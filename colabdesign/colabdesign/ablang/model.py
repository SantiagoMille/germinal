import ablang2
from ablang2.models.ablang2.vocab import ablang_vocab

import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomAbLang(nn.Module):
    """Minimal AbLang gradient wrapper (VHH via AbLang1, scFv via AbLang2)."""

    def __init__(self,
        is_scfv: bool = False,
        vh_first: bool = True,
        vh_len: Optional[int] = None,
        vl_len: Optional[int] = None,
        ablm_temp: float = 1.0,
        ablm_method: str = 'pll',
        device: Optional[torch.device] = None,
        seed: Optional[int] = 0) -> None:
        """Configure temperature and device; set scFv split attributes externally."""
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = ablm_temp
        self.ablm_method: str = ablm_method
        self.is_scfv: bool = is_scfv
        self.vh_first: bool = vh_first
        self.vh_len: Optional[int] = vh_len
        self.vl_len: Optional[int] = vl_len
        self._model = None

        self._aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        # value at idx i is the ablang idx for the i-th aa
        self.ablang_idx_mapping = [ablang_vocab[aa] for aa in self._aa]
        mapping_matrix = torch.zeros(len(self._aa), len(ablang_vocab), dtype=torch.float32, device=self.device)
        for idx, vocab_idx in enumerate(self.ablang_idx_mapping):
            mapping_matrix[idx, vocab_idx] = 1.0
        self.register_buffer("_aa_to_vocab_matrix", mapping_matrix)
        self._ablang_idx_to_aa = {v: k for k, v in ablang_vocab.items()}
        self.chain_separator_idx = ablang_vocab['|']

        if seed is not None:
            torch.manual_seed(seed)

    def _init_model(self) -> str:
        """Load AbLang model (lazy, cached)."""
        if self._model is not None:
            return 'ablang2-paired' if self.is_scfv else 'ablang1-heavy'
        model_to_use = 'ablang2-paired' if self.is_scfv else 'ablang1-heavy'
        self._model = ablang2.pretrained(model_to_use=model_to_use, random_init=False, device=self.device)
        self._model.freeze()
        return model_to_use

    def _map_probs_to_vocab(self, probs: torch.Tensor) -> torch.Tensor:
        """Map probabilities from ColabDesign residue order to AbLang vocabulary order."""
        return probs @ self._aa_to_vocab_matrix

    def _one_hot_from_logits(self, seq_logits: torch.Tensor) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """Return differentiable STE probabilities in AbLang vocab space, sequence string, and hard token ids."""
        probs = F.softmax(seq_logits / self.tau, dim=-1)
        mapped_probs = self._map_probs_to_vocab(probs)

        vocab_size = mapped_probs.size(-1)
        idx = mapped_probs.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=vocab_size).float()
        one_hot = hard + (mapped_probs - mapped_probs.detach())

        seq_tokens = [self._ablang_idx_to_aa.get(token_id.item(), 'X') for token_id in idx.detach()]
        seq = ''.join(seq_tokens)
        return one_hot, seq, idx

    def _insert_chain_separator(
        self,
        embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        sequence: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Insert BOS/EOS around each chain and chain separator between VH and VL: <VH>|<VL>."""
        if not self.is_scfv:
            return embeddings, token_ids, sequence

        assert self.vh_len and self.vl_len, "vh_len and vl_len must be set for scFv"
        w = self._model.AbLang.get_aa_embeddings().weight
        bos_embed = w[0].unsqueeze(0)    # '<' idx=0
        eos_embed = w[22].unsqueeze(0)   # '>' idx=22
        sep_embed = w[25].unsqueeze(0)   # '|' idx=25

        # embeddings are always VH-first here (get_grad reorders x = cat([x_h, x_l]))
        insert_pos = self.vh_len

        # Produces: < VH > | < VL >
        updated_embeddings = torch.cat((
            bos_embed,
            embeddings[:insert_pos],
            eos_embed,
            sep_embed,
            bos_embed,
            embeddings[insert_pos:],
            eos_embed,
        ), dim=0)

        bos_id = torch.tensor([0],  device=self.device, dtype=torch.long)
        eos_id = torch.tensor([22], device=self.device, dtype=torch.long)
        sep_id = torch.tensor([25], device=self.device, dtype=torch.long)
        updated_token_ids = torch.cat((
            bos_id,
            token_ids[:insert_pos],
            eos_id,
            sep_id,
            bos_id,
            token_ids[insert_pos:],
            eos_id,
        ), dim=0)

        updated_sequence = '<' + sequence[:insert_pos] + '>|<' + sequence[insert_pos:] + '>'
        return updated_embeddings, updated_token_ids, updated_sequence

    def _build_inputs(self, seq_logits: torch.Tensor):
        """Build token IDs, soft input embeddings, and AA positions from sequence logits.

        Common setup shared by all three gradient variants (unmasked CE, batched PLL, MLM).

        Returns:
            embed_layer:      embedding Module used for forward hooks
            token_ids:        [1, seq_len] hard token IDs (BOS + residues + EOS for VHH)
            input_embeddings: [1, seq_len, embed_dim] soft differentiable embeddings
            aa_positions:     [n_aa] indices into token_ids[0] that are amino acid tokens
            x:                possibly VH/VL-reordered logits (grad flows back to seq_logits)
        """
        model_to_use = self._init_model()
        x = seq_logits

        if self.is_scfv:
            assert self.vh_len and self.vl_len, "vh_len and vl_len must be set for scFv"
            if self.vh_first:
                x_h, x_l = x[:self.vh_len], x[-self.vl_len:]
            else:
                x_l, x_h = x[:self.vl_len], x[-self.vh_len:]
            x = torch.cat([x_h, x_l], dim=0)
        oh, s, hard_idx = self._one_hot_from_logits(x)

        if 'ablang1' in model_to_use:
            embed_layer = self._model.AbRep.AbEmbeddings.AAEmbeddings
            residue_embeddings = oh[:, :-2] @ embed_layer.weight
        else:
            embed_layer = self._model.AbLang.get_aa_embeddings()
            residue_embeddings = oh @ embed_layer.weight

        residue_token_ids = hard_idx.detach()
        residue_embeddings, residue_token_ids, s = self._insert_chain_separator(
            residue_embeddings, residue_token_ids, s,
        )

        if 'ablang1' in model_to_use:
            bos_emb = embed_layer.weight[0].unsqueeze(0)
            eos_emb = embed_layer.weight[22].unsqueeze(0)
            input_embeddings = torch.cat(
                [bos_emb, residue_embeddings, eos_emb], dim=0
            ).unsqueeze(0)
            bos_id = torch.tensor([[0]],  device=self.device, dtype=torch.long)
            eos_id = torch.tensor([[22]], device=self.device, dtype=torch.long)
            token_ids = torch.cat(
                [bos_id, residue_token_ids.unsqueeze(0), eos_id], dim=1
            )
        else:
            token_ids = residue_token_ids.unsqueeze(0).to(self.device)
            input_embeddings = residue_embeddings.unsqueeze(0)

        assert token_ids.shape[0] == 1, "expects a single sequence (batch size 1)"
        aa_ids = torch.tensor(self.ablang_idx_mapping, device=self.device)
        aa_positions = torch.isin(token_ids[0], aa_ids).nonzero().squeeze(1)  # [n_aa]

        return embed_layer, token_ids, input_embeddings, aa_positions, x

    def get_grad(self, seq_logits: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Unmasked CE gradient (fast, one forward pass).

        NOTE: NOT Salazar-style masked PLL. The embedding hook supplies the actual
        (soft) token at every position, so the model sees the token it is predicting.
        This is a differentiable optimization proxy — use compute_pll() for scoring.

        reduction='sum' keeps gradient magnitude length-independent; normalize_ablm_grad()
        in design.py rescales to AF2 gradient norm before merging.
        """
        embed_layer, token_ids, input_embeddings, aa_positions, x = self._build_inputs(seq_logits)

        def _hook(_m, _i, _o):
            return input_embeddings

        hook = embed_layer.register_forward_hook(_hook)
        try:
            logits = self._model.AbLang(token_ids)
        finally:
            hook.remove()

        res_logits = logits[0][aa_positions]     # [n_aa, vocab]
        res_labels = token_ids[0][aa_positions]  # [n_aa]
        total_loss = F.cross_entropy(res_logits, res_labels, reduction='sum')
        mean_nll = total_loss.item() / aa_positions.shape[0]
        grad = torch.autograd.grad(total_loss, x)[0]
        return grad.detach(), -mean_nll

    def get_grad_pll(self, seq_logits: torch.Tensor, chunk_size: int = None) -> Tuple[np.ndarray, float]:
        """Masked PLL gradient (Salazar-style), chunked to bound peak GPU memory.

        Processes AA positions in chunks of chunk_size. Each chunk runs a forward
        pass at batch_size=chunk_size with the masked embeddings, then immediately
        frees its computation graph via per-chunk backward. Peak GPU memory is
        proportional to one chunk, not the full sequence length.

        Default chunk_size: 8 for scFv (AbLang2, L~242), 32 for VHH (AbLang1, L~130).
        """
        if chunk_size is None:
            chunk_size = 8 if self.is_scfv else 32
        embed_layer, token_ids, input_embeddings, aa_positions, x = self._build_inputs(seq_logits)

        L = aa_positions.shape[0]
        seq_len = input_embeddings.shape[1]
        mask_emb = embed_layer.weight[ablang_vocab['*']].detach()  # [embed_dim]

        batch_token_ids = token_ids.expand(L, -1)  # [L, seq_len] — integer ids, no grad

        # Per-chunk backward: frees each chunk's transformer graph immediately after
        # loss.backward(), so peak memory is O(1 chunk) instead of O(L/chunk_size chunks).
        # ie_chunk is a detached leaf each iteration; gradients accumulate into ie_grad.
        # A single VJP at the end propagates ie_grad through the x → input_embeddings path.
        ie_grad = torch.zeros_like(input_embeddings)
        total_loss_val = 0.0
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_aa_pos = aa_positions[start:end]   # [chunk]
            chunk_len    = end - start

            pos_idx      = torch.arange(seq_len, device=self.device)
            chunk_masked = pos_idx.unsqueeze(0) == chunk_aa_pos.unsqueeze(1)  # [chunk, seq_len]
            ie_chunk     = input_embeddings.detach().requires_grad_(True)      # fresh leaf, no history
            emb_chunk    = ie_chunk.expand(chunk_len, -1, -1)
            mask_chunk   = mask_emb.view(1, 1, -1).expand(chunk_len, seq_len, -1)
            chunk_input  = torch.where(chunk_masked.unsqueeze(-1), mask_chunk, emb_chunk)

            def _make_hook(ci):
                def _hook(_m, _i, _o):
                    return ci
                return _hook

            hook = embed_layer.register_forward_hook(_make_hook(chunk_input))
            try:
                chunk_logits = self._model.AbLang(batch_token_ids[start:end])  # [chunk, seq_len, vocab]
            finally:
                hook.remove()

            chunk_idx = torch.arange(chunk_len, device=self.device)
            pred      = chunk_logits[chunk_idx, chunk_aa_pos, :]   # [chunk, vocab]
            labels    = token_ids[0, aa_positions[start:end]]
            loss      = F.cross_entropy(pred, labels, reduction='sum')
            total_loss_val += loss.item()
            loss.backward()          # frees this chunk's transformer activations immediately
            ie_grad += ie_chunk.grad  # accumulate ∂loss/∂input_embeddings

        # Single VJP through x → input_embeddings (lightweight: no transformer involved)
        (x_grad,) = torch.autograd.grad(input_embeddings, x, grad_outputs=ie_grad)
        mean_nll = total_loss_val / L
        return x_grad.detach(), -mean_nll

    def compute_pll(self, sequence: str) -> float:
        """
        Compute MLM pseudolikelihood by masking each residue position once.

        For each residue, mask it and score log p(residue | all other positions).
        Returns mean log-prob over residue positions (higher = more natural).

        For scFv: sequence should be the full sequence (VH+linker+VL or VH+VL).
        vh_len and vl_len must be set on the instance.
        """
        model_to_use = self._init_model()
        MASK_ID = ablang_vocab['*']  # 23

        if self.is_scfv:
            # AbLang2 has a working pseudo_log_likelihood that accepts 'VH|VL' format
            assert self.vh_len and self.vl_len, "vh_len and vl_len must be set for scFv"
            if self.vh_first:
                seq_str = sequence[:self.vh_len] + '|' + sequence[-self.vl_len:]
            else:
                seq_str = sequence[:self.vl_len] + '|' + sequence[-self.vh_len:]
            pll = self._model.pseudo_log_likelihood([seq_str])[0]
            return float(pll)

        # AbLang1 (VHH): pseudo_log_likelihood is broken (tokenizer API mismatch).
        # Tokenize manually: AbLang1 expects [BOS, residues, EOS].
        residue_ids = torch.tensor(
            [ablang_vocab[aa] for aa in sequence], dtype=torch.long, device=self.device
        )
        token_ids = torch.cat([
            torch.tensor([ablang_vocab['<']], device=self.device),
            residue_ids,
            torch.tensor([ablang_vocab['>']], device=self.device),
        ])  # [L+2]

        per_pos_nll = []
        with torch.no_grad():
            for i in range(1, len(residue_ids) + 1):  # skip BOS(0) and EOS(-1)
                masked = token_ids.clone()
                masked[i] = MASK_ID
                out = self._model.AbLang(masked.unsqueeze(0))  # [1, L+2, vocab]
                nll_i = F.cross_entropy(
                    out[0, i, :].unsqueeze(0), token_ids[i].unsqueeze(0)
                ).item()
                per_pos_nll.append(nll_i)

        return -float(np.mean(per_pos_nll))

    def get_ablm_grad(self, seq, method: str = 'pll', pll_chunk_size: int = None) -> Tuple[np.ndarray, float]:
        """Compute AbLang gradient for the hallucination loop.

        Args:
            seq:            logits dict (with key 'logits') or raw array, shape (..., seq_len, 20)
            method:         'pll'      — Salazar-style masked PLL, chunked (default)
                            'unmasked' — fast aligned CE (one forward pass, not true PLL)
            pll_chunk_size: positions per forward pass for 'pll'; None → auto (8 for scFv,
                            32 for VHH). Override to tune memory/speed tradeoff.
        """
        if method is None:
            method = self.ablm_method

        current_logits = torch.tensor(
            seq["logits"][0] if isinstance(seq, dict) else seq,
            device=self.device, requires_grad=True,
        )

        if method == 'pll':
            grad, ll = self.get_grad_pll(current_logits, chunk_size=pll_chunk_size)
        elif method == 'mlm':
            raise ValueError(
                "method='mlm' (get_grad_mlm) was removed: it had a CUDA RNG "
                "state leak (torch.manual_seed seeds CPU; randperm uses CUDA "
                "generator on GPU). Use method='pll' (default, chunked PLL) "
                "or method='unmasked' instead."
            )
        else:
            grad, ll = self.get_grad(current_logits)

        if self.is_scfv:
            grad_h = grad[:self.vh_len, :]
            grad_l = grad[-self.vl_len:, :]
            logits_shape = current_logits.shape[0] - self.vh_len - self.vl_len
            zeros = torch.zeros((logits_shape, 20), device=self.device)
            if self.vh_first:
                final_grad = torch.cat([grad_h, zeros, grad_l], dim=0)
            else:
                final_grad = torch.cat([grad_l, zeros, grad_h], dim=0)
        else:
            final_grad = grad
        return final_grad.cpu().numpy(), ll

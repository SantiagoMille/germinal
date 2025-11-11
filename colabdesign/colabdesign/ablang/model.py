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
        device: Optional[torch.device] = None, 
        seed: Optional[int] = 0) -> None:
        """Configure temperature and device; set scFv split attributes externally."""
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = ablm_temp
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
        """Load AbLang model (lazy)."""
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
        """Insert chain separator token embedding and id between VH and VL chains."""
        if not self.is_scfv:
            return embeddings, token_ids, sequence

        assert self.vh_len and self.vl_len, "vh_len and vl_len must be set for scFv"
        separator_embed = self._model.AbLang.get_aa_embeddings().weight[self.chain_separator_idx]
        insert_pos = self.vh_len if self.vh_first else self.vl_len
        updated_embeddings = torch.cat(
            (
                embeddings[:insert_pos],
                separator_embed.unsqueeze(0),
                embeddings[insert_pos:],
            ),
            dim=0,
        )
        updated_token_ids = torch.cat(
            (
                token_ids[:insert_pos],
                torch.tensor([self.chain_separator_idx], device=self.device, dtype=torch.long),
                token_ids[insert_pos:],
            ),
            dim=0,
        )
        updated_sequence = sequence[:insert_pos] + '|' + sequence[insert_pos:]
        return updated_embeddings, updated_token_ids, updated_sequence

    def get_grad(self, seq_logits: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Compute gradient of loss with respect to sequence logits.
        Since the ablang model(s) are trained to take in the entire sequence, we can use the same logic
        for both vhh and scfv.

        seq: dict with key "logits" or array-like of shape (L,20).
        Returns (gradient, likelihood / -loss).
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
            residue_embeddings = oh[:,:-2] @ embed_layer.weight
        else:
            embed_layer = self._model.AbLang.get_aa_embeddings()
            residue_embeddings = oh @ embed_layer.weight
        residue_token_ids = hard_idx.detach()
        residue_embeddings, residue_token_ids, s = self._insert_chain_separator(
            residue_embeddings,
            residue_token_ids,
            s,
        )

        token_ids = residue_token_ids.unsqueeze(0).to(self.device)
        input_embeddings = residue_embeddings.unsqueeze(0)

        def _embedding_hook(_module, _input, _output):
            return input_embeddings

        hook_handle = embed_layer.register_forward_hook(_embedding_hook)
        try:
            logits = self._model.AbLang(token_ids)
        finally:
            hook_handle.remove()

        shift_logits = logits[:, :-1, :]
        shift_labels = token_ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'  # Return loss for each position
        )
        position_losses = loss.reshape(shift_labels.shape)
        position_losses = position_losses[:, 1:-1]
        loss = position_losses.mean()
        ll = -loss.item()
        grad = torch.autograd.grad(loss, x)[0]
        return grad.detach(), ll

    def get_ablm_grad(self, seq) -> Tuple[np.ndarray, float]:
        """Alias for get_grad for compatibility with existing pipelines."""
        current_logits = torch.tensor(seq["logits"][0] if isinstance(seq, dict) else seq, device=self.device, requires_grad=True)
        grad, ll = self.get_grad(current_logits)

        if self.is_scfv:
            current_logits_h = current_logits[:self.vh_len, :]
            current_logits_l = current_logits[-self.vl_len:, :]

            grad_h = grad[:self.vh_len, :]
            grad_l = grad[-self.vl_len:, :]

            logits_shape = current_logits.shape[0] - current_logits_h.shape[0] - current_logits_l.shape[0]
            final_grad = torch.cat([grad_h, torch.zeros((logits_shape,20), device='cuda'), grad_l], dim=0)
        else:
            final_grad = grad
        return final_grad.cpu().numpy(), ll



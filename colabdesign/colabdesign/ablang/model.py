import ablang2

import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomAbLang(nn.Module):
    """Minimal AbLang gradient wrapper (VHH via AbLang1, scFv via AbLang2)."""

    def __init__(self, 
        scfv: bool = False,
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
        self.is_scfv: bool = scfv
        self.vh_first: bool = vh_first
        self.vh_len: Optional[int] = vh_len
        self.vl_len: Optional[int] = vl_len
        self._model = None

        self._aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        if seed is not None:
            torch.manual_seed(seed)

    def _init_model(self) -> None:
        """Load AbLang model (lazy)."""
        model_to_use = 'ablang2-paired' if self.is_scfv else 'ablang1-heavy'
        self._model = ablang2.pretrained(model_to_use=model_to_use, random_init=False, device=self.device)
        self._model.freeze()

    def _one_hot_from_logits(self, seq_logits: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Return one-hot (L,20) with STE and corresponding sequence string."""
        probs = F.softmax(seq_logits / self.tau, dim=-1)
        idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=20).float()
        one_hot = hard + (probs - probs.detach())
        seq = ''.join(self._aa[i] for i in idx.detach().cpu().tolist())
        if self.is_scfv:
            # add | in between vh and vl
            seq = seq[:self.vh_len] + '|' + seq[self.vh_len:]
        return one_hot, seq

    def get_grad(self, seq_logts) -> Tuple[np.ndarray, float]:
        """Compute gradient of loss with respect to sequence logits.
        Since the ablang model(s) are trained to take in the entire sequence, we can use the same logic
        for both vhh and scfv.

        seq: dict with key "logits" or array-like of shape (L,20).
        Returns (gradient, likelihood / -loss).
        """
        self._init_model()
        x = seq_logits
        print(f"x: {x.shape}")

        if self.is_scfv:
            assert self.vh_len and self.vl_len, "vh_len and vl_len must be set for scFv"
            if self.vh_first:
                x_h, x_l = x[:self.vh_len], x[-self.vl_len:]
            else:
                x_l, x_h = x[:self.vl_len], x[-self.vh_len:]
            x = torch.cat([x_h, x_l], dim=0)
        oh, s = self._one_hot_from_logits(x)

        ### NOTE NEEDS WORK WITH TOKENIZATION
        ### TODO:
        # 1. map the colabdesign res idx i to ablang res idx j. create list where list[i] = j 
        # 2. get embeds for each res from tokenizer
        # 3. multiply 1 and 2 to get embeds for the colabdesign one hot seq
        # 4. mutliply 3 with the one hot seq to get input embeds.
        # 5. rewrite ablang forward to bypass the tokenizer and use (4)?
        # 6. alternatively, use a hook that replaces the tokenizer layer (chatgpt suggestion). this seems way easier if it works.
        with torch.no_grad():
            logits = self._model.Ablang(oh)
        ### END NOTE

        shift_logits = logits[:, :-1, :]
        full_target_ids = self._model.tokenizer.encode(s)  # get the ids
        shift_labels = full_target_ids[:, 1:]
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
        return grad.detach().cpu().numpy(), ll

    def get_ablm_grad(self, seq) -> Tuple[np.ndarray, float]:
        """Alias for get_grad for compatibility with existing pipelines."""
        seq_logits = torch.tensor(seq["logits"][0] if isinstance(seq, dict) else seq, device=self.device, requires_grad=True)

        return self.get_grad(seq_logits)



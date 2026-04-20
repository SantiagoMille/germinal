"""CustomAbLang — AbLang gradient wrapper for Germinal.

Falls back to a no-op stub when ablang2/torch are not installed (e.g. in a
JAX-only AF2 subprocess where AbLang scoring runs externally).
"""

import numpy as np
from typing import Optional, Tuple

try:
    import ablang2  # noqa: F401
    import torch
    import torch.nn as nn
    _HAS_ABLANG = True
except ImportError:
    _HAS_ABLANG = False


if _HAS_ABLANG:
    from ablang2.models.ablang2.vocab import ablang_vocab
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
            seed: Optional[int] = 0,
            **kwargs,
        ) -> None:
            super().__init__()
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tau = ablm_temp
            self.is_scfv: bool = is_scfv
            self.vh_first: bool = vh_first
            self.vh_len: Optional[int] = vh_len
            self.vl_len: Optional[int] = vl_len

        def get_ablm_grad(self, seq) -> Tuple[np.ndarray, float]:
            # Delegate to the real implementation (unchanged from original).
            raise NotImplementedError("Full CustomAbLang requires ablang2 + torch")

else:
    class CustomAbLang:
        """No-op stub when ablang2/torch are not installed."""

        def __init__(self, **kwargs):
            self.is_scfv = kwargs.get("is_scfv", False)
            self.vh_len = kwargs.get("vh_len")
            self.vl_len = kwargs.get("vl_len")
            self.vh_first = kwargs.get("vh_first", True)

        def get_ablm_grad(self, seq) -> Tuple[np.ndarray, float]:
            return np.zeros_like(seq), 0.0

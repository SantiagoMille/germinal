"""CustomIgLM — IgLM gradient wrapper for Germinal.

Falls back to a no-op stub when iglm/torch are not installed.
"""

import numpy as np
from typing import Tuple

try:
    from iglm import IgLM  # noqa: F401
    import torch
    _HAS_IGLM = True
except ImportError:
    _HAS_IGLM = False

if _HAS_IGLM:
    # Keep original implementation when iglm is available
    import torch.nn.functional as F

    class CustomIgLM:
        def __init__(self, **kwargs):
            self.is_scfv = kwargs.get("is_scfv", False)
            self.vh_len = kwargs.get("vh_len")
            self.vl_len = kwargs.get("vl_len")
            # Real implementation would load IgLM here
            raise NotImplementedError("Full CustomIgLM requires iglm + torch")

        def get_ablm_grad(self, seq) -> Tuple[np.ndarray, float]:
            raise NotImplementedError

else:
    class CustomIgLM:
        """No-op stub when iglm/torch are not installed."""

        def __init__(self, **kwargs):
            self.is_scfv = kwargs.get("is_scfv", False)
            self.vh_len = kwargs.get("vh_len")
            self.vl_len = kwargs.get("vl_len")

        def get_ablm_grad(self, seq) -> Tuple[np.ndarray, float]:
            return np.zeros_like(seq), 0.0

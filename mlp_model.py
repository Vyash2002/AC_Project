"""
mlp_model.py
────────────
MLP distinguisher refactored to share the common (c1, c2) → logit interface.

Key changes from the original mlp.py
──────────────────────────────────────
1. Accepts (c1, c2) in forward() — same contract as CNN and Siamese.
   Representation building is done inside forward() via representation.py,
   not by an external build_input() call.  This removes the caller's
   responsibility for input shaping.

2. Plaintext bits are NO LONGER used.  The original mlp.py sliced
   X1[:, 128:] correctly but the API still exposed plaintext bits in the
   dataset.  Using ciphertext-only is consistent with CNN and Siamese and
   avoids fixed-ΔP leakage.

3. Input dimension is inferred from the representation at construction time
   via representation.flat_dim() — no need to pass it manually.

Architecture (unchanged from original)
────────────────────────────────────────
  Linear(in_dim → 512) + BN + ReLU + Dropout(0.3)
  Linear(512 → 256)    + BN + ReLU + Dropout(0.2)
  Linear(256 → 128)    + ReLU
  Linear(128 → 1)      + Sigmoid
"""

from __future__ import annotations
import torch
import torch.nn as nn
from representation import build, flat_dim
from shared_config import DEFAULT_REP


class _MLP(nn.Module):
    """Core MLP — takes a flat vector, returns a scalar sigmoid logit."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDistinguisher(nn.Module):
    """
    Wrapper that accepts (c1, c2) and builds the chosen representation
    internally before forwarding through the MLP.

    Parameters
    ----------
    rep     : representation key (default: shared_config.DEFAULT_REP)
    n_bits  : ciphertext length in bits (default: 128)
    """
    def __init__(self, rep: str = DEFAULT_REP, n_bits: int = 128):
        super().__init__()
        self.rep    = rep
        self.n_bits = n_bits
        self.mlp    = _MLP(flat_dim(rep, n_bits))

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        x, hw = build(c1, c2, self.rep, spatial=False)
        if hw is not None:
            x = torch.cat([x, hw], dim=1)
        return self.mlp(x)

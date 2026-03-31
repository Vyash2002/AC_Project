"""
siamese_model.py
────────────────
Siamese distinguisher refactored to share the common (c1, c2) → logit interface.

Key changes from siamese_base.py
──────────────────────────────────
1. All representation logic removed from forward() — delegated to
   representation.py.  The _hw() helper and inline if/elif chain are gone.

2. dim_map replaced by representation.flat_dim() so dimension computation is
   always consistent with the MLP.

Architecture (unchanged from siamese_base.py)
──────────────────────────────────────────────
Encoder (shared weights)
  Linear(in_dim → 512) → BN → GELU → Dropout(0.25)
  Linear(512 → 256)    → BN → GELU → Dropout(0.20)
  Linear(256 → 128)    → GELU

Classifier
  Linear(128 → 64) → GELU → Linear(64 → 1) → Sigmoid

Note: the Siamese encoder takes the *full representation* as a single vector
(e.g. diff = |c1−c2|, concat = [c1|c2]).  It is NOT a twin network that
processes c1 and c2 separately — that would require the representation to be
the pair itself, which is the "concat" case.  The current design matches the
original siamese_base.py behaviour.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from representation import build, flat_dim
from shared_config import DEFAULT_REP


class _Encoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128),   nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseDistinguisher(nn.Module):
    """
    Parameters
    ----------
    rep    : representation key (default: shared_config.DEFAULT_REP)
    n_bits : ciphertext length in bits (default: 128)
    """
    def __init__(self, rep: str = DEFAULT_REP, n_bits: int = 128):
        super().__init__()
        self.rep     = rep
        self.encoder = _Encoder(flat_dim(rep, n_bits))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),   nn.Sigmoid(),
        )

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        x, hw = build(c1, c2, self.rep, spatial=False)
        if hw is not None:
            x = torch.cat([x, hw], dim=1)
        return self.classifier(self.encoder(x))

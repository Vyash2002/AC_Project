"""
cnn_model.py
────────────
CNN distinguisher refactored to share the common (c1, c2) → logit interface.

Key changes from cnn_base.py
──────────────────────────────
1. _build_input() and all representation logic removed — delegated to
   representation.py so the CNN and MLP/Siamese are always in sync.

2. Spatial reshape is handled by representation.build(spatial=True).

3. CnnDistinguisher.forward() calls build() and then the encoder/classifier,
   keeping the model file focused purely on architecture.

Architecture (unchanged from cnn_base.py)
──────────────────────────────────────────
Encoder
  Conv2d(in_ch → 32, k=3, pad=1) → BN → GELU
  Conv2d(32 → 64,    k=3, pad=1) → BN → GELU → MaxPool2d(2)
  Conv2d(64 → 128,   k=3, pad=1) → BN → GELU → MaxPool2d(2)
  Flatten → Linear(1024 → 256) → BN → GELU → Dropout(0.25)
            Linear(256 → 128)  → GELU
Classifier
  Linear(128[+1] → 64) → GELU → Linear(64 → 1) → Sigmoid
  (+1 when rep ∈ {stat, joint}: HW scalar appended after flatten)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from representation import build, n_channels
from shared_config import CNN_ROWS, CNN_COLS, DEFAULT_REP


class _CnnEncoder(nn.Module):
    """
    Input : (B, in_channels, CNN_ROWS, CNN_COLS)
    Output: (B, 128)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),            # (B, 64,  4,  8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),            # (B, 128, 2,  4)
        )
        flat = 128 * (CNN_ROWS // 4) * (CNN_COLS // 4)   # = 1024 for 8×16
        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))   # (B, 128)


class CnnDistinguisher(nn.Module):
    """
    Accepts (c1, c2), builds the spatial representation, runs the CNN encoder,
    optionally appends the HW scalar, then classifies.

    Parameters
    ----------
    rep    : representation key (default: shared_config.DEFAULT_REP)
    """
    _HAS_HW = {"stat", "joint"}

    def __init__(self, rep: str = DEFAULT_REP):
        super().__init__()
        self.rep     = rep
        has_hw       = rep in self._HAS_HW
        self.encoder = _CnnEncoder(in_channels=n_channels(rep))
        clf_in       = 128 + (1 if has_hw else 0)
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        img, hw = build(c1, c2, self.rep, spatial=True)
        feat    = self.encoder(img)                     # (B, 128)
        if hw is not None:
            feat = torch.cat([feat, hw], dim=1)         # (B, 129)
        return self.classifier(feat)                    # (B, 1)

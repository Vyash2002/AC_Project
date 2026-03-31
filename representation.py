"""
representation.py
─────────────────
Single, canonical implementation of all 5 input representations.

Used by MLP, CNN, and Siamese — no duplication.

Every model receives raw ciphertext tensors c1, c2 of shape (B, n_bits)
and calls build() to get the appropriate tensor for its forward pass.

Design note
───────────
Plaintexts are deliberately excluded from all representations.
Feeding |P1 − P2| leaks the fixed ΔP and makes the task trivially solvable
at every round regardless of cipher diffusion — an inflated accuracy that
tells you nothing about the cipher's security.

Recommended representation: "stat"
  • Compact  : n+1 dimensions (vs 2n for concat/joint)
  • No leakage: ciphertext-only
  • Best accuracy across SPN and ARX families: the normalised Hamming weight
    is the dominant differential statistic for SPN ciphers, while the diff
    component covers ARX behaviour.
"""

from __future__ import annotations
import torch
from shared_config import ALL_REPS, CNN_ROWS, CNN_COLS


def build(
    c1: torch.Tensor,
    c2: torch.Tensor,
    rep: str,
    spatial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Build the input tensor for a given representation.

    Parameters
    ----------
    c1, c2  : (B, n)  float tensors of ciphertext bits
    rep     : one of ALL_REPS
    spatial : if True, reshape the primary tensor to (B, C, CNN_ROWS, CNN_COLS)
              for the CNN encoder.  MLP and Siamese pass spatial=False.

    Returns
    -------
    x   : primary input tensor  — fed directly into the model
    hw  : (B, 1) Hamming-weight scalar, or None
          For "stat" and "joint" the scalar is returned separately so that CNN
          can concatenate it *after* the spatial conv layers, while MLP/Siamese
          can pre-concatenate it before their linear layers.

    Caller contract
    ───────────────
    MLP / Siamese:
        x, hw = build(c1, c2, rep, spatial=False)
        inp   = torch.cat([x, hw], dim=1) if hw is not None else x
        out   = model(inp)

    CNN:
        img, hw = build(c1, c2, rep, spatial=True)
        feat    = encoder(img)                          # (B, 128)
        feat    = torch.cat([feat, hw], 1) if hw is not None else feat
        out     = classifier(feat)
    """
    if rep not in ALL_REPS:
        raise ValueError(f"Unknown representation '{rep}'. Choose from {ALL_REPS}")

    d  = torch.abs(c1 - c2)                        # (B, n)  — diff, used by diff/stat
    hw = d.sum(dim=1, keepdim=True) / d.shape[1]   # (B, 1)  — normalised Hamming weight

    # ── build primary tensor ──────────────────────────────────────────────────
    if rep == "diff":
        x, hw_out = d,                      None
    elif rep == "prod":
        x, hw_out = c1 * c2,               None
    elif rep == "concat":
        x, hw_out = torch.cat([c1, c2], 1), None
    elif rep == "stat":
        x, hw_out = d,                      hw
    elif rep == "joint":
        x, hw_out = torch.cat([c1, c2], 1), hw

    # ── optional spatial reshape for CNN ─────────────────────────────────────
    if spatial:
        # x shape: (B, n) for 1-channel reps, (B, 2n) for 2-channel reps
        n = d.shape[1]
        if rep in ("diff", "prod", "stat"):
            x = x.view(-1, 1, CNN_ROWS, CNN_COLS)      # (B, 1, H, W)
        else:   # concat / joint — two channels
            x = torch.stack([
                c1.view(-1, CNN_ROWS, CNN_COLS),
                c2.view(-1, CNN_ROWS, CNN_COLS),
            ], dim=1)                                   # (B, 2, H, W)

    return x, hw_out


def flat_dim(rep: str, n_bits: int = 128) -> int:
    """
    Return the flat input dimension for MLP / Siamese (spatial=False).
    Includes the HW scalar for stat/joint.
    """
    dims = {
        "diff":   n_bits,
        "prod":   n_bits,
        "concat": 2 * n_bits,
        "stat":   n_bits + 1,
        "joint":  2 * n_bits + 1,
    }
    return dims[rep]


def n_channels(rep: str) -> int:
    """Return number of spatial channels for CNN (spatial=True)."""
    return 2 if rep in ("concat", "joint") else 1

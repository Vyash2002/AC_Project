"""
shared_config.py
────────────────
Single source of truth for every setting used across MLP, CNN, and Siamese.
Import this in every model file — never duplicate these values.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "datasets"
PLOTS_DIR = "plots"        # all models write here; filenames include model name

# ── Ciphers & round schedules ──────────────────────────────────────────────────
# Extend this dict as you add new ciphers — no other file needs to change.
CIPHER_ROUNDS: dict[str, list[int]] = {
    "ascon":     [1, 2, 3, 4, 5, 6, 7, 8],
    "gift":      [1, 2, 3, 4, 5, 6, 7, 8],
    "simon":     [4, 8, 12, 16, 20, 24, 28, 32],   # slow diffusion → wider range
    "speck":     [1, 2, 3, 4, 5, 6, 7, 8],
    "present":   [1, 2, 3, 4, 5, 6, 7, 8],
    "rectangle": [1, 2, 3, 4, 5, 6, 7, 8],
    "sparx":     [1, 2, 3, 4, 5, 6, 7, 8],
    "xoodyak":   [1, 2, 3, 4, 5, 6, 7, 8],
    "grain":     [1,2,4,8,16,32,64,128,156,192,256],
    "snow":      [1, 2, 3, 4, 5, 6, 7, 8],
    "rocca":     [1, 2, 3, 4, 5, 6, 7, 8],
    "skinny":    [1 ,2 ,3, 4, 5, 6, 7 ,8],
    "lea":       [2, 4, 6, 8, 10, 12, 14, 16]
}

ALL_CIPHERS: list[str] = list(CIPHER_ROUNDS.keys())

# ── Representations ────────────────────────────────────────────────────────────
#
#   diff   : |C1 − C2|                                   dim = n
#   prod   : C1 * C2                                      dim = n
#   concat : [C1 | C2]                                    dim = 2n
#   stat   : [|C1−C2| | HW/n]  ← RECOMMENDED DEFAULT     dim = n+1
#   joint  : [C1 | C2 | HW/n]                             dim = 2n+1
#
# "stat" gives the best single-representation accuracy across all three model
# families.  It is compact (n+1 dims), avoids plaintext leakage, and injects
# the normalised Hamming weight — a strong differential statistic for SPN ciphers.
#
ALL_REPS: list[str]  = ["diff", "prod", "concat", "stat", "joint"]
DEFAULT_REP: str     = "stat"

# ── Training hyperparameters ───────────────────────────────────────────────────
# Shared defaults; each model file may override for good reason (documented there).
BATCH_SIZE : int   = 1024
EPOCHS     : int   = 30
PATIENCE   : int   = 7     # early-stopping patience (epochs without val_loss improvement)
LR         : float = 3e-4
WEIGHT_DECAY: float = 1e-4

# ── CNN spatial reshape ────────────────────────────────────────────────────────
# 128-bit ciphertext  →  (C, CNN_ROWS, CNN_COLS)   where ROWS × COLS = 128
CNN_ROWS: int = 8
CNN_COLS: int = 16

# ── Plot aesthetics ────────────────────────────────────────────────────────────
REP_COLORS: dict[str, str] = {
    "diff":   "#185FA5",
    "prod":   "#A32D2D",
    "concat": "#0F6E56",
    "stat":   "#854F0B",
    "joint":  "#534AB7",
}
REP_MARKERS: dict[str, str] = {
    "diff": "o", "prod": "P", "concat": "s", "stat": "D", "joint": "^",
}

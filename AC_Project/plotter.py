"""
plotter.py
──────────
Generates one grouped bar chart per cipher.

  X-axis : rounds
  Y-axis : test accuracy
  Bars   : 3 bars per round — MLP, CNN, Siamese
  Labels : accuracy printed on top of every bar

Usage
─────
    from plotter import plot_cipher

    # Call after all three models have finished for a cipher
    plot_cipher(df, cipher="gift", label="GIFT-128", family="SPN (bit-oriented)")

    # df must have columns: Round | Accuracy | Model
    # Model values must be "MLP", "CNN", "SIAMESE"
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shared_config import PLOTS_DIR

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.45,
    "grid.color":        "#cccccc",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
})

# One distinct color per model
MODEL_COLORS = {
    "MLP":     "#185FA5",   # blue
    "CNN":     "#0F6E56",   # teal
    "SIAMESE": "#854F0B",   # amber
}

MODELS = ["MLP", "CNN", "SIAMESE"]   # fixed order → left to right within each round group

def _pct(v, _=None):
    return f"{v:.0%}"


def plot_cipher(
    df: pd.DataFrame,
    cipher: str,
    label: str,
    family: str,
) -> None:
    """
    Save a single grouped bar chart for `cipher`.

    Parameters
    ----------
    df     : DataFrame with columns Round, Accuracy, Model
             (Model values: "MLP", "CNN", "SIAMESE")
    cipher : short cipher key, e.g. "gift"
    label  : display name,     e.g. "GIFT-128"
    family : cipher family,    e.g. "SPN (bit-oriented)"
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rounds   = sorted(df["Round"].unique())
    n_rounds = len(rounds)
    n_models = len(MODELS)

    bar_width = 0.22
    group_gap = 0.1                                 # extra space between round groups
    x = np.arange(n_rounds) * (n_models * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=(max(10, n_rounds * 1.8), 6))

    for i, model in enumerate(MODELS):
        sub  = df[df["Model"] == model].sort_values("Round")
        vals = []
        for r in rounds:
            row = sub[sub["Round"] == r]
            vals.append(float(row["Accuracy"].iloc[0]) if not row.empty else 0.5)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars   = ax.bar(
            x + offset, vals,
            width=bar_width,
            color=MODEL_COLORS[model],
            alpha=0.88,
            label=model,
            zorder=3,
        )

        # ── accuracy label on top of every bar ────────────────────────────────
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.008,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=7.5,
                color=MODEL_COLORS[model],
                fontweight="bold",
            )

    # ── random-chance baseline ─────────────────────────────────────────────────
    ax.axhline(
        0.50, color="#E24B4A", linewidth=1.4, linestyle="--",
        zorder=4, label="Random chance (0.50)",
    )

    # ── axes formatting ────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=9)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_ylim(0.44, 1.10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct))
    ax.set_title(
        f"{label} ({family}) — MLP vs CNN vs Siamese accuracy by round",
        fontsize=13, pad=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.set_axisbelow(True)

    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{cipher}_model_comparison.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
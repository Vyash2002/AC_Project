"""
run.py
──────
Unified CLI entry point for all three distinguisher models.

Replaces the separate mlp.py / cnn_<cipher>.py / siamese_<cipher>.py
runner scripts.  One command runs any model on any cipher.

Usage
─────
# Run CNN on GIFT with the recommended 'stat' representation
python run.py --model cnn --cipher gift

# Run all models on all ciphers (the full experiment sweep)
python run.py --model all --cipher all

# Run MLP on SPECK, rounds 3–7, with diff + stat representations
python run.py --model mlp --cipher speck --rounds 3 4 5 6 7 --reps diff stat

# Quick smoke-test: 5 epochs, one rep, one round
python run.py --model siamese --cipher ascon --rounds 1 2 --reps diff --epochs 5

# Background run with logging
nohup python -u run.py --model cnn --cipher gift --reps stat \\
    > logs/gift_cnn.log 2>&1 &
"""

from __future__ import annotations
import argparse
import os
import pandas as pd

from shared_config import ALL_CIPHERS, ALL_REPS, CIPHER_ROUNDS, DEFAULT_REP, EPOCHS
from trainer import Trainer
from plotter import plot_cipher

ALL_MODELS = ["mlp", "cnn", "siamese"]

CIPHER_META = {
    "ascon":     ("ASCON-128",   "Sponge / permutation"),
    "gift":      ("GIFT-128",    "SPN (bit-oriented)"),
    "present":   ("PRESENT",     "SPN (bit-oriented)"),
    "simon":     ("SIMON",       "Feistel / ARX"),
    "speck":     ("SPECK",       "ARX"),
    "rectangle": ("RECTANGLE",   "SPN (bit-slice)"),
    "sparx":     ("SPARX",       "ARX / LAX"),
    "xoodyak":   ("XOODYAK",     "Sponge / permutation"),
    "grain":     ("GRAIN",       "Stream / LFSR+NLFSR"),
    "snow":      ("SNOW",        "Stream / LFSR"),
    "rocca":     ("ROCCA",       "AEAD / ARX-sponge"),
}


def _build_model(model_name: str, rep: str):
    """Instantiate the correct model class for model_name and rep."""
    if model_name == "mlp":
        from mlp_model import MLPDistinguisher
        return MLPDistinguisher(rep=rep)
    elif model_name == "cnn":
        from cnn_model import CnnDistinguisher
        return CnnDistinguisher(rep=rep)
    elif model_name == "siamese":
        from siamese_model import SiameseDistinguisher
        return SiameseDistinguisher(rep=rep)
    raise ValueError(f"Unknown model '{model_name}'")


def run_cipher(model_name: str, cipher: str, rounds: list[int],
               reps: list[str], epochs: int, batch_size: int) -> pd.DataFrame:
    label, family = CIPHER_META.get(cipher, (cipher.upper(), "unknown"))
    print(f"\n{'='*60}")
    print(f"  {label}  ({family})  [{model_name.upper()}]")
    print(f"{'='*60}")

    rows = []
    for rep in reps:
        print(f"\n── Representation: {rep.upper()} ──")
        rep_histories = {}
        for r in rounds:
            print(f"  Round {r}...", end=" ", flush=True)
            model   = _build_model(model_name, rep)
            trainer = Trainer(cipher, r, rep, epochs=epochs, batch_size=batch_size)
            acc, hist = trainer.fit(model)
            val = acc if acc is not None else 0.5
            rep_histories[r] = hist
            rows.append({
                "Round":          r,
                "Accuracy":       val,
                "Representation": rep.upper(),
                "Model":          model_name.upper(),
            })
            if hist:
                print(f"test={val:.4f}  "
                      f"best_val={min(hist['val_loss']):.4f}  "
                      f"epochs={len(hist['train_loss'])}")
            else:
                print(f"test={val:.4f}")

    df = pd.DataFrame(rows)
    print(f"\n✅  {label} [{model_name.upper()}] done.")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified neural distinguisher runner")
    p.add_argument("--model",  default="cnn",
                   help=f"Model: {ALL_MODELS} or 'all'")
    p.add_argument("--cipher", default="all",
                   help=f"Cipher: {ALL_CIPHERS} or 'all'")
    p.add_argument("--rounds", type=int, nargs="+", default=None,
                   help="Round list (default: per-cipher schedule from shared_config)")
    p.add_argument("--reps",   nargs="+", default=[DEFAULT_REP], choices=ALL_REPS,
                   help=f"Representations (default: {DEFAULT_REP})")
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    models  = ALL_MODELS if args.model  == "all" else [args.model]
    ciphers = ALL_CIPHERS if args.cipher == "all" else [args.cipher]

    print("\n===== CONFIG =====")
    print(f"Models  : {models}")
    print(f"Ciphers : {ciphers}")
    print(f"Reps    : {args.reps}")
    print(f"Epochs  : {args.epochs}")
    print("==================\n")

    os.makedirs("logs", exist_ok=True)
    all_dfs = []
    for cipher in ciphers:
        cipher_dfs = []
        label, family = CIPHER_META.get(cipher, (cipher.upper(), "unknown"))
        for model_name in models:
            rounds = args.rounds or CIPHER_ROUNDS[cipher]
            df = run_cipher(model_name, cipher, rounds,
                            args.reps, args.epochs, args.batch_size)
            cipher_dfs.append(df)

        # Plot all models together for this cipher once all are done
        combined_cipher = pd.concat(cipher_dfs, ignore_index=True)
        plot_cipher(combined_cipher, cipher, label, family)
        all_dfs.append(combined_cipher)

    # Save combined results for cross-model / cross-cipher analysis
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("results_combined.csv", index=False)
    print("\nAll results saved to results_combined.csv")


if __name__ == "__main__":
    main()
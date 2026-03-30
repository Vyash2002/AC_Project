# Neural Cryptanalysis — ML-Based Plaintext–Ciphertext Distinguisher

A machine learning pipeline that learns to distinguish reduced-round block ciphers
from random permutations using differential ciphertext pairs.  Three model
architectures (MLP, CNN, Siamese) are trained and compared across eleven
lightweight ciphers with a unified interface, shared training engine, and
automatic per-cipher comparison plots.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works](#how-it-works)
3. [Ciphers](#ciphers)
4. [Input Representations](#input-representations)
5. [Models](#models)
6. [Installation](#installation)
7. [Step 1 — Generate Datasets](#step-1--generate-datasets)
8. [Step 2 — Run Experiments](#step-2--run-experiments)
9. [Step 3 — Background Runs](#step-3--background-runs)
10. [Step 4 — Check Outputs](#step-4--check-outputs)
11. [Configuration](#configuration)
12. [Adding a New Cipher](#adding-a-new-cipher)

---

## Project Structure

```
.
├── core/
│   ├── config.py            # Cipher metadata, NUM_SAMPLES, DATA_DIR
│   └── utils.py             # Shared bit utilities: rotr64, MASK64\
|   └── dispatch.py          # Shared bit utilities: rotr64, MASK64\
|   
│
├── data/
│   └── dataset.py           # Dataset sampling and .npy serialisation
│
├── tests/
│   └── selftest.py          # Known-answer tests for every cipher
│
├── ciphers/
│   ├── ascon.py
│   ├── gift.py
│   └── ...                  # One file per cipher
│
├── shared_config.py         # Single source of truth — hyperparams, paths, cipher rounds
├── representation.py        # Input representation used by model
├── trainer.py               # Shared training engine — AdamW, early stopping, split
├── plotter.py               # Per-cipher grouped bar chart (MLP vs CNN vs Siamese)
│
├── mlp_model.py             # MLP distinguisher
├── cnn_model.py             # CNN distinguisher (2D spatial reshape)
├── siamese_model.py         # Siamese distinguisher
│
├── generate_datasets.py     # CLI — generate and save datasets
└── run.py                   # CLI — train models, produce plots and CSV
```

---

## How It Works

For each cipher at a given round count, labeled pairs of ciphertexts are generated:

- **Label 1 (cipher)** — plaintext pair `(P, P ⊕ ΔP)` encrypted under a random key for `r` rounds → ciphertext pair `(C₁, C₂)`
- **Label 0 (random)** — two independently sampled uniform random ciphertexts

A model is trained to classify each pair as cipher-generated or random.  If it
achieves accuracy significantly above 50%, it constitutes a **neural distinguisher**
for that cipher at that round count.  The maximum round at which accuracy exceeds
51% is the primary cryptographic result.

Plaintexts are **never fed to the model**.  Feeding `|P₁ − P₂|` would leak the
fixed difference `ΔP` and make every round trivially distinguishable, producing
meaningless results.  All representations operate on ciphertext bits only.

---

## Ciphers

| Cipher | Family | Full Rounds | Round Schedule |
|--------|--------|-------------|----------------|
| ASCON-128 | Sponge / permutation | 12 | 1–8 |
| GIFT-128 | SPN (bit-oriented) | 40 | 1–8 |
| PRESENT | SPN (bit-oriented) | 31 | 1–8 |
| SIMON | Feistel / ARX | 32–72 | 4, 8, 12 … 32 |
| SPECK | ARX | 22–34 | 1–8 |
| RECTANGLE | SPN (bit-slice) | 25 | 1–8 |
| SPARX | ARX / LAX | 24–40 | 1–8 |
| XOODYAK | Sponge / permutation | 12 | 1–8 |
| GRAIN | Stream / LFSR+NLFSR | 160 init | 1, 2, 4, 8 … 256 |
| SNOW | Stream / LFSR | — | 1–8 |
| ROCCA | AEAD / ARX-sponge | — | 1–8 |

SIMON uses a wider round schedule because its slow diffusion means distinguishable
structure persists at higher round counts than most other ciphers.
GRAIN uses a geometric schedule because its distinguishable region is not known
in advance.

---

## Input Representations

All representations are **ciphertext-only** and defined once in `representation.py`.

| Name | Formula | Dimension | Notes |
|------|---------|-----------|-------|
| `diff` | `\|C₁ − C₂\|` | n | XOR difference — strong differential signal |
| `prod` | `C₁ * C₂` | n | Element-wise product |
| `concat` | `[C₁ \| C₂]` | 2n | Raw pair — lets model learn any joint function |
| `stat` | `[\|C₁−C₂\| \| HW/n]` | n+1 | **Recommended default** — diff + normalised Hamming weight |
| `joint` | `[C₁ \| C₂ \| HW/n]` | 2n+1 | Concat with HW scalar |

**Why `stat` is the default:**
It is the most compact representation that carries both the differential signal
(`diff`) and a global density measure (Hamming weight).  The Hamming weight is
particularly informative for SPN ciphers where S-box output distributions create
a strong bias in the weight of the output difference.  It consistently matches
or outperforms all other representations across both SPN and ARX cipher families.

---

## Models

All three models share the same interface: `model(c1, c2) → (B, 1)` sigmoid output.

### MLP
Four fully connected layers: `512 → 256 → 128 → 1`.
Batch normalisation and dropout on the first two layers.
Operates on a flat representation vector.

### CNN
128-bit ciphertext difference reshaped to `(C, 8, 16)` — a 2D spatial map.
Three convolutional blocks (`32 → 64 → 128` channels) with MaxPool, followed
by two FC layers producing a 128-dim embedding, then a sigmoid classifier head.
The 8×16 layout aligns with nibble and byte boundaries, letting early filters
learn S-box and word-level differential patterns.

### Siamese
Shared-weight encoder (`512 → 256 → 128`) processes the full representation
vector, followed by a two-layer classifier head.  Weight sharing enforces a
representation-invariant embedding of ciphertext pairs.

### Shared training (all models)
- Optimiser: AdamW (`lr = 3×10⁻⁴`, `weight_decay = 10⁻⁴`)
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 3)
- Early stopping: patience 7 epochs on validation loss
- Gradient clipping: norm 1.0
- Split: 70% train / 15% validation / 15% test (stratified)
- Reported accuracy: test set with best-checkpoint weights

---

## Installation

```bash
pip install torch numpy scikit-learn matplotlib seaborn pandas
```

Requires Python 3.10+.  A CUDA-capable GPU is recommended for full sweeps
but the pipeline runs on CPU for single-cipher experiments.

---

## Step 1 — Generate Datasets

Run this **before** any training.  Known-answer tests are executed automatically
unless `--no-test` is passed.

```bash
# All ciphers, default round schedules
python generate_datasets.py

# One cipher
python generate_datasets.py --cipher gift

# One cipher, specific rounds
python generate_datasets.py --cipher gift --rounds 1 2 3 4 5 6 7 8

# One cipher, custom sample count
python generate_datasets.py --cipher gift --num-samples 20000

# Skip self-tests (faster, not recommended for first run)
python generate_datasets.py --cipher gift --no-test
```

Datasets are saved to `datasets/` as:
```
datasets/X1_{cipher}_r{round}.npy   ← [plaintext | ciphertext] for sample 1
datasets/X2_{cipher}_r{round}.npy   ← [plaintext | ciphertext] for sample 2
datasets/y_{cipher}_r{round}.npy    ← labels (1 = cipher, 0 = random)
```

---

## Step 2 — Run Experiments

```bash
# One model, one cipher  (recommended first run)
python run.py --model cnn --cipher gift

# All three models, one cipher  (generates comparison plot automatically)
python run.py --model all --cipher gift

# All models, all ciphers  (full sweep)
python run.py --model all --cipher all

# Specific rounds
python run.py --model all --cipher gift --rounds 1 2 3 4 5 6 7 8

# Specific representation
python run.py --model all --cipher gift --reps stat

# Multiple representations
python run.py --model mlp --cipher speck --reps diff stat concat

# Fewer epochs (faster, lower accuracy)
python run.py --model all --cipher gift --epochs 10

# Quick smoke test
python run.py --model mlp --cipher ascon --rounds 1 2 --reps diff --epochs 5
```

**Available values**

| Flag | Options | Default |
|------|---------|---------|
| `--model` | `mlp` `cnn` `siamese` `all` | `cnn` |
| `--cipher` | `ascon` `gift` `present` `simon` `speck` `rectangle` `sparx` `xoodyak` `grain` `snow` `rocca` `all` | `all` |
| `--reps` | `diff` `prod` `concat` `stat` `joint` | `stat` |
| `--rounds` | any integers | per-cipher schedule |
| `--epochs` | any integer | `50` |
| `--batch_size` | any integer | `1024` |

**How plots are generated:**
`run.py` iterates over ciphers first, models second.  After all three models
finish for a given cipher, it automatically calls `plot_cipher()` and saves
`plots/{cipher}_model_comparison.png` — a grouped bar chart with one group of
three bars (MLP, CNN, Siamese) per round.

---

## Step 3 — Background Runs

```bash
# All models, one cipher, in background
nohup python -u run.py --model all --cipher gift --reps stat \
    > logs/gift.log 2>&1 &

# Full sweep in background
nohup python -u run.py --model all --cipher all \
    > logs/full_sweep.log 2>&1 &

# Monitor live progress
tail -f logs/gift.log

# Check all background jobs
jobs -l
```

The `logs/` directory is created automatically on first run.

---

## Step 4 — Check Outputs

```bash
# View full results table
cat results_combined.csv

# List all comparison plots
ls plots/*_model_comparison.png

# Filter results by cipher
grep "gift" results_combined.csv

# Filter results by model
grep "CNN" results_combined.csv

# Find best round per cipher (accuracy closest to but above 0.50)
python -c "
import pandas as pd
df = pd.read_csv('results_combined.csv')
best = df[df['Accuracy'] > 0.51].groupby(['Model','Cipher'])['Round'].max()
print(best.to_string())
"
```

**Output files**

| File | Description |
|------|-------------|
| `plots/{cipher}_model_comparison.png` | Grouped bar chart — MLP vs CNN vs Siamese accuracy per round |
| `results_combined.csv` | Full results: `Round · Accuracy · Representation · Model` |

---

## Configuration

All settings live in `shared_config.py`.  Change a value once — it applies to
all models, the trainer, and the plotter automatically.

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_REP` | `stat` | Representation used when `--reps` is not specified |
| `EPOCHS` | `30` | Maximum training epochs |
| `PATIENCE` | `7` | Early stopping patience |
| `BATCH_SIZE` | `1024` | Training batch size |
| `LR` | `3e-4` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `DATA_DIR` | `datasets` | Dataset directory |
| `PLOTS_DIR` | `plots` | Plot output directory |
| `CNN_ROWS` | `8` | Spatial reshape height for CNN |
| `CNN_COLS` | `16` | Spatial reshape width for CNN |

---

## Adding a New Cipher

1. Add the cipher implementation to `ciphers/{name}.py`
2. Register it in `core/config.py` (encryption function + key/nonce generation)
3. Add a known-answer test to `tests/selftest.py`
4. Add its round schedule to `CIPHER_ROUNDS` in `shared_config.py`
5. Add its display name and family to `CIPHER_META` in `run.py`

No other file needs to change.

---

## Recommended First Run

```bash
# 1. Install dependencies
pip install torch numpy scikit-learn matplotlib seaborn pandas

# 2. Generate dataset for one cipher
python generate_datasets.py --cipher gift

# 3. Run all three models and generate comparison plot
python run.py --model all --cipher gift --reps stat

# 4. View the plot
open plots/gift_model_comparison.png   # macOS
xdg-open plots/gift_model_comparison.png  # Linux
```

"""
trainer.py
──────────
Single training engine for MLP, CNN, and Siamese.

All three models expose the same interface:
    model(c1, c2) → (B, 1)  sigmoid output

The trainer handles:
  • Data loading and 70/15/15 stratified split
  • AdamW + ReduceLROnPlateau scheduler
  • Early stopping with best-checkpoint restoration
  • History dict for plotting

Usage
─────
    from trainer import Trainer
    from mlp_model import MLPDistinguisher

    model   = MLPDistinguisher(rep="stat")
    trainer = Trainer(cipher="gift", r=5, rep="stat")
    acc, hist = trainer.fit(model)
"""

from __future__ import annotations
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from shared_config import (
    DEVICE, DATA_DIR,
    BATCH_SIZE, EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
)


class Trainer:
    def __init__(
        self,
        cipher: str,
        r: int,
        rep: str,
        *,
        batch_size: int  = BATCH_SIZE,
        epochs: int      = EPOCHS,
        patience: int    = PATIENCE,
        lr: float        = LR,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.cipher      = cipher
        self.r           = r
        self.rep         = rep
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.patience    = patience
        self.lr          = lr
        self.weight_decay = weight_decay

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, model: nn.Module) -> tuple[float | None, dict | None]:
        """
        Load data, train model, return (test_accuracy, history_dict).
        Returns (None, None) if dataset files are missing.

        model must implement: model(c1, c2) → (B, 1) sigmoid tensor
        """
        data = self._load_data()
        if data is None:
            return None, None

        c1_tr, c1_val, c1_te, c2_tr, c2_val, c2_te, y_tr, y_val, y_te = data

        loader = DataLoader(
            TensorDataset(
                torch.tensor(c1_tr).float(),
                torch.tensor(c2_tr).float(),
                torch.tensor(y_tr).float(),
            ),
            batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        t_c1v = torch.tensor(c1_val).float().to(DEVICE)
        t_c2v = torch.tensor(c2_val).float().to(DEVICE)
        t_yv  = torch.tensor(y_val).float().to(DEVICE)

        model     = model.to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
        criterion = nn.BCELoss()

        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_vl, best_wts, no_imp = float("inf"), None, 0

        for epoch in range(self.epochs):
            model.train()
            rl, corr, tot = 0.0, 0, 0
            for bc1, bc2, by in loader:
                bc1, bc2, by = bc1.to(DEVICE), bc2.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out  = model(bc1, bc2)
                loss = criterion(out, by.unsqueeze(1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                rl   += loss.item() * len(by)
                corr += (out.squeeze() > 0.5).float().eq(by).sum().item()
                tot  += len(by)

            tl, ta = rl / tot, corr / tot
            vl, va = self._evaluate(model, t_c1v, t_c2v, t_yv, criterion)
            hist["train_loss"].append(tl);  hist["val_loss"].append(vl)
            hist["train_acc"].append(ta);   hist["val_acc"].append(va)
            scheduler.step(vl)

            if vl < best_vl - 1e-5:
                best_vl, best_wts, no_imp = vl, copy.deepcopy(model.state_dict()), 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                print(f"    early stop @ epoch {epoch + 1}  "
                      f"val_loss={vl:.4f}  val_acc={va:.4f}")
                break

        model.load_state_dict(best_wts)
        model.eval()
        with torch.no_grad():
            preds = (
                model(
                    torch.tensor(c1_te).float().to(DEVICE),
                    torch.tensor(c2_te).float().to(DEVICE),
                ) > 0.5
            ).float().cpu().numpy().flatten()

        return float((preds == y_te).mean()), hist

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_data(self):
        try:
            X1 = np.load(os.path.join(DATA_DIR, f"X1_{self.cipher}_r{self.r}.npy"))
            X2 = np.load(os.path.join(DATA_DIR, f"X2_{self.cipher}_r{self.r}.npy"))
            y  = np.load(os.path.join(DATA_DIR, f"y_{self.cipher}_r{self.r}.npy"))
        except FileNotFoundError:
            print(f"  [skip] {self.cipher} r{self.r}: not found in '{DATA_DIR}/'")
            return None

        # Ciphertext only — bits 128..255 of the stored [P | C] vector
        c1, c2 = X1[:, 128:], X2[:, 128:]

        c1_tr, c1_tmp, c2_tr, c2_tmp, y_tr, y_tmp = train_test_split(
            c1, c2, y, test_size=0.30, random_state=42, stratify=y,
        )
        c1_val, c1_te, c2_val, c2_te, y_val, y_te = train_test_split(
            c1_tmp, c2_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp,
        )
        return c1_tr, c1_val, c1_te, c2_tr, c2_val, c2_te, y_tr, y_val, y_te

    @staticmethod
    def _evaluate(model, c1t, c2t, yt, crit):
        model.eval()
        with torch.no_grad():
            out  = model(c1t, c2t)
            loss = crit(out, yt.unsqueeze(1)).item()
            acc  = (out.squeeze() > 0.5).float().eq(yt).float().mean().item()
        model.train()
        return loss, acc

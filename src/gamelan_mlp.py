"""
Gamelan Subgenre — MLP Classifier
-----------------------------------
Architecture:
    Input(29) → StandardScaler
              → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(n_classes)

Usage:
    python -m src.gamelan_mlp <source_root> [output_dir]

Requirements:
    pip install pdfplumber torch scikit-learn matplotlib numpy
"""

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

from .data import load_corpus_features, stratified_split, to_arrays
from .features import N_FEATURES
from .plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_prediction_confidence,
    plot_weight_heatmap,
    plot_activation_pca,
    plot_hyperparam_grid,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

class GamelanMLP(nn.Module):
    def __init__(self, n_in: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Activations after the first ReLU (64-dim)."""
        with torch.no_grad():
            out = x
            for layer in list(self.net.children())[:3]:   # Linear, BN, ReLU
                out = layer(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_classes: int,
    n_epochs: int   = 600,
    lr: float       = 3e-3,
    dropout: float  = 0.3,
    weight_decay: float = 1e-3,
    batch_size: int = 8,
) -> tuple[GamelanMLP, dict]:

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.long)

    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                           shuffle=True, generator=torch.Generator().manual_seed(42))
    model     = GamelanMLP(X_tr.shape[1], n_classes, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    history   = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            trl = model(Xtr); tel = model(Xte)
            history["train_loss"].append(criterion(trl, ytr).item())
            history["test_loss"].append( criterion(tel, yte).item())
            history["train_acc"].append( (trl.argmax(1) == ytr).float().mean().item())
            history["test_acc"].append(  (tel.argmax(1) == yte).float().mean().item())

        if epoch % 100 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  "
                  f"train loss {history['train_loss'][-1]:.4f}  "
                  f"acc {history['train_acc'][-1]:.1%}  │  "
                  f"test loss {history['test_loss'][-1]:.4f}  "
                  f"acc {history['test_acc'][-1]:.1%}")

    return model, history


def _hyperparam_grid(X_tr, y_tr, X_te, y_te, n_classes, n_epochs=300):
    lrs      = [1e-3, 3e-3, 1e-2]
    dropouts = [0.0,  0.2,  0.4]
    grid     = np.zeros((len(dropouts), len(lrs)))
    print("  Hyperparameter grid search (lr × dropout) …")
    for i, do in enumerate(dropouts):
        for j, lr in enumerate(lrs):
            _, hist = train(X_tr, y_tr, X_te, y_te, n_classes,
                            n_epochs=n_epochs, lr=lr, dropout=do)
            best = max(hist["test_acc"]) * 100
            grid[i, j] = best
            print(f"    lr={lr:.0e}  dropout={do:.1f}  →  best test {best:.0f}%")
    return grid, lrs, dropouts


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    source  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/mlp")

    if not source.exists():
        print(f"Error: source not found: {source}"); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print(f"\n── Loading corpus from: {source} ──")
    records = load_corpus_features(source)
    if not records:
        print("No pieces loaded."); sys.exit(1)

    by_genre = Counter(r["genre"] for r in records)
    print(f"\nLoaded {len(records)} pieces:")
    for g, n in sorted(by_genre.items()):
        print(f"  {g}: {n} piece(s)")

    # 2. Split
    print("\n── Splitting: 4 train / 1 test per genre ──")
    train_records, test_records = stratified_split(records)
    print(f"  Train: {len(train_records)}  |  Test: {len(test_records)}")

    X_raw_tr, y_tr, le = to_arrays(train_records)
    X_raw_te, y_te, _  = to_arrays(test_records, le)

    # 3. Standardise
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_raw_tr).astype(np.float32)
    X_te   = scaler.transform(X_raw_te).astype(np.float32)

    n_classes = len(le.classes_)
    print(f"\n  Features     : {N_FEATURES}")
    print(f"  Classes      : {n_classes}  ({', '.join(le.classes_)})")
    print(f"  Architecture : {N_FEATURES} → 64 → 32 → {n_classes}")

    # 4. Train
    print("\n── Training MLP (600 epochs) ──")
    model, history = train(X_tr, y_tr, X_te, y_te, n_classes)

    # 5. Evaluate
    model.eval()
    with torch.no_grad():
        proba  = torch.softmax(model(torch.tensor(X_te)), dim=1).numpy()
        y_pred = proba.argmax(axis=1)

    test_acc  = history["test_acc"][-1]
    best_test = max(history["test_acc"])
    print(f"\n  Final train acc : {history['train_acc'][-1]:.1%}")
    print(f"  Final test  acc : {test_acc:.1%}  "
          f"({int(test_acc * len(y_te))}/{len(y_te)} correct)")
    print(f"  Best  test  acc : {best_test:.1%}  (over all epochs)")

    test_ids   = sorted(set(y_te))
    test_names = [le.classes_[i] for i in test_ids]
    report = classification_report(y_te, y_pred, labels=test_ids,
                                   target_names=test_names, zero_division=0)
    print(f"\n── Classification Report ──\n{report}")

    rp = out_dir / "classification_report.txt"
    with open(rp, "w") as f:
        f.write(f"Architecture   : {N_FEATURES} → 64 → 32 → {n_classes}\n")
        f.write(f"Final train acc: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final test  acc: {test_acc:.4f}\n")
        f.write(f"Best  test  acc: {best_test:.4f}\n\n")
        f.write(report)
    print(f"  Saved: {rp.name}")

    # 6. Plots
    print(f"\n── Generating plots → {out_dir}/ ──")
    plot_training_curves(history, out_dir, title="MLP Training Curves")
    plot_confusion_matrix(y_te, y_pred, test_names, test_ids, out_dir,
                          title="Confusion Matrix — Test Set (MLP)")
    plot_prediction_confidence(proba, y_te, test_records, le, out_dir)
    plot_weight_heatmap(model, out_dir)
    plot_activation_pca(model, X_tr, y_tr, X_te, y_te, le, out_dir)

    grid, lrs, dropouts = _hyperparam_grid(X_tr, y_tr, X_te, y_te, n_classes)
    plot_hyperparam_grid(grid, lrs, dropouts, out_dir, filename="06_hyperparam_grid.png")

    print(f"\n✅  Done. {len(list(out_dir.glob('*.png')))} plots + report in: {out_dir}/")


if __name__ == "__main__":
    main()
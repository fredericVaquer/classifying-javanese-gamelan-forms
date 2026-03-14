"""
Gamelan Subgenre — 1D CNN Classifier
--------------------------------------
Takes RAW note sequences as input (no hand-crafted features).

Architecture:
    Input (B, 7, T)
    ├─ Conv1d(7→32,  k=3, pad=1) → BN → ReLU → Dropout
    ├─ Conv1d(32→64, k=3, pad=1) → BN → ReLU → Dropout → MaxPool(2)
    ├─ Conv1d(64→64, k=3, pad=1) → BN → ReLU → AdaptiveAvgPool(1)
    └─ Linear(64→32) → ReLU → Linear(32→n_classes)

Usage:
    python -m src.gamelan_cnn <source_root> [output_dir]

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
from sklearn.metrics import classification_report

from data import load_corpus_sequences, stratified_split, to_tensors
from parser import N_DIMS, PAD_LEN as _PAD_LEN_DEFAULT
from plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_prediction_confidence,
    plot_filter_responses,
    plot_input_sequences,
    plot_cnn_embedding_pca,
    plot_hyperparam_grid,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

class GamelanCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=False):
            layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            if pool:
                layers.append(nn.MaxPool1d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(7,  32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.conv3 = conv_block(64, 64)
        self.pool  = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.head(self.pool(x).squeeze(-1))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """64-dim global-avg-pool embedding (before the head)."""
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.pool(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_tr: torch.Tensor, y_tr: torch.Tensor,
    X_te: torch.Tensor, y_te: torch.Tensor,
    n_classes: int,
    n_epochs: int   = 600,
    lr: float       = 3e-3,
    dropout: float  = 0.3,
    weight_decay: float = 1e-3,
    batch_size: int = 8,
) -> tuple[GamelanCNN, dict]:

    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                           shuffle=True, generator=torch.Generator().manual_seed(42))
    model     = GamelanCNN(n_classes, dropout)
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
            trl = model(X_tr); tel = model(X_te)
            history["train_loss"].append(criterion(trl, y_tr).item())
            history["test_loss"].append( criterion(tel, y_te).item())
            history["train_acc"].append( (trl.argmax(1) == y_tr).float().mean().item())
            history["test_acc"].append(  (tel.argmax(1) == y_te).float().mean().item())

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
            _, hist = train_model(X_tr, y_tr, X_te, y_te, n_classes,
                                  n_epochs=n_epochs, lr=lr, dropout=do)
            best = max(hist["test_acc"]) * 100
            grid[i, j] = best
            print(f"    lr={lr:.0e}  dropout={do:.1f}  →  best test {best:.0f}%")
    return grid, lrs, dropouts


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    source  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Javanese Gamelan Notation")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Gamelan_CNN_Output")

    if not source.exists():
        print(f"Error: source not found: {source}"); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print(f"\n── Loading corpus from: {source} ──")
    records = load_corpus_sequences(source)
    if not records:
        print("No pieces loaded."); sys.exit(1)

    # Dynamic PAD_LEN: 95th-percentile of non-zero sequence lengths
    seq_lengths = [r["seq_len"] for r in records]
    pad_len     = int(np.percentile([l for l in seq_lengths if l > 0], 95))
    n_truncated = sum(1 for l in seq_lengths if l > pad_len)
    print(f"\nSequence lengths — min: {min(seq_lengths)}  "
          f"max: {max(seq_lengths)}  mean: {np.mean(seq_lengths):.0f}  "
          f"p95: {pad_len}  (truncating {n_truncated} outlier(s))")

    by_genre = Counter(r["genre"] for r in records)
    print(f"\nLoaded {len(records)} pieces:")
    for g, n in sorted(by_genre.items()):
        print(f"  {g}: {n} piece(s)")

    # 2. Split
    print("\n── Splitting: 4 train / 1 test per genre ──")
    train_records, test_records = stratified_split(records)
    print(f"  Train: {len(train_records)}  |  Test: {len(test_records)}")

    X_tr, y_tr, le = to_tensors(train_records, pad_len)
    X_te, y_te, _  = to_tensors(test_records,  pad_len, le)

    n_classes = len(le.classes_)
    print(f"\n  Input shape  : {tuple(X_tr.shape)}  (N, channels={N_DIMS}, time={pad_len})")
    print(f"  Classes      : {n_classes}  ({', '.join(le.classes_)})")
    print(f"  Architecture : ({N_DIMS},{pad_len}) → Conv(32) → Conv(64) → Conv(64) → {n_classes}")

    # 3. Train
    print("\n── Training CNN (600 epochs) ──")
    model, history = train_model(X_tr, y_tr, X_te, y_te, n_classes)

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        proba  = torch.softmax(model(X_te), dim=1).numpy()
        y_pred = proba.argmax(axis=1)

    y_te_np   = y_te.numpy()
    test_acc  = history["test_acc"][-1]
    best_test = max(history["test_acc"])
    print(f"\n  Final train acc : {history['train_acc'][-1]:.1%}")
    print(f"  Final test  acc : {test_acc:.1%}  "
          f"({int(test_acc * len(y_te_np))}/{len(y_te_np)} correct)")
    print(f"  Best  test  acc : {best_test:.1%}  (over all epochs)")

    test_ids   = sorted(set(y_te_np))
    test_names = [le.classes_[i] for i in test_ids]
    report = classification_report(y_te_np, y_pred, labels=test_ids,
                                   target_names=test_names, zero_division=0)
    print(f"\n── Classification Report ──\n{report}")

    rp = out_dir / "classification_report.txt"
    with open(rp, "w") as f:
        f.write(f"Architecture   : ({N_DIMS},{pad_len}) → Conv(32) → Conv(64) → Conv(64) → {n_classes}\n")
        f.write(f"PAD_LEN (p95)  : {pad_len}\n")
        f.write(f"Final train acc: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final test  acc: {test_acc:.4f}\n")
        f.write(f"Best  test  acc: {best_test:.4f}\n\n")
        f.write(report)
    print(f"  Saved: {rp.name}")

    # 5. Plots
    print(f"\n── Generating plots → {out_dir}/ ──")
    plot_training_curves(history, out_dir, title="1D-CNN Training Curves")
    plot_confusion_matrix(y_te_np, y_pred, test_names, test_ids, out_dir,
                          title="Confusion Matrix — Test Set (CNN)")
    plot_prediction_confidence(proba, y_te_np, test_records, le, out_dir)
    plot_filter_responses(model, X_te, test_records, le, out_dir)
    plot_input_sequences(X_te, test_records, out_dir)
    plot_cnn_embedding_pca(model, X_tr, y_tr, X_te, y_te, le, out_dir)

    grid, lrs, dropouts = _hyperparam_grid(X_tr, y_tr, X_te, y_te, n_classes)
    plot_hyperparam_grid(grid, lrs, dropouts, out_dir, filename="07_hyperparam_grid.png")

    print(f"\n✅  Done. {len(list(out_dir.glob('*.png')))} plots + report in: {out_dir}/")


if __name__ == "__main__":
    main()
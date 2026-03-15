"""
src/plots.py
------------
All matplotlib plotting functions, shared across classifiers.

Shared plots (DT, MLP, CNN):
  plot_confusion_matrix()
  plot_prediction_confidence()
  plot_embedding_pca()
  plot_hyperparam_grid()

DT-specific:
  plot_feature_importance()
  plot_decision_tree()
  plot_feature_scatter()
  plot_depth_sweep()

MLP-specific:
  plot_training_curves()        (also used by CNN)
  plot_weight_heatmap()
  plot_activation_pca()         (MLP hidden layer variant)

CNN-specific:
  plot_filter_responses()
  plot_input_sequences()
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from .features import FEATURE_NAMES, N_FEATURES


# ── Palette + axis style ──────────────────────────────────────────────────────

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC", "#9B2335",
]


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#F8F5F0")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    if title:  ax.set_title(title,  fontsize=11, fontweight="bold", color="#222222", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9,  color="#555555")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9,  color="#555555")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  Shared plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true, y_pred, classes: list[str], label_ids: list[int],
    out_dir: Path, title: str = "Confusion Matrix — Test Set",
) -> None:
    cm  = confusion_matrix(y_true, y_pred, labels=label_ids)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) * 0.9)))
    fig.patch.set_facecolor("#FAFAF8")
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#111111", pad=10)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "02_confusion_matrix.png")


def plot_prediction_confidence(
    proba: np.ndarray, y_test: np.ndarray,
    test_records: list[dict], le: LabelEncoder, out_dir: Path,
) -> None:
    genres  = le.classes_
    n       = len(test_records)
    xlabels = [f"{r['genre']}\n({r['song_name'][:28]})" for r in test_records]
    y_pred  = proba.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))
    fig.patch.set_facecolor("#FAFAF8")
    x     = np.arange(n)
    width = 0.8 / len(genres)

    for k, genre in enumerate(genres):
        ax.bar(
            x + k * width - (len(genres) - 1) * width / 2,
            proba[:, k], width, label=genre,
            color=PALETTE[k % len(PALETTE)], alpha=0.82,
            edgecolor="white", linewidth=0.5,
        )

    for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
        sym, col = ("✓", "#1a7a1a") if yt == yp else ("✗", "#cc0000")
        ax.text(x[i], 1.02, sym, ha="center", va="bottom",
                fontsize=13, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, ha="center")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Genre", fontsize=8, bbox_to_anchor=(1.01, 1), framealpha=0.7)
    _style(ax, "Softmax Probability per Test Song  (✓=correct  ✗=wrong)",
           ylabel="Probability")
    fig.tight_layout()
    _save(fig, out_dir / "03_prediction_confidence.png")


def plot_embedding_pca(
    emb_tr: np.ndarray, y_tr: np.ndarray,
    emb_te: np.ndarray, y_te: np.ndarray,
    le: LabelEncoder, out_dir: Path,
    title: str = "Learned Embeddings — PCA 2D",
    filename: str = "pca_embeddings.png",
) -> None:
    """Generic PCA scatter for any embeddings (MLP hidden layer or CNN feature map)."""
    all_emb  = np.vstack([emb_tr, emb_te])
    centered = all_emb - all_emb.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs      = centered @ Vt[:2].T
    var_exp  = S[:2] ** 2 / (S ** 2).sum() * 100

    pcs_tr = pcs[:len(emb_tr)]
    pcs_te = pcs[len(emb_tr):]
    classes = le.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#F8F5F0")

    for k, genre in enumerate(classes):
        c = PALETTE[k % len(PALETTE)]
        ax.scatter(pcs_tr[y_tr == k, 0], pcs_tr[y_tr == k, 1],
                   color=c, s=65, alpha=0.75, edgecolors="white",
                   linewidths=0.5, zorder=3, label=genre)
        if np.any(y_te == k):
            ax.scatter(pcs_te[y_te == k, 0], pcs_te[y_te == k, 1],
                       color=c, s=180, marker="*", edgecolors="#111111",
                       linewidths=0.8, zorder=5)

    patches = [mpatches.Patch(color=PALETTE[k % len(PALETTE)], label=g)
               for k, g in enumerate(classes)]
    ax.legend(
        handles=patches + [
            plt.Line2D([0], [0], marker="*", color="gray", markersize=10,
                       linestyle="", label="Test"),
            plt.Line2D([0], [0], marker="o", color="gray", markersize=7,
                       linestyle="", label="Train"),
        ],
        fontsize=8, bbox_to_anchor=(1.01, 1), framealpha=0.7,
    )
    _style(ax, title,
           f"PC1  ({var_exp[0]:.1f}% var)",
           f"PC2  ({var_exp[1]:.1f}% var)")
    ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
    fig.tight_layout()
    _save(fig, out_dir / filename)


def plot_hyperparam_grid(
    grid: np.ndarray,
    lrs: list, dropouts: list,
    out_dir: Path, filename: str = "hyperparam_grid.png",
) -> None:
    """
    Render a pre-computed accuracy grid (dropouts × lrs).
    The caller is responsible for running the grid search and passing results.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#FAFAF8")
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(lrs)));      ax.set_xticklabels([str(l) for l in lrs],      fontsize=9)
    ax.set_yticks(range(len(dropouts))); ax.set_yticklabels([str(d) for d in dropouts], fontsize=9)
    ax.set_xlabel("Learning rate", fontsize=9, color="#555555")
    ax.set_ylabel("Dropout",       fontsize=9, color="#555555")
    ax.set_title("Hyperparameter Grid — Best Test Accuracy (%)",
                 fontsize=11, fontweight="bold", color="#222222", pad=8)
    for i in range(len(dropouts)):
        for j in range(len(lrs)):
            ax.text(j, i, f"{grid[i, j]:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if grid[i, j] > 60 else "black")
    plt.colorbar(im, ax=ax, label="Best test accuracy (%)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / filename)


# ══════════════════════════════════════════════════════════════════════════════
#  Decision Tree plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(clf, X_train: np.ndarray, y_train: np.ndarray,
                             out_dir: Path) -> None:
    from sklearn.inspection import permutation_importance
    from sklearn.tree import DecisionTreeClassifier  # type: ignore

    gini_imp = clf.feature_importances_
    perm     = permutation_importance(clf, X_train, y_train,
                                      n_repeats=30, random_state=42)
    top_n    = min(15, N_FEATURES)
    idx_g    = np.argsort(gini_imp)[::-1]
    idx_p    = np.argsort(perm.importances_mean)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAF8")

    ax1.barh([FEATURE_NAMES[i] for i in idx_g[:top_n]][::-1],
             [gini_imp[i] for i in idx_g[:top_n]][::-1],
             color="#457B9D", alpha=0.85, edgecolor="white")
    _style(ax1, "Gini Feature Importance (top 15)", xlabel="Importance score")
    ax1.yaxis.grid(False)

    means = [perm.importances_mean[i] for i in idx_p[:top_n]][::-1]
    stds  = [perm.importances_std[i]  for i in idx_p[:top_n]][::-1]
    ax2.barh([FEATURE_NAMES[i] for i in idx_p[:top_n]][::-1],
             means, xerr=stds, color="#2A9D8F", alpha=0.85, edgecolor="white",
             error_kw=dict(ecolor="#555555", capsize=3, linewidth=0.8))
    _style(ax2, "Permutation Importance (top 15, ±std)", xlabel="Mean accuracy decrease")
    ax2.yaxis.grid(False)

    fig.suptitle("Feature Importance Analysis", fontsize=13, fontweight="bold",
                 color="#111111", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "02_feature_importance.png")


def plot_decision_tree(clf, classes: list[str], out_dir: Path) -> None:
    from sklearn.tree import plot_tree  # type: ignore

    n_leaves = clf.get_n_leaves()
    depth    = clf.get_depth()
    fig, ax  = plt.subplots(figsize=(max(20, n_leaves * 2.5), max(8, depth * 2.5)))
    fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#F8F5F0")
    plot_tree(clf, ax=ax, feature_names=FEATURE_NAMES, class_names=classes,
              filled=True, rounded=True, impurity=True, proportion=False,
              fontsize=max(5, min(9, 60 // max(n_leaves, 1))))
    ax.set_title(f"Decision Tree  (depth={depth}, leaves={n_leaves})",
                 fontsize=13, fontweight="bold", color="#111111", pad=12)
    fig.tight_layout()
    _save(fig, out_dir / "03_decision_tree.png")


def plot_prediction_confidence_clf(
    clf, X_test: np.ndarray, y_test: np.ndarray,
    test_records: list[dict], le: LabelEncoder, out_dir: Path,
) -> None:
    """Decision-tree variant: uses clf.predict_proba() and clf.predict()."""
    proba  = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    genres = le.classes_
    n      = len(test_records)
    xlabels = [f"{r['genre']}\n({r['song_name'][:30]})" for r in test_records]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 6))
    fig.patch.set_facecolor("#FAFAF8")
    x     = np.arange(n)
    width = 0.8 / len(genres)

    for k, genre in enumerate(genres):
        g_idx = list(le.classes_).index(genre)
        ax.bar(x + k * width - (len(genres) - 1) * width / 2,
               proba[:, g_idx], width, label=genre,
               color=PALETTE[k % len(PALETTE)], alpha=0.82,
               edgecolor="white", linewidth=0.5)

    for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
        sym, col = ("✓", "#1a7a1a") if yt == yp else ("✗", "#cc0000")
        ax.text(x[i], 1.02, sym, ha="center", va="bottom",
                fontsize=13, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, ha="center")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Genre", fontsize=8, bbox_to_anchor=(1.01, 1), framealpha=0.7)
    _style(ax, "Prediction Probability per Test Song  (✓=correct, ✗=wrong)",
           ylabel="Probability")
    fig.tight_layout()
    _save(fig, out_dir / "04_prediction_confidence.png")


def plot_feature_scatter(clf, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          le: LabelEncoder, out_dir: Path) -> None:
    top2 = np.argsort(clf.feature_importances_)[::-1][:2]
    fx, fy = top2[0], top2[1]
    classes = le.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#F8F5F0")

    for k, genre in enumerate(classes):
        c = PALETTE[k % len(PALETTE)]
        ax.scatter(X_train[y_train == k, fx], X_train[y_train == k, fy],
                   color=c, alpha=0.7, s=55, edgecolors="white",
                   linewidths=0.5, label=genre, zorder=3)
        ax.scatter(X_test[y_test == k, fx], X_test[y_test == k, fy],
                   color=c, s=140, marker="*", edgecolors="#111111",
                   linewidths=0.8, zorder=5)

    patches = [mpatches.Patch(color=PALETTE[k % len(PALETTE)], label=g)
               for k, g in enumerate(classes)]
    ax.legend(
        handles=patches + [
            plt.Line2D([0], [0], marker="*", color="gray", markersize=10,
                       linestyle="", label="Test piece"),
            plt.Line2D([0], [0], marker="o", color="gray", markersize=7,
                       linestyle="", label="Train piece"),
        ],
        fontsize=8, bbox_to_anchor=(1.01, 1), framealpha=0.7,
    )
    _style(ax, "Feature Space — top-2 Gini features",
           xlabel=FEATURE_NAMES[fx], ylabel=FEATURE_NAMES[fy])
    ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
    fig.tight_layout()
    _save(fig, out_dir / "05_feature_scatter.png")


def plot_depth_sweep(depths: list[int], train_acc: list[float],
                      test_acc: list[float], best_depth: int,
                      out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.plot(depths, train_acc, color="#457B9D", lw=2,
            marker="o", markersize=5, label="Train accuracy")
    ax.plot(depths, test_acc,  color="#E63946", lw=2,
            marker="s", markersize=5, label="Test accuracy")
    ax.axvline(best_depth, color="#2A9D8F", lw=1.5, linestyle="--",
               label=f"Best depth = {best_depth}")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.7)
    _style(ax, "Tree Depth vs Accuracy", "max_depth", "Accuracy")
    fig.tight_layout()
    _save(fig, out_dir / "06_depth_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MLP + CNN shared: training curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, out_dir: Path,
                          title: str = "Training Curves") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#FAFAF8")

    ax1.plot(epochs, history["train_loss"], color="#457B9D", lw=1.5, label="Train")
    ax1.plot(epochs, history["test_loss"],  color="#E63946", lw=1.5, label="Test")
    ax1.legend(fontsize=9, framealpha=0.7)
    _style(ax1, "Cross-Entropy Loss", "Epoch", "Loss")

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
             color="#457B9D", lw=1.5, label="Train")
    ax2.plot(epochs, [a * 100 for a in history["test_acc"]],
             color="#E63946", lw=1.5, label="Test")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9, framealpha=0.7)
    _style(ax2, "Accuracy", "Epoch", "Accuracy (%)")

    fig.suptitle(title, fontsize=13, fontweight="bold", color="#111111", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "01_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MLP-specific plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_weight_heatmap(model, out_dir: Path) -> None:
    """Visualise W₁ (first Linear layer) of a GamelanMLP."""
    W1 = model.net[0].weight.detach().cpu().numpy()   # (64, N_FEATURES)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#FAFAF8")

    im = ax1.imshow(W1, aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(W1).max(), vmax=np.abs(W1).max())
    ax1.set_xticks(range(N_FEATURES))
    ax1.set_xticklabels(FEATURE_NAMES, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("Hidden neuron (layer 1)", fontsize=9, color="#555555")
    ax1.set_title("W₁  —  Input → Hidden-1 weights  (64 neurons × 29 features)",
                  fontsize=11, fontweight="bold", color="#222222", pad=8)
    ax1.set_facecolor("#F8F5F0")
    plt.colorbar(im, ax=ax1, shrink=0.6, label="Weight value")

    norms = np.linalg.norm(W1, axis=0)
    ax2.bar(range(N_FEATURES), norms,
            color="#457B9D", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax2.set_xticks(range(N_FEATURES))
    ax2.set_xticklabels(FEATURE_NAMES, rotation=60, ha="right", fontsize=7)
    _style(ax2, "Column L₂-norm of W₁  (feature influence proxy)", ylabel="L₂ norm")

    fig.tight_layout()
    _save(fig, out_dir / "04_weight_heatmap.png")


def plot_activation_pca(model, X_tr, y_tr, X_te, y_te,
                         le: LabelEncoder, out_dir: Path) -> None:
    """PCA of MLP first-layer activations."""
    import torch
    model.eval()
    act_tr = model.hidden_activations(
        torch.tensor(X_tr, dtype=torch.float32)).numpy()
    act_te = model.hidden_activations(
        torch.tensor(X_te, dtype=torch.float32)).numpy()
    plot_embedding_pca(
        act_tr, y_tr, act_te, y_te, le, out_dir,
        title="Hidden-Layer Activations — PCA 2D  (internal representation)",
        filename="05_activation_pca.png",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CNN-specific plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_filter_responses(model, X_te, test_records: list[dict],
                           le: LabelEncoder, out_dir: Path) -> None:
    import torch
    model.eval()
    with torch.no_grad():
        acts = model.conv1(X_te).numpy()   # (N_test, 32, T)

    y_pred_genres = [le.classes_[i] for i in
                     model(X_te).argmax(1).detach().numpy()]
    n    = len(test_records)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    fig.patch.set_facecolor("#FAFAF8")
    axes = np.array(axes).flatten()

    for i, rec in enumerate(test_records):
        ax  = axes[i]
        img = acts[i]
        ax.imshow(img, aspect="auto", cmap="RdBu_r",
                  vmin=-np.abs(img).max(), vmax=np.abs(img).max(),
                  interpolation="nearest")
        true_g, pred_g = rec["genre"], y_pred_genres[i]
        colour = "#1a7a1a" if true_g == pred_g else "#cc0000"
        ax.set_title(f"{rec['song_name'][:30]}\ntrue: {true_g}  pred: {pred_g}",
                     fontsize=7.5, color=colour, fontweight="bold")
        ax.set_xlabel("Time (note index)", fontsize=7, color="#555555")
        ax.set_ylabel("Filter index",      fontsize=7, color="#555555")
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Conv-1 Filter Activations per Test Piece  (32 filters × time)",
                 fontsize=12, fontweight="bold", color="#111111", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "04_filter_responses.png")


def plot_input_sequences(X_te, test_records: list[dict],
                          out_dir: Path) -> None:
    dim_labels = ["pitch", "octave", "kenong", "kethuk", "gong", "kempyang", "is_rest"]
    n    = len(test_records)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.5))
    fig.patch.set_facecolor("#FAFAF8")
    axes = np.array(axes).flatten()

    for i, rec in enumerate(test_records):
        ax  = axes[i]
        seq = X_te[i].numpy()   # (7, PAD_LEN)
        nonzero_cols = np.where(seq.any(axis=0))[0]
        end = int(nonzero_cols[-1]) + 1 if len(nonzero_cols) else seq.shape[1]
        ax.imshow(seq[:, :end], aspect="auto", cmap="viridis",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(7))
        ax.set_yticklabels(dim_labels, fontsize=7)
        ax.set_xlabel("Note index", fontsize=7, color="#555555")
        ax.set_title(f"{rec['song_name'][:32]}\n({rec['genre']})",
                     fontsize=7.5, fontweight="bold", color="#222222")
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Raw Input Sequences — what the CNN sees  (7 channels × time)",
                 fontsize=12, fontweight="bold", color="#111111", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "05_input_sequences.png")


def plot_cnn_embedding_pca(model, X_tr, y_tr, X_te, y_te,
                            le: LabelEncoder, out_dir: Path) -> None:
    """PCA of CNN global-avg-pool embeddings."""
    import torch
    model.eval()
    emb_tr = model.feature_map(
        torch.tensor(X_tr, dtype=torch.float32) if not hasattr(X_tr, "numpy") else X_tr
    ).numpy()
    emb_te = model.feature_map(
        torch.tensor(X_te, dtype=torch.float32) if not hasattr(X_te, "numpy") else X_te
    ).numpy()
    plot_embedding_pca(
        emb_tr, y_tr.numpy(), emb_te, y_te.numpy(), le, out_dir,
        title="CNN Learned Embeddings — PCA 2D  (after global avg pool)",
        filename="06_embedding_pca.png",
    )
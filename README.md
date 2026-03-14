# Classifying Javanese Gamelan Forms

Symbolic music analysis pipeline for classifying Javanese gamelan compositions by subgenre (Ayak-Ayak, Bubaran, Ketawang, Ladrang, Lancaran, Sampak, Srepegan) directly from kepatihan (cipher) notation PDFs.

Three classifiers are implemented and can be compared side-by-side:

| Model | Input | Key idea |
|---|---|---|
| Decision Tree | 29 hand-crafted features | Interpretable split rules, depth sweep |
| MLP | 29 hand-crafted features | Learns non-linear boundaries over same features |
| 1D CNN | Raw padded note sequences | No feature engineering — kernels slide over time |

---

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── parser.py            # PDF extraction, Note class, sequence encoding
│   ├── features.py          # 29-dim hand-crafted feature vectors (DT + MLP)
│   ├── data.py              # Corpus loading and stratified train/test split
│   ├── plots.py             # All matplotlib plotting functions
│   ├── gamelan_classifier.py  # Decision Tree model + main
│   ├── gamelan_mlp.py         # MLP model + training + main
│   └── gamelan_cnn.py         # 1D CNN model + training + main
│
├── Javanese Gamelan Notation/   # Input data (not tracked)
│   ├── Ayak Ayak/
│   │   └── Ayak Ayak Nem Slendro Nem/
│   │       └── Ayak Ayak Nem Slendro Nem.pdf
│   ├── Bubaran/
│   └── ...
│
├── Gamelan_Classifier_Output/   # Generated on run
├── Gamelan_MLP_Output/
├── Gamelan_CNN_Output/
└── README.md
```

---

## Data Format

PDF scores use **kepatihan cipher notation** with a custom Balungan TrueType font. The parser (`src/parser.py`) maps font glyphs to pitch and octave:

| Letter range | Octave | Scale degrees |
|---|---|---|
| `A`–`G` | Low (dot below) | 7, 6, 5, 4, 3, 2, 1 |
| `H`–`N` | Mid (no dot) | 1, 2, 3, 4, 5, 6, 7 |
| `O`–`U` | High (dot above) | 1, 2, 3, 4, 5, 6, 7 |

Beat marker characters overlay notes:

| Char | Meaning |
|---|---|
| `)` | Kenong |
| `^` | Kethuk |
| `@` | Gong ageng |
| `(` | Kempyang / gong suwukan |
| `[` / `]` | Repeat section start / end |
| `-` | Rest |

---

## Input Folder Structure

Each genre is a top-level folder. Each piece lives in its own subfolder, and the PDF filename must exactly match the subfolder name:

```
Javanese Gamelan Notation/
└── Ladrang/
    ├── Ladrang Mugi Rahayu Slendro Manyura/
    │   └── Ladrang Mugi Rahayu Slendro Manyura.pdf   ✅ matched
    └── Ladrang Mugi Rahayu Slendro Manyura.pdf        ❌ ignored
```

---

## Installation

```bash
pip install pdfplumber scikit-learn matplotlib numpy torch
```

Python 3.10+ recommended.

---

## Usage

Run all three from the **project root** (parent of `src/`):

```bash
python src/gamelan_classifier "Javanese Gamelan Notation" "Gamelan_Classifier_Output"
python src/gamelan_mlp        "Javanese Gamelan Notation" "Gamelan_MLP_Output"
python src/gamelan_cnn        "Javanese Gamelan Notation" "Gamelan_CNN_Output"
```

The second argument (output directory) is optional and defaults to the names above.

---

## Train / Test Split

All three models use the same stratified split logic (`src/data.py`):

- Songs are sorted alphabetically per genre for reproducibility
- **≥ 5 songs**: first 4 → train, 5th → test
- **4 songs**: first 3 → train, 4th → test *(LOO fallback, printed as ℹ️)*
- **< 4 songs**: all train, no test sample *(printed as ⚠️)*

With the current corpus (7 genres, 33 pieces), this yields **26 train / 7 test**.

---

## Features (Decision Tree + MLP)

The `extract_features()` function in `src/features.py` produces a **29-dimensional float32 vector** per piece:

| Group | Features | Count |
|---|---|---|
| Pitch distribution | % of each scale degree 1–7 | 7 |
| Register | % notes in low / mid / high octave | 3 |
| Complexity | Avg beat markers per note, rest ratio | 2 |
| Intervals | Step ratio, leap ratio, mean absolute interval | 3 |
| Gong structure | Mean / std / count of notes per gong cycle | 3 |
| Repeats | Has repeat bracket (bool), repeat count | 2 |
| Section presence | Buka, Merong, Inggah, Ngelik, Umpak, Suwuk | 6 |
| Pitch entropy | Shannon entropy over pitch distribution | 1 |
| Melodic contour | Ascending ratio, direction change rate | 2 |
| **Total** | | **29** |

---

## Models

### Decision Tree (`src/gamelan_classifier.py`)

A `sklearn.tree.DecisionTreeClassifier` with a depth sweep (depths 1–15) to find the best generalising depth before fitting the final model. No standardisation needed.

**Output plots:**

| File | Description |
|---|---|
| `01_confusion_matrix.png` | Predicted vs actual on test set |
| `02_feature_importance.png` | Gini importance + permutation importance side by side |
| `03_decision_tree.png` | Full tree visualisation with filled, coloured nodes |
| `04_prediction_confidence.png` | Per-song class probabilities with ✓/✗ |
| `05_feature_scatter.png` | 2D scatter on top-2 Gini features |
| `06_depth_sweep.png` | Train vs test accuracy across depths, best depth marked |
| `classification_report.txt` | sklearn report + full tree text rules |

---

### MLP (`src/gamelan_mlp.py`)

A two-hidden-layer PyTorch MLP. Features are z-scored with `StandardScaler` before training (critical for gradient-based optimisation).

**Architecture:**
```
Input(29) → StandardScaler
          → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(n_classes)
```

**Training:** AdamW + CosineAnnealingLR, 600 epochs, batch size 8.

**Output plots:**

| File | Description |
|---|---|
| `01_training_curves.png` | Loss and accuracy over all epochs |
| `02_confusion_matrix.png` | Predicted vs actual on test set |
| `03_prediction_confidence.png` | Softmax probabilities per test song |
| `04_weight_heatmap.png` | W₁ heatmap (64×29) + column L₂ norms |
| `05_activation_pca.png` | First hidden layer activations projected to 2D via PCA |
| `06_hyperparam_grid.png` | 3×3 grid: lr × dropout → best test accuracy |

---

### 1D CNN (`src/gamelan_cnn.py`)

Operates on **raw note sequences** — no hand-crafted features. Each note/rest is encoded as a 7-dim vector:

| Channel | Value |
|---|---|
| 0 | Pitch normalised: `pitch / 7` (0 = rest) |
| 1 | Octave normalised: `(octave + 1) / 2` (0.5 for rests) |
| 2 | Kenong (bool) |
| 3 | Kethuk (bool) |
| 4 | Gong (bool) |
| 5 | Kempyang (bool) |
| 6 | Is rest (bool) |

Sequences are padded / truncated to the **95th-percentile length** across the corpus (to avoid one outlier dictating tensor size for everyone).

**Architecture:**
```
Input (B, 7, T)
├─ Conv1d(7→32,  k=3, pad=1) → BatchNorm → ReLU → Dropout
├─ Conv1d(32→64, k=3, pad=1) → BatchNorm → ReLU → Dropout → MaxPool(2)
├─ Conv1d(64→64, k=3, pad=1) → BatchNorm → ReLU → AdaptiveAvgPool(1)
└─ Linear(64→32) → ReLU → Linear(32→n_classes)
```

**Training:** AdamW + CosineAnnealingLR, 600 epochs, batch size 8.

**Output plots:**

| File | Description |
|---|---|
| `01_training_curves.png` | Loss and accuracy over all epochs |
| `02_confusion_matrix.png` | Predicted vs actual on test set |
| `03_prediction_confidence.png` | Softmax probabilities per test song |
| `04_filter_responses.png` | Conv-1 filter activations (32 filters × time) per test piece |
| `05_input_sequences.png` | Raw 7-channel input heatmap for each test piece |
| `06_embedding_pca.png` | Global-avg-pool embeddings projected to 2D via PCA |
| `07_hyperparam_grid.png` | 3×3 grid: lr × dropout → best test accuracy |

---

## Module Dependency Graph

```
parser.py
├── features.py
│   └── data.py (load_corpus_features, to_arrays)
├── data.py     (load_corpus_sequences, to_tensors)
└── plots.py
    ├── gamelan_classifier.py
    ├── gamelan_mlp.py
    └── gamelan_cnn.py
```

Nothing in `parser.py`, `features.py`, `data.py`, or `plots.py` imports from the model files — the dependency graph is strictly one-directional.

---

## Notes and Caveats

**Small dataset.** With ~33 pieces across 7 genres, all three models are susceptible to overfitting. The depth sweep (DT) and hyperparam grid (MLP, CNN) are specifically designed to surface this. Take test accuracy figures as indicative rather than definitive.

**Uneven genres.** Ayak Ayak and Sampak have 4 pieces each. The LOO fallback uses 3 train / 1 test for these genres so they still contribute a test sample, but their training representation is thinner than the 5-piece genres.

**Parsing edge cases.** `Srepegan Manyura` parses to 0 events (likely a PDF encoding issue). The CNN handles this gracefully with an all-zeros placeholder sequence. `Ayak Ayak Pamungkas` produces ~3400 events (repeated sections parsed as flat sequence) — the p95 truncation ensures this doesn't distort the padded tensor size for the rest of the corpus.

**PyTorch dependency.** The MLP and CNN require `torch`. The Decision Tree runs on `scikit-learn` only. All three share `pdfplumber`, `numpy`, and `matplotlib`.
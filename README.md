# Automatic Classification of Javanese Gamelan Forms from Kepatihan Cipher Notation Using Machine Learning

A symbolic music analysis and classification pipeline for automatically identifying Javanese gamelan **bentuk** (cyclic architectural forms) directly from Kepatihan (cipher) notation PDFs. This project classifies 7 traditional forms — Ayak-Ayak, Bubaran, Ketawang, Ladrang, Lancaran, Sampak, and Srepegan — using both interpretable and deep learning models.

## Motivation

Most research in Music Information Retrieval (MIR) applied to non-Western traditions relies on audio-based deep learning models that act as "black boxes." This project addresses the interpretability gap by working directly with **symbolic notation**, enabling the extraction of musicologically meaningful features and the generation of human-readable decision rules. The approach combines Explainable AI (Decision Trees, Random Forest) with neural models (MLP, 1D CNN) to determine which method best suits the small-data, high-structure regime typical of ethnomusicological datasets.

## Dataset

The project uses the **Javanese Gamelan Notation Dataset** (Kurniawati et al., *Data in Brief*, 2024):

- **35 original compositions** across **7 bentuk** (5 pieces each) in `dataset/`
- Notation in **Kepatihan cipher format** using a custom Balungan TrueType font
- Each piece folder contains the main melody PDF plus instrument-specific notation PDFs (Balungan, Bonang Barung & Bonang Penerus, Peking, Structural Instruments)
- Classification uses only the **main melody PDF** per piece

| Form (Bentuk) | Pieces | Scale (Laras)       | Characteristics                    |
|----------------|--------|---------------------|------------------------------------|
| Ayak-Ayak      | 5      | Slendro             | Irregular gatras per gong          |
| Bubaran        | 5      | Pelog / Slendro     | 4 gatras per gong, similar to Lancaran |
| Ketawang       | 5      | Pelog / Slendro     | 4 gatras per gong, includes kempyang  |
| Ladrang        | 5      | Pelog / Slendro     | 8 gatras per gong, includes kempyang  |
| Lancaran       | 5      | Pelog / Slendro     | 4 gatras per gong                  |
| Sampak         | 5      | Slendro             | 2–4 gatras per gong, fast tempo    |
| Srepegan       | 5      | Slendro             | Irregular gatras per gong          |

### Augmented Datasets

Two augmented versions are provided, both created via **pitch transposition** — each original piece is shifted by ±1–7 scale degrees within the 7-tone Kepatihan system:

| Directory | Pieces | Balance | Generation |
|-----------|--------|---------|------------|
| `dataset_augmented/` | 192 | Unbalanced (11–45 per form) | `--no-balance` |
| `dataset_augmented_balanced/` | 77 | Balanced (11 per form) | Default |

The balanced variant caps every genre to the minimum natural yield (11 pieces, bounded by Ayak-Ayak), eliminating class imbalance entirely. The unbalanced variant retains all valid transpositions and relies on class weighting to compensate.

```bash
python src/make_augmented_dataset.py --dst dataset_augmented_balanced          # balanced (default)
python src/make_augmented_dataset.py --no-balance --dst dataset_augmented      # unbalanced
```

**Citation:**
> Kurniawati, A., Yuniarno, E. M., Suprapto, Y. K., Ifada, N., & Soewidiatmaka, N. I. (2024).
> Notation of Javanese Gamelan dataset for traditional music applications. *Data in Brief*, 53, 110116.
> [DOI: 10.1016/j.dib.2024.110116](https://doi.org/10.1016/j.dib.2024.110116)

## Project Structure

```
.
├── gamelan_classification.ipynb               # Baseline experiment — 35 original pieces
├── gamelan_classification_augmented.ipynb      # Augmented experiment — balanced vs unbalanced (start here)
├── requirements.txt                           # Python dependencies
├── README.md
├── LICENSE
│
├── src/                              # Core Python package
│   ├── __init__.py
│   ├── parser.py                     # PDF extraction, Note class, sequence encoding
│   ├── features.py                   # 29-dim hand-crafted feature vectors
│   ├── data.py                       # Corpus loading, stratified split, LOGO grouping
│   ├── plots.py                      # Matplotlib plotting functions
│   ├── statistical_analysis.py       # Corpus-level statistical analysis and plots
│   ├── gamelan_classifier.py         # Decision Tree (CLI)
│   ├── gamelan_mlp.py                # MLP classifier (CLI)
│   ├── gamelan_cnn.py                # 1D CNN classifier (CLI)
│   └── make_augmented_dataset.py    # Pitch-transposition augmentation script
│
├── dataset/                          # Original Kepatihan notation PDFs (35 pieces)
│   ├── Ayak Ayak/
│   ├── Bubaran/
│   ├── Ketawang/
│   ├── Ladrang/
│   ├── Lancaran/
│   ├── Sampak/
│   └── Srepegan/
│
├── dataset_augmented/                # Pitch-transposed augmented PDFs — unbalanced (192 pieces)
│   └── (same genre structure as dataset/)
│
├── dataset_augmented_balanced/       # Pitch-transposed augmented PDFs — balanced (77 pieces)
│   └── (same genre structure as dataset/)
│
├── docs/                             # Reference papers
│   ├── prototype.pdf                 # Project proposal
│   ├── gamelan-dataset-paper.pdf     # Dataset paper (Kurniawati et al., 2024)
│   └── gamelan-cnn.pdf               # Related work (Savitri et al., 2025)
│
├── plots/                            # Pre-generated statistical analysis plots
└── output/                           # Generated model outputs (gitignored)
```

## Installation

```bash
git clone <repository-url>
cd classifying-javanese-gamelan-forms
pip install -r requirements.txt
```

**Python 3.10+** is required. PyTorch is needed only for the MLP and CNN models; the classical models (Decision Tree, Random Forest, SVM, KNN) require only scikit-learn.

## Usage

### Recommended: Jupyter Notebooks

Two notebooks are provided:

| Notebook | Dataset | Pieces | CV Strategy | Purpose |
|----------|---------|--------|-------------|---------|
| `gamelan_classification.ipynb` | `dataset/` | 35 | LOOCV | Baseline experiment |
| `gamelan_classification_augmented.ipynb` | Both augmented | 77 + 192 | LOGOCV (all 6 models) | Balanced vs unbalanced comparison |

```bash
jupyter notebook gamelan_classification_augmented.ipynb
```

Both notebooks cover dataset exploration, feature extraction, EDA, all six models with evaluation, and a comparative summary. The augmented notebook uses leak-free Leave-One-Group-Out CV to ensure no transposition of a test piece appears in training.

### Generating the Augmented Dataset

The augmented dataset is built from the original 35-piece corpus via pitch transposition:

```bash
python src/make_augmented_dataset.py                          # balanced (default)
python src/make_augmented_dataset.py --no-balance             # keep all valid shifts
python src/make_augmented_dataset.py --src dataset --dst dataset_augmented
```

The script shifts each piece by ±1–7 scale degrees, keeping only transpositions where every note remains within the 3-octave font range. By default it caps all genres to the minimum natural yield for class balance; use `--no-balance` to retain all valid transpositions. Requires `reportlab` for PDF generation.

### Alternative: Command-Line Scripts

Each model can also be run independently from the project root:

```bash
python -m src.gamelan_classifier [dataset_path] [output_dir]
python -m src.gamelan_mlp        [dataset_path] [output_dir]
python -m src.gamelan_cnn        [dataset_path] [output_dir]
```

Default dataset path is `dataset/` and default output directories are under `output/`.

## Kepatihan Notation Encoding

The parser (`src/parser.py`) maps Balungan TrueType font glyphs to pitch and octave:

| Letter Range | Octave         | Scale Degrees  |
|-------------|----------------|----------------|
| `A`–`G`     | Low (dot below)  | 7, 6, 5, 4, 3, 2, 1 |
| `H`–`N`     | Mid (no dot)     | 1, 2, 3, 4, 5, 6, 7 |
| `O`–`U`     | High (dot above) | 1, 2, 3, 4, 5, 6, 7 |

Beat marker characters overlay notes:

| Character | Meaning              |
|-----------|----------------------|
| `)`       | Kenong               |
| `^`       | Kethuk               |
| `@`       | Gong ageng           |
| `(`       | Kempyang / gong suwukan |
| `[` / `]` | Repeat section start / end |
| `-`       | Rest / silence       |

## Feature Engineering

The `extract_features()` function produces a **29-dimensional float32 vector** per piece:

| Group              | Features                                      | Count |
|--------------------|-----------------------------------------------|-------|
| Pitch distribution | Percentage of each scale degree 1–7           | 7     |
| Register           | Percentage in low / mid / high octave         | 3     |
| Complexity         | Avg beat markers per note, rest ratio         | 2     |
| Intervals          | Step ratio, leap ratio, mean absolute interval| 3     |
| Gong structure     | Mean / std / count of notes per gong cycle    | 3     |
| Repeats            | Has repeat bracket (bool), repeat count       | 2     |
| Section presence   | Buka, Merong, Inggah, Ngelik, Umpak, Suwuk   | 6     |
| Pitch entropy      | Shannon entropy over pitch distribution       | 1     |
| Melodic contour    | Ascending ratio, direction change rate        | 2     |
| **Total**          |                                               | **29**|

## Models

| Model          | Input                  | Approach                                         |
|----------------|------------------------|--------------------------------------------------|
| Decision Tree  | 29 features            | Interpretable split rules, depth sweep, `class_weight='balanced'` |
| Random Forest  | 29 features            | Ensemble of decision trees, `class_weight='balanced'` |
| SVM (RBF)      | 29 features (Pipeline) | Kernel-based, `class_weight='balanced'`, per-fold scaling |
| KNN            | 29 features (Pipeline) | Distance-weighted voting, per-fold scaling       |
| MLP            | 29 features (scaled)   | Two-hidden-layer neural network, weighted `CrossEntropyLoss` |
| 1D CNN         | Raw sequences (7×T)    | Convolutional filters over time, weighted `CrossEntropyLoss` |

## Evaluation Protocol

### Baseline (35 pieces)

- **Leave-One-Out Cross-Validation (LOOCV):** For classical models (DT, RF, SVM, KNN). Trains on 34 pieces, tests on 1, repeated 35 times.
- **Stratified hold-out split (4 train / 1 test per genre):** For neural models (MLP, CNN) and visualisation.

### Augmented (77 balanced / 192 unbalanced pieces)

- **Leave-One-Group-Out CV (LOGOCV) for all six models:** Each group = one original piece + all its transpositions. 35 folds — prevents any transposition of a test piece from appearing in training. Neural models (MLP, CNN) use a custom per-fold training loop with 300 epochs.
- **Leak-free stratified hold-out (visualisation only):** Groups by original piece name; all variants of train-originals go to train, all variants of test-originals go to test. Used for training curves and embedding plots.

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

## Notes and Caveats

- **Small dataset.** With 35 original pieces across 7 forms, all models are susceptible to overfitting. LOOCV/LOGOCV and the depth sweep are specifically designed to surface this.
- **Balanced vs unbalanced trade-off.** The balanced dataset (77 pieces) eliminates class imbalance but provides fewer training examples per fold. The unbalanced dataset (192 pieces) provides more data but requires class weighting. The notebook runs both and compares.
- **PDF parsing edge cases.** Some pieces (e.g., Srepegan Manyura) may produce 0 events due to PDF encoding issues. The CNN handles this with zero-padding. Ayak Ayak Pamungkas produces ~3400 events due to repeated sections — the p95 truncation prevents this outlier from distorting tensor dimensions.
- **PyTorch dependency.** The MLP and CNN require PyTorch. The classical models run on scikit-learn only. The `torch` import in `src/data.py` is lazy — classical-only workflows do not require PyTorch.

## References

1. Kurniawati, A., et al. (2024). Notation of Javanese Gamelan dataset for traditional music applications. *Data in Brief*, 53, 110116.
2. Savitri, N. P. D. P., et al. (2025). Classification of Gamelan Selonding Music Using Convolutional Neural Network. *Indonesian Journal of Data and Science*, 6(3).
3. Martopangrawit (1975). *Pengetahuan Karawitan II*. ASKI Surakarta.
4. Supanggah, R. (2002). *Bothèkan Karawitan I*. Masyarakat Seni Pertunjukan Indonesia.

## License

See [LICENSE](LICENSE) for details.

# Automatic Classification of Javanese Gamelan Forms from Kepatihan Cipher Notation Using Machine Learning

A symbolic music analysis and classification pipeline for automatically identifying Javanese gamelan **bentuk** (cyclic architectural forms) directly from Kepatihan (cipher) notation PDFs. This project classifies 7 traditional forms — Ayak-Ayak, Bubaran, Ketawang, Ladrang, Lancaran, Sampak, and Srepegan — using both interpretable and deep learning models.

## Motivation

Most research in Music Information Retrieval (MIR) applied to non-Western traditions relies on audio-based deep learning models that act as "black boxes." This project addresses the interpretability gap by working directly with **symbolic notation**, enabling the extraction of musicologically meaningful features and the generation of human-readable decision rules. The approach combines Explainable AI (Decision Trees, Random Forest) with neural models (MLP, 1D CNN) to determine which method best suits the small-data, high-structure regime typical of ethnomusicological datasets.

## Dataset

The project uses the **Javanese Gamelan Notation Dataset** (Kurniawati et al., *Data in Brief*, 2024):

- **35 compositions** across **7 bentuk** (5 pieces each)
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

**Citation:**
> Kurniawati, A., Yuniarno, E. M., Suprapto, Y. K., Ifada, N., & Soewidiatmaka, N. I. (2024).
> Notation of Javanese Gamelan dataset for traditional music applications. *Data in Brief*, 53, 110116.
> [DOI: 10.1016/j.dib.2024.110116](https://doi.org/10.1016/j.dib.2024.110116)

## Project Structure

```
.
├── gamelan_classification.ipynb       # Main experiment notebook (start here)
├── requirements.txt                  # Python dependencies
├── README.md
├── LICENSE
│
├── src/                              # Core Python package
│   ├── __init__.py
│   ├── parser.py                     # PDF extraction, Note class, sequence encoding
│   ├── features.py                   # 29-dim hand-crafted feature vectors
│   ├── data.py                       # Corpus loading and stratified train/test split
│   ├── plots.py                      # Matplotlib plotting functions
│   ├── statistical_analysis.py       # Corpus-level statistical analysis and plots
│   ├── gamelan_classifier.py         # Decision Tree (CLI)
│   ├── gamelan_mlp.py                # MLP classifier (CLI)
│   └── gamelan_cnn.py                # 1D CNN classifier (CLI)
│
├── dataset/                          # Kepatihan notation PDFs
│   ├── Ayak Ayak/
│   ├── Bubaran/
│   ├── Ketawang/
│   ├── Ladrang/
│   ├── Lancaran/
│   ├── Sampak/
│   └── Srepegan/
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

### Recommended: Jupyter Notebook

The notebook `gamelan_classification.ipynb` provides the complete end-to-end pipeline:

```bash
jupyter notebook gamelan_classification.ipynb
```

It covers dataset exploration, feature extraction, exploratory data analysis, all six models with evaluation, and a comparative summary.

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
| Decision Tree  | 29 features            | Interpretable split rules, depth sweep           |
| Random Forest  | 29 features            | Ensemble of decision trees, reduced variance     |
| SVM (RBF)      | 29 features (scaled)   | Kernel-based max-margin classifier               |
| KNN            | 29 features (scaled)   | Distance-based nearest-neighbor voting           |
| MLP            | 29 features (scaled)   | Two-hidden-layer neural network (64→32→7)        |
| 1D CNN         | Raw sequences (7×T)    | Convolutional filters over time, no feature eng. |

## Evaluation Protocol

Given the small dataset (35 pieces), evaluation uses:

- **Leave-One-Out Cross-Validation (LOOCV):** For classical models (DT, RF, SVM, KNN). Trains on 34 pieces, tests on 1, repeated 35 times. Provides the most unbiased accuracy estimate.
- **Stratified hold-out split (4 train / 1 test per genre):** For neural models (MLP, CNN) and visualization purposes.

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

- **Small dataset.** With 35 pieces across 7 forms, all models are susceptible to overfitting. LOOCV and the depth sweep are specifically designed to surface this.
- **PDF parsing edge cases.** Some pieces (e.g., Srepegan Manyura) may produce 0 events due to PDF encoding issues. The CNN handles this with zero-padding. Ayak Ayak Pamungkas produces ~3400 events due to repeated sections — the p95 truncation prevents this outlier from distorting tensor dimensions.
- **PyTorch dependency.** The MLP and CNN require PyTorch. The classical models run on scikit-learn only.

## References

1. Kurniawati, A., et al. (2024). Notation of Javanese Gamelan dataset for traditional music applications. *Data in Brief*, 53, 110116.
2. Savitri, N. P. D. P., et al. (2025). Classification of Gamelan Selonding Music Using Convolutional Neural Network. *Indonesian Journal of Data and Science*, 6(3).
3. Martopangrawit (1975). *Pengetahuan Karawitan II*. ASKI Surakarta.
4. Supanggah, R. (2002). *Bothèkan Karawitan I*. Masyarakat Seni Pertunjukan Indonesia.

## License

See [LICENSE](LICENSE) for details.

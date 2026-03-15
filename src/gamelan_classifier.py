"""
Gamelan Subgenre — Decision Tree Classifier
--------------------------------------------
Usage:
    python -m src.gamelan_classifier <source_root> [output_dir]

Requirements:
    pip install pdfplumber scikit-learn matplotlib numpy
"""

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

from .data import load_corpus_features, stratified_split, to_arrays
from .features import FEATURE_NAMES
from .plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_prediction_confidence_clf,
    plot_feature_scatter,
    plot_depth_sweep,
)

warnings.filterwarnings("ignore")


def main():
    source  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/decision_tree")

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

    X_train, y_train, le = to_arrays(train_records)
    X_test,  y_test,  _  = to_arrays(test_records, le)

    # 3. Depth sweep → pick best depth
    print("\n── Depth sweep to find best generalising depth ──")
    max_d = min(15, X_train.shape[0] - 1)
    depths = list(range(1, max_d + 1))
    train_accs, test_accs = [], []
    for d in depths:
        clf_d = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf_d.fit(X_train, y_train)
        train_accs.append(clf_d.score(X_train, y_train))
        test_accs.append( clf_d.score(X_test,  y_test))

    best_depth = depths[int(np.argmax(test_accs))]
    print(f"  Best test accuracy {max(test_accs):.1%} at max_depth={best_depth}")

    # 4. Final model
    print(f"\n── Training final Decision Tree (max_depth={best_depth}) ──")
    clf = DecisionTreeClassifier(max_depth=best_depth, criterion="gini", random_state=42)
    clf.fit(X_train, y_train)
    print(f"  Train accuracy: {clf.score(X_train, y_train):.1%}")
    print(f"  Test  accuracy: {clf.score(X_test,  y_test):.1%}  "
          f"({int(clf.score(X_test, y_test) * len(y_test))}/{len(y_test)} correct)")

    # 5. Report
    y_pred         = clf.predict(X_test)
    test_ids       = sorted(set(y_test))
    test_names     = [le.classes_[i] for i in test_ids]
    all_classes    = list(le.classes_)

    report = classification_report(y_test, y_pred, labels=test_ids,
                                   target_names=test_names, zero_division=0)
    print(f"\n── Classification Report ──\n{report}")

    rp = out_dir / "classification_report.txt"
    with open(rp, "w") as f:
        f.write(f"Best depth     : {best_depth}\n")
        f.write(f"Train accuracy : {clf.score(X_train, y_train):.4f}\n")
        f.write(f"Test  accuracy : {clf.score(X_test, y_test):.4f}\n\n")
        f.write(report)
        f.write("\n\n── Tree text rules ──\n")
        f.write(export_text(clf, feature_names=list(FEATURE_NAMES)))
    print(f"  Saved: {rp.name}")

    # 6. Plots
    print(f"\n── Generating plots → {out_dir}/ ──")
    plot_confusion_matrix(y_test, y_pred, test_names, test_ids, out_dir,
                          title="Confusion Matrix — Test Set (Decision Tree)")
    plot_feature_importance(clf, X_train, y_train, out_dir)
    plot_decision_tree(clf, all_classes, out_dir)
    plot_prediction_confidence_clf(clf, X_test, y_test, test_records, le, out_dir)
    plot_feature_scatter(clf, X_train, y_train, X_test, y_test, le, out_dir)
    plot_depth_sweep(depths, train_accs, test_accs, best_depth, out_dir)

    print(f"\n✅  Done. {len(list(out_dir.glob('*.png')))} plots + report in: {out_dir}/")


if __name__ == "__main__":
    main()
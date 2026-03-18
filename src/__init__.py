"""
src — Javanese Gamelan Form Classification Pipeline
====================================================

Modules:
    parser                   PDF extraction, kepatihan tokenisation, Note class, sequence encoding
    features                 29-dim hand-crafted feature vectors
    data                     Corpus loading, stratified split, LOGO grouping
    plots                    Matplotlib plotting functions for all classifiers
    statistical_analysis     Corpus-level statistics and EDA plots
    gamelan_classifier       Decision Tree classifier (CLI)
    gamelan_mlp              MLP classifier (CLI)
    gamelan_cnn              1D CNN classifier (CLI)
    make_augmented_dataset   Pitch-transposition augmentation script
"""

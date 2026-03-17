"""
src/data.py
-----------
Corpus loading and stratified train/test splitting.

Two loaders:
  load_corpus_features()  →  DT / MLP  (extracts 29-dim feature vectors)
  load_corpus_sequences() →  CNN       (stores raw (T, 7) note sequences)

Both share the same stratified_split() logic:
  ≥5 songs  →  4 train / 1 test
   4 songs  →  3 train / 1 test  (LOO fallback)
  <4 songs  →  all train, no test sample
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .parser import (
    extract_raw_text,
    parse_notation,
    pdf_to_sequence,
    pad_or_truncate,
    N_DIMS,
)
from .features import extract_features


# ── Augmentation helpers ─────────────────────────────────────────────────────

import re as _re
_AUG_SUFFIX = _re.compile(r"\s+shift[+-]\d+$")


def original_name(song_name: str) -> str:
    """Strip augmentation suffix (e.g. ' shift+3') to recover the original piece name."""
    return _AUG_SUFFIX.sub("", song_name)


def get_logo_groups(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Assign an integer group ID to each record based on its original (un-shifted)
    piece name.  All transpositions of the same original share the same ID.

    Returns (groups, unique_originals):
      - groups: int array of shape (len(records),)
      - unique_originals: sorted list of unique original piece names
    """
    originals = [original_name(r["song_name"]) for r in records]
    unique = sorted(set(originals))
    mapping = {name: i for i, name in enumerate(unique)}
    return np.array([mapping[n] for n in originals]), unique


def stratified_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Leak-free stratified split that is aware of pitch-transposition augmentation.

    For a plain dataset (no augmentation):
      ≥5 originals  →  first 4 train, 5th test
       4 originals  →  first 3 train, 4th test  (LOO fallback)
      <4 originals  →  all train, warns

    For an augmented dataset (song names end in ` shift±N`):
      Records are first grouped by their *original* piece name.
      The split is decided on originals only — keeping the same 4/1 ratio.
      Every transposition of a train-original goes to train.
      Every transposition of a test-original goes to test.
      The test set contains ONLY the original (no transpositions) so it
      evaluates on unseen, real pieces — preventing data leakage.

    Both cases are handled by the same code path: if no record has a shift
    suffix, every song_name equals its original_name and behaviour is
    identical to the old implementation.
    """
    by_genre: dict[str, list] = defaultdict(list)
    for r in records:
        by_genre[r["genre"]].append(r)

    train, test = [], []

    for genre, songs in sorted(by_genre.items()):
        # Group by original piece name
        by_orig: dict[str, list] = defaultdict(list)
        for r in songs:
            by_orig[original_name(r["song_name"])].append(r)

        # Sort original names alphabetically for reproducibility
        orig_names = sorted(by_orig.keys())
        n_orig = len(orig_names)

        if n_orig >= 5:
            train_origs = orig_names[:4]
            test_origs  = orig_names[4:5]
        elif n_orig == 4:
            print(f"  ℹ️   {genre}: 4 originals → 3 train / 1 test (LOO fallback)")
            train_origs = orig_names[:3]
            test_origs  = orig_names[3:4]
        else:
            print(f"  ⚠️   {genre}: only {n_orig} original(s) — skipping test sample")
            for orig in orig_names:
                train.extend(by_orig[orig])
            continue

        # All transpositions of train-originals → train
        for orig in train_origs:
            train.extend(sorted(by_orig[orig], key=lambda r: r["song_name"]))

        # All variants of test-originals (original + transpositions) → test.
        # No transposition of a test-original ever appears in train, so this
        # is leak-free while giving a larger, more informative test set that
        # explicitly measures transposition generalisation.
        for orig in test_origs:
            test.extend(sorted(by_orig[orig], key=lambda r: r["song_name"]))

        n_train_pieces = sum(len(by_orig[o]) for o in train_origs)
        n_test_pieces  = sum(len(by_orig[o]) for o in test_origs)
        print(f"  {genre}: {n_orig} originals → "
              f"{n_train_pieces} train pieces / {n_test_pieces} test pieces")

    return train, test


# ── Feature-based loader (DT + MLP) ──────────────────────────────────────────

def load_corpus_features(source_root: Path) -> list[dict]:
    """
    Walk source_root/<genre>/<song>/<song>.pdf, extract 29-dim feature vectors.
    Returns list of { genre, song_name, features }.
    """
    records: list[dict] = []
    for category in sorted(source_root.iterdir()):
        if not category.is_dir():
            continue
        genre = category.name
        songs: list[dict] = []
        for song_folder in sorted(category.iterdir()):
            if not song_folder.is_dir():
                continue
            pdf = song_folder / f"{song_folder.name}.pdf"
            if not pdf.exists():
                continue
            try:
                raw    = extract_raw_text(str(pdf))
                parsed = parse_notation(raw)
                feats  = extract_features(parsed)
                songs.append({
                    "genre":     genre,
                    "song_name": pdf.stem,
                    "features":  feats,
                })
                print(f"  ✅  {genre} / {pdf.stem}")
            except Exception as e:
                print(f"  ❌  {pdf.name}: {e}")

        if len(songs) < 5:
            print(f"  ⚠️   {genre}: only {len(songs)} song(s) — need 5 for clean 4/1 split")
        records.extend(songs)
    return records


def to_arrays(
    split_records: list[dict],
    le: LabelEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Stack feature vectors and encode genre labels.
    Fit LabelEncoder on first call; pass fitted encoder for test set.
    """
    X      = np.stack([r["features"] for r in split_records])
    labels = [r["genre"] for r in split_records]
    if le is None:
        le = LabelEncoder()
        y  = le.fit_transform(labels)
    else:
        y = le.transform(labels)
    return X, y.astype(np.int64), le


# ── Sequence-based loader (CNN) ───────────────────────────────────────────────

def load_corpus_sequences(source_root: Path) -> list[dict]:
    """
    Walk source_root/<genre>/<song>/<song>.pdf, extract raw (T, 7) sequences.
    Returns list of { genre, song_name, seq, seq_len }.
    """
    records: list[dict] = []
    for category in sorted(source_root.iterdir()):
        if not category.is_dir():
            continue
        genre = category.name
        songs: list[dict] = []
        for song_folder in sorted(category.iterdir()):
            if not song_folder.is_dir():
                continue
            pdf = song_folder / f"{song_folder.name}.pdf"
            if not pdf.exists():
                continue
            try:
                seq = pdf_to_sequence(str(pdf))
                songs.append({
                    "genre":     genre,
                    "song_name": pdf.stem,
                    "seq":       seq,
                    "seq_len":   seq.shape[0],
                })
                print(f"  ✅  {genre} / {pdf.stem}  ({seq.shape[0]} events)")
            except Exception as e:
                print(f"  ❌  {pdf.name}: {e}")

        if len(songs) < 5:
            print(f"  ⚠️   {genre}: only {len(songs)} song(s) — need 5 for clean 4/1 split")
        records.extend(songs)
    return records


def to_tensors(
    split_records: list[dict],
    pad_len: int,
    le: LabelEncoder | None = None,
):
    """
    Pad/truncate sequences and stack into a channels-first tensor.
    Returns X: (N, N_DIMS, pad_len), y: (N,), le.
    """
    import torch

    X_list = [pad_or_truncate(r["seq"], pad_len) for r in split_records]
    X      = np.stack(X_list, axis=0)       # (N, pad_len, N_DIMS)
    X      = X.transpose(0, 2, 1)           # (N, N_DIMS, pad_len)  channels-first
    labels = [r["genre"] for r in split_records]
    if le is None:
        le = LabelEncoder()
        y  = le.fit_transform(labels)
    else:
        y = le.transform(labels)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        le,
    )
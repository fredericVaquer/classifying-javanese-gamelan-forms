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
import torch
from sklearn.preprocessing import LabelEncoder

from parser import (
    extract_raw_text,
    parse_notation,
    pdf_to_sequence,
    pad_or_truncate,
    N_DIMS,
)
from features import extract_features


# ── Shared split logic ────────────────────────────────────────────────────────

def stratified_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Sort songs alphabetically per genre, then:
      ≥5 songs  →  first 4 train, 5th test
       4 songs  →  first 3 train, 4th test  (LOO fallback, prints ℹ️)
      <4 songs  →  all train, warns
    """
    by_genre: dict[str, list] = defaultdict(list)
    for r in records:
        by_genre[r["genre"]].append(r)

    train, test = [], []
    for genre, songs in sorted(by_genre.items()):
        songs_sorted = sorted(songs, key=lambda x: x["song_name"])
        n = len(songs_sorted)
        if n >= 5:
            train.extend(songs_sorted[:4])
            test.extend(songs_sorted[4:5])
        elif n == 4:
            print(f"  ℹ️   {genre}: 4 songs → 3 train / 1 test (LOO fallback)")
            train.extend(songs_sorted[:3])
            test.extend(songs_sorted[3:4])
        else:
            print(f"  ⚠️   {genre}: only {n} song(s) — skipping test sample")
            train.extend(songs_sorted)
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
) -> tuple[torch.Tensor, torch.Tensor, LabelEncoder]:
    """
    Pad/truncate sequences and stack into a channels-first tensor.
    Returns X: (N, N_DIMS, pad_len), y: (N,), le.
    """
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
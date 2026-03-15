"""
src/features.py
---------------
Hand-crafted feature extraction: one 29-dim vector per piece.
Used by the Decision Tree and MLP classifiers.
The CNN uses raw sequences instead (see src/parser.py).
"""

import math
from collections import Counter

import numpy as np

from .parser import Note


FEATURE_NAMES: list[str] = [
    # Pitch distribution (7)
    "pitch_1_%", "pitch_2_%", "pitch_3_%", "pitch_4_%",
    "pitch_5_%", "pitch_6_%", "pitch_7_%",
    # Register (3)
    "reg_low_%", "reg_mid_%", "reg_high_%",
    # Complexity (2)
    "marker_density", "rest_ratio",
    # Intervals (3)
    "step_ratio", "leap_ratio", "mean_abs_interval",
    # Gong structure (3)
    "gong_cycle_mean", "gong_cycle_std", "gong_cycle_count",
    # Repeats (2)
    "has_repeat", "repeat_count",
    # Section presence (6)
    "has_Buka", "has_Merong", "has_Inggah", "has_Ngelik", "has_Umpak", "has_Suwuk",
    # Pitch entropy (1)
    "pitch_entropy",
    # Melodic contour (2)
    "ascending_ratio", "direction_changes",
]
N_FEATURES = len(FEATURE_NAMES)


def extract_features(parsed: dict) -> np.ndarray:
    """
    Extract a 29-dim float32 feature vector from a parsed notation dict
    (as returned by src.parser.parse_notation).
    """
    all_notes = [n for sec in parsed["sections"]
                 for n in sec["notes"] if isinstance(n, Note)]
    sounding  = [n for n in all_notes if not n.is_rest]
    rests     = [n for n in all_notes if n.is_rest]
    total_raw = len(all_notes)
    pc        = Counter(n.pitch for n in sounding)
    tot_s     = len(sounding) or 1

    # Pitch distribution
    pitch_feats = [pc.get(p, 0) / tot_s for p in range(1, 8)]

    # Register
    rc = Counter(n.octave for n in sounding)
    reg_feats = [rc.get(r, 0) / tot_s for r in (-1, 0, 1)]

    # Complexity
    marker_density = sum(len(n.markers) for n in sounding) / tot_s
    rest_ratio     = len(rests) / (total_raw or 1)

    # Intervals
    pitches   = [n.absolute_pitch for n in sounding if n.absolute_pitch is not None]
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    n_int     = len(intervals) or 1
    steps = sum(1 for iv in intervals if abs(iv) <= 2)
    leaps = sum(1 for iv in intervals if abs(iv) >  2)
    step_ratio   = steps / n_int
    leap_ratio   = leaps / n_int
    mean_abs_int = float(np.mean([abs(iv) for iv in intervals])) if intervals else 0.0

    # Gong structure
    gong_cycles: list[int] = []
    for sec in parsed["sections"]:
        notes = [n for n in sec["notes"] if isinstance(n, Note)]
        gpos  = [i for i, n in enumerate(notes) if "@" in n.markers]
        prev  = 0
        for gp in gpos:
            gong_cycles.append(gp - prev + 1)
            prev = gp + 1
    gong_mean  = float(np.mean(gong_cycles)) if gong_cycles else 0.0
    gong_std   = float(np.std(gong_cycles))  if gong_cycles else 0.0
    gong_count = len(gong_cycles)

    # Repeats
    repeat_count = sum(
        1 for sec in parsed["sections"] for tok in sec["notes"] if tok == "["
    )
    has_repeat = float(repeat_count > 0)

    # Section presence
    sec_names = {sec["name"] for sec in parsed["sections"]}
    sec_feats = [
        float(s in sec_names)
        for s in ("Buka", "Merong", "Inggah", "Ngelik", "Umpak", "Suwuk")
    ]

    # Pitch entropy
    probs         = [v / tot_s for v in pc.values()]
    pitch_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Melodic contour
    ascending = sum(1 for iv in intervals if iv > 0)
    ascending_ratio = ascending / n_int
    direction_changes = sum(
        1 for k in range(1, len(intervals))
        if intervals[k] != 0 and intervals[k - 1] != 0
        and np.sign(intervals[k]) != np.sign(intervals[k - 1])
    ) / n_int

    vec = (
        pitch_feats + reg_feats
        + [marker_density, rest_ratio,
           step_ratio, leap_ratio, mean_abs_int,
           gong_mean, gong_std, gong_count,
           has_repeat, repeat_count]
        + sec_feats
        + [pitch_entropy, ascending_ratio, direction_changes]
    )
    return np.array(vec, dtype=np.float32)
"""
src/parser.py
-------------
PDF extraction, kepatihan tokenisation, Note class, and raw sequence encoding.
All other modules import from here — nothing is duplicated elsewhere.
"""

import re
import numpy as np
import pdfplumber


# ── Kepatihan cipher: letter → (scale_degree, octave) ────────────────────────
NOTE_MAP: dict[str, tuple[int, int]] = {
    # low octave  (dot below, yMin ≈ -214)
    "A": (7, -1), "B": (6, -1), "C": (5, -1), "D": (4, -1),
    "E": (3, -1), "F": (2, -1), "G": (1, -1),
    # mid octave  (no dot,    yMin ≈ -5)
    "H": (1,  0), "I": (2,  0), "J": (3,  0), "K": (4,  0),
    "L": (5,  0), "M": (6,  0), "N": (7,  0),
    # high octave (dot above, yMax ≈ 856)
    "O": (1, +1), "P": (2, +1), "Q": (3, +1), "R": (4, +1),
    "S": (5, +1), "T": (6, +1), "U": (7, +1),
}

SECTION_HEADERS = {"Buka", "Suwuk", "Ngelik", "Umpak", "Merong", "Inggah"}

# ── Sequence encoding ─────────────────────────────────────────────────────────
N_DIMS  = 7    # dims per timestep in the CNN input tensor
PAD_LEN = 200  # fallback; overridden dynamically in gamelan_cnn.py main()

# Channel layout for pdf_to_sequence():
#   0  pitch      normalised  pitch / 7.0          (0 = rest)
#   1  octave     normalised  (octave + 1) / 2.0   (0.5 for rests)
#   2  kenong     bool  ')'
#   3  kethuk     bool  '^'
#   4  gong       bool  '@'
#   5  kempyang   bool  '('
#   6  is_rest    bool


# ── Note class ────────────────────────────────────────────────────────────────

class Note:
    """Single decoded note or rest event."""
    __slots__ = ("raw", "pitch", "octave", "markers", "prefix")

    def __init__(self, raw: str):
        self.raw = raw
        if raw.startswith("-+"):
            self.prefix, self.pitch, self.octave = "-+", 0, 0
            self.markers = "".join(c for c in raw[2:] if c in ")^@(")
        elif raw.startswith("-") and (
            len(raw) == 1 or raw[1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ):
            self.prefix, self.pitch, self.octave = "-", 0, 0
            self.markers = "".join(c for c in raw[1:] if c in ")^@(")
        else:
            self.prefix = raw[0] if raw and raw[0] in "+-" else ""
            letter = raw.lstrip("+-")[:1].upper()
            p, o = NOTE_MAP.get(letter, (0, 0))
            self.pitch, self.octave = p, o
            self.markers = "".join(c for c in raw if c in ")^@(")

    @property
    def is_rest(self) -> bool:
        return self.pitch == 0

    @property
    def absolute_pitch(self):
        """Comparable pitch value: octave*10 + scale_degree (None for rests)."""
        return None if self.is_rest else self.octave * 10 + self.pitch


# ── Low-level tokenisation ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"-\+|-(?![A-Za-z0-9])[@)^(]*|[-+]?[A-Za-z0-9.][@)^(]*|[\[\]]"
)


def tokenize(line: str) -> list[str]:
    return _TOKEN_RE.findall(line)


def decode_tokens(tokens: list[str]) -> list:
    """Convert raw token strings into Note objects (or '['/']' strings)."""
    return [tok if tok in ("[", "]") else Note(tok) for tok in tokens]


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_raw_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


# ── Structured notation parser ────────────────────────────────────────────────

def parse_notation(raw: str) -> dict:
    """
    Returns:
        {
            title    : str,
            laras    : str,
            pathet   : str,
            sections : list of { name, lines, tokens, notes }
        }
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    res   = {"title": "", "laras": "", "pathet": "", "sections": []}
    if not lines:
        return res

    res["title"] = lines[0]
    for key, pat in (("laras", r"laras\s+(\S+)"), ("pathet", r"pathet\s+(\S+)")):
        m = re.search(pat, lines[0], re.I)
        if m:
            res[key] = m.group(1)

    cur = None
    _header_re = re.compile(r"^(" + "|".join(SECTION_HEADERS) + r")\s+(.*)")

    for line in lines[1:]:
        if line in SECTION_HEADERS:
            cur = {"name": line, "lines": [], "tokens": [], "notes": []}
            res["sections"].append(cur)
            continue
        hp = _header_re.match(line)
        if hp:
            toks = tokenize(hp.group(2))
            cur  = {
                "name":   hp.group(1),
                "lines":  [hp.group(2)],
                "tokens": toks,
                "notes":  decode_tokens(toks),
            }
            res["sections"].append(cur)
            continue
        if cur is None:
            cur = {"name": "Intro", "lines": [], "tokens": [], "notes": []}
            res["sections"].append(cur)
        toks = tokenize(line)
        cur["lines"].append(line)
        cur["tokens"].extend(toks)
        cur["notes"].extend(decode_tokens(toks))

    return res


# ── Raw sequence encoder (used by CNN) ───────────────────────────────────────

def pdf_to_sequence(pdf_path: str) -> np.ndarray:
    """
    Parse a PDF directly into a float32 array of shape (T, N_DIMS).
    No feature engineering — each note/rest becomes one row.
    Bracket tokens [ ] are skipped.
    Returns shape (0, N_DIMS) for empty PDFs.
    """
    raw   = extract_raw_text(pdf_path)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    events: list[list[float]] = []
    _sec_re = re.compile(r"^(" + "|".join(SECTION_HEADERS) + r")\s+(.*)")

    for line in lines[1:]:
        if line in SECTION_HEADERS:
            continue
        hp   = _sec_re.match(line)
        toks = tokenize(hp.group(2) if hp else line)

        for tok in toks:
            if tok in ("[", "]"):
                continue
            is_rest = False
            if tok.startswith("-+") or (
                tok.startswith("-") and
                (len(tok) == 1 or tok[1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            ):
                pitch, octave, is_rest = 0, 0, True
                marker_src = tok[2:] if tok.startswith("-+") else tok[1:]
            else:
                letter = tok.lstrip("+-")[:1].upper()
                pitch, octave = NOTE_MAP.get(letter, (0, 0))
                marker_src = tok

            events.append([
                pitch / 7.0,
                (octave + 1) / 2.0 if not is_rest else 0.5,
                float(")" in marker_src),
                float("^" in marker_src),
                float("@" in marker_src),
                float("(" in marker_src),
                float(is_rest),
            ])

    if not events:
        return np.zeros((0, N_DIMS), dtype=np.float32)
    return np.array(events, dtype=np.float32)


def pad_or_truncate(seq: np.ndarray, length: int) -> np.ndarray:
    """Truncate or zero-pad a (T, N_DIMS) array to exactly (length, N_DIMS)."""
    if seq.ndim < 2 or seq.shape[0] == 0 or seq.shape[1] != N_DIMS:
        return np.zeros((length, N_DIMS), dtype=np.float32)
    T = seq.shape[0]
    if T >= length:
        return seq[:length]
    return np.concatenate(
        [seq, np.zeros((length - T, N_DIMS), dtype=np.float32)], axis=0
    )
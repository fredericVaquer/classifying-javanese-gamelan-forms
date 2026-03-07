"""
Gamelan Notation Extractor
--------------------------
Extracts and parses kepatihan (cipher) notation from gamelan PDF scores.
Tested with sléndro/pélog notation using pdfplumber.

Usage:
    python extract_gamelan_notation.py <path_to_pdf>
    python extract_gamelan_notation.py  (uses bundled example)

Requirements:
    pip install pdfplumber
"""

import sys
import re
import pdfplumber


# ── Balungan font glyph → pitch + octave ─────────────────────────────────────
# Discovered by inspecting the embedded TrueType font's glyph bounding boxes:
#   yMin ≈ -214  →  dot below the note  →  low octave  (-1)
#   yMin ≈   -5  →  no dot              →  mid octave  ( 0)
#   yMax ≈  856  →  dot above the note  →  high octave (+1)
#
# Pitch covers the full kepatihan cipher scale 1–7.
# Sléndro uses only 1 2 3 5 6; pélog also uses 4 and 7.
# The letter pattern follows the alphabet in three bands:
#   low  (-1): ... D=4  E=3  F=2  G=1 (gap)  A=?  B=6  C=5
#   mid  ( 0): H=1  I=2  J=3  K=4  L=5  M=6  N=7
#   high (+1): O=1  P=2  Q=3  R=4  S=5  T=6  U=7
# Only the glyphs present in this font were confirmed from the font file;
# the rest (D, G, K, N, R, S, T, U) are inferred from the alphabetic pattern.

NOTE_MAP: dict[str, tuple[int, int]] = {
    # letter : (pitch, octave)   octave: -1=low, 0=mid, +1=high
    # ── low octave (dot below, yMin≈-214) ───────────────────
    "A": (7, -1),  # inferred
    "B": (6, -1),  # confirmed (sléndro + pélog fonts)
    "C": (5, -1),  # confirmed
    "D": (4, -1),  # confirmed (pélog font, yMin=-214)
    "E": (3, -1),  # confirmed
    "F": (2, -1),  # confirmed
    "G": (1, -1),  # confirmed (pélog font, yMin=-214)
    # ── mid octave (no dot, yMin≈-5 to -16) ─────────────────
    "H": (1,  0),  # confirmed
    "I": (2,  0),  # confirmed
    "J": (3,  0),  # confirmed
    "K": (4,  0),  # inferred
    "L": (5,  0),  # confirmed
    "M": (6,  0),  # confirmed
    "N": (7,  0),  # confirmed (pélog font, yMin=-5)
    # ── high octave (dot above, yMax≈856) ────────────────────
    "O": (1, +1),  # confirmed (sléndro font)
    "P": (2, +1),  # confirmed
    "Q": (3, +1),  # confirmed
    "R": (4, +1),  # inferred
    "S": (5, +1),  # inferred
    "T": (6, +1),  # inferred
    "U": (7, +1),  # inferred
}

# ── Marker characters (kept raw on Note, not decoded to booleans) ─────────────
# These are zero-width overlay glyphs in the Balungan font (advance=0),
# confirmed by inspecting the TrueType glyph outlines:
#   )  = arc above the note (yMin=806)              → kempul / kenong
#   ^  = arc above ) (yMin=1010)                    → kethuk
#   @  = full circle spanning note height           → gong ageng (largest unit)
#   (  = arc below the note (yMin=-340)             → kempyang / gong suwukan
#   +  = small cross above the note (yMin=850)      → connector / wilet
#   -  = full-width dash (advance=480, mid-height)  → rest / silence
#
# Structural:
#   [  = start of repeated gongan
#   ]  = end of repeated gongan

SECTION_HEADERS = {"Buka", "Suwuk", "Ngelik", "Umpak", "Merong", "Inggah"}


# ── Note class ────────────────────────────────────────────────────────────────

class Note:
    """A single decoded note event, including rest slots."""

    def __init__(self, raw: str):
        self.raw     = raw                        # original token, e.g. "F)^", "-+", "-^"
        # Determine prefix and letter
        if raw.startswith("-+"):
            # rest-connect: no pitch, marks a silent rhythmic slot before next note
            self.prefix = "-+"
            self.pitch  = 0
            self.octave = 0
            self.markers = "".join(c for c in raw[2:] if c in ")^@(")
        elif raw.startswith("-") and (len(raw) == 1 or raw[1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            # bare rest, possibly with beat markers (e.g. "-^", "-)")
            self.prefix  = "-"
            self.pitch   = 0
            self.octave  = 0
            self.markers = "".join(c for c in raw[1:] if c in ")^@(")
        else:
            self.prefix  = raw[0] if raw and raw[0] in "+-" else ""
            letter       = raw.lstrip("+-")[:1].upper()
            p, o         = NOTE_MAP.get(letter, (0, 0))
            self.pitch   = p    # scale degree 1–7  (0 = unknown)
            self.octave  = o    # -1 / 0 / +1
            self.markers = "".join(c for c in raw if c in ")^@(")

    def __repr__(self):
        if self.pitch == 0:
            # rest slot: show prefix as-is for clarity (-+, -, etc.)
            prefix = self.prefix
        else:
            prefix = self.prefix if self.prefix else " "
        return f"({prefix}{self.pitch},{self.octave:+d},{self.markers!r})"

    def as_dict(self) -> dict:
        return {
            "raw":     self.raw,
            "pitch":   self.pitch,
            "octave":  self.octave,
            "markers": self.markers,
            "prefix":  self.prefix,
        }


# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_raw_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def tokenize(notation_line: str) -> list[str]:
    """
    Split a notation line into raw string tokens.

    Token types:
      -+          rest slot with connect (no note, but a rhythmic position)
      -           rest slot, optionally followed by beat markers (e.g. '-^')
      [-+]?<letter>[@)^(]*   note with optional prefix and beat markers
      [ or ]      structural repeat brackets
    """
    token_pattern = re.compile(r"-\+|-(?![A-Za-z0-9])[@)^(]*|[-+]?[A-Za-z0-9.][@)^(]*|[\[\]]")
    return token_pattern.findall(notation_line)


def decode_tokens(tokens: list[str]) -> list:
    """
    Convert raw string tokens into Note objects (for note letters and rests)
    or plain strings for structural markers [ ].
    """
    decoded = []
    for tok in tokens:
        if tok in ("[", "]"):
            decoded.append(tok)
        else:
            decoded.append(Note(tok))
    return decoded


def parse_notation(raw_text: str) -> dict:
    """
    Parse raw PDF text into a structured notation dict:
      title    : str
      laras    : str           (e.g. "sléndro")
      pathet   : str           (e.g. "nem")
      sections : list of dicts, each containing:
                   name    : str
                   lines   : list[str]       raw text lines
                   tokens  : list[str]       raw token strings
                   notes   : list[Note|str]  decoded note objects (or "["/"]")
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    result = {"title": "", "laras": "", "pathet": "", "sections": []}

    if not lines:
        return result

    title_line = lines[0]
    result["title"] = title_line

    laras_match  = re.search(r"laras\s+(\S+)",  title_line, re.IGNORECASE)
    pathet_match = re.search(r"pathet\s+(\S+)", title_line, re.IGNORECASE)
    if laras_match:
        result["laras"]  = laras_match.group(1)
    if pathet_match:
        result["pathet"] = pathet_match.group(1)

    current_section = None

    for line in lines[1:]:
        if line in SECTION_HEADERS:
            current_section = {"name": line, "lines": [], "tokens": [], "notes": []}
            result["sections"].append(current_section)
            continue

        header_prefix = re.match(
            r"^(" + "|".join(SECTION_HEADERS) + r")\s+(.*)", line
        )
        if header_prefix:
            raw_notation = header_prefix.group(2)
            toks = tokenize(raw_notation)
            current_section = {
                "name":   header_prefix.group(1),
                "lines":  [raw_notation],
                "tokens": toks,
                "notes":  decode_tokens(toks),
            }
            result["sections"].append(current_section)
            continue

        if current_section is None:
            current_section = {"name": "Intro", "lines": [], "tokens": [], "notes": []}
            result["sections"].append(current_section)

        toks = tokenize(line)
        current_section["lines"].append(line)
        current_section["tokens"].extend(toks)
        current_section["notes"].extend(decode_tokens(toks))

    return result


# ── Display ───────────────────────────────────────────────────────────────────

def print_note_map() -> None:
    print("── Note letter → pitch / octave reference ──────────────────")
    print(f"  {'Letter':8} {'Pitch':8} {'Octave':8} {'Confirmed'}")
    print(f"  {'------':8} {'-----':8} {'------':8} {'---------'}")
    for letter, (pitch, octave) in sorted(NOTE_MAP.items()):
        confirmed = "yes" if letter in "BCDEFGHIJLMNOPQ" else "inferred"
        print(f"  {letter:8} {pitch:8} {octave:+8}   {confirmed}")
    print()


def print_notation(parsed: dict) -> None:
    print("=" * 60)
    print(f"Title  : {parsed['title']}")
    print(f"Laras  : {parsed['laras']}")
    print(f"Pathet : {parsed['pathet']}")
    print("=" * 60)
    print("Note format: (<prefix><pitch>,<octave>,<markers>)")
    print("  prefix:  '+' connect, '-' rest, ' ' normal")
    print("  octave:  -1 dot-below, 0 no-dot, +1 dot-above")
    print("  markers: raw beat chars — ')' kenong, '^' kethuk, '@' gong, '(' kempyang")

    for sec in parsed["sections"]:
        print(f"\n── {sec['name']} ──")
        for ln in sec["lines"]:
            print(f"  {ln}")

        print(f"\n  Decoded:")
        row = []
        for item in sec["notes"]:
            if item == "[":
                if row:
                    print("  " + " ".join(str(n) for n in row))
                    row = []
                print("  [ repeat start")
            elif item == "]":
                if row:
                    print("  " + " ".join(str(n) for n in row))
                    row = []
                print("  ] repeat end")
            else:
                row.append(item)
                if isinstance(item, Note) and "@" in item.markers:
                    print("  " + " ".join(str(n) for n in row))
                    row = []
        if row:
            print("  " + " ".join(str(n) for n in row))

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "Javanese Gamelan Notation/Ayak Ayak/Ayak Ayak Nem Slendro Nem/Ayak Ayak Nem Slendro Nem.pdf"

    print(f"Reading: {pdf_path}\n")
    raw    = extract_raw_text(pdf_path)
    parsed = parse_notation(raw)

    print_note_map()
    print_notation(parsed)

    print("── Per-section summary ─────────────────────────────────────")
    for sec in parsed["sections"]:
        notes  = [n for n in sec["notes"] if isinstance(n, Note)]
        gongs  = [n for n in notes if "@" in n.markers]
        by_oct = {-1: 0, 0: 0, 1: 0}
        for n in notes:
            by_oct[n.octave] += 1
        print(
            f"  {sec['name']:10} | {len(notes):3d} notes"
            f" | {len(gongs)} gong(s)"
            f" | low:{by_oct[-1]}  mid:{by_oct[0]}  high:{by_oct[1]}"
        )


if __name__ == "__main__":
    main()
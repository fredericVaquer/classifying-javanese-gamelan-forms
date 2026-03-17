"""
make_augmented_dataset.py
--------------------------
Builds `dataset_augmented/` from an existing `dataset/` folder.

Steps
  1. Scan all pieces and compute valid shifts per piece (shifts that keep
     every note within the 3-octave font range).
  2. Compute the total yield per genre across all valid shifts.
  3. Find the genre with the MINIMUM total yield — this becomes the cap.
  4. For genres above the cap, drop shifts uniformly so every genre ends
     up with the same number of pieces (originals + transpositions).
     Shifts are dropped starting from the extremes (±7, ±6, …) so the
     musically closest transpositions are always kept.
  5. Copy originals verbatim and write transposition PDFs.

Transposition rules
  • Only note letters (A–U) are shifted.
  • Beat markers ) ^ @ ( and structural tokens [ ] - + are untouched.
  • Pitch wraps within the 7-note scale with octave carry:
      new_pitch  = ((pitch − 1 + shift) % 7) + 1
      new_octave = octave + ((pitch − 1 + shift) // 7)
  • A transposition is skipped entirely for a piece if ANY note would
    land outside the 3-octave font range (octave ∉ {−1, 0, +1}).

Usage
  python make_augmented_dataset.py
  python make_augmented_dataset.py --src dataset --dst dataset_augmented
  python make_augmented_dataset.py --no-balance   # skip the cap

Requirements
  pip install pdfplumber reportlab
"""

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ── Kepatihan note map ────────────────────────────────────────────────────────

NOTE_MAP = {
    "A": (7, -1), "B": (6, -1), "C": (5, -1), "D": (4, -1),
    "E": (3, -1), "F": (2, -1), "G": (1, -1),
    "H": (1,  0), "I": (2,  0), "J": (3,  0), "K": (4,  0),
    "L": (5,  0), "M": (6,  0), "N": (7,  0),
    "O": (1, +1), "P": (2, +1), "Q": (3, +1), "R": (4, +1),
    "S": (5, +1), "T": (6, +1), "U": (7, +1),
}
REV_MAP      = {v: k for k, v in NOTE_MAP.items()}
NOTE_LETTERS = set(NOTE_MAP.keys())

SECTION_HEADERS = {"Buka", "Suwuk", "Ngelik", "Umpak", "Merong", "Inggah"}
SEC_RE   = re.compile(r"^(" + "|".join(SECTION_HEADERS) + r")\s+(.*)")
TOKEN_RE = re.compile(
    r"-\+|-(?![A-Za-z0-9])[@)^(]*|[-+]?[A-Za-z0-9.][@)^(]*|[\[\]]"
)

ALL_SHIFTS = [s for s in range(-4, 5) if s != 0]


# ── Transposition core ────────────────────────────────────────────────────────

def shift_letter(letter: str, shift: int):
    pitch, octave = NOTE_MAP[letter]
    raw = pitch - 1 + shift
    return REV_MAP.get((raw % 7 + 1, octave + raw // 7))


def shift_token(tok: str, shift: int):
    if tok in ("[", "]"):
        return tok
    if tok.startswith("-+") or (
        tok.startswith("-") and (len(tok) == 1 or tok[1] not in NOTE_LETTERS)
    ):
        return tok
    prefix = tok[0] if tok and tok[0] in "+-" else ""
    body   = tok[len(prefix):]
    letter = body[:1].upper()
    if letter not in NOTE_LETTERS:
        return tok
    new_letter = shift_letter(letter, shift)
    if new_letter is None:
        return None
    return prefix + new_letter + body[1:]


def collect_letters(lines: list) -> set:
    letters = set()
    for line in lines[1:]:
        s = line.strip()
        if s in SECTION_HEADERS:
            continue
        hp   = SEC_RE.match(s)
        toks = TOKEN_RE.findall(hp.group(2) if hp else s)
        for tok in toks:
            if tok in ("[", "]"):
                continue
            if tok.startswith("-+") or (
                tok.startswith("-") and (len(tok) == 1 or tok[1] not in NOTE_LETTERS)
            ):
                continue
            ltr = tok.lstrip("+-")[:1].upper()
            if ltr in NOTE_LETTERS:
                letters.add(ltr)
    return letters


def can_shift(letters: set, shift: int) -> bool:
    if not letters:
        return False   # empty piece — no notes to shift, treat as invalid
    return all(shift_letter(l, shift) is not None for l in letters)


def apply_shift(lines: list, shift: int) -> list:
    sign   = f"+{shift}" if shift > 0 else str(shift)
    result = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0:
            result.append(f"{s}  [shift: {sign}]")
            continue
        if s in SECTION_HEADERS:
            result.append(s)
            continue
        hp   = SEC_RE.match(s)
        toks = TOKEN_RE.findall(hp.group(2) if hp else s)
        shifted = " ".join(shift_token(t, shift) for t in toks)
        result.append(f"{hp.group(1)} {shifted}" if hp else shifted)
    return result


# ── PDF I/O ───────────────────────────────────────────────────────────────────

def read_lines(pdf_path: Path) -> list:
    with pdfplumber.open(str(pdf_path)) as pdf:
        raw = "\n".join(p.extract_text() or "" for p in pdf.pages)
    return [l for l in raw.splitlines() if l.strip()]


def write_pdf(lines: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setFont("Courier", 10)
    y = A4[1] - 40
    for line in lines:
        if y < 60:
            c.showPage()
            c.setFont("Courier", 10)
            y = A4[1] - 40
        c.drawString(40, y, line)
        y -= 14
    c.save()


# ── Class balancing ───────────────────────────────────────────────────────────

def prioritised_shifts(valid_shifts: list) -> list:
    """
    Sort valid shifts by musical proximity to the original (smallest |shift|
    first), so when we need to drop shifts to meet the cap we always drop
    the most extreme ones first.
    """
    return sorted(valid_shifts, key=lambda s: (abs(s), -s))


def compute_balanced_shifts(
    pieces: list[dict],   # list of {genre, name, lines, valid_shifts}
    balance: bool,
) -> dict:
    """
    Returns {piece_name: [shifts_to_use]} after applying the cap.

    Cap = total pieces in the genre with the minimum natural yield,
    where natural yield = sum of (1 + len(valid_shifts)) per piece in genre.

    Strategy per genre above the cap:
      - Keep all originals (they count as 1 each).
      - Fill remaining slots with transpositions, prioritising smallest |shift|.
      - Distribute slots evenly across pieces within the genre so no single
        piece gets all its transpositions while another gets none.
    """
    # Group by genre
    by_genre: dict[str, list] = defaultdict(list)
    for p in pieces:
        by_genre[p["genre"]].append(p)

    # Natural yield per genre (originals + all valid transpositions).
    # Empty pieces (lines=[]) count as 1 (original only) — they can never
    # contribute transpositions, so excluding them from the cap calculation
    # prevents the vacuous-true bug from inflating yields.
    natural = {g: sum(1 + len(p["valid_shifts"]) for p in ps)
               for g, ps in by_genre.items()}

    if balance:
        # The cap must respect two constraints:
        #   1. cap <= natural yield of every genre (can't create more than available)
        #   2. cap >= n_originals of every genre (can't drop below the originals)
        # We also account for genres with unreadable pieces: those pieces can
        # only contribute 1 (the original copy), so their achievable max is
        # n_originals + sum(valid_shifts for readable pieces only).
        cap = min(natural.values())
        print(f"\n  Class balance cap: {cap} pieces per genre")
        print(f"  Natural yields:    { {g: n for g, n in sorted(natural.items())} }")
    else:
        cap = None

    result = {}   # piece_name → shifts to actually use

    for genre, ps in by_genre.items():
        n_orig = len(ps)

        if cap is None or natural[genre] <= cap:
            # No trimming needed
            for p in ps:
                result[p["name"]] = p["valid_shifts"]
            continue

        # We have more pieces than the cap — need to trim transpositions.
        # Slots available for transpositions after reserving one per original.
        aug_slots = cap - n_orig

        if aug_slots <= 0:
            # Cap is so tight we can't even fit all originals — keep originals only
            for p in ps:
                result[p["name"]] = []
            continue

        # Distribute aug_slots across pieces as evenly as possible,
        # filling each piece's quota with its closest-pitch-distance shifts.
        # Round-robin until we run out of slots or valid shifts.
        assigned: dict[str, list] = {p["name"]: [] for p in ps}
        pool = [(p["name"], s)
                for p in ps
                for s in prioritised_shifts(p["valid_shifts"])]

        for piece_name, shift in pool:
            if aug_slots <= 0:
                break
            assigned[piece_name].append(shift)
            aug_slots -= 1

        for p in ps:
            result[p["name"]] = assigned[p["name"]]

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def build(src: Path, dst: Path, balance: bool) -> None:
    if not src.exists():
        print(f"Error: source directory not found: {src}")
        sys.exit(1)

    if dst.exists():
        print(f"Warning: output directory already exists: {dst}")
        print("         Existing files will be overwritten.\n")

    print(f"Source  : {src.resolve()}")
    print(f"Output  : {dst.resolve()}")
    print(f"Shifts  : {ALL_SHIFTS}")
    print(f"Balance : {'yes — cap all genres to minimum yield' if balance else 'no'}")

    # ── Pass 1: scan all pieces ───────────────────────────────────────────────
    print("\n── Pass 1: scanning corpus ──")
    pieces = []
    read_errors = []

    for genre_dir in sorted(src.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        for song_dir in sorted(genre_dir.iterdir()):
            if not song_dir.is_dir():
                continue
            pdf = song_dir / f"{song_dir.name}.pdf"
            if not pdf.exists():
                continue
            try:
                lines = read_lines(pdf)
            except Exception as e:
                read_errors.append((pdf, e))
                print(f"  ❌  {genre} / {song_dir.name}: {e}")
                continue

            if not lines:
                print(f"  ⚠️   {genre} / {song_dir.name}: empty parse")
                lines = []

            letters      = collect_letters(lines) if lines else set()
            valid_shifts = [s for s in ALL_SHIFTS if can_shift(letters, s)]

            pieces.append({
                "genre":        genre,
                "name":         song_dir.name,
                "pdf":          pdf,
                "lines":        lines,
                "valid_shifts": valid_shifts,
            })
            print(f"  {genre} / {song_dir.name:<46s} "
                  f"{len(valid_shifts):2d} valid shifts")

    # ── Pass 2: compute balanced shift assignments ────────────────────────────
    print("\n── Pass 2: computing shift assignments ──")
    assignments = compute_balanced_shifts(pieces, balance)

    # ── Pass 3: write files ───────────────────────────────────────────────────
    print("\n── Pass 3: writing output ──")
    n_originals = 0
    n_generated = 0
    genre_counts: dict[str, dict] = defaultdict(lambda: {"orig": 0, "aug": 0})

    for piece in pieces:
        genre     = piece["genre"]
        name      = piece["name"]
        pdf       = piece["pdf"]
        lines     = piece["lines"]
        use_shifts = assignments[name]

        # Copy original
        dest = dst / genre / name / pdf.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, dest)
        n_originals += 1
        genre_counts[genre]["orig"] += 1

        if not lines:
            continue

        # Write transpositions
        for shift in use_shifts:
            sign     = f"+{shift}" if shift > 0 else str(shift)
            aug_name = f"{name} shift{sign}"
            out_pdf  = dst / genre / aug_name / f"{aug_name}.pdf"
            try:
                write_pdf(apply_shift(lines, shift), out_pdf)
                n_generated += 1
                genre_counts[genre]["aug"] += 1
            except Exception as e:
                print(f"  ❌  {aug_name}: {e}")

        skipped = len(piece["valid_shifts"]) - len(use_shifts)
        print(f"  {genre} / {name:<46s}"
              f"orig + {len(use_shifts)} shifts"
              + (f"  (dropped {skipped} for balance)" if skipped else ""))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = n_originals + n_generated
    print(f"\n{'─' * 64}")
    print(f"  Originals copied       : {n_originals}")
    print(f"  Transpositions created : {n_generated}")
    print(f"  Total pieces           : {total}")
    if read_errors:
        print(f"  Read errors            : {len(read_errors)}")
    print(f"{'─' * 64}")
    print(f"\n  {'Genre':<22} {'Originals':>10} {'Augmented':>10} {'Total':>8}")
    print(f"  {'─' * 54}")
    for g in sorted(genre_counts):
        c = genre_counts[g]
        t = c["orig"] + c["aug"]
        print(f"  {g:<22} {c['orig']:>10} {c['aug']:>10} {t:>8}")
    print(f"  {'─' * 54}")
    print(f"  {'TOTAL':<22} {n_originals:>10} {n_generated:>10} {total:>8}")

    if balance:
        totals = [genre_counts[g]["orig"] + genre_counts[g]["aug"]
                  for g in genre_counts]
        print(f"\n  Balance check — min: {min(totals)}  max: {max(totals)}"
              + ("  ✅" if min(totals) == max(totals) else "  ⚠️  not perfectly balanced"))

    print(f"\n✅  Done → {dst.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an augmented gamelan dataset by pitch transposition."
    )
    parser.add_argument(
        "--src", type=Path, default=Path("dataset"),
        help="Source dataset root  (default: dataset/)",
    )
    parser.add_argument(
        "--dst", type=Path, default=Path("dataset_augmented"),
        help="Output root  (default: dataset_augmented/)",
    )
    parser.add_argument(
        "--no-balance", dest="balance", action="store_false",
        help="Disable class balancing (keep all valid transpositions)",
    )
    parser.set_defaults(balance=True)
    args = parser.parse_args()
    build(args.src, args.dst, args.balance)
"""
Gamelan Statistical Analysis
------------------------------
Runs musical analysis across genres (top-level category folders) and produces
matplotlib figures saved to an output directory.

Usage:
    python -m src.statistical_analysis [source_root] [output_dir]

    [source_root]  defaults to "dataset"
    [output_dir]   defaults to "output/analysis"
"""

import sys
import math
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .parser import Note, extract_raw_text, parse_notation


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus(source_root: Path) -> dict:
    """
    Returns:
      corpus[genre_name] = list of parsed dicts
    """
    corpus = defaultdict(list)
    for category in sorted(source_root.iterdir()):
        if not category.is_dir():
            continue
        genre = category.name
        for song_folder in sorted(category.iterdir()):
            if not song_folder.is_dir():
                continue
            pdf = song_folder / f"{song_folder.name}.pdf"
            if not pdf.exists():
                continue
            try:
                raw    = extract_raw_text(str(pdf))
                parsed = parse_notation(raw)
                parsed["_source"] = pdf.name
                parsed["_genre"]  = genre
                corpus[genre].append(parsed)
                print(f"  ✅  {genre} / {pdf.stem}")
            except Exception as e:
                print(f"  ❌  {pdf.name}: {e}")
    return dict(corpus)


def all_notes(parsed: dict, include_rests: bool = False) -> list[Note]:
    notes = []
    for sec in parsed["sections"]:
        for n in sec["notes"]:
            if isinstance(n, Note):
                if include_rests or not n.is_rest:
                    notes.append(n)
    return notes


# ── Analysis helpers ──────────────────────────────────────────────────────────

def pitch_distribution(notes: list[Note]) -> Counter:
    return Counter(n.pitch for n in notes if not n.is_rest)


def register_distribution(notes: list[Note]) -> Counter:
    return Counter(n.octave for n in notes if not n.is_rest)


def marker_density(notes: list[Note]) -> dict:
    """Average number of beat markers per sounding note."""
    total = sum(len(n.markers) for n in notes)
    return total / len(notes) if notes else 0.0


def interval_sequence(notes: list[Note]) -> list[int]:
    """Signed intervals in abstract units (octave*10 + scale_degree)."""
    pitches = [n.absolute_pitch for n in notes if n.absolute_pitch is not None]
    return [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]


def categorize_intervals(intervals: list[int]) -> dict:
    steps = sum(1 for i in intervals if abs(i) <= 2)
    leaps = sum(1 for i in intervals if abs(i) >  2)
    total = len(intervals)
    return {
        "steps": steps,
        "leaps": leaps,
        "step_ratio": steps / total if total else 0,
        "leap_ratio": leaps / total if total else 0,
        "mean_abs": np.mean([abs(i) for i in intervals]) if intervals else 0,
    }


def gong_cycle_density(parsed: dict) -> list[float]:
    """Notes-per-gong-cycle for each cycle found in the piece."""
    densities = []
    for sec in parsed["sections"]:
        notes    = [n for n in sec["notes"] if isinstance(n, Note)]
        gong_pos = [i for i, n in enumerate(notes) if "@" in n.markers]
        if not gong_pos:
            continue
        prev = 0
        for gp in gong_pos:
            cycle_len = gp - prev + 1
            densities.append(cycle_len)
            prev = gp + 1
    return densities


def detect_repeats(parsed: dict) -> dict:
    """Detect [...] repeat brackets and count them."""
    bracket_count = 0
    for sec in parsed["sections"]:
        for tok in sec["notes"]:
            if tok == "[":
                bracket_count += 1
    return {"repeat_sections": bracket_count, "has_repeat": bracket_count > 0}


def section_names(parsed: dict) -> list[str]:
    return [sec["name"] for sec in parsed["sections"]]


# ── Aggregation across corpus ─────────────────────────────────────────────────

def aggregate(corpus: dict) -> dict:
    """Pre-compute per-genre statistics."""
    stats = {}
    for genre, pieces in corpus.items():
        genre_notes = [n for p in pieces for n in all_notes(p)]
        genre_intervals = [i for p in pieces for i in interval_sequence(all_notes(p))]
        gong_densities  = [d for p in pieces for d in gong_cycle_density(p)]
        all_section_names = [s for p in pieces for s in section_names(p)]
        repeat_count = sum(detect_repeats(p)["repeat_sections"] for p in pieces)

        stats[genre] = {
            "n_pieces":         len(pieces),
            "total_notes":      len(genre_notes),
            "pitch_dist":       pitch_distribution(genre_notes),
            "register_dist":    register_distribution(genre_notes),
            "marker_density":   marker_density(genre_notes),
            "intervals":        genre_intervals,
            "interval_stats":   categorize_intervals(genre_intervals),
            "gong_densities":   gong_densities,
            "section_counts":   Counter(all_section_names),
            "total_repeats":    repeat_count,
            "pieces_with_repeat": sum(1 for p in pieces if detect_repeats(p)["has_repeat"]),
        }
    return stats


# ── Plotting ──────────────────────────────────────────────────────────────────

# ── Style constants ────────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC", "#6D6875",
    "#B5838D", "#FFCB77", "#17C3B2", "#9B2335",
]

def genre_colors(genres):
    return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(genres))}


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F8F5F0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    if title:   ax.set_title(title, fontsize=11, fontweight="bold", color="#222222", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9, color="#555555")
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9, color="#555555")


def fig_title(fig, text):
    fig.suptitle(text, fontsize=14, fontweight="bold", color="#111111", y=1.01)


# ── 1. Pitch distribution ─────────────────────────────────────────────────────
def plot_pitch_distribution(stats: dict, out_dir: Path, colors: dict):
    genres = sorted(stats)
    pitches = [1, 2, 3, 4, 5, 6, 7]
    n = len(genres)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    fig.patch.set_facecolor("#FAFAF8")
    axes = np.array(axes).flatten()

    for i, genre in enumerate(genres):
        ax  = axes[i]
        pd  = stats[genre]["pitch_dist"]
        tot = sum(pd.values()) or 1
        vals = [pd.get(p, 0) / tot * 100 for p in pitches]
        bars = ax.bar([str(p) for p in pitches], vals,
                      color=colors[genre], alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color="#333333")
        style_ax(ax, title=genre, xlabel="Scale degree", ylabel="% of notes")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pitch Distribution by Genre", fontsize=14, fontweight="bold",
                 color="#111111", y=1.02)
    fig.tight_layout()
    path = out_dir / "01_pitch_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 2. Register usage ─────────────────────────────────────────────────────────
def plot_register_usage(stats: dict, out_dir: Path, colors: dict):
    genres   = sorted(stats)
    registers = [-1, 0, 1]
    labels   = ["Low", "Mid", "High"]
    x        = np.arange(len(genres))
    width    = 0.25

    register_colors = ["#6D6875", "#457B9D", "#E9C46A"]

    fig, ax = plt.subplots(figsize=(max(8, len(genres) * 1.5), 5))
    fig.patch.set_facecolor("#FAFAF8")

    for k, (reg, lab, rc) in enumerate(zip(registers, labels, register_colors)):
        vals = []
        for genre in genres:
            rd  = stats[genre]["register_dist"]
            tot = sum(rd.values()) or 1
            vals.append(rd.get(reg, 0) / tot * 100)
        ax.bar(x + k * width, vals, width,
               label=lab, color=rc, alpha=0.87, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x + width)
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=30, ha="right", fontsize=9)
    ax.legend(title="Register", framealpha=0.7, fontsize=9)
    style_ax(ax, title="Register Usage by Genre", ylabel="% of sounding notes")
    fig.tight_layout()
    path = out_dir / "02_register_usage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 3. Note complexity (marker density) ───────────────────────────────────────
def plot_note_complexity(stats: dict, out_dir: Path, colors: dict):
    genres = sorted(stats)
    vals   = [stats[g]["marker_density"] for g in genres]

    fig, ax = plt.subplots(figsize=(max(7, len(genres) * 1.3), 4.5))
    fig.patch.set_facecolor("#FAFAF8")

    bars = ax.bar(genres, vals,
                  color=[colors[g] for g in genres],
                  alpha=0.85, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=30, ha="right", fontsize=9)
    style_ax(ax, title="Beat Marker Density by Genre",
             ylabel="Avg markers per sounding note")
    fig.tight_layout()
    path = out_dir / "03_note_complexity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 4. Melodic interval analysis ──────────────────────────────────────────────
def plot_interval_analysis(stats: dict, out_dir: Path, colors: dict):
    genres = sorted(stats)
    step_ratios = [stats[g]["interval_stats"]["step_ratio"] * 100 for g in genres]
    leap_ratios = [stats[g]["interval_stats"]["leap_ratio"] * 100 for g in genres]
    mean_abs    = [stats[g]["interval_stats"]["mean_abs"] for g in genres]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(genres) * 2), 5))
    fig.patch.set_facecolor("#FAFAF8")

    x     = np.arange(len(genres))
    width = 0.4
    ax1.bar(x - width/2, step_ratios, width, label="Steps (≤2)", color="#2A9D8F", alpha=0.85, edgecolor="white")
    ax1.bar(x + width/2, leap_ratios, width, label="Leaps (>2)", color="#E63946", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(genres, rotation=30, ha="right", fontsize=9)
    ax1.legend(fontsize=9, framealpha=0.7)
    style_ax(ax1, title="Steps vs Leaps by Genre", ylabel="% of intervals")

    bars2 = ax2.bar(genres, mean_abs,
                    color=[colors[g] for g in genres], alpha=0.85,
                    edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars2, mean_abs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax2.set_xticks(range(len(genres)))
    ax2.set_xticklabels(genres, rotation=30, ha="right", fontsize=9)
    style_ax(ax2, title="Mean Absolute Interval by Genre", ylabel="Interval size (abstract units)")

    fig.suptitle("Melodic Interval Analysis", fontsize=14, fontweight="bold",
                 color="#111111", y=1.02)
    fig.tight_layout()
    path = out_dir / "04_interval_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 5. Gong cycle density ─────────────────────────────────────────────────────
def plot_gong_cycle_density(stats: dict, out_dir: Path, colors: dict):
    genres = sorted(stats)
    # Filter genres that actually have gong data
    genres_with_data = [g for g in genres if stats[g]["gong_densities"]]
    if not genres_with_data:
        print("  Skipping gong cycle plot (no gong markers found).")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(genres_with_data) * 1.6), 5))
    fig.patch.set_facecolor("#FAFAF8")

    data   = [stats[g]["gong_densities"] for g in genres_with_data]
    clrs   = [colors[g] for g in genres_with_data]
    bp     = ax.boxplot(data, patch_artist=True, medianprops=dict(color="#111111", linewidth=1.8),
                        whiskerprops=dict(color="#888888"), capprops=dict(color="#888888"),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgewidth=0))
    for patch, c in zip(bp["boxes"], clrs):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres_with_data, rotation=30, ha="right", fontsize=9)
    style_ax(ax, title="Gong Cycle Length Distribution by Genre",
             ylabel="Notes per gong cycle")
    fig.tight_layout()
    path = out_dir / "05_gong_cycle_density.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 6. Repeat section detection ───────────────────────────────────────────────
def plot_repeat_detection(stats: dict, out_dir: Path, colors: dict):
    genres = sorted(stats)
    n_pieces = [stats[g]["n_pieces"] for g in genres]
    with_rep = [stats[g]["pieces_with_repeat"] for g in genres]
    without  = [n - w for n, w in zip(n_pieces, with_rep)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(genres) * 2), 5))
    fig.patch.set_facecolor("#FAFAF8")

    x     = np.arange(len(genres))
    width = 0.45
    ax1.bar(x, with_rep, width, label="Has repeat", color="#2A9D8F", alpha=0.85, edgecolor="white")
    ax1.bar(x, without,  width, bottom=with_rep, label="No repeat",
            color="#DDDDDD", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(genres, rotation=30, ha="right", fontsize=9)
    ax1.legend(fontsize=9, framealpha=0.7)
    style_ax(ax1, title="Pieces With/Without Repeat Brackets", ylabel="Number of pieces")

    total_reps = [stats[g]["total_repeats"] for g in genres]
    bars2 = ax2.bar(genres, total_reps,
                    color=[colors[g] for g in genres], alpha=0.85,
                    edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars2, total_reps):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(val), ha="center", va="bottom", fontsize=9, color="#333333")
    ax2.set_xticks(range(len(genres)))
    ax2.set_xticklabels(genres, rotation=30, ha="right", fontsize=9)
    style_ax(ax2, title="Total Repeat Brackets per Genre", ylabel="Count")

    fig.suptitle("Repeat Section Detection", fontsize=14, fontweight="bold",
                 color="#111111", y=1.02)
    fig.tight_layout()
    path = out_dir / "06_repeat_detection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 7. Common sections per genre ─────────────────────────────────────────────
def plot_common_sections(stats: dict, out_dir: Path, colors: dict):
    genres = sorted(stats)
    all_sec_types = sorted({s for g in genres for s in stats[g]["section_counts"].keys()})

    fig, ax = plt.subplots(figsize=(max(9, len(genres) * 1.6), max(5, len(all_sec_types) * 0.7)))
    fig.patch.set_facecolor("#FAFAF8")

    matrix = np.array([
        [stats[g]["section_counts"].get(s, 0) for g in genres]
        for s in all_sec_types
    ], dtype=float)

    # Normalise per genre by number of pieces
    for j, genre in enumerate(genres):
        n = stats[genre]["n_pieces"] or 1
        matrix[:, j] /= n

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(genres)))
    ax.set_yticks(range(len(all_sec_types)))
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(all_sec_types, fontsize=9)

    for i in range(len(all_sec_types)):
        for j in range(len(genres)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if val < 3 else "white")

    plt.colorbar(im, ax=ax, label="Avg occurrences per piece", shrink=0.8)
    style_ax(ax, title="Section Occurrence Heatmap (avg per piece)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    path = out_dir / "07_section_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 8. Summary overview ───────────────────────────────────────────────────────
def plot_summary_overview(stats: dict, out_dir: Path, colors: dict):
    """Radar / spider chart comparing genres across 5 normalised dimensions."""
    genres = sorted(stats)
    if len(genres) < 2:
        return

    dimensions = [
        "Pitch\ndiversity",
        "High register\n%",
        "Marker\ndensity",
        "Leap\nratio",
        "Gong cycle\nlength",
    ]

    def extract(g):
        s = stats[g]
        pd    = s["pitch_dist"]
        total = sum(pd.values()) or 1
        pitch_div = len([v for v in pd.values() if v / total > 0.05]) / 7  # normalised 0-1
        hi_reg    = s["register_dist"].get(1, 0) / (sum(s["register_dist"].values()) or 1)
        marker    = min(s["marker_density"] / 2.0, 1.0)
        leap      = s["interval_stats"]["leap_ratio"]
        gong_mean = np.mean(s["gong_densities"]) / 32 if s["gong_densities"] else 0
        gong_norm = min(gong_mean, 1.0)
        return [pitch_div, hi_reg, marker, leap, gong_norm]

    N    = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#F8F5F0")

    for genre in genres:
        vals = extract(genre)
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[genre], linewidth=2, alpha=0.85, label=genre)
        ax.fill(angles, vals, color=colors[genre], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9, color="#333333")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#888888")
    ax.spines["polar"].set_color("#CCCCCC")
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_title("Genre Comparison Radar", fontsize=13, fontweight="bold",
                 color="#111111", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.7)

    fig.tight_layout()
    path = out_dir / "00_summary_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    source  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/analysis")

    if not source.exists():
        print(f"Error: source directory not found: {source}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Loading corpus from: {source} ──")
    corpus = load_corpus(source)

    if not corpus:
        print("No pieces loaded. Check your folder structure.")
        sys.exit(1)

    total_pieces = sum(len(v) for v in corpus.values())
    print(f"\nLoaded {total_pieces} pieces across {len(corpus)} genres.")
    for g, pieces in sorted(corpus.items()):
        print(f"  {g}: {len(pieces)} piece(s)")

    print(f"\n── Aggregating statistics ──")
    stats  = aggregate(corpus)
    colors = genre_colors(corpus.keys())

    print(f"\n── Generating plots → {out_dir}/ ──")
    plot_summary_overview(stats, out_dir, colors)
    plot_pitch_distribution(stats, out_dir, colors)
    plot_register_usage(stats, out_dir, colors)
    plot_note_complexity(stats, out_dir, colors)
    plot_interval_analysis(stats, out_dir, colors)
    plot_gong_cycle_density(stats, out_dir, colors)
    plot_repeat_detection(stats, out_dir, colors)
    plot_common_sections(stats, out_dir, colors)

    print(f"\n✅  All done. {len(list(out_dir.glob('*.png')))} plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
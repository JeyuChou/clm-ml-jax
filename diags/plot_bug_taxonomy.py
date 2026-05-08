"""
plot_bug_taxonomy.py — Bug type distribution for Phase 5b differentiability repair campaign.

Horizontal bar chart: 9 bug types, ordered by count, with a twin-axis scatter
showing average debugging attempts per type. Intended for JAXES paper appendix
(Section: Differentiability Repair Campaign).

Data source: CHANGELOG.md sessions 1-46 (1 Apr – 8 May 2026), mined in
diags/output/agentic_pipeline_analysis.md.

Usage:
    python diags/plot_bug_taxonomy.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"
_PAPER_FIGS = _HERE.parent / "Paper" / "jaxes_paper" / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "text.usetex":       False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Data (all hardcoded from CHANGELOG analysis) ──────────────────────────────
# Columns: (code, short_label, count, percent, avg_attempts, 1_shot_all)
TYPES = [
    ("T1", "NaN Gradient\n(jnp.where)",     17, 35.4, 1.0,  True),
    ("T2", "Zero / Wrong\nGradient",          9, 18.8, 1.3,  False),
    ("T7", "Optimization\nAlgorithm",         4,  8.3, 1.5,  False),
    ("T4", "Memory / OOM",                    3,  6.3, 1.7,  False),
    ("T5", "Gradient\nExplosion",             3,  6.3, 2.0,  False),
    ("T6", "Device–Host\nSync",               3,  6.3, 1.0,  True),
    ("T8", "Crash / Compile",                 3,  6.3, 1.3,  False),
    ("T9", "Diagnostic\nReliability",         3,  6.3, 1.3,  False),
    ("T3", "XLA\nRecompilation",              2,  4.2, 1.0,  True),
]

# Color per type (used consistently across all 3 figures)
TYPE_COLORS = {
    "T1": "#DC2626",   # red      — dominant, dangerous
    "T2": "#EA580C",   # orange
    "T3": "#CA8A04",   # amber
    "T4": "#65A30D",   # olive-green
    "T5": "#7C3AED",   # purple   — hardest
    "T6": "#0891B2",   # cyan
    "T7": "#1D4ED8",   # blue
    "T8": "#6B7280",   # gray
    "T9": "#9CA3AF",   # light gray
}

codes    = [t[0] for t in TYPES]
labels   = [f"{t[0]}  {t[1]}" for t in TYPES]
counts   = [t[2] for t in TYPES]
pcts     = [t[3] for t in TYPES]
avg_att  = [t[4] for t in TYPES]
one_shot = [t[5] for t in TYPES]
colors   = [TYPE_COLORS[c] for c in codes]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.8, 3.5))
ax_r = ax.twinx()

y = np.arange(len(TYPES))
bar_h = 0.62

# Bars: colored by type
bars = ax.barh(y, counts, height=bar_h, color=colors, alpha=0.82, zorder=3)

# Hatch the "always 1-shot" bars to mark autonomy
for bar, is_one in zip(bars, one_shot):
    if is_one:
        bar.set_hatch("///")
        bar.set_edgecolor("white")
        bar.set_linewidth(0.4)

# Count labels inside bars
for i, (bar, n, p) in enumerate(zip(bars, counts, pcts)):
    x_txt = n * 0.5 if n > 2 else n + 0.15
    ha = "center" if n > 2 else "left"
    color = "white" if n > 2 else "#333333"
    ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
            f"{n}  ({p:.0f}%)", va="center", ha=ha,
            color=color, fontsize=7.5, fontweight="bold")

# Right-axis: avg attempts scatter
ax_r.scatter(avg_att, y, color="black", marker="D", s=28, zorder=5,
             label="Avg. attempts")
# Dotted guide lines from bar end to scatter dot
for i, (n, a) in enumerate(zip(counts, avg_att)):
    # map avg_att [1.0, 2.0] to the right-axis range
    pass  # visual guide is enough; no extra line needed

# Vertical reference: overall mean 1.33
ax_r.axvline(1.33, color="#999999", linestyle="--", linewidth=0.9, zorder=2,
             label="Mean (1.33)")

# Axes formatting
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.8)
ax.set_xlabel("Number of bugs  ($n$ = 47 total)", fontsize=8.5)
ax.set_xlim(0, 20)
ax.xaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)

ax_r.set_ylabel("Avg. attempts to fix", fontsize=8.5)
ax_r.set_ylim(-0.6, len(TYPES) - 0.4)
ax_r.set_xlim(0.7, 2.4)
ax_r.set_xticks([1.0, 1.33, 1.5, 2.0])
ax_r.set_xticklabels(["1.0", "1.33\n(mean)", "1.5", "2.0"], fontsize=7.5)
ax_r.spines["top"].set_visible(False)

# Legend
patch_auto = mpatches.Patch(facecolor="white", hatch="///", edgecolor="gray",
                             linewidth=0.5, label="Always 1-shot (autonomous)")
patch_multi = mpatches.Patch(facecolor="#999999", alpha=0.5,
                              label="Multi-attempt (human re-spec.)")
handle_mean = plt.Line2D([0], [0], color="#999999", linestyle="--",
                          linewidth=0.9, label="Mean attempts (1.33)")
handle_dot = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                         markersize=5, label="Avg. attempts (right axis)")
ax.legend(handles=[patch_auto, patch_multi, handle_mean, handle_dot],
          loc="lower right", fontsize=7, framealpha=0.85,
          handlelength=1.4, handletextpad=0.5)

ax.set_title("Phase 5b repair campaign: bug type distribution", fontsize=9,
             pad=6)

fig.subplots_adjust(left=0.27, right=0.88, top=0.92, bottom=0.10)

# ── Save ──────────────────────────────────────────────────────────────────────
_FIGURES_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    out = _FIGURES_DIR / f"bug_taxonomy_chart.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

# Copy to paper figures directory if it exists
if _PAPER_FIGS.is_dir():
    for ext in ("pdf", "png"):
        src = _FIGURES_DIR / f"bug_taxonomy_chart.{ext}"
        dst = _PAPER_FIGS / f"bug_taxonomy_chart.{ext}"
        shutil.copy(src, dst)
        print(f"Copied → {dst}")
else:
    print(f"Paper figures dir not found: {_PAPER_FIGS}  (skipping copy)")

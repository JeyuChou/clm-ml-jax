"""
plot_repair_timeline.py — Differentiability repair campaign timeline for JAXES paper.

Shows session density (bugs fixed per session), cumulative bug count, and phase
annotations over the 5-week repair campaign (1 Apr – 8 May 2026, sessions 1-46).

Data: hardcoded from CHANGELOG.md sessions 1-46.

Usage:
    python diags/plot_repair_timeline.py
"""
from __future__ import annotations

import shutil
from pathlib import Path
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# ── Session data (date, bugs_fixed_in_session) ────────────────────────────────
# Derived from CHANGELOG sessions 1-46. Bugs fixed = distinct bugs resolved in
# that session's code changes. Sessions without code changes not listed.
# Approximate: CHANGELOG provides session-level narrative, not exact per-session counts.
_RAW = [
    # (date_str,  bugs_fixed)
    ("2026-04-01", 12),   # session 1:  T1 NaN batch (9 modules), T8 crash, T6 ×2
    ("2026-04-02",  5),   # sessions 2-3: T1 ×3, T6 ×2
    ("2026-04-03",  4),   # sessions 4-5: T1 ×2, T4(OOM), T1 Monin-Obukhov
    ("2026-04-06",  5),   # session 12: T1 fractional powers batch (5 modules)
    ("2026-04-08",  4),   # sessions 13-14: T2 ×3 (radiation overwrite, diag skip, vcmax25)
    ("2026-04-09",  3),   # sessions 17-18: T5 WUE IFT (B32), T9 spval (B33)
    ("2026-04-10",  5),   # sessions 24-28: T5 Obukhov (B34), T2 float() (B35),
                          #                 T2 MLpftcon (B36), T2 JIT (B37), T3 ×2
    ("2026-04-14",  2),   # sessions 30-31: T5 ci-scan NaN (B38), Jacobian dpai fix
    ("2026-04-22",  2),   # sessions 33-34: T5 Medlyn ci-scan (B38 continued), T3 lru_cache
    ("2026-04-23",  1),   # session 37:  T2 g1_MED explicit arg (B39)
    ("2026-04-24",  1),   # session 38:  T1 alpha_pbot inactive layers (B40)
    ("2026-04-28",  2),   # session 42:  T7 equifinality (B44), step-index (B42)
    ("2026-04-29",  1),   # session 43:  T7 Adam beta2 (B43)
    ("2026-05-08",  3),   # sessions 44-46: T7 multi-step JIT (B41), T8 GPU contention (B46),
                          #                 T8 agent race (B47), T9 timing barrier (B45)
]

dates_raw = [date.fromisoformat(d) for d, _ in _RAW]
bugs_raw  = [b for _, b in _RAW]

# Build daily arrays from start to end
d_start = date(2026, 4, 1)
d_end   = date(2026, 5, 9)
n_days  = (d_end - d_start).days
all_dates = [d_start + timedelta(days=i) for i in range(n_days)]

bugs_per_day = np.zeros(n_days, dtype=int)
for d, b in zip(dates_raw, bugs_raw):
    idx = (d - d_start).days
    bugs_per_day[idx] += b

cumulative = np.cumsum(bugs_per_day)

# ── Phase bands ───────────────────────────────────────────────────────────────
PHASES = [
    (date(2026, 4,  1), date(2026, 4,  6), "#FEF3C7", "NaN elimination\n(T1)"),
    (date(2026, 4,  6), date(2026, 4, 12), "#FECACA", "Zero-gradient\nroot cause (T2)"),
    (date(2026, 4, 12), date(2026, 4, 27), "#DBEAFE", "IFT + param\ninjection (T2, T5)"),
    (date(2026, 4, 27), date(2026, 5,  9), "#DCFCE7", "Calibration /\noptimization (T7)"),
]

# ── Milestones ────────────────────────────────────────────────────────────────
MILESTONES = [
    (date(2026, 4,  1), "First jax.grad\n12 NaN bugs",      0.88),
    (date(2026, 4, 10), "All 5 params\nGPU-verified",         0.64),
    (date(2026, 4, 27), "7-param Jacobian\n(all non-zero)",   0.42),
    (date(2026, 5,  8), "p=10 calibration\n+ Tikhonov",      0.30),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 3.1))
ax_r = ax.twinx()

# Phase bands
for p_start, p_end, color, label in PHASES:
    xs = mdates.date2num([p_start, p_end])
    ax.axvspan(xs[0], xs[1], color=color, alpha=0.85, zorder=0)
    mid = mdates.date2num([p_start + (p_end - p_start) / 2])[0]
    ax.text(mid, -2.7, label, ha="center", va="top", fontsize=6.5,
            style="italic", color="#555555", zorder=4)

# Session stem plot (bugs per session)
dates_plot  = [d_start + timedelta(days=i) for i in range(n_days)]
dates_num   = mdates.date2num(dates_plot)
nonzero = bugs_per_day > 0
for i, (dn, b, nz) in enumerate(zip(dates_num, bugs_per_day, nonzero)):
    if nz:
        ax.bar(dn, b, width=0.6, color="#1D4ED8", alpha=0.75, zorder=3)

# Cumulative line on right axis
cum_dates = mdates.date2num(dates_plot)
ax_r.step(cum_dates, cumulative, where="post", color="#6B7280",
          linewidth=1.4, linestyle="--", zorder=2, label="Cumul. bugs fixed")
ax_r.fill_between(cum_dates, 0, cumulative, step="post",
                  color="#6B7280", alpha=0.08, zorder=1)

# Milestone annotations
for m_date, label, y_frac in MILESTONES:
    mx = mdates.date2num([m_date])[0]
    ax.axvline(mx, color="#555555", linewidth=0.9, linestyle=":", zorder=4,
               alpha=0.7)
    ax_r_max = max(cumulative) * 1.12
    y_pos = ax_r_max * y_frac
    ax_r.annotate(label, xy=(mx, y_pos),
                  xytext=(mx + 0.6, y_pos),
                  fontsize=6.5, color="#333333",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="#aaaaaa", alpha=0.92, linewidth=0.5),
                  zorder=6)

# Axes formatting
ax.set_xlim(mdates.date2num([d_start])[0] - 0.5,
            mdates.date2num([d_end])[0] + 0.5)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
ax.set_ylim(-4, max(bugs_per_day) * 1.25)
ax.set_ylabel("Bugs fixed per session", fontsize=8.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.tick_params(axis="x", rotation=25)

ax_r.set_ylabel("Cumulative bugs fixed", fontsize=8.5, color="#6B7280")
ax_r.tick_params(axis="y", labelcolor="#6B7280")
ax_r.set_ylim(0, max(cumulative) * 1.18)
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(True)

# Legend
bar_patch = mpatches.Patch(color="#1D4ED8", alpha=0.75, label="Bugs fixed (left axis)")
cum_line  = plt.Line2D([0], [0], color="#6B7280", linestyle="--", linewidth=1.4,
                        label="Cumulative (right axis)")
ax.legend(handles=[bar_patch, cum_line], loc="upper left", fontsize=7,
          framealpha=0.9, handlelength=1.4)

ax.set_title("Phase 5b differentiability repair campaign  "
             "(1 Apr – 8 May 2026,  46 sessions)", fontsize=8.5, pad=5)

fig.tight_layout(rect=[0, 0.07, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
_FIGURES_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    out = _FIGURES_DIR / f"repair_timeline.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

if _PAPER_FIGS.is_dir():
    for ext in ("pdf", "png"):
        src = _FIGURES_DIR / f"repair_timeline.{ext}"
        dst = _PAPER_FIGS / f"repair_timeline.{ext}"
        shutil.copy(src, dst)
        print(f"Copied → {dst}")
else:
    print(f"Paper figures dir not found: {_PAPER_FIGS}  (skipping copy)")

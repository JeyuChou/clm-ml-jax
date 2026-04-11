"""
plot_benchmarks.py — Comprehensive performance benchmark figure for CLM-ML-JAX.

Reads two CSV files produced by the benchmark jobs:
  diags/figures/laxscan_benchmark.csv   — diff (lax.scan) vs non-diff (Python loop)
  diags/figures/multisite_benchmark.csv — vmap N-site GPU vs CPU scaling

Produces one figure with three panels:
  Panel A: diff (lax.scan) vs non-diff, Euler and RK4 — steady-state ms/step
           + compile time annotation + speedup labels
  Panel B: ms/site/step vs N sites for GPU and CPU (log–log)
           with reference Fortran single-site time marked
  Panel C: GPU vs CPU throughput at N=1, 16, 32 (sites/second)

Output: diags/figures/benchmark_summary.png  (300 dpi)

Usage (run from project root after benchmark jobs complete):
    python diags/plot_benchmarks.py
    python diags/plot_benchmarks.py --laxscan path/to/laxscan.csv \
                                    --multisite path/to/multisite.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"

# ── Known baseline numbers (previous runs, jobs 7329052 + 7329441) ────────────
# Used as "before" annotation if laxscan CSV not available yet.
_PREV = {
    "Euler": {"diff_ss_ms": 37.2,  "nondiff_ss_ms": 6095.0,  "diff_compile_s": 290.0},
    "RK4":   {"diff_ss_ms": 37.8,  "nondiff_ss_ms": 117164.0, "diff_compile_s": 7200.0},
}

# Fortran reference: 31 days / 1488 steps = 75.7 ms/step (single site, Derecho CPU)
_FORTRAN_MS_STEP = 112600.0 / 1488.0   # 75.7 ms/step

# ── Color palette ─────────────────────────────────────────────────────────────
C_GPU    = "#2563EB"   # blue
C_CPU    = "#DC2626"   # red
C_DIFF   = "#16A34A"   # green  (lax.scan diff mode)
C_NDIFF  = "#9333EA"   # purple (Python loop non-diff)
C_FORT   = "#F59E0B"   # amber  (Fortran reference)
ALPHA    = 0.85


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_laxscan(path: Path) -> dict:
    """Load laxscan_benchmark.csv → {mode: {field: value}}."""
    data = {}
    if not path.exists():
        print(f"  [plot] WARNING: {path} not found — using previous baseline numbers",
              file=sys.stderr)
        return _PREV.copy()
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["mode"]] = {
                "diff_ss_ms":     float(row["diff_ss_ms"]),
                "nondiff_ss_ms":  float(row["nondiff_ss_ms"]),
                "diff_compile_s": float(row["diff_compile_s"]),
                "speedup":        float(row["speedup"]),
            }
    return data


def _load_multisite(path: Path) -> dict[str, list[dict]]:
    """Load multisite_benchmark.csv → {"gpu": [...], "cpu": [...]}."""
    data: dict[str, list[dict]] = {"gpu": [], "cpu": []}
    if not path.exists():
        print(f"  [plot] WARNING: {path} not found — multisite panels will be empty",
              file=sys.stderr)
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            backend = row["backend"].lower()
            if backend in data:
                data[backend].append({
                    "N":           int(row["N"]),
                    "vmap_ss_s":   float(row["vmap_ss_s"]),
                    "seq_ss_s":    float(row["seq_ss_s"]),
                    "speedup":     float(row["speedup"]),
                    "ms_per_site": float(row["ms_per_site"]),
                })
    for backend in data:
        data[backend].sort(key=lambda r: r["N"])
    return data


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _panel_a(ax, laxscan: dict):
    """Panel A: lax.scan diff mode vs Python-loop non-diff — ms/step bar chart."""
    modes   = ["Euler", "RK4"]
    x       = np.arange(len(modes))
    w       = 0.32
    offset  = w / 2 + 0.04

    diff_ms  = [laxscan.get(m, {}).get("diff_ss_ms",   np.nan) for m in modes]
    ndiff_ms = [laxscan.get(m, {}).get("nondiff_ss_ms", np.nan) for m in modes]

    bars_d = ax.bar(x - offset, diff_ms,  width=w, color=C_DIFF,  alpha=ALPHA,
                    label="Diff mode (lax.scan)", zorder=3)
    bars_n = ax.bar(x + offset, ndiff_ms, width=w, color=C_NDIFF, alpha=ALPHA,
                    label="Non-diff (Python loop)", zorder=3)

    # Speedup annotations on top of diff bars
    for i, mode in enumerate(modes):
        sp = laxscan.get(mode, {}).get("speedup", np.nan)
        if np.isfinite(sp):
            ax.text(x[i] - offset, diff_ms[i] * 1.08, f"{sp:.0f}×",
                    ha="center", va="bottom", fontsize=9, color=C_DIFF, fontweight="bold")

    # Compile time annotation
    for i, mode in enumerate(modes):
        ct = laxscan.get(mode, {}).get("diff_compile_s", np.nan)
        if np.isfinite(ct):
            label = f"compile\n{ct:.0f}s" if ct < 3600 else f"compile\n{ct/3600:.1f}h"
            ax.text(x[i] - offset, diff_ms[i] / 2, label,
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_ylabel("Steady-state time (ms / CLM step)", fontsize=10)
    ax.set_title("A.  Diff (lax.scan) vs Non-diff (Python loop)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)

    # Value labels on non-diff bars (they're large)
    for bar, val in zip(bars_n, ndiff_ms):
        if np.isfinite(val):
            label = f"{val/1000:.0f}s" if val > 1000 else f"{val:.0f}ms"
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.05,
                    label, ha="center", va="bottom", fontsize=8, color=C_NDIFF)


def _panel_b(ax, ms: dict[str, list[dict]]):
    """Panel B: ms/site/step vs N (log–log) for GPU and CPU."""
    for backend, color, label in [
        ("gpu", C_GPU, "GPU (A100)"),
        ("cpu", C_CPU, "CPU"),
    ]:
        rows = ms[backend]
        if not rows:
            continue
        Ns  = [r["N"] for r in rows]
        mss = [r["ms_per_site"] for r in rows]
        ax.plot(Ns, mss, "o-", color=color, lw=2, ms=7, label=label, zorder=3)
        for n, m in zip(Ns, mss):
            ax.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                        xytext=(6, 3), fontsize=7.5, color=color)

    # Fortran reference line
    ax.axhline(_FORTRAN_MS_STEP, color=C_FORT, ls="--", lw=1.5,
               label=f"Fortran reference ({_FORTRAN_MS_STEP:.0f} ms, Derecho CPU)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of sites (N)", fontsize=10)
    ax.set_ylabel("ms / site / step", fontsize=10)
    ax.set_title("B.  Multi-site vmap throughput (GPU vs CPU)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, which="both", alpha=0.3)
    ax.xaxis.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)


def _panel_c(ax, ms: dict[str, list[dict]]):
    """Panel C: GPU vs CPU sites/second at N=1, 16, 32 (grouped bar)."""
    target_Ns = [1, 16, 32]
    x = np.arange(len(target_Ns))
    w = 0.32

    def _get(backend, N):
        for r in ms[backend]:
            if r["N"] == N:
                return r["vmap_ss_s"]
        return np.nan

    gpu_thru = [target_Ns[i] / _get("gpu", N) if np.isfinite(_get("gpu", N)) else np.nan
                for i, N in enumerate(target_Ns)]
    cpu_thru = [target_Ns[i] / _get("cpu", N) if np.isfinite(_get("cpu", N)) else np.nan
                for i, N in enumerate(target_Ns)]

    bars_g = ax.bar(x - w/2 - 0.02, gpu_thru, width=w, color=C_GPU, alpha=ALPHA,
                    label="GPU (A100)", zorder=3)
    bars_c = ax.bar(x + w/2 + 0.02, cpu_thru, width=w, color=C_CPU, alpha=ALPHA,
                    label="CPU", zorder=3)

    # GPU/CPU ratio labels
    for i in range(len(target_Ns)):
        g, c = gpu_thru[i], cpu_thru[i]
        if np.isfinite(g) and np.isfinite(c) and c > 0:
            ratio = g / c
            ypos  = max(g, c) * 1.05
            ax.text(x[i], ypos, f"GPU {ratio:.1f}×\nfaster",
                    ha="center", va="bottom", fontsize=8, color=C_GPU, fontweight="bold")

    # Value labels on bars
    for bars, thru in [(bars_g, gpu_thru), (bars_c, cpu_thru)]:
        for bar, v in zip(bars, thru):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in target_Ns], fontsize=11)
    ax.set_ylabel("Throughput (sites / second)", fontsize=10)
    ax.set_title("C.  GPU vs CPU throughput at N=1, 16, 32", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot CLM-ML-JAX benchmark results")
    parser.add_argument("--laxscan",   default=str(_FIGURES_DIR / "laxscan_benchmark.csv"))
    parser.add_argument("--multisite", default=str(_FIGURES_DIR / "multisite_benchmark.csv"))
    parser.add_argument("--out",       default=str(_FIGURES_DIR / "benchmark_summary.png"))
    args = parser.parse_args()

    print("Loading data ...", flush=True)
    laxscan  = _load_laxscan(Path(args.laxscan))
    multisite = _load_multisite(Path(args.multisite))

    print("Plotting ...", flush=True)
    fig = plt.figure(figsize=(17, 5.5))
    fig.suptitle("CLM-ML-JAX GPU Performance Benchmarks (A100)",
                 fontsize=13, fontweight="bold", y=1.02)

    gs = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _panel_a(ax_a, laxscan)
    _panel_b(ax_b, multisite)
    _panel_c(ax_c, multisite)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)

    # Print summary table to stdout
    print("\n── lax.scan summary ─────────────────────────────────────────────")
    print(f"  {'Mode':<8}  {'diff ms':>10}  {'nondiff ms':>14}  {'speedup':>10}  {'compile s':>10}")
    for mode in ("Euler", "RK4"):
        d = laxscan.get(mode, {})
        print(f"  {mode:<8}  {d.get('diff_ss_ms', float('nan')):>10.1f}  "
              f"{d.get('nondiff_ss_ms', float('nan')):>14.1f}  "
              f"{d.get('speedup', float('nan')):>10.0f}x  "
              f"{d.get('diff_compile_s', float('nan')):>10.1f}")

    print("\n── multisite summary (ms/site/step) ─────────────────────────────")
    all_Ns = sorted(set(r["N"] for b in multisite.values() for r in b))
    print(f"  {'N':>5}  {'GPU ms':>10}  {'CPU ms':>10}  {'GPU/CPU speedup':>18}")
    for N in all_Ns:
        g = next((r["ms_per_site"] for r in multisite["gpu"] if r["N"] == N), float("nan"))
        c = next((r["ms_per_site"] for r in multisite["cpu"] if r["N"] == N), float("nan"))
        ratio = c / g if (np.isfinite(g) and np.isfinite(c) and g > 0) else float("nan")
        print(f"  {N:>5}  {g:>10.2f}  {c:>10.2f}  {ratio:>18.1f}x")


if __name__ == "__main__":
    main()

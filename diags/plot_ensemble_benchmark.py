"""
plot_ensemble_benchmark.py — GPU vs CPU ensemble benchmark figure.

Reads diags/figures/ensemble_benchmark.csv produced from log of job 7527971 (A40).
Produces diags/figures/ensemble_benchmark.png (2-panel).

Optionally overlays Fortran reference timing from a CSV produced by the
Fortran benchmark suite (clm-ml-fortran/benchmark/run_ensemble_benchmark.sh):
  --fortran path/to/ensemble_benchmark_fortran.csv

Panel A: ms/sample vs N (log-log) for GPU and CPU vmap
         + optional Fortran sequential and parallel lines
Panel B: GPU/CPU speedup vs N (ms/sample ratio)
         + optional Fortran/GPU speedup lines

Usage:
    python diags/plot_ensemble_benchmark.py
    python diags/plot_ensemble_benchmark.py \
        --fortran path/to/ensemble_benchmark_fortran.csv
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
import matplotlib.ticker

_HERE        = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"

C_GPU       = "#2563EB"   # blue
C_CPU       = "#DC2626"   # red
C_FORT_SEQ  = "#D97706"   # amber  (Fortran sequential)
C_FORT_PAR  = "#92400E"   # brown  (Fortran parallel)
ALPHA       = 0.85


def _load(path: Path):
    gpu, cpu = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            d = {
                "N":            int(row["N"]),
                "ms_per_sample": float(row["ms_per_sample"]) if row["ms_per_sample"] else np.nan,
                "vmap_ss_ms":   float(row["vmap_ss_ms"])   if row["vmap_ss_ms"]   else np.nan,
                "seq_ss_ms":    float(row["seq_ss_ms"])    if row["seq_ss_ms"]    else np.nan,
            }
            if row["backend"] == "gpu":
                gpu.append(d)
            else:
                cpu.append(d)
    return sorted(gpu, key=lambda r: r["N"]), sorted(cpu, key=lambda r: r["N"])


def _load_fortran_ensemble(path: Path | None) -> dict[str, list[dict]]:
    """
    Load ensemble_benchmark_fortran.csv produced by run_ensemble_benchmark.sh.

    Expected columns: backend,N,run_wall_s,ms_per_sample,notes
    backend = 'fortran_seq' or 'fortran_par'

    Returns {"seq": [...], "par": [...]} each sorted by N.
    Returns empty lists if path is None or file does not exist.
    """
    result: dict[str, list[dict]] = {"seq": [], "par": []}
    if path is None:
        return result
    if not path.exists():
        print(f"  [plot] INFO: Fortran ensemble CSV not found: {path}", file=sys.stderr)
        return result
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                N = int(row["N"])
                ms = float(row["ms_per_sample"])
            except (KeyError, ValueError) as e:
                print(f"  [plot] Skipping malformed Fortran row {row}: {e}", file=sys.stderr)
                continue
            backend = row.get("backend", "")
            if "seq" in backend:
                result["seq"].append({"N": N, "ms_per_sample": ms,
                                      "notes": row.get("notes", "")})
            elif "par" in backend:
                result["par"].append({"N": N, "ms_per_sample": ms,
                                      "notes": row.get("notes", "")})
    for k in result:
        result[k].sort(key=lambda r: r["N"])
    print(f"  [plot] Loaded Fortran ensemble data: "
          f"{len(result['seq'])} seq + {len(result['par'])} par N-values from {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Plot CLM-ML-JAX ensemble benchmark results")
    parser.add_argument("--csv",      default=str(_FIGURES_DIR / "ensemble_benchmark.csv"),
                        help="JAX ensemble benchmark CSV (default: diags/figures/ensemble_benchmark.csv)")
    parser.add_argument("--out",      default=str(_FIGURES_DIR / "ensemble_benchmark.png"),
                        help="Output PNG path")
    parser.add_argument(
        "--fortran",
        default=None,
        metavar="FORTRAN_CSV",
        help="Optional: path to ensemble_benchmark_fortran.csv from the Fortran benchmark "
             "suite (clm-ml-fortran/benchmark/run_ensemble_benchmark.sh). "
             "Overlays Fortran sequential and parallel lines on Panel A, "
             "and Fortran/GPU speedup on Panel B.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    gpu, cpu = _load(csv_path)
    fortran = _load_fortran_ensemble(Path(args.fortran) if args.fortran else None)

    gpu_Ns  = [r["N"] for r in gpu]
    gpu_mss = [r["ms_per_sample"] for r in gpu]
    cpu_Ns  = [r["N"] for r in cpu]
    cpu_mss = [r["ms_per_sample"] for r in cpu]

    # Speedup: CPU ms / GPU ms at matching N values
    cpu_by_N = {r["N"]: r["ms_per_sample"] for r in cpu}
    speedup_Ns   = [r["N"] for r in gpu if r["N"] in cpu_by_N]
    speedup_vals = [cpu_by_N[r["N"]] / r["ms_per_sample"]
                    for r in gpu if r["N"] in cpu_by_N]

    # Fortran/GPU speedup: fortran_seq ms / GPU ms
    fort_seq_by_N = {r["N"]: r["ms_per_sample"] for r in fortran["seq"]}
    fort_speedup_Ns   = [r["N"] for r in gpu if r["N"] in fort_seq_by_N]
    fort_speedup_vals = [fort_seq_by_N[r["N"]] / r["ms_per_sample"]
                         for r in gpu if r["N"] in fort_seq_by_N]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Parameter Ensemble Benchmark: GPU vs CPU vmap (NVIDIA A40, Euler 1-substep)",
                 fontsize=11, fontweight="bold")

    # ── Panel A: ms/sample ─────────────────────────────────────────────────────
    ax_a.plot(gpu_Ns, gpu_mss, "o-", color=C_GPU, lw=2, ms=7, label="JAX GPU (A40)", zorder=3)
    ax_a.plot(cpu_Ns, cpu_mss, "s-", color=C_CPU, lw=2, ms=7, label="JAX CPU", zorder=3)
    for n, m in zip(gpu_Ns, gpu_mss):
        ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                      xytext=(6, 3), fontsize=7.5, color=C_GPU)
    for n, m in zip(cpu_Ns, cpu_mss):
        ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                      xytext=(6, -10), fontsize=7.5, color=C_CPU)

    # Optional Fortran lines
    if fortran["seq"]:
        fort_seq_Ns  = [r["N"] for r in fortran["seq"]]
        fort_seq_mss = [r["ms_per_sample"] for r in fortran["seq"]]
        ax_a.plot(fort_seq_Ns, fort_seq_mss, "s--", color=C_FORT_SEQ, lw=1.8, ms=6,
                  label="Fortran sequential (N runs)", zorder=3, alpha=0.9)
        for n, m in zip(fort_seq_Ns, fort_seq_mss):
            ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                          xytext=(6, 5), fontsize=7, color=C_FORT_SEQ)

    if fortran["par"]:
        fort_par_Ns  = [r["N"] for r in fortran["par"]]
        fort_par_mss = [r["ms_per_sample"] for r in fortran["par"]]
        ax_a.plot(fort_par_Ns, fort_par_mss, "^:", color=C_FORT_PAR, lw=1.8, ms=6,
                  label="Fortran parallel (N procs)", zorder=3, alpha=0.9)
        for n, m in zip(fort_par_Ns, fort_par_mss):
            ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                          xytext=(6, -11), fontsize=7, color=C_FORT_PAR)

    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    xticks = sorted(set(gpu_Ns + cpu_Ns + [r["N"] for r in fortran["seq"]]
                        + [r["N"] for r in fortran["par"]]))
    ax_a.set_xticks(xticks)
    ax_a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_a.set_xlabel("Ensemble size N (parameter samples)", fontsize=10)
    ax_a.set_ylabel("ms / sample", fontsize=10)
    ax_a.set_title("A.  Amortized cost per sample", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=9)
    ax_a.yaxis.grid(True, which="both", alpha=0.3)
    ax_a.xaxis.grid(True, which="major", alpha=0.3)
    ax_a.set_axisbelow(True)

    # ── Panel B: speedup ───────────────────────────────────────────────────────
    ax_b.plot(speedup_Ns, speedup_vals, "D-", color="#16A34A", lw=2, ms=7,
              label="JAX GPU speedup vs JAX CPU", zorder=3)
    ax_b.axhline(1.0, color="gray", ls="--", lw=1.2, label="parity (1×)")
    for n, s in zip(speedup_Ns, speedup_vals):
        ax_b.annotate(f"{s:.2f}×", (n, s), textcoords="offset points",
                      xytext=(6, 3), fontsize=8)

    # Optional: Fortran sequential / GPU speedup
    if fort_speedup_Ns:
        ax_b.plot(fort_speedup_Ns, fort_speedup_vals, "s--", color=C_FORT_SEQ,
                  lw=1.8, ms=6, label="JAX GPU speedup vs Fortran seq", zorder=3, alpha=0.9)
        for n, s in zip(fort_speedup_Ns, fort_speedup_vals):
            ax_b.annotate(f"{s:.1f}×", (n, s), textcoords="offset points",
                          xytext=(6, -12), fontsize=7.5, color=C_FORT_SEQ)

    all_speedup_Ns = sorted(set(speedup_Ns + fort_speedup_Ns))
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(all_speedup_Ns if all_speedup_Ns else speedup_Ns)
    ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b.set_xlabel("Ensemble size N", fontsize=10)
    ax_b.set_ylabel("Reference ms/sample  /  GPU ms/sample", fontsize=10)
    ax_b.set_title("B.  GPU speedup over reference (per sample)", fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=9)
    ax_b.yaxis.grid(True, alpha=0.3)
    ax_b.xaxis.grid(True, which="major", alpha=0.3)
    ax_b.set_axisbelow(True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Summary
    print("\n── JAX GPU/CPU speedup summary ──────────────────────────────────")
    for n, s in zip(speedup_Ns, speedup_vals):
        print(f"  N={n:4d}  {s:.2f}×")
    if gpu_mss:
        print(f"\n  JAX GPU wall-clock N=2048: {gpu_mss[-1] * 2048 / 1000:.1f}s")
    if cpu_mss:
        print(f"  JAX CPU sequential N=2048 estimate: {cpu_mss[0] * 2048 / 1000:.1f}s")

    if fort_speedup_Ns:
        print("\n── JAX GPU vs Fortran sequential speedup ────────────────────────")
        for n, s in zip(fort_speedup_Ns, fort_speedup_vals):
            print(f"  N={n:4d}  {s:.2f}×")


if __name__ == "__main__":
    main()

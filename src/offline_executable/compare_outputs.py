"""
Systematic comparison of CLM-ml_v2 Python output files against Fortran validation files.

Usage:
    python compare_outputs.py [--output-dir DIR] [--val-dir DIR] [--tol TOL] [--no-plots]

Tolerance default: 0.001 (matches Fortran energy conservation threshold).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# File schema: (tag, ncols, col_names)
# ---------------------------------------------------------------------------
FLUX_COLS = [
    'time', 'rnet', 'stflx_air', 'shflx', 'lhflx', 'gppveg',
    'ustar', 'swup', 'lwup', 'tair_top', 'gsoi', 'rnsoi',
    'shsoi', 'lhsoi', 'lhflx_tr', 'lhflx_ev', 'beta', 'stflx_veg',
]
AUX_COLS = ['btran', 'lsc_top', 'psis', 'lwp_top', 'lwp_mid', 'fracminlwp']
FSUN_COLS = [
    'zen_deg', 'swvis_total', 'LAI_SAI', 'laisun', 'laisha',
    'swveg_vis', 'swvegsun_vis', 'swvegsha_vis',
    'gppveg', 'gppvegsun', 'gppvegsha',
    'lhveg', 'lhvegsun', 'lhvegsha',
    'shveg', 'shvegsun', 'shvegsha',
    'vcmax25veg', 'vcmax25sun', 'vcmax25sha',
    'gsveg', 'gsvegsun', 'gsvegsha',
    'windveg', 'windvegsun', 'windvegsha',
    'tlveg', 'tlvegsun', 'tlvegsha',
    'taveg', 'tavegsun', 'tavegsha',
]
SOILTEMP_COLS = (['time'] +
    [f'z{i}' for i in range(1, 11)] +
    [f'T{i}' for i in range(1, 11)])
# Reorder soiltemp as written: time, z1, T1, z2, T2, ...
SOILTEMP_COLS = ['time'] + [v for i in range(1, 11) for v in (f'z{i}', f'T{i}')]

FLUXPROFILE_COLS = [
    'time', 'zw', 'shf', 'lhf', 'mflx',
    'swbeam_vis', 'swbeam_nir', 'swdwn_vis', 'swdwn_nir',
    'swupw_vis', 'swupw_nir', 'lwdwn', 'lwupw',
]
# profile.out leaf rows (28 cols):
PROFILE_COLS_LEAF = [
    'time', 'zs', 'fracsun', 'lad', 'lad_sun', 'lad_sha',
    'rnleaf_sun', 'rnleaf_sha', 'shleaf_sun', 'shleaf_sha',
    'lhleaf_sun', 'lhleaf_sha', 'anet_sun', 'anet_sha',
    'apar_sun', 'apar_sha', 'gs_sun', 'gs_sha',
    'lwp_sun', 'lwp_sha', 'tleaf_sun', 'tleaf_sha',
    'vcmax25_sun', 'vcmax25_sha',
    'wind', 'tair', 'qair', 'ra',
]

SPVAL = 1e20  # treat values larger than this as missing


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_simple(path: Path, ncols_expected: int, name: str) -> np.ndarray | None:
    """Parse a space-delimited file into a float array, NaN for spval."""
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            vals = line.split()
            if not vals:
                continue
            try:
                row = [float(v) for v in vals]
            except ValueError:
                print(f"  [WARN] {name} line {lineno}: parse error, skipping")
                continue
            if len(row) != ncols_expected:
                # Allow flexible columns (for profile.out etc.) — handled by caller
                row = row[:ncols_expected] if len(row) > ncols_expected else row
            rows.append(row)
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    arr[np.abs(arr) > SPVAL] = np.nan
    return arr


def _parse_profile(path: Path) -> tuple[np.ndarray | None, bool]:
    """
    Parse profile.out, tolerating rows with 28 or 30 columns.

    Returns (array of shape (N, 28), has_column_bug) where has_column_bug is
    True if any 30-column rows were found.  For 30-col rows the two extra
    missing-value columns (cols 25-26 of the 30) are dropped so the result is
    always 28 columns.
    """
    rows = []
    has_bug = False
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            vals = line.split()
            if not vals:
                continue
            try:
                row = [float(v) for v in vals]
            except ValueError:
                print(f"  [WARN] profile.out line {lineno}: parse error, skipping")
                continue
            n = len(row)
            if n == 28:
                rows.append(row)
            elif n == 30:
                # Drop the two extra missing-value placeholders (positions 24-25, 0-indexed)
                # 30-col layout: [0..5 header+fracsun, 6..25 = 20×mv, 26..29 = wind/tair/qair/ra]
                # 28-col layout: [0..5, 6..23 = 18×mv, 24..27]
                # The last 4 are always at the end; drop cols 24 and 25 (the 2 extra)
                fixed = row[:24] + row[26:]  # keep first 24, skip 2, keep last 4
                rows.append(fixed)
                has_bug = True
            else:
                print(f"  [WARN] profile.out line {lineno}: unexpected {n} columns, skipping")
    if not rows:
        return None, has_bug
    arr = np.array(rows, dtype=float)
    arr[np.abs(arr) > SPVAL] = np.nan
    return arr, has_bug


def _parse_fluxprofile(path: Path, ncols: int = 13) -> np.ndarray | None:
    """Parse fluxprofile.out, replacing spval with NaN."""
    return _parse_simple(path, ncols, 'fluxprofile.out')


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(py: np.ndarray, ft: np.ndarray, tol: float) -> dict:
    """Compute comparison statistics for one variable (1-D arrays)."""
    diff = py - ft
    valid = ~(np.isnan(diff))
    if not np.any(valid):
        return {'n': 0, 'mae': np.nan, 'rmse': np.nan, 'max_diff': np.nan,
                'first_diff': np.nan, 'n_exceed': 0}
    d = diff[valid]
    first = diff[0] if not np.isnan(diff[0]) else np.nan
    return {
        'n':         int(np.sum(valid)),
        'mae':       float(np.mean(np.abs(d))),
        'rmse':      float(np.sqrt(np.mean(d**2))),
        'max_diff':  float(np.max(np.abs(d))),
        'first_diff': float(first),
        'n_exceed':  int(np.sum(np.abs(d) > tol)),
    }


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def _print_table(file_tag: str, col_names: list[str], stats_list: list[dict],
                 tol: float) -> None:
    header = (f"{'Variable':<22} {'N':>5} {'MAE':>10} {'RMSE':>10} "
              f"{'MaxDiff':>10} {'FirstDiff':>12} {'N>tol':>6}")
    sep = '-' * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  {file_tag}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for name, st in zip(col_names, stats_list):
        flag = '*' if st['n_exceed'] > 0 else ' '
        print(
            f"{flag}{name:<21} {st['n']:>5} {st['mae']:>10.4f} {st['rmse']:>10.4f} "
            f"{st['max_diff']:>10.4f} {st['first_diff']:>12.4f} {st['n_exceed']:>6}"
        )
    total_exceed = sum(s['n_exceed'] for s in stats_list)
    print(sep)
    print(f"  Total variables exceeding tol={tol}: {total_exceed}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_file(file_tag: str, col_names: list[str],
               py_arr: np.ndarray, ft_arr: np.ndarray,
               time_col: int | None, skip_cols: set[int],
               out_path: Path) -> None:
    plot_cols = [i for i, n in enumerate(col_names)
                 if i != time_col and i not in skip_cols]
    if not plot_cols:
        return

    time = py_arr[:, time_col] if time_col is not None else np.arange(len(py_arr))

    if not HAS_MATPLOTLIB:
        return

    ncols_plot = min(6, len(plot_cols))
    nrows_plot = (len(plot_cols) + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(ncols_plot * 3.5, nrows_plot * 2.8),
                             squeeze=False)
    fig.suptitle(f'{file_tag}  (blue=Python, orange=Fortran)', fontsize=11)

    for idx, ci in enumerate(plot_cols):
        ax = axes[idx // ncols_plot][idx % ncols_plot]
        ax.plot(time, py_arr[:, ci], lw=1.0, color='steelblue', label='Python')
        ax.plot(time, ft_arr[:, ci], lw=1.0, color='darkorange', linestyle='--', label='Fortran')
        ax.set_title(col_names[ci], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('calday', fontsize=7)

    # Hide unused subplots
    for idx in range(len(plot_cols), nrows_plot * ncols_plot):
        axes[idx // ncols_plot][idx % ncols_plot].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare_file(tag: str, py_path: Path, ft_path: Path,
                 col_names: list[str], tol: float,
                 plot_dir: Path, parse_fn=None, time_col: int = 0,
                 skip_plot_cols: set[int] | None = None) -> list[dict]:
    print(f"\n[{tag}]  {py_path.name}")
    if not py_path.exists():
        print(f"  Python output NOT FOUND: {py_path}")
        return []
    if not ft_path.exists():
        print(f"  Validation NOT FOUND: {ft_path}")
        return []

    if parse_fn is None:
        py = _parse_simple(py_path, len(col_names), tag)
        ft = _parse_simple(ft_path, len(col_names), tag)
    else:
        py = parse_fn(py_path)
        ft = parse_fn(ft_path)

    if py is None or ft is None:
        print(f"  Parse failed — skipping.")
        return []

    nrows = min(len(py), len(ft))
    if len(py) != len(ft):
        print(f"  [WARN] Row count mismatch: Python={len(py)}, Fortran={len(ft)} — comparing first {nrows} rows")
    py = py[:nrows]
    ft = ft[:nrows]

    ncols = min(py.shape[1], ft.shape[1], len(col_names))
    stats_list = []
    for ci in range(ncols):
        st = _stats(py[:, ci], ft[:, ci], tol)
        stats_list.append(st)

    _print_table(tag, col_names[:ncols], stats_list, tol)

    skip = skip_plot_cols or set()
    _plot_file(tag, col_names[:ncols], py, ft, time_col, skip,
               plot_dir / f'comparison_{tag}.png')

    return stats_list


def compare_profile(py_path: Path, ft_path: Path, tol: float,
                    plot_dir: Path) -> list[dict]:
    """Special handler for profile.out (variable column count)."""
    print(f"\n[profile.out]  {py_path.name}")
    if not py_path.exists():
        print(f"  Python output NOT FOUND"); return []
    if not ft_path.exists():
        print(f"  Validation NOT FOUND"); return []

    py, has_bug = _parse_profile(py_path)
    ft, _ = _parse_profile(ft_path)
    if has_bug:
        print("  [NOTE] Python profile.out has 30-column rows (column-count bug) — "
              "extra missing-value columns stripped for comparison.")

    if py is None or ft is None:
        print("  Parse failed — skipping."); return []

    nrows = min(len(py), len(ft))
    if len(py) != len(ft):
        print(f"  [WARN] Row count mismatch: Python={len(py)}, Fortran={len(ft)}, comparing {nrows}")
    py = py[:nrows]; ft = ft[:nrows]

    stats_list = []
    for ci, name in enumerate(PROFILE_COLS_LEAF):
        if ci >= py.shape[1]:
            break
        st = _stats(py[:, ci], ft[:, ci], tol)
        stats_list.append(st)

    _print_table('profile.out', PROFILE_COLS_LEAF[:len(stats_list)], stats_list, tol)

    # Plot a subset of meaningful columns
    skip = {0, 1}  # skip time and zs (not interesting to plot)
    _plot_file('profile', PROFILE_COLS_LEAF[:len(stats_list)], py, ft,
               time_col=None, skip_cols=skip, out_path=plot_dir / 'comparison_profile.png')
    return stats_list


def compare_fluxprofile(py_path: Path, ft_path: Path, tol: float,
                        plot_dir: Path) -> list[dict]:
    """Handler for fluxprofile.out (many spval entries in Python)."""
    print(f"\n[fluxprofile.out]  {py_path.name}")
    if not py_path.exists():
        print(f"  Python output NOT FOUND"); return []
    if not ft_path.exists():
        print(f"  Validation NOT FOUND"); return []

    py = _parse_fluxprofile(py_path, len(FLUXPROFILE_COLS))
    ft = _parse_fluxprofile(ft_path, len(FLUXPROFILE_COLS))
    if py is None or ft is None:
        print("  Parse failed — skipping."); return []

    nrows = min(len(py), len(ft))
    if len(py) != len(ft):
        print(f"  [WARN] Row mismatch: Python={len(py)}, Fortran={len(ft)}, comparing {nrows}")
    py = py[:nrows]; ft = ft[:nrows]

    stats_list = []
    for ci, name in enumerate(FLUXPROFILE_COLS):
        st = _stats(py[:, ci], ft[:, ci], tol)
        stats_list.append(st)

    _print_table('fluxprofile.out', FLUXPROFILE_COLS, stats_list, tol)

    skip = {0, 1}  # skip time and zw
    _plot_file('fluxprofile', FLUXPROFILE_COLS, py, ft,
               time_col=None, skip_cols=skip,
               out_path=plot_dir / 'comparison_fluxprofile.png')
    return stats_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare CLM-ml_v2 outputs vs validation files')
    parser.add_argument('--output-dir', default=None,
                        help='Directory containing Python output files (default: ../output_files)')
    parser.add_argument('--val-dir', default=None,
                        help='Directory containing validation files (default: output_dir/validation_files)')
    parser.add_argument('--tol', type=float, default=0.001,
                        help='Tolerance threshold (default: 0.001)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip matplotlib plots')
    parser.add_argument('--site', default='CHATS7_2007-05',
                        help='Site prefix for output filenames (default: CHATS7_2007-05)')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else here.parent / 'output_files'
    val_dir = Path(args.val_dir) if args.val_dir else out_dir / 'validation_files'
    plot_dir = out_dir / 'comparison_plots'
    if not args.no_plots:
        plot_dir.mkdir(exist_ok=True)

    site = args.site
    tol = args.tol

    print(f"\nCLM-ml_v2 Output Comparison")
    print(f"  Python outputs : {out_dir}")
    print(f"  Validation     : {val_dir}")
    print(f"  Tolerance      : {tol}")
    print(f"  Site prefix    : {site}")

    def py_path(tag):
        return out_dir / f'{site}_{tag}'
    def ft_path(tag):
        return val_dir / f'{site}_{tag}'

    all_stats = {}

    # flux.out
    all_stats['flux'] = compare_file(
        'flux.out', py_path('flux.out'), ft_path('flux.out'),
        FLUX_COLS, tol, plot_dir, time_col=0,
    )

    # aux.out (no time column, col 0 is btran)
    all_stats['aux'] = compare_file(
        'aux.out', py_path('aux.out'), ft_path('aux.out'),
        AUX_COLS, tol, plot_dir, time_col=None,
    )

    # fsun.out (no time column)
    all_stats['fsun'] = compare_file(
        'fsun.out', py_path('fsun.out'), ft_path('fsun.out'),
        FSUN_COLS, tol, plot_dir, time_col=None,
    )

    # soiltemp.out
    all_stats['soiltemp'] = compare_file(
        'soiltemp.out', py_path('soiltemp.out'), ft_path('soiltemp.out'),
        SOILTEMP_COLS, tol, plot_dir, time_col=0,
        skip_plot_cols={0, *range(1, 21, 2)},  # skip time and z columns
    )

    # profile.out (special handler)
    if not args.no_plots:
        all_stats['profile'] = compare_profile(
            py_path('profile.out'), ft_path('profile.out'), tol, plot_dir,
        )
    else:
        all_stats['profile'] = compare_profile(
            py_path('profile.out'), ft_path('profile.out'), tol, plot_dir,
        )

    # fluxprofile.out (special handler)
    all_stats['fluxprofile'] = compare_fluxprofile(
        py_path('fluxprofile.out'), ft_path('fluxprofile.out'), tol, plot_dir,
    )

    # --- Overall summary ---
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY  (tolerance = {tol})")
    print(f"{'='*60}")
    print(f"  {'File':<20} {'Total vars':>10} {'Vars with diffs':>15}")
    print(f"  {'-'*50}")
    grand_total = 0
    grand_bad = 0
    for tag, stats in all_stats.items():
        n_vars = len(stats)
        n_bad = sum(1 for s in stats if s.get('n_exceed', 0) > 0)
        grand_total += n_vars
        grand_bad += n_bad
        flag = ' *' if n_bad > 0 else ''
        print(f"  {tag+'.out':<20} {n_vars:>10} {n_bad:>15}{flag}")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<20} {grand_total:>10} {grand_bad:>15}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

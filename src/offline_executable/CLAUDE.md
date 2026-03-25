# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Python/JAX translation of **CLM-ml_v2** — a multilayer canopy land surface model originally written in Fortran. The Python model must produce numerically equivalent output to the Fortran reference run (validation files in `output_files/validation_files/`).

## Running the Model

From the project root (`CLM-ml_v2_Python/`):

```bash
# Install (once)
pip install -e .

# Run with a namelist file (mirrors Fortran: ./prgm.exe < nl.CHATS7.05.2007)
cd offline_executable
python main.py nl.CHATS7.05.2007
# or
python main.py < nl.CHATS7.05.2007
```

Output files are written to `output_files/` by default (set by `dirout` in the namelist).

## Output Files

Six ASCII files per run, named `{site}_{year}-{month}_{tag}.out`:

| Tag | Contents |
|-----|----------|
| `flux` | Per-timestep surface energy/water fluxes (18 columns) |
| `aux` | Auxiliary diagnostics |
| `profile` | Vertical canopy profiles |
| `fsun` | Sunlit/shaded fraction profiles |
| `fluxprofile` | Vertical flux profiles |
| `soiltemp` | Soil temperature profiles |

Validation (trusted Fortran output): `output_files/validation_files/`

## Architecture

The model mirrors the Fortran module hierarchy:

```
offline_executable/main.py          — Entry point (replaces prgm.exe)
clm_src_cpl/lnd_comp_nuopc.py       — InitializeRealize, ModelAdvance
clm_src_main/clm_initializeMod.py   — Two-phase init (initialize1/initialize2)
clm_src_main/clm_driver.py          — Per-timestep physics orchestration
offline_driver/CLMml_driver.py      — Tower-site driver (init_acclim, TowerVeg, SoilInit, output)
offline_driver/TowerMetMod.py       — Reads tower meteorology from netCDF
offline_driver/clmDataMod.py        — Reads CLM history file data
multilayer_canopy/MLCanopyFluxesMod.py  — Top-level canopy physics loop
multilayer_canopy/MLCanopyTurbulenceMod.py  — Turbulence closure
multilayer_canopy/MLLeafFluxesMod.py        — Leaf-level fluxes
multilayer_canopy/MLLeafPhotosynthesisMod.py — Photosynthesis
multilayer_canopy/MLSolarRadiationMod.py    — Solar radiation transfer
multilayer_canopy/MLRungeKuttaMod.py        — Runge-Kutta time integration
multilayer_canopy/MLFluxProfileSolutionMod.py — Flux-profile solution
clm_src_biogeophys/                  — Soil state, surface resistance, albedo, water, temperature
```

**State pattern**: All physics state is held in immutable NamedTuples (defined in `*Type.py` files). Functions return updated instances via `._replace(field=value)`. Global module-level instances live in `clm_src_main/clm_instMod.py`.

**JAX requirement**: `jax.config.update("jax_enable_x64", True)` is set in `main.py` before any array creation — this is essential for numerical parity with Fortran's `real(r8)` (64-bit).

## Namelist Parameters (nl.CHATS7.05.2007)

Key parameters controlling the run:
- `tower_name`, `start_ymd`, `stop_option`/`stop_n` — site and time range
- `fin_tower` — tower meteorology netCDF (relative paths use `../` from `offline_executable/`)
- `fin_clm` — CLM history file for soil initialization
- `fin_soil_adjust` — optional soil moisture correction
- `met_type=3` — enables 3-point temporal interpolation of met forcing
- `clm_phys='CLM5_0'` — physics version flag
- `pftcon_val=1` — PFT parameter set selector

## Debugging Numerical Differences

The primary debugging target is parity between `output_files/` (Python run) and `output_files/validation_files/` (Fortran reference). Key physics modules to check:

1. **`MLCanopyFluxesMod.py`** — main physics loop; uses Runge-Kutta integration with `nrk` substeps
2. **`MLCanopyTurbulenceMod.py`** — turbulence closure; contains `LookupPsihatINI` lookup table
3. **`MLLeafPhotosynthesisMod.py`** — photosynthesis; sensitive to temperature acclimation
4. **`MLFluxProfileSolutionMod.py`** — flux-profile inversion
5. **`offline_driver/CLMml_driver.py`** — `output()` function controls what/how values are written

When comparing outputs, the `flux` file is the primary diagnostic (surface-level fluxes). Column ordering and unit conversions must match the Fortran `output` subroutine exactly.

`debug_physics.py` (in `offline_executable/`) is a scratchpad for targeted physics debugging.

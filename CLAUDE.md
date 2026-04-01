# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CLM-ML-JAX** is a JAX/Python translation of the Fortran Community Land Model (CLM) with Multi-Layer Canopy physics. Every module directly mirrors a Fortran source file; structural decisions follow Fortran conventions rather than idiomatic Python.

Reference Fortran implementation: [CLM-ML_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS)

## Commands

```bash
# Install (one-time)
pip install -e .

# Run a simulation
clm-ml-offline < offline_executable/nl.CHATS7.05.2007
clm-ml-offline input_files/nl.CHATS7.1day
python -m offline_executable.main input_files/nl.CHATS7.1day

# Debug a single timestep
python src/offline_executable/debug_physics.py

# Run all tests
pytest

# Run a specific test module
pytest tests/multilayer_canopy/

# Run only fast tests
pytest -m "not slow"

# Run with coverage
pytest --cov=src
```

## Architecture

### Package Layout (Fortran → Python)

| Directory | Fortran Equivalent | Role |
|---|---|---|
| `clm_share/` | `csm_share/` | Precision kinds (`r8`/`float64`), orbital params |
| `clm_src_biogeophys/` | `CLM/src/biogeophys/` | State container types (energy, water, temperature, albedo) |
| `clm_src_cpl/` | `CLM/src/cpl/` | `InitializeRealize` / `ModelAdvance` coupling interface |
| `clm_src_main/` | `CLM/src/main/` | Global singletons, decomp, var-constants, driver |
| `clm_src_utils/` | `CLM/src/utils/` | Time manager, orbital variables |
| `multilayer_canopy/` | `CLMml/src/` | All multilayer canopy physics (28 modules) |
| `offline_driver/` | `offline_driver/` | Tower-site driver, namelist control, met/veg ingest |
| `offline_executable/` | `offline_executable/` | Entry point (`main.py` ≡ Fortran `prgm.exe`) |
| `utils/` | N/A | GPU/JAX configuration helpers |

### Data Flow

```
main.py → read_namelist() → controlMod.control()
        → InitializeRealize(bounds)   # initialize1 + initialize2
        → loop: ModelAdvance(bounds, time_indx, fin1, fin2)
                 └→ clm_drv() → MLCanopyFluxes()
                                  └→ Runge-Kutta sub-stepping
```

All model state lives in **module-level singletons** in `clm_src_main/clm_instMod.py` (`atm2lnd_inst`, `mlcanopy_inst`, etc.), initialized to `None` and populated by `clm_instInit`.

### Main Physics Loop (`multilayer_canopy/MLCanopyFluxesMod.py`)

- Runge-Kutta time integration (configurable order via `runge_kutta_type`)
- Solar radiation transfer (Norman or two-stream)
- Leaf photosynthesis (Medlyn/Ball-Berry/WUE stomatal models)
- Turbulence closure (Harman-Finnigan roughness sublayer)
- Soil temperature and water fluxes

### Output Files

Six ASCII files per run, written to `output_files/` (site/date-stamped):
`_flux.out`, `_aux.out`, `_profile.out`, `_fsun.out`, `_fluxprofile.out`, `_soiltemp.out`

Validation files in `src/output_files/validation_files/` for numerical comparison with Fortran reference runs.

## Critical Conventions

### Fortran → Python Mapping

- **Fortran `USE` statements** → `from module import name  # noqa: F401`. Imports carry `# noqa: F401` because they mirror Fortran public interfaces, not necessarily local usage.
- **Fortran module-level variables** → Python module globals, mutated in-place (e.g., `clm_time_manager.dtstep = ...`).
- **Fortran `REAL(r8)`** → `jnp.float64`; **always** enable 64-bit before any JAX array creation: `jax.config.update("jax_enable_x64", True)` (done in `main.py` line 19 — must come first).
- **Fortran 1-based arrays** → Python lists/arrays of length `n+1` with index 0 unused (data starts at index 1).
- **Docstrings** must include the original Fortran module name and Fortran line-number ranges, e.g. `Mirrors Fortran subroutine ``control`` (lines 22-105)`.

### Decomposition (`bounds_type`)

`bounds_type` (NamedTuple in `clm_src_main/decompMod.py`) holds `begg/endg/begl/endl/begc/endc/begp/endp` — always all `1` in standalone (single gridcell/landunit/column/patch) mode.

### State Pattern

Physics state is held in immutable NamedTuples defined in `*Type.py` files. Functional updates use `._replace(field=value)`. Global singletons registered in `clm_src_main/clm_instMod.py`.

### Physics Configuration (`multilayer_canopy/MLclm_varctl.py`)

Change these module globals directly — no config object:
- `gs_type` — stomatal conductance model (0=Medlyn, 1=Ball-Berry, 2=WUE)
- `flux_profile_type` — flux-profile solver (-1=dataset, 0=well-mixed, 1=implicit)
- `runge_kutta_type` — time integration order (10=Euler, 21=2nd-order, 41=4th-order); `nrk = runge_kutta_type // 10`
- `dtime_ml` — sub-step interval (s); must divide CLM timestep evenly

### Namelist Format

```fortran
&clmML_inparm
  tower_name  = 'CHATS7'
  start_ymd   = 20070501
  stop_option = 'ndays'
  stop_n      = 1
  fin_tower   = '../input_files/tower-forcing/CHATS7/2007-05.nc'
  fin_clm     = '../input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1.*.nc'
  clm_phys    = 'CLM5_0'
  dirout      = '../output_files/'
/
```

Paths beginning with `../` are resolved relative to the project root by `main.py:_resolve_path()`.

## Key Files for New Features

- **New physics switch** → `multilayer_canopy/MLclm_varctl.py`
- **New state variable** → appropriate `*Type.py` in `clm_src_biogeophys/` + register singleton in `clm_src_main/clm_instMod.py`
- **New tower site** → `offline_driver/TowerDataMod.py` (increment `ntower`, extend all arrays at 1-based index)
- **RSL psihat look-up table path** → `clm_src_main/clm_varctl.py:rslfile`
- **New output variable** → `clm_src_main/histFileMod.py`

## Tower Sites

15 AmeriFlux + CHATS sites defined in `offline_driver/TowerDataMod.py`. `tower_num` is the 1-based index; matched via `tower_id[i] == tower_name`. Currently primary test site: `CHATS7` (index 14).

## Testing

Tests in `tests/` mirror source structure. `conftest.py` enables 64-bit JAX, initializes CLM parameters, and provides shared fixtures including `bounds` and all state type instances.

Test markers: `slow`, `unit`, `integration`. Use `offline_executable/debug_physics.py` as a template for isolating and inspecting individual physics calls.

## Commit and Push Policy

Commit and push after every meaningful unit of work. This creates a recoverable history if something goes wrong, makes progress visible, and prevents work from being lost if a compute allocation runs out mid-session.

A "meaningful unit" includes: a passing test, a working physics module, a validated output variable, a completed refactor, or any state you would not want to redo. When in doubt, commit.

```bash
git add <specific files>
git commit -m "<short description>"
git push
```

Use descriptive commit messages. Prefix with the affected module or feature (e.g. `MLCanopyFluxes: fix stomatal conductance under low PAR`).

## Progress Tracking (CHANGELOG.md)

Maintain a `CHANGELOG.md` at the project root to preserve cross-session context. Update it at the end of any session that makes meaningful progress — or immediately when a dead end is identified.

**What to track:**

- **Current status** — what is working, what is broken, what is in progress
- **Completed tasks** — what was done and when (date each entry)
- **Failed approaches** — *critical*: what was tried, why it didn't work, and what was switched to instead. Without this, future sessions will re-attempt the same dead ends.
- **Accuracy tables** — numerical comparisons against Fortran reference at key checkpoints (e.g. RMSE per output variable)
- **Known limitations** — edge cases, missing physics, numerical issues

**Entry format:**

```markdown
## YYYY-MM-DD — <short title>

**Status:** <one-line summary of where things stand>

**Completed:**
- <task>

**Failed approaches:**
- Tried <X> for <reason>, but <what went wrong>. Switched to <Y>.

**Accuracy (CHATS7 2007-05):**
| Variable | RMSE | Notes |
|---|---|---|
| H | 12.3 W/m² | ... |

**Known limitations:**
- <limitation>
```

Keep entries in reverse-chronological order (newest first). Do not delete old entries — they are the record of what has been tried.

# Contributing to CLM-ML-JAX

Thank you for your interest in contributing! This guide is aimed at atmospheric and land-surface scientists who want to add physics, fix bugs, or improve documentation.

## Development Setup

```bash
# Clone with submodules (Fortran reference code lives in CLM-ml_v1/)
git clone --recurse-submodules https://github.com/AyaLahlou/clm-ml-jax.git
cd clm-ml-jax

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify the install
clm-ml-offline --help
pytest tests/fortran_validation/ -m "not slow" -q
```

## Repository Layout

```
src/
  multilayer_canopy/    # Within-canopy physics  ← most contributions go here
  clm_src_biogeophys/  # Surface/soil/snow physics
  clm_src_main/        # State containers, global parameters
  offline_driver/       # Tower-site forcing and site definitions
tests/
  fortran_validation/  # 160+ tests vs. Fortran golden values (REL_TOL=1e-9)
  fortran_validation/golden_IO/  # Golden JSON reference data
```

## Adding a Physics Module

1. **Mirror the Fortran structure.** Create `src/multilayer_canopy/MyPhysicsMod.py`.
2. **Add a module docstring** referencing the Fortran source:
   ```python
   """
   JAX translation of Fortran module MyPhysics.
   Mirrors Fortran subroutine ``my_physics_sub`` (lines 45-120 of
   CLM-ml_v1/multilayer_canopy/MyPhysicsMod.F90).

   Key differences from Fortran:
   - DO loops → jax.vmap over layer indices
   - INTENT(INOUT) arguments → function returns new state via ._replace()
   """
   ```
3. **State management**: use NamedTuples; never mutate in-place; use `._replace()`.
4. **Array indexing**: Fortran is 1-based; Python is 0-based. Snow layers go in
   `array[0:nlevsno]`; soil layers in `array[nlevsno:]`.
5. **64-bit precision**: JAX defaults to 32-bit. Ensure `jax.config.update("jax_enable_x64", True)` is called before any JAX operations (done automatically in test fixtures).

## Writing Tests

Test files mirror the source tree:
`src/multilayer_canopy/MyPhysicsMod.py` → `tests/multilayer_canopy/test_MyPhysicsMod.py`

For Fortran validation tests, capture golden I/O from the Fortran build and add a JSON file to `tests/fortran_validation/golden_IO/`, then add a parametrized case in `tests/fortran_validation/test_golden_jax.py`.

Tolerances for validation: `REL_TOL = 1e-9`, `ABS_TOL = 1e-15`.

Run only fast tests during development:
```bash
pytest tests/fortran_validation/ -m "not slow" -q
```

## Code Style

```bash
black --line-length 100 src/ tests/   # auto-format
ruff check src/ tests/                # linting
```

Both are enforced in CI on every pull request.

## Submitting a Pull Request

1. Fork the repository and create a feature branch (`git checkout -b feat/my-physics`).
2. Make your changes with tests.
3. Run the test suite locally (`pytest tests/fortran_validation/ -m "not slow" -q`).
4. Open a PR against `main` with a description of what changed and why.

CI will automatically run lint checks and the fortran validation suite. Please ensure both pass before requesting review.

## Reporting Bugs

Open an issue at https://github.com/AyaLahlou/clm-ml-jax/issues with:
- A minimal reproduction script
- The Python and JAX versions (`python --version`, `python -c "import jax; print(jax.__version__)"`)
- The expected vs. actual output

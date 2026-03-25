"""
JAX translation of clm_varctl Fortran module.

Run control variables for the CLM land surface model.
Contains global configuration constants used throughout the model,
including the log file unit number and the path to the RSL psihat
look-up tables.

Original Fortran module: clm_varctl
Fortran lines 1-17
"""

import os as _os

# "stdout" log file unit number — Fortran line 13
# Used throughout the model as the target for diagnostic output.
iulog: int = 6

# RSL psihat look-up table file path — Fortran line 14
# Path to the roughness sublayer psihat NetCDF look-up tables used by
# MLCanopyTurbulenceMod.LookupPsihatINI.
# Resolve relative to the Python package root (two levels up from this file).
_pkg_root: str = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
rslfile: str = _os.path.join(_pkg_root, 'rsl_lookup_tables', 'psihat.nc')
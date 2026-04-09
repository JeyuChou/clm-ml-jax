#!/bin/bash
# ============================================================
# Portable Fortran CLM-ml-v2 timing script
#
# PURPOSE: Compile and time a 31-day (May 2007, CHATS7) run of
#          the Fortran CLM-ml-v2 reference model on any HPC.
#
# USAGE:
#   1. Copy the entire clm-ml-fortran/ directory to your HPC.
#      The path does NOT need to match the original.
#
#   2. Set the variables in the "CONFIGURE ME" section below.
#
#   3. Submit:   sbatch run_fortran_timing_portable.sh
#      Or run directly: bash run_fortran_timing_portable.sh
#
# WHAT IT NEEDS:
#   - gfortran (or nvfortran — see COMPILER section)
#   - NetCDF-Fortran library (libnetcdff) + NetCDF-C (libnetcdf)
#   - BLAS (libblas or libopenblas)
#   - Input files: tower-forcing and CLM5 IC files (in input_files/)
#
# OUTPUT:
#   - Wall-clock time for the 31-day simulation
#   - Output files written to ../output_files/ (relative to executable)
# ============================================================

#SBATCH --job-name=fortran-timing
#SBATCH --output=fortran_timing_%j.out
#SBATCH --error=fortran_timing_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Adjust --partition, --account, --constraint to match your cluster:
##SBATCH --partition=short
##SBATCH --account=YOUR_ACCOUNT

# ============================================================
# CONFIGURE ME — adjust these paths/modules for your HPC
# ============================================================

# Path to the clm-ml-fortran repository on this machine
# (the directory that contains clm_share/, offline_executable/, etc.)
CLM_FORTRAN_ROOT="${HOME}/clm-ml-fortran"

# NetCDF-Fortran install prefix (directory containing lib/ and include/)
# Find it with: nc-config --prefix  OR  nf-config --prefix
NETCDF_F_PREFIX=""   # e.g. /usr/local  or  /opt/netcdf-fortran-4.6.2

# NetCDF-C install prefix (needed because netcdff links against netcdf-c)
# Find it with: nc-config --prefix
NETCDF_C_PREFIX=""   # e.g. /usr/local  or  /opt/netcdf-c-4.9.3

# Optional: extra module load commands for your cluster, e.g.:
#   module load gcc netcdf-fortran openblas
# Leave empty if not needed.
EXTRA_MODULES=""

# ============================================================
# AUTO-DETECT (tries common locations if prefix is unset)
# ============================================================

_try_find_netcdf() {
    # Returns first directory that has libnetcdff.* under it
    for candidate in \
        /usr/local /usr /opt/homebrew \
        /opt/netcdf-fortran* /burg/opt/netcdf-fortran* \
        $(nf-config --prefix 2>/dev/null) \
        $(nc-config --prefix 2>/dev/null); do
        if ls "${candidate}/lib/libnetcdff"* &>/dev/null 2>&1; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

_try_find_netcdf_c() {
    for candidate in \
        /usr/local /usr /opt/homebrew \
        /opt/netcdf-c* /burg/opt/netcdf-c* \
        $(nc-config --prefix 2>/dev/null); do
        if ls "${candidate}/lib/libnetcdf."* &>/dev/null 2>&1; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

# ============================================================
# SETUP
# ============================================================

set -e

# Load modules if specified
if [ -n "$EXTRA_MODULES" ]; then
    echo "Loading modules: $EXTRA_MODULES"
    eval "$EXTRA_MODULES"
fi

# Resolve NetCDF paths
if [ -z "$NETCDF_F_PREFIX" ]; then
    echo "NETCDF_F_PREFIX not set, attempting auto-detection..."
    NETCDF_F_PREFIX=$(_try_find_netcdf) || {
        echo "ERROR: Could not find NetCDF-Fortran. Set NETCDF_F_PREFIX manually."
        exit 1
    }
    echo "  Found NetCDF-Fortran at: $NETCDF_F_PREFIX"
fi

if [ -z "$NETCDF_C_PREFIX" ]; then
    echo "NETCDF_C_PREFIX not set, attempting auto-detection..."
    NETCDF_C_PREFIX=$(_try_find_netcdf_c) || {
        echo "WARNING: Could not find NetCDF-C. Will try without explicit -L path."
        NETCDF_C_PREFIX=""
    }
    echo "  Found NetCDF-C at: $NETCDF_C_PREFIX"
fi

# Resolve Fortran root
if [ ! -d "$CLM_FORTRAN_ROOT" ]; then
    echo "ERROR: CLM_FORTRAN_ROOT not found: $CLM_FORTRAN_ROOT"
    echo "  Set CLM_FORTRAN_ROOT at the top of this script."
    exit 1
fi

EXEC_DIR="$CLM_FORTRAN_ROOT/offline_executable"
cd "$EXEC_DIR"
echo "Working directory: $(pwd)"

# ============================================================
# ENVIRONMENT REPORT
# ============================================================

echo ""
echo "=== Environment ==="
echo "Hostname: $(hostname)"
echo "Date:     $(date)"
echo "CLM_FORTRAN_ROOT: $CLM_FORTRAN_ROOT"
echo "NETCDF_F: $NETCDF_F_PREFIX"
echo "NETCDF_C: $NETCDF_C_PREFIX"

# Detect compiler
if command -v gfortran &>/dev/null; then
    FC=gfortran
    FC_VER=$(gfortran --version | head -1)
    echo "Compiler: gfortran  ($FC_VER)"
elif command -v nvfortran &>/dev/null; then
    FC=nvfortran
    FC_VER=$(nvfortran --version | head -1)
    echo "Compiler: nvfortran ($FC_VER)"
else
    echo "ERROR: Neither gfortran nor nvfortran found in PATH."
    echo "  Install gfortran or add it to PATH."
    exit 1
fi

echo ""
echo "=== NetCDF libraries found ==="
ls "${NETCDF_F_PREFIX}/lib/libnetcdff"* 2>/dev/null || echo "  WARNING: libnetcdff not found at $NETCDF_F_PREFIX/lib"
[ -n "$NETCDF_C_PREFIX" ] && ls "${NETCDF_C_PREFIX}/lib/libnetcdf."* 2>/dev/null || true

# ============================================================
# COMPILATION
# ============================================================

echo ""
echo "=== Compilation ==="
echo "Cleaning old object files..."
make clean 2>/dev/null || rm -f obj/*.o *.mod 2>/dev/null || true

# Build the compiler flag string
NETCDF_FLAGS="-L${NETCDF_F_PREFIX}/lib -I${NETCDF_F_PREFIX}/include -lnetcdff"
if [ -n "$NETCDF_C_PREFIX" ] && [ "$NETCDF_C_PREFIX" != "$NETCDF_F_PREFIX" ]; then
    NETCDF_FLAGS="$NETCDF_FLAGS -L${NETCDF_C_PREFIX}/lib -lnetcdf"
else
    NETCDF_FLAGS="$NETCDF_FLAGS -lnetcdf"
fi

if [ "$FC" = "gfortran" ]; then
    CMPLR="gfortran -O2 -fno-range-check -ffree-line-length-none ${NETCDF_FLAGS} -lnetcdf -lblas -lm -L/usr/lib64"
else
    # nvfortran
    CMPLR="nvfortran -O2 -L${NETCDF_F_PREFIX}/lib -I${NETCDF_F_PREFIX}/include -lnetcdff -lnetcdf -lblas -lm -L/usr/lib64"
fi

echo "Compiler string: $CMPLR"
echo "Starting compilation at $(date)"

set +e
make "cmplr=${CMPLR}" NETCDF_PATH="${NETCDF_F_PREFIX}" 2>&1
MAKE_STATUS=$?
set -e

if [ $MAKE_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: make failed (status $MAKE_STATUS)"
    echo ""
    echo "Common fixes:"
    echo "  1. Wrong NETCDF_F_PREFIX — check with: nf-config --prefix"
    echo "  2. Missing netcdf-c — set NETCDF_C_PREFIX"
    echo "  3. Missing BLAS — try: module load openblas OR set -lopenblas instead of -lblas"
    echo "  4. Missing module — add 'module load gcc/netcdf-fortran/...' to EXTRA_MODULES"
    exit 1
fi

if [ ! -f prgm.exe ]; then
    echo "ERROR: prgm.exe not produced after successful make"
    exit 1
fi

echo "Compilation succeeded at $(date)"
ls -lh prgm.exe

# ============================================================
# VERIFY INPUT FILES
# ============================================================

echo ""
echo "=== Input file check ==="
NL="nl.CHATS7.05.2007"

if [ ! -f "$NL" ]; then
    echo "ERROR: Namelist not found: $EXEC_DIR/$NL"
    exit 1
fi

# Check the data files referenced in the namelist
FIN_TOWER=$(grep fin_tower "$NL" | sed "s/.*= *'//;s/'.*//")
FIN_CLM=$(grep fin_clm "$NL" | sed "s/.*= *'//;s/'.*//" | sed 's/\*.*//')
echo "Namelist:  $NL"
echo "fin_tower: $FIN_TOWER"
echo "fin_clm:   ${FIN_CLM}..."

# Paths in namelist are relative to offline_executable/
TOWER_FILE="$EXEC_DIR/$FIN_TOWER"
if [ ! -f "$TOWER_FILE" ]; then
    echo "WARNING: Tower forcing file not found: $TOWER_FILE"
    echo "  You may need to copy input_files/ from burg."
else
    echo "Tower forcing file: OK"
fi

mkdir -p "$CLM_FORTRAN_ROOT/output_files"

# ============================================================
# TIMING RUN
# ============================================================

echo ""
echo "=== Timing 31-day simulation (CHATS7, May 2007) ==="
echo "Namelist: $NL"
echo "Start: $(date)"

SECONDS=0

if command -v /usr/bin/time &>/dev/null; then
    /usr/bin/time -v ./prgm.exe < "$NL" 2>&1
else
    ./prgm.exe < "$NL"
fi

SIM_STATUS=$?
ELAPSED=$SECONDS

echo ""
echo "=== Timing Results ==="
echo "Simulation exit status: $SIM_STATUS"
echo "Wall-clock time: ${ELAPSED}s"
printf "Wall-clock time formatted: %d min %d sec\n" $((ELAPSED/60)) $((ELAPSED%60))
echo "End: $(date)"

if [ $SIM_STATUS -ne 0 ]; then
    echo ""
    echo "WARNING: Simulation exited with non-zero status ($SIM_STATUS)"
    echo "Check output above for error messages."
fi

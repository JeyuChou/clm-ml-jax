#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=fortran-timing
#SBATCH --output=logs/%j_fortran_timing.out
#SBATCH --error=logs/%j_fortran_timing.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short

# Experiment 5: time the Fortran CLM-ml-v2 reference model for a 31-day simulation
# No GPU needed — Fortran runs on CPU only

# ── Use gfortran + known NetCDF path (nvfortran not in PATH on this cluster) ──
# gfortran is available at /usr/bin/gfortran (GCC 8.5.0, no module needed)
# NetCDF-Fortran 4.6.2 is at /burg/opt/netcdf-fortran-4.6.2 (confirmed on disk)
# NetCDF-C required by netcdf-fortran: /burg/opt/netcdf-c-4.9.3
export NETCDF_PATH=/burg/opt/netcdf-fortran-4.6.2
export NETCDF_C_PATH=/burg/opt/netcdf-c-4.9.3

echo "=== Environment ==="
which gfortran
gfortran --version | head -1
echo "NETCDF_PATH = $NETCDF_PATH"
ls $NETCDF_PATH/lib/libnetcdff* 2>/dev/null | head -3

# ── Compile the Fortran model ─────────────────────────────────────────────────
FORTRAN_DIR="/burg-archive/home/al4385/clm-ml-jax/clm-ml-fortran/offline_executable"
cd "$FORTRAN_DIR"

echo ""
echo "=== Compilation ==="
echo "Working directory: $(pwd)"

# Clean any old objects
echo "Cleaning old object files..."
make clean 2>/dev/null || rm -f obj/*.o *.mod

# Patch Makefile: override cmplr to use gfortran with equivalent flags
# nvfortran flags → gfortran equivalents:
#   -C            → -fcheck=bounds (optional, slow; omit for timing run)
#   -Minform=inform → (no equivalent, omit)
#   -Ktrap=fp     → -ffpe-trap=invalid,overflow,zero (omit for timing)
#   -O2           → -O2
# Also add NetCDF-C library path (netcdf-fortran depends on netcdf-c)
GFORTRAN_CMPLR="gfortran -O2 -fno-range-check -L${NETCDF_PATH}/lib -I${NETCDF_PATH}/include -L${NETCDF_C_PATH}/lib -lnetcdff -lnetcdf -lblas -lm -L/usr/lib64"

echo "Starting compilation at $(date)"
make "cmplr=${GFORTRAN_CMPLR}" NETCDF_PATH="$NETCDF_PATH" 2>&1
MAKE_STATUS=$?

if [ $MAKE_STATUS -ne 0 ]; then
    echo "ERROR: make failed with status $MAKE_STATUS"
    exit 1
fi

if [ ! -f prgm.exe ]; then
    echo "ERROR: prgm.exe not produced after make"
    exit 1
fi

echo "Compilation succeeded at $(date)"
ls -lh prgm.exe

# ── Run the 31-day simulation and time it ─────────────────────────────────────
echo ""
echo "=== Timing 31-day simulation ==="
echo "Namelist: nl.CHATS7.05.2007"
echo "Start: $(date)"

# Use bash built-in timing for wall-clock time
SECONDS=0
/usr/bin/time -v ./prgm.exe < nl.CHATS7.05.2007
SIM_STATUS=$?
ELAPSED=$SECONDS

echo ""
echo "=== Timing Results ==="
echo "Simulation exit status: $SIM_STATUS"
echo "Wall-clock time (bash SECONDS): ${ELAPSED}s"
printf "Wall-clock time formatted: %d min %d sec\n" $((ELAPSED/60)) $((ELAPSED%60))
echo "End: $(date)"

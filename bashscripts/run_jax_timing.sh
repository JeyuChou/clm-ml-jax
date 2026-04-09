#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=jax-timing
#SBATCH --output=logs/%j_jax_timing.out
#SBATCH --error=logs/%j_jax_timing.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=hpc_test

# ── Load CUDA toolkit ─────────────────────────────────────────────────────────
module load cuda12.8/toolkit/12.8.61

# ── Activate venv ─────────────────────────────────────────────────────────────
source /burg-archive/home/al4385/clm-ml-jax/.venv/bin/activate

# ── Expose JAX's bundled CUDA libraries ──────────────────────────────────────
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

# ── Move to project root ──────────────────────────────────────────────────────
cd /burg-archive/home/al4385/clm-ml-jax

# ── Sanity check ─────────────────────────────────────────────────────────────
nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

# ── Experiment 5: Run timing script (GPU + CPU, with warm-up) ────────────────
echo "=== Starting Experiment 5: JAX timing (GPU + CPU) ==="
cd src
time python ../diags/time_jax_run.py --backend both
echo "=== Experiment 5 complete ==="

# ── Bonus: raw timing for GPU-only 31-day run (no Python wrapper overhead) ───
echo ""
echo "=== Raw GPU 31-day run (time python -m ...) ==="
time python -m offline_executable.main offline_executable/nl.CHATS7.05.2007
echo "=== Raw GPU run complete ==="

# ── CPU baseline (raw) ───────────────────────────────────────────────────────
echo ""
echo "=== Raw CPU 31-day run ==="
JAX_PLATFORM_NAME=cpu time python -m offline_executable.main offline_executable/nl.CHATS7.05.2007
echo "=== Raw CPU run complete ==="

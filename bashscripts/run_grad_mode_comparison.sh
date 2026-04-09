#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=grad-mode-cmp
#SBATCH --output=logs/%j_grad_mode_comparison.out
#SBATCH --error=logs/%j_grad_mode_comparison.err
#SBATCH --time=02:00:00
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

# ── Sanity check GPU ──────────────────────────────────────────────────────────
nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices()); print('backend:', jax.default_backend())"

# ── Run gradient mode comparison ─────────────────────────────────────────────
echo "=== Starting grad_mode_comparison.py ==="
cd src
python ../diags/grad_mode_comparison.py
echo "=== grad_mode_comparison.py complete ==="

#!/bin/bash
#SBATCH --account=glab        
#SBATCH --job-name=clm-ml-jax
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=05:00:00          # adjust to your run length
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --qos=hpc_test  


# ── Load modules (adjust to what your HPC provides) ──────────────────────────
module load anaconda
module load shared
module load cuda12.8/toolkit/12.8.61

# ── Activate environment ──────────────────────────────────────────────────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clm-ml-jax
pip install --quiet netCDF4

# ── Expose JAX's bundled CUDA libraries ──────────────────────────────────────
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia" ]; then
    export LD_LIBRARY_PATH=$(find $SITE_PKGS/nvidia -name "*.so*" -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
fi

# ── Move to project directory ─────────────────────────────────────────────────
cd /burg-archive/home/al4385/clm-ml-jax

# ── Verify JAX sees the GPU (optional sanity check) ──────────────────────────
nvidia-smi
which python
python --version
python -c "import jax, jaxlib; print('jax file:', jax.__file__); print('jaxlib file:', jaxlib.__file__)"
python -c "import sys; print('sys.path:', sys.path)"
python -c "import jax; print('JAX devices:', jax.devices()); print('JAX backend:', jax.default_backend())"

# ── Run the model ─────────────────────────────────────────────────────────────
python -m offline_executable.test_grad
#python -m offline_executable.main src/offline_executable/nl.CHATS7.1day

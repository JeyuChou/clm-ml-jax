#!/bin/bash
#SBATCH --job-name=clm-ml-jax
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00          # adjust to your run length
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --partition=gpu          # change to your HPC's GPU partition name

# ── Load modules (adjust to what your HPC provides) ──────────────────────────
module purge
module load cuda/12.x            # match the cuda version in environment.yml
module load conda                # or: module load anaconda3

# ── Activate environment ──────────────────────────────────────────────────────
conda activate clm-ml-jax

# ── Move to project directory ─────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR

# ── Verify JAX sees the GPU (optional sanity check) ──────────────────────────
python -c "import jax; print('JAX devices:', jax.devices())"

# ── Run the model ─────────────────────────────────────────────────────────────
python -m offline_executable.main input_files/nl.CHATS7.1day

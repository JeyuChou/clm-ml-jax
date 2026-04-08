#!/bin/bash
#SBATCH --account=glab        
#SBATCH --job-name=claude-agent
#SBATCH --output=logs/agent_%j.out
#SBATCH --error=logs/agent_%j.err
#SBATCH --time=24:00:00          # adjust to your run length
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1             # request 1 GPU



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

export TERM=xterm-256color
tmux new-session -d -s claude "claude --dangerously-skip-permissions ; exec bash"
tmux wait-for claude


#### Optional: Attach to the tmux session to interact with the agent
#srun --jobid=7211992 --overlap --pty tmux attach -t claude

#### how to detach form claude session 
# press ctrl+b then d 



####
#example prompts to test the agent
# /ralph-loop:ralph-loop “Please keep working on the task until the success criterion 
#of 0.1% accuracy across the entire parameter range is achieved.”
# --max-iterations 20 --completion-promise “DONE”
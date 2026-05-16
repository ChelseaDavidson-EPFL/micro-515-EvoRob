#!/bin/bash
#SBATCH --job-name=micro515_evo_training
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu  

mkdir -p logs

module load python/3.10          # ← adjust to your cluster's Python module
source "$HOME/venvs/micro515/bin/activate"   # ← adjust to your venv path

PROJECT_DIR="$HOME/micro515_final"           # ← adjust to your project path
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }

export MUJOCO_GL=egl
export DISPLAY=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting training at $(date)"
python final_project_train.py
echo "Done at $(date)"
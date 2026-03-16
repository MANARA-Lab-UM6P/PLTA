#!/bin/bash
# =============================================================================
# run.sh — Launch the PLTA experiment on an HPC cluster (SLURM)
# =============================================================================
# Edit the variables below to match your cluster configuration, then submit:
#   sbatch run.sh

#SBATCH --job-name=PLTA
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --ntasks=201          # 100 task-nodes + 100 worker-nodes + 1 coordinator
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=compute   # adjust to your cluster partition

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10        # or your cluster's Python module
module load openmpi/4.1        # or your MPI module

# Install dependencies into user space if not already present
pip install --user -r requirements.txt --quiet

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
mpirun -n 201 python main.py

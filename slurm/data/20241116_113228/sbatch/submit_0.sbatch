#!/bin/bash
#SBATCH --job-name=20241116_113228
#SBATCH --output=slurm/data/20241116_113228/logs/%a.out
#SBATCH --error=slurm/data/20241116_113228/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=8,20,32,44,56,68,80,92
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241116_113228/config.json" "slurm/data/20241116_113228/output" 129600
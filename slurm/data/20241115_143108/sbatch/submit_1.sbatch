#!/bin/bash
#SBATCH --job-name=20241115_143108
#SBATCH --output=slurm/data/20241115_143108/logs/%a.out
#SBATCH --error=slurm/data/20241115_143108/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=22,45,68,91,114
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241115_143108/config.json" "slurm/data/20241115_143108/output" 129600
#!/bin/bash
#SBATCH --job-name=20241104_203226
#SBATCH --output=slurm/data/20241104_203226/logs/%a.out
#SBATCH --error=slurm/data/20241104_203226/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=0-65
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241104_203226/config.json" "slurm/data/20241104_203226/output"
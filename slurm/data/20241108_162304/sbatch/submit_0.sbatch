#!/bin/bash
#SBATCH --job-name=20241108_162304
#SBATCH --output=slurm/data/20241108_162304/logs/%a.out
#SBATCH --error=slurm/data/20241108_162304/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=0,1,2,3,4
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241108_162304/config.json" "slurm/data/20241108_162304/output" 3600
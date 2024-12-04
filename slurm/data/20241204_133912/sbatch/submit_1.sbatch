#!/bin/bash
#SBATCH --job-name=20241204_133912
#SBATCH --output=slurm/data/20241204_133912/logs/%a.out
#SBATCH --error=slurm/data/20241204_133912/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=3,5,7,11,16,17,19,24,25,27,175,177,179,183,188,189,191,196,197,199,347,349,351,355,360,361,363,368,369,371
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241204_133912/config.json" "slurm/data/20241204_133912/output" 129600
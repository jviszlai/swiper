#!/bin/bash
#SBATCH --job-name=20241116_113228
#SBATCH --output=slurm/data/20241116_113228/logs/%a.out
#SBATCH --error=slurm/data/20241116_113228/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,63,64,65,66,67,69,70,71,72,73,74,75,76,77,78,79,81,82,83,84,85,86,87,88,89,90,91,93,94,95
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241116_113228/config.json" "slurm/data/20241116_113228/output" 129600
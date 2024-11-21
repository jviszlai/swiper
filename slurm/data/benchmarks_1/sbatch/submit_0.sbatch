#!/bin/bash
#SBATCH --job-name=20241119_135256
#SBATCH --output=slurm/data/20241119_135256/logs/%a.out
#SBATCH --error=slurm/data/20241119_135256/logs/%a.out
#SBATCH --account=!!REDACTED!!
#SBATCH --partition=!!REDACTED!!
#SBATCH --array=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/!!REDACTED!!/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241119_135256/config.json" "slurm/data/20241119_135256/output" 129600
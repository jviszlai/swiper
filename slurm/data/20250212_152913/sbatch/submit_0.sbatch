#!/bin/bash
#SBATCH --job-name=20250212_152913
#SBATCH --output=slurm/data/20250212_152913/logs/%a.out
#SBATCH --error=slurm/data/20250212_152913/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=1,4,5,7,10,11,13,16,17,19,22,23,25,28,29,31,34,35,37,40,41,43,46,47,49,52,53,55,58,59,61,64,65,67,70,71,73,76,77,79,82,83,85,88,89,91,94,95,97,100,101,103,106,107,109,112,113,115,118,119,121,124,125
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250212_152913/config.json" "slurm/data/20250212_152913/output" 604800
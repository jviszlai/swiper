#!/bin/bash
#SBATCH --job-name=20250212_153333
#SBATCH --output=slurm/data/20250212_153333/logs/%a.out
#SBATCH --error=slurm/data/20250212_153333/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=4,5,10,11,16,17,22,23,28,29,34,35,40,41,46,47,52,53,58,59,64,65,70,71,76,77,82,83,88,89,94,95,100,101,106,107,112,113,118,119,124,125
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

eval "$(conda shell.bash hook)"
conda activate /home/jchadwick/envs/pySwiper
cd /scratch/viszlai/swiper

python -m slurm.run_simulation "slurm/data/20250212_153333/config.json" "slurm/data/20250212_153333/output" 604800
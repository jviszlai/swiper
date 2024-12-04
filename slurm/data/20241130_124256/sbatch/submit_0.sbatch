#!/bin/bash
#SBATCH --job-name=20241130_124256
#SBATCH --output=slurm/data/20241130_124256/logs/%a.out
#SBATCH --error=slurm/data/20241130_124256/logs/%a.out
#SBATCH --partition=fast-long
#SBATCH --array=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

eval "$(conda shell.bash hook)"
conda activate /home/viszlai/anaconda3/envs/pySwiper/

cd /home/viszlai/slurm/swiper


python slurm/run_simulation.py "slurm/data/20241130_124256/config.json" "slurm/data/20241130_124256/output" 604800
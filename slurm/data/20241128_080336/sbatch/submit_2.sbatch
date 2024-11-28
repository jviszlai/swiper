#!/bin/bash
#SBATCH --job-name=20241128_080336
#SBATCH --output=slurm/data/20241128_080336/logs/%a.out
#SBATCH --error=slurm/data/20241128_080336/logs/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=3,5,7,11,16,17,19,24,25,27,31,33,35,39,44,45,47,52,53,55,347,349,351,355,360,361,363,368,369,371,375,377,379,383,388,389,391,396,397,399,691,693,695,699,704,705,707,712,713,715,719,721,723,727,732,733,735,740,741,743,1035,1037,1039,1043,1048,1049,1051,1056,1057,1059,1207,1209,1211,1215,1220,1221,1223,1228,1229,1231
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241128_080336/config.json" "slurm/data/20241128_080336/output" 129600
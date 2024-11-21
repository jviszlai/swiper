#!/bin/bash
#SBATCH --job-name=20241118_215925
#SBATCH --output=slurm/data/20241118_215925/logs/%a.out
#SBATCH --error=slurm/data/20241118_215925/logs/%a.out
#SBATCH --account=!!REDACTED!!
#SBATCH --partition=!!REDACTED!!
#SBATCH --array=3,6,8,9,12,18,19,21,27,28,29,33,36,38,39,42,48,49,51,57,58,59,369,372,374,375,378,384,385,387,393,394,395,399,402,404,405,408,414,415,417,423,424,425,735,738,740,741,744,750,751,753,759,760,761,765,768,770,771,774,780,781,783,789,790,791,1101,1104,1106,1107,1110,1116,1117,1119,1125,1126,1127,1284,1287,1289,1290,1293,1299,1300,1302,1308,1309,1310
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000

module load python
eval "$(conda shell.bash hook)"
conda activate /project/!!REDACTED!!/projects/envs/pySwiper/

python slurm/run_simulation.py "slurm/data/20241118_215925/config.json" "slurm/data/20241118_215925/output" 129600
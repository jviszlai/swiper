import os, sys
import shutil
import json
import math
import subprocess
import datetime as dt
from functools import reduce

if __name__ == '__main__':
    time = dt.datetime.now()

    data_dir = f'slurm/data/{time.strftime("%Y%m%d_%H%M%S")}'
    config_filename = f'{data_dir}/config.json'
    sbatch_filename = f'{data_dir}/submit.sbatch'
    output_dir = f'{data_dir}/output'
    log_dir = f'{data_dir}/logs'
    benchmark_dir = f'{data_dir}/benchmarks'
    decoder_dist_filename = f'{data_dir}/decoder_dists.json'
    metadata_filename = f'{data_dir}/metadata.txt'
    os.makedirs(output_dir)
    os.makedirs(log_dir)
    os.makedirs(benchmark_dir)

    # Can make a chosen smaller list of these instead
    benchmark_files = []
    for file in os.listdir('benchmarks/cached_schedules/'):
        if file.endswith('.lss'):
            path = os.path.join('benchmarks/cached_schedules/', file)
            newpath = os.path.join(benchmark_dir, file)
            shutil.copyfile(path, newpath)
            benchmark_files.append(newpath)
    
    shutil.copyfile('benchmarks/data/decoder_dists.json', decoder_dist_filename)

    sweep_params = {
        'distance':[21],
        'speculation_latency':[1],
        'speculation_accuracy':[0, 0.99],
        'speculation_mode':['separate'],
        'scheduling_method':['sliding', 'parallel', 'aligned'],
        'max_parallel_processes':[None],
        'benchmark_file':benchmark_files,
        'decoder_dist_filename':[decoder_dist_filename],
        'rng':[0],
    }
    ordered_param_names = list(sorted(sweep_params.keys()))
    total_num_configs = reduce(lambda x,y: x*y, [len(params) for params in sweep_params.values()])

    # Write config file (each Python job will read params from this)
    configs = []
    cur_indices = [0 for _ in ordered_param_names]
    for config_idx in range(total_num_configs):
        rolled_over = [False for _ in ordered_param_names]
        config = {}
        for i,name in enumerate(ordered_param_names):
            idx = cur_indices[i]
            config[name] = sweep_params[name][idx]
            if all(rolled_over[:i]):
                cur_indices[i] += 1
            if cur_indices[i] >= len(sweep_params[name]):
                cur_indices[i] = 0
                rolled_over[i] = True
        configs.append(config)
    with open(config_filename, 'w') as f:
        json.dump(configs, f)

    with open(sbatch_filename, 'w') as f:
        f.write(f'''#!/bin/bash
#SBATCH --job-name={time.strftime("%Y%m%d_%H%M%S")}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array=0-{total_num_configs-1}
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "{config_filename}" "{output_dir}"'''
        )

    p = subprocess.Popen(f'sbatch {sbatch_filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = list(p.stdout.readlines())
    retval = p.wait()
    if retval != 0:
        print(lines)
    job_id = int(lines[-1].decode('utf-8').rstrip().split(' ')[-1])

    with open(metadata_filename, 'w') as f:
        f.write(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Job ID: {job_id}\n')
        f.write(f'Total num. tasks: {total_num_configs}\n')
        f.write(f'Params:\n')
        for name,params in sweep_params.items():
            f.write(f'    {name}: {params}\n')
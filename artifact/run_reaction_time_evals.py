
import os, sys
import shutil
import json
import math
import subprocess
import datetime as dt
import time
from functools import reduce
import numpy as np

################################################################################
# CONFIGURATION
################################################################################

dry_run = True # if True, don't actually submit jobs, just print sbatch commands

# Required configuration
slurm_account = 'CHANGE_ME' # argument to `--account` in `sbatch`
slurm_partition = 'CHANGE_ME' # argument to `--partition` in `sbatch`
path_to_python_env = 'CHANGE_ME' # argument to `conda activate` in sbatch script
path_to_swiper = 'CHANGE_ME' # path to the root of this repository

# Optional configuration
submission_delay = dt.timedelta(minutes=10)
max_tasks_per_job = 800

################################################################################

if not dry_run and any([x == 'CHANGE_ME' for x in [slurm_account, slurm_partition, path_to_python_env, path_to_swiper]]):
    raise ValueError('Please configure the variables at the top of this script.')

if __name__ == '__main__':
    cur_time = dt.datetime.now()

    # USER SETTING: maximum job duration
    max_time = dt.timedelta(hours=1)

    if max_time.days > 0:
        assert max_time.days == 1
        max_time_str = f'1-{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'
    else:
        max_time_str = f'{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'

    data_dir = f'artifact/data/{cur_time.strftime("%Y%m%d_%H%M%S")}'
    config_filename = f'{data_dir}/config.json'
    sbatch_dir = f'{data_dir}/sbatch'
    output_dir = f'{data_dir}/output'
    log_dir = f'{data_dir}/logs'
    benchmark_dir = f'{data_dir}/benchmarks'
    metadata_filename = f'{data_dir}/metadata.txt'
    os.makedirs(sbatch_dir)
    os.makedirs(output_dir)
    os.makedirs(log_dir)
    os.makedirs(benchmark_dir)

    # copy files to data dir to preserve them
    shutil.copyfile('artifact/run_reaction_time_evals.py', f'{data_dir}/run_reaction_time_evals_copy.py')
    shutil.copyfile('artifact/_simulate.py', f'{data_dir}/_simulate_copy.py')

    # Can make a chosen smaller list of these instead
    benchmark_files = []
    benchmark_names = {}
    memory_settings = None
    for file in os.listdir('artifact/data/cached_schedules/'):
        # USER SETTING: filter benchmark files if desired
        if file.endswith('.lss') and file.startswith('random_t_1000_200_0'):
            path = os.path.join('artifact/data/cached_schedules/', file)
            newpath = os.path.join(benchmark_dir, file)
            # copy files to data dir to preserve them
            shutil.copyfile(path, newpath)
            benchmark_files.append(newpath)
            benchmark_names[newpath] = file.split('.')[0]
        elif file == '.memory_settings.json':
            # USER SETTING: change values in this file to add memory for certain benchmarks
            memory_settings = json.load(open(os.path.join('artifact/data/cached_schedules/', file), 'r'))
    assert memory_settings is not None

    for name in set(benchmark_names.values()):
        if name not in memory_settings:
            memory_settings[name] = 4

    # USER SETTING: change parameter sweeps for distance, spec acc, etc.
    sweep_params = {
        'benchmark_file':benchmark_files,
        'distance':[21],
        'scheduling_method':['sliding', 'parallel', 'aligned'],
        'decoder_latency_or_dist_filename':[f'lambda volume: volume*{fac}' for fac in np.geomspace(0.1, 10, 10)],
        'speculation_mode':['separate'],
        'speculation_latency':[1],
        'speculation_accuracy':list(np.linspace(0, 1, 11)),
        'poison_policy':['successors'],
        'missed_speculation_modifier':[1.4],
        'max_parallel_processes':[None],
        'rng':[0],
        'lightweight_setting':[1],
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

    # submit a different sbatch job for each config
    configs_by_mem = {}
    for i,config in enumerate(configs):
        mem_gb = memory_settings[benchmark_names[config['benchmark_file']]]
        configs_by_mem.setdefault(mem_gb, []).append(i)

    last_submit_time = None
    job_ids = []
    submit_idx = 0
    for i,mem_gb in enumerate(sorted(configs_by_mem.keys())):
        config_indices = configs_by_mem[mem_gb]
        num_submissions = math.ceil(len(config_indices) / max_tasks_per_job) # caslake submission limit
        for j in range(num_submissions):
            if last_submit_time:
                time.sleep(max(0, int((last_submit_time + submission_delay - dt.datetime.now()).total_seconds())))
            selected_config_indices = config_indices[j*max_tasks_per_job:(j+1)*max_tasks_per_job]
            print(f'\tSubmitting {len(selected_config_indices)} / {len(configs)} jobs...')
            sbatch_filename = os.path.join(sbatch_dir, f'submit_{submit_idx}.sbatch')
            submit_idx += 1
            with open(sbatch_filename, 'w') as f:
                f.write(f'''#!/bin/bash
#SBATCH --job-name={cur_time.strftime("%Y%m%d_%H%M%S")}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.out
#SBATCH --account={slurm_account}
#SBATCH --partition={slurm_partition}
#SBATCH --array={','.join([str(x) for x in selected_config_indices])}
#SBATCH --time={max_time_str}
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu={mem_gb*1000}

module load python
eval "$(conda shell.bash hook)"
conda activate {path_to_python_env}
cd {path_to_swiper}

python -m artifact._simulate "{config_filename}" "{output_dir}" {int(max_time.total_seconds())}'''
                )

            if dry_run:
                print(f'\t\tDRY RUN: sbatch {sbatch_filename}')
            else:
                p = subprocess.Popen(f'sbatch {sbatch_filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                lines = list(p.stdout.readlines())
                retval = p.wait()
                if retval != 0:
                    print(lines)
                job_ids.append((int(lines[-1].decode('utf-8').rstrip().split(' ')[-1]), mem_gb, selected_config_indices))
                last_submit_time = dt.datetime.now()
                if submission_delay.total_seconds() > 0:
                    print(f'\tSubmitted job {job_ids[-1][0]}')

    with open(metadata_filename, 'w') as f:
        f.write(f'Time: {cur_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Job IDs:\n')
        for i,(job_id, mem_gb, tasks) in enumerate(job_ids):
            f.write(f'    submit_{i}.sbatch: {job_id}. {mem_gb}GB RAM, configs {tasks}\n')
        f.write(f'Max clock time: {max_time_str}\n')
        f.write(f'Total num. tasks: {len(configs)}\n')
        f.write(f'Params:\n')
        for name,params in sweep_params.items():
            f.write(f'    {name}: {params}\n')
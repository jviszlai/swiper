"""This script submits slurm jobs to run benchmarking simulations used in the
paper. The results are saved in the `artifact/data/` directory.

The script generates a large number of "experiment configurations", each of
which corresponds to one SWIPER-SIM simulation (by running
`artifact/_simulate.py`). The script then submits one or more arrays of SLURM
jobs to run these simulations. Each job reads one experiment configuration from
the `config.json` file and runs the corresponding simulation.

Requires a compute cluster with SLURM and a valid Python installation. Python
environment must be configured properly first. The four variables at the start
of this file must be configured to match the user's system. The rest of the
script does not need to be modified to reproduce the results in the paper, but
may require modifications if the user's system has a shorter job time limit or a
limit on the number of jobs that can be submitted at once.

In our experiments, we ran two separate instances of this script with different
parameters, running most benchmarks on a large cluster with a 36-hour job time
limit and running a few larger benchmarks on a smaller cluster with a 7-day job
time limit. Here, we combine them into a single script that runs all benchmarks 
with a 7-day job time limit. See `data/benchmarks1/submit_jobs_copy.py`
and `data/benchmarks2/submit_jobs_copy.py` for the
original job submission scripts used in the paper.

This script is adapted from `slurm/run_simulation.py` and `slurm/submit_jobs.py`
in the original repository.

This script can take between 1 minute and several hours to submit all jobs,
depending on configuration of `submission_delay` and `max_tasks_per_job`. The
SLURM jobs themselves will take several days to complete."""

import os, sys
sys.path.append('.')
import shutil
import json
import math
import subprocess
import datetime as dt
import time
from functools import reduce
import numpy as np
import csv

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
submission_delay = dt.timedelta(minutes=30)
max_tasks_per_job = 500
min_circuit_t_count = None
max_circuit_t_count = None
decoder_dist_source = 'artifact/data/decoder_dists.json' # can change if this has been re-generated elsewhere by running `artifact/run_pymatching_latencies.py`

################################################################################

if not dry_run and any([x == 'CHANGE_ME' for x in [slurm_account, slurm_partition, path_to_python_env, path_to_swiper]]):
    raise ValueError('Please configure the variables at the top of this script.')

if __name__ == '__main__':
    cur_time = dt.datetime.now()

    # USER SETTING: maximum job duration
    max_time = dt.timedelta(hours=24*7)

    if max_time.days > 0:
        max_time_str = f'{max_time.days}-{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'
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
    decoder_dist_filename = f'{data_dir}/{decoder_dist_source.split("/")[-1]}'
    shutil.copyfile(decoder_dist_source, decoder_dist_filename)
    shutil.copyfile('artifact/run_benchmark_evals.py', f'{data_dir}/run_benchmark_evals_copy.py')
    shutil.copyfile('artifact/_simulate.py', f'{data_dir}/_simulate_copy.py')

    with open('artifact/data/benchmark_info.csv', 'r') as f:
        reader = csv.DictReader(f)
        benchmark_info = {row['']:row for row in reader}

    # Can make a chosen smaller list of these instead
    benchmark_files = []
    benchmark_names = {}
    memory_settings = None
    for file in os.listdir('artifact/data/cached_schedules/'):
        # USER SETTING: filter benchmark files if desired
        if file.endswith('.lss') and not file.startswith('memory') and not file.startswith('regular') and not file.startswith('random'):
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
        'distance':[15, 21, 27],
        'scheduling_method':['sliding', 'parallel', 'aligned'],
        'decoder_latency_or_dist_filename':[decoder_dist_filename],
        'speculation_mode':['separate', None],
        'speculation_latency':[1],
        'speculation_accuracy':[0.9, 1.0],
        'poison_policy':['successors'],
        'missed_speculation_modifier':[1.4],
        'max_parallel_processes':[None, 'predict'],
        'rng':list(range(10)),
        'lightweight_setting':[2],
    }
    ordered_param_names = list(sorted(sweep_params.keys()))
    total_num_configs = reduce(lambda x,y: x*y, [len(params) for params in sweep_params.values()])

    # USER SETTING: filter out some combinations of the above parameters
    microbenchmarks = [os.path.join(benchmark_dir, file) for file in ['msd_15to1.lss', 'adder_n4.lss', 'adder_n10.lss', 'adder_n18.lss', 'adder_n28.lss' 'rz_1e-05.lss', 'rz_1e-10.lss', 'rz_1e-15.lss', 'rz_1e-20.lss', 'toffoli.lss', 'qrom_15_15.lss']]
    def config_filter(cfg):
        # TODO: make the logic here more clear...
        return (
            (cfg['distance'] == 21 or (cfg['speculation_accuracy'] == 0.9 and cfg['max_parallel_processes'] == None)) # distance 15 and 27 runs require less data
            and (not (cfg['speculation_accuracy'] == 1.0 and (cfg['speculation_mode'] == None or cfg['max_parallel_processes'] == 'predict')))
            and (not (cfg['speculation_mode'] == None and (cfg['scheduling_method'] == 'sliding' or cfg['max_parallel_processes'] == 'predict'))) # don't want to turn off swiper for sliding window, or for predicting computational cost
            and (min_circuit_t_count is None or (float(benchmark_info[cfg['benchmark_file'].split('/')[-1].split('.')[0]]['T count']) > min_circuit_t_count))
            and (max_circuit_t_count is None or (float(benchmark_info[cfg['benchmark_file'].split('/')[-1].split('.')[0]]['T count']) < max_circuit_t_count))
            and (float(benchmark_info[cfg['benchmark_file'].split('/')[-1].split('.')[0]]['T count']) < 3500 or cfg['rng'] == 0) # only small benchmarks get multiple runs
        )

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
        if config_filter(config):
            configs.append(config)
    print('Generated', len(configs), 'configs')
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
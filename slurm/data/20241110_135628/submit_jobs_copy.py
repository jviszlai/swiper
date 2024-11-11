import os, sys
import shutil
import json
import math
import subprocess
import datetime as dt
from functools import reduce
import numpy as np

if __name__ == '__main__':
    time = dt.datetime.now()

    # USER SETTING: maximum job duration
    max_time = dt.timedelta(hours=36)

    if max_time.days > 0:
        assert max_time.days == 1
        max_time_str = f'1-{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'
    else:
        max_time_str = f'{max_time.seconds // 3600:02d}:{(max_time.seconds % 3600) // 60:02d}:{max_time.seconds % 60:02d}'

    data_dir = f'slurm/data/{time.strftime("%Y%m%d_%H%M%S")}'
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

    # USER SETTING: decoder distribution (can also set to an integer)
    decoder_dist_source = 'benchmarks/data/decoder_dists.json'
    decoder_dist_filename = f'{data_dir}/{decoder_dist_source.split("/")[-1]}'
    shutil.copyfile(decoder_dist_source, decoder_dist_filename)

    # copy files to data dir to preserve them
    shutil.copyfile('slurm/submit_jobs.py', f'{data_dir}/submit_jobs_copy.py')
    shutil.copyfile('slurm/run_simulation.py', f'{data_dir}/run_simulation_copy.py')

    # Can make a chosen smaller list of these instead
    benchmark_files = []
    benchmark_names = {}
    memory_settings = None
    for file in os.listdir('benchmarks/cached_schedules/'):
        # USER SETTING: filter benchmark files if desired
        if file.endswith('.lss'):
            path = os.path.join('benchmarks/cached_schedules/', file)
            newpath = os.path.join(benchmark_dir, file)
            # copy files to data dir to preserve them
            shutil.copyfile(path, newpath)
            benchmark_files.append(newpath)
            benchmark_names[newpath] = file.split('.')[0]
        elif file == '.memory_settings.json':
            # USER SETTING: change values in this file to add memory for certain benchmarks
            memory_settings = json.load(open(os.path.join('benchmarks/cached_schedules/', file), 'r'))
    assert memory_settings is not None

    for name in set(benchmark_names.values()):
        if name not in memory_settings:
            print(f'WARNING: {name} not found in .memory_settings.json, using default 4GB...')
            memory_settings[name] = 4

    # USER SETTING: change parameter sweeps for distance, spec acc, etc.
    sweep_params = {
        'distance':[21],
        'speculation_latency':[1],
        'speculation_accuracy':[0.9],
        'speculation_mode':['separate', None],
        'scheduling_method':['sliding', 'parallel', 'aligned'],
        'max_parallel_processes':[None],
        'benchmark_file':benchmark_files,
        'decoder_latency_or_dist_filename':[decoder_dist_filename],
        'rng':[0],
        'lightweight_setting':[2],
    }
    ordered_param_names = list(sorted(sweep_params.keys()))
    total_num_configs = reduce(lambda x,y: x*y, [len(params) for params in sweep_params.values()])

    # USER SETTING: filter out some combinations of the above parameters
    def config_filter(cfg):
        return not (cfg['speculation_mode'] == None and cfg['scheduling_method'] == 'sliding')

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
    with open(config_filename, 'w') as f:
        json.dump(configs, f)

    # submit a different sbatch job for each config
    configs_by_mem = {}
    for i,config in enumerate(configs):
        mem_gb = memory_settings[benchmark_names[config['benchmark_file']]]
        configs_by_mem.setdefault(mem_gb, []).append(i)

    job_ids = []
    submit_idx = 0
    for i,mem_gb in enumerate(sorted(configs_by_mem.keys())):
        config_indices = configs_by_mem[mem_gb]
        num_submissions = math.ceil(len(config_indices) / 1000) # caslake submission limit
        for j in range(num_submissions):
            selected_config_indices = config_indices[j*1000:(j+1)*1000]
            sbatch_filename = os.path.join(sbatch_dir, f'submit_{submit_idx}.sbatch')
            submit_idx += 1
            with open(sbatch_filename, 'w') as f:
                f.write(f'''#!/bin/bash
#SBATCH --job-name={time.strftime("%Y%m%d_%H%M%S")}
#SBATCH --output={log_dir}/%a.out
#SBATCH --error={log_dir}/%a.out
#SBATCH --account=pi-ftchong
#SBATCH --partition=caslake
#SBATCH --array={','.join([str(x) for x in config_indices])}
#SBATCH --time={max_time_str}
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu={mem_gb*1000}

module load python
eval "$(conda shell.bash hook)"
conda activate /project/ftchong/projects/envs/pySwiper/

python slurm/run_simulation.py "{config_filename}" "{output_dir}" {int(max_time.total_seconds())}'''
                )

            p = subprocess.Popen(f'sbatch {sbatch_filename}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines = list(p.stdout.readlines())
            retval = p.wait()
            if retval != 0:
                print(lines)
            job_ids.append((int(lines[-1].decode('utf-8').rstrip().split(' ')[-1]), mem_gb))

    with open(metadata_filename, 'w') as f:
        f.write(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Job IDs:\n')
        for i,(job_id, mem_gb) in enumerate(job_ids):
            f.write(f'    submit_{i}.sbatch: {job_id}. {mem_gb}GB RAM, configs {configs_by_mem[mem_gb]}\n')
        f.write(f'Max clock time: {max_time_str}\n')
        f.write(f'Total num. tasks: {len(configs)}\n')
        f.write(f'Params:\n')
        for name,params in sweep_params.items():
            f.write(f'    {name}: {params}\n')
import sys, os
import json
import pickle
import datetime as dt
import numpy as np
import math
from swiper.simulator import DecodingSimulator
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule

if __name__ == '__main__':
    args = sys.argv

    config_filename = args[1]
    output_dir = args[2]
    config_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

    start_time = dt.datetime.now()

    with open(config_filename, 'r') as f:
        params = json.load(f)[config_idx]

    assert len(params) == 9, 'Params list changed. Update this file!'
    distance = params['distance']
    max_parallel_processes = params['max_parallel_processes']
    scheduling_method = params['scheduling_method']
    speculation_accuracy = params['speculation_accuracy']
    speculation_latency = params['speculation_latency']
    speculation_mode = params['speculation_mode']
    benchmark_file = params['benchmark_file']
    decoder_dist_filename = params['decoder_dist_filename']
    rng = params['rng']

    generator = np.random.default_rng(rng)

    # decoder_dists[str(d)][str(volume)] = 10,000 sampled latencies, in microseconds
    decoder_dists = json.load(open(decoder_dist_filename, 'r'))
    decoder_dist = {}
    for dist_str, dist_dict in decoder_dists.items():
        if int(dist_str) == distance:
            decoder_dist = {int(k):v for k,v in dist_dict.items()}
    decoding_latency_fn = lambda volume: generator.choice(decoder_dist[max(2, math.ceil(volume / distance))])

    print(f'{start_time.strftime("%Y-%m-%d %H:%M:%S")} | Loading benchmark {benchmark_file}...')
    sys.stdout.flush()

    with open(benchmark_file, 'r') as f:
        benchmark_schedule = LatticeSurgerySchedule.from_str(f.read(), generate_dag_incrementally=True)

    simulator = DecodingSimulator(
        distance=distance,
        decoding_latency_fn=decoding_latency_fn,
        speculation_latency=speculation_latency,
        speculation_accuracy=speculation_accuracy,
        speculation_mode=speculation_mode,
    )

    success, device_data, window_data, decoding_data = simulator.run(
        schedule=benchmark_schedule,
        scheduling_method=scheduling_method,
        max_parallel_processes=max_parallel_processes,
        print_interval=dt.timedelta(seconds=10),
        lightweight_output=True,
        rng=rng,
    )

    with open(os.path.join(output_dir, f'config_{config_idx}_d{distance}_{scheduling_method}_acc{speculation_accuracy}_spec{speculation_latency}_{speculation_mode}_p{max_parallel_processes}_{benchmark_file.split("/")[-1].split('.')[0]}_{rng}.txt'), 'w') as f:
        json.dump({
                'success':success,
                'device_data':device_data.to_dict(),
                'window_data':window_data.to_dict(),
                'decoding_data':decoding_data.to_dict(),
            },
            f
        )

    finish_time = dt.datetime.now()
    print(f'{finish_time.strftime("%Y-%m-%d %H:%M:%S")} | Finished saving output. Done! Total elapsed time: {finish_time - start_time}')
    sys.stdout.flush()
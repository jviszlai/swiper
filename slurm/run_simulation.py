import sys, os
import json
import pickle

from swiper.simulator import DecodingSimulator
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule

if __name__ == '__main__':
    args = sys.argv

    config_filename = args[1]
    output_dir = args[2]
    config_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

    with open(config_filename, 'r') as f:
        params = json.load(f)[config_idx]

    assert len(params) == 8, 'Params list changed. Update this file!'
    distance = params['distance']
    max_parallel_processes = params['max_parallel_processes']
    scheduling_method = params['scheduling_method']
    speculation_accuracy = params['speculation_accuracy']
    speculation_latency = params['speculation_latency']
    speculation_mode = params['speculation_mode']
    benchmark_file = params['benchmark_file']
    rng = params['rng']

    with open(benchmark_file, 'r') as f:
        benchmark_schedule = LatticeSurgerySchedule.from_str(f.read(), generate_dag_incrementally=True)

    simulator = DecodingSimulator(
        distance=distance,
        decoding_latency_fn=lambda _: 0, # TODO: put real decoding latency in here
        speculation_latency=speculation_latency,
        speculation_accuracy=speculation_accuracy,
        speculation_mode=speculation_mode,
    )

    success, device_data, window_data, decoding_data = simulator.run(
        schedule=benchmark_schedule,
        scheduling_method=scheduling_method,
        max_parallel_processes=max_parallel_processes,
        lightweight_output=True,
        rng=rng,
    )

    with open(os.path.join(output_dir, f'config_{config_idx}_d{distance}_p{max_parallel_processes}_{scheduling_method}_acc{speculation_accuracy}_spec{speculation_latency}_{speculation_mode}_{benchmark_file.split("/")[-1]}_{rng}.txt'), 'w') as f:
        json.dump({
                'success':success,
                'device_data':device_data.to_dict(),
                'window_data':window_data.to_dict(),
                'decoding_data':decoding_data.to_dict(),
            },
            f
        )
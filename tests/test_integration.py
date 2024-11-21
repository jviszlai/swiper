"""Integration tests that use all components (device manager, window manager,
decoder manager) together."""
import pytest
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from swiper.simulator import DecodingSimulator
import swiper.plot as plotter
from swiper.schedule_experiments import MemorySchedule, RegularTSchedule, MSD15To1Schedule, RandomTSchedule
from swiper.simulator import DecodingSimulator

def test_sliding_memory():
    d=7
    decoding_time = 14
    speculation_time = 0
    speculation_accuracy = 0
    simulator = DecodingSimulator()

    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=MemorySchedule(d).schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
    )
    assert device_data.num_rounds == d
    assert decoding_data.num_rounds == d+decoding_time

    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=MemorySchedule(2*d).schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
    )
    assert device_data.num_rounds == 2*d
    assert decoding_data.num_rounds == 2*d + 2*decoding_time

    d=7
    decoding_time = 14
    speculation_time = 1
    speculation_accuracy = 1
    simulator = DecodingSimulator()

    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=MemorySchedule(2*d).schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
    )
    assert device_data.num_rounds == 2*d
    assert decoding_data.num_rounds == 2*d + 1 + decoding_time

def test_sliding_regular_T():
    d=7
    decoding_time = 14
    speculation_time = 100
    speculation_accuracy = 0
    simulator = DecodingSimulator()

    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=RegularTSchedule(1, 0).schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    assert device_data.num_rounds == 2*d + 2*decoding_time + d//2+2
    assert decoding_data.num_rounds == 2*d + 2*decoding_time + math.ceil((2*decoding_time + d//2+2) / d) * decoding_time

    d=7
    decoding_time = 14
    speculation_time = 0
    speculation_accuracy = 0
    simulator = DecodingSimulator()

    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=RegularTSchedule(1, 0).schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    assert device_data.num_rounds == 2*d + 2*decoding_time + d//2+2
    assert decoding_data.num_rounds == 2*d + 2*decoding_time + math.ceil((2*decoding_time + d//2+2) / d) * decoding_time

    d=7
    decoding_time = 14
    speculation_time = 0
    speculation_accuracy = 1
    simulator = DecodingSimulator()

    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=RegularTSchedule(1, 0).schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    assert device_data.num_rounds == 2*d + d + decoding_time + d//2+2
    assert decoding_data.num_rounds == device_data.num_rounds + decoding_time


def test_poor_predictor_same_as_slow_predictor_idle():
    """Test that a poor predictor (always gives the wrong answer) gives the same
    total runtime as a slow predictor (takes much longer than decoding).
    """
    d=7
    decoding_time = 10
    regular_t_schedule = MemorySchedule(100)
    
    # Poor predictor
    speculation_time = 1
    speculation_accuracy = 0
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_bad_speculation = decoding_data.num_rounds

    # Slow predictor
    speculation_time = 100*d
    speculation_accuracy = 1
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_slow_speculation = decoding_data.num_rounds

    # No speculation
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode=None,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_no_speculation = decoding_data.num_rounds

    assert num_rounds_bad_speculation == num_rounds_slow_speculation == num_rounds_no_speculation

def test_poor_predictor_same_as_slow_predictor_t():
    """Test that a poor predictor (always gives the wrong answer) gives the same
    total runtime as a slow predictor (takes much longer than decoding).
    """
    d=7
    decoding_time = 10
    regular_t_schedule = RegularTSchedule(2, 0)
    
    # Poor predictor
    speculation_time = 1
    speculation_accuracy = 0
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_bad_speculation = decoding_data.num_rounds

    # Slow predictor
    speculation_time = 100*d
    speculation_accuracy = 1
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_slow_speculation = decoding_data.num_rounds

    # No speculation
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode=None,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_no_speculation = decoding_data.num_rounds

    assert num_rounds_bad_speculation == num_rounds_slow_speculation == num_rounds_no_speculation

def test_integrated_and_separate_consistency_with_bad_predictions():
    """Test that the integrated and separate speculation modes give the same
    total runtime when the predictor is bad.
    """
    d=7
    decoding_time = 3
    speculation_time = 1
    speculation_accuracy = 0
    regular_t_schedule = RegularTSchedule(10, 2*d)
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='integrated',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_integrated = decoding_data.num_rounds

    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        distance=d,
        scheduling_method='sliding',
        decoding_latency_fn=lambda _: decoding_time,
        speculation_mode='separate',
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        max_parallel_processes=None,
        rng=0,
    )
    num_rounds_separate = decoding_data.num_rounds

    assert num_rounds_integrated == num_rounds_separate

def test_lightweight_output():
    d=7
    decoding_time = 14
    speculation_latency = 1
    speculation_accuracy = 0.5
    simulator = DecodingSimulator()
    schedule = MSD15To1Schedule().schedule

    for scheduling_method in ['sliding', 'parallel', 'aligned']:
        success, _, device_data_0, window_data_0, decoding_data_0 = simulator.run(
            schedule=schedule,
            distance=d,
            scheduling_method=scheduling_method,
            decoding_latency_fn=lambda _: decoding_time,
            speculation_mode='separate',
            speculation_latency=speculation_latency,
            speculation_accuracy=speculation_accuracy,
            max_parallel_processes=None,
            rng=0,
            lightweight_setting=0,
        )

        success, _, device_data_1, window_data_1, decoding_data_1 = simulator.run(
            schedule=schedule,
            distance=d,
            scheduling_method=scheduling_method,
            decoding_latency_fn=lambda _: decoding_time,
            speculation_mode='separate',
            speculation_latency=speculation_latency,
            speculation_accuracy=speculation_accuracy,
            max_parallel_processes=None,
            rng=0,
            lightweight_setting=1,
        )

        success, _, device_data_2, window_data_2, decoding_data_2 = simulator.run(
            schedule=schedule,
            distance=d,
            scheduling_method=scheduling_method,
            decoding_latency_fn=lambda _: decoding_time,
            speculation_mode='separate',
            speculation_latency=speculation_latency,
            speculation_accuracy=speculation_accuracy,
            max_parallel_processes=None,
            rng=0,
            lightweight_setting=2,
        )

        # success, device_data_3, window_data_3, decoding_data_3 = simulator.run(
        #     schedule=schedule,
        #     scheduling_method=scheduling_method,
        #     max_parallel_processes=None,
        #     rng=0,
        #     lightweight_setting=3,
        # )

        assert device_data_0.num_rounds == device_data_1.num_rounds == device_data_2.num_rounds
        assert decoding_data_0.num_rounds == decoding_data_1.num_rounds == decoding_data_2.num_rounds
        assert device_data_0.avg_conditioned_decode_wait_time > 0
        assert np.isclose(np.mean(list(device_data_0.conditioned_decode_wait_times.values())), np.mean(list(device_data_1.conditioned_decode_wait_times.values()))), (np.mean(list(device_data_0.conditioned_decode_wait_times.values())), np.mean(list(device_data_1.conditioned_decode_wait_times.values())))
        assert np.isclose(np.mean(list(device_data_0.conditioned_decode_wait_times.values())), device_data_2.avg_conditioned_decode_wait_time)

if __name__ == '__main__':
    import os
    try:
        path_initialized
    except NameError:
        path_initialized = True
        os.chdir('..')
        
    test_lightweight_output()
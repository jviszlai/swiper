"""Integration tests that use all components (device manager, window manager,
decoder manager) together."""
import pytest
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from swiper2.schedule_experiments import MemorySchedule, RegularTSchedule, MSD15To1Schedule
from swiper2.simulator import DecodingSimulator

def test_poor_predictor_same_as_slow_predictor():
    """Test that a poor predictor (always gives the wrong answer) gives the same
    total runtime as a slow predictor (takes much longer than decoding).
    """
    d=7
    decoding_time = 3
    regular_t_schedule = RegularTSchedule(10, 2*d)
    
    # Poor predictor
    speculation_time = 1
    speculation_accuracy = 0
    simulator = DecodingSimulator(
        distance=d,
        decoding_latency_fn=lambda _: decoding_time,
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        speculation_mode='integrated',
    )
    success, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        scheduling_method='sliding',
        enforce_window_alignment=False,
        max_parallel_processes=None,
    )
    num_rounds_bad_speculation = decoding_data.num_rounds

    # Slow predictor
    speculation_time = 100*d
    speculation_accuracy = 1
    simulator = DecodingSimulator(
        distance=d,
        decoding_latency_fn=lambda _: decoding_time,
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        speculation_mode='integrated',
    )
    success, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        scheduling_method='sliding',
        enforce_window_alignment=False,
        max_parallel_processes=None,
    )
    num_rounds_slow_speculation = decoding_data.num_rounds

    assert num_rounds_bad_speculation == num_rounds_slow_speculation

def test_integrated_and_separate_consistency_with_bad_predictions():
    """Test that the integrated and separate speculation modes give the same
    total runtime when the predictor is bad.
    """
    d=7
    decoding_time = 3
    speculation_time = 1
    speculation_accuracy = 0
    regular_t_schedule = RegularTSchedule(10, 2*d)
    simulator = DecodingSimulator(
        distance=d,
        decoding_latency_fn=lambda _: decoding_time,
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        speculation_mode='integrated',
    )
    success, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        scheduling_method='sliding',
        enforce_window_alignment=False,
        max_parallel_processes=None,
    )
    num_rounds_integrated = decoding_data.num_rounds

    simulator = DecodingSimulator(
        distance=d,
        decoding_latency_fn=lambda _: decoding_time,
        speculation_latency=speculation_time,
        speculation_accuracy=speculation_accuracy,
        speculation_mode='separate',
    )
    success, device_data, window_data, decoding_data = simulator.run(
        schedule=regular_t_schedule.schedule,
        scheduling_method='sliding',
        enforce_window_alignment=False,
        max_parallel_processes=None,
    )
    num_rounds_separate = decoding_data.num_rounds

    assert num_rounds_integrated == num_rounds_separate
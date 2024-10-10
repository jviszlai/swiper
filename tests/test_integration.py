"""Integration tests that use all components (device manager, window manager,
decoder manager) together."""
import pytest
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from swiper2.schedule_experiments import MemorySchedule, RegularTSchedule, MSD15To1Schedule
from swiper2.device_manager import DeviceManager
from swiper2.window_builder import WindowBuilder
from swiper2.window_manager import SlidingWindowManager
from swiper2.decoder_manager import DecoderManager

def test_poor_predictor_same_as_slow_predictor():
    """Test that a poor predictor (always gives the wrong answer) gives the same
    total runtime as a slow predictor (takes much longer than decoding).
    """
    d=7
    decoding_time = 3
    speculation_time = 1
    speculation_accuracy = 0
    regular_t_schedule = RegularTSchedule(10, 2*d)
    manager = DeviceManager(d, regular_t_schedule.schedule)
    sliding_manager = SlidingWindowManager(WindowBuilder(d, enforce_alignment=False))
    decoder_manager = DecoderManager(lambda _: decoding_time, speculation_time, speculation_accuracy, speculation_mode='integrated')
    windows_to_decode = 0

    while not manager.is_done() or windows_to_decode > 0:
        # step device forward
        decoder_manager.step(sliding_manager.all_windows, sliding_manager.window_dag)
        fully_decoded_instructions = decoder_manager.get_finished_instruction_indices(sliding_manager.all_windows) - sliding_manager.pending_instruction_indices()

        new_round = manager.get_next_round(fully_decoded_instructions)
        
        # process new round
        sliding_manager.process_round(new_round)
        decoder_manager.update_decoding(sliding_manager.all_windows, sliding_manager.window_dag)
        
        windows_to_decode = len(sliding_manager.all_windows) - len(decoder_manager._window_completion_times)

    num_rounds_bad_speculation = len(decoder_manager.get_data().max_parallel_processes_by_round)


    speculation_time = 100*d
    speculation_accuracy = 1
    manager = DeviceManager(d, regular_t_schedule.schedule)
    sliding_manager = SlidingWindowManager(WindowBuilder(d, enforce_alignment=False))
    decoder_manager = DecoderManager(lambda _: decoding_time, speculation_time, speculation_accuracy, speculation_mode='integrated')
    windows_to_decode = 0

    while not manager.is_done() or windows_to_decode > 0:
        # step device forward
        decoder_manager.step(sliding_manager.all_windows, sliding_manager.window_dag)
        fully_decoded_instructions = decoder_manager.get_finished_instruction_indices(sliding_manager.all_windows) - sliding_manager.pending_instruction_indices()

        new_round = manager.get_next_round(fully_decoded_instructions)
        
        # process new round
        sliding_manager.process_round(new_round)
        decoder_manager.update_decoding(sliding_manager.all_windows, sliding_manager.window_dag)
        
        windows_to_decode = len(sliding_manager.all_windows) - len(decoder_manager._window_completion_times)

    num_rounds_slow_speculation = len(decoder_manager.get_data().max_parallel_processes_by_round)

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
    manager = DeviceManager(d, regular_t_schedule.schedule)
    sliding_manager = SlidingWindowManager(WindowBuilder(d, enforce_alignment=False))
    decoder_manager = DecoderManager(lambda _: decoding_time, speculation_time, speculation_accuracy, speculation_mode='integrated')
    windows_to_decode = 0

    while not manager.is_done() or windows_to_decode > 0:
        # step device forward
        decoder_manager.step(sliding_manager.all_windows, sliding_manager.window_dag)
        fully_decoded_instructions = decoder_manager.get_finished_instruction_indices(sliding_manager.all_windows) - sliding_manager.pending_instruction_indices()

        new_round = manager.get_next_round(fully_decoded_instructions)
        
        # process new round
        sliding_manager.process_round(new_round)
        decoder_manager.update_decoding(sliding_manager.all_windows, sliding_manager.window_dag)
        
        windows_to_decode = len(sliding_manager.all_windows) - len(decoder_manager._window_completion_times)

    num_rounds_integrated = len(decoder_manager.get_data().max_parallel_processes_by_round)

    manager = DeviceManager(d, regular_t_schedule.schedule)
    sliding_manager = SlidingWindowManager(WindowBuilder(d, enforce_alignment=False))
    decoder_manager = DecoderManager(lambda _: decoding_time, speculation_time, speculation_accuracy, speculation_mode='separate')
    windows_to_decode = 0

    while not manager.is_done() or windows_to_decode > 0:
        # step device forward
        decoder_manager.step(sliding_manager.all_windows, sliding_manager.window_dag)
        fully_decoded_instructions = decoder_manager.get_finished_instruction_indices(sliding_manager.all_windows) - sliding_manager.pending_instruction_indices()

        new_round = manager.get_next_round(fully_decoded_instructions)
        
        # process new round
        sliding_manager.process_round(new_round)
        decoder_manager.update_decoding(sliding_manager.all_windows, sliding_manager.window_dag)
        
        windows_to_decode = len(sliding_manager.all_windows) - len(decoder_manager._window_completion_times)

    num_rounds_separate = len(decoder_manager.get_data().max_parallel_processes_by_round)

    assert num_rounds_integrated == num_rounds_separate
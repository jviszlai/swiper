"""TODO: test that WindowBuilder builds windows correctly. Specific cases to
test: 
    1) enforcing alignment: check that alignment is correct, and that every
       window is within desired size bounds.
    ...
"""
import numpy as np
import networkx as nx
from swiper.simulator import DecodingSimulator
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.window_builder import WindowBuilder, DecodingWindow
from swiper.window_manager import WindowManager, WindowData, SlidingWindowManager, ParallelWindowManager, TAlignedWindowManager
from swiper.schedule_experiments import MSD15To1Schedule, RandomTSchedule

distance = 7
decoding_latency = 8
decoding_fn = lambda _: decoding_latency
speculation_latency = 2
speculation_accuracy = 0
speculation_mode = 'integrated'

def xor(a, b):
    return (a and not b) or (not a and b)

def test_spatial_adjacency():
    """"""
    simulator = DecodingSimulator()

    schedule = LatticeSurgerySchedule()
    schedule.idle([(0,0)], distance)
    schedule.merge([(0,1), (1,0)], [(1,1)], duration=distance)

    simulator.initialize_experiment(
        schedule=schedule,
        distance=distance,
        scheduling_method='sliding',
        decoding_latency_fn=decoding_fn,
        speculation_mode=speculation_mode,
        speculation_latency=speculation_latency,
        speculation_accuracy=speculation_accuracy,
        rng=0,
    )
    assert simulator._window_manager

    # Check windows as they are released
    while not simulator.is_done():
        simulator.step_experiment()

    success, _, device_data, window_data, decoding_data = simulator.get_data()

    assert len(window_data.all_constructed_windows) == 4
    windows_by_patch: dict[tuple[int, int], DecodingWindow] = {}
    for window_idx in window_data.all_constructed_windows:
        window = window_data.get_window(window_idx)
        assert len(window.commit_region) == 1
        windows_by_patch[window.commit_region[0].patch] = window
    
    assert windows_by_patch[(0,0)].total_spacetime_volume() == distance
    assert windows_by_patch[(0,0)].overlaps(windows_by_patch[(0,1)]) == False
    assert windows_by_patch[(0,0)].overlaps(windows_by_patch[(1,0)]) == False
    assert windows_by_patch[(0,0)].overlaps(windows_by_patch[(1,1)]) == False

    assert windows_by_patch[(0,1)].commit_region[0].duration == distance
    assert windows_by_patch[(0,1)].overlaps(windows_by_patch[(0,0)]) == False
    assert windows_by_patch[(0,1)].overlaps(windows_by_patch[(1,0)]) == False
    
    assert windows_by_patch[(0,1)].overlaps(windows_by_patch[(1,1)])
    assert windows_by_patch[(1,0)].overlaps(windows_by_patch[(1,1)])
    assert len(windows_by_patch[(0,1)].buffer_regions) + len(windows_by_patch[(1,1)].buffer_regions) + len(windows_by_patch[(1,0)].buffer_regions) == 2

    assert sum(w.total_spacetime_volume() for w in windows_by_patch.values()) == 6*distance

    schedule.merge([(0,0), (0,1)], [], duration=distance)
    schedule.merge([(1,0), (1,1)], [], duration=distance)
    schedule.discard([(0,0), (0,1), (1,0), (1,1)])

    simulator.initialize_experiment(
        schedule=schedule,
        distance=distance,
        scheduling_method='sliding',
        decoding_latency_fn=decoding_fn,
        speculation_mode=speculation_mode,
        speculation_latency=speculation_latency,
        speculation_accuracy=speculation_accuracy,
        rng=0,
    )
    assert simulator._window_manager

    # Check windows as they are released
    while not simulator.is_done():
        simulator.step_experiment()

    success, _, device_data, window_data, decoding_data = simulator.get_data()

    def get_window(patch, round_start) -> DecodingWindow:
        for window_idx in window_data.all_constructed_windows:
            window = window_data.get_window(window_idx)
            if window.commit_region[0].patch == patch and window.commit_region[0].round_start == round_start:
                return window
        raise ValueError(f"Window not found for patch {patch} and round_start {round_start}")
    
    all_syndrome_rounds = [sr for sublist in device_data.generated_syndrome_data for sr in sublist]
    all_commit_regions = [cr for window_idx in window_data.all_constructed_windows for cr in window_data.get_window(window_idx).commit_region]

    for syndrome_round in all_syndrome_rounds:
        assert any(cr.contains_syndrome_round(syndrome_round=syndrome_round) for cr in all_commit_regions)

    for window_idx_1 in window_data.all_constructed_windows:
        window_1 = window_data.get_window(window_idx_1)
        for window_idx_2 in window_data.all_constructed_windows:
            window_2 = window_data.get_window(window_idx_2)
            if window_1 is window_2:
                continue
            if window_1.shares_spacelike_boundary(window_2):
                assert window_1.commit_region[0].round_start == window_2.commit_region[0].round_start
                assert window_1.commit_region[0].merge_instr == window_2.commit_region[0].merge_instr
            if window_1.shares_timelike_boundary(window_2):
                assert window_1.commit_region[0].patch == window_2.commit_region[0].patch
                assert window_1.commit_region[0].round_start != window_2.commit_region[0].round_start
            if window_1.shares_boundary(window_2):
                assert window_1.overlaps(window_2)

    assert len(window_data.all_constructed_windows) == 8
    assert get_window((0,0), 0).shares_timelike_boundary(get_window((0,0), distance))
    assert not any(get_window((0,0), 0).shares_spacelike_boundary(get_window(p, 0)) for p in [(0,1), (1,0), (1,1)])
    assert get_window((0,1), 0).shares_timelike_boundary(get_window((0,1), distance))
    assert get_window((0,1), 0).shares_spacelike_boundary(get_window((1,1), 0))
    
    for patch1 in [(0,0), (0,1), (1,0), (1,1)]:
        for patch2 in [(0,0), (0,1), (1,0), (1,1)]:
            if patch1 == patch2:
                continue
            assert not get_window(patch1, 0).shares_spacelike_boundary(get_window(patch2, distance))
            assert not get_window(patch1, 0).shares_timelike_boundary(get_window(patch2, distance))

    assert get_window((0,0), distance).shares_spacelike_boundary(get_window((0,1), distance))
    assert not get_window((0,1), distance).shares_spacelike_boundary(get_window((1,1), distance))
    assert get_window((1,0), distance).shares_spacelike_boundary(get_window((1,1), distance))

    bbc_000 = get_window((0,0), 0).buffer_boundary_commits()
    assert len(bbc_000) == 1
    assert bbc_000[list(get_window((0,0),0).buffer_regions)[0]] == [get_window((0,0),0).commit_region[0]]

    bbc_011 = get_window((0,1), distance).buffer_boundary_commits()
    assert len(bbc_011) == 1
    assert bbc_011[list(get_window((0,1),distance).buffer_regions)[0]] == [get_window((0,1),distance).commit_region[0]]

    assert get_window((0,0), distance).get_touching_commit_regions(get_window((0,1), distance)) == [get_window((0,1),distance).commit_region[0]]

if __name__ == '__main__':
    test_spatial_adjacency()
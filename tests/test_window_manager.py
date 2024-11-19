"""TODO: test that all variants of WindowManager work correctly. Specific cases
to test:
    1) All: check that windows are processed correctly, e.g. not
       releasing windows until all buffer regions are complete, and not waiting
       excessively long to release a window once it is ready
    2) All: check that final windows are all valid, e.g. each commit region is
       contained in one and only one window, and buffer regions have correct
       overlap. Only specific windows should have commit regions that are not
       buffered by any other window. Commit regions (and commit+buffer) should
       be contiguous.
    3) SlidingWindowManager: check that each window has at most one buffer
       region and that DAG is tree-like
    4) ParallelWindowManager: Check number of layers of parallel windows in
       simple examples
    5) DynamicWindowManager: TODO
    ...
"""
import numpy as np
import networkx as nx
from swiper.simulator import DecodingSimulator
from swiper.device_manager import DeviceData
from swiper.decoder_manager import DecoderData
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.window_builder import WindowBuilder
from swiper.window_manager import WindowManager, WindowData, SlidingWindowManager, ParallelWindowManager, TAlignedWindowManager
from swiper.schedule_experiments import MSD15To1Schedule, RandomTSchedule, RegularTSchedule

idle_schedule = LatticeSurgerySchedule()
idle_schedule.idle([(0,0)], 101)
idle_schedule.discard([(0,0)])

merge_schedule = LatticeSurgerySchedule()
merge_schedule.idle([(0,0)], 10)
merge_schedule.idle([(0,10)], 10)
merge_schedule.merge([(0,0), (0,10)], [(0,i) for i in range(1,10)])
merge_schedule.merge([(0,0), (0,10)], [(0,i) for i in range(1,10)])
merge_schedule.merge([(0,0), (0,10)], [(0,i) for i in range(1,10)])
merge_schedule.discard([(0,0), (0,10)])

distance = 7
decoding_fn = lambda _: 8
speculation_latency = 2
speculation_accuracy = 0.9
speculation_mode = 'integrated'

def do_intermediate_checks(simulator: DecodingSimulator):
   assert simulator._device_manager and simulator._window_manager and simulator._decoding_manager
   for window_idx, window in enumerate(simulator._window_manager.all_windows):
      assert window is None or window.constructed or window_idx in simulator._window_manager.window_construction_wait

   all_syndrome_rounds = [sr for sublist in simulator._device_manager._generated_syndrome_data for sr in sublist]
   assert simulator._window_manager.window_builder._total_rounds_processed == len(all_syndrome_rounds)
   all_commit_regions = [cr for window in simulator._window_manager.all_windows if window for cr in window.commit_region]

   for i,syndrome_round in enumerate(all_syndrome_rounds):
      assert syndrome_round.instruction.name == 'INJECT_T' or i in simulator._window_manager.window_builder._waiting_rounds or any(cr.contains_syndrome_round(syndrome_round=syndrome_round) for cr in all_commit_regions)

   # if isinstance(simulator._window_manager, ParallelWindowManager):
   #    assert simulator._window_manager.layer_indices[1].isdisjoint(simulator._window_manager.layer_indices[2])
   #    assert set(i for i,window in enumerate(simulator._window_manager.all_windows) if window) == simulator._window_manager.layer_indices[0] | simulator._window_manager.layer_indices[1] | simulator._window_manager.layer_indices[2], (set(i for i,window in enumerate(simulator._window_manager.all_windows) if window), simulator._window_manager.layer_indices[0] | simulator._window_manager.layer_indices[1] | simulator._window_manager.layer_indices[2])
   # elif isinstance(simulator._window_manager, SlidingWindowManager):
   #    pass
   # elif isinstance(simulator._window_manager, TAlignedWindowManager):
   #    assert simulator._window_manager.layer_indices[1].isdisjoint(simulator._window_manager.layer_indices[3])
   #    assert set(i for i,window in enumerate(simulator._window_manager.all_windows) if window) == simulator._window_manager.layer_indices[0] | simulator._window_manager.layer_indices[1] | simulator._window_manager.layer_indices[2] | simulator._window_manager.layer_indices[3]
   #    assert len([i for i,window in enumerate(simulator._window_manager.all_windows) if window]) == len(simulator._window_manager.layer_indices[0]) + len(simulator._window_manager.layer_indices[1]) + len(simulator._window_manager.layer_indices[2]) + len(simulator._window_manager.layer_indices[3])


def do_final_checks(device_data: DeviceData, window_data: WindowData, decoder_data: DecoderData, simulator: DecodingSimulator):
   """Standard checks for every experiment, after simulation is complete."""
   # Check that simulation is actually done
   assert simulator._device_manager and simulator._window_manager and simulator._decoding_manager
   assert simulator.is_done()
   assert len(simulator._window_manager.window_builder._waiting_rounds) == 0
   assert len(window_data.all_constructed_windows) == len(set(window_data.all_constructed_windows))
   assert len(window_data.all_constructed_windows) == len(list(w for w in window_data.all_windows if w))
   assert all(window_data.get_window(window_idx).constructed for window_idx in window_data.all_constructed_windows)

   all_syndrome_rounds = [sr for sublist in device_data.generated_syndrome_data for sr in sublist]
   all_commit_regions = [cr for window in simulator._window_manager.all_windows if window for cr in window.commit_region]

   for instr_idx,instr in enumerate(simulator._device_manager.schedule_instructions):
      num_expected_syndromes = len(instr.instruction.patches) * simulator._device_manager._instruction_durations[instr_idx]
      for syndrome_round in all_syndrome_rounds:
         if syndrome_round.instruction == instr.instruction:
            num_expected_syndromes -= 1
      assert num_expected_syndromes == 0, f"Instruction {instr_idx+1}/{len(simulator._device_manager.schedule_instructions)} missing syndrome rounds"

   for i,syndrome_round in enumerate(all_syndrome_rounds):
      assert syndrome_round.instruction.name == 'INJECT_T' or any(cr.contains_syndrome_round(syndrome_round=syndrome_round) for cr in all_commit_regions), f"Syndrome round {i+1}/{len(all_syndrome_rounds)} not contained in any commit region"

   for window_idx in window_data.all_constructed_windows:
      window = window_data.get_window(window_idx)
      assert window.total_spacetime_volume() / distance <= 7

   dependencies_check(window_data)

def dependencies_check(window_data: WindowData):
   """Check that DAG is valid and consistent with window buffer overlaps.
   """
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   assert nx.is_directed_acyclic_graph(window_dag)
   assert set(window_dag.nodes) == set(window_data.all_constructed_windows)
   covered_edges = set()
   for i,window_idx_1 in enumerate(window_data.all_constructed_windows):
      window = window_data.get_window(window_idx_1)
      for window_idx_2 in window_data.all_constructed_windows[:i]:
         w = window_data.get_window(window_idx_2)
         assert not any(cr.overlaps(w_cr) for cr in window.commit_region for w_cr in w.commit_region), (window_idx_1, window_idx_2, window, w)
         if window.shares_boundary(w):
            assert any([cr.overlaps(buff) for cr in window.commit_region for buff in w.buffer_regions]) or any([cr.overlaps(buff) for cr in w.commit_region for buff in window.buffer_regions])
            assert (window_idx_1,window_idx_2) in window_dag.edges or (window_idx_2,window_idx_1) in window_dag.edges
            covered_edges.add((window_idx_1,window_idx_2) if (window_idx_1,window_idx_2) in window_dag.edges else (window_idx_2,window_idx_1))
         else:
            assert not any([cr.overlaps(buff) for cr in window.commit_region for buff in w.buffer_regions]) and not any([cr.overlaps(buff) for cr in w.commit_region for buff in window.buffer_regions])
            assert (window_idx_1,window_idx_2) not in window_dag.edges and (window_idx_2,window_idx_1) not in window_dag.edges
   assert covered_edges == set(window_dag.edges)

class WindowManagerTester(WindowManager):
   def process_round(self, new_rounds):
      pass

def test_window_manager():
   window_manager = WindowManagerTester(WindowBuilder(7))

def test_sliding_idle():
   """Test that SlidingWindowManager can correctly handle idle rounds."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=idle_schedule,
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
      do_intermediate_checks(simulator)

      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         if window and window_idx < len(simulator._window_manager.all_windows)-2:
            assert window.constructed
   
   # Check final windows
   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   device_rounds_covered = np.full(device_data.num_rounds, -1, dtype=int)
   do_final_checks(device_data, window_data, decoding_data, simulator)
   for i,window_idx in enumerate(window_data.all_constructed_windows):
      window = window_data.get_window(window_idx)
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) == 1
      cr = window.commit_region[0]
      if i < len(window_data.all_constructed_windows)-1:
         assert len(window.buffer_regions) == 1
         br = next(iter(window.buffer_regions))
         assert cr.round_start + cr.duration == br.round_start
      else:
         assert len(window.buffer_regions) == 0
      rounds_committed = np.arange(cr.round_start, cr.round_start + cr.duration)
      assert np.all(device_rounds_covered[rounds_committed] == -1)
      device_rounds_covered[rounds_committed] = i

   # TODO: when device is finished, need to flush final incomplete window so
   # that this test passes
   assert np.all(device_rounds_covered >= 0)

def test_parallel_idle():
   """Test that ParallelWindowManager can correctly handle idle rounds."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=idle_schedule,
      distance=distance,
      scheduling_method='parallel',
      decoding_latency_fn=decoding_fn,
      speculation_mode=speculation_mode,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      rng=0,
   )
   assert isinstance(simulator._window_manager, ParallelWindowManager)

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()
      do_intermediate_checks(simulator)
   
   # Check final windows
   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   source_indices = simulator._window_manager.layer_indices[1]
   sink_indices = simulator._window_manager.layer_indices[2]
   device_rounds_covered = np.full(device_data.num_rounds, -1, dtype=int)
   do_final_checks(device_data, window_data, decoding_data, simulator)
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   assert nx.is_bipartite(window_dag)
   assert len(list(nx.topological_generations(window_dag))) == 2
   last_window_idx = max(window_data.all_constructed_windows)
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      if i < last_window_idx:
         assert len(window.commit_region) == 1 or len(window.commit_region) == 3
         if len(window.commit_region) == 1:
            # source
            assert i in source_indices
            cr = window.commit_region[0]
            if i == 0:
               assert len(window.buffer_regions) == 1
               br = next(iter(window.buffer_regions))
               assert br.round_start == cr.round_start + cr.duration
               assert set(window_dag.predecessors(i)) == set()
               assert set(window_dag.successors(i)) == {i+1}
               assert i+1 in sink_indices
            else:
               assert len(window.buffer_regions) == 2
               br_1, br_2 = window.buffer_regions
               if br_2.round_start < br_1.round_start:
                  br_1, br_2 = br_2, br_1
               assert br_1.round_start + br_1.duration == cr.round_start
               assert cr.round_start + cr.duration == br_2.round_start
               assert set(window_dag.predecessors(i)) == set()
               assert set(window_dag.successors(i)) == {i-3, i+1}
               assert i-3 in sink_indices and i+1 in sink_indices
         else:
            # sink
            assert i in sink_indices
            cr_1, cr_2, cr_3 = window.commit_region
            assert len(window.buffer_regions) == 0
            assert cr_1.round_start + cr_1.duration == cr_2.round_start
            assert cr_2.round_start + cr_2.duration == cr_3.round_start

            assert set(window_dag.predecessors(i)) == {i-1, i+3}
            assert set(window_dag.successors(i)) == set()
            assert i-1 in source_indices and i+3 in source_indices
      else:
         assert len(window.buffer_regions) == 0 or len(window.commit_region) == 1
         if len(window.buffer_regions) == 0:
            # sink
            assert i in sink_indices
            assert 1 <= len(window.commit_region) <= 3
            assert set(window_dag.predecessors(i)) == {i-1}
            assert set(window_dag.successors(i)) == set()
            assert i-1 in source_indices
         else:
            # source
            assert i in source_indices
            assert len(window.commit_region) == 1
            assert set(window_dag.predecessors(i)) == set()
            assert set(window_dag.successors(i)) == {i-3}
            assert i-3 in sink_indices
      rounds_committed = np.concatenate([range(cr.round_start, cr.round_start + cr.duration) for cr in window.commit_region])
      assert np.all(device_rounds_covered[rounds_committed] == -1)
      device_rounds_covered[rounds_committed] = i

   # TODO: when device is finished, need to flush final incomplete window so
   # that this test passes
   assert np.all(device_rounds_covered >= 0)

def test_sliding_merge():
   """Test that SlidingWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=merge_schedule,
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
      do_intermediate_checks(simulator)

   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   do_final_checks(device_data, window_data, decoding_data, simulator)
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) == 1
      cr = window.commit_region[0]
      merge_instr = window.merge_instr
      assert 0 <= len(window.buffer_regions) <= 3
      if len(merge_instr) == 2:
         assert cr.patch == (0,0) or cr.patch == (0,10)
      # Check that only connections between the two merges are on the ends
      if cr.patch not in [(0,0), (0,10)]:
         assert len(merge_instr) == 1
         # Commit region is somewhere in merge ancilla region
         dag_neighbors = set(window_dag.successors(i)) | set(window_dag.predecessors(i))
         assert len(dag_neighbors) == 2
         for neighbor in dag_neighbors:
            neighbor_window = window_data.get_window(neighbor)
            cr_n = neighbor_window.commit_region[0]
            assert abs(cr_n.patch[1] - cr.patch[1]) == 1
            merges = neighbor_window.merge_instr
            assert merges == merge_instr or cr_n.patch[1] in [0, 10]

def test_parallel_merge():
   """Test that ParallelWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=merge_schedule,
      distance=distance,
      scheduling_method='parallel',
      decoding_latency_fn=decoding_fn,
      speculation_mode=speculation_mode,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      rng=0,
   )
   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      do_intermediate_checks(simulator)

   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   do_final_checks(device_data, window_data, decoding_data, simulator)
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   assert len(list(nx.topological_generations(window_dag))) == 3
   
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      # TODO: more tests here
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) <= 3

def test_sliding_distillation():
   """Test that SlidingWindowManager can correctly handle the 15-to-1
   distillation schedule."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=MSD15To1Schedule().schedule,
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
      do_intermediate_checks(simulator)

   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   do_final_checks(device_data, window_data, decoding_data, simulator)
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) == 1
      cr = window.commit_region[0]
      merge_instr = window.merge_instr

def test_parallel_distillation():
   """Test that ParallelWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=MSD15To1Schedule().schedule,
      distance=distance,
      scheduling_method='parallel',
      decoding_latency_fn=decoding_fn,
      speculation_mode=speculation_mode,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      rng=0,
   )

   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      do_intermediate_checks(simulator)

   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   do_final_checks(device_data, window_data, decoding_data, simulator)
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   assert len(list(nx.topological_generations(window_dag))) == 3
   
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) <= 3

def test_aligned_randomT():
   """Test that TAlignedWindowManager can correctly handle a RandomT schedule."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=RandomTSchedule(10, 50).schedule,
      distance=distance,
      scheduling_method='aligned',
      decoding_latency_fn=decoding_fn,
      speculation_mode=speculation_mode,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      rng=0,
   )
   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      do_intermediate_checks(simulator)

   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   do_final_checks(device_data, window_data, decoding_data, simulator)
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   assert len(list(nx.topological_generations(window_dag))) == 3
   
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      if window.merge_instr:
         assert simulator._window_manager._get_layer_idx(i) in [2,3]
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) <= 3

def test_aligned_distillation():
   """Test that TAlignedWindowManager can correctly handle a distillation schedule."""
   simulator = DecodingSimulator()
   simulator.initialize_experiment(
      schedule=MSD15To1Schedule().schedule,
      distance=distance,
      scheduling_method='aligned',
      decoding_latency_fn=decoding_fn,
      speculation_mode=speculation_mode,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      rng=0,
   )

   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      do_intermediate_checks(simulator)

   success, _, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   do_final_checks(device_data, window_data, decoding_data, simulator)
   window_dag = nx.DiGraph(window_data.window_dag_edges)
   assert len(list(nx.topological_generations(window_dag))) == 4
   
   for i in window_data.all_constructed_windows:
      window = window_data.get_window(i)
      if any(instr.conditional_dependencies for instr in window.merge_instr):
         assert simulator._window_manager._get_layer_idx(i) in [2, 3]
      assert simulator._window_manager._all_regions_touching(list(window.commit_region))
      assert len(window.commit_region) <= 3

# def test_parallel_regularT():
#    """Test that ParallelWindowManager can correctly handle a merge schedule."""
#    simulator = DecodingSimulator(
#       distance=15,
#       decoding_latency_fn=lambda _: 20,
#       speculation_latency=2,
#       speculation_accuracy=0.99,
#       speculation_mode='integrated',
#    )
#    simulator.initialize_experiment(RegularTSchedule(95, 3).schedule, 'parallel', rng=0)
#    assert simulator._window_manager

#    while not simulator.is_done():
#       simulator.step_experiment()
#       do_intermediate_checks(simulator)

#    success, device_data, window_data, decoding_data = simulator.get_data()
#    assert success
#    do_final_checks(device_data, window_data, decoding_data, simulator)
      
if __name__ == '__main__':
   test_sliding_merge()
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
from swiper2.simulator import DecodingSimulator
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper2.window_builder import WindowBuilder
from swiper2.window_manager import WindowManager, WindowData, SlidingWindowManager, ParallelWindowManager, TAlignedWindowManager
from swiper2.schedule_experiments import MSD15To1Schedule, RandomTSchedule

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
speculation_accuracy = 0
speculation_mode = 'integrated'

def check_dependencies(window_data: WindowData):
   """Check that DAG is valid and consistent with window buffer overlaps.
   """
   assert nx.is_directed_acyclic_graph(window_data.window_dag)
   assert set(window_data.window_dag.nodes) == set(range(len(window_data.all_windows)))
   covered_edges = set()
   for i,window in enumerate(window_data.all_windows):
      for j,w in enumerate(window_data.all_windows[:i]):
         assert not any(cr.overlaps(w_cr) for cr in window.commit_region for w_cr in w.commit_region) 
         if window.shares_boundary(w):
            assert any([cr.overlaps(buff) for cr in window.commit_region for buff in w.buffer_regions]) or any([cr.overlaps(buff) for cr in w.commit_region for buff in window.buffer_regions])
            assert (i,j) in window_data.window_dag.edges or (j,i) in window_data.window_dag.edges
            covered_edges.add((i,j) if (i,j) in window_data.window_dag.edges else (j,i))
         else:
            assert not any([cr.overlaps(buff) for cr in window.commit_region for buff in w.buffer_regions]) and not any([cr.overlaps(buff) for cr in w.commit_region for buff in window.buffer_regions])
            assert (i,j) not in window_data.window_dag.edges and (j,i) not in window_data.window_dag.edges
   assert covered_edges == set(window_data.window_dag.edges)

class WindowManagerTester(WindowManager):
   def process_round(self, new_rounds):
      pass

def test_window_manager():
   window_manager = WindowManagerTester(WindowBuilder(7, False))

def test_sliding_idle():
   """Test that SlidingWindowManager can correctly handle idle rounds."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(idle_schedule, 'sliding', False, rng=0)
   assert simulator._window_manager

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()

      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait
         if window_idx < len(simulator._window_manager.all_windows)-2:
            assert window.constructed
   
   # Check final windows
   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   device_rounds_covered = np.full(device_data.num_rounds, -1, dtype=int)
   check_dependencies(window_data)
   for i,window in enumerate(window_data.all_windows):
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) == 1
      cr = window.commit_region[0]
      if i < len(window_data.all_windows)-1:
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
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(idle_schedule, 'parallel', False, rng=0)
   assert isinstance(simulator._window_manager, ParallelWindowManager)

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()

      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait
   
   # Check final windows
   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   layer1_indices = simulator._window_manager.layer_indices[1]
   layer2_indices = simulator._window_manager.layer_indices[2]
   device_rounds_covered = np.full(device_data.num_rounds, -1, dtype=int)
   check_dependencies(window_data)
   assert nx.is_bipartite(window_data.window_dag)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 2
   for i,window in enumerate(window_data.all_windows):
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      if i < len(window_data.all_windows)-1:
         assert len(window.commit_region) == 1 or len(window.commit_region) == 3
         if len(window.commit_region) == 1:
            # source
            assert i in layer1_indices
            cr = window.commit_region[0]
            if i == 0:
               assert len(window.buffer_regions) == 1
               br = next(iter(window.buffer_regions))
               assert br.round_start == cr.round_start + cr.duration
               assert set(window_data.window_dag.predecessors(i)) == set()
               assert set(window_data.window_dag.successors(i)) == {i+1}
               assert i+1 in layer2_indices
            else:
               assert len(window.buffer_regions) == 2
               br_1, br_2 = window.buffer_regions
               if br_2.round_start < br_1.round_start:
                  br_1, br_2 = br_2, br_1
               assert br_1.round_start + br_1.duration == cr.round_start
               assert cr.round_start + cr.duration == br_2.round_start
               assert set(window_data.window_dag.predecessors(i)) == set()
               assert set(window_data.window_dag.successors(i)) == {i-1, i+1}
               assert i-1 in layer2_indices and i+1 in layer2_indices
         else:
            # sink
            assert i in layer2_indices
            cr_1, cr_2, cr_3 = window.commit_region
            assert len(window.buffer_regions) == 0
            assert cr_1.round_start + cr_1.duration == cr_2.round_start
            assert cr_2.round_start + cr_2.duration == cr_3.round_start

            assert set(window_data.window_dag.predecessors(i)) == {i-1, i+1}
            assert set(window_data.window_dag.successors(i)) == set()
            assert i-1 in layer1_indices and i+1 in layer1_indices
      else:
         assert len(window.buffer_regions) == 0 or len(window.commit_region) == 1
         if len(window.buffer_regions) == 0:
            # sink
            assert i in layer2_indices
            assert 1 <= len(window.commit_region) <= 3
            assert set(window_data.window_dag.predecessors(i)) == {i-1}
            assert set(window_data.window_dag.successors(i)) == set()
            assert i-1 in layer1_indices
         else:
            # source
            assert i in layer1_indices
            assert len(window.commit_region) == 1
            assert set(window_data.window_dag.predecessors(i)) == set()
            assert set(window_data.window_dag.successors(i)) == {i-1}
            assert i-1 in layer2_indices
      rounds_committed = np.concatenate([range(cr.round_start, cr.round_start + cr.duration) for cr in window.commit_region])
      assert np.all(device_rounds_covered[rounds_committed] == -1)
      device_rounds_covered[rounds_committed] = i

   # TODO: when device is finished, need to flush final incomplete window so
   # that this test passes
   assert np.all(device_rounds_covered >= 0)

def test_sliding_merge():
   """Test that SlidingWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(merge_schedule, 'sliding', False, rng=0)
   assert simulator._window_manager

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   check_dependencies(window_data)
   for i,window in enumerate(window_data.all_windows):
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) == 1
      cr = window.commit_region[0]
      merge_instr = window.merge_instr
      if len(merge_instr) == 2:
         assert cr.patch == (0,0) or cr.patch == (0,10)
      # Check that only connections between the two merges are on the ends
      if cr.patch not in [(0,0), (0,10)]:
         assert len(merge_instr) == 1
         # Commit region is somewhere in merge ancilla region
         dag_neighbors = set(window_data.window_dag.successors(i)) | set(window_data.window_dag.predecessors(i))
         print((cr.patch, cr.round_start), [(window_data.all_windows[neighbor].commit_region[0].patch, window_data.all_windows[neighbor].commit_region[0].round_start) for neighbor in dag_neighbors])
         assert len(dag_neighbors) == 2
         for neighbor in dag_neighbors:
            cr_n = window_data.all_windows[neighbor].commit_region[0]
            assert abs(cr_n.patch[1] - cr.patch[1]) == 1
            merges = window_data.all_windows[neighbor].merge_instr
            assert merges == merge_instr or cr_n.patch[1] in [0, 10]

def test_parallel_merge():
   """Test that ParallelWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(merge_schedule, 'parallel', False, rng=0)
   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   check_dependencies(window_data)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 3
   
   for i,window in enumerate(window_data.all_windows):
      # TODO: more tests here
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) <= 3

def test_sliding_distillation():
   """Test that SlidingWindowManager can correctly handle the 15-to-1
   distillation schedule."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(MSD15To1Schedule().schedule, 'sliding', False, rng=0)
   assert simulator._window_manager
 
   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   check_dependencies(window_data)
   for i,window in enumerate(window_data.all_windows):
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) == 1
      cr = window.commit_region[0]
      merge_instr = window.merge_instr

def test_parallel_distillation():
   """Test that ParallelWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(MSD15To1Schedule().schedule, 'parallel', False, rng=0)
   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   check_dependencies(window_data)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 3
   
   for i,window in enumerate(window_data.all_windows):
      # TODO: more tests here
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) <= 6

def test_aligned():
   """Test that TAlignedWindowManager can correctly handle a RandomT schedule."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(RandomTSchedule(10, 50).schedule, 'aligned', False, rng=0)
   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   check_dependencies(window_data)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 3
   
   for i,window in enumerate(window_data.all_windows):
      # TODO: more tests here
      if window.merge_instr:
         assert simulator._window_manager._get_layer_idx(i) == 2
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) <= 3

def test_aligned_distillation():
   """Test that TAlignedWindowManager can correctly handle a distillation schedule."""
   simulator = DecodingSimulator(
      distance=distance,
      decoding_latency_fn=decoding_fn,
      speculation_latency=speculation_latency,
      speculation_accuracy=speculation_accuracy,
      speculation_mode=speculation_mode,
   )
   simulator.initialize_experiment(MSD15To1Schedule().schedule, 'aligned', False, rng=0)
   assert simulator._window_manager

   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   success, device_data, window_data, decoding_data = simulator.get_data()
   assert success
   check_dependencies(window_data)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 4
   
   for i,window in enumerate(window_data.all_windows):
      # TODO: more tests here
      if any(instr.conditional_dependencies for instr in window.merge_instr):
         assert simulator._window_manager._get_layer_idx(i) == 2
      assert simulator._window_manager._is_contiguous(list(window.commit_region))
      assert len(window.commit_region) <= 6
      
if __name__ == '__main__':
   test_parallel_merge()
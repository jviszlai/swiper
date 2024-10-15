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

def test_sliding_idle():
   """Test that SlidingWindowManager can correctly handle idle rounds."""
   simulator = DecodingSimulator(7, lambda _: 14, 2, 0, speculation_mode='separate')
   simulator.initialize_experiment(idle_schedule, 'sliding', False)

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()

      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait
         if window_idx < len(simulator._window_manager.all_windows)-2:
            assert window.constructed
   
   # Check final windows
   device_data, window_data, decoding_data = simulator.get_data()
   device_rounds_covered = np.full(device_data.num_rounds, -1, dtype=int)
   assert nx.is_directed_acyclic_graph(window_data.window_dag)
   assert set(window_data.window_dag.nodes) == set(range(len(window_data.all_windows)))
   for i,window in enumerate(window_data.all_windows):
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
   simulator = DecodingSimulator(7, lambda _: 14, 2, 0, speculation_mode='separate')
   simulator.initialize_experiment(idle_schedule, 'parallel', False)

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()

      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait
   
   # Check final windows
   device_data, window_data, decoding_data = simulator.get_data()
   source_indices = simulator._window_manager.source_indices
   sink_indices = simulator._window_manager.sink_indices
   device_rounds_covered = np.full(device_data.num_rounds, -1, dtype=int)
   assert nx.is_directed_acyclic_graph(window_data.window_dag)
   assert set(window_data.window_dag.nodes) == set(range(len(window_data.all_windows)))
   assert nx.is_bipartite(window_data.window_dag)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 2
   for i,window in enumerate(window_data.all_windows):
      if i < len(window_data.all_windows)-1:
         assert len(window.commit_region) == 1 or len(window.commit_region) == 3
         if len(window.commit_region) == 1:
            # source
            assert i in source_indices
            cr = window.commit_region[0]
            if i == 0:
               assert len(window.buffer_regions) == 1
               br = next(iter(window.buffer_regions))
               assert br.round_start == cr.round_start + cr.duration
               assert set(window_data.window_dag.predecessors(i)) == set()
               assert set(window_data.window_dag.successors(i)) == {i+1}
               assert i+1 in sink_indices
            else:
               assert len(window.buffer_regions) == 2
               br_1, br_2 = window.buffer_regions
               if br_2.round_start < br_1.round_start:
                  br_1, br_2 = br_2, br_1
               assert br_1.round_start + br_1.duration == cr.round_start
               assert cr.round_start + cr.duration == br_2.round_start
               assert set(window_data.window_dag.predecessors(i)) == set()
               assert set(window_data.window_dag.successors(i)) == {i-1, i+1}
               assert i-1 in sink_indices and i+1 in sink_indices
         else:
            # sink
            assert i in sink_indices
            cr_1, cr_2, cr_3 = window.commit_region
            assert len(window.buffer_regions) == 0
            assert cr_1.round_start + cr_1.duration == cr_2.round_start
            assert cr_2.round_start + cr_2.duration == cr_3.round_start

            assert set(window_data.window_dag.predecessors(i)) == {i-1, i+1}
            assert set(window_data.window_dag.successors(i)) == set()
            assert i-1 in source_indices and i+1 in source_indices
      else:
         assert len(window.buffer_regions) == 0 or len(window.commit_region) == 1
         if len(window.buffer_regions) == 0:
            # sink
            assert i in sink_indices
            assert 1 <= len(window.commit_region) <= 3
            assert set(window_data.window_dag.predecessors(i)) == {i-1}
            assert set(window_data.window_dag.successors(i)) == set()
            assert i-1 in source_indices
         else:
            # source
            assert i in source_indices
            assert len(window.commit_region) == 1
            assert set(window_data.window_dag.predecessors(i)) == set()
            assert set(window_data.window_dag.successors(i)) == {i-1}
            assert i-1 in sink_indices
      rounds_committed = np.concatenate([range(cr.round_start, cr.round_start + cr.duration) for cr in window.commit_region])
      assert np.all(device_rounds_covered[rounds_committed] == -1)
      device_rounds_covered[rounds_committed] = i
   # TODO: when device is finished, need to flush final incomplete window so
   # that this test passes
   assert np.all(device_rounds_covered >= 0)

def test_sliding_merge():
   """Test that SlidingWindowManager can correctly handle a merge schedule."""
   simulator = DecodingSimulator(7, lambda _: 14, 2, 0, speculation_mode='separate')
   simulator.initialize_experiment(merge_schedule, 'sliding', False)

   # Check windows as they are released
   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   device_data, window_data, decoding_data = simulator.get_data()
   assert nx.is_directed_acyclic_graph(window_data.window_dag)
   assert set(window_data.window_dag.nodes) == set(range(len(window_data.all_windows)))
   for i,window in enumerate(window_data.all_windows):
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
   simulator = DecodingSimulator(7, lambda _: 14, 2, 0, speculation_mode='separate')
   simulator.initialize_experiment(merge_schedule, 'parallel', False)

   while not simulator.is_done():
      simulator.step_experiment()
      for window_idx, window in enumerate(simulator._window_manager.all_windows):
         assert window.constructed or window_idx in simulator._window_manager.window_buffer_wait

   device_data, window_data, decoding_data = simulator.get_data()
   assert nx.is_directed_acyclic_graph(window_data.window_dag)
   assert set(window_data.window_dag.nodes) == set(range(len(window_data.all_windows)))
   assert nx.is_bipartite(window_data.window_dag)
   assert len(list(nx.topological_generations(window_data.window_dag))) == 2
   
   # # Check that no source-source or sink-sink conflicts exist
   # for window_idx_1, window_1 in enumerate(window_data.all_windows):
   #    for window_idx_2, window_2 in enumerate(window_data.all_windows[:window_idx_1]):
   #       for patch1 in [cr.patch for cr in window_1.commit_region]:
   #             for patch2 in [cr.patch for cr in window_2.commit_region]:
   #                if (abs(patch1[0] - patch2[0]) + abs(patch1[1] - patch2[1]) == 1):
   #                   assert (window_idx_1 in simulator._window_manager.sink_indices and window_idx_2 in simulator._window_manager.source_indices) or (window_idx_1 in simulator._window_manager.source_indices and window_idx_2 in simulator._window_manager.sink_indices)

if __name__ == '__main__':
   test_parallel_merge()
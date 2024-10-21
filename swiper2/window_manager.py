"""

Ideas for performance optimization (if needed later):
- window_idx_dict: dict[DecodingWindow, int] to avoid calling index() on all_windows
- Priority queue for buffer wait
"""
from abc import ABC, abstractmethod
import networkx as nx
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from swiper2.window_builder import WindowBuilder, DecodingWindow, SpacetimeRegion
from swiper2.device_manager import SyndromeRound
from swiper2.lattice_surgery_schedule import Instruction

@dataclass
class WindowData:
    '''
    Data structure to hold information about windows
    '''
    all_windows: list[DecodingWindow]
    window_dag: nx.DiGraph
    window_count_history: NDArray[np.int_]

class WindowManager(ABC):

    def __init__(self, window_builder: WindowBuilder):
        self.all_windows: list[DecodingWindow] = []
        self.window_builder = window_builder
        self.window_dag = nx.DiGraph()
        self.window_end_lookup: dict[tuple[tuple[int, int], int], tuple[int, int]] = {}
        self.window_buffer_wait: dict[int, int] = {}
        self._window_count_history: list[int] = []

    @abstractmethod
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        """Process new syndrome rounds and update the decoding window dependency
        graph as needed.
        """
        raise NotImplementedError

    def pending_instruction_indices(self) -> set[int]:
        """Get the set of instruction indices that are currently generating
        windows.
        """
        return set([instr_idx for window in self.all_windows for instr_idx in window.parent_instr_idx if not window.constructed])

    def get_unconstructed_windows(self):
        return [i for i, window in enumerate(self.all_windows) if not window.constructed]

    def _add_window(self, window: DecodingWindow) -> None:
        """TODO"""

    def _remove_window(self, window: DecodingWindow) -> None:
        """TODO"""

    def _replace_window(self, window: DecodingWindow, new_window: DecodingWindow) -> None:
        """TODO"""

    def _is_contiguous(self, regions: list[SpacetimeRegion]):
        connectivity = []
        for i,cr1 in enumerate(regions):
            for j,cr2 in enumerate(regions):
                if i != j and cr1.shares_boundary(cr2):
                    connectivity.append((i,j))
        g = nx.Graph()
        g.add_nodes_from(range(len(regions)))
        g.add_edges_from(connectivity)
        return nx.is_connected(g)
    
    def _get_adjacent_unconstructed_window_indices(self, window: DecodingWindow) -> list[int]:
        """Get all unconstructed windows that share a boundary with window.
        """
        adjacent_windows = []
        for j, other_window in enumerate(self.all_windows):
            if not other_window.constructed and window.shares_boundary(other_window):
                adjacent_windows.append(j)
        return adjacent_windows

    def _merge_windows(
            self,
            window_1: DecodingWindow,
            window_2: DecodingWindow,
            enforce_contiguous: bool = True,
        ) -> DecodingWindow:
        """Merge two windows into a new window. Removes window_2 from
        all_windows.

        WARNING: like _append_to_buffers or _mark_constructed, this method
        modifies all_windows, window_dag, window_buffer_wait, and other internal
        data structures.

        After merging, the entry for window_1 in self.all_windows will be
        updated to contain the combined commit region and buffer regions.
        window_2 will be removed from self.all_windows.

        Args:
            window_1: Main window
            window_2: Window to merge into window_1
            enforce_contiguous: If True, the commit regions of window_1 and
                window_2 must be contiguous in spacetime.
        
        Returns:
            new_window: The new window created by merging window_1 and window_2.
                Note that this window will already be in self.all_windows.
        """
        assert window_1.constructed == window_2.constructed == False
        window_idx_1 = self.all_windows.index(window_1)
        window_idx_2 = self.all_windows.index(window_2)

        if enforce_contiguous:
            if not self._is_contiguous(list(window_1.commit_region) + list(window_2.commit_region)):
                raise ValueError("Commit regions must be contiguous")

        # Add window 2's attributes to window 1
        for succ in self.window_dag.successors(window_idx_2):
            if succ != window_idx_1:
                self.window_dag.add_edge(window_idx_1, succ)
        for pred in self.window_dag.predecessors(window_idx_2):
            if pred != window_idx_1:
                self.window_dag.add_edge(pred, window_idx_1)
        if not (window_1.constructed and window_2.constructed):
            self.window_buffer_wait[window_idx_1] = max(self.window_buffer_wait.get(window_idx_1, 0), self.window_buffer_wait.get(window_idx_2, 0))
        for k,(w_idx,cr_idx) in self.window_end_lookup.items():
            if w_idx == window_idx_2:
                self.window_end_lookup[k] = (window_idx_1, len(window_1.commit_region)+cr_idx)

        # Construct new window and replace window_1 in all_windows
        new_window = DecodingWindow(
            tuple(list(window_1.commit_region) + list(window_2.commit_region)),
            window_1.buffer_regions | window_2.buffer_regions,
            window_1.merge_instr | window_2.merge_instr,
            window_1.parent_instr_idx | window_2.parent_instr_idx,
            window_1.constructed and window_2.constructed,
        )
        self.all_windows[window_idx_1] = new_window

        # Remove window_2 from all_windows, and update window indices
        self.all_windows.pop(window_idx_2)
        self.window_dag.remove_node(window_idx_2)
        self.window_dag = nx.relabel_nodes(self.window_dag, {i:(i-1 if i > window_idx_2 else i) for i in self.window_dag.nodes})
        if window_idx_2 in self.window_buffer_wait:
            self.window_buffer_wait.pop(window_idx_2)
        self.window_buffer_wait = {(i-1 if i > window_idx_2 else i):wait for i,wait in self.window_buffer_wait.items()}
        for k,(w_idx,cr_idx) in self.window_end_lookup.items():
            if w_idx > window_idx_2:
                self.window_end_lookup[k] = (w_idx-1, cr_idx)
        return new_window

    # def _split_windows(
    #         self,
    #         window: DecodingWindow,
    #         source_commit_regions: list[SpacetimeRegion],
    #         sink_commit_regions: list[SpacetimeRegion],
    #         enforce_contiguous: bool = True,
    #     ) -> tuple[DecodingWindow, DecodingWindow]:
    #     """Split a window into two (or more!) windows that each contain some of the commit
    #     regions of the original window. The original window is removed from
    #     all_windows, and the new windows are added.

    #     Buffer dependency relationships of the original window are preserved in
    #     the new windows. The first new window will be a parent ("source") of the
    #     second new window ("sink").

    #     Args:
    #         window: Window to split
    #         source_commit_regions: Commit regions for the first new window.
    #         sink_commit_regions: Commit regions for the second new window (the
    #             sink). This window will be a dependency of the first new window.
    #         enforce_contiguous: If True, the source and sink commit regions must
    #             be contiguous in spacetime. If False, the source commit regions
    #             can be non-contiguous.
    #     """
    #     window_idx = self.all_windows.index(window)
    #     assert not window.constructed
    #     assert set(source_commit_regions) | set(sink_commit_regions) == set(window.commit_region)
    #     assert len(source_commit_regions) > 0
    #     assert len(sink_commit_regions) > 0

    #     if enforce_contiguous:
    #         assert self._is_contiguous(list(window.commit_region))
    #         if not self._is_contiguous(source_commit_regions):
    #             raise ValueError("Source commit regions must be contiguous")
    #         if not self._is_contiguous(sink_commit_regions):
    #             raise ValueError("Sink commit regions must be contiguous")

    #     source_buffers = set()
    #     sink_buffers = set()
    #     for region in window.buffer_regions:
    #         if any(region.shares_boundary(cr) for cr in source_commit_regions):
    #             source_buffers.add(region)
    #         if any(region.shares_boundary(cr) for cr in sink_commit_regions):
    #             sink_buffers.add(region)
    #     for source_commit in source_commit_regions:
    #         for sink_commit in sink_commit_regions:
    #             if source_commit.shares_boundary(sink_commit):
    #                 source_buffers.add(sink_commit)
    #     assert source_buffers.intersection(sink_commit_regions)

    #     # Construct new windows
    #     source_window = DecodingWindow(
    #         tuple(source_commit_regions),
    #         frozenset(source_buffers),
    #         window.merge_instr,
    #         window.parent_instr_idx,
    #         window.constructed,
    #     )
    #     sink_window = DecodingWindow(
    #         tuple(sink_commit_regions),
    #         frozenset(sink_buffers),
    #         window.merge_instr,
    #         window.parent_instr_idx,
    #         window.constructed,
    #     )

    #     predecessors = list(self.window_dag.predecessors(window_idx))
    #     successors = list(self.window_dag.successors(window_idx))
    #     predecessors_source = []
    #     predecessors_sink = []
    #     successors_source = []
    #     successors_sink = []
    #     for node in predecessors:
    #         w = self.all_windows[node]
    #         if source_window.shares_boundary(w):
    #             predecessors_source.append(node)
    #         if sink_window.shares_boundary(w):
    #             predecessors_sink.append(node)
    #     for node in successors:
    #         w = self.all_windows[node]
    #         if source_window.shares_boundary(w):
    #             successors_source.append(node)
    #         if sink_window.shares_boundary(w):
    #             successors_sink.append(node)

    #     # Add new windows to all_windows and update data structures
    #     self.all_windows[window_idx] = source_window
    #     self.window_dag.remove_edges_from([(p, window_idx) for p in predecessors if p not in predecessors_source])
    #     self.window_dag.remove_edges_from([(window_idx, s) for s in successors if s not in successors_source])
        
    #     self.all_windows.append(sink_window)
    #     sink_idx = len(self.all_windows) - 1
    #     self.window_dag.add_edges_from([(p, sink_idx) for p in predecessors_sink])
    #     self.window_dag.add_edges_from([(sink_idx, s) for s in successors_sink])
    #     self.window_buffer_wait[sink_idx] = self.window_buffer_wait[window_idx]
        
    #     self.window_dag.add_edge(window_idx, sink_idx)

    #     source_cr_indices = [i for i,cr in enumerate(window.commit_region) if cr in source_commit_regions]
    #     sink_cr_indices = [i for i,cr in enumerate(window.commit_region) if cr in sink_commit_regions]
    #     for k,(w_idx,cr_idx) in self.window_end_lookup.items():
    #         if w_idx == window_idx:
    #             if cr_idx in sink_cr_indices:
    #                 self.window_end_lookup[k] = (sink_idx, sink_commit_regions.index(window.commit_region[cr_idx]))
    #             else:
    #                 assert cr_idx in source_cr_indices
    #                 self.window_end_lookup[k] = (window_idx, source_commit_regions.index(window.commit_region[cr_idx]))
    #     return source_window, sink_window

    def _append_to_buffers(self, window: DecodingWindow, region: SpacetimeRegion, constructed: bool=False, inplace=False) -> DecodingWindow:
        """Create new window with region appended to buffer regions
        """
        if region in window.buffer_regions:
            return window
        assert not window.constructed
        new_window = DecodingWindow(window.commit_region, 
                              window.buffer_regions | frozenset([region]), 
                              window.merge_instr, 
                              window.parent_instr_idx,
                              window.constructed | constructed)
        if inplace:
            window_idx = self.all_windows.index(window)
            self.all_windows[window_idx] = new_window
        return new_window

    def _mark_constructed(self, window: DecodingWindow, inplace=False) -> DecodingWindow:
        """Create new window marked constructed to indicate it is ready to be
        decoded.
        """
        window_idx = self.all_windows.index(window)
        if inplace:
            if window_idx in self.window_buffer_wait:
                self.window_buffer_wait.pop(window_idx)
        if window.constructed:
            return window
        new_window = DecodingWindow(window.commit_region,
                              window.buffer_regions,
                              window.merge_instr,
                              window.parent_instr_idx,
                              True)
        if inplace:
            self.all_windows[window_idx] = new_window
        return new_window
    
    def _update_buffer_wait(self) -> None:
        """Look for dangling windows to mark as constructed.

        TODO: might be better to change how this is done. Each window could
        instead be marked as "waiting for more commit regions", "waiting for
        more buffer regions", or "waiting for predecessor to be generated". We
        could then release the window once all conditions are met.
        """
        constructed_windows = set()
        for window_idx in self.window_buffer_wait.keys():
            self.window_buffer_wait[window_idx] -= 1
            if self.window_buffer_wait[window_idx] <= 0:
                constructed_windows.add(self.all_windows[window_idx])
        for window in constructed_windows:
            self._mark_constructed(window, inplace=True)

    def get_data(self) -> WindowData:
        return WindowData(
            all_windows=self.all_windows,
            window_dag=self.window_dag,
            window_count_history=np.array(self._window_count_history, int),
        )

class SlidingWindowManager(WindowManager):
    
    def process_round(self, new_rounds: list[SyndromeRound], discarded_patches: list[tuple[int, int]]) -> None:
        if new_rounds:
            new_commits = self.window_builder.build_windows(new_rounds, discarded_patches)
        else:
            new_commits = self.window_builder.flush()
        
        if new_commits:
            new_window_start = len(self.all_windows)

            # Add new windows
            self.all_windows.extend(new_commits)
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                patch = window.commit_region[0].patch
                end = window.commit_region[0].round_start + window.commit_region[0].duration
                assert (patch, end) not in self.window_end_lookup
                self.window_end_lookup[(patch, end)] = (window_idx, 0)
                self.window_dag.add_node(window_idx)
                self.window_buffer_wait[new_window_start + i] = self.window_builder.d+1
            
            # Process buffers in time (windows covering same patch)
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                patch = window.commit_region[0].patch
                prev_window_end = window.commit_region[0].round_start
                if (patch, prev_window_end) in self.window_end_lookup:
                    prev_window_idx,_ = self.window_end_lookup[(patch, prev_window_end)]
                    prev_window = self.all_windows[prev_window_idx]
                    prev_commit = prev_window.commit_region[0]
                    if prev_commit.discard_after:
                        continue
                    # For sliding window, buffers extend at most one step forward in time
                    # Mark prev window as constructed and ready to decode
                    # TODO: buffer has to be d_m cycles "tall", so we may
                    # need to add more than just the next commit region.
                    # Implementation: do not ever mark this as constructed;
                    # let this happen in update_buffer_wait, and only when
                    # the buffer is tall enough. If buffer exists but is not
                    # tall enough, should upadte_buffer_wait add more to it
                    # every round?
                    assert not prev_window.constructed
                    prev_window = self._append_to_buffers(prev_window, window.commit_region[0], inplace=True)
                    self.window_dag.add_edge(prev_window_idx, window_idx)

            # Process buffers in space (windows covering same MERGE instruction)
            merge_windows = {}
            for i, window in enumerate(new_commits):
                for m_i in window.merge_instr:
                    merge_windows.setdefault(m_i, []).append(new_window_start + i)
            for _, window_idxs in merge_windows.items():
                for i, window_idx_1 in enumerate(window_idxs):
                    for window_idx_2 in window_idxs[i + 1:]:
                        patch1 = self.all_windows[window_idx_1].commit_region[0].patch
                        patch2 = self.all_windows[window_idx_2].commit_region[0].patch
                        if abs(patch1[0] - patch2[0]) + abs(patch1[1] - patch2[1]) == 1:
                            # Adjacent, merged patches
                            assert not self.all_windows[window_idx_1].constructed
                            # Spatial alignment will always be correct (i.e.
                            # commit region will always be wide enough), so we
                            # can just append the neighboring commit region.
                            self._append_to_buffers(self.all_windows[window_idx_1], self.all_windows[window_idx_2].commit_region[0], inplace=True)
                            self.window_dag.add_edge(window_idx_1, window_idx_2)

        for window_idx, window in enumerate(self.all_windows):
            assert window.constructed or window_idx in self.window_buffer_wait

        self._update_buffer_wait()
        self._window_count_history.append(len(self.all_windows))

        if not new_rounds:
            # No new rounds; flush any dangling windows
            for window_idx in list(self.window_buffer_wait.keys()):
                window = self.all_windows[window_idx]
                assert not window.constructed
                self._mark_constructed(window, inplace=True)
            assert all(window.constructed for window in self.all_windows)
        
class ParallelWindowManager(WindowManager):
    """TODO
    
    Ideally, a source contains one commit region and some buffer regions
    surrounding it. A sink contains three commit regions in a line, where the
    start and end are buffered by neighboring sources. However, this gets
    complicated when we have dense merge schedules and more pipe junctions.

    Addition: there are two layers of sources. By default, we only use the
    second source layer and the sinks, but in a big merge operation, we can 
    """
    layer_indices: list[set[int]]

    def __init__(self, window_builder: WindowBuilder):
        self.layer_indices = [set(), set(), set()]
        super().__init__(window_builder)

    def process_round(self, new_rounds: list[SyndromeRound], discarded_patches: list[tuple[int, int]]) -> None:
        if new_rounds:
            new_commits = self.window_builder.build_windows(new_rounds, discarded_patches)
        else:
            new_commits = self.window_builder.flush()

        if new_commits:
            self.all_windows.extend(new_commits)
            for window in new_commits:
                window_idx = self.all_windows.index(window)
                assert len(window.commit_region) == 1
                cr = window.commit_region[0]
                patch, end = cr.patch, cr.round_start + cr.duration
                assert (patch, end) not in self.window_end_lookup
                self.window_end_lookup[(patch, end)] = (window_idx, 0)
                self.window_dag.add_node(window_idx)
                if cr.discard_after:
                    # self.window_buffer_wait[window_idx] = 1 # should this be
                    # 0?
                    self.window_buffer_wait[window_idx] = self.window_builder.d+1
                else:
                    self.window_buffer_wait[window_idx] = self.window_builder.d+1

            # Process buffers in time (windows covering same patch)
            # Make soft decisions to assign each window as a source or sink. May
            # merge some new windows with existing windows, or with each other.
            for window in new_commits:
                assert len(window.commit_region) == 1 # all new windows are single commit regions
                window_idx = self.all_windows.index(window)
                patch = window.commit_region[0].patch
                prev_window_end = window.commit_region[0].round_start
                if (patch, prev_window_end) in self.window_end_lookup:
                    prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                    prev_window = self.all_windows[prev_window_idx]
                    prev_commit = prev_window.commit_region[cr_idx]
                    if prev_commit.discard_after:
                        # No previous window to merge with; this will be a source
                        if any(idx in self.layer_indices[1] for idx in self._get_adjacent_unconstructed_window_indices(window)):
                            self.layer_indices[0].add(window_idx)
                        else:
                            self.layer_indices[1].add(window_idx)
                    elif prev_window_idx in self.layer_indices[2]:
                        if len(prev_window.commit_region) < 3:
                            # Merge with prev sink and remove from all_windows
                            assert not prev_window.constructed
                            self.layer_indices[2].add(window_idx)
                            self._merge_windows(prev_window, window)
                        else:
                            # Sink is full; this will be a source. Add prev sink
                            # as buffer region.
                            # create new source
                            self.layer_indices[1].add(window_idx)
                            # Mark prev window as constructed
                            assert not prev_window.constructed
                            self._append_to_buffers(window, prev_commit, inplace=True)
                            self.window_dag.add_edge(window_idx, prev_window_idx)
                    else:
                        # This will be a sink; add buffer to prev source and
                        # mark as complete
                        assert prev_window_idx in self.layer_indices[0] or prev_window_idx in self.layer_indices[1]
                        assert not prev_window.constructed, (prev_commit.patch, prev_commit.round_start, prev_commit.duration, window.commit_region[0].round_start)
                        # TODO: buffer has to be d_m cycles "tall", so we may
                        # need to add more than just the next commit region
                        self.layer_indices[2].add(window_idx)
                        prev_window = self._append_to_buffers(prev_window, window.commit_region[0], inplace=True)
                        self.window_dag.add_edge(prev_window_idx, window_idx)
                else:
                    # No previous window to merge with; this will be a source
                    if any(idx in self.layer_indices[1] for idx in self._get_adjacent_unconstructed_window_indices(window)):
                        self.layer_indices[0].add(window_idx)
                    else:
                        self.layer_indices[1].add(window_idx)

            assert self.layer_indices[1].isdisjoint(self.layer_indices[2])
            assert set(range(len(self.all_windows))) == self.layer_indices[0] | self.layer_indices[1] | self.layer_indices[2]

            # At this point, every new commit region will only be connected
            # vertically to anything else. We now need to connect horizontally.

            # Naive approach to resolve conflicts: merge any adjacent sinks or
            # sources into each other.
            unconstructed_windows = self.get_unconstructed_windows()
            change_made = True
            while change_made:
                change_made = False
                for i, window_idx_1 in enumerate(unconstructed_windows):
                    if change_made:
                        break
                    window_1 = self.all_windows[window_idx_1]
                    for window_idx_2 in unconstructed_windows[:i]:
                        if change_made:
                            break
                        window_2 = self.all_windows[window_idx_2]
                        if window_1.shares_boundary(window_2) and self._get_layer_idx(window_idx_1) == self._get_layer_idx(window_idx_2):
                            self._merge_windows(window_1, window_2)
                            unconstructed_windows.remove(window_idx_2)
                            unconstructed_windows = [idx - 1 if idx > window_idx_2 else idx for idx in unconstructed_windows]
                            change_made = True
                            break

            # Now, all windows are valid. We finish by adding buffer regions and
            # updating DAG dependencies appropriately.
            unconstructed_windows = self.get_unconstructed_windows()
            for i, window_idx_1 in enumerate(unconstructed_windows):
                window_1 = self.all_windows[window_idx_1]
                for window_idx_2 in unconstructed_windows[:i]:
                    window_2 = self.all_windows[window_idx_2]
                    if window_1.shares_boundary(window_2):
                        layer_idx_1 = self._get_layer_idx(window_idx_1)
                        layer_idx_2 = self._get_layer_idx(window_idx_2)
                        if layer_idx_1 < layer_idx_2: # window_1 is source
                            for region in window_1.get_adjacent_commit_regions(window_2):
                                window_1 = self._append_to_buffers(window_1, region, inplace=True)
                            self.window_dag.add_edge(window_idx_1, window_idx_2)
                        elif layer_idx_1 > layer_idx_2: # window_2 is source
                            for region in window_2.get_adjacent_commit_regions(window_1):
                                window_2 = self._append_to_buffers(window_2, region, inplace=True)
                            self.window_dag.add_edge(window_idx_2, window_idx_1)
                        else:
                            raise ValueError("Invalid merge")

        self._update_buffer_wait()
        self._window_count_history.append(len(self.all_windows))

        if not new_rounds:
            # No new rounds; flush any dangling windows
            for window_idx in list(self.window_buffer_wait.keys()):
                window = self.all_windows[window_idx]
                assert not window.constructed
                self._mark_constructed(window, inplace=True)
            assert all(window.constructed for window in self.all_windows)

    def _get_layer_idx(self, window_idx: int) -> int:
        layer_idx = -1
        for i, layer in enumerate(self.layer_indices):
            if window_idx in layer:
                layer_idx = i
                break
        if layer_idx == -1:
            raise ValueError("Window not in any layer")
        return layer_idx

    def _merge_windows(self, window_1: DecodingWindow, window_2: DecodingWindow) -> DecodingWindow:
        """Wrapper for super()._merge_windows that updates source and sink
        indices.
        """
        window_idx_1 = self.all_windows.index(window_1)
        window_idx_2 = self.all_windows.index(window_2)

        layer_idx_1 = self._get_layer_idx(window_idx_1)
        layer_idx_2 = self._get_layer_idx(window_idx_2)
        if layer_idx_1 != layer_idx_2:
            raise ValueError("Cannot merge windows that are not assigned to same dependency layer")
        self.layer_indices[layer_idx_1].discard(window_idx_2)
        
        for i in range(len(self.layer_indices)):
            self.layer_indices[i] = {idx - 1 if idx > window_idx_2 else idx for idx in self.layer_indices[i]}

        new_window = super()._merge_windows(window_1, window_2)

        return new_window

    # def _split_windows(
    #         self,
    #         window: DecodingWindow,
    #         source_commit_regions: list[SpacetimeRegion],
    #         sink_commit_regions: list[SpacetimeRegion],
    #         source_layer_idx: int,
    #         sink_layer_idx: int,
    #         enforce_contiguous: bool = True,
    #     ) -> tuple[DecodingWindow, DecodingWindow]:
    #     """Wrapper for super()._split_windows that updates source and sink
    #     indices.
    #     """
    #     assert source_layer_idx < sink_layer_idx
    #     window_idx = self.all_windows.index(window)
    #     assert window_idx in self.layer_indices[source_layer_idx] or window_idx in self.layer_indices[sink_layer_idx]
    #     self.layer_indices[self._get_layer_idx(window_idx)].discard(window_idx)
    #     source_window, sink_window = super()._split_windows(window, source_commit_regions, sink_commit_regions, enforce_contiguous)
    #     source_idx = self.all_windows.index(source_window)
    #     sink_idx = self.all_windows.index(sink_window)
    #     self.layer_indices[source_layer_idx].add(source_idx)
    #     self.layer_indices[sink_layer_idx].add(sink_idx)
    #     return source_window, sink_window

class DynamicWindowManager(WindowManager):
    '''TODO'''
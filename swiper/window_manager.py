"""

Ideas for performance optimization (if needed later):
- window_idx_dict: dict[DecodingWindow, int] to avoid calling index() on all_windows
- Priority queue for buffer wait
"""
from abc import ABC, abstractmethod
import networkx as nx
from dataclasses import dataclass, asdict
import numpy as np
from numpy.typing import NDArray

from swiper.window_builder import WindowBuilder, DecodingWindow, SpacetimeRegion
from swiper.device_manager import SyndromeRound
from swiper.lattice_surgery_schedule import Instruction

@dataclass
class WindowData:
    """Data structure to hold information about windows throughout device run.
    """
    all_windows: list[DecodingWindow | None]
    all_constructed_windows: list[int]
    window_dag_edges: list[tuple[int, int]]
    window_construction_times: dict[int, int]
    window_volumes: list[tuple[list[int], list[int]]] # list of (commit_region_durations, buffer_region_durations)

    def to_dict(self):
        return asdict(self)
    
    def get_window(self, window_idx: int) -> DecodingWindow:
        window = self.all_windows[window_idx]
        if window is None:
            raise ValueError("Window does not exist")
        return window

class WindowManager(ABC):

    def __init__(self, window_builder: WindowBuilder, lightweight_output: bool = False):
        self.all_windows: list[DecodingWindow | None] = []
        self.all_constructed_windows: list[int] = []
        self.window_builder = window_builder
        self.window_dag = nx.DiGraph()
        self.window_end_lookup: dict[tuple[tuple[int, int], int], tuple[int, int]] = {}
        self.window_future_buffer_wait: dict[int, int] = {}
        self.window_construction_wait: set[int] = set()
        self.current_round = 0
        self._window_construction_times: dict[int, int] = {}
        self._unconstructed_window_indices: dict[DecodingWindow, int] = {}
        self.lightweight_output = lightweight_output

    @abstractmethod
    def process_round(self, new_rounds: list[SyndromeRound]) -> list[DecodingWindow]:
        """Process new syndrome rounds and update the decoding window dependency
        graph as needed.
        """
        raise NotImplementedError
    
    def _get_window(self, window_idx: int) -> DecodingWindow:
        window = self.all_windows[window_idx]
        if window is None:
            raise ValueError("Window does not exist")
        return window

    def pending_instruction_indices(self) -> set[int]:
        """Get the set of instruction indices that are currently generating
        windows.
        """
        return set([instr_idx for window in self._unconstructed_window_indices.keys() for instr_idx in window.parent_instr_idx])

    def get_unconstructed_windows(self):
        return list(self._unconstructed_window_indices.values())

    def _remove_window(self, window_idx: int) -> None:
        """TODO"""
        # Remove window from all_windows, and update window indices
        window = self._get_window(window_idx)
        if window in self._unconstructed_window_indices:
            self._unconstructed_window_indices.pop(window)
        self.all_windows[window_idx] = None
        self.window_dag.remove_node(window_idx)
        if window_idx in self.window_future_buffer_wait:
            self.window_future_buffer_wait.pop(window_idx)
        if window_idx in self.window_construction_wait:
            self.window_construction_wait.remove(window_idx)
        for cr in window.commit_region:
            if (cr.patch, cr.round_start + cr.duration) in self.window_end_lookup:
                if self.window_end_lookup[(cr.patch, cr.round_start + cr.duration)][0] == window_idx:
                    self.window_end_lookup.pop((cr.patch, cr.round_start + cr.duration))
        del window

    def _all_regions_touching(self, regions: list[SpacetimeRegion]):
        connectivity = []
        for i,cr1 in enumerate(regions):
            for j,cr2 in enumerate(regions):
                if i != j and cr1.shares_boundary(cr2):
                    connectivity.append((i,j))
        g = nx.Graph()
        g.add_nodes_from(range(len(regions)))
        g.add_edges_from(connectivity)
        return nx.is_connected(g)
    
    def _get_touching_unconstructed_window_indices(self, window: DecodingWindow) -> list[int]:
        """Get all unconstructed windows that share a boundary with window.
        """
        adjacent_windows = []
        for other_window, other_idx in self._unconstructed_window_indices.items():
            if window.shares_boundary(other_window):
                adjacent_windows.append(other_idx)
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
        window_idx_1 = self._unconstructed_window_indices[window_1]
        window_idx_2 = self._unconstructed_window_indices[window_2]

        if enforce_contiguous:
            if not self._all_regions_touching(list(window_1.commit_region) + list(window_2.commit_region)):
                raise ValueError("Commit regions must be contiguous")

        # Add window 2's attributes to window 1
        for succ in self.window_dag.successors(window_idx_2):
            if succ != window_idx_1:
                self.window_dag.add_edge(window_idx_1, succ)
        for pred in self.window_dag.predecessors(window_idx_2):
            if pred != window_idx_1:
                self.window_dag.add_edge(pred, window_idx_1)
        if window_idx_2 in self.window_construction_wait:
            self.window_construction_wait.add(window_idx_1)
        for cr_idx,cr in enumerate(window_2.commit_region):
            key = (cr.patch, cr.round_start + cr.duration)
            if key in self.window_end_lookup:
                if self.window_end_lookup[key][0] == window_idx_2:
                    self.window_end_lookup[key] = (window_idx_1, len(window_1.commit_region)+cr_idx)
        # for k,(w_idx,cr_idx) in self.window_end_lookup.items(): # TODO: make this not scale as O(n)
        #     if w_idx == window_idx_2:
        #         self.window_end_lookup[k] = (window_idx_1, len(window_1.commit_region)+cr_idx)

        # Construct new window and replace window_1 in all_windows
        new_window = DecodingWindow(
            commit_region=tuple(list(window_1.commit_region) + list(window_2.commit_region)),
            buffer_regions=window_1.buffer_regions | window_2.buffer_regions,
            merge_instr=window_1.merge_instr | window_2.merge_instr,
            parent_instr_idx=window_1.parent_instr_idx | window_2.parent_instr_idx,
            window_idx=window_idx_1,
            constructed=False,
        )
        self.all_windows[window_idx_1] = new_window
        self._unconstructed_window_indices.pop(window_1)
        self._unconstructed_window_indices[new_window] = window_idx_1

        self._remove_window(window_idx_2)

        return new_window

    def _append_to_buffers(self, window: DecodingWindow, region: SpacetimeRegion) -> DecodingWindow:
        """Create new window with region appended to buffer regions
        """
        assert not window.constructed
        if region in window.buffer_regions:
            return window
        new_window = DecodingWindow(
            commit_region=window.commit_region, 
            buffer_regions=window.buffer_regions | frozenset([region]), 
            merge_instr=window.merge_instr, 
            parent_instr_idx=window.parent_instr_idx,
            window_idx=window.window_idx,
            constructed=False,
        )
        window_idx = self._unconstructed_window_indices[window]
        self.all_windows[window_idx] = new_window
        self._unconstructed_window_indices.pop(window)
        self._unconstructed_window_indices[new_window] = window_idx
        del window
        return new_window

    def _mark_constructed(self, window_idx: int) -> DecodingWindow:
        """Create new window marked constructed to indicate it is ready to be
        decoded.
        """
        window = self._get_window(window_idx)
        assert not window.constructed
        if window_idx in self.window_future_buffer_wait:
            self.window_future_buffer_wait.pop(window_idx)
        if window_idx in self.window_construction_wait:
            self.window_construction_wait.remove(window_idx)
        self.all_constructed_windows.append(window_idx)
        self._window_construction_times[window_idx] = self.current_round
        new_window = DecodingWindow(
            commit_region=window.commit_region,
            buffer_regions=window.buffer_regions,
            merge_instr=window.merge_instr,
            parent_instr_idx=window.parent_instr_idx,
            window_idx=window.window_idx,
            constructed=True,
        )
        self.all_windows[window_idx] = new_window
        self._unconstructed_window_indices.pop(window)
        del window
        return new_window
    
    def purge_windows(self, window_indices) -> None:
        for window_idx in window_indices:
            window = self.all_windows[window_idx]
            if window:
                for cr in window.commit_region:
                    if (cr.patch, cr.round_start + cr.duration) in self.window_end_lookup:
                        if self.window_end_lookup[(cr.patch, cr.round_start + cr.duration)][0] == window_idx:
                            self.window_end_lookup.pop((cr.patch, cr.round_start + cr.duration))
                self.all_windows[window_idx] = None
                self.window_dag.remove_node(window_idx)
                pass
    
    def count_covered_faces(self, window_idx, cr_idx):
        # Need to make sure every face of every commit region either touches
        # another commit region, has an outgoing buffer, has an incoming
        # buffer, or is an initialization/discard/spatial boundary.
        window = self._get_window(window_idx)
        cr = window.commit_region[cr_idx]
        predecessors = list(self.window_dag.predecessors(window_idx))

        num_commit_neighbors = 0
        num_incoming_other_buffers = 0
        num_outgoing_buffers = 0
        num_terminations = 0
        for cr1 in window.commit_region:
            if cr1 != cr and cr1.shares_boundary(cr):
                num_commit_neighbors += 1
        for other_idx in predecessors:
            other_window = self._get_window(other_idx)
            for br,commits in other_window.buffer_boundary_commits().items():
                if br.overlaps(cr):
                    num_incoming_other_buffers += len(commits)
        for br,commits in window.buffer_boundary_commits().items():
            if cr in commits:
                num_outgoing_buffers += 1
        if cr.initialized_patch or cr.prior_t:
            num_terminations += 1
        if cr.discard_after:
            num_terminations += 1
        num_terminations += cr.num_spatial_boundaries

        return num_commit_neighbors, num_incoming_other_buffers, num_outgoing_buffers, num_terminations
    
    def _update_waiting_windows(self) -> None:
        """Look for dangling windows to mark as constructed.
        """
        constructed_windows = set()
        for window_idx in self.window_future_buffer_wait.keys():
            self.window_future_buffer_wait[window_idx] -= 1
            if self.window_future_buffer_wait[window_idx] <= 0:
                constructed_windows.add(self._get_window(window_idx))

        surrounded_windows = set()
        for window_idx in self.window_construction_wait:
            # Need to make sure every face of every commit region either touches
            # another commit region, has an outgoing buffer, has an incoming
            # buffer, or is an initialization/discard/spatial boundary.
            window = self._get_window(window_idx)
            ready_to_construct = True
            for cr_idx in range(len(window.commit_region)):
                num_commit_neighbors, num_incoming_other_buffers, num_outgoing_buffers, num_terminations = self.count_covered_faces(window_idx, cr_idx)
                total = num_commit_neighbors + num_incoming_other_buffers + num_outgoing_buffers + num_terminations
                if total < 6:
                    ready_to_construct = False
                    break

                if total != 6:
                    cr = window.commit_region[cr_idx]
                    raise ValueError(f'Total should be 6, but is {total}. {num_commit_neighbors}, {num_incoming_other_buffers}, {num_outgoing_buffers}, {num_terminations}, {cr.num_spatial_boundaries}, {window}, {cr}')
                assert total == 6
            if ready_to_construct:
                surrounded_windows.add(window_idx)

        for window_idx in surrounded_windows:
            self._mark_constructed(window_idx)

    def _flush_windows(self) -> None:
        # No new rounds; flush any dangling windows
        unconstructed_windows = list(self.window_construction_wait)
        for window_idx in unconstructed_windows:
            window = self._get_window(window_idx)
            assert not window.constructed
            self._mark_constructed(window_idx)
    
    def _clean_old_windows(self, newly_constructed_window_indices) -> None:
        # Remove old windows from all_windows
        assert self.lightweight_output
        for window_idx in newly_constructed_window_indices:
            for neighbor_idx in set(self.window_dag.successors(window_idx)) | set(self.window_dag.predecessors(window_idx)):
                if self.all_windows[neighbor_idx]:
                    can_clean = True
                    for neighbor_neighbor_idx in set(self.window_dag.successors(neighbor_idx)) | set(self.window_dag.predecessors(neighbor_idx)):
                        if neighbor_neighbor_idx in newly_constructed_window_indices:
                            can_clean = False
                            break
                    if can_clean:
                        self.all_windows[neighbor_idx] = None
                # neighbor_window = self.all_windows[neighbor_idx]
                # if neighbor_window is not None and not neighbor_window.constructed:
                #     can_clean = False
                #     break
            # if can_clean:
            #     self.all_windows[window_idx] = None

    def get_data(self) -> WindowData:
        if self.lightweight_output:
            return WindowData(
                all_windows=None,
                all_constructed_windows=None,
                window_dag_edges=None,
                window_construction_times=self._window_construction_times,
                window_volumes=[([cr.duration for cr in window.commit_region], [br.duration for br in window.buffer_regions]) for window in [self._get_window(window_idx) for window_idx in self.all_constructed_windows]]
            )
        else:
            return WindowData(
                all_windows=self.all_windows,
                all_constructed_windows=self.all_constructed_windows,
                window_dag_edges=list(self.window_dag.edges),
                window_construction_times=self._window_construction_times,
                window_volumes=[([cr.duration for cr in window.commit_region], [br.duration for br in window.buffer_regions]) for window in [self._get_window(window_idx) for window_idx in self.all_constructed_windows]]
            )

class SlidingWindowManager(WindowManager):
    def process_round(self, new_rounds: list[SyndromeRound]) -> list[DecodingWindow]:
        constructed_window_count = len(self.all_constructed_windows)

        if new_rounds:
            new_commits = self.window_builder.build_windows(new_rounds)
        else:
            new_commits = self.window_builder.flush()
        
        if new_commits:
            # Add new windows
            new_window_start = len(self.all_windows)
            self.all_windows.extend(new_commits)
            self._unconstructed_window_indices.update({window: new_window_start+i for i,window in enumerate(new_commits)})

            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                patch = window.commit_region[0].patch
                end = window.commit_region[0].round_start + window.commit_region[0].duration
                assert (patch, end) not in self.window_end_lookup
                self.window_end_lookup[(patch, end)] = (window_idx, 0)
                self.window_dag.add_node(window_idx)
                self.window_construction_wait.add(window_idx)
            
            # Process buffers in time (windows covering same patch)
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                patch = window.commit_region[0].patch
                prev_window_end = window.commit_region[0].round_start
                if (patch, prev_window_end) in self.window_end_lookup:
                    prev_window_idx,_ = self.window_end_lookup[(patch, prev_window_end)]
                    prev_window = self._get_window(prev_window_idx)
                    prev_commit = prev_window.commit_region[0]
                    if prev_commit.discard_after:
                        continue
                    # For sliding window, buffers extend at most one step forward in time
                    assert not prev_window.constructed
                    prev_window = self._append_to_buffers(prev_window, window.commit_region[0])
                    self.window_dag.add_edge(prev_window_idx, window_idx)

            # Process buffers in space (windows covering same MERGE instruction)
            unconstructed_window_indices = self.get_unconstructed_windows()
            for window_idx in unconstructed_window_indices:
                window = self._get_window(window_idx)
                cr = window.commit_region[0]
                merge_instr = cr.merge_instr
                if merge_instr:
                    touching_windows = self._get_touching_unconstructed_window_indices(window)
                    for w_idx in touching_windows:
                        other_window = self._get_window(w_idx)
                        other_cr = other_window.commit_region[0]
                        if other_cr.merge_instr == merge_instr: 
                            patch1 = cr.patch
                            patch2 = other_window.commit_region[0].patch
                            if ((patch1, patch2) in merge_instr.merge_faces or (patch2, patch1) in merge_instr.merge_faces) and (patch1[0] >= patch2[0]) and (patch1[1] >= patch2[1]):
                                if other_cr not in window.buffer_regions:
                                    assert not window.constructed and not other_window.constructed
                                    window = self._append_to_buffers(window, other_window.commit_region[0])
                                    cr = window.commit_region[0]
                                    self.window_dag.add_edge(window_idx, w_idx)

        self._update_waiting_windows()

        if not new_rounds:
            self._flush_windows()

        constructed_windows = [self._get_window(window_idx) for window_idx in self.all_constructed_windows[constructed_window_count:]]

        if self.lightweight_output:
            self._clean_old_windows(self.all_constructed_windows[constructed_window_count:])

        self.current_round += 1
        return constructed_windows
        
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

    def __init__(self, window_builder: WindowBuilder, lightweight_output: bool = False):
        self.layer_indices = [set(), set(), set()]
        super().__init__(window_builder, lightweight_output=lightweight_output)

    def process_round(self, new_rounds: list[SyndromeRound]) -> list[DecodingWindow]:
        constructed_window_count = len(self.all_constructed_windows)

        if new_rounds:
            new_commits = self.window_builder.build_windows(new_rounds)
        else:
            new_commits = self.window_builder.flush()

        if new_commits:
            self._add_new_commits(new_commits)
            self._assign_window_layers(new_commits)

            # At this point, every new commit region will only be connected
            # vertically to anything else. We now need to connect horizontally.
            self._merge_adjacent_windows()

            # Now, all windows are valid. We finish by adding buffer regions and
            # updating DAG dependencies appropriately.
            self._update_dependencies_and_dag()

        self._update_waiting_windows()

        if not new_rounds:
            self._flush_windows()

        constructed_windows = [self._get_window(window_idx) for window_idx in self.all_constructed_windows[constructed_window_count:]]

        if self.lightweight_output:
            self._clean_old_windows(self.all_constructed_windows[constructed_window_count:])

        self.current_round += 1
        return constructed_windows

    def _add_new_commits(self, new_commits: list[DecodingWindow]) -> None:
        """TODO"""
        new_window_start = len(self.all_windows)
        self.all_windows.extend(new_commits)
        self._unconstructed_window_indices.update({window: new_window_start+i for i,window in enumerate(new_commits)})
        for i,window in enumerate(new_commits):
            window_idx = new_window_start + i
            assert len(window.commit_region) == 1
            cr = window.commit_region[0]
            patch, end = cr.patch, cr.round_start + cr.duration
            assert (patch, end) not in self.window_end_lookup
            self.window_end_lookup[(patch, end)] = (window_idx, 0)
            self.window_dag.add_node(window_idx)
            self.window_construction_wait.add(window_idx)

    def _choose_layer(self, window_idx: int, possible_layers: list[int], tolerable_resulting_sizes: dict[int, int]) -> int:
        """Choose a layer to assign to a new window (which will result in it
        being merged with neighbors of the same layer) by calculating the
        expected size of the resulting merged window. 
        
        Will prefer adding to existing windows instead of inserting into a new
        layer if the resulting window is below the tolerable size for that
        layer. Assumes that first layer in list is default layer, if none are
        preferred.

        Args:
            window_idx: Index of the new window
            possible_layers: List of layer indices to consider. The first is
                treated as the default.
            tolerable_resulting_sizes: Dictionary of tolerable commit region
                sizes for each layer. If the resulting window is below this
                size, it will be added to the layer with the smallest size.
        """
        window = self._get_window(window_idx)
        layer_sizes = {layer: 0 for layer in possible_layers}
        for idx in self._get_touching_unconstructed_window_indices(window):
            for layer in possible_layers:
                if idx in self.layer_indices[layer]:
                    layer_sizes[layer] += self._get_window(idx).commit_spacetime_volume()
                    break
        if not any(layer_sizes.values()):
            return possible_layers[0]
        elif any(0 < size < tolerable_resulting_sizes[layer] for layer,size in layer_sizes.items()):
            # Prefer adding to existing windows if the resulting window is
            # below the tolerable size for that layer
            return min({layer:size for layer,size in layer_sizes.items() if 0 < size < tolerable_resulting_sizes[layer]}, key=lambda k: layer_sizes[k], default=possible_layers[0])
        else:
            return min(layer_sizes, key=lambda k: layer_sizes[k], default=possible_layers[0])

    def _assign_window_layers(self, new_commits: list[DecodingWindow]) -> None:
        """Process new commits, deciding on their window layer based on previous
        temporally-connected windows.
        """
        new_commits_to_process = new_commits.copy()
        # sort so that we process windows with history first
        def has_history(window):
            patch = window.commit_region[0].patch
            prev_window_end = window.commit_region[0].round_start
            if (patch, prev_window_end) in self.window_end_lookup:
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                return not prev_commit.discard_after
            return False
        new_commits_to_process.sort(key=lambda window: has_history(window), reverse=True)

        for window in new_commits_to_process:
            assert len(window.commit_region) == 1 # all new windows are single commit regions
            window_idx = self._unconstructed_window_indices[window]
            patch = window.commit_region[0].patch
            prev_window_end = window.commit_region[0].round_start
            if (patch, prev_window_end) in self.window_end_lookup:
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                if prev_commit.discard_after:
                    # No previous window to merge with; this will be a source
                    self.layer_indices[self._choose_layer(window_idx, [1,0], {0:3, 1:1})].add(window_idx)
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
                        self._append_to_buffers(window, prev_commit)
                        # self.window_dag.add_edge(window_idx, prev_window_idx)
                else:
                    # This will be a sink; add buffer to prev source and
                    # mark as complete
                    assert prev_window_idx in self.layer_indices[0] or prev_window_idx in self.layer_indices[1]
                    assert not prev_window.constructed
                    self.layer_indices[2].add(window_idx)
                    prev_window = self._append_to_buffers(prev_window, window.commit_region[0])
                    # self.window_dag.add_edge(prev_window_idx, window_idx)
            else:
                # No previous window to merge with; this will be a source
                # Decide layer based on size of touching windows
                self.layer_indices[self._choose_layer(window_idx, [1,0], {0:3, 1:1})].add(window_idx)

    def _merge_adjacent_windows(self) -> None:
        # Naive approach to resolve conflicts: merge any adjacent sinks or
        # sources into each other.
        unconstructed_windows = self.get_unconstructed_windows()
        change_made = True
        while change_made:
            change_made = False
            for i, window_idx_1 in enumerate(unconstructed_windows):
                if change_made:
                    break
                window_1 = self._get_window(window_idx_1)
                for window_idx_2 in unconstructed_windows[:i]:
                    if change_made:
                        break
                    window_2 = self._get_window(window_idx_2)
                    if window_1.shares_boundary(window_2) and self._get_layer_idx(window_idx_1) == self._get_layer_idx(window_idx_2):
                        self._merge_windows(window_1, window_2)
                        unconstructed_windows.remove(window_idx_2)
                        # unconstructed_windows = [idx - 1 if idx > window_idx_2 else idx for idx in unconstructed_windows]
                        change_made = True
                        break

    def _update_dependencies_and_dag(self) -> None:
        unconstructed_windows = self.get_unconstructed_windows()
        for i, window_idx_1 in enumerate(unconstructed_windows):
            window_1 = self._get_window(window_idx_1)
            for window_idx_2 in unconstructed_windows[:i]:
                window_2 = self._get_window(window_idx_2)
                if window_1.shares_boundary(window_2):
                    layer_idx_1 = self._get_layer_idx(window_idx_1)
                    layer_idx_2 = self._get_layer_idx(window_idx_2)
                    assert not window_1.constructed and not window_2.constructed
                    if layer_idx_1 < layer_idx_2: # window_1 is source
                        for region in window_1.get_touching_commit_regions(window_2):
                            window_1 = self._append_to_buffers(window_1, region)
                        self.window_dag.add_edge(window_idx_1, window_idx_2)
                    elif layer_idx_1 > layer_idx_2: # window_2 is source
                        for region in window_2.get_touching_commit_regions(window_1):
                            window_2 = self._append_to_buffers(window_2, region)
                        self.window_dag.add_edge(window_idx_2, window_idx_1)
                    else:
                        raise ValueError("Invalid merge")

    def _get_layer_idx(self, window_idx: int) -> int:
        layer_idx = -1
        for i, layer in enumerate(self.layer_indices):
            if window_idx in layer:
                layer_idx = i
                break
        if layer_idx == -1:
            raise ValueError("Window not in any layer")
        return layer_idx
    
    def _remove_window(self, window_idx: int) -> None:
        """Wrapper for super()._remove_window that updates source and sink
        indices.
        """
        layer_idx = self._get_layer_idx(window_idx)
        self.layer_indices[layer_idx].discard(window_idx)
        return super()._remove_window(window_idx)

    def _merge_windows(self, window_1: DecodingWindow, window_2: DecodingWindow) -> DecodingWindow:
        """Wrapper for super()._merge_windows that updates source and sink
        indices.
        """
        window_idx_1 = self._unconstructed_window_indices[window_1]
        window_idx_2 = self._unconstructed_window_indices[window_2]

        layer_idx_1 = self._get_layer_idx(window_idx_1)
        layer_idx_2 = self._get_layer_idx(window_idx_2)
        if layer_idx_1 != layer_idx_2:
            raise ValueError("Cannot merge windows that are not assigned to same dependency layer")

        new_window = super()._merge_windows(window_1, window_2)

        return new_window

class TAlignedWindowManager(ParallelWindowManager):
    """A version of ParallelWindowManager that enforces that every window
    covering a blocking operation does not have any dependencies on future
    windows. This is to ensure that the blocking operation can be decoded
    ASAP.

    We do this by adding two new layer options to the window manager which will only
    be used for these special windows. The new layers are interleaved with the
    typical layers in ParallelWindowManager. In ParallelWindowManager, the
    layers are 0 (alternate source), 1 (main source), and 2 (sink). In this
    version, the layers are 0 (alternate source), 1 (main source), 2 (blocking
    window 1), 3 (blocking window 2), and 4 (sink). After a blocking window, we always begin a sink. This
    ensures that the blocking window never has any dependencies on future
    windows.

    """
    def __init__(self, window_builder: WindowBuilder, lightweight_output: bool = False):
        super().__init__(window_builder, lightweight_output=lightweight_output)
        self.layer_indices = [set(), set(), set(), set(), set()]

    def _is_blocking_window(self, window: DecodingWindow) -> bool:
        """Check if a window is blocking another window from being decoded.
        """
        return any(instr.conditional_dependencies for instr in window.merge_instr)

    def _assign_window_layers(self, new_commits: list[DecodingWindow]) -> None:
        # Process buffers in time (windows covering same patch)

        new_commits_to_process = new_commits.copy()
        # sort so that we process windows with history first
        def has_history(window):
            patch = window.commit_region[0].patch
            prev_window_end = window.commit_region[0].round_start
            if (patch, prev_window_end) in self.window_end_lookup:
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                return not prev_commit.discard_after
            return False
        new_commits_to_process.sort(key=lambda window: has_history(window), reverse=True)

        for window in new_commits_to_process:
            assert len(window.commit_region) == 1 # all new windows are single commit regions
            window_idx = self._unconstructed_window_indices[window]
            patch = window.commit_region[0].patch
            prev_window_end = window.commit_region[0].round_start
            if (patch, prev_window_end) in self.window_end_lookup:
                prev_window_idx, cr_idx = self.window_end_lookup[(patch, prev_window_end)]
                prev_window = self._get_window(prev_window_idx)
                prev_commit = prev_window.commit_region[cr_idx]
                if prev_commit.discard_after:
                    # No previous window to merge with; this will be a source
                    if self._is_blocking_window(window):
                        self.layer_indices[self._choose_layer(window_idx, [3,2], {3:1, 2:3})].add(window_idx)
                    else:
                        self.layer_indices[self._choose_layer(window_idx, [1,0], {1:1, 0:3})].add(window_idx)
                elif prev_window_idx in self.layer_indices[4]:
                    if self._is_blocking_window(window):
                        self.layer_indices[3].add(window_idx)
                        assert not prev_window.constructed
                        self._append_to_buffers(window, prev_commit)
                        # self.window_dag.add_edge(window_idx, prev_window_idx)
                    elif len(prev_window.commit_region) < 3:
                        # Merge with prev sink and remove from all_windows
                        assert not prev_window.constructed
                        self.layer_indices[4].add(window_idx)
                        self._merge_windows(prev_window, window)
                    else:
                        # Sink is full; this will be a source. Add prev sink
                        # as buffer region.
                        # create new source
                        self.layer_indices[1].add(window_idx)
                        # Mark prev window as constructed
                        assert not prev_window.constructed
                        self._append_to_buffers(window, prev_commit)
                        self.window_dag.add_edge(window_idx, prev_window_idx)
                else:
                    # Previous is a source; this will be a sink
                    assert not prev_window.constructed
                    if self._is_blocking_window(window):
                        assert self._get_layer_idx(prev_window_idx) in [0,1]
                        self.layer_indices[3].add(window_idx)
                        prev_window = self._append_to_buffers(prev_window, window.commit_region[0])
                        # self.window_dag.add_edge(prev_window_idx, window_idx)
                    else:
                        assert self._get_layer_idx(prev_window_idx) in [0,1,2,3]
                        self.layer_indices[4].add(window_idx)
                        prev_window = self._append_to_buffers(prev_window, window.commit_region[0])
                        # self.window_dag.add_edge(prev_window_idx, window_idx)
            else:
                # No previous window to merge with; this will be a source
                if self._is_blocking_window(window):
                    self.layer_indices[self._choose_layer(window_idx, [3,2], {3:1, 2:3})].add(window_idx)
                else:
                    self.layer_indices[self._choose_layer(window_idx, [1,0], {1:1, 0:3})].add(window_idx)
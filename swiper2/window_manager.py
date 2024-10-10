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
    window_end_lookup: dict[int, list[int]]
    window_buffer_wait: dict[int, int]
    window_count_history: NDArray[np.int_]

class WindowManager(ABC):

    def __init__(self, window_builder: WindowBuilder):
        self.all_windows: list[DecodingWindow] = []
        self.window_builder = window_builder
        self.window_dag = nx.DiGraph()
        self.window_end_lookup = {}
        self.window_buffer_wait = {}
        self._window_count_history = []

    @abstractmethod
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        '''
        Process new syndrome rounds and update the decoding window dependency graph as needed
        '''
        raise NotImplementedError

    @abstractmethod
    def update_buffer_wait(self) -> None:
        '''
        Update the buffer wait time for each window
        '''
        raise NotImplementedError
    
    def pending_instruction_indices(self) -> set[int]:
        '''
        Get the set of instruction indices that are currently generating windows
        '''
        return set([instr_idx for window in self.all_windows for instr_idx in window.parent_instr_idx if not window.constructed])

    def _append_to_buffers(self, window: DecodingWindow, region: SpacetimeRegion, constructed: bool=False) -> DecodingWindow:
        '''
        Create new window with region appended to buffer regions
        '''
        return DecodingWindow(window.commit_region, 
                              window.buffer_regions | frozenset([region]), 
                              window.merge_instr, 
                              window.parent_instr_idx,
                              window.constructed | constructed)

    def _mark_constructed(self, window: DecodingWindow) -> DecodingWindow:
        '''
        Create new window marked constructed to indicate it is ready to be decoded
        '''
        if window.constructed:
            return window
        return DecodingWindow(window.commit_region,
                              window.buffer_regions,
                              window.merge_instr,
                              window.parent_instr_idx,
                              True)

class SlidingWindowManager(WindowManager):
    
    def process_round(self, new_rounds: list[SyndromeRound] | None) -> None:
        if new_rounds:
            new_commits = self.window_builder.build_windows(new_rounds)

            new_window_start = len(self.all_windows)

            # Add new windows
            self.all_windows.extend(new_commits)
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                self.window_end_lookup.setdefault(window.commit_region.round_start + window.commit_region.duration, []).append(window_idx)
                self.window_dag.add_node(window_idx)
                self.window_buffer_wait[new_window_start + i] = self.window_builder.d+1
            
            # Process buffers in time
            for i, window in enumerate(new_commits):
                window_idx = new_window_start + i
                patch = window.commit_region.patch
                prev_window_end = window.commit_region.round_start
                if prev_window_end not in self.window_end_lookup:
                    continue
                for prev_window_idx in self.window_end_lookup[prev_window_end]:
                    prev_window = self.all_windows[prev_window_idx]
                    if patch == prev_window.commit_region.patch:
                        # For sliding window, buffers extend at most one step forward in time
                        # Mark prev window as constructed and ready to decode
                        assert not prev_window.constructed
                        self.all_windows[prev_window_idx] = self._append_to_buffers(prev_window, window.commit_region, True)
                        self.window_dag.add_edge(prev_window_idx, window_idx)

            # Process buffers in space
            merge_windows = {}
            for i, window in enumerate(new_commits):
                if window.merge_instr:
                    merge_windows.setdefault(window.merge_instr, []).append(i)
            for _, window_idxs in merge_windows.items():
                for i, window_1 in enumerate(window_idxs):
                    for window_2 in window_idxs[i + 1:]:
                        patch1 = self.all_windows[new_window_start + window_1].commit_region.patch
                        patch2 = self.all_windows[new_window_start + window_2].commit_region.patch
                        if abs(patch1[0] - patch2[0]) + abs(patch1[1] - patch2[1]) == 1:
                            # Adjacent, merged patches
                            assert not self.all_windows[new_window_start + window_1].constructed
                            self.all_windows[new_window_start + window_1] = self._append_to_buffers(self.all_windows[new_window_start + window_1], 
                                                                                                    self.all_windows[new_window_start + window_2].commit_region)
                            self.window_dag.add_edge(new_window_start + window_1, new_window_start + window_2)

        self.update_buffer_wait()
        self._window_count_history.append(len(self.all_windows))

    def update_buffer_wait(self) -> None:
        # Look for dangling windows to mark as constructed
        constructed_windows = []
        for window_idx, buff_time in self.window_buffer_wait.items():
            self.window_buffer_wait[window_idx] -= 1
            if self.window_buffer_wait[window_idx] == 0:
                constructed_windows.append(window_idx)
        for window_idx in constructed_windows:
            self.all_windows[window_idx] = self._mark_constructed(self.all_windows[window_idx])
        self.window_buffer_wait = {k: v for k, v in self.window_buffer_wait.items() if v > 0}

    def get_data(self) -> WindowData:
        return WindowData(
            all_windows=self.all_windows,
            window_dag=self.window_dag,
            window_end_lookup=self.window_end_lookup,
            window_buffer_wait=self.window_buffer_wait,
            window_count_history=np.array(self._window_count_history, int),
        )
        
        
class ParallelWindowManager(WindowManager):
    '''TODO'''
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        new_commits = self.window_builder.build_windows(new_rounds)
        if len(new_commits) == 0:
            # No new windows
            return
        raise NotImplementedError

    
class DynamicWindowManager(WindowManager):
    '''TODO'''
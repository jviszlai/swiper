from abc import ABC, abstractmethod
import networkx as nx

from swiper2.window_builder import WindowBuilder, DecodingWindow, SpacetimeRegion
from swiper2.device_manager import SyndromeRound
from swiper2.lattice_surgery_schedule import Instruction

class WindowManager(ABC):

    def __init__(self, window_builder: WindowBuilder):
        self.all_windows = []
        self.window_builder = window_builder
        self.window_dag = nx.DiGraph()
        self.window_end_lookup = {}

    @abstractmethod
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        '''
        Process new syndrome rounds and update the decoding window dependency graph as needed
        '''
        raise NotImplementedError
    
    def _append_to_buffers(self, window: DecodingWindow, region: SpacetimeRegion) -> DecodingWindow:
        '''
        Create new window with region appended to buffer regions
        '''
        return DecodingWindow(window.commit_region, 
                              window.buffer_regions | frozenset([region]), 
                              window.merge_instr, 
                              window.parent_instr_idx)

class SlidingWindowManager(WindowManager):
    
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        new_commits = self.window_builder.build_windows(new_rounds)
        if len(new_commits) == 0:
            # No new windows
            return
        
        new_window_start = len(self.all_windows)
        
        # Add new windows
        self.all_windows.extend(new_commits)
        for i, window in enumerate(new_commits):
            window_idx = new_window_start + i
            self.window_end_lookup.setdefault(window.commit_region.round_start + window.commit_region.duration, []).append(window_idx)
            self.window_dag.add_node(window_idx)
        
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
                    self.all_windows[prev_window_idx] = self._append_to_buffers(prev_window, window.commit_region)
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
                        self.all_windows[new_window_start + window_1] = self._append_to_buffers(self.all_windows[new_window_start + window_1], 
                                                                                                self.all_windows[new_window_start + window_2].commit_region)
                        self.window_dag.add_edge(new_window_start + window_1, new_window_start + window_2)

            
        
        
class ParallelWindowManager(WindowManager):

    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        new_commits = self.window_builder.build_windows(new_rounds)
        if len(new_commits) == 0:
            # No new windows
            return
        raise NotImplementedError

    
    
    


from abc import ABC, abstractmethod
import networkx as nx

from swiper2.window_builder import WindowBuilder
from swiper2.device_manager import SyndromeRound
from swiper2.lattice_surgery_schedule import Instruction

class WindowManager(ABC):

    def __init__(self, window_builder: WindowBuilder):
        self.all_windows = []
        self.waiting_windows = []
        self.window_builder = window_builder
        self.window_dag = nx.DiGraph()
        self.window_end_lookup = {}

    @abstractmethod
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        '''
        Process new syndrome rounds and update the decoding window dependency graph as needed
        '''
        raise NotImplementedError

class SlidingWindowManager(WindowManager):
    
    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        new_commits = self.window_builder.build_windows(new_rounds)
        if len(new_commits) == 0:
            # No new windows
            return
        
        new_window_start = len(self.all_windows)
        
        # Add new windows
        self.waiting_windows.extend(new_commits)
        self.all_windows.extend(new_commits)
        for i, window in enumerate(new_commits):
            window_idx = new_window_start + i
            self.window_end_lookup.setdefault(window.commit_region.round_start + window.commit_region.duration, []).append(window_idx)
            self.window_dag.add_node(window_idx)
        
        # Process buffers in time
        for i, window in enumerate(new_commits):
            window_idx = new_window_start + i
            patch = window.commit_region.space_footprint[0]
            prev_window_end = window.commit_region.round_start
            if prev_window_end not in self.window_end_lookup:
                continue
            for prev_window_idx in self.window_end_lookup[prev_window_end]:
                prev_window = self.all_windows[prev_window_idx]
                if patch in prev_window.commit_region.space_footprint:
                    prev_window.buffer_regions.append(window.commit_region)
                    self.window_dag.add_edge(prev_window_idx, window_idx)
        
        # Process buffers in space
            
        
        
class ParallelWindowManager(WindowManager):

    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        new_commits = self.window_builder.build_windows(new_rounds)
        if len(new_commits) == 0:
            # No new windows
            return
        raise NotImplementedError

    
    
    


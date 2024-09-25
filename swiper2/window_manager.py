from abc import ABC, abstractmethod
import networkx as nx

from swiper2.window_builder import WindowBuilder
from swiper2.device_manager import SyndromeRound

class WindowManager(ABC):

    def __init__(self, window_builder: WindowBuilder):
        self.waiting_windows = []
        self.window_builder = window_builder
        self.window_dag = nx.DiGraph()

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
        
class ParallelWindowManager(WindowManager):

    def process_round(self, new_rounds: list[SyndromeRound]) -> None:
        new_commits = self.window_builder.build_windows(new_rounds)
        if len(new_commits) == 0:
            # No new windows
            return

    
    
    


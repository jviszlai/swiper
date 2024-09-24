from abc import ABC, abstractmethod

from device_manager import SyndromeRound
from window_manager import DecodingWindow

class WindowBuilder(ABC):

    @abstractmethod
    def build_windows(self, 
                      waiting_rounds: list[SyndromeRound]
                      ) -> tuple[list[SyndromeRound], list[DecodingWindow]]:
        '''
        Args:
            waiting_rounds: List of unassigned syndrome rounds

        Returns:
            List of still unassigned syndrome rounds,
            List of constructed decoding windows
        '''
        raise NotImplementedError
    

class SlidingWindowBuilder(WindowBuilder):

    def build_windows(self, 
                      waiting_rounds: list[SyndromeRound]
                      ) -> tuple[list[SyndromeRound], list[DecodingWindow]]:
        '''
        TODO
        '''
        raise NotImplementedError

class ParallelWindowBuilder(WindowBuilder):

    def build_windows(self, 
                      waiting_rounds: list[SyndromeRound]
                      ) -> tuple[list[SyndromeRound], list[DecodingWindow]]:
        '''
        TODO
        '''
        raise NotImplementedError

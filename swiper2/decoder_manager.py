from dataclasses import dataclass
from typing import Callable
import numpy as np
import networkx as nx
from swiper2.window_builder import DecodingWindow, SpacetimeRegion
from swiper2.lattice_surgery_schedule import Instruction

@dataclass
class DecoderData:
    max_parallel_processes: int | None
    max_parallel_processes_by_round: list[int]
    window_completion_times: dict[DecodingWindow, int]
    window_decoding_times: dict[DecodingWindow, int]

class DecoderManager:
    def __init__(
            self,
            decoding_time_function: Callable[[int], int],
            speculation_time: int,
            speculation_accuracy: float,
            max_parallel_processes: int | None = None,
            speculation_mode: str = 'integrated',
        ) -> None:
        """Initialize the decoder manager.
        
        Args:
            decoding_time_function: A function that returns the number of rounds
                required to decode a given spacetime volume of syndrome
                measurements. The volume is specified in units of rounds*d^2.
            speculation_accuracy: Accuracy of the speculation step
            speculation_time: Number of rounds required to make a speculative
                prediction for artifial defects at the boundary of a decoding
                window.
            max_parallel_processes: Maximum number of parallel decoding processes
                to run. If None, run as many as possible and keep track of
                the maximum number that were run.
            speculation_mode: 'integrated' or 'separate'. If 'integrated', the
                speculation time is included in the decoding time of a window,
                and speculation can only be performed once the decoder starts
                processing the window. If 'separate', the speculation time is
                not included in the decoding, time of a window, and speculation
                can be run independently of decoding. In this case, speculation
                uses a parallel process and counts towards
                max_parallel_processes.
        """
        self.decoding_time_function = decoding_time_function
        self.speculation_time = speculation_time
        self.speculation_accuracy = speculation_accuracy
        self.speculation_mode = speculation_mode

        self.max_parallel_processes = max_parallel_processes
        self._parallel_processes_by_round = []
        self._current_round = 0
        self._window_completion_times: dict[DecodingWindow, int] = {}
        self._window_decoding_times: dict[DecodingWindow, int] = {}
        self._speculation_progress: dict[DecodingWindow, int] = {} # maps each speculated window to the number of rounds elapsed since speculation started
        self._active_window_progress: dict[DecodingWindow, int] = {} # maps each active window to rounds remaining until completion of decoding
        self._speculated_windows: set[DecodingWindow] = set() # windows whose boundaries have been speculated
        self._pending_windows: set[DecodingWindow] = set() # windows with all dependencies satisfied that are now ready to be decoded


    def update_decoding(self, all_windows: list[DecodingWindow], window_idx_dag: nx.DiGraph) -> None:
        """Update state of processing windows and start any new decoding
        processes, if possible.

        Currently, assumes that speculation is performed at the beginning of
        decoding a window. E.g. if speculation takes 2 rounds and decoding a
        window takes 10 rounds, the speculation will be completed 2 rounds after
        the decoder starts decoding the window.

        Args:
            all_windows: List of all decoding windows.
            window_idx_dag: Directed acyclic graph representing the dependencies
                between decoding windows.
        """
        # Step decoders forward; check if any windows have completed
        for window in self._active_window_progress:
            self._active_window_progress[window] -= 1
        completed_windows = []
        poisoned_windows = []

        for window, time_remaining in self._active_window_progress.items():
            if time_remaining <= 0:
                completed_windows.append(window)
                if np.random.choice([True, False], p=[1 - self.speculation_accuracy, self.speculation_accuracy]):
                    # Mis-speculation
                    poisoned_windows.append(window)
                
        for window in completed_windows:
            self._window_completion_times[window] = self._current_round
            self._active_window_progress.pop(window)

        # Step speculation forward; check if any windows are ready to be decoded
        for window in self._speculation_progress:
            self._speculation_progress[window] -= 1
        speculated_windows = []
        for window, time_remaining in self._speculation_progress.items():
            if time_remaining <= 0:
                speculated_windows.append(window)
        for window in speculated_windows:
            self._speculated_windows.add(window)
            self._speculation_progress.pop(window)

        # Update poisoned windows
        for window_idx, window in enumerate(all_windows):
            parents = list(window_idx_dag.predecessors(window_idx))
            if any(all_windows[parent_idx] in poisoned_windows for parent_idx in parents):
                if window in self._window_completion_times:
                    self._window_completion_times.pop(window)
                elif window in self._active_window_progress:
                    self._active_window_progress.pop(window)
                if window in self._speculation_progress:
                    self._speculation_progress.pop(window)
                elif window in self._speculated_windows:
                    self._speculated_windows.remove(window)

        # Check dependencies for all windows
        finished_dependencies = set(self._window_completion_times.keys()) | self._speculated_windows
        for window_idx, window in enumerate(all_windows):
            if not window.constructed:
                continue

            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            if self.speculation_mode == 'separate' and window not in (self._speculated_windows | set(self._speculation_progress.keys())):
                # immediately speculate any window that is ready
                if self.speculation_time > 0:
                    self._speculation_progress[window] = self.speculation_time
                else:
                    self._speculated_windows.add(window)
            
            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            if window not in self._window_completion_times and window not in self._active_window_progress and window not in self._pending_windows:
                # window has not been processed yet
                parents = list(window_idx_dag.predecessors(window_idx))
                if any(all_windows[parent_idx] not in finished_dependencies for parent_idx in parents):
                    continue
                self._active_window_progress[window] = self.decoding_time_function(window.total_spacetime_volume())
                if self.speculation_mode == 'integrated':
                    if self.speculation_time > 0:
                        self._speculation_progress[window] = self.speculation_time
                    else:
                        self._speculated_windows.add(window)

        self._current_round += 1
        if not self.max_parallel_processes:
            self._parallel_processes_by_round.append(len(self._active_window_progress))
    
    def decoded_instruction_idx(self) -> set[int]:
        """Return the set of instruction idx that have been decoded."""
        decoded_instructions = set()
        for window in self._window_completion_times:
            decoded_instructions |= window.parent_instr_idx
        for window in self._active_window_progress:
            decoded_instructions -= window.parent_instr_idx
        return decoded_instructions

    def get_data(self) -> DecoderData:
        return DecoderData(
            self.max_parallel_processes,
            self._parallel_processes_by_round,
            self._window_completion_times,
            self._window_decoding_times,
        )
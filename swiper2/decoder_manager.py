from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from swiper2.window_builder import DecodingWindow, SpacetimeRegion
from swiper2.lattice_surgery_schedule import Instruction

@dataclass
class DecoderData:
    num_rounds: int
    max_parallel_processes: int | None
    parallel_processes_by_round: NDArray[np.int_]
    completed_windows_by_round: NDArray[np.int_]
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
        assert self.speculation_mode in ['integrated', 'separate']

        self.max_parallel_processes = max_parallel_processes
        self._parallel_processes_by_round = []
        self._completed_windows_by_round = []
        self._current_round = 0
        self._window_completion_times: dict[DecodingWindow, int] = {}
        self._window_used_parent_speculations: dict[DecodingWindow, dict[DecodingWindow, bool]] = {}
        self._window_decoding_times: dict[DecodingWindow, int] = {}
        self._speculation_progress: dict[DecodingWindow, int] = {} # maps each speculated window to rounds remaining to complete speculation
        self._active_window_progress: dict[DecodingWindow, int] = {} # maps each active window to rounds remaining to complete decoding
        self._speculated_windows: set[DecodingWindow] = set() # windows whose boundaries have been speculated

    def step(self, all_windows: list[DecodingWindow], window_idx_dag: nx.DiGraph) -> None:
        """Step decoding and speculation forward by one round (without creating
        any new processes)."""
        # Step decoders forward; check if any windows have completed
        for window in self._active_window_progress:
            self._active_window_progress[window] -= 1
        completed_windows = []
        poisoned_speculations = []

        for window, time_remaining in self._active_window_progress.items():
            if time_remaining <= 0:
                completed_windows.append(window)
                if np.random.random() > self.speculation_accuracy:
                    # Mis-speculation
                    # TODO: missed speculations should happen per-buffer-region, not per-window
                    poisoned_speculations.append(window)

        for window in completed_windows:
            # print(f'FINISH DECODING w{all_windows.index(window)}')
            self._window_completion_times[window] = self._current_round
            self._active_window_progress.pop(window)

            # from now on, we can rely on the decoded result, not the faulty speculation
            if window in self._speculated_windows:
                self._speculated_windows.remove(window)
            if window in self._speculation_progress:
                self._speculation_progress.pop(window)

        # Step speculation forward
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
        # For each poisoned window, reset any descendants that used the poisoned
        # speculation.
        active_or_completed_windows = set(self._active_window_progress.keys()) | set(self._window_completion_times.keys())
        for poisoned_window in poisoned_speculations:
            dependents = window_idx_dag.successors(all_windows.index(poisoned_window))
            for dependent_idx in dependents:
                dependent = all_windows[dependent_idx]
                if dependent in active_or_completed_windows:
                    assert poisoned_window in self._window_used_parent_speculations[dependent] and self._window_used_parent_speculations[dependent][poisoned_window]
                    indices_to_reset = [dependent_idx] + list(nx.descendants(window_idx_dag, dependent_idx))
                    # print(f'POISON! RESETTING w{indices_to_reset}')
                    for window_idx in indices_to_reset:
                        window = all_windows[window_idx]
                        if window in self._window_completion_times:
                            self._window_completion_times.pop(window)
                        elif window in self._active_window_progress:
                            self._active_window_progress.pop(window)

        self._current_round += 1
        self._parallel_processes_by_round.append(len(self._active_window_progress))
        self._completed_windows_by_round.append(len(self._window_completion_times))

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
        # Check dependencies and start new speculation and decoding processes
        completed_windows = set(self._window_completion_times.keys())
        unprocessed_windows = set(all_windows) - completed_windows
        while len(unprocessed_windows) > 0:
            window = unprocessed_windows.pop()
            window_idx = all_windows.index(window)

            if window in self._window_completion_times or not window.constructed:
                continue

            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            # begin a speculation
            if self.speculation_mode == 'separate' and window not in (self._speculated_windows | set(self._speculation_progress.keys()) | set(self._window_completion_times.keys())):
                if self.speculation_time > 0:
                    self._speculation_progress[window] = self.speculation_time
                else:
                    self._speculated_windows.add(window)
                    # window_idx = 0 # reset window_idx to 0 to recheck all windows for decoding
                    # continue
                    dependents = {all_windows[w_idx] for w_idx in window_idx_dag.successors(window_idx)}
                    unprocessed_windows |= dependents - completed_windows
            
            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            if window not in self._window_completion_times and window not in self._active_window_progress:
                # window has not been processed yet
                parents = list(window_idx_dag.predecessors(window_idx))
                if any(all_windows[parent_idx] not in (completed_windows | self._speculated_windows) for parent_idx in parents):
                    continue
                # begin decoding
                self._active_window_progress[window] = self.decoding_time_function(window.total_spacetime_volume())
                self._window_used_parent_speculations[window] = {}
                for parent_idx in parents:
                    parent = all_windows[parent_idx]
                    if parent in completed_windows:
                        self._window_used_parent_speculations[window][parent] = False
                    else:
                        assert parent in self._speculated_windows
                        self._window_used_parent_speculations[window][parent] = True
                if self.speculation_mode == 'integrated':
                    # begin a speculation along with decoding
                    if self.speculation_time > 0:
                        self._speculation_progress[window] = self.speculation_time
                    else:
                        self._speculated_windows.add(window)
                        # window_idx = 0 # reset window_idx to 0 to recheck all windows for decoding
                        # continue
                        dependents = {all_windows[w_idx] for w_idx in window_idx_dag.successors(window_idx)}
                        unprocessed_windows |= dependents - completed_windows

    def decoded_windows(self) -> set[DecodingWindow]:
        """Return the set of windows that have been decoded."""
        return set(self._window_completion_times.keys())
    
    def get_finished_instruction_indices(self, all_windows: list[DecodingWindow]) -> set[int]:
        """Return the set of instruction idx that have been decoded."""
        decoded_instructions = set()
        for window in self._window_completion_times:
            decoded_instructions |= window.parent_instr_idx
        for window in all_windows:
            if window not in self._window_completion_times:
                decoded_instructions -= window.parent_instr_idx
        decoded_instructions -= {-1}
        return decoded_instructions

    def decoded_instruction_indices(self) -> set[int]:
        """Return the set of instruction idx that have been decoded."""
        decoded_instructions = set()
        for window in self._window_completion_times:
            decoded_instructions |= window.parent_instr_idx
        for window in self._active_window_progress:
            decoded_instructions -= window.parent_instr_idx
        decoded_instructions -= {-1}
        return decoded_instructions

    def get_data(self) -> DecoderData:
        return DecoderData(
            num_rounds=self._current_round,
            max_parallel_processes=self.max_parallel_processes,
            parallel_processes_by_round=np.array(self._parallel_processes_by_round, int),
            completed_windows_by_round=np.array(self._completed_windows_by_round, int),
            window_completion_times=self._window_completion_times,
            window_decoding_times=self._window_decoding_times,
        )
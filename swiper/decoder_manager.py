from dataclasses import dataclass, field, asdict
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from swiper.window_builder import DecodingWindow, SpacetimeRegion
from swiper.lattice_surgery_schedule import Instruction

@dataclass
class DecoderData:
    num_rounds: int
    max_parallel_processes: int | None
    parallel_processes_by_round: list[int]
    completed_windows_by_round: list[int]
    window_speculation_start_times: dict[int, int]
    window_decoding_start_times: dict[int, int]
    window_decoding_completion_times: dict[int, int]
    missed_speculation_events: list[tuple[int, list[int]]] # list of (round, list of poisoned window indices) tuples

    def to_dict(self):
        return asdict(self)

@dataclass
class DecoderTask:
    window: DecodingWindow
    window_idx: int # window index in DAG
    completed_decoding: bool = False
    completed_speculation: bool = False
    used_parent_speculations: dict[int, bool] = field(default_factory=dict)

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
        self._window_speculation_start_times: dict[int, int] = {}
        self._window_decoding_start_times: dict[int, int] = {}
        self._window_decoding_completion_times: dict[int, int] = {}
        self._missed_speculation_events: list[tuple[int, list[int]]] = []
        self._active_speculation_progress: dict[int, int] = {} # maps each speculated window to rounds remaining to complete speculation
        self._active_window_progress: dict[int, int] = {} # maps each active window to rounds remaining to complete decoding
        self._pending_decode_tasks: set[int] = set()
        self._pending_speculate_tasks: set[int] = set()
        self._tasks_by_idx: list[DecoderTask | None] = []
        self._partially_complete_instructions: set[int] = set()

        self._window_idx_dag = nx.DiGraph()

    def step(self) -> None:
        """Step decoding and speculation forward by one round (without creating
        any new processes)."""
        # Step decoders forward; check if any windows have completed
        for task_idx in self._active_window_progress:
            self._active_window_progress[task_idx] -= 1
        completed_windows = []
        poisoned_speculations = []

        for task_idx, time_remaining in self._active_window_progress.items():
            if time_remaining <= 0:
                completed_windows.append(task_idx)
                if np.random.random() > self.speculation_accuracy:
                    # Mis-speculation
                    # TODO: missed speculations should happen per-buffer-region, not per-window
                    poisoned_speculations.append(task_idx)

        for task_idx in completed_windows:
            task = self._get_task(task_idx)
            # print(f'FINISH DECODING w{all_windows.index(window)}')
            self._window_decoding_completion_times[task_idx] = self._current_round
            self._active_window_progress.pop(task_idx)
            task.completed_decoding = True
            self._partially_complete_instructions |= task.window.parent_instr_idx

            # from now on, we can rely on the decoded result, not the faulty
            # speculation
            task.completed_speculation = False
            if task_idx in self._active_speculation_progress:
                self._active_speculation_progress.pop(task_idx)
            if task_idx in self._pending_speculate_tasks:
                self._pending_speculate_tasks.remove(task_idx)

        # Step speculation forward
        for task_idx in self._active_speculation_progress:
            self._active_speculation_progress[task_idx] -= 1
        speculated_windows = []
        for task_idx, time_remaining in self._active_speculation_progress.items():
            if time_remaining <= 0:
                speculated_windows.append(task_idx)
        for task_idx in speculated_windows:
            task = self._get_task(task_idx)
            task.completed_speculation = True
            self._active_speculation_progress.pop(task_idx)

        # Update poisoned windows
        # For each poisoned window, reset any descendants that used the poisoned
        # speculation.
        active_windows = set(self._active_window_progress.keys())
        for poisoned_task_idx in poisoned_speculations:
            dependents = self._window_idx_dag.successors(poisoned_task_idx)
            all_poisoned_indices = [poisoned_task_idx]
            for dependent_idx in dependents:
                dependent = self._get_task_or_none(dependent_idx)
                if dependent and (dependent.completed_decoding or dependent_idx in active_windows):
                    assert dependent.used_parent_speculations[poisoned_task_idx]
                    indices_to_reset = [dependent_idx] + list(nx.descendants(self._window_idx_dag, dependent_idx))
                    for task_idx in indices_to_reset:
                        task = self._get_task_or_none(task_idx)
                        if task:
                            all_poisoned_indices.append(task_idx)
                            if task.completed_decoding:
                                self._window_decoding_completion_times.pop(task_idx)
                                task.completed_decoding = False
                            elif task_idx in self._active_window_progress:
                                self._active_window_progress.pop(task_idx)
                            self._pending_decode_tasks.add(task_idx)
                            # TODO: how exactly does a wrong speculation actually
                            # affect descendants? Do we need to completely re-do
                            # decoding of grandchildren?
            self._missed_speculation_events.append((self._current_round, all_poisoned_indices))

        self._current_round += 1
        self._parallel_processes_by_round.append(len(self._active_window_progress))
        self._completed_windows_by_round.append(len(self._window_decoding_completion_times))

    def update_decoding(self, new_windows: dict[DecodingWindow, int], window_idx_dag: nx.DiGraph) -> None:
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
        # completed_windows = set(self._window_completion_times.keys())
        # unprocessed_windows = set(all_windows) - completed_windows
        self._window_idx_dag = window_idx_dag
        new_task_indices = set(new_windows.values())
        new_tasks = [DecoderTask(window=window, window_idx=window_idx) for window, window_idx in new_windows.items()]
        self._pending_decode_tasks |= set(new_task_indices)
        self._pending_speculate_tasks |= set(new_task_indices)
        
        new_task_len = max(len(self._tasks_by_idx), max((task.window_idx for task in new_tasks), default=0) + 1)
        if len(self._tasks_by_idx) < new_task_len:
            self._tasks_by_idx += [None] * (new_task_len - len(self._tasks_by_idx))
        for task in new_tasks:
            self._tasks_by_idx[task.window_idx] = task
        unprocessed_task_indices = self._pending_decode_tasks | self._pending_speculate_tasks
        while len(unprocessed_task_indices) > 0:
            task_idx = unprocessed_task_indices.pop()
            task = self._get_task(task_idx)
            assert task.window_idx == task_idx

            if task.completed_decoding or not task.window.constructed:
                continue

            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            # begin a speculation
            if task_idx in self._pending_speculate_tasks and self.speculation_mode == 'separate' and not self._completed_decoding(task_idx):
                assert task_idx not in self._active_speculation_progress
                self._pending_speculate_tasks.remove(task_idx)
                self._window_speculation_start_times[task_idx] = self._current_round
                if self.speculation_time > 0:
                    self._active_speculation_progress[task_idx] = self.speculation_time
                else:
                    self._get_task(task_idx).completed_speculation = True
                    unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
            
            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            if not self._completed_decoding(task_idx) and task_idx not in self._active_window_progress:
                # window has not been processed yet
                parents = list(self._window_idx_dag.predecessors(task.window_idx))
                if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents):
                    continue
                # begin decoding
                assert task_idx in self._pending_decode_tasks
                self._pending_decode_tasks.remove(task_idx)
                self._window_decoding_start_times[task_idx] = self._current_round
                self._active_window_progress[task_idx] = self.decoding_time_function(task.window.total_spacetime_volume())
                task.used_parent_speculations = {}
                for parent_idx in parents:
                    parent = self._tasks_by_idx[parent_idx]
                    if parent.completed_decoding:
                        task.used_parent_speculations[parent_idx] = False
                    else:
                        assert parent.completed_speculation
                        task.used_parent_speculations[parent_idx] = True
                if self.speculation_mode == 'integrated':
                    # begin a speculation along with decoding
                    self._window_speculation_start_times[task_idx] = self._current_round
                    if self.speculation_time > 0:
                        self._active_speculation_progress[task_idx] = self.speculation_time
                    else:
                        self._get_task(task_idx).completed_speculation = True
                        unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
    
    def _completed_decoding(self, task_idx: int) -> bool:
        if task_idx >= len(self._tasks_by_idx):
            return False
        task = self._tasks_by_idx[task_idx]
        if task:
            return task.completed_decoding
        return False
    
    def _completed_speculation(self, task_idx: int) -> bool:
        if task_idx >= len(self._tasks_by_idx):
            return False
        task = self._tasks_by_idx[task_idx]
        if task:
            return task.completed_speculation
        return False
    
    def _get_task(self, task_idx: int) -> DecoderTask:
        if task_idx >= len(self._tasks_by_idx):
            raise RuntimeError(f'Invalid task index: {task_idx}')
        task = self._tasks_by_idx[task_idx]
        if not task:
            raise RuntimeError(f'Invalid task index: {task_idx}')
        return task
    
    def _get_task_or_none(self, task_idx: int) -> DecoderTask | None:
        if task_idx >= len(self._tasks_by_idx):
            return None
        return self._tasks_by_idx[task_idx]

    def get_finished_instruction_indices(self) -> set[int]:
        """Return the set of instruction idx that have been decoded."""
        # TODO: send unfinished instructions instead
        decoded_instructions = self._partially_complete_instructions.copy()
        for idx in set(self._active_window_progress.keys()) | self._pending_decode_tasks:
            task = self._get_task(idx)
            decoded_instructions -= task.window.parent_instr_idx
        decoded_instructions -= {-1}
        return decoded_instructions

    def get_data(self, lightweight_output: bool = False) -> DecoderData:
        return DecoderData(
            num_rounds=self._current_round,
            max_parallel_processes=self.max_parallel_processes,
            parallel_processes_by_round=self._parallel_processes_by_round,
            completed_windows_by_round=self._completed_windows_by_round,
            window_speculation_start_times=self._window_speculation_start_times.copy(),
            window_decoding_start_times=self._window_decoding_start_times.copy(),
            window_decoding_completion_times=self._window_decoding_completion_times.copy(),
            missed_speculation_events=self._missed_speculation_events.copy(),
        )
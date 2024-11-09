from dataclasses import dataclass, field, asdict
from typing import Callable
import numpy as np
import networkx as nx
import itertools
from swiper.window_builder import DecodingWindow

@dataclass
class DecoderData:
    num_rounds: int
    max_parallel_processes: int
    num_completed_windows: int
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
    decoding_start_time: int = -1
    decoding_completion_time: int = -1
    completed_speculation: bool = False
    speculation_start_time: int = -1
    speculation_completion_time: int = -1
    used_parent_speculations: dict[int, bool] = field(default_factory=dict)

class DecoderManager:
    def __init__(
            self,
            decoding_time_function: Callable[[int], int],
            speculation_time: int,
            speculation_accuracy: float,
            max_parallel_processes: int | None = None,
            speculation_mode: str = 'integrated',
            lightweight_setting: int = 0,
            rng: int | np.random.Generator = np.random.default_rng(),
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
            speculation_mode: 'integrated', 'separate', or None. If 'integrated', the
                speculation time is included in the decoding time of a window,
                and speculation can only be performed once the decoder starts
                processing the window. If 'separate', the speculation time is
                not included in the decoding, time of a window, and speculation
                can be run independently of decoding. In this case, speculation
                uses a parallel process and counts towards
                max_parallel_processes. If None, no speculation is performed.
            rng: Random number generator, or integer seed.
        """
        self.decoding_time_function = decoding_time_function
        self.speculation_time = speculation_time
        self.speculation_accuracy = speculation_accuracy
        self.speculation_mode = speculation_mode
        assert self.speculation_mode in ['integrated', 'separate', None]
        self.lightweight_setting = lightweight_setting
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng

        self.max_parallel_processes = max_parallel_processes
        self._max_parallel_processes_used = 0
        self._parallel_processes_by_round: list[int] = []
        self._completed_windows_by_round: list[int] = []
        self._num_completed_windows = 0
        self._current_round = 0
        self._missed_speculation_events: list[tuple[int, list[int]]] = []
        self._active_speculation_progress: dict[int, int] = {} # maps each speculated window to rounds remaining to complete speculation
        self._active_window_progress: dict[int, int] = {} # maps each active window to rounds remaining to complete decoding
        self._pending_decode_tasks: set[int] = set()
        self._pending_speculate_tasks: set[int] = set()
        self._finalized_frontier: set[int] = set()
        self._tasks_by_idx: list[DecoderTask | None] = []
        self._partially_complete_instructions: set[int] = set()

        self._window_idx_dag = nx.DiGraph()

    def step(self) -> list[int]:
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
                if self.speculation_mode:
                    for successor_idx in self._window_idx_dag.successors(task_idx):
                        successor = self._get_task_or_none(successor_idx)
                        if successor and successor.decoding_start_time != -1 and self.rng.random() > 1-(1-self.speculation_accuracy)**self._get_task(successor_idx).window.count_touching_faces(self._get_task(task_idx).window):
                            # Missed speculation
                            assert successor.used_parent_speculations[task_idx]
                            poisoned_speculations.append(successor_idx)

        self._num_completed_windows += len(completed_windows)
        for task_idx in completed_windows:
            task = self._get_task(task_idx)
            # self._window_decoding_completion_times[task_idx] = self._current_round
            task.decoding_completion_time = self._current_round
            self._active_window_progress.pop(task_idx)
            task.completed_decoding = True
            self._partially_complete_instructions |= task.window.parent_instr_idx

        # Step speculation forward
        if self.speculation_mode:
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
            all_poisoned_indices = []
            for poisoned_task_idx in poisoned_speculations:
                poisoned_task = self._get_task_or_none(poisoned_task_idx)
                if poisoned_task and poisoned_task.decoding_start_time != -1:
                    if poisoned_task.completed_decoding:
                        all_poisoned_indices += self._poisoned_task_reset_children_that_used_decoding(poisoned_task_idx)
                    self._reset_decode_task(poisoned_task_idx)
                    all_poisoned_indices.append(poisoned_task_idx)
            if self.lightweight_setting == 0:
                self._missed_speculation_events.append((self._current_round, all_poisoned_indices))

        self._current_round += 1
        if self.lightweight_setting == 0:
            self._parallel_processes_by_round.append(len(self._active_window_progress))
            self._completed_windows_by_round.append((self._completed_windows_by_round[-1] if self._current_round > 1 else 0) + len(completed_windows))
        self._max_parallel_processes_used = max(self._max_parallel_processes_used, len(self._active_window_progress))

        if self.lightweight_setting == 3:
            deleted_task_indices = self._advance_finalized_frontier()
            return deleted_task_indices
        return []

    def _advance_finalized_frontier(self):
        """Advance the frontier of finalized windows, and remove any windows
        that are no longer needed.
        
        A window is considered finalized when it and all of its ancestors have
        completed decoding. This means we will not have any missed speculations
        happen in the future that would impact it. Every task in
        self._finalized_frontier is guaranteed to have all of its ancestors
        finalized.
        """
        assert self.lightweight_setting == 3
        finalized_task_indices = []
        change_made = True
        while change_made and self._finalized_frontier:
            change_made = False
            task_idx = self._finalized_frontier.pop()
            task = self._get_task_or_none(task_idx)
            if task and task.completed_decoding:
                finalized_task_indices.append(task_idx)
                for successor in self._window_idx_dag.successors(task_idx):
                    # print(f'Checking successor {successor} of task {task_idx}')
                    # if not self._get_task_or_none(successor):
                    #     assert self._get_task_or_none(successor)
                    if all(self._get_task(parent_idx).completed_decoding for parent_idx in nx.ancestors(self._window_idx_dag, successor) if self._get_task_or_none(parent_idx)):
                        self._finalized_frontier.add(successor)
                change_made = True
            else:
                self._finalized_frontier.add(task_idx)
        for task_idx in finalized_task_indices:
            # print(f'Finalized task {task_idx}')
            self._tasks_by_idx[task_idx] = None
            # self._window_idx_dag.remove_node(task_idx)
        return finalized_task_indices

    def _reset_decode_task(self, task_idx):
        task = self._get_task(task_idx)
        task.decoding_start_time = -1
        if task_idx in self._active_window_progress:
            self._active_window_progress.pop(task_idx)
        else:
            # self._window_decoding_completion_times.pop(task_idx)
            task.decoding_completion_time = -1
            self._num_completed_windows -= 1
            task.completed_decoding = False
        task.used_parent_speculations = {}
        self._pending_decode_tasks.add(task_idx)
        assert task_idx not in self._active_window_progress and task.decoding_start_time == -1 and task.decoding_completion_time == -1 and not task.completed_decoding
        return task_idx

    def _poisoned_task_reset_children_that_used_decoding(self, task_idx):
        """Recursively reset children of a completed-and-then-poisoned task (if
        children used the completed decoding result).
        """
        poisoned_indices = []
        for child_idx in self._window_idx_dag.successors(task_idx):
            child = self._get_task_or_none(child_idx)
            if child and child.decoding_start_time != -1 and not child.used_parent_speculations[task_idx]:
                if child.completed_decoding:
                    poisoned_indices += self._poisoned_task_reset_children_that_used_decoding(child_idx)
                self._reset_decode_task(child_idx)
        return poisoned_indices

    def update_decoding(self, new_windows: list[DecodingWindow], window_idx_dag: nx.DiGraph) -> None:
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
        self._window_idx_dag = window_idx_dag
        new_task_indices = set(w.window_idx for w in new_windows)
        new_tasks = [DecoderTask(window=window, window_idx=window.window_idx) for window in new_windows]
        self._pending_decode_tasks |= new_task_indices
        if self.speculation_mode:
            self._pending_speculate_tasks |= new_task_indices
        
        new_task_len = max(len(self._tasks_by_idx), max((task.window_idx for task in new_tasks), default=-1) + 1)
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
                raise RuntimeError(f'Task is complete, but marked as unprocessed: {task_idx}')

            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            # begin a speculation
            if task_idx in self._pending_speculate_tasks and self.speculation_mode == 'separate' and not self._completed_decoding(task_idx):
                assert task_idx not in self._active_speculation_progress
                self._pending_speculate_tasks.remove(task_idx)
                task.speculation_start_time = self._current_round
                if self.speculation_time > 0:
                    self._active_speculation_progress[task_idx] = self.speculation_time
                else:
                    task.completed_speculation = True
                    task.speculation_completion_time = self._current_round
                    unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
            
            if self.max_parallel_processes and len(self._active_window_progress) >= self.max_parallel_processes:
                break

            if task_idx in self._pending_decode_tasks:
                # window has not been processed yet
                parents = list(self._window_idx_dag.predecessors(task.window_idx))
                if any(not (self._completed_decoding(parent_idx) or self._completed_speculation(parent_idx)) for parent_idx in parents):
                    continue
                # begin decoding
                assert not self._completed_decoding(task_idx) and task_idx not in self._active_window_progress
                self._pending_decode_tasks.remove(task_idx)
                task.decoding_start_time = self._current_round
                self._active_window_progress[task_idx] = self.decoding_time_function(task.window.total_spacetime_volume())
                if self.lightweight_setting == 3 and not any(ancestor_idx in self._finalized_frontier for ancestor_idx in nx.ancestors(self._window_idx_dag, task_idx)):
                    self._finalized_frontier.add(task_idx)
                task.used_parent_speculations = {}
                for parent_idx in parents:
                    parent = self._get_task(parent_idx)
                    if parent.completed_decoding:
                        task.used_parent_speculations[parent_idx] = False
                    else:
                        assert parent.completed_speculation
                        task.used_parent_speculations[parent_idx] = True
                if self.speculation_mode == 'integrated' and task_idx in self._pending_speculate_tasks:
                    # begin a speculation along with decoding
                    assert task_idx not in self._active_speculation_progress
                    self._pending_speculate_tasks.remove(task_idx)
                    task.speculation_start_time = self._current_round
                    if self.speculation_time > 0:
                        self._active_speculation_progress[task_idx] = self.speculation_time
                    else:
                        self._get_task(task_idx).completed_speculation = True
                        unprocessed_task_indices |= {w_idx for w_idx in self._window_idx_dag.successors(task.window_idx) if not (self._get_task_or_none(w_idx) is None or self._get_task(w_idx).completed_decoding)}
    
    def _completed_decoding(self, task_idx: int) -> bool:
        task = self._get_task_or_none(task_idx)
        if task:
            return task.completed_decoding
        return False
    
    def _completed_speculation(self, task_idx: int) -> bool:
        task = self._get_task_or_none(task_idx)
        if task:
            return task.completed_speculation
        return False

    def _get_task(self, task_idx: int) -> DecoderTask:
        if task_idx >= len(self._tasks_by_idx):
            raise RuntimeError(f'Invalid task index: {task_idx}')
        task = self._tasks_by_idx[task_idx]
        if task is None:
            raise RuntimeError(f'Invalid or deleted task index: {task_idx}')
        else:
            return task
    
    def _get_task_or_none(self, task_idx: int) -> DecoderTask | None:
        if task_idx >= len(self._tasks_by_idx):
            return None
        task = self._tasks_by_idx[task_idx]
        if task is None:
            return None
        return task

    def get_finished_instruction_indices(self) -> set[int]:
        """Return the set of instruction idx that have been decoded."""
        # TODO: send unfinished instructions instead
        decoded_instructions = self._partially_complete_instructions.copy()
        not_ready_instructions = set(itertools.chain.from_iterable(self._get_task(idx).window.parent_instr_idx for idx in itertools.chain.from_iterable(nx.descendants(self._window_idx_dag, idx) for idx in set(self._active_window_progress.keys()) | self._pending_decode_tasks) if idx < len(self._tasks_by_idx) and self._tasks_by_idx[idx]))
        decoded_instructions -= not_ready_instructions
        decoded_instructions -= {-1}
        return decoded_instructions
    
    def get_incomplete_instruction_indices(self) -> set[int]:
        """Return the set of instruction idx that have not been decoded."""
        incomplete_task_instructions = set(itertools.chain.from_iterable(self._get_task(idx).window.parent_instr_idx for idx in set(self._active_window_progress.keys()) | self._pending_decode_tasks))
        incomplete_descendant_instructions = set(itertools.chain.from_iterable(self._get_task(idx).window.parent_instr_idx for idx in itertools.chain.from_iterable(nx.descendants(self._window_idx_dag, task_idx) for task_idx in set(self._active_window_progress.keys()) | self._pending_decode_tasks) if idx < len(self._tasks_by_idx) and self._tasks_by_idx[idx]))
        return incomplete_task_instructions | incomplete_descendant_instructions

    def get_data(self) -> DecoderData:
        if self.lightweight_setting == 0:
            return DecoderData(
                num_rounds=self._current_round,
                max_parallel_processes=self._max_parallel_processes_used,
                num_completed_windows=self._num_completed_windows,
                parallel_processes_by_round=self._parallel_processes_by_round,
                completed_windows_by_round=self._completed_windows_by_round,
                window_speculation_start_times={task_idx:task.speculation_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_start_times={task_idx:task.decoding_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_completion_times={task_idx:task.decoding_completion_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                missed_speculation_events=self._missed_speculation_events,
            )
        elif self.lightweight_setting == 1:
            return DecoderData(
                num_rounds=self._current_round,
                max_parallel_processes=self._max_parallel_processes_used,
                num_completed_windows=self._num_completed_windows,
                parallel_processes_by_round=None,
                completed_windows_by_round=None,
                window_speculation_start_times={task_idx:task.speculation_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_start_times={task_idx:task.decoding_start_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                window_decoding_completion_times={task_idx:task.decoding_completion_time for task_idx,task in enumerate(self._tasks_by_idx) if task},
                missed_speculation_events=self._missed_speculation_events,
            )
        elif self.lightweight_setting == 2 or self.lightweight_setting == 3:
            return DecoderData(
                num_rounds=self._current_round,
                max_parallel_processes=self._max_parallel_processes_used,
                num_completed_windows=self._num_completed_windows,
                parallel_processes_by_round=None,
                completed_windows_by_round=None,
                window_speculation_start_times=None,
                window_decoding_start_times=None,
                window_decoding_completion_times=None,
                missed_speculation_events=None,
            )
        else:
            raise RuntimeError('Invalid lightweight setting')
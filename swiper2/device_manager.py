from dataclasses import dataclass
import networkx as nx
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
import copy

@dataclass
class SyndromeRound:
    """A syndrome round for a given patch"""
    patch: tuple[int, int]
    round: int
    instruction: Instruction
    instruction_idx: int
    initialized_patch: bool
    is_unwanted_idle: bool = False
    discard_after: bool = False

@dataclass
class DeviceData:
    """Data containing the history of a device."""
    d: int
    num_rounds: int
    instructions: list[Instruction]
    all_patch_coords: set[tuple[int, int]]
    syndrome_count_by_round: NDArray[np.int_]
    instruction_count_by_round: NDArray[np.int_]
    generated_syndrome_data: list[list[SyndromeRound]]
    patches_initialized_by_round: dict[int, set[tuple[int, int]]]

class DeviceManager:
    def __init__(self, d_t: int, schedule: LatticeSurgerySchedule, rng: int | np.random.Generator = np.random.default_rng()):
        """TODO

        Args:
            d_t: Temporal distance of the code.
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
        """
        self.d_t = d_t
        self.schedule = schedule
        self.schedule_dag = schedule.to_dag()
        self._patches_initialized_by_instr = {i: self._get_initialized_patches(i) for i in range(len(self.schedule.all_instructions))}
        self.current_round = 0

        self._syndrome_count_by_round = []
        self._instruction_count_by_round = []
        self._all_patch_coords = set()
        self._generated_syndrome_data = []

        self._completed_instructions = dict()
        self._active_instructions = dict()
        self._active_patches = set()
        self._instruction_frontier = set()

        if isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

        self._instruction_durations = [self._get_duration(i) for i in range(len(self.schedule.all_instructions))]
        
        # Begin by starting the first instruction
        first_instruction_idx = self._find_first_instruction_idx()
        self._active_instructions[first_instruction_idx] = self._instruction_durations[first_instruction_idx]
        self._update_active_instructions()

    def _get_duration(self, instruction_idx: int) -> int:
        """Return the duration of an instruction."""
        if self.schedule.all_instructions[instruction_idx].name == 'CONDITIONAL_S':
            if self.rng.random() < 0.5:
                return 0

        duration = self.schedule.all_instructions[instruction_idx].duration
        if isinstance(duration, int):
            return duration
        elif duration == Duration.HALF_D_ROUNDS:
            return self.d_t // 2 + 2
        elif duration == Duration.D_ROUNDS:
            return self.d_t
        elif duration == Duration.HALF_D_ROUNDS_ROUNDED_DOWN:
            return self.d_t // 2
        elif duration == Duration.HALF_D_ROUNDS_ROUNDED_UP:
            return self.d_t // 2 + 2
        else:
            raise ValueError(f"Invalid instruction duration: {self.schedule.all_instructions[instruction_idx].duration}")

    def _get_initialized_patches(self, instruction_idx: int) -> set[tuple[int, int]]:
        """Return the set of patches initialized by an instruction."""
        if instruction_idx == 0:
            return set(self.schedule.all_instructions[0].patches)
        
        active_patches = set()
        for i in range(instruction_idx):
            if self.schedule.all_instructions[i].name == 'DISCARD':
                active_patches -= set(self.schedule.all_instructions[i].patches)
            else:
                active_patches.update(self.schedule.all_instructions[i].patches)
        return set(self.schedule.all_instructions[instruction_idx].patches) - active_patches

    def _is_startup_instruction(self, instruction_idx: int) -> bool:
        """Return whether an instruction is a startup instruction."""
        return len(self._patches_initialized_by_instr[instruction_idx]) == len(self.schedule.all_instructions[instruction_idx].patches)

    def _find_first_instruction_idx(self) -> int:
        schedule_longest_path = nx.dag_longest_path(self.schedule.to_dag(d=self.d_t, dummy_final_node=True))
        return schedule_longest_path[0]

    def _predict_instruction_start_times(self):
        """For each not-yet-started instruction in the frontier, get number of
        rounds from now at which we expect it to begin, assuming no unexpected
        delays happen. Instructions which should be started immediately are
        assigned round 0.
        """
        first_round = dict()
        after_last_round = dict()
        instruction_queue = list(range(len(self.schedule.all_instructions)))
        while len(instruction_queue) > 0:
            instruction_idx = instruction_queue.pop(0)
            is_startup_instruction = self._is_startup_instruction(instruction_idx)
            if instruction_idx in self._completed_instructions:
                # already completed
                first_round[instruction_idx] = self._completed_instructions[instruction_idx] - self._instruction_durations[instruction_idx] + 1
                after_last_round[instruction_idx] = self._completed_instructions[instruction_idx] + 1
            elif instruction_idx in self._active_instructions:
                # currently active
                first_round[instruction_idx] = self._active_instructions[instruction_idx] + self.current_round - self._instruction_durations[instruction_idx]
                after_last_round[instruction_idx] = self._active_instructions[instruction_idx] + self.current_round
            elif is_startup_instruction and all(inst_idx in first_round for inst_idx in self.schedule_dag.successors(instruction_idx)):
                # if startup instruction; schedule ALAP (before soonest successor)
                successor_first_rounds = [first_round[inst_idx] for inst_idx in self.schedule_dag.successors(instruction_idx)]
                first_round[instruction_idx] = min(successor_first_rounds, default=0) - self._instruction_durations[instruction_idx]
                first_round[instruction_idx] = max(first_round[instruction_idx], self.current_round)
                after_last_round[instruction_idx] = first_round[instruction_idx] + self._instruction_durations[instruction_idx]
            elif not is_startup_instruction and all(inst_idx in after_last_round for inst_idx in self.schedule_dag.predecessors(instruction_idx) if not self._is_startup_instruction(inst_idx)):
                # standard operation; schedule ASAP (after last predecessor)
                predecessor_last_rounds = []
                for inst_idx in self.schedule_dag.predecessors(instruction_idx):
                    if inst_idx in after_last_round:
                        predecessor_last_rounds.append(after_last_round[inst_idx])
                    else:
                        # end time not specified yet; only ok if it is a startup instruction
                        assert self._is_startup_instruction(inst_idx)
                first_round[instruction_idx] = max(predecessor_last_rounds, default=0)
                first_round[instruction_idx] = max(first_round[instruction_idx], self.current_round)
                after_last_round[instruction_idx] = first_round[instruction_idx] + self._instruction_durations[instruction_idx]
            else:
                # not ready to be processed; push to back
                instruction_queue = instruction_queue + [instruction_idx]

        return first_round, after_last_round

    def _update_active_instructions(self, fully_decoded_instructions: set[int] = set()) -> None:
        """Add new instructions to the active set if they are ready to start.
        Immediately complete any instructions with duration 0. Instructions with
        conditional dependencies cannot be started if any of the instructions
        they are conditioned on are still being decoded.

        Args:
            fully_decoded_instructions: Set of instruction indices whose data
                has been fully decoded.
        """ 
        patches_in_use = set()
        for instruction_idx in self._active_instructions.keys():
            patches_in_use.update(self.schedule.all_instructions[instruction_idx].patches)

        start_times, finish_times = self._predict_instruction_start_times()
        for instruction_idx, start_time in start_times.items():
            if instruction_idx not in self._active_instructions and instruction_idx not in self._completed_instructions and start_time <= self.current_round:
                if set(self.schedule.all_instructions[instruction_idx].patches) & patches_in_use:
                    # at least one patch is already in use
                    continue
                elif not self.schedule.all_instructions[instruction_idx].conditioned_on_idx.issubset(fully_decoded_instructions):
                    # decoding dependency not yet satisfied
                    continue
                elif not self.schedule.all_instructions[instruction_idx].conditioned_on_completion_idx.issubset(set(self._completed_instructions.keys())):
                    # dependency not yet satisfied
                    continue
                elif set(self.schedule_dag.predecessors(instruction_idx)) - set(self._completed_instructions.keys()):
                    # not all predecessors have been completed
                    continue
                elif finish_times[instruction_idx] == self.current_round:
                    # zero-duration instruction; complete immediately
                    self._completed_instructions[instruction_idx] = self.current_round
                    if self.schedule.all_instructions[instruction_idx].name == 'DISCARD':
                        self._active_patches -= set(self.schedule.all_instructions[instruction_idx].patches)
                else:
                    self._active_instructions[instruction_idx] = self._instruction_durations[instruction_idx]
                    patches_in_use.update(self.schedule.all_instructions[instruction_idx].patches)

    def _generate_syndrome_round(self) -> tuple[list[SyndromeRound], set[int]]:
        generated_syndrome_rounds = []

        self._instruction_count_by_round.append(0)
        patches_used_this_round = set()
        completed_instructions = set()
        for instruction_idx in self._active_instructions.keys():
            assert self._active_instructions[instruction_idx] > 0
            generated_syndrome_rounds.extend([
                SyndromeRound(coords, 
                              self.current_round, 
                              self.schedule.all_instructions[instruction_idx], 
                              instruction_idx,
                              initialized_patch=(coords not in self._active_patches)) 
                for coords in self.schedule.all_instructions[instruction_idx].patches
                ])
            patches_used_this_round.update(self.schedule.all_instructions[instruction_idx].patches)
            self._active_patches.update(self.schedule.all_instructions[instruction_idx].patches)
            self._active_instructions[instruction_idx] -= 1
            self._instruction_count_by_round[-1] += 1
            if self._active_instructions[instruction_idx] == 0:
                completed_instructions.add(instruction_idx)
        self._all_patch_coords.update(patches_used_this_round)

        generated_syndrome_rounds.extend([
            SyndromeRound(coords, 
                          self.current_round, 
                          Instruction('UNWANTED_IDLE', frozenset([coords]), 1), 
                          -1,
                          initialized_patch=False, 
                          is_unwanted_idle=True) 
            for coords in self._active_patches - patches_used_this_round
            ])
        self._instruction_count_by_round[-1] += len(self._active_patches - patches_used_this_round)
        self._syndrome_count_by_round.append(len(generated_syndrome_rounds))
        self._generated_syndrome_data.append(generated_syndrome_rounds)

        return generated_syndrome_rounds, completed_instructions
    
    def _clean_completed_instructions(self, completed_instructions: set[int] = set()):
        for instruction_idx in completed_instructions:
            self._completed_instructions[instruction_idx] = self.current_round
            self._active_instructions.pop(instruction_idx)

    def get_next_round(self, fully_decoded_instructions: set[int]) -> tuple[list[SyndromeRound], set[tuple[int, int]]]:
        """Return another round of syndrome measurements, starting new
        instructions if possible.

        Args:
            fully_decoded_instructions: Set of instruction indices whose data
                has been fully decoded.
        
        Returns:
            generated_syndrome_rounds: List of SyndromeRound objects for the
                current round.
            discarded_patches: Set of patches that were discarded after the
                current round.
        """
        if self.is_done():
            return [], set()

        # self._update_active_instructions(fully_decoded_instructions)
        init_active_patches = copy.deepcopy(self._active_patches)
        generated_syndrome_rounds, completed_instructions = self._generate_syndrome_round()
        self._clean_completed_instructions(completed_instructions)

        if not self.is_done():
            self.current_round += 1

        self._update_active_instructions(fully_decoded_instructions)
        discarded_patches = init_active_patches - self._active_patches
    
        return generated_syndrome_rounds, discarded_patches
    
    def is_done(self) -> bool:
        """Return whether all instructions have been completed."""
        return len(self._active_instructions) == 0 and len(self._completed_instructions) == len(self.schedule.all_instructions)
    
    def get_data(self):
        """Return all relevant dataregarding device history."""
        patches_initialized_by_round = {round_idx: set() for round_idx in range(self.current_round+2)}
        for instr, round_idx in self._predict_instruction_start_times()[0].items():
            if round_idx <= self.current_round:
                patches_initialized_by_round[round_idx] |= self._patches_initialized_by_instr[instr]

        return DeviceData(
            d=self.d_t,
            num_rounds=self.current_round,
            instructions=copy.deepcopy(self.schedule.all_instructions),
            all_patch_coords=self._all_patch_coords,
            syndrome_count_by_round=np.array(self._syndrome_count_by_round, int),
            instruction_count_by_round=np.array(self._instruction_count_by_round, int),
            generated_syndrome_data=self._generated_syndrome_data,
            patches_initialized_by_round=patches_initialized_by_round,
        )
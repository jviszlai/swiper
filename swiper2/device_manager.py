from dataclasses import dataclass
import networkx as nx
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy

@dataclass
class SyndromeRound:
    """A syndrome round for a given patch"""
    patch: tuple[int, int]
    round: int
    instruction: Instruction
    is_unwanted_idle: bool = False

@dataclass
class DeviceData:
    """Data containing the history of a device."""
    d: int
    num_rounds: int
    instructions: list[Instruction]
    all_patch_coords: set[tuple[int, int]]
    syndrome_count_by_round: list[int]
    instruction_count_by_round: list[int]
    generated_syndrome_data: list[list[SyndromeRound]]

class DeviceManager:
    def __init__(self, d_t: int, schedule: LatticeSurgerySchedule):
        """TODO

        Args:
            d_t: Temporal distance of the code.
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
        """
        self.d_t = d_t
        self.schedule = schedule
        self.schedule_dag = schedule.to_dag()
        self._is_startup_instruction_dict = {i: self._is_startup_instruction(i) for i in range(len(self.schedule.all_instructions))}
        self.current_round = 0

        self._syndrome_count_by_round = []
        self._instruction_count_by_round = []
        self._all_patch_coords = set()
        self._generated_syndrome_data = []

        self._completed_instructions = dict()
        self._active_instructions = dict()
        self._active_patches = set()
        self._instruction_frontier = set()

        self._active_instructions[0] = self._get_duration(0)
        self._update_active_instructions()

    def _get_duration(self, instruction_idx: int) -> int:
        """Return the duration of an instruction."""
        duration = self.schedule.all_instructions[instruction_idx].duration
        if isinstance(duration, int):
            return duration
        elif duration == Duration.HALF_D_ROUNDS:
            return self.d_t // 2 + 2
        elif duration == Duration.D_ROUNDS:
            return self.d_t
        else:
            raise ValueError(f"Invalid instruction duration: {self.schedule.all_instructions[instruction_idx].duration}")

    def _is_startup_instruction(self, instruction_idx: int) -> int:
        """Return whether an instruction initializes its patch. This can be true
        if it is either the first instruction to use the patch, or if it follows
        a DISCARD instruction on the same patch.
        """
        if instruction_idx == 0:
            return True
        
        active_patches = set()
        for i in range(instruction_idx):
            if self.schedule.all_instructions[i].name == 'DISCARD':
                active_patches -= set(self.schedule.all_instructions[i].patches)
            else:
                active_patches.update(self.schedule.all_instructions[i].patches)
        if set(self.schedule.all_instructions[instruction_idx].patches) & active_patches:
            return False
        return True
    
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
            is_startup_instruction = self._is_startup_instruction_dict[instruction_idx]
            if instruction_idx in self._completed_instructions:
                # already completed
                first_round[instruction_idx] = self._completed_instructions[instruction_idx] - self._get_duration(instruction_idx) + 1
                after_last_round[instruction_idx] = self._completed_instructions[instruction_idx] + 1
            elif instruction_idx in self._active_instructions:
                # currently active
                first_round[instruction_idx] = self._active_instructions[instruction_idx] + self.current_round - self._get_duration(instruction_idx)
                after_last_round[instruction_idx] = self._active_instructions[instruction_idx] + self.current_round
            elif is_startup_instruction and all(inst_idx in first_round for inst_idx in self.schedule_dag.successors(instruction_idx)):
                # if startup instruction; schedule ALAP (before soonest successor)
                successor_first_rounds = [first_round[inst_idx] for inst_idx in self.schedule_dag.successors(instruction_idx)]
                first_round[instruction_idx] = min(successor_first_rounds, default=0) - self._get_duration(instruction_idx)
                first_round[instruction_idx] = max(first_round[instruction_idx], self.current_round)
                after_last_round[instruction_idx] = first_round[instruction_idx] + self._get_duration(instruction_idx)
            elif not is_startup_instruction and all(inst_idx in after_last_round for inst_idx in self.schedule_dag.predecessors(instruction_idx) if not self._is_startup_instruction_dict[inst_idx]):
                # standard operation; schedule ASAP (after last predecessor)
                predecessor_last_rounds = []
                for inst_idx in self.schedule_dag.predecessors(instruction_idx):
                    if inst_idx in after_last_round:
                        predecessor_last_rounds.append(after_last_round[inst_idx])
                    else:
                        assert self._is_startup_instruction_dict[inst_idx]
                first_round[instruction_idx] = max(predecessor_last_rounds, default=0)
                first_round[instruction_idx] = max(first_round[instruction_idx], self.current_round)
                after_last_round[instruction_idx] = first_round[instruction_idx] + self._get_duration(instruction_idx)
            else:
                # not ready to be processed; push to back
                instruction_queue = instruction_queue + [instruction_idx]

        return first_round, after_last_round

    def _update_active_instructions(self, not_yet_decoded_instructions: set[int] = set()):
        """Add new instructions to the active set if they are ready to start.
        Immediately complete any instructions with duration 0. Instructions with
        conditional dependencies cannot be started if any of the instructions
        they are conditioned on are still being decoded.

        Args:
            not_yet_decoded_instructions: Set of instruction indices that
                are not yet finished decoding (either pending or active).
        """ 
        start_times, finish_times = self._predict_instruction_start_times()
        for instruction_idx, start_time in start_times.items():
            if start_time == self.current_round:
                if self.schedule.all_instructions[instruction_idx].conditioned_on_idx is not None and self.schedule.all_instructions[instruction_idx].conditioned_on_idx in not_yet_decoded_instructions:
                    continue
                elif finish_times[instruction_idx] == self.current_round:
                    self._completed_instructions[instruction_idx] = self.current_round
                    if self.schedule.all_instructions[instruction_idx].name == 'DISCARD':
                        self._active_patches -= set(self.schedule.all_instructions[instruction_idx].patches)
                else:
                    self._active_instructions[instruction_idx] = self._get_duration(instruction_idx)

    def get_next_round(self, not_yet_decoded_instructions: set[int]) -> list[SyndromeRound] | None:
        """Return another round of syndrome measurements, starting new
        instructions if possible.

        Args:
            not_yet_decoded_instructions: Set of instruction indices that
                are not yet finished decoding (either pending or active).
        
        Returns:
            List of tuples, each containing a syndrome measurement coordinate,
            the current round index, and the index of the instruction that it is
            associated with.
        """
        if self.is_done():
            return None

        generated_syndrome_rounds = []

        self._instruction_count_by_round.append(0)
        patches_used_this_round = set()
        completed_instructions = set()
        for instruction_idx in self._active_instructions.keys():
            assert self._active_instructions[instruction_idx] > 0
            instruction = self.schedule.all_instructions[instruction_idx]
            generated_syndrome_rounds.extend([SyndromeRound(coords, self.current_round, instruction) for coords in instruction.patches])
            patches_used_this_round.update(self.schedule.all_instructions[instruction_idx].patches)
            self._active_patches.update(self.schedule.all_instructions[instruction_idx].patches)
            self._active_instructions[instruction_idx] -= 1
            self._instruction_count_by_round[-1] += 1
            if self._active_instructions[instruction_idx] == 0:
                completed_instructions.add(instruction_idx)
        self._all_patch_coords.update(patches_used_this_round)

        for instruction_idx in completed_instructions:
            self._completed_instructions[instruction_idx] = self.current_round
            self._active_instructions.pop(instruction_idx)

        # any unused (but active) patches must idle
        generated_syndrome_rounds.extend([SyndromeRound(coords, self.current_round, -1, is_unwanted_idle=True) for coords in self._active_patches - patches_used_this_round])
        self._instruction_count_by_round[-1] += len(self._active_patches - patches_used_this_round)
    
        self._syndrome_count_by_round.append(len(generated_syndrome_rounds))
        self._generated_syndrome_data.append(generated_syndrome_rounds)
        
        self.current_round += 1
        self._update_active_instructions(not_yet_decoded_instructions)

        return generated_syndrome_rounds
    
    def is_done(self) -> bool:
        """Return whether all instructions have been completed."""
        return len(self._active_instructions) == 0 and len(self._completed_instructions) == len(self.schedule.all_instructions)
    
    def get_data(self):
        """Return all relevant dataregarding device history."""
        return DeviceData(
            d=self.d_t,
            num_rounds=self.current_round,
            instructions=copy.deepcopy(self.schedule.all_instructions),
            all_patch_coords=self._all_patch_coords,
            syndrome_count_by_round=self._syndrome_count_by_round,
            instruction_count_by_round=self._instruction_count_by_round,
            generated_syndrome_data=self._generated_syndrome_data,
        )
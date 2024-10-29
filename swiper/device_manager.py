from dataclasses import dataclass, asdict
import networkx as nx
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule, Duration, Instruction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
import copy
import random

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

    def __repr__(self):
        return f'SyndromeRound({self.patch}, r={self.round}, instr={self.instruction_idx}, init={self.initialized_patch}, discard={self.discard_after})'

@dataclass
class DeviceData:
    """Data containing the history of a device."""
    d: int
    num_rounds: int
    instructions: list[Instruction]
    instruction_start_times: list[int]
    all_patch_coords: list[tuple[int, int]]
    syndrome_count_by_round: list[int]
    instruction_count_by_round: list[int]
    generated_syndrome_data: list[list[SyndromeRound]]
    patches_initialized_by_round: dict[int, list[tuple[int, int]]]

    def to_dict(self):
        return asdict(self)

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
        self.schedule_instructions = schedule.full_instructions()
        self.schedule_dag = schedule.to_dag(self.d_t)
        self._patches_initialized_by_instr = {i: self._get_initialized_patches(i) for i in range(len(self.schedule_instructions))}
        self.current_round = 0

        self._syndrome_count_by_round = []
        self._instruction_count_by_round = []
        self._all_patch_coords = set()
        self._generated_syndrome_data = []

        self._completed_instructions = dict()
        self._active_instructions = dict()
        self._active_patches = set()

        if isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

        self._instruction_durations: list[int] = [self.schedule.get_true_duration(instr.duration, self.d_t) for instr in self.schedule_instructions]
        
        # Begin by starting the first instruction
        first_instruction_idx = self._find_first_instruction_idx()
        self._active_instructions[first_instruction_idx] = self._instruction_durations[first_instruction_idx]
        self._instruction_frontier = set(next(nx.topological_generations(self.schedule_dag))) | set(self.schedule_dag.successors(first_instruction_idx)) - set([first_instruction_idx])
        self._update_active_instructions()

    def _get_initialized_patches(self, instruction_idx: int) -> set[tuple[int, int]]:
        """Return the set of patches initialized by an instruction."""
        if instruction_idx == 0:
            return set(self.schedule_instructions[0].patches)
        
        active_patches = set()
        for i in range(instruction_idx):
            if self.schedule_instructions[i].name == 'DISCARD':
                active_patches -= set(self.schedule_instructions[i].patches)
            else:
                active_patches.update(self.schedule_instructions[i].patches)
        return set(self.schedule_instructions[instruction_idx].patches) - active_patches

    def _is_startup_instruction(self, instruction_idx: int) -> bool:
        """Return whether an instruction is a startup instruction."""
        return len(self._patches_initialized_by_instr[instruction_idx]) == len(self.schedule_instructions[instruction_idx].patches)

    def _find_first_instruction_idx(self) -> int:
        schedule_longest_path = nx.dag_longest_path(self.schedule_dag)
        return schedule_longest_path[0]

    def _predict_instruction_start_time(self, instruction_idx: int, first_round: dict[int, int], recur: bool = False) -> tuple[dict[int, int], list[int]]:
        """Update first_round with the expected start time of
        instruction_idx."""
        instructions_to_process = []
        if instruction_idx in first_round:
            pass
        elif instruction_idx in self._active_instructions:
            start_time = self._active_instructions[instruction_idx] + self.current_round - self._instruction_durations[instruction_idx]
            first_round[instruction_idx] = start_time
        elif instruction_idx in self._completed_instructions:
            start_time = self._completed_instructions[instruction_idx] - self._instruction_durations[instruction_idx] + 1
            first_round[instruction_idx] = start_time
        else:
            valid = True
            expected_start = None
            if self._is_startup_instruction(instruction_idx):
                # startup instruction; schedule ALAP
                for inst_idx in self.schedule_dag.successors(instruction_idx):
                    if inst_idx in first_round:
                        if valid:
                            start = first_round[inst_idx]
                            this_start = start - self._instruction_durations[instruction_idx]
                            if not expected_start or this_start < expected_start:
                                expected_start = this_start
                    else:
                        expected_start = None
                        valid = False
                        instructions_to_process.append(inst_idx)
            else:
                # standard operation; schedule ASAP
                for inst_idx in self.schedule_dag.predecessors(instruction_idx):
                    if inst_idx in first_round:
                        if valid:
                            start = first_round[inst_idx]
                            this_start = start + self._instruction_durations[inst_idx]
                            if not expected_start or this_start > expected_start:
                                expected_start = this_start
                    elif self._is_startup_instruction(inst_idx):
                        if valid:
                            if inst_idx in self._completed_instructions:
                                start = self._completed_instructions[inst_idx]
                                this_start = start
                                if not expected_start or this_start > expected_start:
                                    expected_start = this_start
                            else:
                                expected_start = expected_start if expected_start else 0
                    else:
                        expected_start = None
                        valid = False
                        instructions_to_process.append(inst_idx)
            if expected_start is not None:
                expected_start = max(expected_start, self.current_round)
                # print(instruction_idx, expected_start)
                first_round[instruction_idx] = expected_start
            else:
                instructions_to_process.append(instruction_idx)

        if recur and len(instructions_to_process) > 0:
            while len(instructions_to_process) > 0:
                instr = instructions_to_process.pop(0)
                first_round, new_instructions_to_process = self._predict_instruction_start_time(instr, first_round, recur=True)
                instructions_to_process.extend([instr for instr in new_instructions_to_process if instr not in instructions_to_process])
        assert not recur or len(instructions_to_process) == 0
        return first_round, instructions_to_process

    def _predict_instruction_start_times(self):
        """For each not-yet-started instruction in the frontier, get number of
        rounds from now at which we expect it to begin, assuming no unexpected
        delays happen. Instructions which should be started immediately are
        assigned round 0.
        """
        first_round = dict()
        after_last_round = dict()

        instruction_queue = list(self._instruction_frontier)

        iters = 0
        while len(instruction_queue) > 0:
            iters += 1
            if iters > 10:
                pass
            if iters > 1000:
                raise Exception('Infinite loop in _predict_instruction_start_times', instruction_queue, first_round)
            instruction_idx = instruction_queue.pop(0)
            first_round, instructions_to_process = self._predict_instruction_start_time(instruction_idx, first_round, recur=True)
            assert len(instructions_to_process) == 0
            instruction_queue.extend([instr for instr in instructions_to_process if instr not in instruction_queue])
        
        return first_round

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
            patches_in_use.update(self.schedule_instructions[instruction_idx].patches)

        start_times = self._predict_instruction_start_times()
        done_with_zero_duration_instructions = False
        while not done_with_zero_duration_instructions:
            done_with_zero_duration_instructions = True
            for instruction_idx in self._instruction_frontier:
                if instruction_idx not in self._active_instructions and instruction_idx not in self._completed_instructions and start_times[instruction_idx] <= self.current_round:
                    if set(self.schedule_instructions[instruction_idx].patches) & patches_in_use:
                        # at least one patch is already in use
                        pass
                    elif not self.schedule_instructions[instruction_idx].conditioned_on_idx.issubset(fully_decoded_instructions):
                        # decoding dependency not yet satisfied
                        pass
                    elif not self.schedule_instructions[instruction_idx].conditioned_on_completion_idx.issubset(set(self._completed_instructions.keys())):
                        # dependency not yet satisfied
                        pass
                    elif set(self.schedule_dag.predecessors(instruction_idx)) - set(self._completed_instructions.keys()):
                        # not all predecessors have been completed
                        pass
                    elif self._instruction_durations[instruction_idx] == 0:
                        done_with_zero_duration_instructions = False
                        assert instruction_idx in self._instruction_frontier, (instruction_idx, self._instruction_frontier, self._completed_instructions)
                        self._completed_instructions[instruction_idx] = self.current_round - 1
                        if self.schedule_instructions[instruction_idx].name == 'DISCARD':
                            self._active_patches -= set(self.schedule_instructions[instruction_idx].patches)
                        self._instruction_frontier -= set([instruction_idx])
                        new_instructions = set(self.schedule_dag.successors(instruction_idx)) - self._instruction_frontier
                        for instr in new_instructions:
                            start_times, _ = self._predict_instruction_start_time(instr, start_times, recur=True)
                        self._instruction_frontier.update(new_instructions)
                        break

        for instruction_idx in self._instruction_frontier.copy():
            if instruction_idx not in self._active_instructions and instruction_idx not in self._completed_instructions and start_times[instruction_idx] <= self.current_round:
                if set(self.schedule_instructions[instruction_idx].patches) & patches_in_use:
                    # at least one patch is already in use
                    pass
                elif not self.schedule_instructions[instruction_idx].conditioned_on_idx.issubset(fully_decoded_instructions):
                    # decoding dependency not yet satisfied
                    pass
                elif not self.schedule_instructions[instruction_idx].conditioned_on_completion_idx.issubset(set(self._completed_instructions.keys())):
                    # dependency not yet satisfied
                    pass
                elif set(self.schedule_dag.predecessors(instruction_idx)) - set(self._completed_instructions.keys()):
                    # not all predecessors have been completed
                    pass
                else:
                    self._active_instructions[instruction_idx] = self._instruction_durations[instruction_idx]
                    patches_in_use.update(self.schedule_instructions[instruction_idx].patches)
                    self._instruction_frontier -= set([instruction_idx])
                    new_instructions = set(self.schedule_dag.successors(instruction_idx)) - self._instruction_frontier
                    for instr in new_instructions:
                        start_times, _ = self._predict_instruction_start_time(instr, start_times, recur=True)
                    self._instruction_frontier.update(new_instructions)

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
                              self.schedule_instructions[instruction_idx], 
                              instruction_idx,
                              initialized_patch=(coords not in self._active_patches)) 
                for coords in self.schedule_instructions[instruction_idx].patches
                ])
            patches_used_this_round.update(self.schedule_instructions[instruction_idx].patches)
            self._active_patches.update(self.schedule_instructions[instruction_idx].patches)
            self._active_instructions[instruction_idx] -= 1
            self._instruction_count_by_round[-1] += 1
            if self._active_instructions[instruction_idx] == 0:
                completed_instructions.add(instruction_idx)
        self._all_patch_coords.update(patches_used_this_round)

        generated_syndrome_rounds.extend([
            SyndromeRound(coords, 
                          self.current_round, 
                          Instruction('UNWANTED_IDLE', -1, frozenset([coords]), 1), 
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

    def get_next_round(self, fully_decoded_instructions: set[int]) -> list[SyndromeRound]:
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
            return []

        # self._update_active_instructions(fully_decoded_instructions)
        init_active_patches = copy.deepcopy(self._active_patches)
        generated_syndrome_rounds, completed_instructions = self._generate_syndrome_round()
        self._clean_completed_instructions(completed_instructions)

        if not self.is_done():
            self.current_round += 1

        self._update_active_instructions(fully_decoded_instructions)
        discarded_patches = init_active_patches - self._active_patches
        for dp in discarded_patches:
            syndrome_round = [sr for sr in generated_syndrome_rounds if sr.patch == dp][0]
            syndrome_round.discard_after = True
    
        return generated_syndrome_rounds
    
    def is_done(self) -> bool:
        """Return whether all instructions have been completed."""
        return len(self._active_instructions) == 0 and len(self._completed_instructions) == len(self.schedule_instructions)
    
    def _postprocess_idle_data(self, syndrome_data: list[list[SyndromeRound]]) -> list[list[SyndromeRound]]:
        """Rename UNWANTED_IDLE syndrome rounds to either DECODE_IDLE (if they
        happen before a conditional gate, while waiting for a decode) or IDLE
        (otherwise).
        """
        # collect continuous groups of syndrome data for each patch
        data_by_patch = {patch: [[]] for patch in self._all_patch_coords}
        for round_idx,round_data in enumerate(syndrome_data):
            used_patches = set()
            for i,sr in enumerate(round_data):
                used_patches.add(sr.patch)
                data_by_patch[sr.patch][-1].append((sr, round_idx, i))
            for patch in self._all_patch_coords - used_patches:
                if len(data_by_patch[patch]) > 0:
                    data_by_patch[patch].append([])

        for patch, patch_data in data_by_patch.items():
            for i,continuous_data in enumerate(patch_data):
                for j,(sr,_,_) in enumerate(continuous_data):
                    if sr.is_unwanted_idle:
                        patch_data[i][j][0].instruction = sr.instruction.rename('IDLE')

        for patch, patch_data in data_by_patch.items():
            for i,continuous_data in enumerate(patch_data):
                for j,(sr,_,_) in enumerate(continuous_data):
                    if sr.instruction.conditioned_on_idx:
                        for jj in range(j-1,-1,-1):
                            if patch_data[i][jj][0].is_unwanted_idle:
                                patch_data[i][jj][0].instruction = patch_data[i][jj][0].instruction.rename('DECODE_IDLE')
                                # data_by_patch[patch][i][jj] = (sr, round_idx, sr_idx, 'DECODE_IDLE')
                            else:
                                break

        # reconstruct syndrome data
        new_syndrome_data = [[None for _ in syndrome_data[i]] for i in range(len(syndrome_data))]
        for patch, patch_data in data_by_patch.items():
            for i,continuous_data in enumerate(patch_data):
                for sr,round_idx,sr_idx in continuous_data:
                    assert sr.instruction.name != 'UNWANTED_IDLE'
                    new_syndrome_data[round_idx][sr_idx] = sr

        return new_syndrome_data

    def get_data(self, lightweight_output: bool = False):
        """Return all relevant data regarding device history."""
        patches_initialized_by_round = {round_idx: set() for round_idx in range(self.current_round+2)}
        for instr, round_idx in self._predict_instruction_start_times().items():
            if round_idx <= self.current_round:
                patches_initialized_by_round[round_idx] |= self._patches_initialized_by_instr[instr]

        if lightweight_output:
            return DeviceData(
                d=self.d_t,
                num_rounds=self.current_round,
                instructions=None,
                instruction_start_times=[self._completed_instructions[i]-self._instruction_durations[i]+1 for i in range(len(self.schedule_instructions))],
                all_patch_coords=list(self._all_patch_coords),
                syndrome_count_by_round=self._syndrome_count_by_round,
                instruction_count_by_round=self._instruction_count_by_round,
                generated_syndrome_data=None,
                patches_initialized_by_round={k: list(v) for k,v in patches_initialized_by_round.items()},
            )
        else:
            return DeviceData(
                d=self.d_t,
                num_rounds=self.current_round,
                instructions=copy.deepcopy(self.schedule_instructions),
                instruction_start_times=[self._completed_instructions[i]-self._instruction_durations[i]+1 for i in range(len(self.schedule_instructions))],
                all_patch_coords=list(self._all_patch_coords),
                syndrome_count_by_round=self._syndrome_count_by_round,
                instruction_count_by_round=self._instruction_count_by_round,
                generated_syndrome_data=self._postprocess_idle_data(self._generated_syndrome_data),
                patches_initialized_by_round={k: list(v) for k,v in patches_initialized_by_round.items()},
            )
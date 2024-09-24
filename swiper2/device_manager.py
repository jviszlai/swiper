from dataclasses import dataclass
import networkx as nx
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule, Duration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

@dataclass
class SyndromeRound:
    """A syndrome round for a given patch"""
    patch: tuple[int, int]
    round: int
    instruction_idx: int

@dataclass
class DeviceData:
    """Data containing the history of a device."""
    d: int
    num_rounds: int
    num_instructions: int
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
        self.current_round = 0

        self._syndrome_count_by_round = []
        self._instruction_count_by_round = []
        self._generated_syndrome_data = []

        self._completed_instructions = set()
        self._active_instructions = dict()
        self._instruction_frontier = set()

        for instruction_idx in next(nx.topological_generations(self.schedule_dag)):
            self._active_instructions[instruction_idx] = self._get_duration(instruction_idx)
            self._instruction_frontier.update(nx.descendants(self.schedule_dag, instruction_idx))

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

    def get_next_round(self, unfinished_decoding_instructions: set[int]) -> list[SyndromeRound]:
        """Return another round of syndrome measurements, starting new
        instructions if possible.

        Args:
            unfinished_decoding_instructions: Set of instruction indices that
                are not yet finished decoding (either pending or active).
        
        Returns:
            List of tuples, each containing a syndrome measurement coordinate,
            the current round index, and the index of the instruction that it is
            associated with.
        """
        generated_syndrome_rounds = []

        self._instruction_count_by_round.append(0)
        completed_instructions = set()
        for instruction_idx in self._active_instructions.keys():
            assert self._active_instructions[instruction_idx] > 0
            generated_syndrome_rounds.extend([SyndromeRound(coords, self.current_round, instruction_idx) for coords in self.schedule.all_instructions[instruction_idx].patches])
            self._active_instructions[instruction_idx] -= 1
            self._instruction_count_by_round[-1] += 1
            if self._active_instructions[instruction_idx] == 0:
                completed_instructions.add(instruction_idx)

        for instruction_idx in completed_instructions:
            self._completed_instructions.add(instruction_idx)
            self._active_instructions.pop(instruction_idx)
        
        new_active_instructions = set()
        for instruction_idx in self._instruction_frontier:
            if (all(inst_idx in self._completed_instructions for inst_idx in nx.ancestors(self.schedule_dag, instruction_idx))
                and not (self.schedule.all_instructions[instruction_idx].conditioned_on_idx in unfinished_decoding_instructions)):
                new_active_instructions.add(instruction_idx)

        for instruction_idx in new_active_instructions:
            duration = self._get_duration(instruction_idx)
            if duration > 0:
                self._active_instructions[instruction_idx] = self._get_duration(instruction_idx)
            self._instruction_frontier.remove(instruction_idx)
            self._instruction_frontier.update(nx.descendants(self.schedule_dag, instruction_idx))
    
        self.current_round += 1
        self._syndrome_count_by_round.append(len(generated_syndrome_rounds))

        self._generated_syndrome_data.append(generated_syndrome_rounds)

        return generated_syndrome_rounds
    
    def is_done(self) -> bool:
        """Return whether all instructions have been completed."""
        return len(self._active_instructions) == 0 and len(self._instruction_frontier) == 0
    
    def get_data(self):
        """Return all relevant dataregarding device history."""
        return DeviceData(
            d=self.d_t,
            num_rounds=self.current_round,
            num_instructions=len(self.schedule.all_instructions),
            syndrome_count_by_round=self._syndrome_count_by_round,
            instruction_count_by_round=self._instruction_count_by_round,
            generated_syndrome_data=self._generated_syndrome_data,
        )
    
def plot_data(data: DeviceData):
    pass
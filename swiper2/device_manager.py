import networkx as nx
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule

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

        self._completed_instructions = set()
        self._active_instructions = dict()
        self._instruction_frontier = set()

        for instruction_idx in next(nx.topological_generations(self.schedule_dag)):
            self._active_instructions[instruction_idx] = (self.d_t if self.schedule.all_instructions[instruction_idx].full_duration else self.d_t // 2 + 2)
            self._instruction_frontier.update(nx.descendants(self.schedule_dag, instruction_idx))

    def get_next_round(self, unfinished_decoding_instructions: set[int]) -> list[tuple[tuple[int, int], int]]:
        """Return another round of syndrome measurements, starting new
        instructions if possible.

        Args:
            unfinished_decoding_instructions: Set of instruction indices that
                are not yet finished decoding (either pending or active).
        
        Returns:
            List of tuples, each containing a syndrome measurement coordinate
            and the index of the instruction that it is associated with.
        """
        generated_syndrome_coords = []

        completed_instructions = set()
        for instruction_idx in self._active_instructions.keys():
            generated_syndrome_coords.extend([(coords, instruction_idx) for coords in self.schedule.all_instructions[instruction_idx].patches])
            self._active_instructions[instruction_idx] -= 1
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
            self._active_instructions[instruction_idx] = (self.d_t if self.schedule.all_instructions[instruction_idx].full_duration else self.d_t // 2 + 2)
            self._instruction_frontier.remove(instruction_idx)
            self._instruction_frontier.update(nx.descendants(self.schedule_dag, instruction_idx))
    
        return generated_syndrome_coords
from dataclasses import dataclass, field
import networkx as nx

@dataclass
class Instruction:
    name: str
    patches: list[tuple[int, int]]
    full_duration: bool
    conditioned_on_idx: int | None = None
    conditional_dependencies: list[int] = field(default_factory=list)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    def __init__(self):
        self.all_instructions = []

    def inject_T(self, patch_coords: tuple[int, int]):
        instruction = Instruction('INJECT_T', [patch_coords], True)
        self.all_instructions.append(instruction)

    def conditional_S(self, patch_coords: tuple[int, int], conditioned_on_idx: int):
        instruction = Instruction('CONDITIONAL_S', [patch_coords], False, conditioned_on_idx)
        self.all_instructions.append(instruction)
        
        conditioned_on_instruction = self.all_instructions[conditioned_on_idx]
        conditioned_on_instruction.conditional_dependencies.append(len(self.all_instructions) - 1)

    def add_op(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]],
            advance_time: bool = False,
        ):
        instruction = Instruction('MERGE', active_qubits + routing_qubits, True)
        self.all_instructions.append(instruction)

    def to_dag(self):
        dag = nx.DiGraph()
        for i,instruction in enumerate(self.all_instructions):
            dag.add_node(i)
            for j,instr in enumerate(self.all_instructions[:i]):
                if set(instruction.patches) & set(instr.patches):
                    dag.add_edge(j, i)
        
        return dag
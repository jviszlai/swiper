from dataclasses import dataclass, field
import networkx as nx
from enum import Enum

class Duration(Enum):
    HALF_D_ROUNDS = 1
    D_ROUNDS = 2

@dataclass
class Instruction:
    name: str
    patches: list[tuple[int, int]]
    duration: Duration | int
    conditioned_on_idx: int | None = None
    conditional_dependencies: list[int] = field(default_factory=list)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    def __init__(self):
        self.all_instructions: list[Instruction] = []

    def inject_T(self, patch_coords: tuple[int, int]):
        instruction = Instruction('INJECT_T', [patch_coords], Duration.D_ROUNDS)
        self.all_instructions.append(instruction)

    def conditional_S(self, patch_coords: tuple[int, int], conditioned_on_idx: int):
        instruction = Instruction('CONDITIONAL_S', [patch_coords], Duration.HALF_D_ROUNDS, conditioned_on_idx)
        self.all_instructions.append(instruction)
        
        conditioned_on_instruction = self.all_instructions[conditioned_on_idx]
        conditioned_on_instruction.conditional_dependencies.append(len(self.all_instructions) - 1)

    def merge(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]],
        ):
        instruction = Instruction('MERGE', active_qubits + routing_qubits, Duration.D_ROUNDS)
        self.all_instructions.append(instruction)

    def discard(self, patches: list[tuple[int, int]]):
        instruction = Instruction('DISCARD', patches, 0)
        self.all_instructions.append(instruction)

    def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D_ROUNDS):
        instruction = Instruction('IDLE', patches, num_rounds)
        self.all_instructions.append(instruction)

    def to_dag(self):
        dag = nx.DiGraph()
        for i,instruction in enumerate(self.all_instructions):
            dag.add_node(i)
            hidden_patches = set() # patches we will no longer draw connections to
            for j,instr in reversed(list(enumerate(self.all_instructions[:i]))):
                if (set(instruction.patches) & set(instr.patches)) - hidden_patches:
                    dag.add_edge(j, i)

                hidden_patches |= set(instr.patches)
        
        return dag
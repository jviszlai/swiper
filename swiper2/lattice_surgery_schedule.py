from dataclasses import dataclass, field
import networkx as nx
from enum import Enum

class Duration(Enum):
    HALF_D_ROUNDS = 1
    D_ROUNDS = 2

@dataclass(frozen=True)
class Instruction:
    name: str
    patches: frozenset[tuple[int, int]]
    duration: Duration | int
    conditioned_on_idx: int | None = None
    conditional_dependencies: frozenset[int] = field(default_factory=frozenset)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    def __init__(self):
        self.all_instructions: list[Instruction] = []

    def inject_T(self, patch_coords: tuple[int, int]):
        instruction = Instruction('INJECT_T', frozenset([patch_coords]), Duration.D_ROUNDS)
        self.all_instructions.append(instruction)

    def conditional_S(self, patch_coords: tuple[int, int], conditioned_on_idx: int):
        instruction = Instruction('CONDITIONAL_S', frozenset([patch_coords]), Duration.HALF_D_ROUNDS, conditioned_on_idx)
        self.all_instructions.append(instruction)
        
        update_instr = self.all_instructions[conditioned_on_idx]
        self.all_instructions[conditioned_on_idx] = Instruction(update_instr.name, update_instr.patches, update_instr.duration,
                                                                update_instr.conditioned_on_idx,
                                                                update_instr.conditional_dependencies | frozenset([len(self.all_instructions) - 1]))
                                                            

    def merge(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]],
        ):
        instruction = Instruction('MERGE', frozenset(active_qubits + routing_qubits), Duration.D_ROUNDS)
        self.all_instructions.append(instruction)

    def discard(self, patches: list[tuple[int, int]]):
        instruction = Instruction('DISCARD', frozenset(patches), 0)
        self.all_instructions.append(instruction)

    def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D_ROUNDS):
        if isinstance(num_rounds, Duration) or num_rounds > 0:
            instruction = Instruction('IDLE', frozenset(patches), num_rounds)
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
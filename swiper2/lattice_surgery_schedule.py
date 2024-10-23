from dataclasses import dataclass, field
import networkx as nx
from enum import Enum

class Duration(Enum):
    HALF_D_ROUNDS = 1
    D_ROUNDS = 2
    HALF_D_ROUNDS_ROUNDED_DOWN = 3
    HALF_D_ROUNDS_ROUNDED_UP = 4

@dataclass(frozen=True)
class Instruction:
    name: str
    patches: frozenset[tuple[int, int]]
    duration: Duration | int
    conditioned_on_idx: frozenset[int] = field(default_factory=frozenset)
    conditional_dependencies: frozenset[int] = field(default_factory=frozenset)
    conditioned_on_completion_idx: frozenset[int] = field(default_factory=frozenset)
    conditional_completion_dependencies: frozenset[int] = field(default_factory=frozenset)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    def __init__(self):
        self.all_instructions: list[Instruction] = []

    def inject_T(self, patch_coords: tuple[int, int]):
        instruction = Instruction('INJECT_T', frozenset([patch_coords]), Duration.D_ROUNDS)
        self.all_instructions.append(instruction)

    # def inject_T(self, patches: list[tuple[int, int]]):
    #     for patch in patches:
    #         instruction = Instruction('INJECT_T', frozenset([patch]), Duration.D_ROUNDS)
    #         self.all_instructions.append(instruction)

    def conditional_S(self, patch_coords: tuple[int, int], conditioned_on_idx: int):
        instruction = Instruction('CONDITIONAL_S', frozenset([patch_coords]), Duration.HALF_D_ROUNDS, frozenset([conditioned_on_idx]))
        self.all_instructions.append(instruction)
        
        update_instr = self.all_instructions[conditioned_on_idx]
        self.all_instructions[conditioned_on_idx] = Instruction(update_instr.name, update_instr.patches, update_instr.duration,
                                                                update_instr.conditioned_on_idx,
                                                                update_instr.conditional_dependencies | frozenset([len(self.all_instructions) - 1]))
                                                            
    def merge(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]],
            duration: Duration | int = Duration.D_ROUNDS,
        ):
        instruction = Instruction('MERGE', frozenset(active_qubits + routing_qubits), duration)
        self.all_instructions.append(instruction)
        self.discard(routing_qubits) 

    def discard(self, patches: list[tuple[int, int]], conditioned_on_idx: set[int] = set()):
        if len(patches) == 0:
            return
        instruction = Instruction('DISCARD', frozenset(patches), 0, conditioned_on_completion_idx=frozenset(conditioned_on_idx))
        self.all_instructions.append(instruction)
        for idx in conditioned_on_idx:
            update_instr = self.all_instructions[idx]
            self.all_instructions[idx] = Instruction(
                update_instr.name,
                update_instr.patches,
                update_instr.duration,
                update_instr.conditioned_on_idx,
                update_instr.conditional_dependencies,
                update_instr.conditioned_on_completion_idx,
                update_instr.conditional_completion_dependencies  | frozenset([len(self.all_instructions) - 1]),
            )

    def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D_ROUNDS):
        if isinstance(num_rounds, Duration) or num_rounds > 0:
            instruction = Instruction('IDLE', frozenset(patches), num_rounds)
            self.all_instructions.append(instruction)

    # def discard(self, patches: list[tuple[int, int]]):
    #     if len(patches) == 0:
    #         return
    #     for patch in patches:
    #         instruction = Instruction('DISCARD', frozenset([patch]), 0)
    #         self.all_instructions.append(instruction)

    # def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D_ROUNDS):
    #     if isinstance(num_rounds, Duration) or num_rounds > 0:
    #         for patch in patches:
    #             instruction = Instruction('IDLE', frozenset([patch]), num_rounds)
    #             self.all_instructions.append(instruction)

    def to_dag(self, d: int | None = None, dummy_final_node: bool = False):
        """Generate a DAG representations of the instruction indices, with
        'duration' attributes on the nodes. Each edge weight is set to the
        duration of the source instruction of the edge.
        
        Args:
            dummy_final_node: If True, add a dummy final node that all
            instructions point to. This is useful when calculating critical
            paths in the graph, because we do this using edge weights leading
            out of each node (so we need a terminal node to point to for the
            last instructions).
        """
        dag = nx.DiGraph()
        for i,instruction in enumerate(self.all_instructions):
            dag.add_node(i, duration=self.get_true_duration(instruction.duration, distance=d))
            hidden_patches = set() # patches we will no longer draw connections to
            for j,instr in reversed(list(enumerate(self.all_instructions[:i]))):
                if (set(instruction.patches) & set(instr.patches)) - hidden_patches:
                    dag.add_edge(j, i, weight=self.get_true_duration(instr.duration, distance=d))

                hidden_patches |= set(instr.patches)
        if dummy_final_node:
            for i,instruction in enumerate(self.all_instructions):
                dag.add_edge(i, len(self.all_instructions), weight=self.get_true_duration(instruction.duration, distance=d))
        return dag
    
    def total_duration(self, distance: int):
        dag = self.to_dag(d=distance, dummy_final_node=True)
        return nx.dag_longest_path_length(dag)
    
    def get_true_duration(self, duration: Duration | int, distance: int | None = None):
        if distance is None:
            return duration
        if isinstance(duration, Duration):
            if duration == Duration.HALF_D_ROUNDS:
                return distance // 2 + 2
            elif duration == Duration.D_ROUNDS:
                return distance
            elif duration == Duration.HALF_D_ROUNDS_ROUNDED_DOWN:
                return distance // 2
            elif duration == Duration.HALF_D_ROUNDS_ROUNDED_UP:
                return distance // 2 + 2
        return duration
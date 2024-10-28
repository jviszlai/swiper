from dataclasses import dataclass, field
import networkx as nx
import numpy as np
from enum import Enum

class Duration(Enum):
    D = 1
    HALF_D = 2
    HALF_D_PLUS_2 = 3

@dataclass(frozen=True)
class Instruction:
    name: str
    idx: int
    patches: frozenset[tuple[int, int]]
    duration: Duration | int
    conditioned_on_idx: frozenset[int] = field(default_factory=frozenset)
    conditional_dependencies: frozenset[int] = field(default_factory=frozenset)
    conditioned_on_completion_idx: frozenset[int] = field(default_factory=frozenset)
    conditional_completion_dependencies: frozenset[int] = field(default_factory=frozenset)
    merge_faces: frozenset[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=frozenset)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    def __init__(self):
        self.all_instructions: list[Instruction] = []

    # def inject_T(self, patch_coords: tuple[int, int]):
    #     instruction = Instruction('INJECT_T', frozenset([patch_coords]), Duration.D_ROUNDS)
    #     self.all_instructions.append(instruction)

    def inject_T(self, patches: list[tuple[int, int]]):
        for patch in patches:
            instruction = Instruction('INJECT_T', len(self.all_instructions), frozenset([patch]), Duration.D)
            self.all_instructions.append(instruction)

    def conditional_S(self, patch_coords: tuple[int, int], conditioned_on_idx: int):
        instruction = Instruction('CONDITIONAL_S', len(self.all_instructions), frozenset([patch_coords]), Duration.HALF_D_PLUS_2, frozenset([conditioned_on_idx]))
        self.all_instructions.append(instruction)
        
        update_instr = self.all_instructions[conditioned_on_idx]
        self.all_instructions[conditioned_on_idx] = Instruction(
            update_instr.name,
            update_instr.idx,
            update_instr.patches,
            update_instr.duration,
            update_instr.conditioned_on_idx,
            update_instr.conditional_dependencies | frozenset([len(self.all_instructions) - 1]),
            update_instr.conditioned_on_completion_idx,
            update_instr.conditional_completion_dependencies,
            update_instr.merge_faces,
        )
                                                            
    def merge(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]],
            merge_faces: set[tuple[tuple[int, int]]] | None = None,
            duration: Duration | int = Duration.D,
        ):
        if not merge_faces:
            merge_faces = set()
            if len(routing_qubits) > 0:
                for qubit in active_qubits:
                    found_match = False
                    for routing in routing_qubits:
                        if np.linalg.norm(np.array(qubit) - np.array(routing)) == 1:
                            if found_match:
                                raise ValueError('Multiple connections to routing space for one logical qubit; can\'t handle this case.')
                            found_match = True
                            merge_faces.add((qubit, routing))
                for i,routing_1 in enumerate(routing_qubits):
                    for routing_2 in routing_qubits[:i]:
                        if np.linalg.norm(np.array(routing_1) - np.array(routing_2)) == 1:
                            merge_faces.add((routing_1, routing_2))
            else:
                if len(active_qubits) == 2:
                    merge_faces.add(tuple(active_qubits))
                else:
                    raise ValueError('Can only merge two patches without routing patches.')
        for patch in active_qubits + routing_qubits:
            assert sum(patch in face for face in merge_faces) <= 4, (patch, merge_faces)
        instruction = Instruction(
            name='MERGE',
            idx=len(self.all_instructions),
            patches=frozenset(active_qubits + routing_qubits),
            duration=duration,
            merge_faces=frozenset(merge_faces),
        )
        self.all_instructions.append(instruction)
        self.discard(routing_qubits) 


    def discard(self, patches: list[tuple[int, int]], conditioned_on_idx: set[int] = set()):
        if len(patches) == 0:
            return
        for patch in patches:
            instruction = Instruction('DISCARD', len(self.all_instructions), frozenset([patch]), 0, conditioned_on_completion_idx=frozenset(conditioned_on_idx))
            self.all_instructions.append(instruction)
            for idx in conditioned_on_idx:
                update_instr = self.all_instructions[idx]
                self.all_instructions[idx] = Instruction(
                    update_instr.name,
                    update_instr.idx,
                    update_instr.patches,
                    update_instr.duration,
                    update_instr.conditioned_on_idx,
                    update_instr.conditional_dependencies,
                    update_instr.conditioned_on_completion_idx,
                    update_instr.conditional_completion_dependencies  | frozenset([len(self.all_instructions) - 1]),
                    update_instr.merge_faces,
                )

    # def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D_ROUNDS):
    #     if isinstance(num_rounds, Duration) or num_rounds > 0:
    #         instruction = Instruction('IDLE', frozenset(patches), num_rounds)
    #         self.all_instructions.append(instruction)

    # def discard(self, patches: list[tuple[int, int]]):
    #     if len(patches) == 0:
    #         return
    #     for patch in patches:
    #         instruction = Instruction('DISCARD', frozenset([patch]), 0)
    #         self.all_instructions.append(instruction)

    def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D):
        if isinstance(num_rounds, Duration) or num_rounds > 0:
            for patch in patches:
                instruction = Instruction('IDLE', len(self.all_instructions), frozenset([patch]), num_rounds)
                self.all_instructions.append(instruction)

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
            if duration == Duration.D:
                return distance
            elif duration == Duration.HALF_D:
                return distance // 2
            elif duration == Duration.HALF_D_PLUS_2:
                return distance // 2 + 2
        return duration
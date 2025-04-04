from dataclasses import dataclass, field
import networkx as nx
import numpy as np
from enum import Enum
import copy

class Duration(Enum):
    D = 1
    HALF_D = 2
    HALF_D_PLUS_2 = 3

    def __str__(self):
        if self == Duration.D:
            return 'D'
        elif self == Duration.HALF_D:
            return 'HALF_D'
        elif self == Duration.HALF_D_PLUS_2:
            return 'HALF_D_PLUS_2'
        
    @classmethod
    def from_str(cls, s):
        if s == 'D':
            return cls.D
        elif s == 'HALF_D':
            return cls.HALF_D
        elif s == 'HALF_D_PLUS_2':
            return cls.HALF_D_PLUS_2
        elif s.isdigit():
            return int(s)
        else:
            raise ValueError(f'Invalid duration string: {s}')

    @staticmethod
    def get_true_duration(duration: 'Duration | int', distance: int):
        if isinstance(duration, Duration):
            if duration == Duration.D:
                return distance
            elif duration == Duration.HALF_D:
                return distance // 2
            elif duration == Duration.HALF_D_PLUS_2:
                return distance // 2 + 2
        return duration

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
    group_instr_indices: frozenset[int] = field(default_factory=frozenset)
    group_name: str = ''

    def rename(self, new_name) -> 'Instruction':
        return Instruction(
            name=new_name,
            idx=self.idx,
            patches=self.patches,
            duration=self.duration,
            conditioned_on_idx=self.conditioned_on_idx,
            conditional_dependencies=self.conditional_dependencies,
            conditioned_on_completion_idx=self.conditioned_on_completion_idx,
            conditional_completion_dependencies=self.conditional_completion_dependencies,
            merge_faces=self.merge_faces,
            group_instr_indices=self.group_instr_indices,
        )
    
    def __str__(self):
        return f'{self.name} {self.idx} {str(list(self.patches)).replace(" ", "")} {self.duration} {str(list(self.conditioned_on_idx)).replace(" ", "")} {str(list(self.conditional_dependencies)).replace(" ", "")} {str(list(self.conditioned_on_completion_idx)).replace(" ", "")} {str(list(self.conditional_completion_dependencies)).replace(" ", "")} {str(list(self.merge_faces)).replace(" ", "")} {str(list(self.group_instr_indices)).replace(" ", "")} {self.group_name if self.group_name else "none"}'

    @classmethod
    def from_str(cls, s):
        s = s.split()
        return Instruction(
            name=s[0],
            idx=int(s[1]),
            patches=frozenset(eval(s[2])),
            duration=Duration.from_str(s[3]),
            conditioned_on_idx=frozenset(eval(s[4])),
            conditional_dependencies=frozenset(eval(s[5])),
            conditioned_on_completion_idx=frozenset(eval(s[6])),
            conditional_completion_dependencies=frozenset(eval(s[7])),
            merge_faces=frozenset(eval(s[8])),
            group_instr_indices=frozenset(eval(s[9])),
            group_name=s[10] if s[10] != 'none' else '',
        )
    
    def spacetime_volume(self, distance: int) -> int:
        return len(self.patches) * Duration.get_true_duration(self.duration, distance)

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    instructions: list[Instruction]
    _instructions_by_patch: dict[tuple[int, int], list[int]]

    def __init__(self, generate_dag_incrementally: bool = True):
        self.instructions: list[Instruction] = []
        self._instructions_by_patch = {}
        self.generate_dag_incrementally = generate_dag_incrementally
        self._generated_dag = nx.DiGraph()

    def __len__(self):
        return len(self.instructions)
    
    def __str__(self):
        return '\n'.join(str(instr) for instr in self.full_schedule().instructions)
    
    def __eq__(self, other):
        return self.full_schedule().instructions == other.full_schedule().instructions
    
    # copy
    def __copy__(self):
        instance = LatticeSurgerySchedule(generate_dag_incrementally=self.generate_dag_incrementally)
        instance.instructions = copy.deepcopy(self.instructions)
        instance._instructions_by_patch = copy.deepcopy(self._instructions_by_patch)
        instance._generated_dag = copy.deepcopy(self._generated_dag)
        return instance
    
    @classmethod
    def from_str(cls, s, generate_dag_incrementally: bool = False):
        instance = cls(generate_dag_incrementally=generate_dag_incrementally)
        for line in s.split('\n'):
            instance._add_instruction(Instruction.from_str(line))
        return instance

    def full_schedule(self) -> 'LatticeSurgerySchedule':
        """Return a list of all instructions, with all the necessary DISCARDS
        placed at the end.
        """
        full_sched = self.__copy__()
        full_sched.discard(full_sched._get_final_active_patches())
        return full_sched

    def inject_T(self, patches: list[tuple[int, int]]):
        for patch in patches:
            if patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[patch][-1]].name != 'DISCARD':
                raise ValueError(f'Tried to inject T gate on patch {patch} at instruction {len(self.instructions)}, but it was already active. If this was intended, make sure to DISCARD the patch first.')
            instruction = Instruction('INJECT_T', len(self.instructions), frozenset([patch]), Duration.D)
            self._add_instruction(instruction)

    def Y_meas(self, patch_coords: tuple[int, int], conditioned_on_idx: int=None):
        instruction = Instruction(
            name='Y_MEAS',
            idx=len(self.instructions),
            patches=frozenset([patch_coords]),
            duration=Duration.HALF_D_PLUS_2,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_name=('Y_MEAS' if not conditioned_on_idx else 'CONDITIONAL_S')
        )
        self._add_instruction(instruction)
        
        update_instr = self.instructions[conditioned_on_idx]
        self.instructions[conditioned_on_idx] = Instruction(
            update_instr.name,
            update_instr.idx,
            update_instr.patches,
            update_instr.duration,
            update_instr.conditioned_on_idx,
            update_instr.conditional_dependencies | frozenset([len(self.instructions) - 1]),
            update_instr.conditioned_on_completion_idx,
            update_instr.conditional_completion_dependencies,
            update_instr.merge_faces,
            update_instr.group_instr_indices,
            update_instr.group_name,
        )
    
    def S(self, target_patch: tuple[int, int], ancilla_patch: tuple[int, int], conditioned_on_idx: int | None = None):
        if ancilla_patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[ancilla_patch][-1]].name != 'DISCARD':
            raise ValueError(f'Tried to initialize ancilla patch {ancilla_patch} for S instruction {len(self.instructions)}, but it is already active. If this was intended, make sure to DISCARD the patch first.')
        group_instr_indices = frozenset([len(self.instructions), len(self.instructions) + 1, len(self.instructions) + 2, len(self.instructions) + 3])
        merge_instr = Instruction(
            name='MERGE',
            idx=len(self.instructions),
            patches=frozenset([target_patch, ancilla_patch]),
            duration=Duration.D,
            merge_faces=frozenset([(target_patch, ancilla_patch)]),
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
        )
        y_cube_instr = Instruction(
            name='Y_MEAS',
            idx=len(self.instructions) + 1,
            patches=frozenset([ancilla_patch]),
            duration=Duration.HALF_D_PLUS_2,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
        )
        idle = Instruction(
            name='IDLE',
            idx=len(self.instructions) + 2,
            patches=frozenset([target_patch]),
            duration=Duration.HALF_D_PLUS_2,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
        )
        discard = Instruction(
            name='DISCARD',
            idx=len(self.instructions) + 3,
            patches=frozenset([ancilla_patch]),
            duration=0,
            conditioned_on_idx=frozenset([conditioned_on_idx]) if conditioned_on_idx else frozenset(),
            group_instr_indices=group_instr_indices,
            group_name='CONDITIONAL_S',
        )
        if conditioned_on_idx:
            update_instr = self.instructions[conditioned_on_idx]
            self.instructions[conditioned_on_idx] = Instruction(
                update_instr.name,
                update_instr.idx,
                update_instr.patches,
                update_instr.duration,
                update_instr.conditioned_on_idx,
                update_instr.conditional_dependencies | group_instr_indices,
                update_instr.conditioned_on_completion_idx,
                update_instr.conditional_completion_dependencies,
                update_instr.merge_faces,
                update_instr.group_instr_indices,
                update_instr.group_name,
            )
        self._add_instruction(merge_instr)
        self._add_instruction(y_cube_instr)
        self._add_instruction(idle)
        self._add_instruction(discard)
                                                            
    def merge(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]] = [],
            merge_faces: set[tuple[tuple[int, int]]] | None = None,
            duration: Duration | int = Duration.D,
        ) -> int:
        """Lattice surgery merge-and-split operation involving two or more
        logical qubits.
        
        Args:
            active_qubits: List of logical qubits (patch coords) to merge.
            routing_qubits: List of routing patches that connect the active
                qubits. Must not already be active patches.
            merge_faces: If provided, a set of tuples of tuples, where each
                tuple contains two patch coordinates that share a merged
                boundary. If not provided, the merge faces are inferred from
                the active and routing qubits.
            duration: Duration of the merge operation. After the merge, the
                routing patches are discarded.
        
        Returns:
            The index of the merge instruction in the schedule.
        """
        if len(routing_qubits) == 0 and len(active_qubits) != 2:
            raise ValueError(f'No routing patches provided for merge instruction {len(self.instructions)}, but more than two active patches {active_qubits}.')
        if len(routing_qubits) == 0 and np.linalg.norm(np.array(active_qubits[0]) - np.array(active_qubits[1])) != 1:
            raise ValueError(f'No routing patches provided for merge instruction {len(self.instructions)}, but active patches {active_qubits} are not adjacent.')
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
        for patch in routing_qubits:
            if patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[patch][-1]].name != 'DISCARD':
                raise ValueError(f'Tried to initialize routing patch {patch} for merge instruction {len(self.instructions)}, but it is already active. If this was intended, make sure to DISCARD the patch first.')
        instruction = Instruction(
            name='MERGE',
            idx=len(self.instructions),
            patches=frozenset(active_qubits + routing_qubits),
            duration=duration,
            merge_faces=frozenset(merge_faces),
        )
        self._add_instruction(instruction)
        idx = len(self.instructions) - 1
        self.discard(routing_qubits) 
        return idx

    def discard(self, patches: list[tuple[int, int]], conditioned_on_idx: set[int] = set()):
        if len(patches) == 0:
            return
        for patch in patches:
            instruction = Instruction('DISCARD', len(self.instructions), frozenset([patch]), 0, conditioned_on_completion_idx=frozenset(conditioned_on_idx))
            if patch in self._instructions_by_patch and self.instructions[self._instructions_by_patch[patch][-1]].name == 'DISCARD':
                raise ValueError(f'Tried to discard the same patch {patch} twice at instruction {len(self.instructions)}.')
            self._add_instruction(instruction)
            for idx in conditioned_on_idx:
                update_instr = self.instructions[idx]
                self.instructions[idx] = Instruction(
                    update_instr.name,
                    update_instr.idx,
                    update_instr.patches,
                    update_instr.duration,
                    update_instr.conditioned_on_idx,
                    update_instr.conditional_dependencies,
                    update_instr.conditioned_on_completion_idx,
                    update_instr.conditional_completion_dependencies  | frozenset([len(self.instructions) - 1]),
                    update_instr.merge_faces,
                    update_instr.group_instr_indices,
                    update_instr.group_name,
                )

    def idle(self, patches: list[tuple[int, int]], num_rounds: Duration | int = Duration.D):
        if isinstance(num_rounds, Duration) or num_rounds > 0:
            for patch in patches:
                instruction = Instruction('IDLE', len(self.instructions), frozenset([patch]), num_rounds)
                self._add_instruction(instruction)
        elif num_rounds < 0:
            raise ValueError('Number of rounds must be nonnegative.')

    def to_dag(self, d: int | None = None) -> nx.DiGraph:
        """Generate a DAG representations of the instruction indices, with
        'duration' attributes on the nodes. Each edge weight is set to the
        duration of the source instruction of the edge.

        Args:
            d: Temporal code distance. If given, instruction durations will be
                converted from abstract Duration values to actual integer
                durations.

        Returns:
            A directed acyclic graph representing the schedule. Each node
            corresponds to an instruction index and has a 'duration' attribute.
            Each edge corresponds to a dependency between instructions and has
            a 'weight' attribute corresponding to the duration of the source
            instruction.
        """
        if self.generate_dag_incrementally:
            dag = self._generated_dag.copy()
            nx.set_edge_attributes(
                G=dag, 
                values={e: self.get_true_duration(self.instructions[e[0]].duration, distance=d) for e in dag.edges()},
                name='weight',
            )
            nx.set_node_attributes(
                G=dag,
                values={idx: self.get_true_duration(self.instructions[idx].duration, distance=d) for idx in dag.nodes()},
                name='duration',
            )
            return dag
        else:
            full_schedule = self.full_schedule()
            dag = nx.DiGraph()
            for i,instruction in enumerate(full_schedule.instructions):
                dag.add_node(i, duration=self.get_true_duration(instruction.duration, distance=d))
                hidden_patches = set() # patches we will no longer draw connections to
                for j,instr in reversed(list(enumerate(full_schedule.instructions[:i]))):
                    if (set(instruction.patches) & set(instr.patches)) - hidden_patches:
                        dag.add_edge(j, i, weight=self.get_true_duration(instr.duration, distance=d))

                    hidden_patches |= set(instr.patches)
            return dag
    
    def total_duration(self, distance: int):
        """Calculate the duration of the longest path in the schedule DAG."""
        dag = self.to_dag(d=distance)
        return nx.dag_longest_path_length(dag)
    
    def get_true_duration(self, duration: Duration | int, distance: int | None = None):
        """Convert abstract Duration values to actual integer durations.

        Args:
            duration: Abstract Duration value or integer duration.
            distance: Temporal code distance. If given, abstract Duration values
                will be converted to actual integer durations.
        
        Returns:
            Converted duration, or input duration if distance is not given.
        """
        if distance is None:
            return duration
        else:
            return Duration.get_true_duration(duration, distance)
    
    def count_instructions(self, name: str):
        return sum(instr.name == name for instr in self.instructions)
    
    def total_space_footprint(self):
        return len(set(patch for instr in self.instructions for patch in instr.patches))

    def total_instruction_volume(self, distance: int, name: str | None = None):
        return sum(instr.spacetime_volume(distance) for instr in self.instructions if name is None or instr.name == name)

    def _add_instruction(self, instruction: Instruction):
        idx = len(self.instructions)
        self.instructions.append(instruction)

        if self.generate_dag_incrementally:
            self._generated_dag.add_node(idx, duration=instruction.duration)
            for patch in instruction.patches:
                if patch in self._instructions_by_patch and (self._instructions_by_patch[patch][-1], idx) not in self._generated_dag.edges():
                    prev_instruction = self._instructions_by_patch[patch][-1]
                    assert prev_instruction < idx
                    self._generated_dag.add_edge(prev_instruction, idx, weight=self.get_true_duration(self.instructions[prev_instruction].duration))

        for patch in instruction.patches:
            self._instructions_by_patch.setdefault(patch, []).append(idx)

    def _get_final_active_patches(self):
        patches = []
        for patch, instr_idxs in self._instructions_by_patch.items():
            instr = self.instructions[instr_idxs[-1]]
            if instr.name != 'DISCARD':
                patches.append(patch)
        return patches
from dataclasses import dataclass

@dataclass
class Patch:
    coords: tuple[int, int]

@dataclass
class Instruction:
    name: str
    patches: list[tuple[int, int]]
    duration: str
    conditioned_on_idx: int | None = None
    conditional_dependencies: list[int] = []

class LatticeSurgerySchedule:
    """Represents a planned series of lattice surgery operations."""
    def __init__(self):
        self.layers = [[]]
        self.active_patches_by_layer = [[]]
        self.blocked_patches_by_layer = [set()]
        self._current_active_patches = set()
        self._all_instructions = []

    def inject_T(self, patch_coords: tuple[int, int]):
        assert patch_coords not in self._current_active_patches
        self._current_active_patches.add(patch_coords)
        instruction = Instruction('INJECT_T', [patch_coords], 'd')
        self.layers[-1].append(instruction)
        self._all_instructions.append(instruction)
        self.active_patches_by_layer[-1] = list(self._current_active_patches)

    def conditional_S(self, patch_coords: tuple[int, int], conditioned_on_idx: int):
        self._current_active_patches.add(patch_coords)
        instruction = Instruction('CONDITIONAL_S', [patch_coords], 'd', conditioned_on_idx)
        self.layers[-1].append(instruction)
        self._all_instructions.append(instruction)
        
        conditioned_on_instruction = self._all_instructions[conditioned_on_idx]
        conditioned_on_instruction.conditional_dependencies.append(len(self._all_instructions) - 1)

        self.active_patches_by_layer[-1] = list(self._current_active_patches)

    def advance_time(self):
        self.active_patches_by_layer.append([])
        self.layers.append([])
        self.blocked_patches_by_layer.append(set())

    def deactivate_qubit(self, patch_coords: Patch):
        self._current_active_patches.remove(patch_coords)

    def add_op(
            self,
            active_qubits: list[tuple[int, int]],
            routing_qubits: list[tuple[int, int]],
            blocked_qubits_after_op: list[tuple[int, int]] = [],
            advance_time: bool = False,
        ):
        self._current_active_patches.update(active_qubits)
        instruction = Instruction('MERGE', active_qubits + routing_qubits, 'd')
        self.layers[-1].append(instruction)
        self._all_instructions.append(instruction)
        self.blocked_patches_by_layer[-1].update(blocked_qubits_after_op)
        self.active_patches_by_layer[-1] = list(self._current_active_patches)

        if advance_time:
            self.advance_time()
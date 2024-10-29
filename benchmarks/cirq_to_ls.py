# from lsqecc.logical_lattice_ops import logical_lattice_ops
# from lsqecc.patches.patches import PatchType
# from lsqecc.pipeline.lattice_surgery_compilation_pipeline import compile_str
# from lsqecc.simulation.logical_patch_state_simulation import SimulatorType
# from lsqecc.gates import gates
import json
import parse
import re
import subprocess
import os
import shutil
import cirq

from swiper.lattice_surgery_schedule import LatticeSurgerySchedule

class Cell:

    def __init__(self, slice_idx, row, col, cell_info):
        self.slice_idx = slice_idx
        self.row = row
        self.col = col
        self.activity = cell_info['activity']['activity_type']
        self.patch_type = cell_info['patch_type']
        if self.patch_type == 'Qubit':
            self.qubit_id = parse.parse('Id: {}', cell_info['text'])[0]
            is_m = lambda edge: 'Stiched' in edge # rip typo
        elif self.patch_type == 'Ancilla':
            is_m = lambda edge: 'Join' in edge
        else:
            is_m = lambda _: False
        self.bottom_m = is_m(cell_info['edges']['Bottom'])
        self.left_m = is_m(cell_info['edges']['Left'])
        self.right_m = is_m(cell_info['edges']['Right'])
        self.top_m = is_m(cell_info['edges']['Top'])
    
    def __repr__(self) -> str:
        return json.dumps(self, 
                          default=lambda o: o.__dict__, 
                          indent=4)

class LLInstruction:

    def __init__(self, slice_idx, label, args):
        self.slice_idx = slice_idx
        self.label = label
        self.args = args
    
    def __repr__(self) -> str:
        return json.dumps(self, 
                          default=lambda o: o.__dict__, 
                          indent=4)

S_GATE_MAPPING = {
    'H' : cirq.H,
    'I' : cirq.I,
    'S' : cirq.S,
    'X' : cirq.X,
    'T' : cirq.T,
    'Z' : cirq.Z,
}

def _get_gridsynth_sequence(op: cirq.Operation, rads: float, precision: float = 1e-10):
    assert len(op.qubits) == 1
    # pi / x = rz_angle => x = pi / rz_angle
    angle_str = f'{rads}' if rads >= 0 else f'({rads})'
    command = ["../benchmarks/gridsynth", angle_str, f'--epsilon={precision}']
    output = subprocess.check_output(command)
    
    ss = str(output)[2:-3].strip('W')[::-1]
    
    # Merge double S gate
    new_s = ''
    for i in range(1, len(ss)):
        if ss[i] == ss[i-1] == 'S':
            new_s += 'Z'
        else:
            new_s += ss[i]
    
    # Build a circuit from this
    approx_seq = []
    for s in new_s:
        approx_seq.append(S_GATE_MAPPING[s].on(op.qubits[0]))
    return approx_seq

def _get_merge(cell_program, endpoint: Cell):
    if endpoint.patch_type != 'Qubit':
        raise Exception('Endpoint must be data qubit cell')
    merge_faces = set()
    def step(coords: tuple[int,int], from_dir: str):
        cell = cell_program[coords[0]][coords[1]][slice]
        if cell.bottom_m and not from_dir == 'top':
            merge_faces.add(((coords[0], coords[1]), (coords[0]+1, coords[1])))
            return (cell.row+1, cell.col), 'bottom'
        elif cell.top_m and not from_dir == 'bottom':
            merge_faces.add(((coords[0], coords[1]), (coords[0]-1, coords[1])))
            return (cell.row-1, cell.col), 'top'
        elif cell.right_m and not from_dir == 'left':
            merge_faces.add(((coords[0], coords[1]), (coords[0], coords[1]+1)))
            return (cell.row, cell.col+1), 'right'
        elif cell.left_m and not from_dir == 'right':
            merge_faces.add(((coords[0], coords[1]), (coords[0], coords[1]-1)))
            return (cell.row, cell.col-1), 'left'
        return None
    slice = endpoint.slice_idx
    curr = (endpoint.row, endpoint.col)
    data_cells = [curr]
    routing_cells = []
    from_dir = None
    while ret := step(curr, from_dir):
        curr, from_dir = ret
        if cell_program[curr[0]][curr[1]][slice].patch_type == 'Ancilla':
            routing_cells.append(curr)
        elif cell_program[curr[0]][curr[1]][slice].patch_type == 'Qubit':
            data_cells.append(curr)
        else:
            raise Exception('Unexpected merged cell type found')
    is_inject = (False, None, None)
    if slice < 2:
        return data_cells, routing_cells, merge_faces, is_inject
    for data in data_cells:
        old_t_data = cell_program[data[0]][data[1]][slice - 2]
        old_s_data = cell_program[data[0]][data[1]][slice - 1]
        curr_activity = cell_program[data[0]][data[1]][slice].activity
        if curr_activity != 'Measurement':
            continue
        if old_s_data and old_s_data.patch_type == 'Qubit' and old_s_data.activity == 'Unitary':
            # There does not seem to be a better way to check for S gate injection. 
            # Corresponding slices between lli and json seems very unreliable 
            # (e.g. Y state injection is two slices in lli but T state injection is 1 slice in lli
            #  both are 2 slices in json)
            # TODO: See if this creates problematic edge cases
            is_inject = (True, 'Y', data) 
            break
        elif old_t_data and old_t_data.patch_type == 'DistillationQubit':
            is_inject = (True, 'T', data)
            break
        # data_id = cell_program[data[0]][data[1]][slice].qubit_id
        # for instr in lli_program[slice]: # Why is T not injected in the previous timeslice, but Y is
        #     if instr.label == 'RequestMagicState':
        #         for arg in instr.args:
        #             if data_id == arg.split(' ')[0]: #{qubit_id} {_}
        #                 is_inject = (True, 'T', data)
        #                 break
        # for instr in lli_program[slice - 1]:
        #     if instr.label == 'RequestYState':
        #         for arg in instr.args:
        #             if data_id == arg.split(' ')[0]: #{qubit_id} {_}
        #                 is_inject = (True, 'Y', data)
        #                 break
            
    return data_cells, routing_cells, merge_faces, is_inject



def cirq_to_ls(circ: cirq.Circuit) -> LatticeSurgerySchedule:
    qbit_mapping = {q: f'q_{i}' for i, q in enumerate(circ.all_qubits())}
    bad_ops = []
    for i, moment in enumerate(circ.moments):
        for op in moment:
            no_control_op = op.without_classical_controls()
            try:
                test_qasm = no_control_op._qasm_(cirq.QasmArgs(qubit_id_map=qbit_mapping))
                if 'ry' in test_qasm:
                    raise Exception('Ry gate') # TODO: Properly convert Ry gates
                if isinstance(op.gate, cirq.ZPowGate) and op.gate._exponent != 1:
                    # Undecomposed gate TODO
                    if op.gate == cirq.T or op.gate == cirq.S:
                        continue
                    raise Exception('Undecomposed Gate')
                if isinstance(op.gate, cirq.XPowGate) and op.gate._exponent != 1:
                    # Undecomposed gate TODO
                    raise Exception('Undecomposed Gate')
                if isinstance(op.gate, cirq.YPowGate) and op.gate._exponent != 1:
                    # Undecomposed gate TODO
                    raise Exception('Undecomposed Gate')
            except Exception:
                bad_ops.append((i, op))
    circ.batch_remove(bad_ops)
    def decomp(op: cirq.Operation) -> cirq.OP_TREE:
        return cirq.decompose(op, keep=lambda op: len(op.qubits) <= 2)
    def map_approx_rz(op: cirq.Operation) -> cirq.OP_TREE:
        if isinstance(op.gate, cirq.Rz):
            return _get_gridsynth_sequence(op, op.gate._rads)
        return op
    def make_qasm_compat(op: cirq.Operation) -> cirq.OP_TREE:
        return op.without_classical_controls()

    circ = circ.map_operations(decomp).map_operations(map_approx_rz).map_operations(make_qasm_compat)

    os.makedirs('../benchmarks/tmp')
    circ.save_qasm('../benchmarks/tmp/prog.qasm')
    #subprocess.call(['../benchmarks/lsqecc_slicer', '-q', '-i', '../benchmarks/tmp/prog.qasm', '-L', 'edpc', '--disttime', '1', '--nostagger', '-P', 'wave', '--printlli', 'sliced', '-o', '../benchmarks/tmp/lli.txt'])
    subprocess.call(['../benchmarks/lsqecc_slicer', '-q', '-i', '../benchmarks/tmp/prog.qasm', '-L', 'edpc', '--disttime', '1', '--nostagger', '-P', 'wave', '-o', '../benchmarks/tmp/compiled.json'])

    prog_data = json.load(open('../benchmarks/tmp/compiled.json', 'rb'))
    #prog_instrs = open('../benchmarks/tmp/lli.txt', 'r').readlines()

    # lli_program = []
    # for slice_idx, slice_data in enumerate(prog_instrs):
    #     lli_program.append([])
    #     instrs = slice_data.split(';')
    #     for instr in instrs:
    #         try:
    #             label, arg_list = parse.parse('{} {}', instr)
    #             args = re.split(',(?!\d+\))', arg_list) # look-ahead to ignore commas in location tuples
    #             lli_program[slice_idx].append(LLInstruction(slice_idx, label, args))
    #         except:
    #             continue

    cell_major_program = [[[None for _ in range(len(prog_data))]            
                            for _ in range(len(prog_data[0]))]
                            for _ in range(len(prog_data[0][0]))]
    slice_major_program = [[[None for _ in range(len(prog_data[0][0]))]            
                            for _ in range(len(prog_data[0]))]
                            for _ in range(len(prog_data))]

    for i, slice in enumerate(prog_data):
        for r, row in enumerate(slice):
            for c, cell in enumerate(row):
                if cell:
                    cell_major_program[r][c][i] = Cell(i, r, c, cell)
                    slice_major_program[i][r][c] = cell_major_program[r][c][i]
    
    # Ok, data processing done. Convert all instructions 
    schedule = LatticeSurgerySchedule()
    t_inject_history = {}
    for slice_idx, slice in enumerate(slice_major_program):
        processed_cells = []
        for r, row in enumerate(slice):
            for c, cell in enumerate(row):
                if cell and cell.patch_type == 'Qubit' and (r, c) not in processed_cells:
                    if cell.bottom_m or cell.left_m or cell.right_m or cell.top_m:
                        data, routing, merge_faces, (is_inject, inject_type, inject_cell) = _get_merge(cell_major_program, cell)
                        if is_inject:
                            if len(data) != 2:
                                raise Exception(f'Expected two cells for injection operation, got {len(data)}.')
                            other_cell = [cell for cell in data if cell != inject_cell][0]
                            if inject_type == 'T':
                                schedule.inject_T([inject_cell])
                                merge_idx = schedule.merge(data, routing, merge_faces, return_merge_idx=True)
                                if other_cell not in t_inject_history:
                                    t_inject_history[other_cell] = [(False, None) for _ in range(len(slice_major_program))]
                                t_inject_history[other_cell][slice_idx] = (True, merge_idx)
                                schedule.discard([inject_cell])
                            elif inject_type == 'Y':
                                if other_cell in t_inject_history:
                                    # TODO: Assume conditional for now....
                                    i = 1
                                    while slice_idx - i > 0 and not t_inject_history[other_cell][slice_idx - i][0]:
                                        i += 1
                                    conditional, instr_idx = t_inject_history[other_cell][slice_idx - i]
                                    if conditional:
                                        schedule.conditional_S(other_cell, instr_idx)
                                    else:
                                        pass # TODO: Non-conditional S gates
                        else:
                            schedule.merge(data, routing, merge_faces)
                            for data_coords in data:
                                data_cell = cell_major_program[data_coords[0]][data_coords[1]][slice_idx]
                                if data_cell.activity == 'Measurement':
                                    schedule.discard([data_coords])
                        processed_cells.extend(data)
                        processed_cells.extend(routing)
                    elif cell.activity == 'Measurement':
                        schedule.discard([cell])
                        processed_cells.append(cell)
                    elif cell.activity == 'Unitary':
                        pass # TODO: Need to handle Y state prep

    # Discard remaining data qubits
    for r, row in enumerate(cell_major_program):
        for c, cell_history in enumerate(row):
            if cell_history[-1] and cell_history[-1].patch_type == 'Qubit':
                schedule.discard([(r,c)])

    shutil.rmtree('../benchmarks/tmp', ignore_errors=True)

    return schedule
                        







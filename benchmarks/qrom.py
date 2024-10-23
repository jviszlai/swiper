from qualtran.bloqs.data_loading.qrom import QROM
import cirq
import numpy as np

from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule

class QROM_Schedule:

    def __init__(self, data: list[int], select_bitsize: int, target_bitsize: int):
        qrom_bloq = QROM(data, selection_bitsizes=(select_bitsize,), target_bitsizes=(target_bitsize,))
        circuit = qrom_bloq.as_composite_bloq().flatten().to_cirq_circuit()
        self.schedule = self._build_ls_schedule(circuit)

    def _build_ls_schedule(self, circuit: cirq.Circuit) -> LatticeSurgerySchedule:
        qbit_idx = {qbit: i for i, qbit in enumerate(circuit.all_qubits())}
        schedule = LatticeSurgerySchedule()
        for i, op in enumerate(circuit.all_operations()):
            if isinstance(op.gate, cirq.ZPowGate):
                # T Gates

                ancilla_qbit = (0, qbit_idx[op.qubits[0]])
                data_qbit = (1, qbit_idx[op.qubits[0]])
                schedule.inject_T(ancilla_qbit)
                schedule.merge([ancilla_qbit, data_qbit], [])
                schedule.conditional_S(data_qbit, len(schedule.all_instructions) - 1)
                schedule.discard([ancilla_qbit])
            elif isinstance(op.gate, cirq.CXPowGate):
                # CNOT Gates

                min_qbit = min(qbit_idx[qbit] for qbit in op.qubits)
                max_qbit = max(qbit_idx[qbit] for qbit in op.qubits)
                data_qbits = [(1, qbit_idx[qbit]) for qbit in op.qubits]
                schedule.merge(data_qbits, [(2, qbit_loc) for qbit_loc in range(min_qbit, max_qbit + 1)])
            else:
                # TODO:
                continue
        for data_loc in qbit_idx.values():
            schedule.discard([(1, data_loc)])
        return schedule

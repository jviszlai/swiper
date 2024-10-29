import math
import numpy as np
import cirq
import os
from abc import ABC, abstractmethod
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran import Register, Signature, QAny, DecomposeTypeError, DecomposeNotImplementedError
from qualtran.bloqs.data_loading.qrom import QROM

from pyLIQTR.BlockEncodings.CarlemanLinearization   import Carleman_Linearization
from pyLIQTR.ProblemInstances.NonlinearODE          import FOperators
from pyLIQTR.utils.circuit_decomposition  import keep, generator_decompose
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding   import *
from pyLIQTR.utils.resource_analysis          import estimate_resources
from pyLIQTR.utils.circuit_decomposition      import circuit_decompose_multi
from pyLIQTR.qubitization.qubitized_gates     import QubitizedWalkOperator
from pyLIQTR.utils.printing                 import openqasm

from pyLIQTR.pest_interface.pest_python import pw_to_dpw_cutoff

from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule
from benchmarks.cirq_to_ls import cirq_to_ls

def _decompose_circuit(circuit) -> cirq.Circuit:
    out = cirq.Circuit()
    for op in generator_decompose(circuit, keep=keep, on_stuck_raise=None):
        out.append(op)
    return out

class Benchmark(ABC):

    @abstractmethod
    def get_schedule(self) -> LatticeSurgerySchedule:
        raise NotImplementedError

class QROM_15(Benchmark):

    def __init__(self, data: list[int] | None = None, select_bitsize: int = 15, target_bitsize: int = 15) -> None:
        if data is None:
            data = [2 ** np.arange(select_bitsize)]
        qrom_bloq = QROM(data, selection_bitsizes=(select_bitsize,), target_bitsizes=(target_bitsize,))
        def is_atomic(binst):
            try: 
                binst.bloq.decompose_bloq()
                return True
            except (DecomposeTypeError, DecomposeNotImplementedError):
                return False
            
        circuit = cirq.Circuit(qrom_bloq.as_composite_bloq().flatten(pred=is_atomic).to_cirq_circuit())
        self.schedule = cirq_to_ls(circuit)

    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule

class CarlemanEncoding(Benchmark):

    def __init__(self) -> None:
        # Source: pyLIQTR/Examples/Applications/NonLinearODE/Carleman_encoding_details.ipynb
        n = 2
        K = 4

        a0_in = n
        a1 = n
        a2_in = math.ceil(n/2)
        a2_out = math.ceil(n)

        alpha0 = 1
        alpha1 = 1
        alpha2 = 1

        ancilla_register = Register("ancilla", QAny(bitsize = 7 + max(a0_in, a1, a2_in, a2_out) + math.ceil(np.log2(K))))
        data_register = Register("data", QAny(bitsize = n*K+1))

        signature = Signature([ancilla_register, data_register])
        registers = get_named_qubits(signature)

        carlemanLinearization = Carleman_Linearization(FOperators(n, K, (a0_in, a1, a2_in, a2_out), (alpha0, alpha1, alpha2)), K)
        circuit = cirq.Circuit(carlemanLinearization.on_registers(**registers))

        self.schedule = cirq_to_ls(_decompose_circuit(circuit))

    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule

class ElectronicStructure(Benchmark):

    def __init__(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Source: pyLIQTR/Examples/ApplicationInstances/PeriodicChemistry/resource_estimates-electronic_structure-mg_slab.ipynb
        hamhdf5     =  f"{dir_path}/data/magnesium-4x4x2-Ecut_0.5_scale_1.0.ham.hdf5"
        gridhdf5    =  f"{dir_path}/data/magnesium-4x4x2-Ecut_0.5_scale_1.0.grid.hdf5"

        mg_slab = getInstance('ElectronicStructure', filenameH=hamhdf5, filenameG=gridhdf5)
        energy_error = 1e-3 # Hartree units
        mg_slab_LinearTEncoding = getEncoding(VALID_ENCODINGS.LinearT,instance = mg_slab, energy_error=energy_error, control_val=1)
        registers = get_named_qubits(mg_slab_LinearTEncoding.signature)
        walk_op = QubitizedWalkOperator(mg_slab_LinearTEncoding,multi_control_val=0)
        walk_circuit = cirq.Circuit(walk_op.on_registers(**registers))

        self.schedule = cirq_to_ls(_decompose_circuit(walk_circuit))
    
    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
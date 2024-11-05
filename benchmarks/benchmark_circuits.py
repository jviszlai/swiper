import math
import numpy as np
import cirq
import qiskit
from qiskit.qasm2 import dumps
from qiskit.transpiler.passes import RemoveBarriers, Decompose
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import os
from abc import ABC, abstractmethod
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran import Register, Signature, QAny, DecomposeTypeError, DecomposeNotImplementedError
from qualtran.bloqs.data_loading.qrom import QROM as QROM_bloq
from    pyLIQTR.clam.lattice_definitions                      import   CubicLattice, SquareLattice, TriangularLattice
from    pyLIQTR.BlockEncodings.getEncoding                    import   getEncoding, VALID_ENCODINGS
from pyLIQTR.BlockEncodings.CarlemanLinearization   import Carleman_Linearization
from pyLIQTR.ProblemInstances.NonlinearODE          import FOperators
from pyLIQTR.utils.circuit_decomposition  import keep, generator_decompose
from pyLIQTR.utils.get_hdf5 import get_hdf5
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding   import *
from pyLIQTR.utils.resource_analysis          import estimate_resources
from pyLIQTR.utils.circuit_decomposition      import circuit_decompose_multi
from pyLIQTR.qubitization.qubitized_gates     import QubitizedWalkOperator
from pyLIQTR.utils.printing                 import openqasm

from pyLIQTR.pest_interface.pest_python import pw_to_dpw_cutoff

from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.schedule_experiments import RegularTSchedule, MSD15To1Schedule, MemorySchedule
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
    
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

class QASMBenchmark(Benchmark):

    def __init__(self, qasm_file: str, eps=1e-10) -> None:
        with open(qasm_file) as f_in:
            qasm_str = f_in.read()
        qiskit_circ = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
        cirq_circ = circuit_from_qasm(dumps(qiskit.transpile(RemoveBarriers()(qiskit_circ), basis_gates=['cx', 'u3'])))
        # cirq.to_json(cirq_circ, f'benchmarks/data/mqt/{file[:-5]}.json')
        self.schedule = cirq_to_ls(_decompose_circuit(cirq_circ), eps=eps)
        self._name = qasm_file.split("/")[-1].split(".")[0]

    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
    
    def name(self) -> str:
        return self._name

class QROM(Benchmark):

    def __init__(self, data: list[int] | None = None, select_bitsize: int = 15, target_bitsize: int = 15) -> None:
        self.select_bitsize = select_bitsize
        self.target_bitsize = target_bitsize
        print(select_bitsize, target_bitsize)
        if data is None:
            data = [2 ** np.arange(select_bitsize)]
        qrom_bloq = QROM_bloq(data, selection_bitsizes=(select_bitsize,), target_bitsizes=(target_bitsize,))
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
    
    def name(self) -> str:
        return f"qrom_{self.select_bitsize}_{self.target_bitsize}"

class CarlemanEncoding(Benchmark):

    def __init__(self, n: int = 2, K: int = 4) -> None:
        # Source: pyLIQTR/Examples/Applications/NonLinearODE/Carleman_encoding_details.ipynb
        self.n = n
        self.K = K

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
    
    def name(self) -> str:
        return f"carleman_encoding_{self.n}_{self.K}"

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
    
    def name(self) -> str:
        return "electronic_structure"
    
class FermiHubbardEncoding(Benchmark):

    def __init__(self, shape: tuple[int, int] = (2,2)) -> None:
        # recommended shapes: (2,2), (3,3), (4,4)?
        self.shape = shape
        model  =  getInstance('FermiHubbard',shape=shape, J=-1.0, U=4.0,cell=SquareLattice)
        block_encoding = getEncoding(VALID_ENCODINGS.PauliLCU)(model)

        registers = get_named_qubits(block_encoding.signature)
        circuit = cirq.Circuit(block_encoding.on_registers(**registers))
        self.schedule = cirq_to_ls(_decompose_circuit(circuit))

    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
    
    def name(self) -> str:
        return f"fermi_hubbard_{self.shape[0]}_{self.shape[1]}"
    
class HeisenbergEncoding(Benchmark):

    def __init__(self, N: int = 3) -> None:
        self.N = N
        J_x  =  J_y  =  -0.5;              J_z = -1.0
        h_x  =  1.0;      h_y = 0.0;       h_z = 0.5
        model  =  getInstance( "Heisenberg", 
                            shape=(N,N), 
                            J=(J_x,J_y,J_z), 
                            h=(h_x,h_y,h_z), 
                            cell=SquareLattice)
        block_encoding    =  getEncoding(VALID_ENCODINGS.PauliLCU)(model)

        registers = get_named_qubits(block_encoding.signature)
        circuit = cirq.Circuit(block_encoding.on_registers(**registers))
        self.schedule = cirq_to_ls(_decompose_circuit(circuit))

    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
    
    def name(self) -> str:
        return f"heisenberg_{self.N}"
    
class ChemicalHamiltonianEncoding(Benchmark):
    
    def __init__(self, index):
        assert index in [112, 140, 146]
        mol = get_hdf5(index, filename='benchmarks/data/chemistry_instances.csv')
        self.molecule = mol.name.split('_')[0]
        mol_instance    =   getInstance("ChemicalHamiltonian",mol_ham=mol.get_molecular_hamiltonian())
        walk_operator  =  QubitizedWalkOperator(getEncoding(instance=mol_instance, \
                                                    encoding=VALID_ENCODINGS.PauliLCU,instantiate=False))
        registers = get_named_qubits(walk_operator.signature)
        circuit = cirq.Circuit(walk_operator.on_registers(**registers))
        self.schedule = cirq_to_ls(_decompose_circuit(circuit))
    
    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
    
    def name(self) -> str:
        return self.molecule

class RegularT(Benchmark):

    def __init__(self, num_ts: int = 100, idle_between_ts: int = 0) -> None:
        self.num_ts = num_ts
        self.idle_between_ts = idle_between_ts
        self.schedule = RegularTSchedule(num_ts, idle_between_ts).schedule

    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
    
    def name(self) -> str:
        return f"regular_t_{self.num_ts}_{self.idle_between_ts}"
    
class Memory(Benchmark):
    
        def __init__(self, rounds: int = 10000) -> None:
            self.rounds = rounds
            self.schedule = MemorySchedule(rounds).schedule
    
        def get_schedule(self) -> LatticeSurgerySchedule:
            return self.schedule
        
        def name(self) -> str:
            return f"memory_{self.rounds}"
        
class MSD15To1(Benchmark):
    
    def __init__(self) -> None:
        self.schedule = MSD15To1Schedule().schedule
    
    def get_schedule(self) -> LatticeSurgerySchedule:
        return self.schedule
    
    def name(self) -> str:
        return "msd_15to1"
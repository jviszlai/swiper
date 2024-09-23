from typing import Callable
import networkx as nx
from swiper2.lattice_surgery import LatticeSurgerySchedule

class DecodingSimulator:
    def __init__(
            self,
            distance: int,
            decoding_latency_fn: Callable[[int], float],
            speculation_latency: float,
            
        ):
        """Initialize the decoding simulator.
        
        Args:
            distance: The distance of the surface code (which also specifies the
                number of QEC rounds for each lattice surgery operation).
            decoding_latency_fn: A function that returns a (possibly
                randomly-sampled) decoding latency given the spacetime volume of
                the decoding problem (e.g. dxdx2d => volume 2). Returned latency
                is in units of rounds of QEC.
            speculation_latency: The latency of a speculative prediction, in
                units of rounds of QEC.
        """
        self.distance = distance
        self.decoding_latency_fn = decoding_latency_fn
        self.speculation_latency = speculation_latency

    def run(
            self,
            schedule: LatticeSurgerySchedule,
            scheduling_method: str,
            max_parallel_processes: int | None = None,
        ):
        """TODO
        
        Args:
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
            scheduling_method: Window scheduling method. 'sliding', 'parallel', 
                or 'dynamic'.
            max_parallel_processes: Maximum number of parallel decoding
                processes to run. If None, run as many as possible.
        """
        decoding_graph = nx.DiGraph()
        # Node type: 3D spacetime coords. Third coordinate specifies round at
        # which commit region begins.
        # Node attributes:
        #   num_commit_rounds: int
        #   pre_buffer_size: int
        #   post_buffer_size: int
        #   ready_to_decode: bool
        # Edge direction represents flow of virtual defect data.

        finished_nodes = set()
        active_nodes = set() # nodes being actively decoded. Each is a tuple of (node, round_started_decoding)
        patch_latest_node = dict() # map from each patch to its most recent node in decoding graph. If patch is not in this dict, it has not yet been initialized.

        num_instructions = len(schedule.layers)
        pending_instructions = []
        instruction_window_dependencies: dict[int, list[tuple[int, int, int]]] = dict()
        current_instruction_layer_idx = 0
        current_round = 0
        while current_instruction_idx < num_instructions:
            if current_round % self.distance == 0:
                # old windows are finished

                # apply a new layer of instructions
                instructions = schedule.layers[current_instruction_layer_idx]
                for instruction in instructions:
                    # create new nodes in decoding graph according to scheduling
                    # method.

                    # conditional: cannot be performed until entire history
                    # before the completion of the conditioned-on instruction
                    # has been decoded.

                    raise NotImplementedError

                current_instruction_idx += 1


            current_round += 1
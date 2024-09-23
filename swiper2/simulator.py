from typing import Callable
import networkx as nx
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper2.device_manager import DeviceManager

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
        device_manager = DeviceManager(self.distance, schedule)
        window_manager = WindowManager(...)
        decoding_manager = DecodingManager(...)

        unassigned_syndrome_data = []
        unfinished_decoding_instructions: set[int] = set()
        all_windows = []
        while not done:
            new_syndrome_data = device_manager.get_next_round(unfinished_decoding_instructions)
            unassigned_syndrome_data += new_syndrome_data
            all_windows, unassigned_syndrome_data = window_manager.update_windows(unassigned_syndrome_data)
            all_windows = decoding_manager.update_decoding(all_windows)
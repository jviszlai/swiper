from typing import Callable
import networkx as nx
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper2.device_manager import DeviceData, DeviceManager

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
        ) -> tuple[DeviceData, WindowData, DecodingData]:
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
        window_manager = WindowManager(..., scheduling_method=scheduling_method)
        decoding_manager = DecodingManager(..., max_parallel_processes=max_parallel_processes)

        unfinished_instructions: set[int] = set()
        while not (device_manager.is_done() and window_manager.is_done() and decoding_manager.is_done()):
            new_syndrome_data = device_manager.get_next_round(unfinished_instructions)
            new_windows = window_manager.get_new_windows(new_syndrome_data)
            decoding_manager.update_decoding(new_windows)

            unfinished_window_instructions = window_manager.get_unfinished_instructions()
            unfinished_decoding_instructions = decoding_manager.get_unfinished_instructions()
            unfinished_instructions = unfinished_window_instructions | unfinished_decoding_instructions

        device_data = device_manager.get_data()
        window_data = window_manager.get_data()
        decoding_data = decoding_manager.get_data()
        return device_data, window_data, decoding_data
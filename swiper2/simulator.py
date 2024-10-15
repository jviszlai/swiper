from typing import Callable
import networkx as nx
import tqdm
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper2.device_manager import DeviceData, DeviceManager
from swiper2.decoder_manager import DecoderData, DecoderManager
from swiper2.window_manager import WindowData, SlidingWindowManager, ParallelWindowManager, DynamicWindowManager
from swiper2.window_builder import WindowBuilder

class DecodingSimulator:
    def __init__(
            self,
            distance: int,
            decoding_latency_fn: Callable[[int], int],
            speculation_latency: int,
            speculation_accuracy: float,
            speculation_mode: str,
        ):
        """Initialize the decoding simulator.
        
        Args:
            distance: The distance of the surface code (which also specifies the
                number of QEC rounds for each lattice surgery operation).
            decoding_latency_fn: A function that returns a (possibly
                randomly-sampled) decoding latency given the spacetime volume of
                the decoding problem, in units of rounds*d^2 (e.g. dxdx2d =>
                volume 2d). Returned latency is in units of rounds of QEC.
            speculation_latency: The latency of a speculative prediction, in
                units of rounds of QEC.
            speculation_accuracy: The probability that a speculative prediction
                is correct.
            speculation_mode: 'integrated' or 'separate'. If 'integrated', the
                speculation time is included in the decoding time of a window,
                and speculation can only be performed once the decoder starts
                processing the window. If 'separate', the speculation time is
                not included in the decoding, time of a window, and speculation
                can be run independently of decoding. In this case, speculation
                uses a parallel process and counts towards
                max_parallel_processes.
        """
        self.distance = distance
        self.decoding_latency_fn = decoding_latency_fn
        self.speculation_latency = speculation_latency
        self.speculation_accuracy = speculation_accuracy
        assert speculation_mode in ['integrated', 'separate']
        self.speculation_mode = speculation_mode

        self._device_manager: DeviceManager | None = None
        self._window_manager: SlidingWindowManager | ParallelWindowManager | DynamicWindowManager | None = None
        self._decoding_manager: DecoderManager | None = None

    def run(
            self,
            schedule: LatticeSurgerySchedule,
            scheduling_method: str,
            enforce_window_alignment: bool,
            max_parallel_processes: int | None = None,
            progress_bar: bool = False,
        ) -> tuple[DeviceData, WindowData, DecoderData]:
        """TODO
        
        Args:
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
            scheduling_method: Window scheduling method. 'sliding', 'parallel', 
                or 'dynamic'.
            max_parallel_processes: Maximum number of parallel decoding
                processes to run. If None, run as many as possible.
            progress_bar: If True, display a progress bar for the simulation.
        """
        self.initialize_experiment(
            schedule=schedule,
            scheduling_method=scheduling_method,
            enforce_window_alignment=enforce_window_alignment,
            max_parallel_processes=max_parallel_processes,
        )
        self._window_manager.source_indices = set()

        if progress_bar:
            pbar_r = tqdm.tqdm(desc='Surface code rounds')
            # pbar_i = tqdm.tqdm(total=len(schedule.all_instructions), desc='Scheduled instructions complete')

        while not self.is_done():
            self.step_experiment()
            if progress_bar and self._decoding_manager._current_round % 100 == 0:
                pbar_r.update(100)
                # pbar_i.update(len(fully_decoded_instructions) - pbar_i.n)
                # pbar_i.refresh()
            
        if progress_bar:
            pbar_r.update(self._decoding_manager._current_round - pbar_r.n)
            # pbar_i.update(pbar_i.total - pbar_i.n)
            pbar_r.close()
            # pbar_i.close()

        return self.get_data()
    
    def initialize_experiment(
            self,
            schedule: LatticeSurgerySchedule,
            scheduling_method: str,
            enforce_window_alignment: bool,
            max_parallel_processes: int | None = None,
        ) -> None:
        self._device_manager = DeviceManager(self.distance, schedule)
        if scheduling_method == 'sliding':
            self._window_manager = SlidingWindowManager(WindowBuilder(self.distance, enforce_alignment=enforce_window_alignment))
        elif scheduling_method == 'parallel':
            self._window_manager = ParallelWindowManager(WindowBuilder(self.distance, enforce_alignment=enforce_window_alignment))
        elif scheduling_method == 'dynamic':
            raise NotImplementedError
            # self._window_manager = DynamicWindowManager(WindowBuilder(self.distance, enforce_alignment=enforce_window_alignment))
        else:
            raise ValueError(f"Unknown scheduling method: {scheduling_method}")
        self._decoding_manager = DecoderManager(
            decoding_time_function=self.decoding_latency_fn,
            speculation_time=self.speculation_latency,
            speculation_accuracy=self.speculation_accuracy,
            max_parallel_processes=max_parallel_processes,
            speculation_mode=self.speculation_mode,
        )

    def step_experiment(self) -> None:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")

        if self.is_done():
            raise ValueError("Experiment is already done. Run run() to start a new experiment.")

        # step device forward
        self._decoding_manager.step(self._window_manager.all_windows, self._window_manager.window_dag)
        fully_decoded_instructions = self._decoding_manager.get_finished_instruction_indices(self._window_manager.all_windows) - self._window_manager.pending_instruction_indices()

        syndrome_rounds, discarded_patches = self._device_manager.get_next_round(fully_decoded_instructions)

        # process new round
        self._window_manager.process_round(syndrome_rounds, discarded_patches)
        self._decoding_manager.update_decoding(self._window_manager.all_windows, self._window_manager.window_dag)

    def is_done(self) -> bool:
        return self._device_manager.is_done() and len(self._window_manager.all_windows) - len(self._decoding_manager._window_completion_times) == 0

    def get_data(self) -> tuple[DeviceData, WindowData, DecoderData]:
        device_data = self._device_manager.get_data()
        window_data = self._window_manager.get_data()
        decoding_data = self._decoding_manager.get_data()
        return device_data, window_data, decoding_data
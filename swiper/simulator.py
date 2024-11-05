from typing import Callable
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.device_manager import DeviceData, DeviceManager
from swiper.decoder_manager import DecoderData, DecoderManager
from swiper.window_manager import WindowData, SlidingWindowManager, ParallelWindowManager, TAlignedWindowManager
from swiper.window_builder import WindowBuilder
import swiper.plot as plotter

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
            speculation_mode: 'integrated', 'separate', or None. If 'integrated', the
                speculation time is included in the decoding time of a window,
                and speculation can only be performed once the decoder starts
                processing the window. If 'separate', the speculation time is
                not included in the decoding, time of a window, and speculation
                can be run independently of decoding. In this case, speculation
                uses a parallel process and counts towards
                max_parallel_processes. If None, no speculation is performed.
        """
        self.distance = distance
        self.decoding_latency_fn = decoding_latency_fn
        self.speculation_latency = speculation_latency
        self.speculation_accuracy = speculation_accuracy
        assert speculation_mode in ['integrated', 'separate', None]
        self.speculation_mode = speculation_mode

        self._device_manager: DeviceManager | None = None
        self._window_manager: SlidingWindowManager | ParallelWindowManager | TAlignedWindowManager | None = None
        self._decoding_manager: DecoderManager | None = None

        self.sent_windows = []

    def run(
            self,
            schedule: LatticeSurgerySchedule,
            scheduling_method: str,
            max_parallel_processes: int | None = None,
            progress_bar: bool = False,
            pending_window_count_cutoff: int = 0,
            save_animation_frames: bool = False,
            lightweight_output: bool = False,
            rng: int | np.random.Generator = np.random.default_rng(),
        ) -> tuple[bool, DeviceData, WindowData, DecoderData]:
        """TODO
        
        Args:
            schedule: LatticeSurgerySchedule encoding operations to be
                performed.
            scheduling_method: Window scheduling method. 'sliding', 'parallel', 
                or 'aligned'.
            max_parallel_processes: Maximum number of parallel decoding
                processes to run. If None, run as many as possible.
            progress_bar: If True, display a progress bar for the simulation.
            pending_window_count_cutoff: If the number of pending windows
                exceeds this value, the simulation is considered to have failed
                and will return early.
            save_animation_frames: If using in Jupyter notebook, use %%capture
                TODO: broken
            lightweight_output: If True, avoid returning certain large data
                structures (e.g. window_dag, window_completion_times) in
                outputs. Useful for large-scale simulations.
            rng: Random number generator.
        """
        self.initialize_experiment(
            schedule=schedule,
            scheduling_method=scheduling_method,
            max_parallel_processes=max_parallel_processes,
            lightweight_output=lightweight_output,
            rng=rng,
        )
        assert self._device_manager is not None
        assert self._window_manager is not None
        assert self._decoding_manager is not None

        if progress_bar:
            pbar_r = tqdm.tqdm(desc='Surface code rounds')
            # pbar_i = tqdm.tqdm(total=len(schedule.all_instructions), desc='Scheduled instructions complete')

        if save_animation_frames:
            fig = plt.figure()
            self.frame_data = []

        while not self.is_done():
            self.step_experiment(pending_window_count_cutoff=pending_window_count_cutoff)
            if progress_bar and self._decoding_manager._current_round % 100 == 0:
                pbar_r.update(100)
                # pbar_i.update(len(fully_decoded_instructions) - pbar_i.n)
                # pbar_i.refresh()
            if save_animation_frames:
                ax = plotter.plot_device_schedule_trace(self._device_manager.get_data(), spacing=1, default_fig=fig)
                ax.set_zticks([])
                self.frame_data.append(ax)
            
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
            max_parallel_processes: int | None = None,
            lightweight_output: bool = False,
            rng: int | np.random.Generator = np.random.default_rng(),
        ) -> None:
        self.failed = False
        self._device_manager = DeviceManager(self.distance, schedule, lightweight_output=lightweight_output, rng=rng)
        if scheduling_method == 'sliding':
            self._window_manager = SlidingWindowManager(WindowBuilder(self.distance), lightweight_output=lightweight_output)
        elif scheduling_method == 'parallel':
            self._window_manager = ParallelWindowManager(WindowBuilder(self.distance), lightweight_output=lightweight_output)
        elif scheduling_method == 'aligned':
            self._window_manager = TAlignedWindowManager(WindowBuilder(self.distance), lightweight_output=lightweight_output)
        else:
            raise ValueError(f"Unknown scheduling method: {scheduling_method}")
        self._decoding_manager = DecoderManager(
            decoding_time_function=self.decoding_latency_fn,
            speculation_time=self.speculation_latency,
            speculation_accuracy=self.speculation_accuracy,
            max_parallel_processes=max_parallel_processes,
            speculation_mode=self.speculation_mode,
            lightweight_output=lightweight_output,
            rng=rng,
        )

    def step_experiment(self, pending_window_count_cutoff: int = 0) -> None:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")

        if self.is_done():
            raise ValueError("Experiment is already done. Run run() to start a new experiment.")

        pending_window_count = len(self._window_manager.all_windows) - len(self._decoding_manager._window_decoding_completion_times)
        if pending_window_count_cutoff > 0 and pending_window_count > pending_window_count_cutoff:
            self.failed = True
            return

        # step device forward
        self._decoding_manager.step()
        incomplete_instructions = set(self._device_manager._active_instructions.keys()) | self._window_manager.window_builder.get_incomplete_instructions() | self._window_manager.pending_instruction_indices() | self._decoding_manager.get_incomplete_instruction_indices()

        syndrome_rounds = self._device_manager.get_next_round(incomplete_instructions)

        # process new round
        newly_constructed_windows = self._window_manager.process_round(syndrome_rounds)
        self.sent_windows.extend(w.window_idx for w in newly_constructed_windows)
        self._decoding_manager.update_decoding(newly_constructed_windows, self._window_manager.window_dag)

    def is_done(self) -> bool:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")
        return self.failed or (self._device_manager.is_done() and len(self._window_manager.all_constructed_windows) - len(self._decoding_manager._window_decoding_completion_times) == 0)

    def get_data(self) -> tuple[bool, DeviceData, WindowData, DecoderData]:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")
        device_data = self._device_manager.get_data()
        window_data = self._window_manager.get_data()
        decoding_data = self._decoding_manager.get_data()
        return not self.failed, device_data, window_data, decoding_data
    
    def get_frame_data(self) -> list[plt.Axes]:
        return self.frame_data
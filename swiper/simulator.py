import sys
from typing import Callable
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
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
            print_interval: dt.timedelta | None = None,
            pending_window_count_cutoff: int = 0,
            device_rounds_cutoff: int = 0,
            clock_timeout: dt.timedelta | None = None,
            save_animation_frames: bool = False,
            lightweight_setting: int = 0,
            ultralight_output: bool = False,
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
            device_rounds_cutoff: If the number of device rounds exceeds this
                value, the simulation is considered to have failed and will
                return early.
            clock_timeout: If given, stop simulation after this much time has
                elapsed.
            save_animation_frames: If using in Jupyter notebook, use %%capture
                TODO: broken
            lightweight_setting: Optimization level for memory usage. Affects
                runtime memory usage and output data size. Some output data will
                not be available at higher settings.
                0: No optimization.
                1: Avoid data structures that scale with the total number of
                    device rounds, but keep some data structures that scale with
                    the number of windows.
                2: Avoid any data structures that scales with simulation
                    duration.
            rng: Random number generator.
        """
        start_time = dt.datetime.now()

        if print_interval is not None:
            print(f'{start_time.strftime("%Y-%m-%d %H:%M:%S")} | Starting simulation')
            sys.stdout.flush()

        self.initialize_experiment(
            schedule=schedule,
            scheduling_method=scheduling_method,
            max_parallel_processes=max_parallel_processes,
            lightweight_setting=lightweight_setting,
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
            self.step_experiment(pending_window_count_cutoff=pending_window_count_cutoff, device_rounds_cutoff=device_rounds_cutoff, print_interval=print_interval)
            if progress_bar and self._decoding_manager._current_round % 100 == 0:
                pbar_r.update(100)
                # pbar_i.update(len(fully_decoded_instructions) - pbar_i.n)
                # pbar_i.refresh()
            if save_animation_frames:
                ax = plotter.plot_device_schedule_trace(self._device_manager.get_data(), spacing=1, default_fig=fig)
                ax.set_zticks([])
                self.frame_data.append(ax)
            if clock_timeout is not None and dt.datetime.now() > start_time + clock_timeout:
                self.failed = True
        
        if print_interval is not None:
            print(f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Finished simulation')
            sys.stdout.flush()
            
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
            lightweight_setting: int = 0,
            rng: int | np.random.Generator = np.random.default_rng(),
        ) -> None:
        self.failed = False
        self._device_manager = DeviceManager(self.distance, schedule, lightweight_setting=lightweight_setting, rng=rng)
        if scheduling_method == 'sliding':
            self._window_manager = SlidingWindowManager(WindowBuilder(self.distance, lightweight_setting=lightweight_setting), lightweight_setting=lightweight_setting)
        elif scheduling_method == 'parallel':
            self._window_manager = ParallelWindowManager(WindowBuilder(self.distance, lightweight_setting=lightweight_setting), lightweight_setting=lightweight_setting)
        elif scheduling_method == 'aligned':
            self._window_manager = TAlignedWindowManager(WindowBuilder(self.distance, lightweight_setting=lightweight_setting), lightweight_setting=lightweight_setting)
        else:
            raise ValueError(f"Unknown scheduling method: {scheduling_method}")
        self._decoding_manager = DecoderManager(
            decoding_time_function=self.decoding_latency_fn,
            speculation_time=self.speculation_latency,
            speculation_accuracy=self.speculation_accuracy,
            max_parallel_processes=max_parallel_processes,
            speculation_mode=self.speculation_mode,
            lightweight_setting=lightweight_setting,
            rng=rng,
        )

        self.start_time = dt.datetime.now()
        self.last_print_time = dt.datetime.now() - dt.timedelta(days=1)

    def step_experiment(self, pending_window_count_cutoff: int = 0, device_rounds_cutoff: int = 0, print_interval: dt.timedelta | None = None) -> None:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")

        if self.is_done():
            raise ValueError("Experiment is already done. Run run() to start a new experiment.")

        pending_window_count = len(self._window_manager.all_windows) - self._decoding_manager._num_completed_windows
        if pending_window_count_cutoff > 0 and pending_window_count > pending_window_count_cutoff:
            self.failed = True
            return
        if device_rounds_cutoff > 0 and self._device_manager.current_round > device_rounds_cutoff:
            self.failed = True
            return

        # step device forward
        completed_window_indices = self._decoding_manager.step()
        purged_indices = self._window_manager.purge_windows(completed_window_indices)
        incomplete_instructions = set(self._device_manager._active_instructions.keys()) | self._window_manager.window_builder.get_incomplete_instructions() | self._window_manager.pending_instruction_indices() | self._decoding_manager.get_incomplete_instruction_indices()

        syndrome_rounds = self._device_manager.get_next_round(incomplete_instructions)
        
        cur_time = dt.datetime.now()
        if print_interval is not None and cur_time - self.last_print_time >= print_interval:
            num_complete_instructions = self._device_manager._completed_instruction_count
            print(f'{cur_time.strftime("%Y-%m-%d %H:%M:%S")} | Simulation update: decoder round {self._decoding_manager._current_round}, completed instructions: {num_complete_instructions}/{len(self._device_manager.schedule)}, actively running or decoding instructions: {len(incomplete_instructions)}, waiting windows: {pending_window_count}/{len(self._window_manager.all_windows)}. Max active instruction index: {max(incomplete_instructions)}')
            sys.stdout.flush()
            self.last_print_time = cur_time
            
        # process new round
        newly_constructed_windows = self._window_manager.process_round(syndrome_rounds)
        self.sent_windows.extend(w.window_idx for w in newly_constructed_windows)
        self._decoding_manager.update_decoding(newly_constructed_windows, purged_indices, self._window_manager.window_dag)

    def is_done(self) -> bool:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")
        return self.failed or (self._device_manager.is_done() and len(self._window_manager.all_constructed_windows) - self._decoding_manager._num_completed_windows == 0)

    def get_data(self) -> tuple[bool, DeviceData, WindowData, DecoderData]:
        if self._device_manager is None or self._window_manager is None or self._decoding_manager is None:
            raise ValueError("Experiment not initialized properly. Run initialize_experiment() first.")
        device_data = self._device_manager.get_data()
        window_data = self._window_manager.get_data()
        decoding_data = self._decoding_manager.get_data()
        return not self.failed, device_data, window_data, decoding_data
    
    def get_frame_data(self) -> list[plt.Axes]:
        return self.frame_data
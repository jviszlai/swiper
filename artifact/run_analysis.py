"""This script generates all data-related plots that appear in the paper. The
data directory names provided in main() can be configured to point to
newly-generated data if desired.

The code is largely a combination of various parts of the notebooks found in
`notebooks/`, although the heavier data generation has been separated out into
the `artifact/run_*` scripts. The plots are generated using matplotlib and saved
to the `artifact/figures/` directory."""

import os, sys
sys.path.append('.')
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize
from typing import Any
import scipy
import pandas as pd
import stim
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from swiper.lattice_surgery_schedule import LatticeSurgerySchedule
from swiper.schedule_experiments import RegularTSchedule, MSD15To1Schedule
from swiper.simulator import DecodingSimulator
import swiper.plot as plotter

def main():
    ############################################################################
    # CONFIGURATION
    ############################################################################

    # For fig 3. Can be regenerated using `artifact/run_reaction_time_evals.py`
    decoder_dist_filename = 'artifact/data/decoder_dists.json'

    # For fig 4. Can be regenerated using `artifact/run_predictor_accuracy.py`
    one_step_data_filename = 'artifact/data/processed_01-step-predictor-results.pkl'
    two_step_data_filename = 'artifact/data/processed_02-step-predictor-results.pkl'
    three_step_data_filename = 'artifact/data/processed_03-step-predictor-results.pkl'

    # For fig 7.
    fpga_data_filename = 'artifact/data/fpga_data.json'

    # For fig 8b. Can be regenerated using `artifact/run_reaction_time_evals.py`
    mispredict_data_filename = 'artifact/data/mispredict_data.json'

    # For figs 10, 14, 15. Can be regenerated using `artifact/run_reaction_time_evals.py`
    benchmark_directories = ['artifact/data/benchmarks1', 'artifact/data/benchmarks2']

    # For fig 10 (runtime of SWIPER-SIM), we only wanted runtime data from a single
    # cluster (the two benchmark datasets were run on different clusters)
    benchmark_runtime_directories = ['artifact/data/benchmarks2']

    # For fig 12. Can be regenerated using `artifact/run_reaction_time_evals.py`
    reaction_times_directory = 'artifact/data/reaction_times'

    ############################################################################

    # Generate all plots
    print('\nRUNNING ANALYSIS\n')
    plot_3(decoder_dist_filename)
    plot_4(one_step_data_filename, two_step_data_filename, three_step_data_filename)
    plot_7(fpga_data_filename)
    plot_8b(mispredict_data_filename)
    plot_9()
    plot_10_14abc_15ab(benchmark_directories, benchmark_runtime_directories)
    plot_12ab(reaction_times_directory, decoder_dist_filename)
    plot_11ab_12c()

def plot_3(decoder_dist_filename):
    """See `notebooks/03_decoder_distribution.ipynb` for original plotting
    code."""
    print(f'Loading decoder distribution data from {decoder_dist_filename}...')
    with open(decoder_dist_filename, 'r') as f:
        decoder_dists = json.load(f)

    print(f'Generating figure 3\n')
    fig,ax = plt.subplots(figsize=(5,3))

    distances = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    volumes = [2,3,4,5,6,7]
    for i,dist in reversed(list(enumerate(distances))):
        for volume_str,data in decoder_dists[str(dist)].items():
            volume = int(volume_str)
            parts = ax.violinplot(np.array(data)/dist, positions=[i], vert=True, showextrema=False, points=1000)
            for pc in parts['bodies']:
                pc.set_alpha((volumes.index(volume)+1) / len(volumes))
                pc.set_facecolor(f'C2')

    # discrete color scale using C0, showing that each volume is a different alpha
    for i,volume in reversed(list(enumerate(volumes))):
        ax.hist([-1], color='C2', label=f'{volume}'+r'd$^3$', alpha=(i+1) / len(volumes))

    plt.axhline(1, color='black', linestyle='--', linewidth=1)
    # ax.set_yscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Factor longer than\nlogical cycle time')
    ax.set_xlabel('Code distance')
    # inward ticks
    ax.tick_params(direction='in', which='both', right=True, top=True)
    # ax.set_xticks([1,5,10,15], ['1x', '5x', '10x', '15x'])
    ax.set_yticks([1,10,50], ['1x', '10x', '50x'])
    ax.set_ylim(0.5, 70)
    ax.set_xlim(-0.5, len(distances)-0.5)
    ax.set_xticks(range(len(distances)), distances)

    # reverse entried in legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(reversed(handles), reversed(labels))
    ax.legend(title='Volume', frameon=False, ncol=1, bbox_to_anchor=(1, 1.05), loc='upper left')

    plt.savefig('artifact/figures/3.png', bbox_inches='tight')
    plt.close()

def plot_4(one_step_data_filename, two_step_data_filename, three_step_data_filename):
    """See `notebooks/04_predictor_accuracy.ipynb` for original plotting code."""
    ## Predictor Accuracy
    import pickle as pkl
    d_range = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    p_range = [1e-3]

    ### Plotting
    print(f'Loading predictor data from {one_step_data_filename}, {two_step_data_filename}, {three_step_data_filename}...')
    one_step_data = pkl.load(open(one_step_data_filename, 'rb'))
    two_step_data = pkl.load(open(two_step_data_filename, 'rb'))    
    three_step_data = pkl.load(open(three_step_data_filename, 'rb'))

    print(f'Generating figure 4\n')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.patches as mpatches
    plt.rcParams["font.family"] = "serif"
    color_list = ['#0072B2', '#CC79A7', '#009E73', '#E69F00', '#56B4E9', '#D55E00', '#F0E442']
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_list)

    bs = 0.4 # bar spacing
    bw = 0.3 # bar width
    plt.figure(figsize=(6, 3.7))
    titles = {
        0: 'Naive',
        1: 'Binned',
        2: 'Binned+'
    }
    for i, results in enumerate([one_step_data, two_step_data, three_step_data]):
        for p in p_range:
            speculation_accuracy = {d: results[(p,d)][1].count(True)/len(results[(p,d)][1]) for d in d_range}
            plt.plot([(d-bs)+bs*i for d in d_range], [speculation_accuracy[d] for d in d_range], marker='o', color=color_list[0], markeredgecolor='black', markeredgewidth=.5)#, label=f'{i+1}-Step Predictor')
            plt.text((31.3 - bs)+bs*i, speculation_accuracy[31] - 0.003, f'{i+1}-Step Predictor', color='black')

            if p == 1e-3:
                for d in d_range:
                    lb = speculation_accuracy[d]
                    false_neg, false_pos, both = results[(p,d)][2]
                    num_failures = results[(p,d)][1].count(False)
                    pos_h = (1 - lb) * (false_pos / num_failures)
                    plt.bar(x=(d-bs)+bs*i, width=bw, height=pos_h, bottom=lb, color=color_list[2], label='False Positives' if d == 13 and i == 0 else None, edgecolor='black', linewidth=.5)
                    false_h = (1 - lb) * (false_neg / num_failures)
                    plt.bar(x=(d-bs)+bs*i, width=bw, height=false_h, bottom=lb+pos_h, color=color_list[5], label='False Negatives' if d == 13 and i == 0 else None, edgecolor='black', linewidth=.5)
                    plt.bar(x=(d-bs)+bs*i, width=bw, height=(1 - lb) * both / num_failures, bottom=lb+false_h+pos_h, color=color_list[4], label='False Negative and Positive' if d == 13 and i == 0 else None, edgecolor='black', linewidth=.5)

    plt.xlim(12, 38.5)
    plt.ylim(0.5, 1)
    plt.legend(loc='lower left')
    plt.xticks(d_range)
    plt.ylabel('Accuracy')
    plt.xlabel('Code Distance')
    plt.title(f'Predictor Performance')
    plt.savefig('artifact/figures/4.png', bbox_inches='tight')
    plt.close()

def plot_7(fpga_data_filename):
    """See `notebooks/11_fpga_costs.ipynb` for original plotting code."""
    def log1(d, c1, c0, a):
        return a * d**3 * (c1 * np.log(d) + c0)

    def log2(d, c2, c1, c0, a):
        return a * d**3 * (c2 * np.log(d)**2 + c1 * np.log(d) + c0)

    def log3(d, c3, c2, c1, c0, a):
        return a * d**3 * (c3 * np.log(d)**3 + c2 * np.log(d)**2 + c1 * np.log(d) + c0)

    def poly3(d, c3, c2, c1, c0):
        return c3 * d**3 + c2 * d**2 + c1 * d + c0
    from matplotlib.ticker import MultipleLocator

    fig,ax = plt.subplots(figsize=(3,2))

    print(f'Loading FPGA data from {fpga_data_filename}...')
    with open(fpga_data_filename, 'r') as f:
        data = json.load(f)
        ds = data['ds']
        luts = data['luts']
        regs = data['regs']

    print('Fitting curves to FPGA data...')
    ds1 = [9, 13, 17, 21]
    helios_luts = [52111, 165718, 448314, 898715]
    helios_regs = [13754, 47211, 122028, 238939]
    warnings.filterwarnings('ignore', category=scipy.optimize.OptimizeWarning)
    fit_helios_luts,_ = scipy.optimize.curve_fit(poly3, ds1, helios_luts)
    fit_helios_regs,_ = scipy.optimize.curve_fit(poly3, ds1, helios_regs)

    ds2 = [3, 5, 7, 9, 11, 13, 15]
    mb_luts = [4000, 21000, 66000, 156000, 314000, 553000, 867000]
    fit_mb,_ = scipy.optimize.curve_fit(log1, ds2, mb_luts)

    ds_fit = np.arange(9, 29, 2)

    print(f'Generating figure 7\n')
    ax.plot(ds, luts, '^-', color='C0', label='3-step predictor LUTs')
    ax.plot(ds, regs, '^--', color='C0', markerfacecolor='none', label='3-step predictor registers')
    ax.plot(ds1, helios_luts, 'o-', color='C1', label='Helios LUTs')
    ax.plot(ds1, helios_regs, 'o--', color='C1', markerfacecolor='none', label='Helios registers')
    ax.plot(ds_fit[-4:], poly3(ds_fit[-4:], *fit_helios_luts), ':', color='C1')
    ax.plot(ds_fit[-4:], poly3(ds_fit[-4:], *fit_helios_regs), ':', color='C1')
    ax.plot(ds2, mb_luts, 's-', color='C2', label='Micro Blossom LUTs')
    ax.plot(ds_fit, log1(ds_fit, *fit_mb), ':', color='C2')
    ax.set_xlabel('Code distance')
    ax.set_ylabel('Resource usage')

    # legend off to right
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set_ylim(0, 2e6)
    ax.set_xlim(6, 28)
    ax.set_xticks(range(7, 29, 4))
    ax.xaxis.set_minor_locator(MultipleLocator(2, 1))

    ax.grid(axis='both', alpha=0.5, zorder=-10, linestyle=':', which='both')
    ax.tick_params(direction='in', axis='y')

    plt.savefig('artifact/figures/7.png', bbox_inches='tight')
    plt.close()

def plot_8b(mispredict_data_filename):
    """See `notebooks/04_predictor_accuracy.ipynb` for original plotting code."""
    plt.rcParams["font.family"] = "serif"
    color_list = ['#0072B2', '#CC79A7', '#009E73', '#E69F00', '#56B4E9', '#D55E00', '#F0E442']
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_list)
    
    print(f'Loading misprediction data from {mispredict_data_filename}...')
    with open(mispredict_data_filename, 'r') as f:
        data = json.load(f)
        decode_times = data['decode_times']
        num_nodes = data['num_nodes']
        pes_procs = {int(k):v for k,v in data['pes_procs'].items()}
        pes_classicals = {int(k):v for k,v in data['pes_classicals'].items()}
        adj_procs = {int(k):v for k,v in data['adj_procs'].items()}
        adj_classicals = {int(k):v for k,v in data['adj_classicals'].items()}
        opt_procs = {int(k):v for k,v in data['opt_procs'].items()}
        opt_classicals = {int(k):v for k,v in data['opt_classicals'].items()}

    ### Plotting
    print(f'Generating figure 8b\n')
    fig, axes = plt.subplots(3, 1, figsize=(3.2, 6))
    plt.subplots_adjust(hspace=0.3)

    def interpolate(color1, color2, alpha):
        return LinearSegmentedColormap.from_list('_', [color1, color2])(alpha)

    for i, decode_time in enumerate(decode_times[::-1]):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        real = decode_time * num_nodes

        ax1.bar(0.5, np.mean(pes_procs[decode_time]), color=color_list[0], width=0.2, edgecolor='black')
        ax2.bar(0.7, np.mean(pes_classicals[decode_time]) - real, bottom=real, color=interpolate('w', color_list[5], 0.5), 
            edgecolor='black', width=0.2, hatch='//', label='Wasted Compute' if i == 0 else None)
        ax2.bar(0.7, real, color=color_list[5], width=0.2, edgecolor='black', label='Valid Compute' if i == 0 else None)

        ax1.bar(1.1, np.mean(adj_procs[decode_time]), color=color_list[0], width=0.2, edgecolor='black')
        ax2.bar(1.3, real, color=color_list[5], width=0.2, edgecolor='black')
        ax2.bar(1.3, np.mean(adj_classicals[decode_time]) - real, bottom=real, color=interpolate('w', color_list[5], 0.5), edgecolor='black', width=0.2, hatch='//')

        ax1.bar(1.7, np.mean(opt_procs[decode_time]), color=color_list[0], width=0.2, edgecolor='black')
        ax2.bar(1.9, real, color=color_list[5], width=0.2, edgecolor='black')
        ax2.bar(1.9, np.mean(opt_classicals[decode_time]) - real, bottom=real, color=interpolate('w', color_list[5], 0.5), edgecolor='black', width=0.2, hatch='//')

        ax1.tick_params(axis='y', labelcolor=color_list[0])
        ax2.tick_params(axis='y', labelcolor=color_list[5])

        ax1.set_xticks([0.6, 1.2, 1.8])
        if i == 2:
            ax1.set_xticklabels(['Pessimistic', 'Adjacent', 'Optimistic'])
            ax1.set_xlabel('Speculation Strategy', fontsize=12)
        else:
            ax1.set_xticklabels([])

        if i == 0:
            ax2.legend()

        if i == 1:
            ax1.set_ylabel('Max # Concurrent Processors', color=color_list[0])
            ax2.set_ylabel('Active Decoders × Active Rounds', color=color_list[5])
        
        ax1.set_title(f'Decode Time: {decode_time // 13} Cycles', fontsize=10)

    plt.savefig('artifact/figures/8_b.png', bbox_inches='tight')
    plt.close()

def plot_9():
    """See `notebooks/10_bandwidth_power.ipynb` for original plotting code."""

    def sparse_compress(sample):
        non_zero_idx = np.where(sample)[0]
        g_addr_len = len(sample).bit_length()
        bit_vector = np.zeros(len(non_zero_idx) * g_addr_len, dtype=bool)
        for i, index in enumerate(non_zero_idx):
            bit_vector[g_addr_len*i:g_addr_len*(i+1)] = np.array([b == '1' for b in np.base_repr(index, 2, padding=g_addr_len - int(index).bit_length())], dtype=bool)
        return bit_vector
    p = 1e-3

    def estimate_power(bandwidth):
        # Per cable - 10 gbps
        #           - 31 mW
        num_cables = np.ceil(bandwidth / 1)
        return num_cables * 31 # mW

    def sim(d, L, num_samples):
        # d - code distance
        # L - number surface codes
        circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=2, 
                                    after_clifford_depolarization=p, 
                                    before_measure_flip_probability=p, 
                                    after_reset_flip_probability=p)

        sampler = circ.detector_error_model().compile_sampler()
        det_coords = circ.get_detector_coordinates()
        samples = sampler.sample(num_samples, return_errors=True)

        round_0_dets = []
        for det, coords_2x in det_coords.items():
            if coords_2x[2] == 0:
                round_0_dets.append(det)
                continue

        round_0_errors = []
        for i, err in enumerate(circ.detector_error_model()):
            if err.type == 'error':
                for det in err.targets_copy():
                    if det.is_relative_detector_id() and det_coords[det.val][2] == 0:
                        round_0_errors.append(i)
                        break

        det_samples = samples[0][:,len(round_0_dets):]
        err_samples = samples[2][:,len(round_0_errors):]

        results = { # Optimizations are cumulative e.g. PFU includes cryo control
            'naive': [0,0],
            'cryo_control': [0,0],
            'PFU': [0,0],
            'band_reduce': [0,0]
        }

        for i in range(num_samples):
            dets = det_samples[i]
            errs = err_samples[i]

            gates = d ** 2 * 16 * np.log2(d ** 2) # d^2 ancilla with 4 CX gates and 2 H. Qubit address is log_2(d^2)
            controls = np.count_nonzero(errs) * 2 * np.log2(d ** 2)# Controls for physical corrections X/Z w/ log_2(d^2) address (no PFU)
            full_syndrome = len(dets)
            sparse_syndrome = len(sparse_compress(dets))

            naive_bandwidth = (L * (gates + controls) * 1e-3, L * full_syndrome * 1e-3) # Gbps
            cryo_bandwidth = (L * (controls) * 1e-3, L * full_syndrome * 1e-3)
            PFU_bandwidth = (0, L * (full_syndrome) * 1e-3)
            band_reduce_bandwidth = (0, L * (sparse_syndrome) * 1e-3)


            results['naive'][0] += (naive_bandwidth[0] + naive_bandwidth[1]) / num_samples
            results['naive'][1] += (estimate_power(naive_bandwidth[0]) + estimate_power(naive_bandwidth[1])) / num_samples
                                    
            results['cryo_control'][0] += (cryo_bandwidth[0] + cryo_bandwidth[1]) / num_samples
            results['cryo_control'][1] += (estimate_power(cryo_bandwidth[0]) + estimate_power(cryo_bandwidth[1])) / num_samples

            results['PFU'][0] += (PFU_bandwidth[0] + PFU_bandwidth[1]) / num_samples
            results['PFU'][1] += (estimate_power(PFU_bandwidth[0]) + estimate_power(PFU_bandwidth[1])) / num_samples

            results['band_reduce'][0] += (band_reduce_bandwidth[0] + band_reduce_bandwidth[1]) / num_samples
            results['band_reduce'][1] += (estimate_power(band_reduce_bandwidth[0]) + estimate_power(band_reduce_bandwidth[1])) / num_samples
        

        return results

    print(f'Estimating I/O power costs...')
    d = 21
    num_samples = 1_000
    L_range = 2 * np.arange(1, 51)
    d_range = [9, 13, 17, 21, 25, 29]
    results = {}

    for d in d_range:
        for L in L_range:
            results[(d,L)] = sim(d, L, num_samples)
    
    ### Plotting
    print(f'Generating figure 9\n')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax1 = plt.subplots(figsize=(4.2,3.1))
    ax2 = ax1.twinx() 

    ax1.plot(d_range, [results[(d,50)]['cryo_control'][0] for d in d_range], color="tab:blue",marker='o', label='Naive')
    ax1.plot(d_range, [results[(d,50)]['PFU'][0] for d in d_range], color="tab:orange",marker='s', label='Pauli Frames')
    ax1.plot(d_range, [results[(d,50)]['band_reduce'][0] for d in d_range], color="tab:green",marker='d', label='Syndrome Compression')

    ax2.plot(np.arange(-5, 35), [1500 for i in np.arange(-5, 35)], color='black', linestyle='dashed')
    ax2.text(9, 1550, '4K Power Budget', color='black')

    ax1.arrow(29, results[29,50]['cryo_control'][0], 0, results[29,50]['PFU'][0] - results[29,50]['cryo_control'][0], 
            length_includes_head=True, head_width=0.5, head_length=3, zorder=5, color='black')
    ax1.text(29.4, (results[29,50]['cryo_control'][0] + results[29,50]['PFU'][0])/2, 
            f"{(results[29,50]['PFU'][0] / results[29,50]['cryo_control'][0]):.1f}×")
    ax1.arrow(29, results[29,50]['PFU'][0], 0, results[29,50]['band_reduce'][0] - results[29,50]['PFU'][0], 
            length_includes_head=True, head_width=0.5, head_length=3, zorder=5, color='black')
    ax1.text(29.4, (results[29,50]['PFU'][0] + results[29,50]['band_reduce'][0])/2, 
            f"{(results[29,50]['band_reduce'][0] / results[29,50]['PFU'][0]):.1f}×")


    plt.xticks(d_range, [str(L) for L in d_range])
    plt.xlim(8, 32.1)

    ax1.set_ylim(-1, 80)
    ax2.set_ylim(-1, 80 * 31)

    ax1.set_ylabel('Bandwidth (Gbps)')
    ax1.set_xlabel('Code distance')
    ax2.set_ylabel('Power (mW)')
    ax1.legend(loc='upper left')
    plt.title('4K - 300K Decoding I/O Costs')
    plt.savefig('artifact/figures/9_inset.png', bbox_inches='tight')
    plt.close()

def plot_10_14abc_15ab(benchmark_directories, benchmark_runtime_directories):
    """See `notebooks/09_slurm_data_benchmarks.ipynb` for original plotting
    code."""
    
    print(f'Loading benchmark evaluation data...\n')
    # Benchmark plots
    exclude_benchmarks = [[], [], []]
    config = []
    config_idx_offset = 0
    data_by_config = {}
    for i,directory in enumerate(benchmark_directories):
        with open(f'{directory}/config.json', 'r') as f:
            dir_config = json.load(f)
        config.extend(dir_config)
        for file in os.listdir(f'{directory}/output/'):
            with open(f'{directory}/output/{file}', 'r') as f:
                contents = f.read()
                if len(contents) == 0:
                    print('Empty file!')
                    continue
                data = json.loads(contents)
            config_idx = int(file.split('_')[0][6:])
            if config[config_idx + config_idx_offset]['benchmark_file'].split('/')[-1].split('.')[0] in exclude_benchmarks[i]:
                continue
            data_by_config[config_idx + config_idx_offset] = data
        config_idx_offset = len(config)

    def benchmark_name(conf):
        return conf['benchmark_file'].split('/')[-1].split('.')[0]

    def get_config_idx(config, schedule_name, config_match):
        indices = []
        for idx, conf in enumerate(config):
            if benchmark_name(conf) == schedule_name and all(conf[k] == v for k, v in config_match.items()):
                indices.append(idx)
        if len(indices) == 1:
            return indices[0]
        else:
            print(f'Found {len(indices)} matches for {schedule_name} and {config_match}: {indices}')
        return None

    def get_data(config, data_by_config, schedule_name, config_match):
        config_idx = get_config_idx(config, schedule_name, config_match)
        if config_idx is None:
            return None
        return data_by_config[config_idx]

    hatches = ['', '//', '/////', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    def to_str(key_name, key_val):
        if key_name == 'scheduling_method':
            return key_val
        elif key_name == 'speculation_mode':
            return 'SWIPER' if key_val else 'Default'
        elif key_name == 'benchmark':
            return key_val
        else:
            raise NotImplementedError(f'Unknown key name {key_name}')  

    from matplotlib.colors import LinearSegmentedColormap
    rename_benchmarks = {'grover_ancilla_5':'grover', 'fermi_hubbard_4_4':'fermi_hubbard', 'H2':'H2_molecule', 'rz_1e-10':'rz', 'qrom_15_15':'qrom', 'carleman_2_4':'carleman', 'adder_n18':'adder_8bit', 'qft_10':'qft', 'qpeexact_10':'qpe', 'fermi_hubbard_4_4_Square':'fh_square', 'fermi_hubbard_2_2_Kagome':'fh_kagome', 'heisenberg_3': 'heisenberg'}
    colors = ['C1', 'C0']
    cmaps = [LinearSegmentedColormap.from_list('cmap0', ['white', colors[0]]), LinearSegmentedColormap.from_list('cmap1', ['white', colors[1]])]

    def interpolate(color1, color2, alpha):
        return LinearSegmentedColormap.from_list('_', [color1, color2])(alpha)

    colors_by_keys = {
        ('parallel', None):'C0',
        ('aligned', None):'C1',
        ('parallel', 'separate'):interpolate('white', 'C0', 1.0),
        ('aligned', 'separate'):interpolate('white', 'C1', 1.0),
        ('sliding', 'separate'):interpolate('white', 'C3', 1.0),
    }
    hatches_by_keys = {
        ('parallel', None):'',
        ('aligned', None):'',
        ('parallel', 'separate'):'//',
        ('aligned', 'separate'):'///',
        ('sliding', 'separate'):'/////',
    }
    def plot_data(
            data_by_config,
            config,
            group_by: str,
            group_by_2: str,
            filter_dict: dict[str, Any] = {},
            relative_to: dict[str, Any] | None = None,
            compare_filter_dict: dict[str, Any] = {},
            sorted_keys: list[str] = [],
            sorted_keys_2: list[str] = [],
            sorted_benchmarks: list[str] = [],
            custom_plot_val: tuple[str, str] = ('device_data', 'num_rounds'),
            x_offsets_after: dict[str, float] = {},
            modifiers: dict[str, Any] = {},
            ax = None,
            hidden_benchmarks: list[str] = [],
        ):
        benchmarks = []
        group_keys = []
        group_keys_2 = []
        results = {}
        limited_proc_results = {}
        for key, data in data_by_config.items():
            if not all(config[key][k] == v for k, v in filter_dict.items()):
                continue
            if not data['success']:
                continue
            benchmark = benchmark_name(config[key])
            runtime = data[custom_plot_val[0]][custom_plot_val[1]]
            group_key = config[key][group_by]
            group_key_2 = config[key][group_by_2]
            benchmarks.append(benchmark)
            group_keys.append(group_key)
            group_keys_2.append(group_key_2)

            if data['simulator_params']['max_parallel_processes'] is None:
                results.setdefault((benchmark, group_key, group_key_2), []).append(runtime)  
            else:
                # limited processors
                limited_proc_results.setdefault((benchmark, group_key, group_key_2), []).append(runtime)

        compare_results = {}
        if compare_filter_dict:
            for key, data in data_by_config.items():
                if not all(config[key][k] == v for k, v in compare_filter_dict.items()):
                    continue
                if not data['success']:
                    continue
                benchmark = benchmark_name(config[key])
                runtime = data[custom_plot_val[0]][custom_plot_val[1]]
                group_key = config[key][group_by]
                group_key_2 = config[key][group_by_2]
                
                compare_results.setdefault((benchmark, group_key, group_key_2), []).append(runtime)

        # Remove those that do not have all data
        expected_num_data = len(set((k,k2) for b,k,k2 in results.keys()))
        for benchmark in set(benchmarks) | set(sorted_benchmarks):
            if len(set((k,k2) for b,k,k2 in results.keys() if b == benchmark)) != expected_num_data:
                benchmarks = [b for b in benchmarks if b != benchmark]
                results = {k:v for k,v in results.items() if k[0] != benchmark}
                compare_results = {k:v for k,v in compare_results.items() if k[0] != benchmark}
                if benchmark in sorted_benchmarks:
                    sorted_benchmarks = [b for b in sorted_benchmarks if b != benchmark]

        result_means = {}
        result_stdevs = {}
        for key in results.keys():
            result_means[key] = np.mean(results[key])
            result_stdevs[key] = np.std(results[key]) if len(results[key]) > 1 else 0
        limited_proc_result_means = {}
        limited_proc_result_stdevs = {}
        for key in limited_proc_results.keys():
            limited_proc_result_means[key] = np.mean(limited_proc_results[key])
            limited_proc_result_stdevs[key] = np.std(limited_proc_results[key])
        compare_result_means = {}
        for key in compare_results.keys():
            compare_result_means[key] = np.mean(compare_results[key])

        # sort benchmarks and group keys by average 
        if len(sorted_benchmarks) < len(set(benchmarks)):
            benchmark_avgs = {b:np.mean([r for (bench,_,_),r in result_means.items() if bench == b]) for b in set(benchmarks) if b not in sorted_benchmarks}
            sorted_benchmarks = sorted_benchmarks + sorted(benchmark_avgs, key=lambda k: benchmark_avgs[k])
        sorted_benchmarks = [x for x in sorted_benchmarks if not (x.startswith('memory') or x.startswith('random') or x.startswith('regular') or (x.startswith('rz') and x != 'rz_1e-10'))]
        sorted_benchmarks = [x for x in sorted_benchmarks if x not in hidden_benchmarks]
        if len(sorted_keys) < len(set(group_keys)):
            key_avgs = {k:np.mean([r for (_,key,_),r in result_means.items() if key == k]) for k in set(group_keys)}
            sorted_keys = sorted(key_avgs, key=lambda x: key_avgs[x], reverse=True)
        if len(sorted_keys_2) < len(set(group_keys_2)):
            key_2_avgs = {k:np.mean([r for (_,_,key),r in result_means.items() if key == k]) for k in set(group_keys_2)}
            sorted_keys_2 = sorted(key_2_avgs, key=lambda x: key_2_avgs[x], reverse=True)

        # colors_by_keys = {key:{key2:cmaps[j]((len(sorted_keys_2)-k)/(len(sorted_keys_2))) for k,key2 in enumerate(sorted_keys_2)} for j,key in enumerate(sorted_keys)}

        limited_proc_val_differences = []

        if ax is None:
            fig,ax = plt.subplots(figsize=(9,2))
        
        relative_to_result_means = {}
        x_offset = 0
        centers = []
        for i,benchmark in enumerate(sorted_benchmarks):
            center = i*(len(sorted_keys)+len(sorted_keys_2)+2) + (len(sorted_keys)+len(sorted_keys_2))/2 + x_offset + 1/2
            centers.append(center)
            prev_x = i*(len(sorted_keys)+len(sorted_keys_2)+2) + x_offset
            for j,key in enumerate(sorted_keys):
                for k,key_2 in enumerate(sorted_keys_2):
                    failed = False
                    prev_x += 1
                    try:
                        relative_to_val = result_means[(benchmark, relative_to[group_by], relative_to[group_by_2])] if relative_to is not None else 1
                        result_val = result_means[(benchmark,key,key_2)] / relative_to_val * modifiers.get(benchmark, 1)
                        result_stdev_val = result_stdevs[(benchmark,key,key_2)] / relative_to_val * modifiers.get(benchmark, 1)
                        assert (benchmark,key,key_2) not in relative_to_result_means
                        relative_to_result_means[(benchmark,key,key_2)] = relative_to_val
                        mpl.rcParams['hatch.linewidth'] = 1.0
                        mpl.rcParams['hatch.color'] = 'white'
                    except KeyError as e:
                        failed = True
                        prev_x -= 1
                    if not failed:
                        ax.bar(prev_x, result_val, yerr=result_stdev_val, edgecolor='k', color=colors_by_keys[(key_2,key)], width=1, hatch=hatches_by_keys[(key_2,key)], label=f'{to_str(group_by, key)} {to_str(group_by_2, key_2)}' if i == 0 else None, zorder=5)

                        if (benchmark,key,key_2) in compare_results:
                            ax.bar(prev_x, result_val, yerr=result_stdev_val, edgecolor='k', color='w', alpha=0.5, width=1, zorder=5.1)
                            compare_val = compare_result_means[(benchmark,key,key_2)] / relative_to_val * modifiers.get(benchmark, 1)
                            # ax.bar(prev_x, compare_val, edgecolor='k', width=1, color='none', zorder=6)
                            ax.bar(prev_x, compare_val, edgecolor='k', color=colors_by_keys[(key_2,key)], width=1, hatch=hatches_by_keys[(key_2,key)], zorder=6)
                    
                        if (benchmark,key,key_2) in limited_proc_results and result_stdevs[(benchmark,key,key_2)] > 0:
                            limited_proc_val_differences += [(result - result_means[(benchmark,key,key_2)]) / result_means[(benchmark,key,key_2)] for i,result in enumerate(limited_proc_results[(benchmark,key,key_2)]) if results[(benchmark,key,key_2)][i] > 10**4]
            x_offset += x_offsets_after.get(benchmark, 0)
        ax.set_xticks(centers, [(rename_benchmarks[b] if b in rename_benchmarks else b) for b in sorted_benchmarks], rotation=20, ha='right', fontsize=9)
        # ax.set_xlim(-1, prev_x+1)
        ax.tick_params(direction='in')

        return ax, result_means, result_stdevs, limited_proc_result_means, limited_proc_result_stdevs, sorted_benchmarks, sorted_keys, sorted_keys_2, colors_by_keys, limited_proc_val_differences, compare_result_means, relative_to_result_means
    
    print('Sorting benchmarks by volume...')
    benchmark_info = pd.read_csv('benchmarks/benchmark_info.csv', index_col=0)
    benchmark_info['density'] = benchmark_info['Ideal volume'] / (benchmark_info['Space footprint'] * benchmark_info['Ideal time'])
    benchmarks_sorted_by_volume = benchmark_info.sort_values('T count').index
    
    fig,ax = plt.subplots(2,1, figsize=(10,4), sharex=True)
    d = 21

    microbench = ['msd_15to1', 'toffoli', 'rz_1e-10']
    # hidden_benchmarks = ['adder_n4', 'adder_n10', 'adder_n28', 'qpeinexact_5', 'qpeexact_5', 'fermi_hubbard_2_2', 'qpeinexact_10']
    hidden_benchmarks = ['adder_n4', 'adder_n10', 'adder_n28', 'fermi_hubbard_2_2_Square', 'qpeexact_5']

    print('Generating figure 14a')

    _, result_means, result_stdevs, limited_proc_result_means, limited_proc_result_stdevs, sorted_benchmarks, sorted_keys, sorted_keys_2, colors_by_keys, limited_proc_val_differences, compare_means, relative_to_means, = plot_data(
        data_by_config=data_by_config,
        config=config,
        group_by='speculation_mode',
        group_by_2='scheduling_method',
        filter_dict={'speculation_accuracy':0.9, 'distance': d},
        relative_to={'scheduling_method':'parallel', 'speculation_mode':None},
        compare_filter_dict={'speculation_accuracy': 1.0},
        sorted_benchmarks=microbench + [b for b in benchmarks_sorted_by_volume if b not in microbench],
        x_offsets_after={
            'rz_1e-10':2.0,
        },
        ax=ax[1],
        hidden_benchmarks=hidden_benchmarks,
    )
    ax[1].set_ylabel('Relative runtime', fontsize=10)
    ax[1].axvline(21.5, color='k', linewidth=1)
    # ytick font size
    plt.setp(ax[1].get_yticklabels(), fontsize=9)

    handles, labels = ax[1].get_legend_handles_labels()
    order = [0,1,2,3,4]
    ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, edgecolor='black', frameon=False)
    # ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=9, edgecolor='black', frameon=False)

    ax_inset = ax[0].figure.add_axes([0.63, 0.93, 0.1, 0.07])
    ax_inset.hist(limited_proc_val_differences, bins=20, color='C2')
    ax_inset.tick_params(direction='in', which='both')
    ax_inset.set_xlim(-0.1, 0.1)
    ax_inset.set_xticks([-0.05, 0, 0.05], ['-5%', '0', '5%'], fontsize=8)
    ax_inset.set_yticks([])

    ax[1].grid(axis='y', linestyle=':', alpha=0.5, zorder=-10)
    ax[1].axhline(0.6, color='C2', linestyle='-', linewidth=1, alpha=0.5)
    ax[1].text(ax[1].get_xlim()[1]+1.5, 0.7, '40%\nreduction', fontsize=9, color='C2', va='center', ha='left')
    ax[1].annotate('', (1.01, 0.9), xytext=(1.01, 0.5), xycoords='axes fraction',
                arrowprops=dict(arrowstyle="<-", color='C2'))
    ax[0].grid(axis='y', linestyle=':', alpha=0.5, zorder=-10, which='both')

    ax[0].text(0.74, 0.945, 'Relative performance\nwith fixed processor limit', transform=fig.transFigure, fontsize=9)

    _, _, _, _, _, _, _, _, _, _, _, _, = plot_data(
        data_by_config=data_by_config,
        config=config,
        group_by='speculation_mode',
        group_by_2='scheduling_method',
        filter_dict={'distance': d},
        compare_filter_dict={'speculation_accuracy': 1.0},
        sorted_benchmarks=sorted_benchmarks,
        x_offsets_after={
            'rz_1e-10':2.0,
        },
        ax=ax[0],
        hidden_benchmarks=hidden_benchmarks,
    )

    plt.subplots_adjust(hspace=0.1)
    ax[0].set_ylabel(r'Runtime ($\mu$s)', fontsize=10)
    ax[0].axvline(21.5, color='k', linewidth=1)
    ax[0].tick_params(direction='in', which='both')
    ax[0].set_yscale('log')
    plt.setp(ax[0].get_yticklabels(), fontsize=9)
    plt.savefig('artifact/figures/14_a.png', bbox_inches='tight')
    plt.close()

    def geo_mean(iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    improvements_aligned = []
    improvements_swiper_parallel = []
    improvements_swiper_aligned = []
    improvements_swiper_sliding = []
    for benchmark in sorted_benchmarks:
        improvements_aligned.append(1 - result_means[(benchmark, None, 'aligned')] / result_means[(benchmark, None, 'parallel')])
        improvements_swiper_parallel.append(1 - result_means[(benchmark, 'separate', 'parallel')] / result_means[(benchmark, None, 'parallel')])
        improvements_swiper_aligned.append(1 - result_means[(benchmark, 'separate', 'aligned')] / result_means[(benchmark, None, 'parallel')])
        improvements_swiper_sliding.append(1 - result_means[(benchmark, 'separate', 'sliding')] / result_means[(benchmark, None, 'parallel')])

    print('Benchmark performance improvements:')

    print(f'\tImprovements aligned: {min(improvements_aligned)*100:0.1f}% to {max(improvements_aligned)*100:0.1f}% (mean {np.mean(improvements_aligned)*100:0.1f}%)')
    print(f'\tImprovements SWIPER parallel: {min(improvements_swiper_parallel)*100:0.1f}% to {max(improvements_swiper_parallel)*100:0.1f}% (geometric mean {geo_mean(improvements_swiper_parallel)*100:0.1f}%)')
    print(f'\tImprovements SWIPER aligned: {min(improvements_swiper_aligned)*100:0.1f}% to {max(improvements_swiper_aligned)*100:0.1f}% (geometric mean {geo_mean(improvements_swiper_aligned)*100:0.1f}%)')
    print(f'\tImprovements SWIPER sliding: {min(improvements_swiper_sliding)*100:0.1f}% to {max(improvements_swiper_sliding)*100:0.1f}% (geometric mean {geo_mean(improvements_swiper_sliding)*100:0.1f}%)')

    # greater than 1000 Ts
    improvements_aligned = []
    improvements_swiper_parallel = []
    improvements_swiper_aligned = []
    improvements_swiper_sliding = []
    for benchmark in sorted_benchmarks:
        if benchmark_info.loc[benchmark]['T count'] <= 1000:
            continue
        improvements_aligned.append(1 - result_means[(benchmark, None, 'aligned')] / result_means[(benchmark, None, 'parallel')])
        improvements_swiper_parallel.append(1 - result_means[(benchmark, 'separate', 'parallel')] / result_means[(benchmark, None, 'parallel')])
        improvements_swiper_aligned.append(1 - result_means[(benchmark, 'separate', 'aligned')] / result_means[(benchmark, None, 'parallel')])
        improvements_swiper_sliding.append(1 - result_means[(benchmark, 'separate', 'sliding')] / result_means[(benchmark, None, 'parallel')])

    print('\n\tBenchmarks with more than 1000 Ts:')
    print(f'\tImprovements aligned: {min(improvements_aligned)*100:0.1f}% to {max(improvements_aligned)*100:0.1f}% (geometric mean {geo_mean(improvements_aligned)*100:0.1f}%)')
    print(f'\tImprovements SWIPER parallel: {min(improvements_swiper_parallel)*100:0.1f}% to {max(improvements_swiper_parallel)*100:0.1f}% (geometric mean {geo_mean(improvements_swiper_parallel)*100:0.1f}%)')
    print(f'\tImprovements SWIPER aligned: {min(improvements_swiper_aligned)*100:0.1f}% to {max(improvements_swiper_aligned)*100:0.1f}% (geometric mean {geo_mean(improvements_swiper_aligned)*100:0.1f}%)')
    print(f'\tImprovements SWIPER sliding: {min(improvements_swiper_sliding)*100:0.1f}% to {max(improvements_swiper_sliding)*100:0.1f}% (geometric mean {geo_mean(improvements_swiper_sliding)*100:0.1f}%)')

    # overheads due to missed speculations
    losses_swiper_parallel = []
    losses_swiper_aligned = []
    losses_swiper_sliding = []
    for benchmark in sorted_benchmarks:
        if benchmark_info.loc[benchmark]['T count'] <= 1000:
            continue
        losses_swiper_parallel.append(1 - compare_means[(benchmark, 'separate', 'parallel')] / result_means[(benchmark, 'separate', 'parallel')])
        losses_swiper_aligned.append(1 - compare_means[(benchmark, 'separate', 'aligned')] / result_means[(benchmark, 'separate', 'parallel')])
        losses_swiper_sliding.append(1 - compare_means[(benchmark, 'separate', 'sliding')] / result_means[(benchmark, 'separate', 'parallel')])

    print('\n\tOverheads due to missed speculations (benchmarks with at least 1000 Ts):')
    print(f'\tOverheads SWIPER parallel: {min(losses_swiper_parallel)*100:0.1f}% to {max(losses_swiper_parallel)*100:0.1f}% (geometric mean {geo_mean(losses_swiper_parallel)*100:0.1f}%)')
    print(f'\tOverheads SWIPER aligned: {min(losses_swiper_aligned)*100:0.1f}% to {max(losses_swiper_aligned)*100:0.1f}% (geometric mean {geo_mean(losses_swiper_aligned)*100:0.1f}%)')
    print(f'\tOverheads SWIPER sliding: {min(losses_swiper_sliding)*100:0.1f}% to {max(losses_swiper_sliding)*100:0.1f}% (geometric mean {geo_mean(losses_swiper_sliding)*100:0.1f}%)')
    print(f'\tOverall overheads: {min(losses_swiper_parallel + losses_swiper_aligned + losses_swiper_sliding)*100:0.1f}% to {max(losses_swiper_parallel + losses_swiper_aligned + losses_swiper_sliding)*100:0.1f}% (geometric mean {geo_mean(losses_swiper_parallel + losses_swiper_aligned + losses_swiper_sliding)*100:0.1f}%)')
    
    print('Generating figure 14b')
    fig,ax = plt.subplots(2,1,figsize=(3,3), sharex=True)
    markers = {
        None: {
            'parallel': '^',
            'aligned': 'v',
        },
        'separate': {
            'parallel': 's',
            'aligned': 'p',
            'sliding': 'h',
        }
    }
    made_labels = {key:{key2:False for key2 in set([k for _,_,k in result_means.keys() if _ == key])} for key in set([k for _,k,_ in result_means.keys()])}
    for key,mean in result_means.items():
        stdev = result_stdevs[key]
        if stdev == 0:
            continue
        try:
            t_count = benchmark_info.loc[key[0], 'T count']
            relative_to = result_means[(key[0], None, 'parallel')]
            ax[0].errorbar(t_count, mean / relative_to, yerr=stdev / relative_to, marker=markers[key[1]][key[2]], linestyle='none', markeredgecolor='k' if key[1] else None, markeredgewidth=1, color=colors_by_keys[(key[2],key[1])], label=f'{to_str("speculation_mode", key[1])} {to_str("scheduling_method", key[2])}' if not made_labels[key[1]][key[2]] else None)
            made_labels[key[1]][key[2]] = True
        except KeyError as e:
            continue
    # ax[0].loglog()
    # handles, labels = ax[0].get_legend_handles_labels()
    # order = [1,4,0,2,3]
    # ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', bbox_to_anchor=(1.0, -0.05), fontsize=10, frameon=False)
    # ax[0].set_xlabel('T count')
    ax[0].grid(axis='both', alpha=0.5, zorder=-10, linestyle=':', which='both')
    ax[0].set_ylabel('Relative runtime')
    ax[0].tick_params(direction='in', which='both')
    ax[0].axhline(0.6, color='C2', linestyle='-', linewidth=1, alpha=0.5)
    ax[0].text(ax[1].get_xlim()[1]*1.4, 0.6, '40%\nreduction', fontsize=9, color='C2', va='center')

    xs = []
    ys = []
    made_labels = {key:{key2:False for key2 in set([k for _,_,k in result_means.keys() if _ == key])} for key in set([k for _,k,_ in result_means.keys()])}
    for key,mean in result_means.items():
        stdev = result_stdevs[key]
        if stdev == 0:
            continue
        try:
            t_count = benchmark_info.loc[key[0], 'T count']
        except KeyError as e:
            continue
        xs.append(t_count)
        ys.append(stdev / mean)
        ax[1].errorbar(t_count, stdev / mean, marker=markers[key[1]][key[2]], linestyle='none', markeredgecolor='k' if key[1] else None, markeredgewidth=1, color=colors_by_keys[(key[2],key[1])], label=f'{to_str("speculation_mode", key[1])} {to_str("scheduling_method", key[2])}' if not made_labels[key[1]][key[2]] else None)
        made_labels[key[1]][key[2]] = True
    ax[1].loglog()

    def poly(x, a, k):
        return a*x**k
    result = scipy.optimize.curve_fit(poly, xs, ys)

    ax[1].set_xlabel('Benchmark T count')
    ax[1].set_ylabel(r'$\sigma$ / $\mu$')
    ax[1].grid(axis='both', alpha=0.5, zorder=-10, linestyle=':', which='both')
    ax[1].tick_params(direction='in', which='both')
    ax[1].set_xticks([10, 100, 1000], ['10', '100', '1000'])
    ax[1].set_yticks([0.1, 0.01, 0.001], ['10%', '1%', '0.1%'])

    handles, labels = ax[1].get_legend_handles_labels()
    order = [1,0,4,2,3]
    ax[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.5, -0.4), fontsize=10, frameon=False, ncol=2)

    plt.savefig('artifact/figures/14_b.png', bbox_inches='tight')
    plt.close()

    benchmark_info['t merge ratio'] = benchmark_info['T volume'] / benchmark_info['Merge volume']

    benchmark_info[benchmark_info.index.isin(sorted_benchmarks)].sort_values('T count')

    print(f'Collecting runtimes for runs in {benchmark_runtime_directories}...')
    import datetime as dt

    space_fps = []
    instr_counts = []
    instr_volumes = []
    runtimes = []
    config_idx_offset = 0
    for i,directory in enumerate(benchmark_runtime_directories):
        with open(f'{directory}/config.json', 'r') as f:
            dir_config = json.load(f)
        for conf_idx in range(config_idx_offset, len(dir_config) + config_idx_offset):
            if conf_idx in data_by_config:
                data = data_by_config[conf_idx]
                if not data['success']:
                    continue
                benchmark = benchmark_name(dir_config[conf_idx])
                # if benchmark.startswith('rz'):
                #     continue
                with open(f'{directory}/logs/{conf_idx - config_idx_offset}.out', 'r') as f:
                    try:
                        # 0:40:30.184628
                        timestr = list(f.readlines())[-1].split(' ')[-1].strip()
                        if timestr[1] == ':':
                            timestr = '0' + timestr
                        t = dt.datetime.strptime(timestr, '%H:%M:%S.%f')
                        delta = dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
                        runtimes.append(delta.total_seconds())
                    
                        space_fps.append(benchmark_info.loc[benchmark, 'Space footprint'])
                        instr_counts.append(benchmark_info.loc[benchmark, 'Instruction count'])
                        instr_volumes.append(benchmark_info.loc[benchmark, 'Ideal volume'])
                    except KeyError as e:
                        runtimes = runtimes[:-1]
                        continue
                    except ValueError as e:
                        continue

    print('Generating figure 10 inset')
    fig,ax = plt.subplots(figsize=(2, 1.5))
    ax.plot(np.array(space_fps) * np.array(instr_counts), runtimes, marker='o', linestyle='', markersize=2, alpha=0.5)
    ax.set_xlabel(r'# patches $\times$ instr. count', fontsize=9)

    ax.loglog()
    ax.set_xticks([10**2, 10**3, 10**4, 10**5, 10**6, 10**7])
    ax.set_yticks([1, 60, 60*60, 24*60*60], ['1 sec', '1 min', '1 hour', '1 day'])
    ax.grid(axis='both', linestyle=':', alpha=0.5, zorder=-10, which='both')
    ax.set_title('SWIPER-SIM runtime', fontsize=10)
    ax.tick_params(direction='in', which='both')
    plt.savefig('artifact/figures/10_inset.png', bbox_inches='tight')
    plt.close()

    print('Collecting data on wasted computation due to missed speculations...')
    ## Looking at wasted computation due to missed speculations
    fig,ax = plt.subplots(figsize=(3,2.5))

    parallel_procs = {}
    for key, data in data_by_config.items():
        if not data['success']:
            continue
        benchmark = benchmark_name(config[key])
        procs = data['decoding_data']['max_parallel_decoders']
        group_key = config[key]['speculation_mode']
        group_key_2 = config[key]['scheduling_method']
        if config[key]['max_parallel_processes'] is None:
            # unlimited
            parallel_procs.setdefault((benchmark, group_key, group_key_2, False), []).append(data['decoding_data']['max_parallel_decoders'])
        else:
            parallel_procs.setdefault((benchmark, group_key, group_key_2, True), []).append(data['simulator_params']['max_parallel_processes'])

    xs = []
    ys = [[], [], [], [], [], []]

    for i,benchmark in enumerate(sorted_benchmarks):
        init_len = len(xs)
        try:
            xs.append(np.mean(parallel_procs[(benchmark, None, 'parallel', False)]))
            ys[0].append(np.mean(parallel_procs[(benchmark, 'separate', 'parallel', True)]))
            ys[1].append(np.mean(parallel_procs[(benchmark, 'separate', 'aligned', True)]))
            ys[2].append(np.mean(parallel_procs[(benchmark, 'separate', 'sliding', True)]))
            ys[3].append(np.mean(parallel_procs[(benchmark, 'separate', 'parallel', False)]))
            ys[4].append(np.mean(parallel_procs[(benchmark, 'separate', 'aligned', False)]))
            ys[5].append(np.mean(parallel_procs[(benchmark, 'separate', 'sliding', False)]))
        except:
            xs = xs[:init_len]
            ys = [y[:init_len] for y in ys]

    print('Generating figure 15a')
    # for i,benchmark in enumerate(sorted_benchmarks):
    ax.plot(xs, ys[0], marker=markers['separate']['parallel'], linestyle='none', color=colors_by_keys[('parallel','separate')], label=f'SWIPER parallel (applied limit)')
    ax.plot(xs, ys[1], marker=markers['separate']['aligned'], linestyle='none', color=colors_by_keys[('aligned','separate')], label=f'SWIPER aligned (applied limit)')
    ax.plot(xs, ys[2], marker=markers['separate']['sliding'], linestyle='none', color=colors_by_keys[('sliding','separate')], label=f'SWIPER sliding (applied limit)')
    ax.plot(xs, ys[3], marker=markers['separate']['parallel'], linestyle='none', markerfacecolor='none', color=interpolate('k',colors_by_keys[('parallel','separate')],0.7), markeredgewidth=1, label=f'SWIPER parallel (unlimited usage)')
    ax.plot(xs, ys[4], marker=markers['separate']['aligned'], linestyle='none', markerfacecolor='none', color=interpolate('k',colors_by_keys[('aligned','separate')],0.7), markeredgewidth=1, label=f'SWIPER aligned (unlimited usage)')
    ax.plot(xs, ys[5], marker=markers['separate']['sliding'], linestyle='none', markerfacecolor='none', color=interpolate('k',colors_by_keys[('sliding','separate')],0.7), markeredgewidth=1, label=f'SWIPER sliding (unlimited usage)')

    all_data = np.hstack(ys)
    all_xs = np.hstack([xs for y in ys])
    def fixed_line(x, slope):
        return slope*x
    result = scipy.optimize.curve_fit(fixed_line, all_xs, all_data)
    r2 = 1 - np.sum((all_data - fixed_line(all_xs, *result[0]))**2) / np.sum((all_data - np.mean(all_data))**2)
    ax.plot([0, max(xs)], [0, max(xs)*result[0][0]], color='C2', linestyle='--', label=f'Linear fit: y = {result[0][0]:0.2f}x' + r', $R^2=$' + f'{r2:0.2f}', zorder=-1)

    # ax.plot(xs, np.array(ys[0])+10+np.random.random()*5, marker=markers['separate']['parallel'], linestyle='none', color=colors_by_keys['separate']['parallel'], label=f'SWIPER parallel')
    # ax.plot(xs, np.array(ys[1])+10+np.random.random()*5, marker=markers['separate']['aligned'], linestyle='none', color=colors_by_keys['separate']['aligned'], label=f'SWIPER aligned')
    # ax.plot(xs, np.array(ys[2])+10+np.random.random()*5, marker=markers['separate']['sliding'], linestyle='none', color=colors_by_keys['separate']['sliding'], label=f'SWIPER sliding')
    ax.set_xlim(0)
    ax.set_ylim(0)
    # ax.loglog()
    # ax.set_title('Max. parallel processes')
    ax.grid(axis='both', linestyle=':', alpha=0.5, zorder=-10, which='both')
    ax.set_xlabel('Default parallel window max. proc.')
    ax.set_ylabel('SWIPER max. proc.')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), fontsize=10, frameon=False)
    ax.tick_params(direction='in', which='both')
    plt.savefig('artifact/figures/15_a.png', bbox_inches='tight')
    plt.close()

    print('Generating figure 15b\n')
    fig,ax = plt.subplots(figsize=(3, 0.7))

    wasted_fracs = {
        'sliding': [],
        'aligned': [],
        'parallel': [],
    }

    for key, data in data_by_config.items():
        if not (
            data['success']
            and data['simulator_params']['speculation_mode'] == 'separate'
            and data['simulator_params']['speculation_accuracy'] == 0.9
            and data['simulator_params']['distance'] == 21
            and data['simulator_params']['max_parallel_processes'] is None
        ):
            continue
        decode_vol = data['decoding_data']['decode_process_volume']
        wasted_vol = data['decoding_data']['wasted_decode_volume']
        wasted_fracs[data['simulator_params']['scheduling_method']].append(wasted_vol / decode_vol)

    bins = np.linspace(min(min(wasted_fracs['sliding']), min(wasted_fracs['aligned']), min(wasted_fracs['parallel'])), max(max(wasted_fracs['sliding']), max(wasted_fracs['aligned']), max(wasted_fracs['parallel'])), 40)
    plt.hist(wasted_fracs['sliding'], bins=bins, zorder=5, color='C3', alpha=0.5)
    plt.hist(wasted_fracs['sliding'], bins=bins, zorder=5, color='C3', histtype='step')
    plt.hist(wasted_fracs['parallel'], bins=bins, zorder=5, color='C0', alpha=0.5)
    plt.hist(wasted_fracs['parallel'], bins=bins, zorder=5, color='C0', histtype='step')
    plt.hist(wasted_fracs['aligned'], bins=bins, zorder=5, color='C1', alpha=0.5)
    plt.hist(wasted_fracs['aligned'], bins=bins, zorder=5, color='C1', histtype='step')
    plt.xlabel('Wasted decode fraction')
    plt.yticks([])
    plt.grid(axis='y', linestyle=':', alpha=0.5, zorder=-10)
    plt.savefig('artifact/figures/15_b.png', bbox_inches='tight')
    plt.close()

def plot_12ab(reaction_times_directory, decoder_dist_filename):
    """See `notebooks/08_slurm_data_reaction_time.ipynb` for original plotting
    code."""
    print(f'Loading reaction time data from {reaction_times_directory}...')
    with open(f'{reaction_times_directory}/config.json', 'r') as f:
        config = json.load(f)
    data_by_config = {}

    for file in os.listdir(f'{reaction_times_directory}/output/'):
        with open(f'{reaction_times_directory}/output/{file}', 'r') as f:
            contents = f.read()
            if len(contents) == 0:
                print('Empty file!')
                continue
            data = json.loads(contents)
        config_idx = int(file.split('_')[0][6:])
        data_by_config[config_idx] = data

    ordered_configs = [
        ('parallel', True),
        ('aligned', True),
        ('sliding', True),
    ]

    broken_configs = []
    all_data = {conf:{} for conf in ordered_configs}
    for (scheduling_method,spec_on) in ordered_configs:
        made_label = False
        for config_idx,data in data_by_config.items():
            spec = config[config_idx]['speculation_mode'] != None
            sched = config[config_idx]['scheduling_method']
            if sched != scheduling_method or spec != spec_on:
                continue
            decoding_latency = float(config[config_idx]['decoder_latency_or_dist_filename'].split('*')[-1])
            speculation_accuracy = config[config_idx]['speculation_accuracy']
            success = data['success']
            if success:
                vals = data['device_data']['conditioned_decode_wait_times']
                # assert len(vals) == 1000
                if np.mean([vals[k] for k in list(sorted(vals.keys()))[-100:]]) > 1.2 * np.mean([vals[k] for k in list(sorted(vals.keys()))[100:200]]):
                    success = False
            if success:
                # use avg
                all_data[(scheduling_method,spec_on)].setdefault(speculation_accuracy, {})[decoding_latency] = data['device_data']['avg_conditioned_decode_wait_time']
            else:
                # use last
                # all_data[(scheduling_method,spec_on)].setdefault(speculation_accuracy, {})[decoding_latency] = data['device_data']['conditioned_decode_wait_times']
                all_data[(scheduling_method,spec_on)].setdefault(speculation_accuracy, {})[decoding_latency] = 10**15
                broken_configs.append((scheduling_method,spec_on,speculation_accuracy,decoding_latency))
            # TODO: should mark any whose values continually increase over
            # experiment duration as "not scalable" (use hatches to mark on graph,
            # or color gray)

    print(f'Loading decoder distance data from {decoder_dist_filename}...')
    with open(decoder_dist_filename, 'r') as f:
        decoder_dists = json.load(f)

    distances = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    volumes = [2,3,4,5,6,7]

    from scipy.optimize import curve_fit
    def f(x, a, b):
        return a*x+b

    print('Fitting relative latency factors to PyMatching data...')
    fit_rs = []
    for i,dist in enumerate(distances):
        xs = []
        ys = []
        for volume_str,data in decoder_dists[str(dist)].items():
            volume = int(volume_str)
            for latency in data:
                xs.append(volume*dist)
                ys.append(latency)
        popt, pcov = curve_fit(f, xs, ys)
        a,b = popt
        fit_rs.append(a)
        # print(f'\tb = {b:.2f} +- {pcov[1,1]:.2f}')
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib as mpl

    print('Generating figure 12a')
    fig,ax = plt.subplots(figsize=(5,4))
    colors = ['C0', 'C1', 'C3']
    def interpolate(color1, color2, alpha):
        return LinearSegmentedColormap.from_list('_', [color1, color2])(alpha)
    def interpolate_white(color, alpha):
        return interpolate('white', color, alpha)
    zorders = [2,1.9,1.8]
    x_offsets = [-0.5, 0.25, 1.0]
    plot_lines = []
    legend_lines = []
    for i,(scheduling_method,spec_on) in enumerate(ordered_configs):
        data = all_data[(scheduling_method,spec_on)]
        for j,spec_acc in enumerate(sorted(data.keys())):
            keys = sorted(data[spec_acc].keys())
            xvals = keys.copy()
            if len(xvals) == 10:
                xvals[-1] += x_offsets[i]
            if spec_acc == 0:
                alpha = 1
                linestyle = '--'
            else:
                alpha = spec_acc
                linestyle = '-'
            if j+1 < len(data):
                vals_next = data[sorted(data.keys())[j+1]]
                for k,dec_lat in enumerate(sorted(data[spec_acc].keys())):
                    if k != len(data[spec_acc])-1:
                        dec_lat_next = sorted(data[spec_acc].keys())[k+1]
                        # if (scheduling_method, spec_on, spec_acc, dec_lat_next) in broken_configs:
                        #     color = 'gray'
                        # else:
                        color = interpolate_white(colors[i], alpha)
                        ax.fill_between(
                            [xvals[k], xvals[k+1]],
                            [data[spec_acc][dec_lat], data[spec_acc][dec_lat_next]],
                            [vals_next[dec_lat], vals_next[dec_lat_next]],
                            color=color,
                            zorder=zorders[i]+0.001*(len(data)-j),
                            alpha=0.5,
                        )
            # alpha = (alpha+1)/2 if spec_acc > 0 else alpha
            lines, = ax.plot(xvals, [data[spec_acc][k] for k in sorted(data[spec_acc].keys())], label=(f'{scheduling_method}' if spec_acc == 1 else None), color=interpolate_white(colors[i], alpha) if spec_acc > 0 else colors[i], linestyle=linestyle, zorder=(zorders[i]+0.01+0.001*(len(data)-j)))
            if spec_acc == 1:
                plot_lines.append(lines)
            if i == 0:
                lines, = ax.plot([-2, -1], [0, 0], label=(f'{spec_acc*100:.0f}%' if spec_acc > 0 else '0%'), color=interpolate_white('gray', alpha) if spec_acc > 0 else 'gray', linestyle=linestyle)
                legend_lines.append(lines)
    leg1 = ax.legend(plot_lines, [method for method,_ in ordered_configs], title='Scheduling\nmethod', edgecolor='black', frameon=False)
    leg2 = ax.legend(legend_lines, [f'{spec_acc*100:.0f}%' for spec_acc in sorted(data.keys())], title='SWIPER\nspeculation\naccuracy', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    ax.add_artist(leg1)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xticks(sorted(all_data[('parallel', True)][1.0].keys()),
    # sorted(all_data[('parallel', True)][1.0].keys()))
    plt.xticks([0.1, 1.0, 10], ['0.1', '1.0', '10'])
    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.xlabel('Decoding latency / code distance')
    plt.ylabel(r'Avg. reaction time ($\mu$s)')
    plt.ylim(2e1, 2e3)
    plt.xlim(0.1, 11)

    textcolor1 = interpolate('C2', 'black', 0.3)
    textcolor2 = interpolate('C5', 'black', 0.3)
    # for dist,r in zip(distances, fit_rs):
    #     plt.plot([r, r], [plt.ylim()[0], 23], color=textcolor, linestyle='-', linewidth=1)
    #     plt.text(r, 23, f'{dist}', va='bottom', ha='center', fontsize=9, color=textcolor)
    # plt.text(0.43, 30, 'PyMatching equivalent latency', va='bottom', ha='left', fontsize=9, color=textcolor)
    # plt.text(0.43, 23, 'at d =', va='bottom', ha='left', fontsize=9,
    # color=textcolor)
    y1 = 23
    y2 = 35

    dist_lers = {13: -9, 19: -12, 25: -15}
    for dist,r in zip(distances, fit_rs):
        if dist in [13, 19, 25]:
            plt.plot([r, r], [plt.ylim()[0], y1], color=textcolor2, linestyle='-', linewidth=1, alpha=0.5)
            plt.text(r, y1, r'$10^{' + f'{dist_lers[dist]}' + r'}$', va='bottom', ha='center', fontsize=9, color=textcolor2)
    plt.text(0.22, y1, 'RNN at LER:', va='bottom', ha='left', fontsize=9, color=textcolor2)

    dist_lers = {15: -9, 21: -12, 27: -15}
    for dist,r in zip(distances, fit_rs):
        if dist in [15, 21, 27]:
            plt.plot([r, r], [plt.ylim()[0], y2], color=textcolor1, linestyle='-', linewidth=1, alpha=0.5)
            plt.text(r, y2, r'$10^{' + f'{dist_lers[dist]}' + r'}$', va='bottom', ha='center', fontsize=9, color=textcolor1)
    plt.text(0.4, y2, 'PyM at LER:', va='bottom', ha='left', fontsize=9, color=textcolor1)

    plt.tick_params(direction='in', which='both')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.savefig('artifact/figures/12_a.png', bbox_inches='tight')
    plt.close()

    print('Running decoder headroom analysis...')
    rs = sorted(all_data[('parallel', True)][0.0].keys())
    ys_baseline = [all_data[('parallel', True)][0.0][x] for x in rs]
    fit_baseline = np.polyfit(rs, ys_baseline, 1)
    invert_fit = [1/fit_baseline[0], -fit_baseline[1]/fit_baseline[0]]
    min_react_time = min(ys_baseline)

    fig,ax = plt.subplots(figsize=(5,2))
    # parallel_headrooms = []
    # aligned_headrooms = []
    # accuracies = np.linspace(0, 1, 11)
    accuracies = [0.9, 1]
    # alphas = np.linspace(0, 1, 11)
    alphas = np.linspace(0.7, 1, 2)
    linestyles = ['-.', '-']
    # alphas[0] = 1
    plot_lines = []
    legend_lines = []
    for i,accuracy in enumerate(accuracies):
        ys_swiper_parallel = [all_data[('parallel', True)][accuracy][x] for x in rs]
        ys_swiper_aligned = [all_data[('aligned', True)][accuracy][x] for x in rs]
        ys_swiper_sliding = [all_data[('sliding', True)][accuracy][x] for x in rs]
        parallel_xs = []
        parallel_headrooms = []
        aligned_xs = []
        aligned_headrooms = []
        sliding_xs = []
        sliding_headrooms = []
        for r,react_time in zip(rs, ys_swiper_parallel):
            if react_time >= min_react_time:
                parallel_xs.append(react_time)
                parallel_headrooms.append(r / np.polyval(invert_fit, react_time))
        for r,react_time in zip(rs, ys_swiper_aligned):
            if react_time >= min_react_time:
                aligned_xs.append(react_time)
                aligned_headrooms.append(r / np.polyval(invert_fit, react_time))
        for r,react_time in zip(rs, ys_swiper_sliding):
            if react_time >= min_react_time:
                sliding_xs.append(react_time)
                sliding_headrooms.append(r / np.polyval(invert_fit, react_time))

        lines1, = ax.plot(parallel_xs, parallel_headrooms, alpha=alphas[i], color='C0', linestyle=linestyles[i], zorder=1, label='parallel')
        lines2, = ax.plot(aligned_xs, aligned_headrooms, alpha=alphas[i], color='C1', linestyle=linestyles[i], zorder=2, label='aligned')
        if accuracy >= 0.7:
            lines3, = ax.plot(sliding_xs, sliding_headrooms, alpha=alphas[i], color='C3', linestyle=linestyles[i], zorder=3, label='sliding')
        if accuracy == 1:
            plot_lines.append(lines1)
            plot_lines.append(lines2)
            plot_lines.append(lines3)

    print('Generating figure 12b\n')
    xlim = ax.get_xlim()
    for i,spec_acc in enumerate(accuracies):
        lines, = ax.plot([-2, -1], [1, 1], label=(f'{spec_acc*100:.0f}%' if spec_acc > 0 else '0%'), color=interpolate_white('gray', alphas[i]) if spec_acc > 0 else 'gray', linestyle=linestyles[i])
        legend_lines.append(lines)
    ax.set_xlim(0)

    # ax.set_xscale('log')
    ax.set_xlabel(r'Fixed reaction time ($\mu$s)')
    ax.set_ylabel('Decoder headroom')
    ax.set_yticks([1,2,3,4,5], [r'$1\times$', r'$2\times$', r'$3\times$', r'$4\times$', r'$5\times$'])
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.tick_params(direction='in', which='both')
    leg1 = ax.legend(plot_lines, ['parallel', 'aligned', 'sliding'], title='Scheduling\nmethod', bbox_to_anchor=(1.02, 0.99), loc='upper right', frameon=False)
    leg2 = ax.legend(legend_lines, ['90%', '100%'], title='Speculation\naccuracy', bbox_to_anchor=(0.76, 0.99), loc='upper right', frameon=False, ncol=1, fontsize=10)
    ax.add_artist(leg1)
    plt.savefig('artifact/figures/12_b.png', bbox_inches='tight')
    plt.close()

def plot_11ab_12c():
    """See `notebooks/00_window_test.ipynb` for original plotting code."""

    print('Simulating RegularTSchedule...')

    scheduling_method = 'sliding'
    schedule = LatticeSurgerySchedule(True)
    schedule.idle([(0,0)], 15)
    schedule.inject_T([(1,0)])
    idx = schedule.merge([(0,0), (1,0)])
    schedule.discard([(1,0)])
    schedule.Y_meas((0,0), idx)
    schedule.discard([(0,0)])

    simulator = DecodingSimulator()
    success, sim_params, device_data, window_data, decoding_data = simulator.run(
        schedule=RegularTSchedule(10, 0).schedule,
        distance=7,
        scheduling_method='parallel',
        decoding_latency_fn=lambda x: 7,
        speculation_mode='separate',
        speculation_latency=1,
        speculation_accuracy=0.9,
        max_parallel_processes=None,
        progress_bar=False,
        rng=0,
        lightweight_setting=0,
    )

    print('Generating figure 12')
    ax = plotter.plot_device_schedule_trace(device_data, spacing=1, z_min=120)
    ax.set_axis_off()
    plt.savefig('artifact/figures/12_c.png', bbox_inches='tight', dpi=300)
    plt.close()

    print('Simulating MSD15To1Schedule...')
    ## 15 - 1 Factory
    speculation_mode, scheduling_method = 'separate', 'aligned'
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=MSD15To1Schedule().schedule,
        distance=7,
        scheduling_method=scheduling_method,
        decoding_latency_fn=lambda x: 14,
        speculation_mode=speculation_mode,
        speculation_latency=1,
        speculation_accuracy=0.9,
        poison_policy='successors',
        max_parallel_processes=None,
        progress_bar=False,
        rng=0,
        lightweight_setting=0,
    )

    print('Generating figure 11a')
    ax = plotter.plot_device_schedule_trace(device_data, spacing=1, z_max=106)
    plt.savefig(f'artifact/figures/11_a.png', bbox_inches='tight', dpi=300)
    plt.close()

    print('Simulating MSD15To1Schedule...')
    speculation_mode, scheduling_method = None, 'parallel'
    simulator = DecodingSimulator()
    success, _, device_data, window_data, decoding_data = simulator.run(
        schedule=MSD15To1Schedule().schedule,
        distance=7,
        scheduling_method=scheduling_method,
        decoding_latency_fn=lambda x: 14,
        speculation_mode=speculation_mode,
        speculation_latency=1,
        speculation_accuracy=0.9,
        poison_policy='successors',
        max_parallel_processes=None,
        progress_bar=False,
        rng=0,
        lightweight_setting=0,
    )

    print('Generating figure 11b\n')
    ax = plotter.plot_device_schedule_trace(device_data, spacing=1, z_max=106)
    plt.savefig(f'artifact/figures/11_b.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
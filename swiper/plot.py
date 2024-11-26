import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from swiper.device_manager import DeviceData
from swiper.window_builder import DecodingWindow
from swiper.window_manager import WindowData
from swiper.decoder_manager import DecoderData

plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#0072B2', '#CC79A7', '#009E73', '#E69F00', '#56B4E9', '#D55E00', '#F0E442']) 
plt.rcParams['font.family'] = 'serif'

def plot_device_schedule_trace(
        data: DeviceData,
        spacing: int = 1,
        do_z_offset: bool = False,
        color_dict: dict[str, str] = {
            'MERGE': 'gray',
            'INJECT_T': 'green',
            'Y_MEAS': 'gold',
            'IDLE': 'white',
            'DECODE_IDLE': 'firebrick',
        },
        edgecolor_dict: dict[str, str] = {
            'MERGE': 'dimgray',
            'INJECT_T': 'darkgreen',
            'Y_MEAS': 'orange',
            'IDLE': 'lightgray',
            'DECODE_IDLE': 'maroon',
        },
        windows: list[DecodingWindow] = [],
        window_schedule_times: list[int] = [],
        window_cmap: str | None = 'viridis',
        window_buffers_to_highlight: list[int] = [],
        selected_window_colors: list[str] = [
            'firebrick', 'pink', 'orange',
        ],
        hide_z_ticks: bool = False,
        default_fig: plt.Figure | None = None,
        z_min: int | None = None,
        z_max: int | None = None,
        ax: mpl.axes.Axes | None = None,
    ):
    if default_fig and not ax:
        fig = default_fig
        ax = fig.add_subplot(111, projection='3d')
    elif ax and not default_fig:
        fig = ax.figure
    elif not ax and not default_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = default_fig
    assert ax and fig

    min_x = min([r for r,c in data.all_patch_coords])
    min_y = min([c for r,c in data.all_patch_coords])
    rows = max([r for r,c in data.all_patch_coords]) - min_x + 1
    cols = max([c for r,c in data.all_patch_coords]) - min_y + 1

    num_rounds = data.num_rounds
    if z_max:
        num_rounds = z_max
    if z_min:
        num_rounds -= z_min
    x,y,z = np.meshgrid(np.cumsum([0]+[data.d, spacing]*cols), np.cumsum([0]+[data.d, spacing]*rows), np.arange((spacing+1)*num_rounds+2))

    volume = np.zeros((2*rows, 2*cols, (spacing+1)*num_rounds+1))
    colors = np.empty_like(volume, dtype=object)
    edgecolors = np.empty_like(volume, dtype=object)
    linewidth = (0.5 if len(windows) == 0 else 1)

    def get_color(window_idx):
        if window_idx in window_buffers_to_highlight:
            return selected_window_colors[window_buffers_to_highlight.index(window_idx)]
        elif window_cmap:
            return plt.cm.get_cmap(window_cmap)(window_schedule_times[window_idx] / max(window_schedule_times))         
        else:
            return 'white'

    z_offset = 0
    increased_z = False
    for round_idx, round_data in enumerate(data.generated_syndrome_data):
        if z_max and round_idx > z_max:
            continue
        if z_min:
            if round_idx < z_min:
                continue
            round_idx -= z_min

        # if there is a discard operation in the previous round, bump up z coord
        # to avoid connecting the patches
        if do_z_offset and round_idx > 0 and np.any([syndrome.initialized_patch for syndrome in data.generated_syndrome_data[round_idx]]):
            print('WARNING: experimental feature do_z_offset is enabled, but is not yet fully implemented')
            z_offset += spacing
            increased_z = True

        for i, syndrome in enumerate(round_data):
            coords = (syndrome.patch[0] - min_x, syndrome.patch[1] - min_y, round_idx)
            name = syndrome.instruction.name

            if len(windows) == 0:
                # color by instruction
                alpha = 1.0
                color = color_dict[name]
                edgecolor = edgecolor_dict[name]
            else:
                # color by window
                alpha = 0.5
                containing_window_idx = -10**10
                for window_idx, window in enumerate(windows):
                    if any(cr.contains_syndrome_round(syndrome_round=syndrome) for cr in window.commit_region):
                        if containing_window_idx >= 0:
                            print(f'WARNING: multiple commit regions contain the same syndrome round! Syndrome: {syndrome}, Windows: {containing_window_idx}, {window_idx}')
                        containing_window_idx = window_idx
                if containing_window_idx < 0:
                    # not in a commit region
                    continue
                color = get_color(containing_window_idx)

                containing_buffer_idx = -10**10
                for window_idx in window_buffers_to_highlight:
                    if any(buffer.contains_syndrome_round(syndrome_round=syndrome) for buffer in windows[window_idx].buffer_regions):
                        if containing_buffer_idx >= 0:
                            print(f'WARNING: multiple highlighted buffer regions contain the same syndrome round! Syndrome: {syndrome}, Windows: {containing_buffer_idx}, {window_idx}')
                        containing_buffer_idx = window_idx
                if containing_buffer_idx >= 0:
                    edgecolor = get_color(containing_buffer_idx)
                else:
                    edgecolor = (color, 0.3)
                    
            if name == 'MERGE':
                # fill extra space between patches of same instruction
                for j, syndrome2 in enumerate(round_data):
                    if syndrome2.instruction == syndrome.instruction:
                        if syndrome2.patch[0] == coords[0] and abs(syndrome2.patch[1] - coords[1]) == 1:
                            # same x coordinate, adjacent y coordinates
                            y_fill = min(syndrome2.patch[1], coords[1])
                            volume[coords[0]*2, max(0, 2*y_fill+1), coords[2]+z_offset] = 1
                            colors[coords[0]*2, max(0, 2*y_fill+1), coords[2]+z_offset] = color
                            edgecolors[coords[0]*2, max(0, 2*y_fill+1), coords[2]+z_offset] = edgecolor
                        elif syndrome2.patch[1] == coords[1] and abs(syndrome2.patch[0] - coords[0]) == 1:
                            # same y coordinate, adjacent x coordinates
                            x_fill = min(syndrome2.patch[0], coords[0])
                            volume[max(0, 2*x_fill+1), coords[1]*2, coords[2]+z_offset] = 1
                            colors[max(0, 2*x_fill+1), coords[1]*2, coords[2]+z_offset] = color
                            edgecolors[max(0, 2*x_fill+1), coords[1]*2, coords[2]+z_offset] = edgecolor

            volume[coords[0]*2, coords[1]*2, coords[2]+z_offset] = 1
            colors[coords[0]*2, coords[1]*2, coords[2]+z_offset] = color
            edgecolors[coords[0]*2, coords[1]*2, coords[2]+z_offset] = edgecolor

            # if we increased z_offset this round, connect any pre-existing
            # patches across the z_offset
            if increased_z:
                if not syndrome.initialized_patch:
                    volume[coords[0]*2, coords[1]*2, coords[2]+z_offset-spacing:coords[2]+z_offset] = 1
                    colors[coords[0]*2, coords[1]*2, coords[2]+z_offset-spacing:coords[2]+z_offset] = color
                    edgecolors[coords[0]*2, coords[1]*2, coords[2]+z_offset-spacing:coords[2]+z_offset] = edgecolor
    ax.voxels(x,y,z, filled=volume, facecolors=colors, edgecolors=edgecolors, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45), alpha=alpha, linewidths=linewidth)
    
    if z_min:
        ax.set_zlim(bottom=z_min)
    if z_max:
        ax.set_zlim(top=z_max)
    ax.set_aspect('equal')

    ax.view_init(elev=15, azim=30)
    ax.set_xticks([])
    ax.set_yticks([])
    if hide_z_ticks:
        ax.set_zticks([])

    return ax

def plot_windows(
        device_data: DeviceData,
        window_data: WindowData,
        decoder_data: DecoderData,
        window_buffers_to_highlight: list[int] = [],
        selected_window_colors: list[str] = [
            'firebrick', 'pink', 'orange',
        ],
        **kwargs,
    ):
    return plot_device_schedule_trace(
        data=device_data,
        windows=[window_data.get_window(w_idx) for w_idx in window_data.all_constructed_windows], 
        window_schedule_times=[decoder_data.window_decoding_completion_times[w_idx] for w_idx in window_data.all_constructed_windows],
        window_buffers_to_highlight=window_buffers_to_highlight,
        selected_window_colors=selected_window_colors,
        **kwargs,
    )

################################################################################
# TODO: redo with plot_cube_at? Should be much faster
################################################################################

def cuboid_data(pos, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/42611693
    # (in turn taken from https://stackoverflow.com/a/35978146/4124317)
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plot_cube_at(ax, pos, size=(1,1,1), color: str | tuple = 'b', alpha=1.0):
    # adapted from https://stackoverflow.com/a/42611693
    X, Y, Z = cuboid_data(pos, size=size)
    ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=alpha)
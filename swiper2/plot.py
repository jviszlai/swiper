import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from swiper2.device_manager import DeviceData
from swiper2.window_builder import DecodingWindow

def plot_device_schedule_trace(
        data: DeviceData,
        spacing: int = 1,
        do_z_offset: bool = False,
        color_dict: dict[str, str] = {
            'MERGE': 'gray',
            'INJECT_T': 'green',
            'CONDITIONAL_S': 'gold',
            'IDLE': 'white',
            'UNWANTED_IDLE': 'firebrick'
        },
        edgecolor_dict: dict[str, str] = {
            'MERGE': 'dimgray',
            'INJECT_T': 'darkgreen',
            'CONDITIONAL_S': 'orange',
            'IDLE': 'lightgray',
            'UNWANTED_IDLE': 'maroon'
        },
        windows: list[DecodingWindow] = [],
        window_schedule_times: list[int] = [],
        window_cmap: str = 'viridis',
        window_buffers_to_highlight: list[int] = [],
        selected_window_colors: list[str] = [
            'firebrick', 'pink', 'orange',
        ],
        default_fig: plt.Figure | None = None,
    ):
    fig = plt.figure() if not default_fig else default_fig
    ax = fig.add_subplot(111, projection='3d')

    rows = max([r for r,c in data.all_patch_coords]) - min([r for r,c in data.all_patch_coords]) + 1
    cols = max([c for r,c in data.all_patch_coords]) - min([c for r,c in data.all_patch_coords]) + 1

    x,y,z = np.meshgrid(np.cumsum([0]+[data.d, spacing]*cols), np.cumsum([0]+[data.d, spacing]*rows), np.arange((spacing+1)*data.num_rounds+2))

    volume = np.zeros((2*rows, 2*cols, (spacing+1)*data.num_rounds+1))
    colors = np.empty_like(volume, dtype=object)
    edgecolors = np.empty_like(volume, dtype=object)
    linewidth = (0.5 if len(windows) == 0 else 1)

    def get_color(window_idx):
        if window_idx in window_buffers_to_highlight:
            return selected_window_colors[window_buffers_to_highlight.index(window_idx)]
        else:
            return plt.cm.get_cmap(window_cmap)(window_schedule_times[window_idx] / max(window_schedule_times))         

    z_offset = 0
    increased_z = False
    for round_idx, round_data in enumerate(data.generated_syndrome_data):

        # if there is a discard operation in the previous round, bump up z coord
        # to avoid connecting the patches
        if do_z_offset and round_idx > 0 and np.any([syndrome.initialized_patch for syndrome in data.generated_syndrome_data[round_idx]]):
            print('WARNING: experimental feature do_z_offset is enabled, but is not yet fully implemented')
            z_offset += spacing
            increased_z = True

        for i, syndrome in enumerate(round_data):
            coords = syndrome.patch + (round_idx,)
            if syndrome.is_unwanted_idle:
                name = 'UNWANTED_IDLE'
            else:
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
                    if any(cr.contains_syndrome_round(syndrome) for cr in window.commit_region):
                        if containing_window_idx >= 0:
                            print(f'WARNING: multiple commit regions contain the same syndrome round! Syndrome: {syndrome}, Windows: {containing_window_idx}, {window_idx}')
                        containing_window_idx = window_idx
                if containing_window_idx < 0:
                    # not in a commit region
                    continue
                color = get_color(containing_window_idx)

                containing_buffer_idx = -10**10
                for window_idx in window_buffers_to_highlight:
                    if any(buffer.contains_syndrome_round(syndrome) for buffer in windows[window_idx].buffer_regions):
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
    ax.set_aspect('equal')
    ax.view_init(elev=15, azim=30)
    ax.set_xticks([])
    ax.set_yticks([])
    # flip both x and y axes to match the orientation of the device
    # ax.invert_xaxis()
    # ax.invert_yaxis()

    return ax

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

def plot_windows(
        windows: list[DecodingWindow] = [],
        window_schedule_layers: list[int] = [],
        discrete_window_colors: list[str] = [
            'green', 'gold', 'firebrick', 'navy', 'pink',
        ],
        window_cmap: str = 'viridis',
        space_multiplier: int = 1,
        window_buffers_to_highlight: list[int] = [],
        buffer_offset: float = 0.1,
    ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    use_cmap = (len(np.unique(window_schedule_layers)) > len(discrete_window_colors))

    def get_color(window_idx):
        if use_cmap:
            return plt.cm.get_cmap(window_cmap)(window_schedule_layers[window_idx] / (len(np.unique(window_schedule_layers))-1))
        else:
            return discrete_window_colors[window_schedule_layers[window_idx]]

    for window_idx, window in enumerate(windows):
        for space_coords in window.commit_region.space_footprint:
            plot_coords = tuple([coord * space_multiplier for coord in space_coords])
            plot_cube_at(ax, space_coords + (window.commit_region.round_start,), size=(1,1,window.commit_region.duration), color=get_color(window_idx), alpha=0.5)
        avg_commit_spatial_position_x = np.mean([coords[0] for coords in window.commit_region.space_footprint])
        avg_commit_spatial_position_y = np.mean([coords[1] for coords in window.commit_region.space_footprint])
        avg_commit_coords = (avg_commit_spatial_position_x, avg_commit_spatial_position_y, window.commit_region.round_start + window.commit_region.duration/2)

        for window_idx_2, window_2 in enumerate(windows):
            if window_idx_2 in window_buffers_to_highlight:
                for buffer_region in window_2.buffer_regions:
                    for space_coords in buffer_region.space_footprint:
                        plot_coords = tuple([coord * space_multiplier for coord in space_coords])
                        plot_cube_at(ax, space_coords + (buffer_region.round_start,), size=(1,1,buffer_region.duration), color='black', alpha=0.5)

        # for buffer_region in window.buffer_regions:
        #     for space_coords in buffer_region.space_footprint:
        #         avg_z = buffer_region.round_start + buffer_region.duration/2
        #         x_y_or_z = -1
        #         sign = 0
        #         extent = 0
        #         if np.isclose(avg_commit_coords[0], space_coords[0]) and np.isclose(avg_commit_coords[1], space_coords[1]):
        #             x_y_or_z = 2
        #             if avg_z < avg_commit_coords[2]:
        #                 sign = -1
        #             else:
        #                 sign = 1
        #             extent = buffer_region.duration
        #         elif np.isclose(avg_commit_coords[0], space_coords[0]):
        #             assert np.isclose(avg_commit_coords[2], avg_z)
        #             x_y_or_z = 1
        #             if space_coords[1] < avg_commit_coords[1]:
        #                 sign = -1
        #             else:
        #                 sign = 1
        #             extent = 1
        #         else:
        #             assert np.isclose(avg_commit_coords[1], space_coords[1])
        #             assert np.isclose(avg_commit_coords[2], avg_z)
        #             x_y_or_z = 0
        #             if space_coords[0] < avg_commit_coords[0]:
        #                 sign = -1
        #             else:
        #                 sign = 1
        #             extent = 1

        #         if sign > 0:
        #             # window_mask = [-0.55, -0.3, 0.3, 0.55]
        #             # step = 0.1
        #             window_mask = np.linspace(-0.55, 0.55, 20)
        #             step = (window_mask[1] - window_mask[0]) / 4
        #             # window_mask = window_mask[np.arange(len(window_mask)) % 4 == 0]
        #         else:
        #             # window_mask = [-0.55, -0.3, 0.3, 0.55]
        #             # step = 0.1
        #             window_mask = np.linspace(-0.55, 0.55, 20)
        #             step = (window_mask[1] - window_mask[0]) / 4
        #             # window_mask = window_mask[np.arange(len(window_mask)) % 4 == 2]

        #         # plot skinny rectangles around this cube
        #         offsets = np.linspace(-0.5, 0.5, 10)
        #         width = 0.01
        #         if x_y_or_z == 2:
        #             # do four vertical faces; each rectangle is long in z
        #             # direction
        #             for offset in offsets:
        #                 plot_cube_at(ax, (space_coords[0] + offset, space_coords[1]+0.8+width/2, avg_z), size=(width, width, extent), color=color)
        #                 plot_cube_at(ax, (space_coords[0] + offset, space_coords[1]-0.8+width/2, avg_z), size=(width, width, extent), color=color)
        #                 plot_cube_at(ax, (space_coords[0] + 0.8+width/2, space_coords[1] + offset, avg_z), size=(width, width, extent), color=color)
        #                 plot_cube_at(ax, (space_coords[0] - 0.8+width/2, space_coords[1] + offset, avg_z), size=(width, width, extent), color=color)
from swiper2.device_manager import DeviceData
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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
    ):
    # TODO: make vertical pipes disconnected if the patches are routing space,
    # or if there is a discard operation in between.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rows = max([r for r,c in data.all_patch_coords]) - min([r for r,c in data.all_patch_coords]) + 1
    cols = max([c for r,c in data.all_patch_coords]) - min([c for r,c in data.all_patch_coords]) + 1

    x,y,z = np.meshgrid(np.cumsum([0]+[data.d, spacing]*cols), np.cumsum([0]+[data.d, spacing]*rows), np.arange((spacing+1)*data.num_rounds+2))

    volume = np.zeros((2*rows, 2*cols, (spacing+1)*data.num_rounds+1))
    colors = np.empty_like(volume, dtype=object)
    edgecolors = np.empty_like(volume, dtype=object)

    z_offset = 0
    increased_z = False
    for round_idx, round_data in enumerate(data.generated_syndrome_data):

        # if there is a discard operation in the previous round, bump up z coord
        # by 1 to avoid connecting the patches
        # if do_z_offset and round_idx > 0 and round_idx in
        # data.patches_initialized_by_round and
        # len(data.patches_initialized_by_round[round_idx]) > 0:
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
            color = color_dict[name]
            
            if name == 'MERGE':
                # fill extra space between patches of same instruction
                for j, syndrome2 in enumerate(round_data):
                    if syndrome2.instruction == syndrome.instruction:
                        if syndrome2.patch[0] == coords[0] and abs(syndrome2.patch[1] - coords[1]) == 1:
                            # same x coordinate, adjacent y coordinates
                            y_fill = min(syndrome2.patch[1], coords[1])
                            volume[coords[0]*2, max(0, 2*y_fill+1), coords[2]+z_offset] = 1
                            colors[coords[0]*2, max(0, 2*y_fill+1), coords[2]+z_offset] = color
                            edgecolors[coords[0]*2, max(0, 2*y_fill+1), coords[2]+z_offset] = edgecolor_dict[name]
                        elif syndrome2.patch[1] == coords[1] and abs(syndrome2.patch[0] - coords[0]) == 1:
                            # same y coordinate, adjacent x coordinates
                            x_fill = min(syndrome2.patch[0], coords[0])
                            volume[max(0, 2*x_fill+1), coords[1]*2, coords[2]+z_offset] = 1
                            colors[max(0, 2*x_fill+1), coords[1]*2, coords[2]+z_offset] = color
                            edgecolors[max(0, 2*x_fill+1), coords[1]*2, coords[2]+z_offset] = edgecolor_dict[name]
                            
            volume[coords[0]*2, coords[1]*2, coords[2]+z_offset] = 1
            colors[coords[0]*2, coords[1]*2, coords[2]+z_offset] = color
            edgecolors[coords[0]*2, coords[1]*2, coords[2]+z_offset] = edgecolor_dict[name]

            # if we increased z_offset this round, connect any pre-existing
            # patches across the z_offset
            if increased_z:
                if not syndrome.initialized_patch:
                    volume[coords[0]*2, coords[1]*2, coords[2]+z_offset-spacing:coords[2]+z_offset] = 1
                    colors[coords[0]*2, coords[1]*2, coords[2]+z_offset-spacing:coords[2]+z_offset] = color
                    edgecolors[coords[0]*2, coords[1]*2, coords[2]+z_offset-spacing:coords[2]+z_offset] = edgecolor_dict[name]
    ax.voxels(x,y,z, filled=volume, facecolors=colors, edgecolors=edgecolors, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax.set_aspect('equal')
    ax.view_init(elev=15, azim=30)
    ax.set_xticks([])
    ax.set_yticks([])
    # flip both x and y axes to match the orientation of the device
    # ax.invert_xaxis()
    # ax.invert_yaxis()

    return ax
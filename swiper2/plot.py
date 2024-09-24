from swiper2.device_manager import DeviceData
import matplotlib.pyplot as plt
import numpy as np

def plot_device_schedule_trace(
        data: DeviceData,
        block_width: int = 7,
        spacing: int = 1,
        color_dict: dict[str, str] = {
            'MERGE': 'gray',
            'INJECT_T': 'green',
            'CONDITIONAL_S': 'gold',
            'IDLE': 'white',
            'UNWANTED_IDLE': 'black'
        }
    ):
    # TODO: redo passing x,y,z arrays to voxel (makes plotting complexity independent of d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rows = max([r for r,c in data.all_patch_coords]) - min([r for r,c in data.all_patch_coords]) + 1
    cols = max([c for r,c in data.all_patch_coords]) - min([c for r,c in data.all_patch_coords]) + 1

    volume = np.zeros((rows*(block_width+spacing), cols*(block_width+spacing), data.num_rounds))
    colors = np.empty_like(volume, dtype=object)

    for round_idx, round_data in enumerate(data.generated_syndrome_data):
        for i, syndrome in enumerate(round_data):
            coords = syndrome.patch + (round_idx,)
            if syndrome.is_unwanted_idle:
                name = 'UNWANTED_IDLE'
            else:
                name = data.instructions[syndrome.instruction_idx].name
            color = color_dict[name]
            
            if name == 'MERGE':
                # fill extra space between patches of same instruction
                for j, syndrome2 in enumerate(round_data):
                    if syndrome2.instruction_idx == syndrome.instruction_idx:
                        if syndrome2.patch[0] == coords[0] and abs(syndrome2.patch[1] - coords[1]) == 1:
                            # same x coordinate, adjacent y coordinates
                            y_fill = max(syndrome2.patch[1], coords[1])
                            volume[coords[0]*(block_width+spacing):coords[0]*(block_width+spacing)+block_width, max(0, y_fill*(block_width+spacing)-spacing):y_fill*(block_width+spacing), coords[2]] = 1
                            colors[coords[0]*(block_width+spacing):coords[0]*(block_width+spacing)+block_width, max(0, y_fill*(block_width+spacing)-spacing):y_fill*(block_width+spacing), coords[2]] = color
                        elif syndrome2.patch[1] == coords[1] and abs(syndrome2.patch[0] - coords[0]) == 1:
                            # same y coordinate, adjacent x coordinates
                            x_fill = max(syndrome2.patch[0], coords[0])
                            volume[max(0, x_fill*(block_width+spacing)-spacing):x_fill*(block_width+spacing), coords[1]*(block_width+spacing):coords[1]*(block_width+spacing)+block_width, coords[2]] = 1
                            colors[max(0, x_fill*(block_width+spacing)-spacing):x_fill*(block_width+spacing), coords[1]*(block_width+spacing):coords[1]*(block_width+spacing)+block_width, coords[2]] = color

            volume[coords[0]*(block_width+spacing):coords[0]*(block_width+spacing)+block_width, coords[1]*(block_width+spacing):coords[1]*(block_width+spacing)+block_width, coords[2]] = 1
            colors[coords[0]*(block_width+spacing):coords[0]*(block_width+spacing)+block_width, coords[1]*(block_width+spacing):coords[1]*(block_width+spacing)+block_width, coords[2]] = color    

    ax.voxels(volume, facecolors=colors)
    ax.set_aspect('equal')

    return ax
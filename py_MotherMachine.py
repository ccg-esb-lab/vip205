import matplotlib.pyplot as plt
import numpy as np
import imageio
import imageio.v2 as imageio  # Importing ImageIO v2 to avoid deprecation warning
import pickle
import os
import matplotlib.cm as cm
import random

##########################################


pathCFP = 'montagesCFP/'
pathTIF = 'montagesTIF/'
pathPKL = 'MontagesPKL/'

# List of montage paths
montage_files = [f for f in os.listdir(pathTIF) if f.endswith('.tif')]
montage_paths = [
    (os.path.join(pathTIF, f), os.path.join(pathCFP, f.replace('_DIC', '_CFP')))
    for f in montage_files
]
print(montage_paths)

random.shuffle(montage_paths)
current_montage_index = 0
use_cfp = False

# Global variables to store clicked positions and breakpoints
clicked_positions = []
y_breakpoints = []
x_breakpoints = []
track_id = 0
from_frame = 0
to_frame = 100

y_pos1 = 25
part_height = 10

channel_width = 11
y_size = 400

max_frames = 100
frames = range(0, max_frames + 1)
debug = False

clicked_positions = []


showing_all=False

cmap = plt.colormaps.get_cmap('tab10')

# Global variables to store breakpoints for all cells
all_breakpoints = []

is_dragging = False
drag_start = None


##########################################
def on_click(event):
    global is_dragging, drag_start
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        drag_start = (x, y)
        is_dragging = True
        print(f"Mouse clicked at ({x}, {y})")

def on_motion(event):
    global is_dragging, drag_start
    if is_dragging and event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Mouse dragging at ({x}, {y})")

def on_release(event):
    global is_dragging, drag_start, track_id, x_breakpoints, y_breakpoints
    if is_dragging:
        is_dragging = False
        
        if event.xdata is not None and event.ydata is not None:
            x_end, y_end = int(event.xdata), int(event.ydata)
        else:
            # Use the last frame as the endpoint if released outside the plot area
            x_end = int(ax.get_xlim()[1])
            y_end = int(ax.get_ylim()[0])
        
        x_start, y_start = drag_start
        drag_start = None
        print(f"Mouse released at ({x_end}, {y_end})")
        
        if abs(x_end - x_start) < 5 and abs(y_end - y_start) < 5:
            # This is considered a click, not a drag
            handle_click(x_start, y_start, event)
        else:
            # This is a drag
            handle_drag(x_start, y_start, x_end, y_end, event)


def handle_click(x, y, event):
    global showing_all, track_id, x_breakpoints, y_breakpoints
    this_frame = x // channel_width
    print(f"Handling click at ({x}, {y}): {this_frame}")

    clicked_positions.append((x, y))
    # Update the plot with the clicked points
    ax.plot(x, y, 'ro')
    fig.canvas.draw()
    
    # Estimate the frame from the x position
    frame = x // channel_width

    if 'shift' in event.modifiers:
        print("Inserting new breakpoints")

        # Ensure unique track_id
        new_track_id = track_id + 1
        while any(track['id'] == new_track_id for track in all_breakpoints):
            new_track_id += 1

        track_id = new_track_id

        # Insert new breakpoints
        y_breakpoints, x_breakpoints = track_positions(y, frame, frames, montage, channel_width, y_size, part_height, debug)

        print("Saving breakpoints for new cell")
        all_breakpoints.append({
            'id': track_id,
            'x_breakpoints': x_breakpoints.copy(),
            'y_breakpoints': y_breakpoints.copy()
        })
        showing_all = True    
        plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif 'control' in event.modifiers or 'ctrl' in event.modifiers:
        print("Trimming breakpoints after frame %s" % this_frame)
        x_breakpoints_filtered = []
        y_breakpoints_filtered = []
        for xb, yb in zip(x_breakpoints, y_breakpoints):
            if xb <= x:
                x_breakpoints_filtered.append(xb)
                y_breakpoints_filtered.append(yb)
        x_breakpoints[:] = x_breakpoints_filtered
        y_breakpoints[:] = y_breakpoints_filtered

        # Update the current track
        for track in all_breakpoints:
            if track['id'] == track_id:
                track['x_breakpoints'] = x_breakpoints.copy()
                track['y_breakpoints'] = y_breakpoints.copy()
                break

        plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif 'alt' in event.modifiers:
        print("Correcting only this breakpoint")
        # Update the specific breakpoint
        if this_frame * channel_width in x_breakpoints:
            index = x_breakpoints.index(this_frame * channel_width)
            y_breakpoints[index] = y
        else:
            x_breakpoints.append(this_frame * channel_width)
            y_breakpoints.append(y)

        # Update the current track
        for track in all_breakpoints:
            if track['id'] == track_id:
                track['x_breakpoints'] = x_breakpoints.copy()
                track['y_breakpoints'] = y_breakpoints.copy()
                break

        plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    else:
        print("Correcting breakpoints")
        showing_all = False

        # Corrector: Erase breakpoints from this frame onwards if it is the second click
        if len(clicked_positions) > 0:
            correct_from_frame = frames[this_frame]
            print("Track from %s" % correct_from_frame)
            x_breakpoints_filtered = []
            y_breakpoints_filtered = []
            print(">>%s" % track_id, x_breakpoints)
            for xb, yb in zip(x_breakpoints, y_breakpoints):
                if xb < correct_from_frame * channel_width:
                    x_breakpoints_filtered.append(xb)
                    y_breakpoints_filtered.append(yb)
            x_breakpoints[:] = x_breakpoints_filtered
            y_breakpoints[:] = y_breakpoints_filtered

        # Track positions based on the new clicked position
        new_y_breakpoints, new_x_breakpoints = track_positions(y, frame, frames, montage, channel_width, y_size, part_height, debug)

        # Update the global breakpoints list
        y_breakpoints.extend(new_y_breakpoints)
        x_breakpoints.extend(new_x_breakpoints)

        # Update the current track
        for track in all_breakpoints:
            if track['id'] == track_id:
                track['x_breakpoints'] = x_breakpoints.copy()
                track['y_breakpoints'] = y_breakpoints.copy()
                break
        showing_all = True
        plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)


def handle_drag(x_start, y_start, x_end, y_end, event):
    global track_id, x_breakpoints, y_breakpoints
    print(f"Handling drag from ({x_start}, {y_start}) to ({x_end}, {y_end})")

    # Assume this is a horizontal drag to set breakpoints for each frame
    x_values = np.linspace(x_start, x_end, num=(x_end - x_start) // channel_width + 1)
    y_values = np.linspace(y_start, y_end, num=(x_end - x_start) // channel_width + 1)
    
    if len(clicked_positions) > 0:
        correct_from_frame = int(x_start) // channel_width
        print("Track from %s" % correct_from_frame)
        x_breakpoints_filtered = []
        y_breakpoints_filtered = []
        print(">>%s" % track_id, x_breakpoints)
        for xb, yb in zip(x_breakpoints, y_breakpoints):
            if xb < correct_from_frame * channel_width:
                x_breakpoints_filtered.append(xb)
                y_breakpoints_filtered.append(yb)
        x_breakpoints[:] = x_breakpoints_filtered
        y_breakpoints[:] = y_breakpoints_filtered

    # Replace with new breakpoints based on the drag
    for x, y in zip(x_values, y_values):
        frame = int(x) // channel_width
        x_breakpoints.append(frame * channel_width)
        y_breakpoints.append(int(y))

    # Update the current track
    for track in all_breakpoints:
        if track['id'] == track_id:
            track['x_breakpoints'] = x_breakpoints.copy()
            track['y_breakpoints'] = y_breakpoints.copy()
            break

    plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)


def on_key(event):
    global track_id, from_frame, to_frame, y_breakpoints, x_breakpoints, all_breakpoints, current_montage_index, showing_all, use_cfp
    print(f"Key pressed: {event.key}")

    if event.key == 'shift':
        print("Shift key pressed")
    elif event.key == 'enter' and 'shift' not in event.modifiers and 'alt' not in event.modifiers:
        print("Updating breakpoints for %s" % track_id)
        # Update the current track
        for track in all_breakpoints:
            if track['id'] == track_id:
                track['x_breakpoints'] = x_breakpoints.copy()
                track['y_breakpoints'] = y_breakpoints.copy()
                break
        showing_all = True
        plot_all_breakpoints(from_frame, to_frame)
    elif event.key == 'enter' and 'shift' in event.modifiers:
        print("Saving breakpoints for new cell")
        all_breakpoints.append({
            'id': track_id,
            'x_breakpoints': x_breakpoints.copy(),
            'y_breakpoints': y_breakpoints.copy()
        })
        # Reset breakpoints for the next cell
        y_breakpoints = []
        x_breakpoints = []
        showing_all = True
        plot_all_breakpoints(from_frame, to_frame)
    elif event.key == '+':
        print("Zooming in")
        from_frame = max(from_frame - 50, 0)
        to_frame = min(to_frame + 50, max_frames)
        if showing_all:
            plot_all_breakpoints(from_frame, to_frame)
        else:
            plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif event.key == '-':
        print("Zooming out")
        if showing_all:
            plot_all_breakpoints(from_frame=None, to_frame=None)
        else:
            plot_breakpoints(x_breakpoints, y_breakpoints, from_frame=None, to_frame=None)
    elif event.key == 'left':
        print("Moving left")
        from_frame = max(from_frame - 50, 0)
        to_frame = min(from_frame + 50, max_frames)
        if showing_all:
            plot_all_breakpoints(from_frame, to_frame)
        else:
            plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif event.key == 'right':
        print("Moving right")
        to_frame = min(to_frame + 50, max_frames)
        from_frame = max(to_frame - 50, 0)
        if showing_all:
            plot_all_breakpoints(from_frame, to_frame)
        else:
            plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif event.key == 'escape':
        save_all_breakpoints()
        print("Saving all breakpoints to file")
        exit()
    elif event.key == ' ':
        print("Plotting all breakpoints")
        if showing_all:
            plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
            showing_all = False
        else:
            plot_all_breakpoints(from_frame, to_frame)
            showing_all = True

    elif event.key == 'c':

        use_cfp = not use_cfp  # Toggle the flag
        print("Switching channel use_cfp=%s"%use_cfp)
        if showing_all:
            plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
        else:
            plot_all_breakpoints(from_frame, to_frame)
    elif event.key == '<':
        print("**%s" % current_montage_index)
        if current_montage_index > 0:
            print("Switching to previous montage")
            save_all_breakpoints()
            current_montage_index -= 1
            load_next_montage()
            showing_all = True
            plot_all_breakpoints(from_frame=None, to_frame=None)
    elif event.key == '>':
        if current_montage_index < len(montage_paths) - 1:
            print("Switching to next montage")
            save_all_breakpoints()
            current_montage_index += 1
            load_next_montage()
            showing_all = True
            plot_all_breakpoints(from_frame=None, to_frame=None)
    elif event.key == 'up':
        if len(all_breakpoints) > 0:
            current_indices = [i for i, track in enumerate(all_breakpoints) if track['id'] == track_id]
            print(">%s" % (current_indices), track_id)
            if current_indices:
                current_index = current_indices[0]
                new_index = (current_index - 1 + len(all_breakpoints)) % len(all_breakpoints)  # Adjust for negative index
                track_id = all_breakpoints[new_index]['id']
                load_active_track()
                plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
            else:
                track_id = all_breakpoints[0]['id']
                load_active_track()
                plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif event.key == 'down':
        if len(all_breakpoints) > 0:
            current_indices = [i for i, track in enumerate(all_breakpoints) if track['id'] == track_id]
            if current_indices:
                current_index = current_indices[0]
                new_index = (current_index + 1) % len(all_breakpoints)
                track_id = all_breakpoints[new_index]['id']
                load_active_track()
                plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
            else:
                track_id = all_breakpoints[0]['id']
                load_active_track()
                plot_breakpoints(x_breakpoints, y_breakpoints, from_frame, to_frame)
    elif event.key == 'd':
        if len(all_breakpoints) > 0:
            print(all_breakpoints)
            print(f"Deleting track {track_id}")
            all_breakpoints = [track for track in all_breakpoints if track['id'] != track_id]
            if len(all_breakpoints) > 0:
                track_id = all_breakpoints[0]['id']
                print(f"New track_id={track_id}")
                load_active_track()
                plot_all_breakpoints(from_frame, to_frame)
            else:
                y_breakpoints = []
                x_breakpoints = []
                plot_all_breakpoints(from_frame, to_frame)


def load_next_montage():
    global current_montage_index, montage, clicked_positions, y_breakpoints, x_breakpoints, track_id, from_frame, to_frame, all_breakpoints, fig, ax, use_cfp
    
    if 0 <= current_montage_index < len(montage_paths):
        #montage_path = montage_paths[current_montage_index]
        #montage = imageio.imread(montage_path)
        
        tif_path, cfp_path = montage_paths[current_montage_index]
        montage_path = cfp_path if use_cfp else tif_path
        montage = imageio.imread(montage_path)

        # Reset variables
        clicked_positions = []
        y_breakpoints = []
        x_breakpoints = []
        track_id = 0
        from_frame = 0
        to_frame = 100
        all_breakpoints = []

        # Extract base filename and construct PKL filename
        base_filename = os.path.splitext(os.path.basename(montage_path))[0]
        pkl_filename = os.path.join(pathPKL, base_filename + ".pkl")

        # Load the PKL file if it exists
        if os.path.exists(pkl_filename):
            with open(pkl_filename, 'rb') as f:
                all_breakpoints = pickle.load(f)
            print(f"Loaded breakpoints from {pkl_filename}")

        # Set the initial active track if breakpoints are loaded
        if all_breakpoints:
            load_active_track()

        # Close the previous figure if it exists
        try:
            plt.close(fig)
        except NameError:
            pass

        # Plot the new montage
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.imshow(np.flipud(montage), cmap='gray')

        plot_all_breakpoints(from_frame, to_frame)

        # Connect the key press event to the handler
        fig.canvas.mpl_connect('key_press_event', on_key)
        #fig.canvas.mpl_connect('button_press_event', on_click)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        plt.show()

        print(f"Loaded montage: {montage_path}")
    else:
        print("No more montages to load.")

def save_all_breakpoints():
    """
    Save all breakpoints for all cells to a PKL file.
    """
    
    tif_path, cfp_path = montage_paths[current_montage_index]
    montage_path = cfp_path if use_cfp else tif_path

    base_filename = os.path.splitext(os.path.basename(montage_path))[0]

    pkl_filename = os.path.join(pathPKL, base_filename + ".pkl")
    with open(pkl_filename, 'wb') as f:
        pickle.dump(all_breakpoints, f)
    print(f"All breakpoints saved to {pkl_filename}")

def load_active_track():
    global x_breakpoints, y_breakpoints, all_breakpoints, track_id
    for track in all_breakpoints:
        if track['id'] == track_id:
            x_breakpoints = track['x_breakpoints']
            y_breakpoints = track['y_breakpoints']
            break



def get_channel_profile(channel_image):
    """
    Computes the channel profile of a given channel image.

    Parameters:
    channel_image (ndarray): The image of the specific channel.

    Returns:
    tuple: Two numpy arrays containing the y positions and the corresponding pixel intensities.
    """
    # Calculate the mean intensity for each row (y position)
    y_positions = np.arange(channel_image.shape[0])
    pixel_intensities = np.mean(channel_image, axis=1)

    return y_positions, pixel_intensities


def plot_channel_profile(channel_image, y_positions, pixel_intensities):
    """
    Plots the channel image and its y-profile side by side with shared axes.

    Parameters:
    channel_image (ndarray): The image of the specific channel.
    y_positions (ndarray): The y positions.
    pixel_intensities (ndarray): The corresponding pixel intensities.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})

    # Plot the channel image
    ax1.imshow(channel_image, cmap='gray')
    ax1.set_xticks([])
    ax1.set_ylabel('Y')
    ax1.axis('on')

    # Plot the channel profile
    ax2.plot(pixel_intensities, y_positions, marker='.',color='k')
    ax2.set_xlabel('Pixel Intensity')


    ax2.set_ylabel('Y')
    ax2.invert_yaxis()  # Invert y-axis to match the image orientation
    ax2.grid(True)

    # Set the y-axis limits to be the same for both subplots
    y_lim = ax1.get_ylim()
    ax2.set_ylim(y_lim)

    plt.title(channel_image)
    plt.tight_layout()
    plt.show()


def get_channel_image(montage, channel_width, frame):
    """
    Extracts and returns the image of a specific channel from the montage,
    with the y-axis inverted.

    Parameters:
    montage (ndarray): The montage image.
    channel_width (int): The width of each channel in the montage.
    frame (int): The frame number to extract.

    Returns:
    ndarray: The image of the specified channel with y-axis inverted.
    """
    # Calculate the start and end x positions for the channel
    start_x = frame * channel_width
    end_x = (frame + 1) * channel_width

    # Debug print statements to check the start and end x positions
    #print(f"Frame: {frame}, Start_x: {start_x}, End_x: {end_x}")

    # Extract the channel image and invert the y-axis
    channel_image = montage[:, start_x:end_x]
    channel_image = np.flipud(channel_image)

    return channel_image

def get_channel_window(channel_image, y_pos, y_size=10):
    """
    Extracts a window centered around a reference point from the given channel image.

    Parameters:
    channel_image (ndarray): The image of the specific channel.
    y_pos (int): The center y position of the window.
    y_size (int): The height of the window. Default is 20.

    Returns:
    ndarray: The extracted window of the channel image.
    """
    # Calculate the start and end y positions
    start_y = int(y_pos - y_size / 2)
    end_y = int(y_pos + y_size / 2)

    # Ensure the window is within the bounds of the image
    if start_y < 0:
        start_y = 0
    if end_y > channel_image.shape[0]:
        end_y = channel_image.shape[0]


    # Extract the window from the channel image
    channel_window = channel_image[start_y:end_y, :]

    return channel_window


def calculate_window_difference(window1, window2, debug=False):
    """
    Calculates the fraction of pixels where the difference exceeds a threshold between two windows.
    
    Parameters:
    window1 (ndarray): The first window.
    window2 (ndarray): The second window.
    debug (bool): If True, displays the windows being compared along with the difference colormap.
    
    Returns:
    float: The fraction of pixels above the threshold.
    """
    threshold = 100
    difference = (window1 - window2) ** 2
    count_above_threshold = np.sum(difference > threshold)
    total_pixels = np.size(difference)
    
    if total_pixels == 0:
        fraction_above_threshold = 0
    else:
        fraction_above_threshold = count_above_threshold / total_pixels
    
    if debug:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(window1, cmap='gray')
        axs[0].set_title('Window 1')
        axs[1].imshow(window2, cmap='gray')
        axs[1].set_title('Window 2')
        cax = axs[2].imshow(difference, cmap='cividis', interpolation='nearest')
        axs[2].set_title('Difference')
        fig.colorbar(cax, ax=axs[2])
        plt.show()
        
    return fraction_above_threshold


def find_optimal_displacement(channel_window1, y_pos1, channel_window2, part_height, debug=False):
    """
    Finds the optimal y-displacement that minimizes the difference between a small window from channel_window1
    and the rows in channel_window2.

    Parameters:
    channel_window1 (ndarray): The first channel window.
    y_pos1 (int): The y position of the center of the window in channel_window1 to compare.
    channel_window2 (ndarray): The second channel window.
    part_height (int): The height of the small window for comparison.

    Returns:
    int: The optimal y-displacement.
    float: The minimum difference.
    """

    # Extract the small part from channel_window1
    start_y1 = y_pos1 - part_height // 2
    part_window1 = channel_window1[start_y1:start_y1 + part_height, :]

    # Initialize variables for the optimal displacement
    min_difference = float('inf')
    optimal_displacement = 0

    # Calculate the differences for each possible displacement
    for displacement in range(y_pos1 - part_height*2, y_pos1 + part_height*2+1):
        if debug:
          print("displacement=%s"%displacement)
        if displacement < 0:
            part_window2_slice = channel_window2[:part_height + displacement, :]
            part_window1_slice = part_window1[-displacement:, :]
        else:
            part_window2_slice = channel_window2[displacement:displacement + part_height, :]
            part_window1_slice = part_window1[:part_height, :]

        # Ensure both parts have the same shape
        if part_window1.shape[0] != part_window2_slice.shape[0]:
            continue

        difference = calculate_window_difference(part_window1, part_window2_slice, debug)
        #print(y_pos1, displacement, difference, min_difference)


        if difference < min_difference:
            #if debug:
              #print("**")
            min_difference = difference
            optimal_displacement = y_pos1-(displacement + part_height // 2)
        #print(displacement, difference, optimal_displacement)
    return optimal_displacement, min_difference

def plot_comparison(channel_window1, y_pos1, channel_window2, optimal_displacement, part_height):
    """
    Plots the small window of pixels from channel_window1 and the corresponding window in channel_window2 after shifting.

    Parameters:
    channel_window1 (ndarray): The first channel window.
    y_pos1 (int): The y position of the center of the window in channel_window1.
    channel_window2 (ndarray): The second channel window.
    optimal_displacement (int): The optimal y-displacement for alignment.
    part_height (int): The height of the small window for comparison.
    """
    # Extract the specified small window from channel_window1
    start_y1 = y_pos1 - part_height // 2
    part_window1 = channel_window1[start_y1:start_y1 + part_height, :]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot channel_window1 with a red box indicating the selected part
    axs[0].imshow(channel_window1, cmap='gray')
    axs[0].add_patch(plt.Rectangle((-0.5, start_y1-0.5), channel_window1.shape[1], part_height,
                                   edgecolor='red', facecolor='none', linewidth=2))
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')

    # Plot channel_window2 with a yellow box at the initial position and a solid red box at the optimal displacement
    initial_y_pos2 = y_pos1 - part_height // 2
    axs[1].imshow(channel_window2, cmap='gray')
    axs[1].add_patch(plt.Rectangle((-0.5, initial_y_pos2-0.5), channel_window2.shape[1], part_height,
                                   edgecolor='red', linestyle='-', facecolor='none', linewidth=2))
    axs[1].add_patch(plt.Rectangle((-0.5, initial_y_pos2 + optimal_displacement-0.5), channel_window2.shape[1], part_height,
                                   edgecolor='red', linestyle='--',facecolor='none', linewidth=2))
    axs[1].set_xlabel('X Position')
    axs[1].set_ylabel('Y Position')

    plt.tight_layout()
    plt.show()


        


def track_positions(y_pos1, start_frame, frames, montage, channel_width, y_size, part_height, debug):
    y_positions = [y_pos1]
    x_positions = [start_frame * channel_width]
    
    for iframe in range(start_frame, len(frames) - 1):
        frame1 = frames[iframe]
        frame2 = frames[iframe + 1]
        channel_image1 = get_channel_image(montage, channel_width, frame1)
        channel_window1 = get_channel_window(channel_image1, y_pos1, y_size)

        channel_image2 = get_channel_image(montage, channel_width, frame2)
        channel_window2 = get_channel_window(channel_image2, y_pos1, y_size)  # Ensure same window size

        optimal_displacement, min_difference = find_optimal_displacement(channel_window1, y_pos1, channel_window2, part_height, debug)

        if debug:
            print(f"Optimal y-displacement: {optimal_displacement}")
            print(f"Minimum difference: {min_difference}")
            print(f"frames: [{frame1, frame2}]")
            print(f"y_pos:{y_pos1}")

        y_pos1 -= optimal_displacement
        if y_pos1<0:
            y_pos1=0 
        y_positions.append(y_pos1)  # Store the y position
        x_positions.append((iframe + 1) * channel_width)

        if debug:
            plot_comparison(channel_window1, y_pos1, channel_window2, optimal_displacement, part_height)
    
    return y_positions, x_positions

def plot_breakpoints(x_breakpoints, y_breakpoints, from_frame=None, to_frame=None):

    global track_id, cmap, use_cfp


    # Extract the filename from the current montage path
    tif_path, cfp_path = montage_paths[current_montage_index]
    print("tif:",tif_path)
    montage_path = cfp_path if use_cfp else tif_path
    montage = imageio.imread(montage_path)
    filename = os.path.basename(montage_path)
    print(use_cfp," ",filename)


    # Clear the current plot
    ax.clear()
    
    # Re-plot the montage
    ax.imshow(np.flipud(montage), cmap='gray')

    # Define a colormap
    color = cmap(int(track_id))  # Normalize the track_id to get a color
    # Plot the updated breakpoints with colors based on track_id
    for i, (x, y) in enumerate(zip(x_breakpoints, y_breakpoints)):
        
        ax.hlines(y, x, x + channel_width, colors=color, linestyles='-', linewidth=2)
    
    # Set x-axis limits if specified
    if from_frame is not None and to_frame is not None:
        ax.set_xlim(from_frame * channel_width, (to_frame + 1) * channel_width)
    

    # Set the title with the filename
    ax.set_title(f"Montage: {filename}")

    fig.canvas.draw()

def plot_all_breakpoints(from_frame=None, to_frame=None):
    global cmap, use_cfp

    # Clear the current plot
    ax.clear()

    # Extract the filename from the current montage path
    tif_path, cfp_path = montage_paths[current_montage_index]
    print("tif:",tif_path)
    montage_path = cfp_path if use_cfp else tif_path
    montage = imageio.imread(montage_path)
    filename = os.path.basename(montage_path)
    print(use_cfp," ",filename)
    
    # Re-plot the montage
    ax.imshow(np.flipud(montage), cmap='gray')
    
    # Plot the breakpoints for all tracks
    for i, track in enumerate(all_breakpoints):
        itrack=int(track['id'])
        x_breakpoints = track['x_breakpoints']
        y_breakpoints = track['y_breakpoints']
        color = cmap(itrack)  # Normalize to get a color
        for x, y in zip(x_breakpoints, y_breakpoints):
            ax.hlines(y, x, x + channel_width, colors=color, linestyles='-', linewidth=2)
    
    # Set x-axis limits if specified
    if from_frame is not None and to_frame is not None:
        ax.set_xlim(from_frame * channel_width, (to_frame + 1) * channel_width)
    
    # Set the title with the filename
    ax.set_title(f"Montage: {filename}")

    fig.canvas.draw()




##########################################


# Initialize the first montage
load_next_montage()
plot_all_breakpoints(from_frame, to_frame)


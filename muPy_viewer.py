import os
import matplotlib.pyplot as plt
from code.io_utils import ImageLoader, save_segmentation_outputs
from code.event_handler import EventHandler
import numpy as np
import scipy.ndimage as ndi

# Define experiment ID and path to TIF directory
experiment_id = "grx_xy18"
pathTIF = "./data/TIF/"

# Load images using the new ImageLoader
image_loader = ImageLoader(pathTIF)
image_loader.load_images(experiment_id)

debug=False

# Extract loaded images and channels
images = image_loader.images
default_channel = image_loader.default_channel 

# Initialize Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figsize for larger display
ax.set_xlim([0, 500])  # Adjust these values for zoom effect
ax.set_ylim([500, 0])  # Flip y-axis to match image orientation

# Ensure the first display is initialized correctly
current_time_index = 0  # Start from first timepoint
current_channel = default_channel if default_channel else list(images.keys())[0]  # Fallback if no DIC

# **Initialize EventHandler First!**
event_handler = EventHandler(
    fig, 
    ax, 
    image_loader.images, 
    None,  # Placeholder, update below
    list(image_loader.images.keys()),
    image_loader.path,          # Store path
    image_loader.default_channel,  # Store default channel
    experiment_id,               # Store experiment ID
    debug
)

# **Now Define `update_display` and Pass `event_handler`**
def update_display(time_index, channel=None, show_mask=True, show_cells=False, current_cmap='nipy_spectral', shuffled_mask=None):
    """Updates the displayed image for the given time index and channel, preserving zoom."""
    
    ax.clear()

    # Ensure channel is valid
    if channel is None or channel not in images:
        if len(images) == 0:
            if debug:
                print("[ERROR] No images available.")
            return
        channel = sorted(images.keys())[0]  # Default to the first available channel

    img_stack = images.get(channel, None)
    if img_stack is None:
        if debug:
            print(f"Warning: Channel '{channel}' not found. Defaulting to first channel.")
        img_stack = list(images.values())[0]

    if isinstance(img_stack, tuple):
        img_stack = img_stack[0]  # Extract the NumPy array if it's a tuple

    # Select the correct time index
    if isinstance(img_stack, np.ndarray):
        max_time = img_stack.shape[0] - 1 if img_stack.ndim >= 3 else 0
        time_index = min(time_index, max_time)
        img = img_stack[time_index, :, :] if img_stack.ndim >= 3 else img_stack
    else:
        if debug:
            print(f"[ERROR] Invalid data type for img_stack: {type(img_stack)}")
        return

    # Normalize intensities
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * 255
        img = img.astype(np.uint8)

    ax.imshow(img, cmap=None if img.ndim == 3 else 'gray')

    # Show MASK
    if show_mask and 'MASK' in images:
        mask_layer = shuffled_mask if shuffled_mask is not None else images['MASK']
        mask_layer = mask_layer[min(time_index, mask_layer.shape[0] - 1), :, :] if mask_layer.ndim >= 3 else mask_layer
        masked_layer = np.ma.masked_where(mask_layer == 0, mask_layer)
        ax.imshow(masked_layer, alpha=0.4, cmap=current_cmap)

    # Extract the current MASK frame
    if show_mask and 'MASK' in images:
        mask_layer = shuffled_mask if shuffled_mask is not None else images['MASK']
        mask = mask_layer[min(time_index, mask_layer.shape[0] - 1), :, :] if mask_layer.ndim >= 3 else mask_layer

        # If centers are enabled, overlay them
        if event_handler.show_centers:
            centers = get_cell_centers(mask)

            # Normalize mask values to match colormap
            norm = plt.Normalize(vmin=mask.min(), vmax=mask.max())
            cmap = plt.get_cmap(current_cmap)

            for cell_id, (y, x) in centers.items():
                if cell_id > 0:  # Exclude background (0)
                    color = cmap(norm(cell_id))  # Get colormap color for the cell
                    ax.scatter(x, y, color=color, s=20, edgecolors='black', linewidth=0.5)




    # If CELLS is enabled, also overlay MASK in white at alpha=0.1
    if show_cells and 'MASK' in images:
        mask_layer = images['MASK']
        if isinstance(mask_layer, np.ndarray):
            if mask_layer.ndim >= 3:
                mask_layer = mask_layer[min(time_index, mask_layer.shape[0] - 1), :, :]

            # Convert MASK to binary: Set all non-zero pixels to 1.0 for display
            binary_mask = np.where(mask_layer > 0, 65535.0, 0.0)
            # Mask the background so it doesn't darken the underlying image.
            masked_binary = np.ma.masked_where(binary_mask == 0, binary_mask)
            ax.imshow(masked_binary, alpha=0.2, cmap='gray', vmin=0, vmax=65535.)
            print("****")


    # Overlay CELLS
    if show_cells and 'CELLS' in images:
        cells_layer = images['CELLS'][time_index, :, :] if images['CELLS'].ndim >= 3 else images['CELLS']
        masked_cells = np.ma.masked_where(cells_layer == 0, cells_layer)
        ax.imshow(masked_cells, alpha=0.8, cmap='cubehelix', vmin=0, vmax=65535)

    ax.set_title(f"Channel: {channel}, Time {time_index + 1}/{max_time + 1}")

    # **Apply Zoom Only If Already Set**
    if event_handler.current_xlim and event_handler.current_ylim:
        ax.set_xlim(event_handler.current_xlim)
        ax.set_ylim(event_handler.current_ylim)

    fig.canvas.draw_idle()
    plt.pause(0.001)


def get_cell_centers(mask):
    """
    Computes the center (centroid) of each segmented cell in the mask.

    Parameters:
    - mask (np.ndarray): A 2D array where each unique value (excluding 0) represents a cell.

    Returns:
    - dict: A dictionary where keys are cell labels and values are (y, x) coordinates of the cell center.
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

    centers = {label: ndi.center_of_mass(mask == label) for label in unique_labels}
    return centers

def plot_mask_with_centers(mask, ax):
    """
    Overlays cell centers on top of the MASK layer.

    Parameters:
    - mask (np.ndarray): The segmentation mask.
    - ax (matplotlib.axes.Axes): Matplotlib axis to draw the overlay.

    Returns:
    - None
    """
    centers = get_cell_centers(mask)

    for label, (y, x) in centers.items():
        ax.scatter(x, y, color='yellow', s=20, edgecolors='black', linewidth=0.5)

# *Now Link `update_display` to `event_handler`**
event_handler.update_display = update_display

if debug:
    print(f"[DEBUG] Displaying initial channel: {default_channel}")
update_display(current_time_index, default_channel, show_mask=False, show_cells=False)

# Show interactive plot
plt.show()

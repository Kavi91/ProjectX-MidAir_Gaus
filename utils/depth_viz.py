import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Define the path to your depth files
depth_dir = "/home/krkavinda/Datasets/KITTI_raw/kitti_data/scan/00/depth/"  # Adjust '00' to the sequence you want to visualize

# Find all .npy files in the depth directory
depth_files = glob.glob(os.path.join(depth_dir, "*.npy"))
depth_files.sort()  # Sort files for consistent ordering

# Check if files are found
if not depth_files:
    print(f"No .npy files found in {depth_dir}")
    exit()

# Visualize a few depth maps (e.g., first 5)
num_samples = min(5, len(depth_files))  # Visualize up to 5 depth maps

# Create a figure for visualization
fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))

# If only one sample, wrap axes in a list for consistent iteration
if num_samples == 1:
    axes = [axes]

# Load, visualize, and get resolution for each depth map
for i in range(num_samples):
    # Load the depth map
    depth_path = depth_files[i]
    depth_map = np.load(depth_path)
    
    # Get resolution (height, width)
    if len(depth_map.shape) == 2:  # Single-channel depth map (H, W)
        height, width = depth_map.shape
        print(f"Depth file {os.path.basename(depth_path)}: Resolution = ({height}, {width})")
    elif len(depth_map.shape) == 3:  # Depth map with channels (H, W, C)
        height, width, channels = depth_map.shape
        print(f"Depth file {os.path.basename(depth_path)}: Resolution = ({height}, {width}), Channels = {channels}")
        # If multi-channel, take the first channel for visualization
        depth_map = depth_map[:, :, 0]
    else:
        print(f"Unexpected shape for {depth_path}: {depth_map.shape}")
        continue
    
    # Visualize the depth map as a heatmap
    im = axes[i].imshow(depth_map, cmap='viridis')  # 'viridis' colormap for depth visualization
    axes[i].set_title(f"Depth Map: {os.path.basename(depth_path)}")
    axes[i].axis('off')  # Hide axes for cleaner visualization
    plt.colorbar(im, ax=axes[i], label='Depth Value')

# Adjust layout and display
plt.tight_layout()
plt.show()
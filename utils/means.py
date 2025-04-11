import os
import glob
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from params import par

# Define climate sets and trajectories (use training set only)
climate_sets = par.climate_sets  # ['Kite_training/cloudy', 'Kite_training/foggy', 'Kite_training/sunny', 'Kite_training/sunset']
sample_limit = 1000  # Limit the number of images to process for efficiency

# Initialize accumulators for RGB mean and std as PyTorch tensors
mean_03 = torch.zeros(3)  # For image_rgb (left)
mean_02 = torch.zeros(3)  # For image_rgb_right (right)
m2_03 = torch.zeros(3)    # For Welford's method (variance computation)
m2_02 = torch.zeros(3)

# Initialize accumulators for depth mean and std
depth_max = 100.0  # Same as in data_helper.py
depth_values = []  # Store depth values for mean and std computation
count_rgb = 0
count_depth = 0

# Transform to convert images to tensors
transform = transforms.ToTensor()

# Iterate over training climate sets and specified trajectories
for climate_set in climate_sets:
    # Use only the trajectories specified in par.train_traj_ids for this climate set
    traj_list = par.train_traj_ids.get(climate_set, [])
    print(f"Processing {climate_set} with {len(traj_list)} trajectories: {traj_list}")

    for traj in traj_list:
        # Load file paths
        fpaths_03 = glob.glob(f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/image_rgb/*.JPEG')
        fpaths_02 = glob.glob(f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/image_rgb_right/*.JPEG')
        fpaths_depth = glob.glob(f'/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/{traj}/depth/*.PNG')
        fpaths_03.sort()
        fpaths_02.sort()
        fpaths_depth.sort()

        # Process a subset of RGB images
        for i, (fpath_03, fpath_02) in enumerate(zip(fpaths_03, fpaths_02)):
            if count_rgb >= sample_limit:
                break

            # Load and transform images
            img_03 = transform(Image.open(fpath_03))  # Shape: [3, H, W]
            img_02 = transform(Image.open(fpath_02))

            # Flatten the images to compute per-channel statistics
            img_03 = img_03.view(3, -1)  # Shape: [3, H*W]
            img_02 = img_02.view(3, -1)

            # Welford's online algorithm for mean and variance
            count_rgb += 1
            delta_03 = img_03 - mean_03[:, None]
            mean_03 += delta_03.sum(dim=1) / (count_rgb * img_03.size(1))
            delta2_03 = img_03 - mean_03[:, None]
            m2_03 += (delta_03 * delta2_03).sum(dim=1)

            delta_02 = img_02 - mean_02[:, None]
            mean_02 += delta_02.sum(dim=1) / (count_rgb * img_02.size(1))
            delta2_02 = img_02 - mean_02[:, None]
            m2_02 += (delta_02 * delta2_02).sum(dim=1)

        # Process a subset of depth maps
        for fpath_depth in fpaths_depth:
            if count_depth >= sample_limit:
                break

            # Load depth map
            depth_map = np.array(Image.open(fpath_depth)).astype(np.float32) / 1000.0  # Mid-Air depth in meters
            depth_map = depth_map.flatten()
            depth_map = depth_map[depth_map > 0]  # Exclude zero values (invalid depth)
            if len(depth_map) > 0:
                depth_values.extend(depth_map)
                count_depth += 1

        if count_rgb >= sample_limit and count_depth >= sample_limit:
            break

    if count_rgb >= sample_limit and count_depth >= sample_limit:
        break

# Compute RGB standard deviation
std_03 = torch.sqrt(m2_03 / (count_rgb * img_03.size(1)))
std_02 = torch.sqrt(m2_02 / (count_rgb * img_02.size(1)))

# Compute depth mean and std
depth_values = np.array(depth_values)
depth_mean = depth_values.mean() if len(depth_values) > 0 else 0.0
depth_std = depth_values.std() if len(depth_values) > 0 else 1.0

# Convert RGB results to tuples for params.py
mean_03 = tuple(mean_03.tolist())
mean_02 = tuple(mean_02.tolist())
std_03 = tuple(std_03.tolist())
std_02 = tuple(std_02.tolist())

# Print results
print(f"Computed statistics for Kite_training dataset:")
print(f"RGB (based on {count_rgb} images):")
print(f"img_means_03: {mean_03}")
print(f"img_stds_03: {std_03}")
print(f"img_means_02: {mean_02}")
print(f"img_stds_02: {std_02}")
print(f"Depth (based on {count_depth} depth maps):")
print(f"depth_mean: {depth_mean}")
print(f"depth_std: {depth_std}")
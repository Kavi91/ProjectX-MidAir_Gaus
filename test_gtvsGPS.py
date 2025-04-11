#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import to_ned_pose

def main():
    # Hardcoded file paths (adjust if needed)
    gt_path = "/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_0008.npy"
    gps_path = "/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/trajectory_0008/gps.npy"
    
    # Load ground truth; it is assumed to be a [N, 6] array, where columns 3-5 (x, y, z) are in NED.
    ground_truth = np.load(gt_path)
    print("Loaded ground truth shape:", ground_truth.shape)
    gt_positions = ground_truth[:, 3:6]  # Translation component in NED

    # Load raw GPS data; assumed shape [M, 6] (position + velocity).
    gps_raw = np.load(gps_path)
    print("Loaded raw GPS shape:", gps_raw.shape)
    
    # Convert raw GPS data to NED using the provided helper function.
    gps_tensor = torch.tensor(gps_raw, dtype=torch.float32)
    gps_ned = to_ned_pose(gps_tensor, is_absolute=True)
    gps_positions = gps_ned[:, :3].cpu().numpy()  # Use position part (NED)

    # Compute offset to align the initial positions:
    # For example, subtract the difference between the first ground truth and first GPS position.
    offset = gt_positions[0] - gps_positions[0]
    print("Computed offset:", offset)

    # Apply the offset correction to the GPS positions.
    gps_positions_aligned = gps_positions + offset

    # Create a 3D plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth as a blue line.
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
            label="Ground Truth", color="blue", linewidth=2)
    
    # Plot original GPS positions in red dashed line.
    ax.plot(gps_positions[:, 0], gps_positions[:, 1], gps_positions[:, 2],
            label="Original GPS", color="red", marker="o", linestyle="--", markersize=6)
    
    # Plot aligned GPS positions in green dashed line.
    ax.plot(gps_positions_aligned[:, 0], gps_positions_aligned[:, 1], gps_positions_aligned[:, 2],
            label="Aligned GPS", color="green", marker="^", linestyle="--", markersize=6)
    
    ax.set_title("3D Plot: Ground Truth vs. Original & Aligned GPS (NED)")
    ax.set_xlabel("North")
    ax.set_ylabel("East")
    ax.set_zlabel("Down")
    ax.legend()
    
    # Save the plot instead of showing it.
    save_path = "plot_gt_gps_aligned.png"
    plt.savefig(save_path)
    print("Plot saved as:", save_path)

if __name__ == "__main__":
    main()

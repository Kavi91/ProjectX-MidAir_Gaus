import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from params import par

# Define the output directory for saving plots
output_dir = "/home/krkavinda/ProjectX-midAir/ground_thruth_plots"
os.makedirs(output_dir, exist_ok=True)

# Define climate sets (from params.py)
climate_sets = par.climate_sets

# Iterate over all climate sets and trajectories
for climate_set in climate_sets:
    # Get list of trajectories for this climate set
    traj_list = [d for d in os.listdir(f"/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/") if d.startswith("trajectory_")]
    traj_list.sort()

    for traj in traj_list:
        # Load ground truth poses
        traj_id = traj.split('_')[1]  # e.g., '3000' from 'trajectory_3000'
        poses_path = f"/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/{climate_set}/poses/poses_{traj_id}.npy"
        if not os.path.exists(poses_path):
            print(f"Poses file not found for {climate_set}/{traj}, skipping.")
            continue

        poses = np.load(poses_path)  # Shape: (num_frames, 6) [roll, pitch, yaw, x, y, z]

        # Extract X, Y, and Z positions for 3D trajectory plot
        x_positions = poses[:, 3]  # X (North in NED frame)
        y_positions = poses[:, 4]  # Y (East in NED frame)
        z_positions = poses[:, 5]  # Z (Down in NED frame)

        # Create the 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(x_positions, y_positions, z_positions, label=f"{climate_set}/{traj}", color='blue')
        ax.scatter(x_positions[0], y_positions[0], z_positions[0], color='green', label='Start', marker='o', s=100)
        ax.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', label='End', marker='x', s=100)

        # Set labels and title
        ax.set_xlabel('X Position (m) [North]')
        ax.set_ylabel('Y Position (m) [East]')
        ax.set_zlabel('Z Position (m) [Down]')
        ax.set_title(f"Ground Truth 3D Trajectory: {climate_set}/{traj}")
        ax.legend()
        ax.grid(True)

        # Adjust the view angle for better visualization
        ax.view_init(elev=30, azim=45)

        # Save the plot
        plot_filename = os.path.join(output_dir, f"trajectory_3d_{climate_set.replace('/', '_')}_{traj_id}.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"Saved 3D plot to {plot_filename}")
        plt.close()  # Close the figure to free memory

print("All ground truth 3D trajectory plots have been saved.")
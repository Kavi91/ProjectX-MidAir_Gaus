import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Define output directory
output_dir = "/home/krkavinda/ProjectX-midAir/comparison_plots"
os.makedirs(output_dir, exist_ok=True)

# File paths
pred_path = '/home/krkavinda/ProjectX-midAir/result/out_Kite_training_sunny_trajectory_0008.txt'
gt_path = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_0008.npy'
traj_id = '3008'
climate_set = 'Kite_training/cloudy'

# Load data
print("Loading predicted data...")
pred_data = np.loadtxt(pred_path, delimiter=',')
pred_data = pred_data[:, ~np.all(np.isnan(pred_data), axis=0)]
print("First few rows of predicted data:\n", pred_data[:5])

print("\nLoading ground truth data...")
gt_data = np.load(gt_path)
print("First few rows of ground truth data:\n", gt_data[:5])

print(f"\nPredicted data shape: {pred_data.shape}")
print(f"Ground truth data shape: {gt_data.shape}")

# Extract positions (columns 3,4,5 for x,y,z)
gt_x = gt_data[:, 3]  # North
gt_y = gt_data[:, 4]  # East
gt_z = gt_data[:, 5]  # Down
pred_x = pred_data[:, 3]  # North
pred_y = pred_data[:, 4]  # East
pred_z = pred_data[:, 5]  # Down

# Print data ranges
print("\nData ranges:")
print(f"Ground truth X: {min(gt_x):.2f} to {max(gt_x):.2f}")
print(f"Ground truth Y: {min(gt_y):.2f} to {max(gt_y):.2f}")
print(f"Ground truth Z: {min(gt_z):.2f} to {max(gt_z):.2f}")
print(f"Predicted X: {min(pred_x):.2f} to {max(pred_x):.2f}")
print(f"Predicted Y: {min(pred_y):.2f} to {max(pred_y):.2f}")
print(f"Predicted Z: {min(pred_z):.2f} to {max(pred_z):.2f}")

# Try scaling predictions (assuming meters vs kilometers or similar)
scale_factor = 1 # Try 100 as a starting point (could be 1000, etc.)
pred_x_scaled = pred_x * scale_factor
pred_y_scaled = pred_y * scale_factor
pred_z_scaled = pred_z * scale_factor

print(f"\nScaled Predicted ranges (x{scale_factor}):")
print(f"Scaled Predicted X: {min(pred_x_scaled):.2f} to {max(pred_x_scaled):.2f}")
print(f"Scaled Predicted Y: {min(pred_y_scaled):.2f} to {max(pred_y_scaled):.2f}")
print(f"Scaled Predicted Z: {min(pred_z_scaled):.2f} to {max(pred_z_scaled):.2f}")

# Create 3D trajectory plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot ground truth
ax.plot(gt_x, gt_y, gt_z, 'r-', label='Ground Truth', alpha=0.6)
ax.scatter(gt_x[0], gt_y[0], gt_z[0], color='lime', label='Start (GT)', marker='o', s=100)
ax.scatter(gt_x[-1], gt_y[-1], gt_z[-1], color='red', label='End (GT)', marker='x', s=100)

# Plot scaled predictions
ax.plot(pred_x_scaled, pred_y_scaled, pred_z_scaled, 'b-', label=f'Predicted (x{scale_factor})', alpha=0.6)
ax.scatter(pred_x_scaled[0], pred_y_scaled[0], pred_z_scaled[0], color='green', label='Start (Pred)', marker='o', s=100)
ax.scatter(pred_x_scaled[-1], pred_y_scaled[-1], pred_z_scaled[-1], color='blue', label='End (Pred)', marker='x', s=100)

ax.set_xlabel('X Position (m) [North]')
ax.set_ylabel('Y Position (m) [East]')
ax.set_zlabel('Z Position (m) [Down]')
ax.set_title(f"Trajectory Comparison: {climate_set}/trajectory_{traj_id}")
ax.legend()
ax.grid(True)
ax.view_init(elev=30, azim=45)
ax.set_box_aspect([1,1,1])

# Save 3D plot
plot_3d_filename = os.path.join(output_dir, f"comparison_3d_{climate_set.replace('/', '_')}_{traj_id}.png")
plt.savefig(plot_3d_filename, dpi=300, bbox_inches='tight')
print(f"Saved 3D comparison plot to {plot_3d_filename}")

# Plot position errors with scaled predictions
fig2 = plt.figure(figsize=(12, 4))
time = np.arange(min(len(pred_x), len(gt_x)))
pred_pos_scaled = np.column_stack((pred_x_scaled, pred_y_scaled, pred_z_scaled))
gt_pos = np.column_stack((gt_x, gt_y, gt_z))
errors = np.linalg.norm(pred_pos_scaled[:len(time)] - gt_pos[:len(time)], axis=1)

plt.plot(time, errors, 'g-', label='Position Error')
plt.xlabel('Time Step')
plt.ylabel('Error Magnitude (m)')
plt.title(f"Position Error Over Time: {climate_set}/trajectory_{traj_id}")
plt.legend()
plt.grid(True)

# Save error plot
plot_error_filename = os.path.join(output_dir, f"error_{climate_set.replace('/', '_')}_{traj_id}.png")
plt.savefig(plot_error_filename, dpi=300, bbox_inches='tight')
print(f"Saved error plot to {plot_error_filename}")

plt.close('all')

# Print statistics
print("Mean Position Error:", np.mean(errors))
print("Max Position Error:", np.max(errors))
print("Std Dev of Error:", np.std(errors))
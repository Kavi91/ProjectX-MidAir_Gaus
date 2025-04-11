import os
import numpy as np
from scipy.spatial.transform import Rotation  # Import Rotation
from params import par

# Define directories
input_dir = "/home/krkavinda/ProjectX-midAir/result/"  # Match the test script's output directory
output_dir = "kitti-odom-eval/result/test_01/"
gt_dir = "/home/krkavinda/ProjectX-midAir/kitti-odom-eval/gt/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Debug: List files in the input directory
print(f"Listing files in {input_dir}:")
if os.path.exists(input_dir):
    files = os.listdir(input_dir)
    print(files)
else:
    print(f"Input directory {input_dir} does not exist.")

# Map Mid-Air trajectory IDs to climate
traj_to_climate = {
    '3008': 'Kite_training/cloudy',
    '2008': 'Kite_training/foggy',
    '0008': 'Kite_training/sunny',
    '1008': 'Kite_training/sunset'
}

def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles (roll, pitch, yaw) to a rotation matrix."""
    roll, pitch, yaw = euler_angles
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

# Iterate over each climate and trajectory in test_traj_ids
for climate, traj_ids in par.test_traj_ids.items():
    for traj_id in traj_ids:
        # Sanitize the climate name for the filename (replace '/' with '_')
        climate_sanitized = climate.replace('/', '_')
        input_file = os.path.join(input_dir, f"out_{climate_sanitized}_{traj_id}.txt")
        # Extract the numeric part of the trajectory ID (e.g., 'trajectory_3008' -> '3008')
        traj_num = traj_id.replace('trajectory_', '')
        # Use the numeric part for the output file name to match KITTI format
        output_file = os.path.join(output_dir, f"{traj_num}.txt")
        # Load the corresponding ground truth file to determine the number of poses
        gt_file = os.path.join(gt_dir, f"{traj_num}.txt")
        
        if not os.path.exists(input_file):
            print(f"File {input_file} not found. Skipping.")
            continue
        
        if not os.path.exists(gt_file):
            print(f"Ground truth file {gt_file} not found. Skipping.")
            continue

        # Load the ground truth poses to determine the number of poses
        with open(gt_file, 'r') as f_gt:
            gt_poses = [line.strip().split() for line in f_gt if line.strip()]
        num_gt_poses = len(gt_poses)
        print(f"Sequence {traj_num}: Ground truth poses = {num_gt_poses}")

        # Load the predicted poses
        print(f"Processing {input_file}...")
        with open(input_file, "r") as fin:
            pred_poses = [line.strip() for line in fin if line.strip()]
            for i, line in enumerate(pred_poses):
                pred_poses[i] = [float(x) for x in line.split(",") if x.strip()]
                if len(pred_poses[i]) != 6:
                    print("Skipping line (unexpected number of values):", line)
                    pred_poses[i] = None
            # Filter out invalid lines
            pred_poses = [pose for pose in pred_poses if pose is not None]
        
        num_pred_poses = len(pred_poses)
        print(f"Sequence {traj_num}: Predicted poses (before truncation) = {num_pred_poses}")

        # Truncate predicted poses to match the number of ground truth poses
        pred_poses = pred_poses[:num_gt_poses]
        num_pred_poses = len(pred_poses)
        print(f"Sequence {traj_num}: Predicted poses (after truncation) = {num_pred_poses}")

        # Convert and save the predicted poses
        with open(output_file, "w") as fout:
            for pose in pred_poses:
                # First three values are Euler angles, last three are translation
                euler_angles = pose[:3]  # [roll, pitch, yaw] in radians
                translation = pose[3:]   # [x, y, z] in meters
                
                # Convert Euler angles to rotation matrix
                R = euler_to_rotation_matrix(euler_angles)  # 3x3 matrix
                
                # Form the 3x4 transformation matrix [R | t]
                T = np.concatenate([R, np.array(translation).reshape(3, 1)], axis=1)  # 3x4 matrix
                
                # Flatten the matrix into 12 numbers (row-major order) and write space-separated
                T_flat = T.flatten()
                fout.write(" ".join(map(str, T_flat)) + "\n")
        print(f"Saved converted file to {output_file}")
import os
import numpy as np
from scipy.spatial.transform import Rotation
from params import par

# Define directories
midair_gt_base_dir = "/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/"
kitti_gt_dir = "/home/krkavinda/ProjectX-midAir/kitti-odom-eval/gt/"

# Ensure output directory exists
os.makedirs(kitti_gt_dir, exist_ok=True)

def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles (roll, pitch, yaw) to a rotation matrix."""
    roll, pitch, yaw = euler_angles
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

# Iterate over each climate and trajectory in test_traj_ids
for climate, traj_ids in par.test_traj_ids.items():
    for traj_id in traj_ids:
        # Extract the numeric part of the trajectory ID (e.g., 'trajectory_3008' -> '3008')
        traj_num = traj_id.replace('trajectory_', '')
        
        # Load the Mid-Air ground truth .npy file
        midair_gt_file = os.path.join(midair_gt_base_dir, climate, 'poses', f'poses_{traj_num}.npy')
        if not os.path.exists(midair_gt_file):
            print(f"Mid-Air ground truth file not found: {midair_gt_file}, skipping.")
            continue
        
        # Load the .npy file (Mid-Air format: [roll, pitch, yaw, x, y, z])
        gt_poses = np.load(midair_gt_file)  # Shape: (num_frames, 6)
        
        # Convert to KITTI format (3x4 transformation matrices)
        kitti_poses = []
        for pose in gt_poses:
            euler_angles = pose[:3]  # [roll, pitch, yaw] in radians
            translation = pose[3:]   # [x, y, z] in meters
            # Convert Euler angles to rotation matrix
            R = euler_to_rotation_matrix(euler_angles)  # 3x3 matrix
            # Form the 3x4 transformation matrix [R | t]
            T = np.concatenate([R, np.array(translation).reshape(3, 1)], axis=1)  # 3x4 matrix
            # Flatten the matrix into 12 numbers (row-major order)
            T_flat = T.flatten()
            kitti_poses.append(T_flat)
        
        # Save the converted poses in KITTI format
        kitti_gt_file = os.path.join(kitti_gt_dir, f"{traj_num}.txt")
        with open(kitti_gt_file, 'w') as f:
            for pose in kitti_poses:
                f.write(" ".join(map(str, pose)) + "\n")
        print(f"Saved converted ground truth to {kitti_gt_file}")

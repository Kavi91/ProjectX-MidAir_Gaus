import numpy as np
import os

# Base directory for ground truth poses
base_path = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training'

# Define climates under Kite_training
climates = ['cloudy', 'foggy', 'sunny', 'sunset']

# Map trajectories to climate sets (based on the provided script)
traj_mapping = {
    'cloudy': [f'trajectory_{i:04d}' for i in range(3000, 3030)],
    'foggy': [f'trajectory_{i:04d}' for i in range(2000, 2030)],
    'sunny': [f'trajectory_{i:04d}' for i in range(0, 30)],
    'sunset': [f'trajectory_{i:04d}' for i in range(1000, 1030)],
}

# Check pose counts for each climate
for climate in climates:
    print(f"\nChecking ground truth poses for climate: {climate}")
    print("=" * 50)
    
    # Get the list of trajectories for this climate
    traj_ids = traj_mapping.get(climate, [])
    if not traj_ids:
        print(f"No trajectories defined for climate {climate}, skipping.")
        continue
    
    # Path to the poses directory
    poses_dir = f'{base_path}/{climate}/poses'
    if not os.path.exists(poses_dir):
        print(f"Poses directory not found: {poses_dir}, skipping.")
        continue
    
    # Iterate over each trajectory
    for traj_id in traj_ids:
        traj_num = traj_id.replace('trajectory_', '')
        pose_file = f'{poses_dir}/poses_{traj_num}.npy'
        
        if not os.path.exists(pose_file):
            print(f"Pose file not found: {pose_file}, skipping.")
            continue
        
        # Load the ground truth poses
        poses = np.load(pose_file)
        pose_count = poses.shape[0]
        print(f"Trajectory {traj_id}: {pose_count} poses")
        print(f"First 5 ground truth poses for {traj_id}: {poses[:5]}")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from params import par

plt.switch_backend('Agg')
pose_GT_dir = par.pose_dir
predicted_result_dir = './result/'

def plot_route(ax, gt, out, c_gt='g', c_out='r', label_gt='Ground Truth', label_out='DeepVO'):
    # Using NED: indices 3,4,5 for North, East, Down
    x_gt, y_gt, z_gt = gt[:, 3], gt[:, 4], gt[:, 5]
    x_out, y_out, z_out = out[:, 3], out[:, 4], out[:, 5]
    # Align trajectories by initial offset
    x_out = x_out - (x_out[0] - x_gt[0])
    y_out = y_out - (y_out[0] - y_gt[0])
    z_out = z_out - (z_out[0] - z_gt[0])
    ax.plot(x_gt, y_gt, z_gt, color=c_gt, label=label_gt)
    ax.plot(x_out, y_out, z_out, color=c_out, label=label_out)
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Down (m)')
    x_mid, y_mid, z_mid = np.mean(x_gt), np.mean(y_gt), np.mean(z_gt)
    max_range = max(np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)) / 2
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    ax.legend()

trajectories_to_plot = [
    ('PLE_training/spring', 'trajectory_5005'),
]

for climate, traj_id in trajectories_to_plot:
    print('='*50)
    print(f'Plotting Trajectory: {climate}/{traj_id}')
    traj_num = traj_id.split('_')[1]
    GT_pose_path = f"{pose_GT_dir}/{climate}/poses/poses_{traj_num}.npy"
    gt = np.load(GT_pose_path)
    climate_sanitized = climate.replace('/', '_')
    pose_result_path = f"{predicted_result_dir}/out_{climate_sanitized}_{traj_id}.txt"
    with open(pose_result_path, 'r') as f:
        out_lines = f.readlines()
    out = np.array([[float(val) for val in line.strip().split(',')] for line in out_lines])
    min_len = min(len(gt), len(out))
    gt = gt[:min_len]
    out = out[:min_len]
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    plot_route(ax, gt, out)
    plt.title(f"Trajectory Comparison: {climate}/{traj_id}")
    plt.savefig(f"{predicted_result_dir}/traj_{climate_sanitized}_{traj_id}.png")
    plt.close()

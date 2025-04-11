from params import par
from model_2 import StereoAdaptiveVO
import numpy as np
from PIL import Image
import os
import time
import torch
from data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def compute_mse(pred_traj, gt_traj):
    return np.mean((pred_traj - gt_traj) ** 2)

def compute_ate(pred_traj, gt_traj):
    pred_trans = pred_traj[:, 3:]
    gt_trans = gt_traj[:, 3:6]
    trans_errors = pred_trans - gt_trans
    return np.sqrt(np.mean(np.sum(trans_errors ** 2, axis=1)))

def plot_trajectories(pred_traj, gt_traj, climate, traj_id, save_dir):
    pred_x, pred_y = pred_traj[:, 3], pred_traj[:, 4]
    gt_x, gt_y = gt_traj[:, 3], gt_traj[:, 4]
    plt.figure(figsize=(10, 8))
    plt.plot(pred_x, pred_y, 'b--', label='Predicted Trajectory')
    plt.plot(gt_x, gt_y, 'g-', label='Ground Truth Trajectory')
    plt.xlabel('North (m)')
    plt.ylabel('East (m)')
    plt.title(f'Trajectory Comparison: {climate}/{traj_id}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    climate_sanitized = climate.replace('/', '_')
    plt.savefig(f'{save_dir}/traj_{climate_sanitized}_{traj_id}.png')
    plt.close()

if __name__ == '__main__':
    trajectories_to_test = [
        ('PLE_training/spring', 'trajectory_5005'),
    ]
    load_model_path = par.load_model_path
    save_dir = 'result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    M_deepvo = StereoAdaptiveVO(par.img_h, par.img_w, par.batch_norm, hidden_size=par.rnn_hidden_size, num_layers=2)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location='cuda', weights_only=True))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location='cpu', weights_only=True))
    M_deepvo.eval()
    n_workers = 1
    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
    overlap = seq_len - 1
    batch_size = par.batch_size

    for climate, test_traj_id in trajectories_to_test:
        train_df, valid_df, test_df = get_data_info(
            climate_sets=[climate],
            seq_len_range=[seq_len, seq_len],
            overlap=overlap,
            sample_times=1,
            shuffle=False,
            sort=False,
            include_test=True
        )
        df = test_df[df['seq_len'] == seq_len]
        dataset = ImageSequenceDataset(
            df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
            par.img_means_02, par.img_stds_02, par.minus_point_5
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        
        traj_num = test_traj_id.split('_')[1]
        gt_pose = np.load(f"{par.pose_dir}/{climate}/poses/poses_{traj_num}.npy")
        all_pred_poses = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                _, (x_03, x_02, x_depth, x_imu, x_gps), y = batch
                if use_cuda:
                    x_03 = x_03.cuda()
                    x_02 = x_02.cuda()
                    x_depth = x_depth.cuda() if x_depth.numel() > 0 else x_depth
                    x_imu = x_imu.cuda() if x_imu.numel() > 0 else x_imu
                    x_gps = x_gps.cuda() if x_gps.numel() > 0 else x_gps
                x = (x_03, x_02, x_depth, x_imu, x_gps)
                pred_pose = M_deepvo.forward(x)
                all_pred_poses.append(pred_pose.cpu())
        all_pred_poses = torch.cat(all_pred_poses, dim=0)
        total_frames = all_pred_poses.shape[0] * all_pred_poses.shape[1]
        all_pred_poses = all_pred_poses.view(total_frames, 6)
        all_pred_poses = all_pred_poses.unsqueeze(0)
        absolute_poses = M_deepvo.compute_absolute_poses(all_pred_poses)
        absolute_poses = absolute_poses.squeeze(0).numpy()
        min_len = min(len(absolute_poses), len(gt_pose))
        absolute_poses = absolute_poses[:min_len]
        gt_pose = gt_pose[:min_len]
        mse = compute_mse(absolute_poses, gt_pose)
        ate = compute_ate(absolute_poses, gt_pose)
        print(f"MSE: {mse:.4f}, ATE: {ate:.4f}")
        plot_trajectories(absolute_poses, gt_pose, climate, test_traj_id, save_dir)

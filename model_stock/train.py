import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
from params import par
from model_5 import StereoAdaptiveVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info
import wandb
from tqdm import tqdm

# Initialize WandB
wandb.init(project="deepvo_training", config=vars(par))
wandb.config.update({"start_time": time.time()})

# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n' + '=' * 50 + '\n')
    f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
    f.write('\n' + '=' * 50 + '\n')

# Prepare Data
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
    print('Load data info from {}'.format(par.train_data_info_path))
    train_df = pd.read_pickle(par.train_data_info_path)
    valid_df = pd.read_pickle(par.valid_data_info_path)
else:
    print('Create new data info')
    if par.partition is not None:
        partition = par.partition
        train_df, valid_df = get_partition_data_info(partition, par.train_video, par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=True)
    else:
        train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)    
        valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
    train_df.to_pickle(par.train_data_info_path)
    valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(
    train_df, 
    par.resize_mode, 
    (par.img_h, par.img_w), 
    par.img_means_03, 
    par.img_stds_03, 
    par.img_means_02, 
    par.img_stds_02, 
    par.minus_point_5
)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(
    valid_df, 
    par.resize_mode, 
    (par.img_h, par.img_w), 
    par.img_means_03, 
    par.img_stds_03, 
    par.img_means_02, 
    par.img_stds_02, 
    par.minus_point_5
)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('Number of training batches: ', len(train_dl))
print('Number of validation batches: ', len(valid_dl))

# Model
M_deepvo = StereoAdaptiveVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo = M_deepvo.cuda()

# Load FlowNet weights pretrained with FlyingChairs
if par.pretrained_flownet and not par.resume:
    if use_cuda:
        pretrained_w = torch.load(par.pretrained_flownet, weights_only=True)
    else:
        pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu', weights_only=True)
    print('Load FlowNet pretrained model')
    model_dict = M_deepvo.state_dict()
    update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)

# Create optimizer
if par.optim['opt'] == 'Adam':
    optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
elif par.optim['opt'] == 'Adagrad':
    optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
elif par.optim['opt'] == 'Cosine':
    optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
    T_iter = par.optim['T'] * len(train_dl)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Add learning rate scheduler
warmup_epochs = 5
lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

# Load trained model and optimizer
if par.resume:
    M_deepvo.load_state_dict(torch.load(par.load_model_path, weights_only=True))
    optimizer.load_state_dict(torch.load(par.load_optimizer_path, weights_only=True))
    print('Load model from: ', par.load_model_path)
    print('Load optimizer from: ', par.load_optimizer_path)

# Train
print('Record loss in: ', par.record_path)
min_loss_t = 1e10
min_loss_v = 1e10
M_deepvo.train()

# Initialize variables for ETA calculation
total_start_time = time.time()
epoch_times = []

for ep in range(par.epochs):
    epoch_start_time = time.time()
    print('=' * 50)
    # Train
    M_deepvo.train()
    loss_mean = 0
    t_loss_list = []
    with tqdm(train_dl, desc=f"Epoch {ep+1}/{par.epochs} [Train]", unit="batch") as tbar:
        for batch_idx, (_, (t_x_03, t_x_02, t_x_depth, t_x_lidar), t_y) in enumerate(tbar):
            if use_cuda:
                t_x_03 = t_x_03.cuda(non_blocking=par.pin_mem)
                t_x_02 = t_x_02.cuda(non_blocking=par.pin_mem)
                t_x_depth = t_x_depth.cuda(non_blocking=par.pin_mem)
                t_x_lidar = t_x_lidar.cuda(non_blocking=par.pin_mem) if par.enable_lidar else None
                t_y = t_y.cuda(non_blocking=par.pin_mem)
            
            # Debug: Log shapes and depth feature statistics (first batch of first epoch)
            if ep == 0 and batch_idx == 0:
                print(f"Input shapes - t_x_03: {t_x_03.shape}, t_x_02: {t_x_02.shape}, t_x_depth: {t_x_depth.shape}, t_x_lidar: {t_x_lidar.shape if par.enable_lidar else 'disabled'}, t_y: {t_y.shape}")
                with torch.no_grad():
                    t_x_depth_encoded = torch.cat((t_x_depth[:, :-1], t_x_depth[:, 1:]), dim=2)
                    new_seq_len_depth = t_x_depth_encoded.size(1)
                    batch_size = t_x_depth_encoded.size(0)
                    t_x_depth_encoded = t_x_depth_encoded.view(batch_size * new_seq_len_depth, t_x_depth_encoded.size(2), t_x_depth_encoded.size(3), t_x_depth_encoded.size(4))
                    depth_features = M_deepvo.encode_depth(t_x_depth_encoded)
                    print(f"Encoded depth features mean: {depth_features.mean().item():.4f}, std: {depth_features.std().item():.4f}")
                    print(f"Encoded depth features min: {depth_features.min().item():.4f}, max: {depth_features.max().item():.4f}")
            
            ls = M_deepvo.step((t_x_03, t_x_02, t_x_depth, t_x_lidar), t_y, optimizer)
            ls = ls.data.cpu().numpy()
            t_loss_list.append(float(ls))
            loss_mean += float(ls)
            
            tbar.set_postfix({'loss': f"{float(ls):.4f}"})
    train_time = time.time() - epoch_start_time
    print('Train take {:.1f} sec'.format(train_time))
    loss_mean /= len(train_dl)

    # Validation
    valid_start_time = time.time()
    M_deepvo.eval()
    loss_mean_valid = 0
    v_loss_list = []
    # Lists to store confidence scores for the epoch
    alpha_C_list, alpha_D_list, alpha_L_list = [], [], []
    with tqdm(valid_dl, desc=f"Epoch {ep+1}/{par.epochs} [Valid]", unit="batch") as vbar:
        for batch_idx, (_, (v_x_03, v_x_02, v_x_depth, v_x_lidar), v_y) in enumerate(vbar):
            if use_cuda:
                v_x_03 = v_x_03.cuda(non_blocking=par.pin_mem)
                v_x_02 = v_x_02.cuda(non_blocking=par.pin_mem)
                v_x_depth = v_x_depth.cuda(non_blocking=par.pin_mem)
                v_x_lidar = v_x_lidar.cuda(non_blocking=par.pin_mem) if par.enable_lidar else None
                v_y = v_y.cuda(non_blocking=par.pin_mem)
            v_pose, v_pose_covar, v_modality_covars = M_deepvo.forward((v_x_03, v_x_02, v_x_depth, v_x_lidar), return_uncertainty=True)
            v_ls = M_deepvo.get_loss((v_x_03, v_x_02, v_x_depth, v_x_lidar), v_y).data.cpu().numpy()
            v_loss_list.append(float(v_ls))
            loss_mean_valid += float(v_ls)
            
            # Extract confidence scores (alpha_C, alpha_D, alpha_L) from the fusion module
            batch_alpha = M_deepvo.fusion_module.last_alpha  # Shape: [batch_size * seq_len, num_modalities, 1, 1, 1]
            alpha_C = batch_alpha[:, 0].mean().item()  # Mean over batch and sequence
            alpha_D = batch_alpha[:, 1].mean().item()
            alpha_L = batch_alpha[:, 2].mean().item()
            alpha_C_list.append(alpha_C)
            alpha_D_list.append(alpha_D)
            alpha_L_list.append(alpha_L)
            
            # Log uncertainty estimates and confidence scores to the progress bar
            vbar.set_postfix({
                'loss': f"{float(v_ls):.4f}",
                'rgb_unc': v_modality_covars['rgb'].mean().item() if 'rgb' in v_modality_covars else 0.0,
                'depth_unc': v_modality_covars['depth'].mean().item() if 'depth' in v_modality_covars else 0.0,
                'lidar_unc': v_modality_covars['lidar'].mean().item() if 'lidar' in v_modality_covars else 0.0,
                'pose_unc': torch.log(torch.diagonal(v_pose_covar, dim1=-2, dim2=-1)).mean().item(),
                'alpha_C': alpha_C,
                'alpha_D': alpha_D,
                'alpha_L': alpha_L
            })
    valid_time = time.time() - valid_start_time
    print('Valid take {:.1f} sec'.format(valid_time))
    loss_mean_valid /= len(valid_dl)

    # Calculate epoch time and update ETA
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = par.epochs - (ep + 1)
    eta_seconds = avg_epoch_time * remaining_epochs
    eta_minutes = eta_seconds / 60
    eta_hours = eta_minutes / 60
    print(f"Epoch {ep+1}/{par.epochs} completed in {epoch_time:.1f} sec")
    print(f"Average time per epoch: {avg_epoch_time:.1f} sec")
    print(f"ETA to complete training: {eta_minutes:.1f} minutes ({eta_hours:.1f} hours)")

    # Compute average confidence scores for the epoch
    avg_alpha_C = np.mean(alpha_C_list)
    avg_alpha_D = np.mean(alpha_D_list)
    avg_alpha_L = np.mean(alpha_L_list)

    # Log to WandB
    wandb.log({
        "epoch": ep + 1,
        "train_loss_mean": loss_mean,
        "train_loss_std": np.std(t_loss_list),
        "valid_loss_mean": loss_mean_valid,
        "valid_loss_std": np.std(v_loss_list),
        "epoch_time": epoch_time,
        "eta_minutes": eta_minutes,
        "rgb_uncertainty": v_modality_covars['rgb'].mean().item() if 'rgb' in v_modality_covars else 0.0,
        "depth_uncertainty": v_modality_covars['depth'].mean().item() if 'depth' in v_modality_covars else 0.0,
        "lidar_uncertainty": v_modality_covars['lidar'].mean().item() if 'lidar' in v_modality_covars else 0.0,
        "pose_uncertainty": torch.log(torch.diagonal(v_pose_covar, dim1=-2, dim2=-1)).mean().item(),
        "alpha_C": avg_alpha_C,  # Log average confidence scores
        "alpha_D": avg_alpha_D,
        "alpha_L": avg_alpha_L
    })

    with open(par.record_path, 'a') as f:
        f.write('=' * 50 + '\n')
        f.write(f'Epoch {ep + 1}\ntrain loss mean: {loss_mean}, std: {np.std(t_loss_list):.2f}\nvalid loss mean: {loss_mean_valid}, std: {np.std(v_loss_list):.2f}\n')
        print(f'Epoch {ep + 1}\ntrain loss mean: {loss_mean}, std: {np.std(t_loss_list):.2f}\nvalid loss mean: {loss_mean_valid}, std: {np.std(v_loss_list):.2f}\n')

    # Save model
    check_interval = 1
    if loss_mean_valid < min_loss_v and ep % check_interval == 0:
        min_loss_v = loss_mean_valid
        print(f'Save model at ep {ep + 1}, mean of valid loss: {loss_mean_valid}')
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.valid')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.valid')
    if loss_mean < min_loss_t and ep % check_interval == 0:
        min_loss_t = loss_mean
        print(f'Save model at ep {ep + 1}, mean of train loss: {loss_mean}')
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.train')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.train')

    # Apply warmup scheduler for the first few epochs
    if ep < warmup_epochs:
        lr_scheduler_warmup.step()
    else:
        lr_scheduler.step(loss_mean_valid)

wandb.finish()
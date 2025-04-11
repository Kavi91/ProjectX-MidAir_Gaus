import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import pickle
from params import par
from model_2 import StereoAdaptiveVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset
import wandb
from tqdm import tqdm
from torch.amp import GradScaler, autocast

wandb.init(project="deepvo_training_midair", config=vars(par))
wandb.config.update({"start_time": time.time()})

mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
    f.write('\n' + '=' * 50 + '\n')
    f.write('\n'.join(f"{k}: {v}" for k, v in vars(par).items()))
    f.write('\n' + '=' * 50 + '\n')

if par.batch_size > 16:
    print(f"Warning: Batch size {par.batch_size} may lead to high memory usage. Consider adjusting batch size, number of workers, or using AMP.")

if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
    print('Loading data info from:', par.train_data_info_path)
    train_df = pd.read_pickle(par.train_data_info_path)
    valid_df = pd.read_pickle(par.valid_data_info_path)
else:
    print('Creating new data info...')
    train_df, valid_df = get_data_info(
        climate_sets=par.climate_sets, 
        seq_len=par.seq_len, 
        overlap=par.overlap,
        sample_times=par.sample_times, 
        shuffle=True, 
        sort=True, 
        include_test=False
    )
    train_df.to_pickle(par.train_data_info_path)
    valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(
    train_df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
    par.img_means_02, par.img_stds_02, par.minus_point_5, is_training=True
)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_batch_size = max(par.batch_size // 2, 8)
valid_sampler = SortedRandomBatchSampler(valid_df, valid_batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(
    valid_df, par.resize_mode, (par.img_h, par.img_w), par.img_means_03, par.img_stds_03,
    par.img_means_02, par.img_stds_02, par.minus_point_5, is_training=False
)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

stats_pickle_path = "datainfo/dataset_stats.pickle"
if os.path.exists(stats_pickle_path):
    print(f"Loading dataset sizes from {stats_pickle_path}")
    with open(stats_pickle_path, 'rb') as f:
        stats = pickle.load(f)
    if 'num_train_samples' in stats:
        num_train_samples = stats['num_train_samples']
        num_valid_samples = stats['num_valid_samples']
        num_train_batches = stats['num_train_batches']
        num_valid_batches = stats['num_valid_batches']
    else:
        num_train_samples = len(train_df.index)
        num_valid_samples = len(valid_df.index)
        num_train_batches = len(train_dl)
        num_valid_batches = len(valid_dl)
        stats.update({
            'num_train_samples': num_train_samples,
            'num_valid_samples': num_valid_samples,
            'num_train_batches': num_train_batches,
            'num_valid_batches': num_valid_batches
        })
        with open(stats_pickle_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Updated dataset statistics in {stats_pickle_path}")
else:
    num_train_samples = len(train_df.index)
    num_valid_samples = len(valid_df.index)
    num_train_batches = len(train_dl)
    num_valid_batches = len(valid_dl)
    stats = {
        'num_train_samples': num_train_samples,
        'num_valid_samples': num_valid_samples,
        'num_train_batches': num_train_batches,
        'num_valid_batches': num_valid_batches
    }
    with open(stats_pickle_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Saved dataset sizes to {stats_pickle_path}")

print('Training samples:', num_train_samples)
print('Validation samples:', num_valid_samples)
print('Training batches:', num_train_batches)
print('Validation batches:', num_valid_batches)

M_deepvo = StereoAdaptiveVO(
    img_h=par.img_h,
    img_w=par.img_w,
    batch_norm=par.batch_norm,
    input_channels=6,
    hidden_size=par.rnn_hidden_size,
    num_layers=2
)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available; using GPU.')
    M_deepvo = M_deepvo.cuda()
else:
    print('CUDA not available; using CPU.')

if par.pretrained_flownet and not par.resume:
    try:
        if use_cuda:
            pretrained_w = torch.load(par.pretrained_flownet, map_location='cuda')
        else:
            pretrained_w = torch.load(par.pretrained_flownet, map_location='cpu')
        print('Loading FlowNet pretrained model...')
        
        if 'state_dict' in pretrained_w:
            pretrained_dict = pretrained_w['state_dict']
        else:
            pretrained_dict = pretrained_w
        #print("Pretrained FlowNet keys (sample):", list(pretrained_dict.keys())[:10])
        
        model_dict = M_deepvo.state_dict()
        #print("Model keys (sample):", list(model_dict.keys())[:10])
        
        # Updated key mapping based on KITTI DeepVO
        key_mapping = {
            'conv1.0.weight': 'rgb_conv.0.0.weight',
            'conv1.1.weight': 'rgb_conv.0.1.weight',
            'conv1.1.bias': 'rgb_conv.0.1.bias',
            'conv1.1.running_mean': 'rgb_conv.0.1.running_mean',
            'conv1.1.running_var': 'rgb_conv.0.1.running_var',
            'conv2.0.weight': 'rgb_conv.1.0.weight',
            'conv2.1.weight': 'rgb_conv.1.1.weight',
            'conv2.1.bias': 'rgb_conv.1.1.bias',
            'conv2.1.running_mean': 'rgb_conv.1.1.running_mean',
            'conv2.1.running_var': 'rgb_conv.1.1.running_var',
            'conv3.0.weight': 'rgb_conv.2.0.weight',
            'conv3.1.weight': 'rgb_conv.2.1.weight',
            'conv3.1.bias': 'rgb_conv.2.1.bias',
            'conv3.1.running_mean': 'rgb_conv.2.1.running_mean',
            'conv3.1.running_var': 'rgb_conv.2.1.running_var',
            'conv3_1.0.weight': 'rgb_conv.3.0.weight',
            'conv3_1.1.weight': 'rgb_conv.3.1.weight',
            'conv3_1.1.bias': 'rgb_conv.3.1.bias',
            'conv3_1.1.running_mean': 'rgb_conv.3.1.running_mean',
            'conv3_1.1.running_var': 'rgb_conv.3.1.running_var',
            'conv4.0.weight': 'rgb_conv.4.0.weight',
            'conv4.1.weight': 'rgb_conv.4.1.weight',
            'conv4.1.bias': 'rgb_conv.4.1.bias',
            'conv4.1.running_mean': 'rgb_conv.4.1.running_mean',
            'conv4.1.running_var': 'rgb_conv.4.1.running_var',
            'conv4_1.0.weight': 'rgb_conv.5.0.weight',
            'conv4_1.1.weight': 'rgb_conv.5.1.weight',
            'conv4_1.1.bias': 'rgb_conv.5.1.bias',
            'conv4_1.1.running_mean': 'rgb_conv.5.1.running_mean',
            'conv4_1.1.running_var': 'rgb_conv.5.1.running_var',
            'conv5.0.weight': 'rgb_conv.6.0.weight',
            'conv5.1.weight': 'rgb_conv.6.1.weight',
            'conv5.1.bias': 'rgb_conv.6.1.bias',
            'conv5.1.running_mean': 'rgb_conv.6.1.running_mean',
            'conv5.1.running_var': 'rgb_conv.6.1.running_var',
            'conv5_1.0.weight': 'rgb_conv.7.0.weight',
            'conv5_1.1.weight': 'rgb_conv.7.1.weight',
            'conv5_1.1.bias': 'rgb_conv.7.1.bias',
            'conv5_1.1.running_mean': 'rgb_conv.7.1.running_mean',
            'conv5_1.1.running_var': 'rgb_conv.7.1.running_var',
            'conv6.0.weight': 'rgb_conv.8.0.weight',
            'conv6.1.weight': 'rgb_conv.8.1.weight',
            'conv6.1.bias': 'rgb_conv.8.1.bias',
            'conv6.1.running_mean': 'rgb_conv.8.1.running_mean',
            'conv6.1.running_var': 'rgb_conv.8.1.running_var',
        }
        
        update_dict = {}
        for pretrained_key, model_key in key_mapping.items():
            if pretrained_key in pretrained_dict and model_key in model_dict:
                if pretrained_dict[pretrained_key].shape == model_dict[model_key].shape:
                    update_dict[model_key] = pretrained_dict[pretrained_key]
                    #print(f"Mapped {pretrained_key} to {model_key}")
                else:
                    print(f"Shape mismatch for {model_key}: {pretrained_dict[pretrained_key].shape} vs {model_dict[model_key].shape}")
        
        missing_keys = set(model_dict.keys()) - set(update_dict.keys())
        if missing_keys:
            print("Warning: The following keys were not found in the pretrained checkpoint:", missing_keys)
        model_dict.update(update_dict)
        M_deepvo.load_state_dict(model_dict)
    except Exception as e:
        print("Error loading pretrained weights:", e)
else:
    print('Skipping FlowNet pretrained weights loading.')

optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=par.optim['lr'], weight_decay=par.optim.get('weight_decay', 0))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=par.epochs, eta_min=1e-6)
scaler = GradScaler()

print('Recording training metrics in:', par.record_path)
min_loss_t = 1e10
min_loss_v = 1e10
patience = 20
best_val_loss = min_loss_v
epochs_no_improve = 0

M_deepvo.train()
print(f"Training with climate sets: {par.climate_sets}")
print(f"Total epochs: {par.epochs}")

epoch_times = []
total_start_time = time.time()

for ep in range(par.epochs):
    epoch_start_time = time.time()
    print('=' * 50)
    
    # --- Training Phase ---
    M_deepvo.train()
    t_loss_list = []
    with tqdm(train_dl, desc=f"Epoch {ep+1}/{par.epochs} [Train]", unit="batch") as tbar:
        for batch_idx, (_, (t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps), t_y) in enumerate(tbar):
            if use_cuda:
                t_x_03 = t_x_03.cuda(non_blocking=par.pin_mem)
                t_x_02 = t_x_02.cuda(non_blocking=par.pin_mem)
                t_x_depth = t_x_depth.cuda(non_blocking=par.pin_mem)
                t_x_imu = t_x_imu.cuda(non_blocking=par.pin_mem)
                t_x_gps = t_x_gps.cuda(non_blocking=par.pin_mem)
                t_y = t_y.cuda(non_blocking=par.pin_mem)
            if ep == 0 and batch_idx == 0:
                print(f"Input shapes: t_x_03: {t_x_03.shape}, t_x_02: {t_x_02.shape}")
            loss_val = M_deepvo.step((t_x_03, t_x_02, t_x_depth, t_x_imu, t_x_gps), t_y, optimizer, scaler)
            t_loss_list.append(float(loss_val))
            tbar.set_postfix({'loss': f"{loss_val:.4f}"})
            torch.cuda.empty_cache()
    
    train_time = time.time() - epoch_start_time
    print('Training phase completed in {:.1f} sec'.format(train_time))
    train_loss_mean = np.mean(t_loss_list)
    train_loss_std = np.std(t_loss_list)
    
    # --- Validation Phase ---
    M_deepvo.eval()
    v_loss_list = []
    valid_start_time = time.time()
    with tqdm(valid_dl, desc=f"Epoch {ep+1}/{par.epochs} [Valid]", unit="batch") as vbar:
        for batch_idx, (_, (v_x_03, v_x_02, v_x_depth, v_x_imu, v_x_gps), v_y) in enumerate(vbar):
            if use_cuda:
                v_x_03 = v_x_03.cuda(non_blocking=par.pin_mem)
                v_x_02 = v_x_02.cuda(non_blocking=par.pin_mem)
                v_x_depth = v_x_depth.cuda(non_blocking=par.pin_mem)
                v_x_imu = v_x_imu.cuda(non_blocking=par.pin_mem)
                v_x_gps = v_x_gps.cuda(non_blocking=par.pin_mem)
                v_y = v_y.cuda(non_blocking=par.pin_mem)
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=True):
                    v_loss = M_deepvo.get_loss((v_x_03, v_x_02, v_x_depth, v_x_imu, v_x_gps), v_y)
            v_loss_val = float(v_loss.data.cpu().numpy())
            v_loss_list.append(v_loss_val)
            vbar.set_postfix({'loss': f"{v_loss_val:.4f}"})
            torch.cuda.empty_cache()
    
    valid_time = time.time() - valid_start_time
    print('Validation phase completed in {:.1f} sec'.format(valid_time))
    valid_loss_mean = np.mean(v_loss_list)
    valid_loss_std = np.std(v_loss_list)
    
    print(f"Epoch {ep+1} Summary:")
    print(f"   Train Loss: Mean = {train_loss_mean:.4f}, Std = {train_loss_std:.4f}")
    print(f"   Val   Loss: Mean = {valid_loss_mean:.4f}, Std = {valid_loss_std:.4f}")
    
    scheduler.step()
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {ep+1} completed in {epoch_time:.1f} sec, Learning Rate: {current_lr:.6f}")
    
    wandb.log({
        "epoch": ep + 1,
        "train_loss_mean": train_loss_mean,
        "train_loss_std": train_loss_std,
        "valid_loss_mean": valid_loss_mean,
        "valid_loss_std": valid_loss_std,
        "epoch_time": epoch_time,
        "learning_rate": current_lr,
    })
    
    with open(par.record_path, 'a') as f:
        f.write(f'\nEpoch {ep + 1}\nTrain Loss: Mean = {train_loss_mean:.4f}, Std = {train_loss_std:.4f}\n'
                f'Validation Loss: Mean = {valid_loss_mean:.4f}, Std = {valid_loss_std:.4f}\n'
                f'Learning Rate: {current_lr}\n')
    
    if valid_loss_mean < min_loss_v:
        min_loss_v = valid_loss_mean
        print(f"Validation loss improved at epoch {ep+1}; saving model...")
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.valid')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.valid')
    if train_loss_mean < min_loss_t:
        min_loss_t = train_loss_mean
        print(f"Training loss improved at epoch {ep+1}; saving model...")
        torch.save(M_deepvo.state_dict(), par.save_model_path + '.train')
        torch.save(optimizer.state_dict(), par.save_optimzer_path + '.train')
    
    if valid_loss_mean < best_val_loss:
        best_val_loss = valid_loss_mean
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}: No improvement for {patience} epochs.")
            break

wandb.finish()
print("Training complete in {:.1f} sec".format(time.time() - total_start_time))
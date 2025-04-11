import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from params import par
from helper import to_ned_pose, integrate_relative_poses

# Adaptive Gated Attention Fusion Module for multi-modal features.
class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels, depth_gate_scaling=par.depth_gate_scaling, imu_gate_scaling=par.imu_gate_scaling):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.lidar_channels = lidar_channels
        self.total_channels = rgb_channels + depth_channels + lidar_channels

        self.rgb_gate = nn.Linear(self.total_channels, rgb_channels)
        self.depth_gate = nn.Linear(self.total_channels, depth_channels)
        self.lidar_gate = nn.Linear(self.total_channels, lidar_channels)

        self.depth_gate_scaling = depth_gate_scaling
        self.imu_gate_scaling = imu_gate_scaling

        self.fusion_layer = nn.Linear(self.total_channels, self.total_channels)

    def forward(self, rgb_features, depth_features, lidar_features):
        combined = torch.cat((rgb_features, depth_features, lidar_features), dim=-1)
        rgb_attention = torch.sigmoid(self.rgb_gate(combined))
        depth_attention = torch.sigmoid(self.depth_gate(combined)) * self.depth_gate_scaling
        lidar_attention = torch.sigmoid(self.lidar_gate(combined)) * self.imu_gate_scaling
        rgb_weighted = rgb_features * rgb_attention
        depth_weighted = depth_features * depth_attention
        lidar_weighted = lidar_features * lidar_attention
        fused = torch.cat((rgb_weighted, depth_weighted, lidar_weighted), dim=-1)
        fused = self.fusion_layer(fused)
        return fused

# DeepVO-style convolutional block
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

# Stereo Adaptive Visual Odometry Model with DeepVO feature extractor
class StereoAdaptiveVO(nn.Module):
    def __init__(self, img_h, img_w, batch_norm, input_channels=6, hidden_size=512, num_layers=2):
        super(StereoAdaptiveVO, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # DeepVO-style RGB feature extraction for stacked stereo input
        self.rgb_conv = nn.Sequential(
            conv(self.batch_norm, 6, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0]),
            conv(self.batch_norm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1]),
            conv(self.batch_norm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2]),
            conv(self.batch_norm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3]),
            conv(self.batch_norm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4]),
            conv(self.batch_norm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5]),
            conv(self.batch_norm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6]),
            conv(self.batch_norm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7]),
            conv(self.batch_norm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        )
        # Compute feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, img_h, img_w)
            dummy_output = self.rgb_conv(dummy_input)
            self.rgb_feature_size = dummy_output.view(1, -1).size(1)
        self.rgb_fc = nn.Linear(self.rgb_feature_size, 512)

        # Depth feature extraction (if enabled)
        if par.enable_depth:
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.depth_feature_size = (img_h // 8) * (img_w // 8) * 128
            self.depth_fc = nn.Linear(self.depth_feature_size, 256)
        else:
            self.depth_fc = nn.Identity()

        # IMU feature extraction (if enabled)
        if par.enable_imu:
            self.imu_fc = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
        else:
            self.imu_fc = nn.Identity()

        # GPS feature extraction (if enabled)
        if par.enable_gps:
            self.gps_fc = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )
        else:
            self.gps_fc = nn.Identity()

        # Fusion module
        self.fusion_module = AdaptiveGatedAttentionFusion(
            rgb_channels=512,
            depth_channels=256 if par.enable_depth else 0,
            lidar_channels=(128 if par.enable_imu else 0) + (128 if par.enable_gps else 0)
        )

        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=512 + (256 if par.enable_depth else 0) + (128 if par.enable_imu else 0) + (128 if par.enable_gps else 0),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(hidden_size, 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_03, x_02, x_depth, x_imu, x_gps = x

        # Stack left and right RGB images
        B, seq_len, C, H, W = x_03.shape
        x_stacked = torch.cat((x_03, x_02), dim=2)  # [B, seq_len, 6, H, W]
        x_stacked = x_stacked.view(B * seq_len, 6, H, W)
        rgb_features = self.rgb_conv(x_stacked)
        rgb_features = rgb_features.view(B * seq_len, -1)
        rgb_features = self.rgb_fc(rgb_features)
        rgb_features = rgb_features.view(B, seq_len, -1)

        # Process Depth
        if par.enable_depth:
            B, seq_len, C_depth, H_depth, W_depth = x_depth.shape
            x_depth = x_depth.view(B * seq_len, C_depth, H_depth, W_depth)
            depth_features = self.depth_conv(x_depth)
            depth_features = depth_features.view(B * seq_len, -1)
            depth_features = self.depth_fc(depth_features)
            depth_features = depth_features.view(B, seq_len, -1)
        else:
            depth_features = torch.zeros(B, seq_len, 0, device=x_stacked.device)

        # Process IMU
        if par.enable_imu:
            B, seq_len, _ = x_imu.shape
            x_imu = x_imu.view(B * seq_len, -1)
            imu_features = self.imu_fc(x_imu)
            imu_features = imu_features.view(B, seq_len, -1)
        else:
            imu_features = torch.zeros(B, seq_len, 0, device=x_stacked.device)

        # Process GPS
        if par.enable_gps:
            B, seq_len, _ = x_gps.shape
            x_gps = x_gps.view(B * seq_len, -1)
            gps_features = self.gps_fc(x_gps)
            gps_features = gps_features.view(B, seq_len, -1)
        else:
            gps_features = torch.zeros(B, seq_len, 0, device=x_stacked.device)

        # Combine IMU and GPS features
        lidar_features = torch.cat((imu_features, gps_features), dim=-1)

        # Fuse features
        combined_features = torch.zeros(B, seq_len, self.fusion_module.total_channels, device=x_stacked.device)
        for t in range(seq_len):
            fused = self.fusion_module(
                rgb_features[:, t, :],
                depth_features[:, t, :],
                lidar_features[:, t, :]
            )
            combined_features[:, t, :] = fused

        # RNN
        out, _ = self.rnn(combined_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)

        out = to_ned_pose(out, is_absolute=False)
        return out

    def compute_absolute_poses(self, relative_poses):
        absolute_poses = integrate_relative_poses(relative_poses)
        return absolute_poses

    def get_loss(self, x, y):
        predicted = self.forward(x)  # [batch_size, seq_len, 6]
        y = y[:, 1:, :]  # [batch_size, seq_len-1, 6]
        predicted_relative = predicted[:, 1:, :]  # [batch_size, seq_len-1, 6]

        # Base losses
        angle_loss = F.mse_loss(predicted_relative[:, :, :3], y[:, :, :3])
        translation_loss = F.mse_loss(predicted_relative[:, :, 3:], y[:, :, 3:])
        l2_lambda = par.l2_lambda
        l2_loss = l2_lambda * sum(torch.norm(param) for param in self.parameters() if param.requires_grad)

        # Load statistics for normalization
        with open("datainfo/dataset_stats.pickle", 'rb') as f:
            stats = pickle.load(f)

        # Depth consistency loss (normalize predictions)
        if par.enable_depth:
            depth_data = x[2]
            depth_diff = depth_data[:, 1:, :, :, :] - depth_data[:, :-1, :, :, :]
            depth_diff_mean = depth_diff.mean(dim=[2, 3, 4])
            pred_trans_z = predicted_relative[:, :, 5]
            # Normalize pred_trans_z to match depth_data's scale
            pred_trans_z_norm = (pred_trans_z - stats['depth_mean'] / stats['depth_max']) / (stats['depth_std'] / stats['depth_max'])
            depth_loss = F.mse_loss(pred_trans_z_norm, depth_diff_mean)
        else:
            depth_loss = torch.tensor(0.0, device=predicted.device)

        # GPS loss (normalize predictions)
        if par.enable_gps:
            gps_data = x[4]
            predicted_absolute = self.compute_absolute_poses(predicted)
            predicted_absolute = predicted_absolute[:, 1:, :]
            pred_pos = predicted_absolute[:, :, 3:]
            # Normalize pred_pos to match gps_data's scale
            pred_pos_norm = (pred_pos - stats['gps_pos_mean']) / stats['gps_pos_std']
            gps_loss = F.mse_loss(pred_pos_norm, gps_data[:, :, :3])
        else:
            gps_loss = torch.tensor(0.0, device=predicted.device)

        # Combine losses
        base_loss = par.R_factor * angle_loss + par.T_factor * translation_loss + l2_loss
        total_loss = base_loss + par.depth_consistency_loss_weight * depth_loss + par.gps_loss_weight * gps_loss
        return total_loss

    def step(self, x, y, optimizer, scaler=None):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            loss = self.get_loss(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

if __name__ == "__main__":
    model = StereoAdaptiveVO(img_h=300, img_w=400, batch_norm=True, input_channels=6, hidden_size=512, num_layers=2)
    print(model)
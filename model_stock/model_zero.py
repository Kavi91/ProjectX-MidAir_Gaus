import torch
import torch.nn as nn
import os
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None, dropout=0):
    if padding is None:
        padding = (kernel_size-1)//2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

class CrossModalAttention(nn.Module):
    def __init__(self, query_channels, key_value_channels, num_heads=8, reduction=16):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.query_channels = query_channels
        self.key_value_channels = key_value_channels
        self.head_dim = query_channels // num_heads
        assert self.head_dim * num_heads == query_channels, "query_channels must be divisible by num_heads"

        # Query projection (based on query modality)
        self.query = nn.Conv2d(query_channels, query_channels, kernel_size=1)
        # Key/Value projections (based on key/value modality)
        self.key = nn.Conv2d(key_value_channels, query_channels, kernel_size=1)  # Project to query space
        self.value = nn.Conv2d(key_value_channels, query_channels, kernel_size=1)
        self.scale = (self.head_dim) ** -0.5

        # Output projection
        self.out_proj = nn.Conv2d(query_channels, query_channels, kernel_size=1)

    def forward(self, query_features, key_features, value_features):
        B, C_q, H, W = query_features.size()
        B, C_kv, H, W = key_features.size()
        assert C_q == self.query_channels, f"Query channels mismatch: expected {self.query_channels}, got {C_q}"
        assert C_kv == self.key_value_channels, f"Key/Value channels mismatch: expected {self.key_value_channels}, got {C_kv}"

        # Project features
        Q = self.query(query_features).view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        K = self.key(key_features).view(B, self.num_heads, self.head_dim, H * W)      # [B, num_heads, head_dim, H*W]
        V = self.value(value_features).view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]

        # Attention scores
        scores = torch.einsum('bnhd,bnkd->bnhk', Q, K) * self.scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.einsum('bnhk,bnkd->bnhd', attn, V)  # [B, num_heads, H*W, head_dim]
        out = out.view(B, self.query_channels, H, W)  # [B, C_q, H, W]

        # Output projection
        out = self.out_proj(out)
        return out

class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels=0, reduction=16):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.total_channels = rgb_channels + depth_channels + lidar_channels
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.lidar_channels = lidar_channels
        self.temperature = 2.0  # Temperature for softmax to amplify differences
        self.min_confidence = 0.05  # Minimum confidence to prevent complete exclusion
        self.last_alpha = None  # To store the last computed alpha values for logging

        # Cross-modal attention for each modality pair
        self.rgb_to_depth = CrossModalAttention(query_channels=rgb_channels, key_value_channels=depth_channels, num_heads=8, reduction=reduction) if depth_channels > 0 else None
        self.rgb_to_lidar = CrossModalAttention(query_channels=rgb_channels, key_value_channels=lidar_channels, num_heads=8, reduction=reduction) if lidar_channels > 0 else None
        self.depth_to_rgb = CrossModalAttention(query_channels=depth_channels, key_value_channels=rgb_channels, num_heads=8, reduction=reduction) if rgb_channels > 0 else None
        self.depth_to_lidar = CrossModalAttention(query_channels=depth_channels, key_value_channels=lidar_channels, num_heads=8, reduction=reduction) if lidar_channels > 0 else None
        self.lidar_to_rgb = CrossModalAttention(query_channels=lidar_channels, key_value_channels=rgb_channels, num_heads=8, reduction=reduction) if rgb_channels > 0 and lidar_channels > 0 else None
        self.lidar_to_depth = CrossModalAttention(query_channels=lidar_channels, key_value_channels=depth_channels, num_heads=8, reduction=reduction) if depth_channels > 0 and lidar_channels > 0 else None

        # SE blocks for channel-wise attention
        self.se_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels, rgb_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(rgb_channels // reduction, rgb_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.se_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depth_channels, depth_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(depth_channels // reduction, depth_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.se_lidar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(lidar_channels, lidar_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(lidar_channels // reduction, lidar_channels, kernel_size=1),
            nn.Sigmoid()
        ) if lidar_channels > 0 else None

        # Gating mechanism (fallback if uncertainty is not used)
        self.gate = nn.Sequential(
            nn.Conv2d(self.total_channels + 3, 3, kernel_size=1),
            nn.Sigmoid()
        )

        # Final fusion
        self.fusion = nn.Conv2d(self.total_channels, self.total_channels, kernel_size=1)

    def forward(self, rgb_features, depth_features, lidar_features=None, modality_covars=None):
        # Ensure spatial dimensions match
        if rgb_features.size()[2:] != depth_features.size()[2:]:
            depth_features = F.interpolate(depth_features, size=rgb_features.size()[2:], mode='bilinear', align_corners=False)
        if lidar_features is not None and rgb_features.size()[2:] != lidar_features.size()[2:]:
            lidar_features = F.interpolate(lidar_features, size=rgb_features.size()[2:], mode='bilinear', align_corners=False)

        # Cross-modal attention
        # RGB attends to depth and LiDAR
        rgb_cross = rgb_features
        if self.rgb_to_depth is not None:
            rgb_cross = rgb_cross + self.rgb_to_depth(rgb_features, depth_features, depth_features)
        if self.rgb_to_lidar is not None and lidar_features is not None:
            rgb_cross = rgb_cross + self.rgb_to_lidar(rgb_features, lidar_features, lidar_features)

        # Depth attends to RGB and LiDAR
        depth_cross = depth_features
        if self.depth_to_rgb is not None:
            depth_cross = depth_cross + self.depth_to_rgb(depth_features, rgb_features, rgb_features)
        if self.depth_to_lidar is not None and lidar_features is not None:
            depth_cross = depth_cross + self.depth_to_lidar(depth_features, lidar_features, lidar_features)

        # LiDAR attends to RGB and depth
        lidar_cross = lidar_features
        if lidar_features is not None:
            if self.lidar_to_rgb is not None:
                lidar_cross = lidar_cross + self.lidar_to_rgb(lidar_features, rgb_features, rgb_features)
            if self.lidar_to_depth is not None:
                lidar_cross = lidar_cross + self.lidar_to_depth(lidar_features, depth_features, depth_features)

        # SE blocks (channel-wise attention)
        rgb_refined = rgb_cross * self.se_rgb(rgb_cross)
        depth_refined = depth_cross * self.se_depth(depth_cross)
        lidar_refined = lidar_cross * self.se_lidar(lidar_cross) if lidar_cross is not None else None

        # Normalize features to have unit norm
        rgb_norm = torch.norm(rgb_refined, p=2, dim=1, keepdim=True)
        rgb_refined = rgb_refined / (rgb_norm + 1e-6)  # Avoid division by zero
        depth_norm = torch.norm(depth_refined, p=2, dim=1, keepdim=True)
        depth_refined = depth_refined / (depth_norm + 1e-6)
        if lidar_refined is not None:
            lidar_norm = torch.norm(lidar_refined, p=2, dim=1, keepdim=True)
            lidar_refined = lidar_refined / (lidar_norm + 1e-6)

        # Compute confidence scores using Bayesian uncertainties if provided
        if modality_covars is not None:
            confidences = []
            if 'rgb' in modality_covars:
                rgb_logvar = modality_covars['rgb']  # [batch_size, seq_len, 1]
                rgb_conf = -rgb_logvar  # Higher log-variance -> lower confidence
                confidences.append(rgb_conf)
            else:
                confidences.append(torch.zeros_like(depth_refined[:, 0, 0, 0]))
            if 'depth' in modality_covars:
                depth_logvar = modality_covars['depth']
                depth_conf = -depth_logvar
                confidences.append(depth_conf)
            else:
                confidences.append(torch.zeros_like(depth_refined[:, 0, 0, 0]))
            if 'lidar' in modality_covars and lidar_refined is not None:
                lidar_logvar = modality_covars['lidar']
                lidar_conf = -lidar_logvar
                confidences.append(lidar_conf)
            else:
                confidences.append(torch.zeros_like(depth_refined[:, 0, 0, 0]) if lidar_refined is not None else None)

            # Remove None entries and stack confidences
            confidences = [c for c in confidences if c is not None]
            confidences = torch.stack(confidences, dim=1)  # [batch_size, num_modalities, seq_len, 1]
            confidences = confidences / self.temperature  # Apply temperature scaling
            alpha = torch.softmax(confidences, dim=1)  # [batch_size, num_modalities, seq_len, 1]
            alpha = alpha * (1 - self.min_confidence * confidences.size(1)) + self.min_confidence  # Apply minimum confidence
            alpha = alpha.unsqueeze(-1)  # [batch_size, num_modalities, seq_len, 1, 1]
            self.last_alpha = alpha  # Store for logging

            # Reshape alpha to match the flattened batch dimension of modality features
            batch_size, num_modalities, seq_len, _, _ = alpha.shape
            alpha = alpha.view(batch_size * seq_len, num_modalities, 1, 1, 1)  # [batch_size * seq_len, num_modalities, 1, 1, 1]
        else:
            # Fallback to quality metrics
            depth_sparsity = (depth_features == 0).float().mean(dim=[1, 2, 3], keepdim=True)
            rgb_variance = rgb_features.var(dim=[1, 2, 3], keepdim=True)
            lidar_sparsity = (lidar_features == 0).float().mean(dim=[1, 2, 3], keepdim=True) if lidar_features is not None else torch.zeros_like(depth_sparsity)
            quality_metrics = torch.cat((depth_sparsity, rgb_variance, lidar_sparsity), dim=1)
            quality_metrics = quality_metrics.expand(-1, -1, rgb_features.size(2), rgb_features.size(3))

            # Concatenate features for gating
            features_to_concat = [rgb_refined, depth_refined]
            if lidar_refined is not None:
                features_to_concat.append(lidar_refined)
            combined = torch.cat(features_to_concat, dim=1)
            combined_with_metrics = torch.cat((combined, quality_metrics), dim=1)

            # Gating
            alpha = self.gate(combined_with_metrics)  # [batch_size * seq_len, 3, H, W]
            self.last_alpha = alpha  # Store for logging

        # Apply gates
        gate_idx = 0
        rgb_weighted = rgb_refined * alpha[:, gate_idx:gate_idx+1]  # [batch_size * seq_len, 2048, H, W] * [batch_size * seq_len, 1, 1, 1]
        gate_idx += 1
        depth_weighted = depth_refined * alpha[:, gate_idx:gate_idx+1]
        gate_idx += 1
        lidar_weighted = lidar_refined * alpha[:, gate_idx:gate_idx+1] if lidar_refined is not None else None

        # Final fusion
        weighted_features = [rgb_weighted, depth_weighted]
        if lidar_weighted is not None:
            weighted_features.append(lidar_weighted)
        fused_features = self.fusion(torch.cat(weighted_features, dim=1))
        return fused_features

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO, self).__init__()
        self.batchNorm = batchNorm
        self.clip = par.clip
        self.conv1   = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        fused_feature_size = int(np.prod(__tmp.size())) * 2
        
        self.rnn = nn.LSTM(
            input_size=fused_feature_size, 
            hidden_size=par.rnn_hidden_size, 
            num_layers=2, 
            dropout=par.rnn_dropout_between, 
            batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, return_uncertainty=False):
        # DeepVO expects x to be a tuple of (x_03, x_02), but StereoAdaptiveVO passes (x_03, x_02, x_depth, x_lidar)
        # We'll only use x_03 and x_02 here
        if isinstance(x, tuple) and len(x) >= 2:
            x_03, x_02 = x[:2]
        else:
            raise ValueError("Input x must be a tuple of at least (x_03, x_02)")
        
        x_03 = torch.cat((x_03[:, :-1], x_03[:, 1:]), dim=2)
        x_02 = torch.cat((x_02[:, :-1], x_02[:, 1:]), dim=2)
        
        batch_size = x_03.size(0)
        seq_len = x_03.size(1)
        
        x_03 = x_03.view(batch_size * seq_len, x_03.size(2), x_03.size(3), x_03.size(4))
        x_02 = x_02.view(batch_size * seq_len, x_02.size(2), x_03.size(3), x_03.size(4))
        
        features_03 = self.encode_image(x_03)
        features_02 = self.encode_image(x_02)
        
        fused_features = torch.cat((features_03, features_02), dim=1)
        fused_features = fused_features.view(batch_size, seq_len, -1)
        
        out, hc = self.rnn(fused_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        
        if return_uncertainty:
            # DeepVO doesn't compute uncertainty, so return None for pose_covar and modality_covars
            return out, None, None
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

class StereoAdaptiveVO(DeepVO):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(StereoAdaptiveVO, self).__init__(imsize1, imsize2, batchNorm)
        
        self.depth_conv1 = conv(self.batchNorm, 2, 32, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.depth_conv2 = conv(self.batchNorm, 32, 64, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.depth_conv3 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.depth_conv4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        
        # LiDAR processing with pretrained ResNet18
        if par.enable_lidar:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            pretrained_weights = resnet.conv1.weight.clone()
            resnet.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight[:, :3, :, :] = pretrained_weights
                for i in range(3, 10):
                    resnet.conv1.weight[:, i, :, :] = pretrained_weights[:, i % 3, :, :]
            self.lidar_resnet = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )
            self.lidar_adapt = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        else:
            self.lidar_resnet = None
            self.lidar_adapt = None
        
        # Compute feature sizes
        __tmp_rgb = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp_rgb = self.encode_image(__tmp_rgb)
        rgb_channels = __tmp_rgb.size(1) * 2 if par.enable_rgb else 0  # 2048
        rgb_h, rgb_w = __tmp_rgb.size(2), __tmp_rgb.size(3)
        rgb_feature_size = int(np.prod(__tmp_rgb.size())) * 2
        print(f"Computed fused RGB feature size: {rgb_feature_size} (channels: {rgb_channels}, H: {rgb_h}, W: {rgb_w})")
        
        __tmp_depth = Variable(torch.zeros(1, 2, imsize1, imsize2))
        __tmp_depth = self.encode_depth(__tmp_depth)
        depth_channels = __tmp_depth.size(1) if par.enable_depth else 0  # 256
        depth_h, depth_w = __tmp_depth.size(2), __tmp_depth.size(3)
        depth_feature_size = int(np.prod(__tmp_depth.size()))
        print(f"Computed depth feature size: {depth_feature_size} (channels: {depth_channels}, H: {depth_h}, W: {depth_w})")
        
        lidar_channels = 0
        if par.enable_lidar:
            __tmp_lidar = Variable(torch.zeros(1, 10, 64, 900))
            __tmp_lidar = self.encode_lidar(__tmp_lidar)
            lidar_channels = __tmp_lidar.size(1)  # 128
            lidar_h, lidar_w = __tmp_lidar.size(2), __tmp_lidar.size(3)
            lidar_feature_size = int(np.prod(__tmp_lidar.size()))
            print(f"Computed LiDAR feature size: {lidar_feature_size} (channels: {lidar_channels}, H: {lidar_h}, W: {lidar_w})")
        else:
            lidar_h, lidar_w = rgb_h, rgb_w  # Match RGB dimensions for fusion
        
        self.fusion_module = AdaptiveGatedAttentionFusion(rgb_channels, depth_channels, lidar_channels)
        
        __tmp_rgb_fused = torch.zeros(1, rgb_channels, rgb_h, rgb_w)
        __tmp_depth_resized = torch.zeros(1, depth_channels, rgb_h, rgb_w)
        __tmp_lidar_resized = torch.zeros(1, lidar_channels, rgb_h, rgb_w) if par.enable_lidar else None
        __tmp_fused = self.fusion_module(__tmp_rgb_fused, __tmp_depth_resized, __tmp_lidar_resized)
        rnn_input_size = int(np.prod(__tmp_fused.size()))
        print(f"Adjusted LSTM input size (RGB + depth + LiDAR after fusion): {rnn_input_size}")
        
        self.rnn = nn.LSTM(
            input_size=rnn_input_size, 
            hidden_size=par.rnn_hidden_size, 
            num_layers=2, 
            dropout=par.rnn_dropout_between, 
            batch_first=True)
        
        # Uncertainty heads
        self.rgb_uncertainty_head = nn.Conv2d(rgb_channels, 1, kernel_size=1) if par.enable_rgb else None
        self.depth_uncertainty_head = nn.Conv2d(depth_channels, 1, kernel_size=1) if par.enable_depth else None
        self.lidar_uncertainty_head = nn.Conv2d(lidar_channels, 1, kernel_size=1) if par.enable_lidar else None
        self.pose_uncertainty_head = nn.Linear(par.rnn_hidden_size, 6)  # Output log-variances for 6-DoF pose

        for m in self.rnn.modules():
            if isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

    def encode_depth(self, x):
        out_conv2 = self.depth_conv2(self.depth_conv1(x))
        out_conv3 = self.depth_conv3(out_conv2)
        out_conv4 = self.depth_conv4(out_conv3)
        return out_conv4

    def encode_lidar(self, x):
        if not par.enable_lidar:
            return None
        out = self.lidar_resnet(x)  # [batch_size * (new_seq_len + 1), 512, H', W']
        out = self.lidar_adapt(out)  # [batch_size * new_seq_len, 128, H', W']
        return out

    def forward(self, x, return_uncertainty=False):
        x_03, x_02, x_depth, x_lidar = x
        
        if x_03.size(1) < 2 or x_02.size(1) < 2 or x_depth.size(1) < 2 or (par.enable_lidar and x_lidar.size(1) < 2):
            raise ValueError(f"Sequence length too short: RGB_03={x_03.size(1)}, RGB_02={x_02.size(1)}, Depth={x_depth.size(1)}, LiDAR={x_lidar.size(1)}")
        
        x_03 = torch.cat((x_03[:, :-1], x_03[:, 1:]), dim=2)
        x_02 = torch.cat((x_02[:, :-1], x_02[:, 1:]), dim=2)
        new_seq_len = x_03.size(1)
        
        x_depth = torch.cat((x_depth[:, :-1], x_depth[:, 1:]), dim=2)
        if x_depth.size(1) != new_seq_len:
            print(f"Warning: Adjusting depth sequence length from {x_depth.size(1)} to {new_seq_len}")
            if x_depth.size(1) > new_seq_len:
                x_depth = x_depth[:, :new_seq_len]
            else:
                padding = torch.zeros(x_depth.size(0), new_seq_len - x_depth.size(1), 
                                      x_depth.size(2), x_depth.size(3), x_depth.size(4), device=x_depth.device)
                x_depth = torch.cat((x_depth, padding), dim=1)
        
        batch_size = x_03.size(0)
        x_03 = x_03.view(batch_size * new_seq_len, x_03.size(2), x_03.size(3), x_03.size(4))
        x_02 = x_02.view(batch_size * new_seq_len, x_02.size(2), x_02.size(3), x_02.size(4))
        x_depth = x_depth.view(batch_size * new_seq_len, x_depth.size(2), x_depth.size(3), x_depth.size(4))
        
        features_03 = self.encode_image(x_03)
        features_02 = self.encode_image(x_02)
        rgb_features = torch.cat((features_03, features_02), dim=1)
        
        depth_features = self.encode_depth(x_depth)
        
        if par.enable_lidar:
            x_lidar = x_lidar[:, :new_seq_len + 1].view(batch_size * (new_seq_len + 1), x_lidar.size(2), x_lidar.size(3), x_lidar.size(4))
            lidar_features = self.encode_lidar(x_lidar[:batch_size * new_seq_len])
        else:
            lidar_features = None
        
        # Debug shapes
        print(f"rgb_features shape: {rgb_features.shape}")
        print(f"depth_features shape: {depth_features.shape}")
        if lidar_features is not None:
            print(f"lidar_features shape: {lidar_features.shape}")
        
        # Compute per-modality uncertainties (log-variances)
        modality_covars = {}
        if self.rgb_uncertainty_head is not None:
            rgb_logvar = self.rgb_uncertainty_head(rgb_features)  # [batch_size*seq_len, 1, H, W]
            rgb_logvar = rgb_logvar.mean(dim=[2, 3])  # [batch_size*seq_len, 1]
            modality_covars['rgb'] = rgb_logvar.view(batch_size, new_seq_len, -1)
        if self.depth_uncertainty_head is not None:
            depth_logvar = self.depth_uncertainty_head(depth_features)
            depth_logvar = depth_logvar.mean(dim=[2, 3])
            modality_covars['depth'] = depth_logvar.view(batch_size, new_seq_len, -1)
        if self.lidar_uncertainty_head is not None and lidar_features is not None:
            lidar_logvar = self.lidar_uncertainty_head(lidar_features)
            lidar_logvar = lidar_logvar.mean(dim=[2, 3])
            modality_covars['lidar'] = lidar_logvar.view(batch_size, new_seq_len, -1)
        
        # Debug modality covariances
        for modality, covar in modality_covars.items():
            print(f"{modality}_covar shape: {covar.shape}")
        
        # Fusion with uncertainty-based weights
        fused_features = self.fusion_module(rgb_features, depth_features, lidar_features, modality_covars)
        fused_features = fused_features.view(batch_size, new_seq_len, -1)
        
        # RNN and Pose Prediction
        out, hc = self.rnn(fused_features)
        out = self.rnn_drop_out(out)
        pose = self.linear(out)  # [batch_size, seq_len, 6]
        
        if not return_uncertainty:
            return pose
        
        # Compute pose uncertainty
        pose_features = self.rnn_drop_out(out)
        pose_logvar = self.pose_uncertainty_head(pose_features)  # [batch_size, seq_len, 6]
        pose_logvar = torch.clamp(pose_logvar, min=-10, max=10)  # Prevent numerical issues
        pose_covar = torch.diag_embed(torch.exp(pose_logvar))  # [batch_size, seq_len, 6, 6]
        
        return pose, pose_covar, modality_covars

    def get_loss(self, x, y):
        pose, pose_covar, _ = self.forward(x, return_uncertainty=True)
        y = y[:, 1:, :]  # [batch_size, seq_len, 6]
        
        # Negative Log-Likelihood Loss
        diff = y - pose  # [batch_size, seq_len, 6]
        logvar = torch.log(torch.diagonal(pose_covar, dim1=-2, dim2=-1))  # [batch_size, seq_len, 6]
        precision = torch.exp(-logvar)  # [batch_size, seq_len, 6]
        
        # Apply weighting: scale angle errors (first 3 dimensions) by 100
        weights = torch.ones_like(diff)
        weights[:, :, :3] = 100.0  # Weight angle dimensions
        loss = 0.5 * (weights * precision * (diff**2) + logvar)
        return loss.mean()

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss
import torch
import torch.nn as nn
import os
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
import torch.nn.functional as F

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None, dropout=0):
    # Default padding to (kernel_size-1)//2 if not specified
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

class AdaptiveGatedAttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, lidar_channels=0, reduction=16):
        super(AdaptiveGatedAttentionFusion, self).__init__()
        self.total_channels = rgb_channels + depth_channels + lidar_channels
        self.gate = nn.Sequential(
            nn.Conv2d(self.total_channels + 3, 3, kernel_size=1),
            nn.Sigmoid()
        )
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
        self.fusion = nn.Conv2d(self.total_channels, self.total_channels, kernel_size=1)

    def forward(self, rgb_features, depth_features, lidar_features=None):
        if rgb_features.size()[2:] != depth_features.size()[2:]:
            depth_features = F.interpolate(depth_features, size=rgb_features.size()[2:], mode='bilinear', align_corners=False)
        if lidar_features is not None and rgb_features.size()[2:] != lidar_features.size()[2:]:
            lidar_features = F.interpolate(lidar_features, size=rgb_features.size()[2:], mode='bilinear', align_corners=False)
        
        depth_sparsity = (depth_features == 0).float().mean(dim=[1, 2, 3], keepdim=True)
        rgb_variance = rgb_features.var(dim=[1, 2, 3], keepdim=True)
        lidar_sparsity = (lidar_features == 0).float().mean(dim=[1, 2, 3], keepdim=True) if lidar_features is not None else torch.zeros_like(depth_sparsity)
        quality_metrics = torch.cat((depth_sparsity, rgb_variance, lidar_sparsity), dim=1)
        quality_metrics = quality_metrics.expand(-1, -1, rgb_features.size(2), rgb_features.size(3))
        
        features_to_concat = [rgb_features, depth_features]
        if lidar_features is not None:
            features_to_concat.append(lidar_features)
        combined = torch.cat(features_to_concat, dim=1)
        combined_with_metrics = torch.cat((combined, quality_metrics), dim=1)
        gate_weights = self.gate(combined_with_metrics)
        rgb_gate, depth_gate, lidar_gate = gate_weights[:, 0:1], gate_weights[:, 1:2], gate_weights[:, 2:3]

        rgb_refined = rgb_features * self.se_rgb(rgb_features)
        depth_refined = depth_features * self.se_depth(depth_features)
        lidar_refined = lidar_features * self.se_lidar(lidar_features) if lidar_features is not None else None

        rgb_weighted = rgb_refined * rgb_gate
        depth_weighted = depth_refined * depth_gate
        lidar_weighted = lidar_refined * lidar_gate if lidar_refined is not None else None

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

    def forward(self, x):
        x_03, x_02 = x
        
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
        
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss

class StereoAdaptiveVO(DeepVO):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(StereoAdaptiveVO, self).__init__(imsize1, imsize2, batchNorm)
        
        self.depth_conv1 = conv(self.batchNorm, 2, 32, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.depth_conv2 = conv(self.batchNorm, 32, 64, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.depth_conv3 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.depth_conv4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        
        # LiDAR processing (only if enabled)
        if par.enable_lidar:
            self.lidar_conv1 = conv(self.batchNorm, 10, 32, kernel_size=3, stride=(1,2), padding=(1,0), dropout=par.conv_dropout[0])
            self.lidar_conv2 = conv(self.batchNorm, 32, 64, kernel_size=3, stride=(1,2), padding=(1,0), dropout=par.conv_dropout[1])
            self.lidar_conv3 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=(1,2), padding=(1,0), dropout=par.conv_dropout[2])
            self.lidar_conv4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=(2,2), padding=(1,0), dropout=par.conv_dropout[4])
            self.lidar_conv5 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=(2,2), padding=(1,0), dropout=par.conv_dropout[5])
            self.lidar_conv6 = conv(self.batchNorm, 512, 128, kernel_size=1, stride=1, padding=(1,0), dropout=par.conv_dropout[6])
        else:
            self.lidar_conv1 = self.lidar_conv2 = self.lidar_conv3 = self.lidar_conv4 = self.lidar_conv5 = self.lidar_conv6 = None
        
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
        out_conv2 = self.lidar_conv2(self.lidar_conv1(x))
        out_conv3 = self.lidar_conv3(out_conv2)
        out_conv4 = self.lidar_conv4(out_conv3)
        out_conv5 = self.lidar_conv5(out_conv4)
        out_conv6 = self.lidar_conv6(out_conv5)
        return out_conv6

    def forward(self, x):
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
        
        fused_features = self.fusion_module(rgb_features, depth_features, lidar_features)
        fused_features = fused_features.view(batch_size, new_seq_len, -1)
        
        out, hc = self.rnn(fused_features)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss
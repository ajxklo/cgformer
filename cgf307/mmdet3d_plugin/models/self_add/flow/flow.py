import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d

class FeatureWarping(nn.Module):
    def __init__(self, in_channels):
        super(FeatureWarping, self).__init__()
        self.offset_conv = nn.Conv2d(2, 18, kernel_size=3, padding=1)  # 计算偏移量
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, feature_t0, flow):
        """
        :param feature_t0: [batch, C, H_feat, W_feat]  # 第0帧的特征
        :param flow: [batch, 2, H_img, W_img]  # 原始图像尺度的光流
        :return: 变换后的 t1 特征
        """
        # 计算可变形卷积的偏移量 (offsets)
        offsets = self.offset_conv(flow)  # [batch, 18, H_feat, W_feat]

        # 进行 Deformable Convolution
        feature_t1 = self.deform_conv(feature_t0, offsets)

        return feature_t1
    # 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        attn = F.relu(self.conv1(x))
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)  # Apply sigmoid to get attention map (0-1)
        return attn

    # 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        avg_pool = avg_pool.view(avg_pool.size(0), -1)  # Flatten
        fc1_out = F.relu(self.fc1(avg_pool))
        channel_attn = self.sigmoid(self.fc2(fc1_out))  # Apply sigmoid to get channel attention
        channel_attn = channel_attn.view(channel_attn.size(0), channel_attn.size(1), 1,
                                         1)  # Reshape to match feature map
        return channel_attn

    # 光流图尺寸调整函数
def resize_optical_flow(flow, target_size=(48, 160)):
    # flow: [H, W, 2], target_size: (H', W')
    flow_resized = F.interpolate(flow.permute(0,3, 1, 2), size=target_size, mode='bilinear',
                                 align_corners=False)
    return flow_resized.squeeze(0).permute(1, 2, 0)  # 返回[H', W', 2]

class OpticalFlowAlignmentWithAttention(nn.Module):
    def __init__(self):
        super(OpticalFlowAlignmentWithAttention, self).__init__()
        self.spatial_attention = SpatialAttention(2)  # 处理光流图
        self.channel_attention = ChannelAttention(640)  # 假设图像有640个通道
        self.flow = FeatureWarping(640)

    def forward(self, image_0, flow):
        num = image_0.dim()
        if num ==5:
            b,n,c,h,w=image_0.shape
            image_0 = image_0.view(b*n,c,h,w)
        # 调整光流图的尺寸
        flow_resized = resize_optical_flow(flow, target_size=(image_0.shape[2], image_0.shape[3]))

        flow_resized = flow_resized.permute(2,0,1).unsqueeze(0)

        # 获取空间注意力图
        flow_attention = self.spatial_attention(flow_resized)

        # 将空间注意力应用到光流图上
        flow_resized = flow_resized * flow_attention

        # 使用光流图将第0帧图像对齐到第1帧

        warped_image = self.flow(image_0,flow_resized)

        # 获取通道注意力图
        image_attention = self.channel_attention(warped_image)

        # 将通道注意力应用到对齐后的图像
        warped_image = warped_image * image_attention


        return warped_image





import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


# from mmdet3d_plugin.models.self_add.flow.flow_mask import compute_occlusion_mask
#
#
def compute_cosine_similarity(feat1, feat2):
    """计算余弦相似度"""
    cos_sim = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
    return cos_sim.unsqueeze(1)  # 添加通道维度
#
#
# def warp_feature_with_flow(feature, flow):
#     B, C, H, W = feature.shape
#
#     # 生成网格
#     grid_x, grid_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
#     grid_x = grid_x.float().to(feature.device)
#     grid_y = grid_y.float().to(feature.device)
#
#     # 将grid_x, grid_y扩展到批次维度
#     grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # 扩展为 (B, H, W)
#     grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # 扩展为 (B, H, W)
#
#     # 加上光流
#     grid_x = grid_x + flow[:, :, :, 0]  # flow[:, :, :, 0]的尺寸是 (B, H, W)
#     grid_y = grid_y + flow[:, :, :, 1]  # flow[:, :, :, 1]的尺寸是 (B, H, W)
#
#     # 归一化到 [-1, 1] 范围
#     grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
#     grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
#     grid = torch.stack((grid_x, grid_y), dim=-1)  # (B, H, W, 2)
#
#     # 使用grid_sample进行光流扭曲
#     warped_feature = F.grid_sample(feature, grid, mode='bilinear', align_corners=False)
#     return warped_feature
#
# def FlowGuidedWarp(feature, feature_last, flow):
#     """
#     :param feature: 当前帧特征 (B, C, H, W)
#     :param feature_last: 下一帧特征 (B, C, H, W)
#     :param flow: 光流 (B, H, W, 2)
#     :return: warp_feature, weight
#     """
#     B, C, H, W = feature.shape
#
#     # 1. 利用光流对特征进行扭曲
#     warped_feature = warp_feature_with_flow(feature_last, flow[0])
#
#     # 2. 计算 warped_feature 和 feature_last 的余弦相似度
#     similarity = compute_cosine_similarity(warped_feature, feature)
#
#     aggregated_feat = warped_feature * similarity
#     return aggregated_feat,
#
#
# def temporal_feature_aggregation(current_feat, feature_last, flow):
#     """使用光流引导特征聚合"""
#     devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     num = current_feat.dim()
#     if num == 5:
#         B, N, C, H, W = current_feat.shape
#         current_feat = current_feat.view(B * N, C, H, W).to(devices)
#         feature_last = feature_last.view(B * N, C, H, W).to(devices)
#
#     mask = compute_occlusion_mask(flow[0], flow[1])
#
#     aggregated_feat = FlowGuidedWarp(current_feat, feature_last, flow)
#
#     # aggregated_feat = warp_feature * weight + current_feat * (1 - weight)
#     # if num == 5:
#     #     aggregated_feat = aggregated_feat.unsqueeze(0)
#
#     return aggregated_feat
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
        # batch, C, H_feat, W_feat = feature_t0.shape
        #
        # # 调整光流到特征图尺度
        # flow_feat = F.interpolate(flow, size=(H_feat, W_feat), mode="bilinear", align_corners=True)

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


# def warp_feature_with_flow(feature, flow, img_size=(48,160), align_corners=True):
#     """
#     使用光流对特征图进行变换
#     :param feature: 形状 [batch_size, C, H_feat, W_feat]，第 0 帧的特征图
#     :param flow:    形状 [batch_size, 2, H_img, W_img]，原始图像尺寸的光流
#     :param img_size: 原始图像尺寸 (H_img, W_img)
#     :param align_corners: 是否对齐插值
#     :return: 变换后的特征图 [batch_size, C, H_feat, W_feat]
#     """
#     batch_size, C, H_feat, W_feat = feature.shape
#     H_img, W_img = img_size  # 原始图像尺寸
#
#     # 计算特征图相对于图像的缩放比例
#     scale_x = W_feat / W_img
#     scale_y = H_feat / H_img
#
#     # 下采样光流，使其匹配特征图大小
#     flow_feat = F.interpolate(flow, size=(H_feat, W_feat), mode="bilinear", align_corners=align_corners)
#     flow_feat[:, 0, :, :] *= scale_x  # 缩放 x 方向
#     flow_feat[:, 1, :, :] *= scale_y  # 缩放 y 方向
#
#     # 生成特征图级别的网格坐标
#     grid_y, grid_x = torch.meshgrid(torch.arange(H_feat), torch.arange(W_feat), indexing="ij")
#     grid_x = grid_x.float().to(feature.device)
#     grid_y = grid_y.float().to(feature.device)
#
#     # 计算新的坐标
#     new_grid_x = grid_x + flow_feat[:, 0, :, :]
#     new_grid_y = grid_y + flow_feat[:, 1, :, :]
#
#     # 归一化到 [-1, 1]
#     new_grid_x = 2.0 * new_grid_x / (W_feat - 1) - 1.0
#     new_grid_y = 2.0 * new_grid_y / (H_feat - 1) - 1.0
#
#     # 组合网格，并添加 batch 维度
#     new_grid = torch.stack((new_grid_x, new_grid_y), dim=-1)  # [batch_size, H_feat, W_feat, 2]
#
#     # 使用 `grid_sample` 进行双线性插值，得到 t1 时刻的特征图
#     warped_feature = F.grid_sample(feature, new_grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
#
#     return warped_feature



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
        # warped_image = warp_feature_with_flow(image_0,flow_resized)
        # model = FeatureWarping(in_channels=640).cuda()
        warped_image = self.flow(image_0,flow_resized)

        # 获取通道注意力图
        image_attention = self.channel_attention(warped_image)

        # 将通道注意力应用到对齐后的图像
        warped_image = warped_image * image_attention
        # sim = compute_cosine_similarity(warped_image,image_0)
        # warped_image = warped_image * sim

        return warped_image





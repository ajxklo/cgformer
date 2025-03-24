import torch
import torch.nn.functional as F
from torch import nn

class LambdaPredictor(nn.Module):
    def __init__(self, in_channels):
        super(LambdaPredictor, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 预测一个通道的 lambda
        self.sigmoid = nn.Sigmoid()  # 限制在 (0,1) 之间

    def forward(self, Ft, Fwarp):
        x = torch.cat([Ft, Fwarp], dim=1)  # 拼接特征
        lambda_factor = self.sigmoid(self.conv(x))  # 预测每个像素点的 lambda
        return lambda_factor


class NeighborhoodCrossAttention(nn.Module):
    def __init__(self, kernel_size=3,init_lambda=0.5):
        """
        邻域交叉注意机制 (NCA)
        :param kernel_size: 邻域窗口大小 (默认为 3×3)
        """
        super(NeighborhoodCrossAttention, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # 确保窗口能覆盖中心点
        self.lambda_factor = LambdaPredictor(640*2)

    def extract_local_patches(self, x):
        """
        提取邻域窗口
        :param x: [B, C, H, W] 特征图
        :return: [B, C, H, W, k, k] 邻域窗口
        """
        B, C, H, W = x.shape
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        patches = x_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)  # [B, C, H, W, k, k]
        return patches

    def forward(self, Ft, Fwarp, mask):
        """
        :param Ft: [B, C, H, W] 当前帧特征 (Query)
        :param Fwarp: [B, C, H, W] 历史帧对齐后的特征 (Key, Value)
        :param mask: [B, 1, H, W] 遮挡掩码 (1-遮挡, 0-非遮挡)
        :return: [B, C, H, W] 融合后的特征
        """
        num = Ft.dim()
        if num ==5:
            b,n,c,h,w=Ft.shape
            Ft = Ft.view(b*n,c,h,w)
        B, C, H, W = Ft.shape

        # 1. 采样历史帧的邻域特征
        Fwarp_patches = self.extract_local_patches(Fwarp)  # [B, C, H, W, k, k]

        # 2. 计算 Query 与 Key 之间的相似度
        Ft_expanded = Ft.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]
        similarity = (Ft_expanded * Fwarp_patches).sum(dim=1)  # [B, H, W, k, k]
        similarity = similarity / (C ** 0.5)  # 归一化

        # 3. 应用掩码 (mask) 只关注非遮挡区域
        mask_patches = self.extract_local_patches(mask)  # [B, 1, H, W, k, k]
        similarity = similarity.masked_fill(mask_patches.squeeze(1) == 1, float('-inf'))  # 遮挡区域置为 -inf

        # **避免 NaN 传播**
        similarity = similarity.view(B, H, W, -1)
        valid_mask = torch.isfinite(similarity).any(dim=-1, keepdim=True)  # 检查是否有可用数值
        similarity = torch.where(valid_mask, similarity, torch.tensor(-1e8, device=similarity.device))  # 避免全 -inf

        # 4. Softmax 计算注意力权重
        attn_weights = F.softmax(similarity, dim=-1)  # 计算 softmax

        attn_weights = torch.where(valid_mask, attn_weights, torch.zeros_like(attn_weights))  # [B, H, W, k*k]
        attn_weights = attn_weights.view(B, H, W, 3, 3)

        # 5. 计算加权和
        Fwarp_weighted = (Fwarp_patches * attn_weights.unsqueeze(1)).sum(dim=(-1, -2))  # [B, C, H, W]
        lambda_factor = self.lambda_factor(Fwarp_weighted,Ft)
        F_final = lambda_factor * Ft + (1 - lambda_factor) * Fwarp_weighted

        return F_final
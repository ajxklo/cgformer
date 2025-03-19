from torch import nn


class DepthRefinement(nn.Module):
    """
    Residual Inverse Depth Module for precise metric depth recovery.
    用于精确恢复深度的残差逆深度模块。
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化DepthRefinement模块。

        参数:
        - args: 可变位置参数。
        - kwargs: 可变关键字参数。
        """
        super(DepthRefinement, self).__init__(*args, **kwargs)
        self.min_depth = 0.1  # 最小深度，用于限制深度值的范围
        self.max_depth = 80   # 最大深度，用于限制深度值的范围

    def forward(self, pseudo_depth, res_inv_depth, vis=False):
        """
        前向传播计算精确深度。

        参数:
        - pseudo_depth (torch.Tensor): 伪深度估计值，形状为 (B, H, W)。
        - res_inv_depth (torch.Tensor): 残差逆深度估计值，形状为 (B, H, W)。
        - vis (bool): 是否启用可视化权重掩码（默认值为 False）。

        返回:
        - depth (torch.Tensor): 修正后的深度值，形状为 (B, H, W)。
        """
        # 限制伪深度值在 [min_depth, max_depth] 范围内
        pseudo_depth = pseudo_depth.clamp(min=self.min_depth, max=self.max_depth)

        # 对残差逆深度进行缩放，减小影响幅度
        res_inv_depth /= 10

        # 计算伪深度的逆（1 / 伪深度）
        inv_pseudo_depth = 1. / pseudo_depth

        # 如果启用可视化模式
        if vis:
            # 基于伪深度计算权重掩码，深度越远权重越小
            weight_mask = 1 - pseudo_depth / self.max_depth
            # 应用权重掩码调整残差逆深度
            res_inv_depth *= weight_mask

        # 根据伪深度逆和残差逆深度计算最终深度
        depth = 1. / (inv_pseudo_depth + res_inv_depth + 1e-8)  # 防止除零

        return depth

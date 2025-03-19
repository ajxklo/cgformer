import torch
import matplotlib.pyplot as plt


def compute_occlusion_mask(flow_fwd, flow_bwd, threshold=10.0):
    """
    计算遮挡掩码
    :param flow_fwd: 前向光流 (B, H, W, 2)
    :param flow_bwd: 后向光流 (B, H, W, 2)
    :param threshold: 判定遮挡的阈值（默认1.0像素）
    :return: 遮挡掩码 (B, H, W), 1表示遮挡
    """
    B, H, W, _ = flow_fwd.shape

    diff = flow_fwd + flow_bwd  # (B, H, W, 2)
    diff_norm = torch.norm(diff, dim=-1)  # (B, H, W)

    occlusion_mask = (diff_norm > threshold).float()  # (B, H, W)
    # 可视化调试 (可选)
    plt.imshow(occlusion_mask[0].cpu().numpy(), cmap='jet')
    plt.colorbar()
    plt.title(f'Occlusion Mask (Threshold={threshold})')
    plt.show()

    return occlusion_mask

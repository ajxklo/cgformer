import torch
from kornia.geometry.depth import depth_to_3d
import torch.nn.functional as F
from matplotlib import pyplot as plt


def warp(img_r, depth_t, depth_r, pose_t2r, K):
    """
    warp a reference image to the target image.
    Args:
        img_r: [B, 3, H, W], the reference image (where to sample pixels)
        depth_t: [B, 1, H, W], target depth map
        depth_r: [B, 1, H, W], reference depth map
        pose_t2r: [B, 3, 4], relative pose from target image to reference image
        K: [B, 3, 3], camera intrinsic matrix

    Returns:

    """
    B, _, H, W = img_r.size()

    KT = torch.matmul(K.float(), pose_t2r.float())  # [B, 3, 4]

    p_cam = depth_to_3d(depth_t, K)  # [B, 3, H, W], 3D points of target image
    p_cam = p_cam.float()
    p_cam = torch.cat([p_cam, torch.ones(B, 1, H, W).type_as(p_cam)], 1)  # [B, 4, H, W]
    p_ref = torch.matmul(KT, p_cam.view(B, 4, -1))  # =KTP, [B, 3, HxW]
    pix_coords = p_ref[:, :2, :] / (p_ref[:, 2, :].unsqueeze(1) + 1e-7)  # [B, 2, HxW]
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]

    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    projected_img = F.grid_sample(img_r, pix_coords, align_corners=False)
    projected_depth = F.grid_sample(depth_r, pix_coords, mode='nearest',align_corners=False)
    computed_depth = p_ref[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    epsilon = 1e-4  # 容差值
    valid_points = pix_coords.abs().max(dim=-1)[0] <= 1 + epsilon
    valid_mask = valid_points.unsqueeze(1).float()

    # diff1 = depth_t - computed_depth
    # diff2 = depth_r - computed_depth
    plt.subplot(2, 2, 1)
    plt.imshow(depth_t[0, 0].cpu().detach().numpy(), cmap="jet")
    plt.title("Target Depth")

    plt.subplot(2, 2, 2)
    plt.imshow(depth_r[0, 0].cpu().detach().numpy(), cmap="jet")
    plt.title("Reference Depth")

    plt.subplot(2, 2, 3)
    plt.imshow(computed_depth[0, 0].cpu().detach().numpy(), cmap="jet")
    plt.title("computed_depth")

    plt.subplot(2, 2, 4)
    plt.imshow(projected_depth[0, 0].cpu().detach().numpy(), cmap="jet")
    plt.title("projected_depth")
    plt.show()
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(diff1[0, 0].cpu().detach().numpy(), cmap="jet")
    # plt.title("diff1")
    # plt.show()
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(diff2[0, 0].cpu().detach().numpy(), cmap="jet")
    # plt.title("diff2")
    # plt.show()
    # def visualize_depth(depth_map, title="Depth Map"):
    #     depth_map = depth_map.squeeze().detach().cpu().numpy()
    #     plt.imshow(depth_map, cmap="plasma")
    #     plt.colorbar()
    #     plt.title(title)
    #     plt.show()
    #
    # # 可视化 computed_depth 和 projected_depth
    # visualize_depth(computed_depth[0], title="Computed Depth")
    # visualize_depth(projected_depth[0], title="Projected Depth")
    #
    # # 可视化误差
    # visualize_depth((computed_depth - projected_depth).abs()[0], title="Depth Difference")

    return projected_img, projected_depth, computed_depth, valid_mask

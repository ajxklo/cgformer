import torch
from torch import nn

from mmdet3d_plugin.models.self_add.vipocc.util.wrap import warp


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0.).type_as(diff)
    return mean_value


class time_loss(nn.Module):
    def __init__(self, temporal_loss_weight=0.5, geometry_loss_weight=0.5):
        super(time_loss, self).__init__()
        self.temporal_loss_weight = temporal_loss_weight
        self.geometry_loss_weight = geometry_loss_weight

    def forward(self, data):
        depth_r = data['predicted_depth'][0]
        depth_t = data['predicted_depth'][1]
        img_r = data["imgs1"]
        img_t = data["imgs2"]
        pose_t2r = torch.inverse(data["poses"][0]) @ data["poses"][1]
        pose_t2r = pose_t2r[:, :3, :]

        b, n, c, h, w = img_r.shape
        img_r = img_r.view(b * n, c, h, w)
        img_t = img_t.view(b * n, c, h, w)
        K = data["k1"]

        warped_img_r2t, projected_depth, computed_depth, valid_mask = warp(img_r, depth_t, depth_r, pose_t2r, K)

        # Photometric loss (using L2 loss)
        # diff_img = (img_t - warped_img_r2t).pow(2).clamp(0, 1)  # L2 loss
        # # Geometry consistency loss (using L1 loss)
        # diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth + 1e-8)).clamp(0, 1)
        # photometric loss
        diff_img = (img_t - warped_img_r2t).abs().clamp(0, 1)
        diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

        # Apply weight mask
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

        # Compute temporal and geometry consistency loss
        temporal_loss = mean_on_mask(diff_img, valid_mask)
        geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

        # Combine losses with weights
        loss_temp_align = (self.temporal_loss_weight * temporal_loss +
                           self.geometry_loss_weight * geometry_consistency_loss)

        return loss_temp_align
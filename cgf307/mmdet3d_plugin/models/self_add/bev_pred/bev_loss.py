import torch
import torch.nn.functional as F

def bev_CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    Binary Cross-Entropy Loss with support for class weights and ignore index.

    Args:
        pred (torch.Tensor): Predictions (logits), shape (N, C, H, W).
        target (torch.Tensor): Ground truth, same shape as pred or (N, H, W) for class indices.
        class_weights (torch.Tensor or None): Class weights, shape (C,).
        ignore_index (int): Value in target to ignore during loss computation.

    Returns:
        torch.Tensor: Computed average loss.
    """
    # Ensure pred and target are on the same device and type
    device = pred.device
    pred = pred.float()
    target = target.float().to(device)

    # Create a mask to ignore invalid regions
    if ignore_index is not None:
        valid_mask = (target != ignore_index).float()
        target = target * valid_mask  # Zero out ignored targets to prevent impact on loss
    else:
        valid_mask = torch.ones_like(target)

    # Apply class weights if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        weight_map = class_weights.view(1, -1, 1, 1)  # Reshape to broadcast over pred and target
    else:
        weight_map = torch.ones_like(pred)  # Set uniform weight as a tensor

    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight_map, reduction='none')

    # Apply the mask to the loss
    loss = loss * valid_mask

    # Compute the mean loss, accounting for the valid region only
    average_loss = loss.sum() / (valid_mask.sum() + 1e-6)  # Avoid division by zero

    return average_loss

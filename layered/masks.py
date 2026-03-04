import torch


def create_half_masks(latent):
    """
    Creates fixed left/right masks in latent space.

    latent: [B, C, H, W]
    Returns:
        mask_A, mask_B → [1, 1, H, W]
    """

    B, C, H, W = latent.shape
    device = latent.device

    mask_A = torch.zeros((1, 1, H, W), device=device)
    mask_B = torch.zeros((1, 1, H, W), device=device)

    mask_A[:, :, :, : W // 2] = 1.0
    mask_B[:, :, :, W // 2 :] = 1.0

    return mask_A, mask_B

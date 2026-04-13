import torch
import torch.nn.functional as F


def ssim_loss(x, y, c1=0.01 ** 2, c2=0.03 ** 2, window_size=11):
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("x and y must be 4D tensors [B, C, H, W]")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-12)
    return 1.0 - ssim_map.mean()


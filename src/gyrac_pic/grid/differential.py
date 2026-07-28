import torch


def negative_gradient(phi: torch.Tensor, spacing, mask: torch.Tensor) -> torch.Tensor:
    dx, dy, dz = spacing
    padded = torch.nn.functional.pad(torch.where(mask, phi, 0.0), (1, 1, 1, 1, 1, 1))
    ex = -(padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2 * dx)
    ey = -(padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2 * dy)
    ez = -(padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2 * dz)
    return torch.stack((ex, ey, ez), -1) * mask[..., None]

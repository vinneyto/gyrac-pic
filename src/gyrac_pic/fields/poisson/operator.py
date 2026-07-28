import torch


def apply_negative_laplacian(phi: torch.Tensor, interior_mask: torch.Tensor, spacing) -> torch.Tensor:
    dx, dy, dz = spacing
    p = torch.where(interior_mask, phi, 0.0)
    out = torch.zeros_like(phi)
    c = p[1:-1, 1:-1, 1:-1]
    out[1:-1, 1:-1, 1:-1] = ((2*c-p[:-2,1:-1,1:-1]-p[2:,1:-1,1:-1])/dx**2 +
                                      (2*c-p[1:-1,:-2,1:-1]-p[1:-1,2:,1:-1])/dy**2 +
                                      (2*c-p[1:-1,1:-1,:-2]-p[1:-1,1:-1,2:])/dz**2)
    return out * interior_mask

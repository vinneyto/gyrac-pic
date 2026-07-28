from dataclasses import dataclass
import torch


@dataclass
class CylindricalPECDomain:
    radius_m: float
    length_m: float
    grid_shape: tuple[int, int, int]
    boundary_potential: float = 0.0

    @property
    def bounds(self):
        return ((-self.radius_m, self.radius_m), (-self.radius_m, self.radius_m),
                (-self.length_m / 2, self.length_m / 2))

    def interior_mask(self, device, dtype=torch.float32):
        x = torch.linspace(*self.bounds[0], self.grid_shape[0], device=device, dtype=dtype)
        y = torch.linspace(*self.bounds[1], self.grid_shape[1], device=device, dtype=dtype)
        z = torch.linspace(*self.bounds[2], self.grid_shape[2], device=device, dtype=dtype)
        return ((x[:, None, None] ** 2 + y[None, :, None] ** 2 < self.radius_m ** 2) &
                (z[None, None, :].abs() < self.length_m / 2))

    def particle_inside(self, p):
        return ((p[:, 0] ** 2 + p[:, 1] ** 2 < self.radius_m ** 2) &
                (p[:, 2].abs() < self.length_m / 2))

    def boundary_types(self, positions):
        radial = positions[:, 0] ** 2 + positions[:, 1] ** 2 >= self.radius_m ** 2
        positive = positions[:, 2] >= self.length_m / 2
        return ["side_wall" if r else "positive_end_cap" if p else "negative_end_cap"
                for r, p in zip(radial.cpu().tolist(), positive.cpu().tolist())]

    def scene_renderers(self):
        return []

from dataclasses import dataclass
import torch


@dataclass
class BoxDomain:
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]
    grid_shape: tuple[int, int, int]
    boundary_potential: float = 0.0

    @property
    def bounds(self):
        return self.x_bounds, self.y_bounds, self.z_bounds

    def interior_mask(self, device, dtype=torch.float32):
        mask = torch.ones(self.grid_shape, device=device, dtype=torch.bool)
        mask[[0, -1], :, :] = False
        mask[:, [0, -1], :] = False
        mask[:, :, [0, -1]] = False
        return mask

    def particle_inside(self, p):
        return ((p[:, 0] > self.x_bounds[0]) & (p[:, 0] < self.x_bounds[1]) &
                (p[:, 1] > self.y_bounds[0]) & (p[:, 1] < self.y_bounds[1]) &
                (p[:, 2] > self.z_bounds[0]) & (p[:, 2] < self.z_bounds[1]))

    def boundary_types(self, positions):
        return ["box_wall"] * len(positions)

    def scene_renderers(self):
        return []

from dataclasses import dataclass
import torch


@dataclass
class CartesianGrid:
    shape: tuple[int, int, int]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    device: torch.device
    dtype: torch.dtype

    @property
    def spacing(self):
        return tuple((hi - lo) / (n - 1) for (lo, hi), n in zip(self.bounds, self.shape))

    @property
    def cell_volume(self):
        dx, dy, dz = self.spacing
        return dx * dy * dz

    def coordinates(self):
        return tuple(torch.linspace(lo, hi, n, device=self.device, dtype=self.dtype)
                     for (lo, hi), n in zip(self.bounds, self.shape))

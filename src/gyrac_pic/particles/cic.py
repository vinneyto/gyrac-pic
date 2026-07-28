import itertools
import torch


def _coordinates(positions, bounds, shape):
    lower = positions.new_tensor([b[0] for b in bounds])
    spacing = positions.new_tensor([(b[1] - b[0]) / (n - 1) for b, n in zip(bounds, shape)])
    u = (positions - lower) / spacing
    base = torch.floor(u).long()
    maximum = torch.tensor(shape, device=positions.device) - 2
    base = torch.maximum(torch.zeros_like(base), torch.minimum(base, maximum))
    return base, (u - base).clamp(0, 1)


def deposit_charge_cic(positions, charges, shape, bounds, *, alive=None):
    base, frac = _coordinates(positions, bounds, shape)
    if alive is not None:
        charges = charges * alive.to(charges.dtype)
    flat = torch.zeros(shape[0] * shape[1] * shape[2], device=positions.device, dtype=positions.dtype)
    for ox, oy, oz in itertools.product((0, 1), repeat=3):
        offset = positions.new_tensor((ox, oy, oz), dtype=torch.long)
        idx = base + offset
        choice = positions.new_tensor((ox, oy, oz), dtype=positions.dtype)
        weights = torch.where(choice.bool(), frac, 1 - frac).prod(-1)
        linear = idx[:, 0] * shape[1] * shape[2] + idx[:, 1] * shape[2] + idx[:, 2]
        flat.scatter_add_(0, linear, charges * weights)
    spacing = [(b[1] - b[0]) / (n - 1) for b, n in zip(bounds, shape)]
    return flat.reshape(shape) / (spacing[0] * spacing[1] * spacing[2])


def gather_field_cic(field, positions, bounds):
    shape = field.shape[:3]
    base, frac = _coordinates(positions, bounds, shape)
    result = torch.zeros_like(positions)
    for ox, oy, oz in itertools.product((0, 1), repeat=3):
        idx = base + positions.new_tensor((ox, oy, oz), dtype=torch.long)
        choice = positions.new_tensor((ox, oy, oz), dtype=positions.dtype)
        weights = torch.where(choice.bool(), frac, 1 - frac).prod(-1)
        result += weights[:, None] * field[idx[:, 0], idx[:, 1], idx[:, 2]]
    return result

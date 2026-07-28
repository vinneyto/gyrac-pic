import torch
from ..constants import SPEED_OF_LIGHT


def normalized_momentum_squared(momentum, mass):
    """Return |p/(mc)|² without squaring tiny dimensional fp32 values first."""
    normalized_momentum = momentum / (mass * SPEED_OF_LIGHT)
    return (normalized_momentum * normalized_momentum).sum(-1, keepdim=True)


def velocity_from_momentum(momentum, mass):
    gamma = torch.sqrt(1 + normalized_momentum_squared(momentum, mass))
    return momentum / (gamma * mass)


def kinetic_energy(momentum, mass):
    momentum_squared = normalized_momentum_squared(momentum, mass).squeeze(-1)
    gamma = torch.sqrt(1 + momentum_squared)
    # This is algebraically equal to (gamma - 1) m c², but avoids catastrophic
    # cancellation for non-relativistic particles when the simulation uses fp32.
    return (
        momentum_squared
        / (gamma + 1)
        * mass
        * SPEED_OF_LIGHT ** 2
    )


def relativistic_boris_push(position, momentum, electric_field, magnetic_field, charge, mass, dt):
    p_minus = momentum + charge * electric_field * (dt / 2)
    gamma = torch.sqrt(1 + normalized_momentum_squared(p_minus, mass))
    t = charge * magnetic_field * dt / (2 * gamma * mass)
    s = 2 * t / (1 + (t * t).sum(-1, keepdim=True))
    p_prime = p_minus + torch.cross(p_minus, t, dim=-1)
    p_plus = p_minus + torch.cross(p_prime, s, dim=-1)
    new_momentum = p_plus + charge * electric_field * (dt / 2)
    return position + velocity_from_momentum(new_momentum, mass) * dt, new_momentum

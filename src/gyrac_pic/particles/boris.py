import torch
from ..constants import SPEED_OF_LIGHT


def velocity_from_momentum(momentum, mass):
    gamma = torch.sqrt(1 + (momentum * momentum).sum(-1, keepdim=True) / (mass * mass * SPEED_OF_LIGHT ** 2))
    return momentum / (gamma * mass)


def kinetic_energy(momentum, mass):
    normalized_momentum_squared = (momentum * momentum).sum(-1) / (
        mass * mass * SPEED_OF_LIGHT ** 2
    )
    gamma = torch.sqrt(1 + normalized_momentum_squared)
    # This is algebraically equal to (gamma - 1) m c², but avoids catastrophic
    # cancellation for non-relativistic particles when the simulation uses fp32.
    return (
        normalized_momentum_squared
        / (gamma + 1)
        * mass
        * SPEED_OF_LIGHT ** 2
    )


def relativistic_boris_push(position, momentum, electric_field, magnetic_field, charge, mass, dt):
    p_minus = momentum + charge * electric_field * (dt / 2)
    gamma = torch.sqrt(1 + (p_minus * p_minus).sum(-1, keepdim=True) / (mass * mass * SPEED_OF_LIGHT ** 2))
    t = charge * magnetic_field * dt / (2 * gamma * mass)
    s = 2 * t / (1 + (t * t).sum(-1, keepdim=True))
    p_prime = p_minus + torch.cross(p_minus, t, dim=-1)
    p_plus = p_minus + torch.cross(p_prime, s, dim=-1)
    new_momentum = p_plus + charge * electric_field * (dt / 2)
    return position + velocity_from_momentum(new_momentum, mass) * dt, new_momentum

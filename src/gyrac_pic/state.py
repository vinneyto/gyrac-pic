from dataclasses import dataclass
import torch


@dataclass
class ParticleSpeciesState:
    name: str
    positions: torch.Tensor
    momenta: torch.Tensor
    alive: torch.Tensor
    physical_charge_c: float
    physical_mass_kg: float
    macro_weight: float
    macro_charge_c: float
    macro_mass_kg: float


@dataclass
class GridState:
    rho: torch.Tensor
    phi: torch.Tensor
    electric_field: torch.Tensor


@dataclass
class SimulationState:
    step: int
    time_s: float
    species: dict[str, ParticleSpeciesState]
    grid: GridState

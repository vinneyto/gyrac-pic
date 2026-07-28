from dataclasses import dataclass


@dataclass
class Diagnostics:
    mean_electron_energy_ev: float = 0.0
    max_electron_energy_ev: float = 0.0
    mean_proton_energy_ev: float = 0.0
    num_alive_electrons: int = 0
    num_alive_protons: int = 0
    lost_electrons: int = 0
    lost_protons: int = 0
    poisson_iterations: int = 0
    poisson_relative_residual: float = 0.0
    poisson_converged: bool = True

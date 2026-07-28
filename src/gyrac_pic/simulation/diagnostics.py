from dataclasses import dataclass


@dataclass
class Diagnostics:
    mean_electron_energy_ev: float = 0.0
    max_electron_energy_ev: float = 0.0
    mean_proton_energy_ev: float = 0.0
    max_proton_energy_ev: float = 0.0
    max_electron_speed_m_per_s: float = 0.0
    max_proton_speed_m_per_s: float = 0.0
    num_alive_electrons: int = 0
    num_alive_protons: int = 0
    lost_electrons: int = 0
    lost_protons: int = 0
    poisson_iterations: int = 0
    poisson_relative_residual: float = 0.0
    poisson_converged: bool = True
    poisson_final_residual: float = 0.0
    poisson_elapsed_seconds: float = 0.0
    total_grid_charge_c: float = 0.0
    max_abs_charge_density_c_per_m3: float = 0.0
    rms_electric_field_v_per_m: float = 0.0
    max_electric_field_v_per_m: float = 0.0
    max_abs_potential_v: float = 0.0

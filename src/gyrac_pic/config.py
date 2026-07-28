from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

from .constants import ELEMENTARY_CHARGE, ELECTRON_MASS, PROTON_MASS


@dataclass
class GridConfig:
    shape: tuple[int, int, int] = (48, 48, 64)
    dtype_name: str = "float32"


@dataclass
class SpeciesConfig:
    name: str
    count: int
    density_m3: float
    temperature_ev: float
    physical_charge_c: float
    physical_mass_kg: float


@dataclass
class PoissonConfig:
    relative_tolerance: float = 1e-5
    absolute_tolerance: float = 0.0
    max_iterations: int = 200
    preconditioner: str = "jacobi"
    solve_stride: int = 1


@dataclass
class VisualizationConfig:
    mode: str = "external"
    spawn_viewer: bool = True
    recording_path: Path | None = None
    visualization_stride: int = 10
    diagnostics_stride: int = 1
    max_particles_per_species: int = 50_000
    log_grid: bool = True
    log_domain: bool = True
    log_field_vectors: bool = False
    log_charge_density: bool = False
    log_potential_slice: bool = True
    potential_color_percentile: float = 99.0
    energy_color_mode: str = "running_percentile"
    electron_color_min_ev: float = 0.0
    electron_color_max_ev: float = 1e6
    proton_color_min_ev: float = 0.0
    proton_color_max_ev: float = 1e4

    def __post_init__(self) -> None:
        valid = {"disabled", "external", "file", "external_and_file"}
        if self.mode not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}")
        if "file" in self.mode and self.recording_path is None:
            raise ValueError("a recording_path is required for file visualization")
        if not 0.0 < self.potential_color_percentile <= 100.0:
            raise ValueError("potential_color_percentile must be in (0, 100]")


@dataclass
class ResonatorConfig:
    radius_m: float = 0.04687
    length_m: float = 0.10
    frequency_hz: float = 2.4e9
    electric_field_amplitude_v_per_m: float = 3e5
    rf_ramp_cycles: int = 10


@dataclass
class MagneticFieldConfig:
    B0_tesla: float = ELECTRON_MASS * (2 * math.pi * 2.4e9) / ELEMENTARY_CHARGE
    delta_B_max_tesla: float = 0.05
    ramp_time_s: float = 100e-6
    mirror_ratio: float = 1.02
    nonphysical_scaled_ramp: bool = False


@dataclass
class PlasmaConfig:
    radius_m: float = 0.004
    length_m: float = 0.04
    center_z_m: float = 0.0


@dataclass
class ExperimentConfig:
    name: str = "classic_gyrac_x_smoke"
    dt_s: float = 0.8e-12
    num_steps: int = 100
    grid: GridConfig = field(default_factory=GridConfig)
    species: list[SpeciesConfig] = field(default_factory=list)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    resonator: ResonatorConfig = field(default_factory=ResonatorConfig)
    magnetic_field: MagneticFieldConfig = field(default_factory=MagneticFieldConfig)
    plasma: PlasmaConfig = field(default_factory=PlasmaConfig)
    checkpoint_stride: int = 0
    random_seed: int = 1234
    self_consistent_field_enabled: bool = True
    device: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        path = result["visualization"]["recording_path"]
        result["visualization"]["recording_path"] = str(path) if path else None
        return result


def _species(count: int) -> list[SpeciesConfig]:
    return [
        SpeciesConfig("electrons", count, 4e16, 20.0, -ELEMENTARY_CHARGE, ELECTRON_MASS),
        SpeciesConfig("protons", count, 4e16, 1.0, ELEMENTARY_CHARGE, PROTON_MASS),
    ]


def make_smoke_config() -> ExperimentConfig:
    return ExperimentConfig(species=_species(10_000))


def make_classic_gyrac_x_smoke_config() -> ExperimentConfig:
    return make_smoke_config()


def make_development_config() -> ExperimentConfig:
    return ExperimentConfig(name="classic_gyrac_x_development", grid=GridConfig((128, 128, 160)), species=_species(100_000))


def make_production_config() -> ExperimentConfig:
    return ExperimentConfig(name="classic_gyrac_x_production", grid=GridConfig((192, 192, 256)), species=_species(500_000), num_steps=1000)

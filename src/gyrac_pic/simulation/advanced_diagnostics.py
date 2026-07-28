"""Energy balance and field/trajectory diagnostics for validation experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..constants import ELEMENTARY_CHARGE, EPSILON_0, SPEED_OF_LIGHT
from ..particles import kinetic_energy, velocity_from_momentum


class AdvancedDiagnosticsCollector:
    """Collect a compact, serializable energy balance without changing PIC state."""

    def __init__(self, experiment, rf_module, sample_stride=10, report_stride=100):
        self.experiment = experiment
        self.rf_module = rf_module
        self.sample_stride = sample_stride
        self.report_stride = report_stride
        self.samples: list[dict] = []
        self.cumulative_external_work_j = 0.0
        self.cumulative_rf_work_j = 0.0
        self._previous_time = None
        self._previous_external_power = None
        self._previous_rf_power = None
        self._initial_total_energy_j = None
        self._tracked_indices = torch.arange(
            min(24, len(experiment.state.species["electrons"].positions)),
            device=experiment.device,
        )
        self._trails: list = []

    def _module_power(self, module, time):
        total = 0.0
        e_magnitudes = []
        b_magnitudes = []
        for species in self.experiment.state.species.values():
            live = species.alive
            positions = species.positions[live]
            velocity = velocity_from_momentum(
                species.momenta[live], species.physical_mass_kg
            )
            electric = module.electric_field(positions, time)
            magnetic = module.magnetic_field(positions, time)
            total += float(
                species.macro_weight
                * species.physical_charge_c
                * torch.sum(electric * velocity)
            )
            if species.name == "electrons":
                e_magnitudes.append(torch.linalg.vector_norm(electric, dim=-1))
                b_magnitudes.append(torch.linalg.vector_norm(magnetic, dim=-1))
        return total, torch.cat(e_magnitudes), torch.cat(b_magnitudes)

    def observe(self, force_report=False):
        state = self.experiment.state
        if not force_report and state.step % self.sample_stride:
            return None
        time = state.time_s
        external_power = 0.0
        rf_power = 0.0
        rf_e = rf_b = None
        for module in self.experiment.modules:
            power, e_magnitude, b_magnitude = self._module_power(module, time)
            external_power += power
            if module is self.rf_module:
                rf_power = power
                rf_e, rf_b = e_magnitude, b_magnitude
        if self._previous_time is not None:
            elapsed = time - self._previous_time
            self.cumulative_external_work_j += (
                0.5 * (external_power + self._previous_external_power) * elapsed
            )
            self.cumulative_rf_work_j += (
                0.5 * (rf_power + self._previous_rf_power) * elapsed
            )
        self._previous_time = time
        self._previous_external_power = external_power
        self._previous_rf_power = rf_power

        if not force_report and state.step % self.report_stride:
            return None
        kinetic = {}
        radial_rms = {}
        for name, species in state.species.items():
            live = species.alive
            energy = kinetic_energy(
                species.momenta[live], species.physical_mass_kg
            )
            kinetic[name] = float(energy.sum() * species.macro_weight)
            radial_rms[name] = float(
                torch.sqrt(torch.mean(torch.sum(species.positions[live, :2] ** 2, dim=-1)))
            )
        self_field_energy = float(
            0.5
            * EPSILON_0
            * torch.sum(state.grid.electric_field**2)
            * self.experiment.grid.cell_volume
        )
        total_energy = sum(kinetic.values()) + self_field_energy
        if self._initial_total_energy_j is None:
            self._initial_total_energy_j = total_energy
        balance_error = (
            total_energy
            - self._initial_total_energy_j
            - self.cumulative_external_work_j
        )
        electrons = state.species["electrons"]
        live_electrons = electrons.alive
        electron_positions = electrons.positions[live_electrons]
        electron_momentum = electrons.momenta[live_electrons]
        external_b = torch.zeros_like(electron_positions)
        for module in self.experiment.modules:
            external_b += module.magnetic_field(electron_positions, time)
        if len(electron_momentum):
            b_magnitude = torch.linalg.vector_norm(external_b, dim=-1).clamp_min(1e-20)
            b_direction = external_b / b_magnitude[:, None]
            normalized_p = electron_momentum / (
                electrons.physical_mass_kg * SPEED_OF_LIGHT
            )
            parallel_p = torch.sum(normalized_p * b_direction, dim=-1, keepdim=True)
            perpendicular_p = normalized_p - parallel_p * b_direction
            perpendicular_u = torch.linalg.vector_norm(perpendicular_p, dim=-1)
            larmor_radius = (
                perpendicular_u
                * electrons.physical_mass_kg
                * SPEED_OF_LIGHT
                / (abs(electrons.physical_charge_c) * b_magnitude)
            )
            magnetic_moment = (
                perpendicular_u**2
                * electrons.physical_mass_kg
                * SPEED_OF_LIGHT**2
                / (2 * b_magnitude)
            )
            transverse_u = torch.linalg.vector_norm(
                normalized_p[:, :2], dim=-1
            ).clamp_min(1e-20)
            gyro_phase_coherence = torch.sqrt(
                torch.mean(normalized_p[:, 0] / transverse_u) ** 2
                + torch.mean(normalized_p[:, 1] / transverse_u) ** 2
            )
            larmor_mean, larmor_max = float(larmor_radius.mean()), float(larmor_radius.max())
            magnetic_moment_mean = float(magnetic_moment.mean())
            phase_coherence = float(gyro_phase_coherence)
        else:
            larmor_mean = larmor_max = magnetic_moment_mean = phase_coherence = 0.0
        sample = {
            "step": state.step,
            "time_s": time,
            "electron_mean_energy_ev": self.experiment.latest_diagnostics.mean_electron_energy_ev,
            "electron_max_energy_ev": self.experiment.latest_diagnostics.max_electron_energy_ev,
            "proton_mean_energy_ev": self.experiment.latest_diagnostics.mean_proton_energy_ev,
            "electron_kinetic_energy_j": kinetic["electrons"],
            "proton_kinetic_energy_j": kinetic["protons"],
            "self_field_energy_j": self_field_energy,
            "tracked_total_energy_j": total_energy,
            "instantaneous_external_power_w": external_power,
            "instantaneous_rf_power_w": rf_power,
            "cumulative_external_work_j": self.cumulative_external_work_j,
            "cumulative_rf_work_j": self.cumulative_rf_work_j,
            "energy_balance_error_j": balance_error,
            "relative_energy_balance_error": balance_error
            / max(abs(self._initial_total_energy_j), 1e-30),
            "electron_radial_rms_m": radial_rms["electrons"],
            "proton_radial_rms_m": radial_rms["protons"],
            "electron_mean_larmor_radius_m": larmor_mean,
            "electron_max_larmor_radius_m": larmor_max,
            "electron_mean_magnetic_moment_j_per_t": magnetic_moment_mean,
            "electron_gyro_phase_coherence": phase_coherence,
            "rf_envelope": self.rf_module.envelope(time),
            "rf_e_rms_at_electrons_v_per_m": (
                float(torch.sqrt(torch.mean(rf_e**2))) if len(rf_e) else 0.0
            ),
            "rf_e_max_at_electrons_v_per_m": float(rf_e.max()) if len(rf_e) else 0.0,
            "rf_b_rms_at_electrons_t": (
                float(torch.sqrt(torch.mean(rf_b**2))) if len(rf_b) else 0.0
            ),
            "rf_b_max_at_electrons_t": float(rf_b.max()) if len(rf_b) else 0.0,
            "poisson_converged": self.experiment.latest_diagnostics.poisson_converged,
            "poisson_relative_residual": self.experiment.latest_diagnostics.poisson_relative_residual,
            "alive_electrons": self.experiment.latest_diagnostics.num_alive_electrons,
            "lost_electrons": self.experiment.latest_diagnostics.lost_electrons,
        }
        self.samples.append(sample)
        print("GYRAC_ADVANCED " + json.dumps(sample))
        self._log_rerun(sample)
        return sample

    def _log_rerun(self, sample):
        sink = self.experiment.visualizer
        if not sink.connected or sink.rr is None:
            return
        try:
            rr = sink.rr
            for key in (
                "electron_mean_energy_ev", "self_field_energy_j",
                "cumulative_rf_work_j", "relative_energy_balance_error",
                "rf_e_rms_at_electrons_v_per_m", "electron_radial_rms_m",
            ):
                rr.log(f"advanced/{key}", rr.Scalars(sample[key]))
            positions = self.experiment.state.species["electrons"].positions[
                self._tracked_indices
            ].detach().cpu().numpy()
            momenta = self.experiment.state.species["electrons"].momenta[
                self._tracked_indices
            ].detach().cpu().numpy()
            for index in range(min(8, len(positions))):
                rr.log(f"tracked/e{index}/x_m", rr.Scalars(float(positions[index, 0])))
                rr.log(f"tracked/e{index}/y_m", rr.Scalars(float(positions[index, 1])))
                rr.log(f"tracked/e{index}/px", rr.Scalars(float(momenta[index, 0])))
                rr.log(f"tracked/e{index}/py", rr.Scalars(float(momenta[index, 1])))
            self._trails.append(positions)
            self._trails = self._trails[-256:]
            if len(self._trails) > 1:
                strips = [
                    [frame[index].tolist() for frame in self._trails]
                    for index in range(len(self._tracked_indices))
                ]
                rr.log("tracked/electron_trails", rr.LineStrips3D(strips, radii=2e-5))
        except Exception:
            # Advanced visualization is auxiliary; JSON diagnostics remain valid.
            return

    def save_summary(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "format_version": 1,
            "metadata": metadata or {},
            "initial": self.samples[0] if self.samples else None,
            "final": self.samples[-1] if self.samples else None,
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        path.write_text(json.dumps(summary, indent=2))
        print("GYRAC_SUMMARY_PATH " + str(path))
        print("GYRAC_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "samples"}))
        return path

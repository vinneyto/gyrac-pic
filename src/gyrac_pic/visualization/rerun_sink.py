from __future__ import annotations
import importlib
import importlib.util
import logging

from .static_scene import log_static_scene

log = logging.getLogger(__name__)


class RerunSink:
    """Optional Rerun adapter; viewer disconnects never stop the simulation."""
    def __init__(self, config, recording_id):
        self.config, self.recording_id, self.rr, self.connected = config, recording_id, None, False
        self._static_logged = False
        self._registered_series = set()

    def initialize(self):
        if self.config.mode == "disabled" or self.connected:
            return
        if importlib.util.find_spec("rerun") is None:
            log.warning("Rerun SDK is not installed; visualization is disabled")
            return
        try:
            rr = importlib.import_module("rerun")
            self.rr = rr
            rr.init("gyrac_pic", recording_id=self.recording_id, spawn=False)
            if self.config.mode in {"external", "external_and_file"} and self.config.spawn_viewer:
                rr.spawn()
            if self.config.mode in {"file", "external_and_file"}:
                self.config.recording_path.parent.mkdir(parents=True, exist_ok=True)
                supports_multiple_sinks = all(
                    hasattr(rr, name) for name in ("set_sinks", "GrpcSink", "FileSink")
                )
                if self.config.mode == "external_and_file" and supports_multiple_sinks:
                    rr.set_sinks(
                        rr.GrpcSink(),
                        rr.FileSink(str(self.config.recording_path)),
                    )
                else:
                    if self.config.mode == "external_and_file":
                        log.warning(
                            "This Rerun SDK has no multi-sink API; file output takes "
                            "precedence over the live viewer. Upgrade rerun-sdk."
                        )
                    rr.save(str(self.config.recording_path))
            self._send_blueprint()
            self.connected = True
        except Exception as error:
            log.warning("Rerun unavailable; simulation continues: %s", error)

    def log_frame(self, state, diagnostics, max_particles):
        self.initialize()
        if not self.connected: return
        try:
            self.rr.set_time("step", sequence=state.step)
            self.rr.set_time("physical_time", duration=state.time_s)
            for name, species in state.species.items():
                points = species.positions[species.alive][:max_particles].detach().cpu().numpy()
                self.rr.log(f"particles/{name}", self.rr.Points3D(points, radii=0.00015))
            self.log_scalar(
                "plots/energy_ev/electron_mean", diagnostics.mean_electron_energy_ev,
                "Mean electron energy [eV]",
            )
            self.log_scalar(
                "plots/energy_ev/electron_max", diagnostics.max_electron_energy_ev,
                "Maximum electron energy [eV]",
            )
            self.log_scalar(
                "plots/energy_ev/proton_mean", diagnostics.mean_proton_energy_ev,
                "Mean proton energy [eV]",
            )
            self.log_scalar(
                "plots/particle_count/alive_electrons", diagnostics.num_alive_electrons,
                "Alive electron macroparticles [count]",
            )
            self.log_scalar(
                "plots/solver/poisson_relative_residual",
                diagnostics.poisson_relative_residual,
                "Poisson relative residual [1]",
            )
        except Exception as error:
            log.warning("Rerun connection lost; disabling live output: %s", error)
            self.connected = False

    def log_static_scene(self, domain, grid, modules):
        self.initialize()
        if not self.connected or self._static_logged:
            return
        try:
            log_static_scene(self.rr, domain, grid, modules, self.config)
            self._static_logged = True
        except Exception as error:
            log.warning("Could not log static Rerun scene: %s", error)

    def log_scalar(self, path, value, legend_name=None):
        """Log a one-element scalar batch to a leaf entity path."""
        if legend_name and path not in self._registered_series:
            if hasattr(self.rr, "SeriesLines"):
                try:
                    self.rr.log(
                        path,
                        self.rr.SeriesLines(names=legend_name),
                        static=True,
                    )
                except Exception as error:
                    log.warning("Could not set Rerun series legend for %s: %s", path, error)
            self._registered_series.add(path)
        self.rr.log(path, self.rr.Scalars([float(value)]))

    def set_time(self, step, time_s):
        self.rr.set_time("step", sequence=int(step))
        self.rr.set_time("physical_time", duration=float(time_s))

    def _send_blueprint(self):
        """Install explicit views so scalar leaves are never selected as parents."""
        rr = self.rr
        if not hasattr(rr, "blueprint") or not hasattr(rr, "send_blueprint"):
            return
        try:
            rrb = rr.blueprint
            plots = rrb.Vertical(
                rrb.TimeSeriesView(name="Particle energy [eV]", origin="/plots/energy_ev"),
                rrb.TimeSeriesView(name="Particle count [count]", origin="/plots/particle_count"),
                rrb.TimeSeriesView(name="Poisson convergence [1]", origin="/plots/solver"),
                rrb.TimeSeriesView(name="Energy accounting [J]", origin="/advanced/energy_j"),
                rrb.TimeSeriesView(name="External power [W]", origin="/advanced/power_w"),
                rrb.TimeSeriesView(name="RF electric field [V/m]", origin="/advanced/electric_field_v_per_m"),
                rrb.TimeSeriesView(name="Particle radii [m]", origin="/advanced/radius_m"),
                rrb.TimeSeriesView(name="Dimensionless validation", origin="/advanced/dimensionless"),
            )
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial3DView(name="GYRAC 3D scene", origin="/"),
                        rrb.TextDocumentView(
                            name="Experiment parameters", origin="/experiment/parameters"
                        ),
                        row_shares=[3, 2],
                    ),
                    plots,
                    column_shares=[3, 2],
                ),
                collapse_panels=True,
            )
            rr.send_blueprint(blueprint)
        except Exception as error:
            log.warning("Could not install Rerun blueprint: %s", error)

    def reset(self):
        self.connected = False
        self._static_logged = False
        self._registered_series.clear()
        self.initialize()

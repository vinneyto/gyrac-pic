from __future__ import annotations
import importlib
import importlib.util
import logging
import numpy as np
import torch

from .static_scene import log_static_scene
from .color_maps import signed_potential_rgba
from ..particles import gather_field_cic

log = logging.getLogger(__name__)


class RerunSink:
    """Optional Rerun adapter; viewer disconnects never stop the simulation."""
    def __init__(self, config, recording_id):
        self.config, self.recording_id, self.rr, self.connected = config, recording_id, None, False
        self._static_logged = False
        self._registered_series = set()
        self._grid = None
        self._domain = None
        self._modules = []

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
            self._log_potential_slice(state)
            self._log_transverse_field_arrows(state)
        except Exception as error:
            log.warning("Rerun connection lost; disabling live output: %s", error)
            self.connected = False

    def log_static_scene(self, domain, grid, modules, experiment_config):
        self.initialize()
        if not self.connected or self._static_logged:
            return
        try:
            self._grid = grid
            self._domain = domain
            self._modules = list(modules)
            log_static_scene(
                self.rr,
                domain,
                grid,
                modules,
                self.config,
                experiment_config,
            )
            self._static_logged = True
        except Exception as error:
            log.warning("Could not log static Rerun scene: %s", error)

    def _log_potential_slice(self, state):
        if not self.config.log_potential_slice or self._grid is None:
            return
        try:
            z_coordinates = self._grid.coordinates()[2]
            middle_index = int(z_coordinates.abs().argmin())
            # Rerun image rows are y and columns are x, hence the transpose.
            potential = (
                state.grid.phi[:, :, middle_index]
                .detach()
                .cpu()
                .numpy()
                .T
            )
            rgba, scale = signed_potential_rgba(
                potential,
                percentile=self.config.potential_color_percentile,
            )
            self.rr.log("fields/potential_midplane_z/rgba", self.rr.Image(rgba))
            self.log_scalar(
                "fields/potential_midplane_z/color_scale_v",
                scale,
                "Potential color scale ± [V]",
            )
        except Exception as error:
            log.warning("Could not log potential slice image: %s", error)

    @staticmethod
    def _scaled_arrow_vectors(field, max_length_m, percentile):
        magnitude = torch.linalg.vector_norm(field, dim=-1)
        nonzero = magnitude[magnitude > 0]
        if len(nonzero) == 0:
            return torch.zeros_like(field), 0.0
        scale_value = float(
            np.percentile(nonzero.detach().cpu().numpy(), percentile)
        )
        scale_value = max(scale_value, torch.finfo(field.dtype).tiny)
        scale = field.new_tensor(scale_value)
        direction = field / torch.clamp(magnitude[:, None], min=scale * 1e-12)
        length = max_length_m * torch.clamp(magnitude / scale, max=1.0)
        return direction * length[:, None], scale_value

    def _log_transverse_field_arrows(self, state):
        if (
            not self.config.log_field_vectors
            or self._grid is None
            or self._domain is None
        ):
            return
        try:
            nx, ny = self.config.field_vector_grid_shape
            x = torch.linspace(
                *self._domain.bounds[0], nx,
                device=self._grid.device, dtype=self._grid.dtype,
            )
            y = torch.linspace(
                *self._domain.bounds[1], ny,
                device=self._grid.device, dtype=self._grid.dtype,
            )
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            z_nodes = self._grid.coordinates()[2]
            z = z_nodes[z_nodes.abs().argmin()]
            origins = torch.stack(
                (xx.reshape(-1), yy.reshape(-1), torch.full_like(xx.reshape(-1), z)),
                dim=-1,
            )
            inside = self._domain.particle_inside(origins)
            origins = origins[inside]
            self_field = gather_field_cic(
                state.grid.electric_field, origins, self._grid.bounds
            )
            rf_field = torch.zeros_like(self_field)
            for module in self._modules:
                if "resonator" in module.scene_renderers():
                    rf_field += module.electric_field(origins, state.time_s)
            # This view is explicitly transverse even if E_self has a local Ez.
            self_field[:, 2] = 0
            rf_field[:, 2] = 0
            for path, field, color, label in (
                ("fields/self_electric/transverse_arrows", self_field,
                 [65, 155, 255, 220], "Self E arrow scale [V/m]"),
                ("fields/rf_electric/transverse_arrows", rf_field,
                 [255, 125, 35, 235], "RF E arrow scale [V/m]"),
            ):
                vectors, scale = self._scaled_arrow_vectors(
                    field,
                    self.config.field_arrow_max_length_m,
                    self.config.field_arrow_scale_percentile,
                )
                self.rr.log(
                    path,
                    self.rr.Arrows3D(
                        origins=origins.detach().cpu().numpy(),
                        vectors=vectors.detach().cpu().numpy(),
                        colors=color,
                        radii=8e-5,
                    ),
                )
                self.log_scalar(path.rsplit("/", 1)[0] + "/scale_v_per_m", scale, label)
        except Exception as error:
            log.warning("Could not log transverse electric-field arrows: %s", error)

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
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial3DView(name="GYRAC 3D scene", origin="/"),
                        rrb.Horizontal(
                            rrb.TextDocumentView(
                                name="Experiment parameters",
                                origin="/experiment/parameters",
                            ),
                            rrb.Spatial2DView(
                                name="Mid-plane potential φ(x,y) [V]",
                                origin="/fields/potential_midplane_z",
                            ),
                            column_shares=[3, 2],
                        ),
                        row_shares=[3, 2],
                    ),
                    rrb.TimeSeriesView(
                        name="Particle energy [eV]",
                        origin="/plots/energy_ev",
                    ),
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

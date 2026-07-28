from __future__ import annotations
import logging

log = logging.getLogger(__name__)


class RerunSink:
    """Optional Rerun adapter; viewer disconnects never stop the simulation."""
    def __init__(self, config, recording_id):
        self.config, self.recording_id, self.rr, self.connected = config, recording_id, None, False

    def initialize(self):
        if self.config.mode == "disabled" or self.connected:
            return
        try:
            import rerun as rr
            self.rr = rr
            rr.init("gyrac_pic", recording_id=self.recording_id, spawn=False)
            if self.config.mode in {"external", "external_and_file"} and self.config.spawn_viewer:
                rr.spawn()
            if self.config.mode in {"file", "external_and_file"}:
                self.config.recording_path.parent.mkdir(parents=True, exist_ok=True)
                rr.save(str(self.config.recording_path))
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
            self.rr.log("diagnostics/electron_mean_energy_ev", self.rr.Scalars(diagnostics.mean_electron_energy_ev))
        except Exception as error:
            log.warning("Rerun connection lost; disabling live output: %s", error)
            self.connected = False

    def reset(self):
        self.connected = False
        self.initialize()

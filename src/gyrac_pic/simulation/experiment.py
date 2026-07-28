from __future__ import annotations
import json, math
from pathlib import Path
import torch

from ..config import ExperimentConfig
from ..constants import ELEMENTARY_CHARGE, ELECTRON_MASS, EPSILON_0, SPEED_OF_LIGHT
from ..device import dtype_from_name, select_device
from ..grid import CartesianGrid, negative_gradient
from ..particles import deposit_charge_cic, gather_field_cic, kinetic_energy, relativistic_boris_push, velocity_from_momentum
from ..state import GridState, ParticleSpeciesState, SimulationState
from ..fields.poisson import solve_pcg
from ..visualization import RerunSink
from .diagnostics import Diagnostics


class Experiment:
    def __init__(self, config, domain, modules):
        self.config, self.domain, self.modules = config, domain, list(modules)
        self.device = select_device(config.device)
        self.dtype = dtype_from_name(config.grid.dtype_name)
        self.grid = CartesianGrid(domain.grid_shape, domain.bounds, self.device, self.dtype)
        self.mask = domain.interior_mask(self.device, self.dtype)
        self.state = None
        self.latest_diagnostics = Diagnostics()
        self.loss_events = []
        self.visualizer = RerunSink(config.visualization, config.name)

    @classmethod
    def create(cls, config: ExperimentConfig, domain, modules, visualization_config=None):
        if visualization_config is not None: config.visualization = visualization_config
        if tuple(config.grid.shape) != tuple(domain.grid_shape):
            raise ValueError("config and domain grid shapes differ")
        return cls(config, domain, modules)

    def initialize(self):
        torch.manual_seed(self.config.random_seed)
        nmax = max(s.count for s in self.config.species)
        u = torch.rand((nmax, 3), device=self.device, dtype=self.dtype)
        radius = self.config.plasma.radius_m * torch.sqrt(u[:, 0])
        positions = torch.stack((radius*torch.cos(2*math.pi*u[:,1]), radius*torch.sin(2*math.pi*u[:,1]),
                                 self.config.plasma.center_z_m+self.config.plasma.length_m*(u[:,2]-0.5)), -1)
        volume = math.pi*self.config.plasma.radius_m**2*self.config.plasma.length_m
        species = {}
        for spec in self.config.species:
            pos = positions[:spec.count].clone()
            std = math.sqrt(spec.temperature_ev*ELEMENTARY_CHARGE/spec.physical_mass_kg)
            velocity = torch.randn_like(pos)*std
            velocity -= velocity.mean(0, keepdim=True)
            speed2 = (velocity*velocity).sum(-1, keepdim=True)
            gamma = 1/torch.sqrt(1-speed2/SPEED_OF_LIGHT**2)
            momentum = gamma*spec.physical_mass_kg*velocity
            weight = spec.density_m3*volume/spec.count
            species[spec.name] = ParticleSpeciesState(spec.name, pos, momentum, torch.ones(spec.count,device=self.device,dtype=torch.bool),
                spec.physical_charge_c, spec.physical_mass_kg, weight, weight*spec.physical_charge_c, weight*spec.physical_mass_kg)
        zeros = torch.zeros(self.grid.shape, device=self.device, dtype=self.dtype)
        self.state = SimulationState(0, 0.0, species, GridState(zeros, zeros.clone(), torch.zeros((*self.grid.shape,3),device=self.device,dtype=self.dtype)))
        self._update_field()
        self._diagnose(self._last_solve)
        return self.state

    def _update_field(self):
        rho = torch.zeros(self.grid.shape, device=self.device, dtype=self.dtype)
        for s in self.state.species.values():
            charges = torch.full((len(s.positions),), s.macro_charge_c, device=self.device, dtype=self.dtype)
            rho += deposit_charge_cic(s.positions, charges, self.grid.shape, self.grid.bounds, alive=s.alive)
        rho *= self.mask
        if self.config.self_consistent_field_enabled:
            pc = self.config.poisson
            phi, info = solve_pcg(rho/EPSILON_0, self.mask, self.grid.spacing, self.state.grid.phi,
                                   pc.relative_tolerance, pc.absolute_tolerance, pc.max_iterations)
            field = negative_gradient(phi, self.grid.spacing, self.mask)
        else:
            phi, field, info = torch.zeros_like(rho), torch.zeros((*rho.shape,3),device=self.device,dtype=self.dtype), None
        self.state.grid = GridState(rho, phi, field); self._last_solve = info

    def step(self):
        field_time = self.state.time_s + self.config.dt_s/2
        for s in self.state.species.values():
            e = gather_field_cic(self.state.grid.electric_field, s.positions, self.grid.bounds)
            b = torch.zeros_like(e)
            for module in self.modules:
                e += module.electric_field(s.positions, field_time)
                b += module.magnetic_field(s.positions, field_time)
            new_pos, new_mom = relativistic_boris_push(s.positions, s.momenta, e, b, s.physical_charge_c, s.physical_mass_kg, self.config.dt_s)
            moving = s.alive[:,None]
            s.positions = torch.where(moving, new_pos, s.positions); s.momenta = torch.where(moving, new_mom, s.momenta)
            newly_lost = s.alive & ~self.domain.particle_inside(s.positions)
            s.alive &= ~newly_lost
        self.state.step += 1; self.state.time_s = self.state.step*self.config.dt_s
        if self.state.step % self.config.poisson.solve_stride == 0: self._update_field()
        self._diagnose(self._last_solve)

    def _diagnose(self, info):
        def stats(name):
            s = self.state.species[name]
            live_momenta = s.momenta[s.alive]
            values = kinetic_energy(live_momenta, s.physical_mass_kg) / ELEMENTARY_CHARGE
            speeds = torch.linalg.vector_norm(
                velocity_from_momentum(live_momenta, s.physical_mass_kg), dim=-1
            )
            return (
                float(values.mean()) if len(values) else 0.0,
                float(values.max()) if len(values) else 0.0,
                float(speeds.max()) if len(speeds) else 0.0,
                int(s.alive.sum()),
                len(s.alive) - int(s.alive.sum()),
            )
        e=stats("electrons"); p=stats("protons")
        field_magnitude = torch.linalg.vector_norm(self.state.grid.electric_field, dim=-1)
        self.latest_diagnostics = Diagnostics(
            mean_electron_energy_ev=e[0],
            max_electron_energy_ev=e[1],
            mean_proton_energy_ev=p[0],
            max_proton_energy_ev=p[1],
            max_electron_speed_m_per_s=e[2],
            max_proton_speed_m_per_s=p[2],
            num_alive_electrons=e[3],
            num_alive_protons=p[3],
            lost_electrons=e[4],
            lost_protons=p[4],
            poisson_iterations=info.iterations if info else 0,
            poisson_relative_residual=info.relative_residual if info else 0.0,
            poisson_converged=info.converged if info else True,
            poisson_final_residual=info.final_residual if info else 0.0,
            poisson_elapsed_seconds=info.elapsed_seconds if info else 0.0,
            total_grid_charge_c=float(self.state.grid.rho.sum() * self.grid.cell_volume),
            max_abs_charge_density_c_per_m3=float(self.state.grid.rho.abs().max()),
            rms_electric_field_v_per_m=float(torch.sqrt(torch.mean(field_magnitude**2))),
            max_electric_field_v_per_m=float(field_magnitude.max()),
            max_abs_potential_v=float(self.state.grid.phi.abs().max()),
        )

    def run(self, num_steps=None):
        if self.state is None: self.initialize()
        self.print_diagnostics(label="run_start")
        requested_steps = self.config.num_steps if num_steps is None else num_steps
        for _ in range(requested_steps):
            with torch.inference_mode(): self.step()
            if self.state.step % self.config.visualization.visualization_stride == 0:
                self.visualizer.log_frame(self.state,self.latest_diagnostics,self.config.visualization.max_particles_per_species)
                self.print_diagnostics(label="progress")
            if self.config.checkpoint_stride and self.state.step % self.config.checkpoint_stride == 0:
                self.save_checkpoint(f"runs/{self.config.name}/checkpoints/step_{self.state.step:08d}.pt")
        if requested_steps and self.state.step % self.config.visualization.visualization_stride:
            self.print_diagnostics(label="run_end")
        return self.state

    def visualize_initial_state(self):
        if self.state is None: raise RuntimeError("call initialize() before visualization")
        self.visualizer.log_frame(self.state,self.latest_diagnostics,self.config.visualization.max_particles_per_species)

    def reset_visualization_recording(self): self.visualizer.reset()

    def physical_scales(self):
        n=next(s.density_m3 for s in self.config.species if s.name=="electrons"); t=next(s.temperature_ev for s in self.config.species if s.name=="electrons")
        debye=math.sqrt(EPSILON_0*t*ELEMENTARY_CHARGE/(n*ELEMENTARY_CHARGE**2)); omega=math.sqrt(n*ELEMENTARY_CHARGE**2/(ELECTRON_MASS*EPSILON_0))
        return {"rf_period_s":1/self.config.resonator.frequency_hz,"electron_plasma_omega_rad_s":omega,"debye_length_m":debye,
                "grid_over_debye":tuple(d/debye for d in self.grid.spacing),"omega_pe_dt":omega*self.config.dt_s,"steps_per_rf_period":1/self.config.resonator.frequency_hz/self.config.dt_s}

    def print_configuration(self): print(json.dumps(self.config.to_dict(),indent=2))
    def print_physical_scales(self): print(json.dumps(self.physical_scales(),indent=2))

    def diagnostics_dict(self, label=None):
        from dataclasses import asdict

        result = {
            "label": label,
            "step": self.state.step,
            "time_s": self.state.time_s,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            **asdict(self.latest_diagnostics),
        }
        return result

    def print_diagnostics(self, label=None):
        if self.state is None:
            raise RuntimeError("experiment is not initialized")
        print("GYRAC_DIAGNOSTICS " + json.dumps(self.diagnostics_dict(label)))

    def save_checkpoint(self, path):
        if self.state is None: raise RuntimeError("experiment is not initialized")
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"format_version":1,"step":self.state.step,"time_s":self.state.time_s,"species":{k:{**vars(v),"positions":v.positions.cpu(),"momenta":v.momenta.cpu(),"alive":v.alive.cpu()} for k,v in self.state.species.items()},
                    "grid":{"rho":self.state.grid.rho.cpu(),"phi":self.state.grid.phi.cpu(),"electric_field":self.state.grid.electric_field.cpu()},"rng":{"torch_cpu":torch.get_rng_state()},"experiment_config":self.config.to_dict(),"module_states":{m.name:m.state_dict() for m in self.modules}},path)
        path.with_name("run_config.json").write_text(json.dumps(self.config.to_dict(),indent=2))
        return path

    def restore_checkpoint(self, path):
        data=torch.load(path,map_location=self.device,weights_only=False)
        species={k:ParticleSpeciesState(**{**v,"positions":v["positions"].to(self.device),"momenta":v["momenta"].to(self.device),"alive":v["alive"].to(self.device)}) for k,v in data["species"].items()}
        g=data["grid"]; grid=GridState(*(g[k].to(self.device) for k in ("rho","phi","electric_field")))
        self.state=SimulationState(data["step"],data["time_s"],species,grid)
        for m in self.modules: m.load_state_dict(data["module_states"].get(m.name,{}))
        self._diagnose(None); return self

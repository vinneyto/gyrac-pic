# GYRAC PIC

GYRAC PIC is a research-oriented 3D electrostatic Particle-in-Cell model of
hydrogen plasma. Its PyTorch core supports CUDA, MPS, and CPU execution and
provides a relativistic Boris pusher, vectorized CIC operations, a matrix-free
PCG solver, pluggable TE111 and magnetic-mirror fields, checkpoints, and
optional Rerun visualization.

> The cylinder parameters in the `classic_gyrac_x` profile describe an idealized
> model cavity tuned near 2.4 GHz; they are not measured dimensions of the
> apparatus.

## Installation

```bash
pip install -e '.[test,visualization]'
pytest
```

## Experiments

The repository contains three directly comparable experiment entry points:

```bash
# Self-consistent plasma field only
python experiments/01_self_field_expansion.py

# Self-consistent field and analytic rotating TE111 mode, without magnets
python experiments/02_te111_without_magnets.py

# Full validated scenario: self-consistent field, TE111 mode, and magnetic mirror
python experiments/03_full_te111_autoresonance.py
```

All three experiments use the same cylindrical domain, initial plasma,
self-consistent Poisson solve, 10,000-step loop, advanced diagnostics, particle
trails, checkpoint format, and output layout. They differ only in their external
field modules:

| Experiment | Self-consistent field | TE111 resonator | Magnetic mirror |
| --- | --- | --- | --- |
| `01_self_field_expansion.py` | Yes | No | No |
| `02_te111_without_magnets.py` | Yes | Yes | No |
| `03_full_te111_autoresonance.py` | Yes | Yes | Yes |

Common command-line options are:

```text
--steps N          number of simulation steps (default: 10000)
--report-stride N  advanced diagnostic reporting interval (default: 100)
--no-viewer        record to a file without spawning the Rerun viewer
```

Each run creates a unique UTC-timestamped directory under `runs/` containing:

- `run.rrd` — the Rerun visualization recording;
- `final.pt` — the physical simulation checkpoint;
- `run_config.json` — the serialized experiment configuration;
- `summary.json` — advanced diagnostics and run metadata.

The process also prints `GYRAC_SUMMARY` and `GYRAC_SUMMARY_PATH` lines. Sharing
the summary line or `summary.json` is normally sufficient for an initial review
of a run.

## Diagnostics and visualization

Standard diagnostics include particle energies and speeds for both species,
alive and lost particle counts, grid charge, electric field, potential, and PCG
convergence information. Advanced diagnostics add the energy balance, external
and RF work, plasma-cloud radii, gyro-phase statistics, tracked trajectories,
and fields sampled at electron positions.

The main Rerun plots are stored under `/plots`, advanced plots under
`/advanced/plots`, and individual particle coordinates and momenta under
`/tracked/plots`. The initial blueprint displays only the main particle-energy
plot; all recorded series remain available in the entity tree and can be added
to a view manually.

The 3D view is assembled from the enabled modules. It always shows sparse lines
from the real Cartesian grid and the domain boundary. It adds the translucent
PEC cylinder only when the resonator module is enabled, and adds coils and a
magnetic-field indicator only when the magnetic module is enabled.

The global grid display is intentionally limited to at most 11 nodes per axis.
A bright local section around the initial plasma shows every actual simulation
cell so that the visual spacing cannot be mistaken for `dx`, `dy`, or `dz`.
The parameter area also contains a dynamic transverse potential slice at the
grid plane nearest `z=0`: blue is negative, white is zero, and red is positive.
The color range is normalized symmetrically in volts. This is the
self-consistent electrostatic plasma potential `phi` obtained from the Poisson
solve, not a scalar potential for the external RF mode.

On the transverse plane near `z=0`, blue arrows show the self-consistent plasma
electric field and orange arrows show the resonator RF electric field. Arrow
directions are physical, while lengths are normalized for visualization using
the 95th percentile and capped at 4 mm. The corresponding physical scale in
V/m is recorded as `scale_v_per_m`. These arrows do not replace the simulation
grid and do not affect the calculation.

A `.rrd` file contains visualization data only. Resume the physical simulation
from the `.pt` checkpoint, not from the Rerun recording. At the standard time
step, a physical 100 microsecond magnetic ramp requires approximately 125
million steps and is therefore not run by default. Any deliberately accelerated
demonstration ramp must be marked with `nonphysical_scaled_ramp=True`.

## Numerical limitations

The model does not include the plasma self-magnetic field, collisions,
ionization, radiation, secondary emission, or feedback from the plasma to the
resonator mode. The cylindrical PEC boundary is represented as a stair-step
mask on a Cartesian grid. Before interpreting results, check convergence with
respect to grid resolution, particle count, time step, and PCG tolerance.

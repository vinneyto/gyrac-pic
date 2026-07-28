import torch
from gyrac_pic import *


def tiny_config():
    c=make_smoke_config(); c.grid.shape=(10,10,12); c.species[0].count=c.species[1].count=64
    c.num_steps=2; c.poisson.max_iterations=50; c.visualization.mode="disabled"; c.grid.dtype_name="float64"
    return c


def test_quiet_start_modules_and_checkpoint(tmp_path):
    c=tiny_config(); d=BoxDomain((-.05,.05),(-.05,.05),(-.05,.05),c.grid.shape)
    e=Experiment.create(c,d,[]); e.initialize()
    assert torch.equal(e.state.species["electrons"].positions,e.state.species["protons"].positions)
    assert e.state.step==0 and e.state.time_s==0
    e.run(1); path=e.save_checkpoint(tmp_path/"state.pt")
    restored=Experiment.create(c,d,[]).restore_checkpoint(path)
    assert restored.state.step==1 and torch.equal(restored.state.grid.phi,e.state.grid.phi)


def test_external_module_composition():
    c=tiny_config(); p=torch.tensor([[.001,.002,0.]],dtype=torch.float64)
    r=TE111CylindricalResonator(c.resonator.radius_m,c.resonator.length_m,c.resonator.frequency_hz,3e5)
    m=RampedMirrorMagneticField(c.magnetic_field.B0_tesla,.05,1e-4,1.02,.1)
    assert r.electric_field(p,1e-8).shape==(1,3) and m.magnetic_field(p,0).shape==(1,3)


def test_analytic_te111_is_normalized_on_axis_and_zero_outside():
    c = tiny_config()
    mode = AnalyticRotatingTE111(
        c.resonator.radius_m,
        c.resonator.length_m,
        c.resonator.frequency_hz,
        c.resonator.electric_field_amplitude_v_per_m,
        rf_ramp_cycles=1,
    )
    # One full cycle is after the ramp and returns to the initial modal phase.
    time = 1.0 / c.resonator.frequency_hz
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2 * c.resonator.radius_m, 0.0, 0.0]],
        dtype=torch.float64,
    )
    electric, magnetic = mode.fields(positions, time)
    assert torch.allclose(
        torch.linalg.vector_norm(electric[0]),
        torch.tensor(c.resonator.electric_field_amplitude_v_per_m, dtype=torch.float64),
        rtol=1e-10,
    )
    assert torch.equal(electric[1], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(magnetic[1], torch.zeros(3, dtype=torch.float64))


def test_analytic_te111_rotates_in_quadrature():
    c = tiny_config()
    mode = AnalyticRotatingTE111(
        c.resonator.radius_m,
        c.resonator.length_m,
        c.resonator.frequency_hz,
        1.0,
        rf_ramp_cycles=0,
    )
    position = torch.zeros((1, 3), dtype=torch.float64)
    e0 = mode.electric_field(position, 1.0 / c.resonator.frequency_hz)
    quarter = mode.electric_field(position, 1.25 / c.resonator.frequency_hz)
    assert torch.allclose(e0, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(quarter, torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64), atol=1e-12)

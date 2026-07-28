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

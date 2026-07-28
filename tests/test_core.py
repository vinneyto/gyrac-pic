import math
import torch
from gyrac_pic.domains import BoxDomain, CylindricalPECDomain
from gyrac_pic.particles import deposit_charge_cic, gather_field_cic, relativistic_boris_push
from gyrac_pic.fields.poisson import apply_negative_laplacian, solve_pcg
from gyrac_pic.constants import ELECTRON_MASS, ELEMENTARY_CHARGE


def test_cic_conserves_charge_and_gathers_constant():
    torch.manual_seed(1); shape=(8,9,10); bounds=((-1,1),(-2,2),(-3,3))
    p=torch.rand(100,3,dtype=torch.float64); p=p*torch.tensor((2.,4.,6.))-torch.tensor((1.,2.,3.))
    q=torch.randn(100,dtype=torch.float64)
    rho=deposit_charge_cic(p,q,shape,bounds)
    volume=math.prod((b[1]-b[0])/(n-1) for b,n in zip(bounds,shape))
    assert torch.allclose(rho.sum()*volume,q.sum(),atol=1e-12)
    field=torch.ones((*shape,3),dtype=torch.float64)*torch.tensor((2.,3.,4.))
    assert torch.allclose(gather_field_cic(field,p,bounds),torch.tensor((2.,3.,4.)).expand_as(p))


def test_domains():
    box=BoxDomain((-1,1),(-1,1),(-1,1),(7,7,7)); assert box.interior_mask(torch.device("cpu")).sum()==125
    cylinder=CylindricalPECDomain(1,2,(21,21,21)); mask=cylinder.interior_mask(torch.device("cpu"))
    assert mask[10,10,10] and not mask[0,0,10] and not mask[10,10,0]


def test_pcg_manufactured_solution():
    domain=BoxDomain((-1,1),(-1,1),(-1,1),(12,12,12)); mask=domain.interior_mask(torch.device("cpu"))
    exact=torch.zeros((12,12,12),dtype=torch.float64); exact[1:-1,1:-1,1:-1]=torch.rand((10,10,10),dtype=torch.float64)
    spacing=(2/11,)*3; rhs=apply_negative_laplacian(exact,mask,spacing)
    got,info=solve_pcg(rhs,mask,spacing,relative_tolerance=1e-10,max_iterations=500)
    assert info.converged and torch.allclose(got,exact,atol=1e-8)


def test_boris_free_particle_and_magnetic_energy():
    p=torch.tensor([[0.,0.,0.]]); momentum=torch.tensor([[1e-23,0.,0.]])
    zero=torch.zeros_like(p); magnetic=torch.tensor([[0.,0.,.1]])
    _,m=relativistic_boris_push(p,momentum,zero,magnetic,-ELEMENTARY_CHARGE,ELECTRON_MASS,1e-12)
    assert torch.allclose(torch.linalg.vector_norm(m),torch.linalg.vector_norm(momentum),rtol=1e-6)

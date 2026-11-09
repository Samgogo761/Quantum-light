"""Physical and independent numerical invariants, not implementation snapshots."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ql_tip_1d.core import (A0_NM, EH_EV, Grid, analytic_well, bound_states,
    well_grid, surface_potential, integrated_near_field, enhancement,
    CrankNicolson, norm, fft_free_step)


def test_bound_state_count_and_second_order_energy():
    target=analytic_well()['energy_au']
    errors=[]
    for cells in [12,24,48]:
        g=well_grid(cells)
        energy,states=bound_states(g,surface_potential(g.x_au))
        assert len(energy)==1
        assert abs(norm(states[:,0],g.dx_au)-1)<1e-12
        errors.append(abs(energy[0]-target))
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2


def test_integrated_field_gradient_and_surface_continuity():
    x=np.array([0.1,1.,10.,100.])/A0_NM
    step=1e-3
    derivative=(integrated_near_field(x+step)-integrated_near_field(x-step))/(2*step)
    np.testing.assert_allclose(derivative,enhancement(x),rtol=2e-9,atol=2e-9)
    np.testing.assert_array_equal(integrated_near_field(np.array([-1.,0.])),[0.,0.])
    assert abs(integrated_near_field(np.array([1e-10]))[0])<4e-10


def test_cn_static_phase_including_time_discretization():
    g=well_grid(24)
    energy,states=bound_states(g,surface_potential(g.x_au))
    psi0=states[:,0].astype(complex)
    dt,steps=.2,300
    cn=CrankNicolson(g,dt,surface_potential(g.x_au))
    psi=psi0.copy()
    for _ in range(steps):
        psi=cn.step(psi)
    exact_cn_phase=np.exp(-2j*steps*np.arctan(energy[0]*dt/2))
    assert np.sqrt(norm(psi-exact_cn_phase*psi0,g.dx_au))<1e-10
    # Do not let norm preservation conceal a nonzero phase error to exp(-iHt).
    exact_phase=np.exp(-1j*energy[0]*dt*steps)
    assert np.sqrt(norm(psi-exact_phase*psi0,g.dx_au))>1e-4


def test_cn_absorbing_probability_identity():
    # The CN midpoint continuity identity tests both CAP sign and probability units.
    g=well_grid(12)
    x=g.x_au
    psi=np.exp(-x*x/20).astype(complex)
    psi/=np.sqrt(norm(psi,g.dx_au))
    cap=.01*(1+np.sin(x)**2)
    cn=CrankNicolson(g,.17,-1j*cap)
    after=cn.step(psi)
    mid=(after+psi)/2
    loss=2*.17*np.sum(cap*abs(mid)**2)*g.dx_au
    assert norm(after,g.dx_au)<norm(psi,g.dx_au)
    assert abs(norm(psi,g.dx_au)-norm(after,g.dx_au)-loss)<2e-14


def test_fft_direction_width_and_momentum():
    n,dx=2048,.1
    x=(np.arange(n)-n/2)*dx
    x0,sigma,p0=-12.,4.,.4
    psi=(2*np.pi*sigma*sigma)**(-.25)*np.exp(-(x-x0)**2/(4*sigma*sigma)+1j*p0*x)
    out=fft_free_step(psi,dx,30.)
    rho=abs(out)**2
    centroid=np.sum(x*rho)*dx
    variance=np.sum((x-centroid)**2*rho)*dx
    assert abs(centroid-(x0+p0*30))<1e-11
    assert abs(variance-(sigma*sigma+30**2/(4*sigma*sigma)))<1e-10
    assert abs(norm(out,dx)-1)<1e-13


def test_cn_free_packet_separate_space_and_time_refinement():
    def propagate(dx,dt):
        x=np.arange(-80.,80.,dx)
        g=Grid(x,dx,0)
        psi=(2*np.pi*4**2)**(-.25)*np.exp(-(x+12.)**2/(4*4**2)+.8j*x)
        cn=CrankNicolson(g,dt,np.zeros(len(x)))
        for _ in range(round(20/dt)):
            psi=cn.step(psi)
        return abs(float(np.sum(x*abs(psi)**2)*dx)-4.)
    # Hold one discretization accurate while refining the other independently.
    space=[propagate(dx,.01) for dx in [.4,.2,.1]]
    time=[propagate(.01,dt) for dt in [.8,.4,.2]]
    assert space[0]>3.8*space[1]>3.8**2*space[2]
    assert time[0]>3.8*time[1]>3.8**2*time[2]

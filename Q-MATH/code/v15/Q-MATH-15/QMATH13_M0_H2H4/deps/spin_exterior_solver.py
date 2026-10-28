#!/usr/bin/env python3
"""Independent, spin-factorized shared-two-k M0 unitary propagator.
Electron tensor=(Lambda^2 C4_up ⊗ Lambda^2 C4_down)_k ⊗(... )_-k, 6^4=1296.
Photon SAME pump + H2 parallel/perpendicular across all four factors.
No claim that finite switched cavity modes are asymptotic photons.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import time, json, sys
from itertools import combinations
import numpy as np
from scipy import linalg as la
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import q8_m0_model as m
import qmath04_two_pol as old
from shared import photon_data
PAIRS=np.array(list(combinations(range(4),2)),dtype=int)
SPINS=[[0,2,4,6],[1,3,5,7]]

def wedge2(U):
    # U shape (...,4,4), returns exterior-square unitary (...,6,6)
    A=PAIRS[:,0]; B=PAIRS[:,1]
    return (U[...,A[:,None],A[None,:]]*U[...,B[:,None],B[None,:]]-
            U[...,A[:,None],B[None,:]]*U[...,B[:,None],A[None,:]])

def transform_photons(arr, mats):
    """Apply local tensor matrices on last 3 axes, preserving other axes."""
    for axis,M in zip((-3,-2,-1),mats):
        ax=arr.ndim+axis
        arr=np.moveaxis(np.tensordot(arr,M,axes=([ax],[0])), -1, ax)
    return arr

def spin_initial(h4):
    _,vectors=la.eigh(h4)
    Q=vectors[:,:2]
    return Q[PAIRS[:,0],0]*Q[PAIRS[:,1],1]-Q[PAIRS[:,0],1]*Q[PAIRS[:,1],0]

def eval_from_states(psi,nump,nharm):
    B=psi.shape[0]
    pp=psi.reshape(B,1296,nump,nharm*nharm)
    rhos=np.einsum('beai,beaj->bij',pp,pp.conj(), optimize=True)
    a,_,_=old.oscillator(nharm)
    an=np.kron(a,np.eye(nharm));bn=np.kron(np.eye(nharm),a)
    oa=an.conj().T@an; ob=bn.conj().T@bn
    out=[]
    for k,rho in enumerate(rhos):
        ev=lambda o:np.trace(rho@o)
        ba=ev(an);bb=ev(bn)
        mcr=ev(an@bn)-ba*bb
        na=float(ev(oa).real);nb=float(ev(ob).real)
        nfa=na-abs(ba)**2;nfb=nb-abs(bb)**2
        maa=ev(an@an)-ba**2
        mbb=ev(bn@bn)-bb**2
        pd=np.real(np.diag(rho).reshape(nharm,nharm))
        par_edge=float(pd[-1,:].sum())
        perp_edge=float(pd[:,-1].sum())
        vacuum_comm_par=1-nharm*par_edge
        vacuum_comm_perp=1-nharm*perp_edge
        varpar=float(nfa+vacuum_comm_par/2-abs(maa))
        varperp=float(nfb+vacuum_comm_perp/2-abs(mbb))
        pt=rho.reshape(nharm,nharm,nharm,nharm).transpose(0,3,2,1).reshape(nharm**2,nharm**2)
        neg_spec=np.linalg.eigvalsh((pt+pt.conj().T)/2)
        negativity=float(np.sum(np.maximum(-neg_spec,0)))
        out.append(dict(n_parallel=na,n_perp=nb,
                beta_parallel=[ba.real,ba.imag],beta_perp=[bb.real,bb.imag],
                m_cross=[mcr.real,mcr.imag],raw_bb=[(mcr+ba*bb).real,(mcr+ba*bb).imag],
                n_fluct_parallel=float(nfa),n_fluct_perp=float(nfb),
                m_single_parallel=[float(maa.real),float(maa.imag)],
                m_single_perp=[float(mbb.real),float(mbb.imag)],
                varmin_parallel=varpar,varmin_perp=varperp,
                photon_comm_parallel=float(vacuum_comm_par),photon_comm_perp=float(vacuum_comm_perp),
                negativity=negativity,pt_min_eigenvalue=float(neg_spec[0]),
                rho_min_eigenvalue=float(np.linalg.eigvalsh(rho)[0]),
                classical_bound_ratio=float(abs(mcr)**2/max(nfa*nfb,1e-90)),
                photon_corner=float(np.diag(rho).real.reshape(nharm,nharm)[-1,:].sum()+np.diag(rho).real.reshape(nharm,nharm)[:,-1].sum()),
                pump_edge=float(abs(pp[k,:,-1,:]).__pow__(2).sum()),
                norm=float(abs(np.trace(rho)-1.))))
    return out

def calc(npump=11,nharm=3,nsteps=80,phase=np.pi/2,typ='seeded',mean=.7,nsqueeze=.2,omega=.18,
         ncycles=2,azp=.05,azh=.005,neel=1,kfrac=(.4158,.1683),two_parities=True):
    t0=time.perf_counter(); geo=m.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in')
    k=geo.k_cart(kfrac)
    modes=[]
    for kk in (k,-k):
        hh=m.h8(geo,m.R0,kk,neel)
        modes.extend([spin_initial(hh[np.ix_(S,S)]) for S in SPINS])
    el=modes[0]
    for state in modes[1:]:el=np.kron(el,state)
    assert abs(la.norm(el)-1)<1e-12
    phases=[phase,phase+np.pi] if two_parities else [phase]
    pump_states=[]; initial_info=[]
    for phi in phases:
        pu,nbar,edge=old.pump_ket(npump,typ,mean,nsqueeze,phi)
        pump_states.append(pu);initial_info.append(dict(phase=float(phi),actual_mean=float(nbar),input_edge=float(edge)))
    B=len(phases)
    ap,qp0,Vp=old.oscillator(npump)
    aa,qa0,Va=old.oscillator(nharm)
    ax,qx0,Vx=old.oscillator(nharm)
    qp=np.repeat(qp0,nharm*nharm)
    qa=np.tile(np.repeat(qa0,nharm),npump)
    qx=np.tile(qx0,npump*nharm)
    nfree=(np.arange(npump)[:,None,None]+2*np.arange(nharm)[None,:,None]+2*np.arange(nharm)[None,None,:]).reshape(-1)
    phdim=len(nfree)
    ep=np.array(geo.a2_cart[:2]);ep/=la.norm(ep)
    et=np.array([ep[1],-ep[0]])
    shifts=azp*qp[:,None]*ep + azh*qa[:,None]*ep+azh*qx[:,None]*et
    vacuum=np.zeros(nharm*nharm,complex);vacuum[0]=1
    psi=np.stack([np.outer(el,np.kron(pu,vacuum)) for pu in pump_states],axis=0)
    T=ncycles*(2*np.pi/omega);dt=T/nsteps
    free=np.exp(-.5j*dt*omega*nfree)
    norm_max=0.;pe_max=0
    for it in range(nsteps):
        gate=np.sin(np.pi*(it+.5)/nsteps)**2
        hs_plus=m.h8(geo,m.R0,k+gate*shifts,neel)
        hs_minus=m.h8(geo,m.R0,-k+gate*shifts,neel)
        Us=[]
        for hh in (hs_plus,hs_minus):
            for S in SPINS:
                h=hh[...,S,:][...,S]
                eig,V=np.linalg.eigh(h)
                single=np.einsum('zai,zi,zbi->zab',V,np.exp(-1j*dt*eig),V.conj(),optimize=True)
                Us.append(wedge2(single))
        y=(psi*free[None,None,:]).reshape(B,1296,npump,nharm,nharm)
        y=transform_photons(y,(Vp.conj(),Va.conj(),Vx.conj()))
        q=y.reshape(B,6,6,6,6,phdim).transpose(0,5,1,2,3,4)
        for axis,U in enumerate(Us):
            v=np.moveaxis(q,2+axis,2)
            v=np.einsum('zai,bzid->bzad',U,v.reshape(B,phdim,6,-1),optimize=True).reshape(B,phdim,6,6,6,6)
            q=np.moveaxis(v,2,2+axis)
        q=q.transpose(0,2,3,4,5,1).reshape(B,1296,phdim)
        q=transform_photons(q.reshape(B,1296,npump,nharm,nharm),(Vp.T,Va.T,Vx.T))
        psi=q.reshape(B,1296,phdim)*free[None,None,:]
        if (it+1)%max(1,nsteps//5)==0:
            norm_max=max(norm_max,float(np.max(np.abs(np.sum(abs(psi)**2,axis=(1,2))-1))))
            pe_max=max(pe_max,float(np.max(np.sum(abs(psi.reshape(B,1296,npump,nharm*nharm)[:,:,-1,:])**2,axis=(1,2)))))
    outputs=eval_from_states(psi,npump,nharm)
    for i,o in enumerate(outputs):o.update(initial_info[i])
    odds={}
    if two_parities:
        A,Bv=outputs
        for name in ('m_cross','raw_bb'):
            x=complex(*A[name]);y=complex(*Bv[name]);z=(x-y)/2
            odds[name]=[float(z.real),float(z.imag)]
        odds['abs_m_odd']=float(abs(complex(*odds['m_cross'])))
    return dict(params=dict(npump=npump,nharm=nharm,nsteps=nsteps,phase=phase,typ=typ,mean=mean,nsqueeze=nsqueeze,neel=neel,kfrac=kfrac),
                outputs=outputs,odd=odds,norm_max=norm_max,pump_edge_max=pe_max,wall_sec=round(time.perf_counter()-t0,2))

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--npump',type=int,default=11);a.add_argument('--nharm',type=int,default=3);a.add_argument('--steps',type=int,default=80);a.add_argument('--type',default='seeded');a.add_argument('--phase',type=float,default=float(np.pi/2));a.add_argument('--out',default='spin_solver.json');z=a.parse_args()
    d=calc(npump=z.npump,nharm=z.nharm,nsteps=z.steps,typ=z.type,phase=z.phase)
    Path(z.out).write_text(json.dumps(d,indent=2,ensure_ascii=False));print(json.dumps({'odd':d['odd'],'outputs':d['outputs'],'wall':d['wall_sec'],'norm':d['norm_max']},ensure_ascii=False,indent=2),flush=True)

#!/usr/bin/env python3
"""Q-MATH-04: real M0 4e + one quantized pump + H2 two orthogonal photon modes.
Both harmonic modes are canonical truncated Fock oscillators, not far-field outgoing modes.
All two-polarization modes couple through spectrally evaluated Peierls h8(k+quantum A).
"""
from __future__ import annotations
import json, sys, time, math, argparse
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy import linalg as la
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import q8_m0_model as m
import qmath03_exact_unitary as previous

@lru_cache(maxsize=None)
def oscillator(n):
    a=previous.annihilator(n)
    d,V=la.eigh(a+a.conj().T)
    return a,d,V

def pump_ket(n,type_,mean=0.7,nsqueeze=0.2,phase=0.,squeeze_phase=0.):
    a,_,_=oscillator(n)
    if type_=='vacuum':
        v=np.eye(n,dtype=complex)[:,0]
    elif type_=='coherent':
        alpha=np.sqrt(mean)*np.exp(1j*phase)
        v=la.expm(alpha*a.conj().T-alpha.conjugate()*a)[:,0]
    elif type_ in ('seeded','bsv'):
        n_sq=(mean if type_=='bsv' else nsqueeze)
        alpha=(0 if type_=='bsv' else np.sqrt(mean-n_sq)*np.exp(1j*phase))
        r=np.arcsinh(np.sqrt(n_sq))
        z=r*np.exp(1j*squeeze_phase)
        Sq=la.expm(0.5*(z.conjugate()*(a@a)-z*(a.conj().T@a.conj().T)))
        vac=Sq[:,0]
        v=vac if type_=='bsv' else la.expm(alpha*a.conj().T-alpha.conjugate()*a)@vac
    else: raise ValueError(type_)
    v=v/la.norm(v)
    input_p=np.real(np.sum(np.arange(n)*abs(v)**2))
    edge=np.real(abs(v[-1])**2)
    return v, float(input_p),float(edge)


def photon_ops(nump,np_,nx_):
    ap,qp,Vp=oscillator(nump); aa,qa,Va=oscillator(np_); ax,qx,Vx=oscillator(nx_)
    # coordinate eigenmode k_index corresponds to pump,parallel,perpendicular
    W=np.kron(np.kron(Vp,Va),Vx)
    pp=np.repeat(qp,np_*nx_)
    pa=np.tile(np.repeat(qa,nx_),nump)
    px=np.tile(qx,nump*np_)
    P=np.eye(nump)
    A=np.kron(aa,np.eye(nx_));B=np.kron(np.eye(np_),ax)
    Npar=np.kron(aa.conj().T@aa,np.eye(nx_)); Nper=np.kron(np.eye(np_),ax.conj().T@ax)
    nfree=(np.arange(nump)[:,None,None]+2*np.arange(np_)[None,:,None]+2*np.arange(nx_)[None,None,:]).reshape(-1)
    return W,pp,pa,px,A,B,Npar,Nper,nfree


def eframe_and_h(geo,k,neel,G,basis):
    return previous.build_initial_e(geo,k,neel,G,basis)[0]


def compute(geo,G,basis,*,neel=1,kfrac=(.4158,.1683),type_='seeded',mean=.7,nsqueeze=.2,phase=0.,squeeze_phase=0.,npump=19,nharm=3,nsteps=120,azp=.05,azh=.005,omega=.18,ncycles=2,diagnostic=False):
    start=time.monotonic()
    k=geo.k_cart(kfrac)
    ep=np.asarray(geo.a2_cart[:2]);ep=ep/np.linalg.norm(ep)
    eh=np.array([ep[1],-ep[0]])
    fock,in_mean,in_edge=pump_ket(npump,type_,mean,nsqueeze,phase,squeeze_phase)
    E=eframe_and_h(geo,k,neel,G,basis)
    basis0=np.eye(nharm,dtype=complex)[:,0]
    state=np.kron(E,np.kron(fock,np.kron(basis0,basis0))).reshape(36,-1)
    W,qp,qa,qx,A,B,Npar,Nper,nfree=photon_ops(npump,nharm,nharm)
    T=2*np.pi*ncycles/omega;dt=T/nsteps
    half=np.exp(-.5j*dt*omega*nfree)
    shifts=azp*qp[:,None]*ep+azh*qa[:,None]*ep+azh*qx[:,None]*eh
    Hc=m.h8(geo,m.R0,k[None,:],neel)[0]
    max_norm=0.; max_pump_edge=0.; max_h2_edge=0.
    for it in range(nsteps):
        tm=(it+.5)*dt
        gate=np.sin(np.pi*tm/T)**2
        h=m.h8(geo,m.R0,k[None,:]+gate*shifts,neel)
        hel=np.einsum('zij,ijab->zab',h,G,optimize=True)
        lam,vec=np.linalg.eigh(hel)
        q=(state*half[None,:])@W.conj()
        # phase-exact unitary spin sector at each photon coordinate z
        q=np.einsum('zai,zi,zbi,bz->az',vec,np.exp(-1j*dt*lam),vec.conj(),q,optimize=True)
        state=(q@W.T)*half[None,:]
        if diagnostic and (it+1)%max(1,nsteps//8)==0:
            max_norm=max(max_norm,float(abs(np.vdot(state,state).real-1)))
            pp=abs(state.reshape(36,npump,nharm,nharm))**2
            max_pump_edge=max(max_pump_edge,float(pp[:,-1,:,:].sum()))
            max_h2_edge=max(max_h2_edge,float(pp[:,:,-1,:].sum()+pp[:,:,:,-1].sum()))
    data=state.reshape(36,npump,nharm*nharm).reshape(36*npump,nharm*nharm)
    rho=data.T @data.conj()
    rho=(rho+rho.conj().T)/2
    def ev(op):return np.trace(rho@op)
    beta_a=ev(A); beta_b=ev(B)
    n_a=ev(Npar).real;n_b=ev(Nper).real
    x=ev(A.conj().T@B)-beta_a.conjugate()*beta_b
    s=ev(A@B)-beta_a*beta_b
    ma=ev(A@A)-beta_a**2
    mb=ev(B@B)-beta_b**2
    # Exact (truncated-mode) partial transpose in second H2 polarization.
    pt=rho.reshape(nharm,nharm,nharm,nharm).transpose(0,3,2,1).reshape(nharm**2,nharm**2)
    vals=np.linalg.eigvalsh((pt+pt.conj().T)/2)
    negativity=float(np.sum(np.maximum(-vals,0)))
    # Minimal single-mode quadrature variances: vacuum is 1/2.
    var_a=.5+n_a-abs(ma)
    var_b=.5+n_b-abs(mb)
    n2a=ev(Npar@(Npar-np.eye(nharm*nharm))).real
    n2b=ev(Nper@(Nper-np.eye(nharm*nharm))).real
    distp=np.real(np.diag(rho).reshape(nharm,nharm)).sum(axis=1)
    distx=np.real(np.diag(rho).reshape(nharm,nharm)).sum(axis=0)
    end_prob=abs(state.reshape(36,npump,nharm,nharm))**2
    return dict(type=type_,N=neel,k_frac=list(kfrac),phase=phase,squeeze_phase=squeeze_phase,
                nominal_mean=mean,actual_mean=in_mean,input_pump_edge=in_edge,nsqueeze=nsqueeze,
                n_parallel=float(n_a),n_perpendicular=float(n_b),
                g2_parallel=float(n2a/n_a**2) if n_a>1e-13 else None,
                g2_perpendicular=float(n2b/n_b**2) if n_b>1e-13 else None,
                beta_parallel=[float(beta_a.real),float(beta_a.imag)],beta_perpendicular=[float(beta_b.real),float(beta_b.imag)],
                n_cross=[float(x.real),float(x.imag)],m_cross=[float(s.real),float(s.imag)],
                cov_xx=float(x.real+s.real),
                m_parallel=[float(ma.real),float(ma.imag)],m_perpendicular=[float(mb.real),float(mb.imag)],
                var_min_parallel=float(var_a),var_min_perpendicular=float(var_b),
                two_mode_negativity=negativity,pt_min_eigenvalue=float(min(vals)),
                normal_classical_cauchy_gap=float(abs(s)**2-n_a*n_b),
                prob_par=distp.tolist(),prob_per=distx.tolist(),
                output_pump_edge=float(end_prob[:,-1,:,:].sum()),output_harm_edge=float(end_prob[:,:,-1,:].sum()+end_prob[:,:,:,-1].sum()),
                max_pump_edge_during=max_pump_edge,max_harm_edge_during=max_h2_edge,
                norm_err=float(abs(np.trace(rho).real-1)),max_norm_error_during=max_norm,
                dims=[36,npump,nharm,nharm],nsteps=nsteps,elapsed_s=round(time.monotonic()-start,3))


def main():
    args=argparse.ArgumentParser();args.add_argument('--quick',action='store_true');args.add_argument('--out',default=str(BASE/'quick_seed.json'));ns=args.parse_args()
    geo=m.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in')
    basis,G=previous.make_ops()
    settings=dict(npump=15 if ns.quick else 21,nharm=3,nsteps=64 if ns.quick else 160)
    rows=[]
    for typ in ('vacuum','coherent','bsv','seeded'):
        for phase in ([0.] if typ!='seeded' else [0.,np.pi/2,np.pi,3*np.pi/2]):
            for N in (1,-1):
                v=compute(geo,G,basis,type_=typ,phase=phase,neel=N,diagnostic=True,**settings)
                rows.append(v)
                print(f'{typ:8} N={N:+} phi={phase:.3f} npar={v["n_parallel"]:.8g} nperp={v["n_perpendicular"]:.8g} m={v["m_cross"]} elapsed={v["elapsed_s"]}',flush=True)
    Path(ns.out).write_text(json.dumps({'params':settings,'results':rows},indent=2),encoding='utf-8')
if __name__=='__main__':main()

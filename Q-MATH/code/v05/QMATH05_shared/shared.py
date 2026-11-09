#!/usr/bin/env python3
"""Two k-block M0 with a SINGLE shared pump + two H2 polarizations.
Hilbert = (wedge^2 C4_up tensor wedge^2 C4_down)^{tensor 2} tensor Fpump tensor Fpar tensor Fperp.
Switched finite mode test, not far field or full BZ. No interaction between electrons except sharing photon operators.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys, math, json,time
from pathlib import Path
import numpy as np
from scipy import linalg as la
from functools import lru_cache
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import q8_m0_model as m
import qmath03_exact_unitary as old
import qmath04_two_pol as prev

@lru_cache(maxsize=10)
def photon_data(npump,nharm):
    ap,qp,Vp=prev.oscillator(npump)
    aa,qa,Va=prev.oscillator(nharm)
    ax,qx,Vx=prev.oscillator(nharm)
    W=np.kron(np.kron(Vp,Va),Vx)
    qp=np.repeat(qp,nharm*nharm)
    qa=np.tile(np.repeat(qa,nharm),npump)
    qx=np.tile(qx,npump*nharm)
    n=npump*nharm*nharm
    na=np.kron(aa,np.eye(nharm))
    nb=np.kron(np.eye(nharm),ax)
    nfree=(np.arange(npump)[:,None,None]+2*np.arange(nharm)[None,:,None]+2*np.arange(nharm)[None,None,:]).reshape(-1)
    return W,qp,qa,qx,na,nb,nfree

def exp36(h,dt):
    eig,U=np.linalg.eigh(h)
    return np.einsum('zij,zj,zkj->zik',U,np.exp(-1j*dt*eig),U.conj(),optimize=True)

def calc(neel,typ='seeded',phase=0., kfrac=(0.4158,0.1683),npump=9,nharm=3,nsteps=48,
         omega=.18,ncycles=2,azp=.05,azh=.005,mean=.7,nsqueeze=.2,weight=1.,
         diagnostic=True,solo=False):
    t0=time.perf_counter()
    geo=m.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in')
    basis,G=old.make_ops()
    k=geo.k_cart(kfrac)
    initial_k=old.build_initial_e(geo,k,neel,G,basis)[0]
    initial_mk=old.build_initial_e(geo,-k,neel,G,basis)[0]
    D=36 if solo else 1296
    ph,inmean,inedge=prev.pump_ket(npump,typ,mean,nsqueeze,phase)
    w,qp,qa,qx,an,bn,nfree=photon_data(npump,nharm)
    nmode=len(nfree)
    ep=np.array(geo.a2_cart[:2]);ep/=la.norm(ep)
    et=np.array([ep[1],-ep[0]])
    shifts=azp*qp[:,None]*ep+azh*qa[:,None]*ep+azh*qx[:,None]*et
    vacuum=np.zeros(nharm*nharm,complex);vacuum[0]=1
    el=initial_k if solo else np.kron(initial_k,initial_mk)
    psi=np.outer(el,np.kron(ph,vacuum))
    T=ncycles*(2*np.pi/omega);dt=T/nsteps
    free=np.exp(-.5j*dt*omega*nfree)
    norm_errors=[]
    for t in range(nsteps):
        gate=np.sin(np.pi*(t+.5)/nsteps)**2
        hs1=m.h8(geo,m.R0,k+gate*shifts,neel)
        H1=np.einsum('zij,ijab->zab',hs1,G,optimize=True)
        U1=exp36(H1,dt)
        if not solo:
            hs2=m.h8(geo,m.R0,-k+gate*shifts,neel)
            H2=np.einsum('zij,ijab->zab',hs2,G,optimize=True)
            U2=exp36(H2,dt)
        y=(psi*free[None,:])@w.conj()
        if solo:
            q=np.einsum('zai,iz->az',U1,y,optimize=True)
        else:
            q=y.reshape(36,36,nmode).transpose(2,0,1)
            q=np.einsum('zai,zij,zbj->zab',U1,q,U2,optimize=True)
            q=q.transpose(1,2,0).reshape(D,nmode)
        psi=(q@w.T)*free[None,:]
        if diagnostic and (t+1)%max(1,nsteps//4)==0:
            norm_errors.append(float(abs(np.vdot(psi,psi).real-1.)))
    yr=psi.reshape(D,npump,nharm*nharm).reshape(D*npump,nharm*nharm)
    rho=yr.T@yr.conj();rho=(rho+rho.conj().T)/2
    def ev(x):return np.trace(rho@x)
    ba=ev(an); bb=ev(bn)
    na=ev(an.conj().T@an).real;nb=ev(bn.conj().T@bn).real
    mc=ev(an@bn)-ba*bb
    nc=ev(an.conj().T@bn)-ba.conjugate()*bb
    edgeb=np.real(np.diag(rho).reshape(nharm,nharm))
    pumpedge=float((abs(psi.reshape(D,npump,nharm*nharm))**2)[:,-1,:].sum())
    return dict(N=neel,typ=typ,phase=phase,solo=solo,kfrac=list(kfrac),pumpdim=npump,harmdim=nharm,nsteps=nsteps,
                n_parallel=float(na),n_perp=float(nb),m_cross=[float(mc.real),float(mc.imag)],n_cross=[float(nc.real),float(nc.imag)],
                beta_parallel=[float(ba.real),float(ba.imag)],beta_perp=[float(bb.real),float(bb.imag)],
                photon_corner=float(edgeb[-1,:].sum()+edgeb[:,-1].sum()),pump_edge=pumpedge,input_edge=inedge,actual_mean=inmean,
                norm_error=max(norm_errors) if norm_errors else None,wall=round(time.perf_counter()-t0,2))

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--npump',type=int,default=9);a.add_argument('--nharm',type=int,default=3);a.add_argument('--steps',type=int,default=48);a.add_argument('--out',default='results.json');v=a.parse_args()
    rec=[]
    for typ,phi in [('vacuum',0.),('bsv',0.),('coherent',np.pi/2),('seeded',np.pi/2),('seeded',0.)]:
        for neel in [1,-1]:
            r=calc(neel,typ,phi,npump=v.npump,nharm=v.nharm,nsteps=v.steps)
            rec.append(r)
            print(typ,round(phi,3),neel,r['n_parallel'],r['n_perp'],r['m_cross'],'norm',r['norm_error'],'dt',r['wall'],flush=True)
    Path(v.out).write_text(json.dumps(rec,indent=2,ensure_ascii=False))

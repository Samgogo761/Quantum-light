#!/usr/bin/env python3
"""Finite-mode exactly unitary (within Fock cutoffs) M0 8-band, four-electron quantum-light benchmark.
Electron single-k spin conserving 2up+2down (36D) + pump + H2 photons.
Fully quantized Peierls coupling h8(k + env(t)[A_zpf,p Xp ep + A_zpf,h Xh eh]);
no Gaussian output assumption. Photon modes are switched free after envelope.
"""
import os, json, argparse, time, math
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, expm_multiply
import sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import q8_m0_model as m

def fermion_cij(basis,i,j):
    idx={v:a for a,v in enumerate(basis)}; n=len(basis); O=np.zeros((n,n),complex)
    for c, mask in enumerate(basis):
        if not(mask&(1<<j)):continue
        s1=(-1)**((mask & ((1<<j)-1)).bit_count()); inter=mask^(1<<j)
        if inter&(1<<i):continue
        s2=(-1)**((inter & ((1<<i)-1)).bit_count())
        O[idx[inter|(1<<i)],c]=s1*s2
    return O

def make_ops():
    up=[0,2,4,6];down=[1,3,5,7]
    basis=sorted((sum(1<<x for x in su)+sum(1<<x for x in sd) for su in __import__('itertools').combinations(up,2) for sd in __import__('itertools').combinations(down,2)))
    G=np.zeros((8,8,36,36),complex)
    for i in range(8):
        for j in range(8):
            if (i-j)%2==0:G[i,j]=fermion_cij(basis,i,j)
    return basis,G

def annihilator(n):return np.diag(np.sqrt(np.arange(1,n)),1).astype(complex)

def squeezed_vac(nbar,n):
    r=np.arcsinh(np.sqrt(nbar)); v=np.zeros(n,complex)
    for m in range((n+1)//2):
        n2=2*m
        v[n2]=(-np.tanh(r))**m * np.exp(0.5*(math.lgamma(2*m+1)-2*math.lgamma(m+1)) - m*np.log(2))/np.sqrt(np.cosh(r))
    before=np.vdot(v,v).real
    return v/np.linalg.norm(v),1-before

def coherent(nbar,n):
    alpha=np.sqrt(nbar); v=np.array([alpha**i/math.sqrt(math.factorial(i)) for i in range(n)],complex)*np.exp(-nbar/2)
    loss=max(0.0,1-np.vdot(v,v).real)
    return v/np.linalg.norm(v),loss

def build_initial_e(geo,k,neel,G,basis):
    h=m.h8(geo,m.R0,k,neel)
    en,u=la.eigh(h)
    occ=u[:,:4]
    he=np.einsum('ij,ijab->ab',h,G,optimize=True)
    es,vs=la.eigh(he)
    phi=vs[:,0]
    # For Slater occupation projection, occupied frame minors
    frame=np.zeros(36,complex)
    for q,mask in enumerate(basis):
        rows=[i for i in range(8) if mask&(1<<i)]
        frame[q]=np.linalg.det(occ[rows,:])
    err=min(np.linalg.norm(phi-frame),np.linalg.norm(phi+frame))
    # phase insensitive
    overlap=abs(np.vdot(phi,frame)); energy_span=np.real(np.vdot(frame,he@frame))
    assert abs(1-overlap)<1e-8, f'wrong frame {overlap}'
    return frame/np.linalg.norm(frame),float(en[4]-en[3]),float(energy_span),float(abs(1-overlap))

def simulate(geo,G,basis,k,neel,typ,npump,nout,mean,azp,azout,omega,ncycles,nsteps,det_polar):
    t0=time.monotonic(); N=npump*nout; k=np.asarray(k,float)
    ep=np.asarray(geo.a2_cart[:2]);ep=ep/np.linalg.norm(ep)
    eh= ep if det_polar=='parallel' else np.array([ep[1],-ep[0]])
    ap=annihilator(npump);ao=annihilator(nout)
    xp=ap+ap.conj().T; xo=ao+ao.conj().T
    qa,Wa=la.eigh(xp);qb,Wb=la.eigh(xo)
    W=np.kron(Wa,Wb)
    assert np.linalg.norm(W.conj().T@W-np.eye(N))<1e-12
    Avalues=(azp*np.repeat(qa,nout)[:,None]*ep + azout*np.tile(qb,npump)[:,None]*eh)
    ed, gap, energy, initial_overlap_error=build_initial_e(geo,k,neel,G,basis)
    pu,tail=(coherent(mean,npump) if typ=='coherent' else squeezed_vac(mean,npump) if typ=='squeezed' else ((np.eye(npump,dtype=complex)[int(typ.split(':')[1])],0.) if typ.startswith('fock:') else (np.r_[1.+0j,np.zeros(npump-1)],0.)))
    psi=np.kron(ed,np.kron(pu,np.r_[1.+0j,np.zeros(nout-1)]))
    # energy & operators in number basis
    nf=np.kron(np.diag(np.arange(npump)),np.eye(nout)); nh=np.kron(np.eye(npump),np.diag(np.arange(nout)))
    hb=omega*nf+(2*omega)*nh
    tt=2*np.pi*ncycles/omega; dt=tt/nsteps
    # Strang splitting of photon free vs electron Peierls coupling; photons remain canonical as finite oscillator limit, test cutoff.
    # H_e(A) quadrature spectral diagonal; approximate exp(-idt H_e) via eigendecomp in each Q-sector; exact per midpoint for electron operator.
    # Free field applied half-step in number basis; fast and unitarity-preserving time propagator.
    free_half=np.exp(-0.5j*dt*np.diag(hb))
    maxnorm=0; maxexc=0
    for it in range(nsteps):
        tm=(it+0.5)*dt
        gate=np.sin(np.pi*tm/tt)**2
        pts=k[None,:] + gate*Avalues
        hs=m.h8(geo,m.R0,pts,neel)
        Hm=np.einsum('kij,ijab->kab',hs,G,optimize=True)
        # evolve one time step exact electronic at each quadrature-sector eigenvalue
        eig,uv=np.linalg.eigh(Hm)
        # State electron x photon Fock; left half step free, transform to quadrature rep
        y=psi.reshape(36,N)*free_half[None,:]
        y=y@W.conj()
        # U_k(t) each photon sector
        y=np.einsum('kaj,kj,kbj,bk->ak',uv,np.exp(-1j*dt*eig),uv.conj(),y,optimize=True)
        y=(y@W.T)*free_half[None,:]
        psi=y.reshape(-1)
        if (it+1)%max(1,nsteps//8)==0:
            maxnorm=max(maxnorm,abs(np.vdot(psi,psi).real-1))
            prob=np.sum(abs(y.reshape(36,npump,nout))**2,axis=(0,2))
            maxexc=max(maxexc,float(prob[-1]))
    # reduced photon nout density, trace over electron + pump
    t=psi.reshape(36,npump,nout)
    rho=np.einsum('epi,epj->ij',t,t.conj())
    rho=(rho+rho.conj().T)/2
    prob=np.real(np.diag(rho));ph=np.sum(prob*np.arange(nout)); n2=np.sum(prob*np.arange(nout)*(np.arange(nout)-1))
    bmean=np.sum(np.diag(rho,1)*np.sqrt(np.arange(1,nout))) # trace rho*a =? index rho_{j,j+1} <a> =sum rho_{j+1,j} sqrt(j+1), so conjugate, magnitude only
    pumpP=np.sum(abs(t)**2,axis=(0,2))
    res={"neel":neel,"pump":typ,"detector":det_polar,"k_frac":[0.4158,0.1683],"npump":npump,"nout":nout,"ncycles":ncycles,"nsteps":nsteps,"omega":omega,"mean_input_nominal":mean,"mean_input_actual":float(np.sum(np.arange(npump)*abs(pu)**2)),"input_tail_lost":float(tail),"input_parity_odd_prob":float(np.sum(abs(pu[1::2])**2)),"azp":azp,"azout":azout,"initial_band_gap_eV":gap,"initial_e_energy":energy,"frame_error":initial_overlap_error,"P_H2":prob.tolist(),"n_H2":float(ph),"g2_H2":float(n2/ph**2) if ph>1e-14 else None,"factorial2_H2":float(n2),"abs_mean_b_H2":float(abs(bmean)),"purity_H2":float(np.real(np.trace(rho@rho))),"pump_n_final":float(np.sum(np.arange(npump)*pumpP)),"pump_edge_probability":float(pumpP[-1]),"H2_edge_probability":float(prob[-1]),"max_mid_norm_error":maxnorm,"max_pump_edge_intermediate":maxexc,"elapsed_s":round(time.monotonic()-t0,2)}
    return res

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--results',default=str(BASE/'results.json'));a=p.parse_args()
    geo=m.build_geometry(Path(__file__).resolve().parent/'CrI3_scf_soc_a1h_clean.in')
    basis,G=make_ops()
    k=geo.k_cart([.4158,.1683]);
    quick=a.quick
    run=[]
    configs=[('coherent',1),('squeezed',1),('vacuum',1),('coherent',-1),('squeezed',-1),('vacuum',-1)]
    for typ,N in configs:
        d=simulate(geo,G,basis,k,N,typ, npump=13 if quick else 25, nout=4 if quick else 5, mean=.7,azp=.05,azout=.005,omega=.18,ncycles=2,nsteps=80 if quick else 160,det_polar='parallel')
        run.append(d)
        print(f"{typ:10s} N={N:+d} nH2={d['n_H2']:.7g} g2={d['g2_H2']:.5g} edge={d['H2_edge_probability']:.3g} pump_edge={d['pump_edge_probability']:.3g} time={d['elapsed_s']:.2f}s",flush=True)
    Path(a.results).write_text(json.dumps({'model': 'M0 4-electron single k, quantum pump+H2, unitary quadrature Peierls, switched coupling', 'note':'not far-field HHG', 'records':run},indent=2),encoding='utf-8')
if __name__=='__main__':main()

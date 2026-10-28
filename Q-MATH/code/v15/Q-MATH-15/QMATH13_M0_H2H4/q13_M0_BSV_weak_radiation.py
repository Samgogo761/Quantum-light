#!/usr/bin/env python3
"""Q-MATH-13: nonperturbative pure BSV pump and *leading* weak outgoing radiation.
Full production M0 at a fixed k in 2-up/2-down four-electron Slater sector.
The Hilbert dimension is 36*Np and all Peierls bonds use exact Fock projections.
Outputs are O(epsilon) complex amplitudes for finite detector mode profiles,
NOT a true far-field solution nor full photon statistics.
"""
import os
for x in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[x]='1'
import sys, json, argparse, time, math
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import norm
from scipy.special import gammaln
ROOT=Path(__file__).resolve().parent
SOURCE=(ROOT/'deps') if (ROOT/'deps'/'qmath07_projected_shared.py').exists() else (ROOT.parent/'qmath13_work'/'q07'/'QMATH07_projected')
sys.path.insert(0,str(SOURCE))
import qmath07_projected_shared as q
import q8_m0_model as m
from spin_exterior_solver import SPINS, spin_initial

def bsv_state(npump,mean=.7,phase=0):
    r=np.arcsinh(np.sqrt(mean)); t=np.tanh(r)
    j=np.arange((npump+1)//2); nn=2*j
    vv=np.zeros(npump,complex)
    vv[nn]=(-t*np.exp(1j*phase))**j*np.exp(0.5*gammaln(nn+1)-gammaln(j+1)-j*np.log(2))/np.sqrt(np.cosh(r))
    tail=float(max(0,1-np.vdot(vv,vv).real))
    vv/=norm(vv)
    return vv, float(np.dot(np.arange(npump),abs(vv)**2)),tail

class M0Source:
    def __init__(self,npump,neel,kfrac=(.4158,.1683),pump_coupling=.05,omega=.18,nk=1):
        self.np=npump;self.omega=omega;self.nk=nk;self.neel=neel
        self.geo=m.build_geometry(SOURCE/'CrI3_scf_soc_a1h_clean.in')
        offs,terms,a2=q.electron_terms(tuple(kfrac),neel,nk)
        self.terms=list(terms.items()); self.offs=offs;self.nE=36**nk
        self.ep=np.asarray(a2[:2],float);self.ep/=norm(self.ep)
        self.et=np.array([self.ep[1],-self.ep[0]])
        self.azp=pump_coupling
        self.base=sp.kron(offs,sp.eye(npump,format='csr'),format='csr')+sp.kron(sp.eye(self.nE,format='csr'),sp.diags(omega*np.arange(npump),format='csr'),format='csr')
        self.spin_start=[]; k=self.geo.k_cart(kfrac)
        for kk in ((k,-k) if nk==2 else (k,)):
            hh=m.h8(self.geo,m.R0,kk,neel)
            for s in (0,1):self.spin_start.append(spin_initial(hh[np.ix_(SPINS[s],SPINS[s])]))
        el=np.array([1.+0j]);
        for ss in self.spin_start:el=np.kron(el,ss)
        self.initial_e=el
        # Check exact zero-field electronic Hamiltonian
        h0=offs.copy()
        for _,o in self.terms:h0=h0+o+o.getH()
        self.zero_field_hermiticity=float(sp.linalg.norm(h0-h0.getH()))
        self.zero_field_ref=float(q.audit_zero_field(nk=nk)[0]) if tuple(kfrac)==(.4158,.1683) and neel==1 else None

    def assemble(self,gate):
        # Sources indexed by transverse polarizations. Gate applies both pump coupling
        # and weak output coupling; source vertices use exact derivative of full projected Peierls.
        n=self.np;H=self.base.copy()
        Jp=sp.csr_matrix(H.shape,dtype=complex); Jt=sp.csr_matrix(H.shape,dtype=complex)
        for delta,O in self.terms:
            d=np.asarray(delta[:2],float)
            dis=q.exact_disp(n,gate*self.azp*float(self.ep@d))
            A=sp.kron(O,sp.csr_matrix(dis),format='csr')
            Ad=A.getH();H=H+A+Ad
            a=gate*float(self.ep@d);b=gate*float(self.et@d)
            Jp=Jp+(1j*a)*(A-Ad)
            Jt=Jt+(1j*b)*(A-Ad)
        return H,Jp,Jt

def one_run(npump=13,steps=64,mean=.7,phases=(0.,np.pi/2,np.pi,3*np.pi/2),neels=(1,-1),nk=1,kfrac=(.4158,.1683),ncycles=2):
    t0=time.time();report={'config':dict(npump=npump,nsteps=steps,mean_pump=mean,phases=list(phases),neels=list(neels),nk=nk,kfrac=list(kfrac),ncycles=ncycles,
         omega_p=.18,pump_az=.05,source_gate='sin^2(pi*t/T)',radiation_form='leading weak output epsilon, beta=-i epsilon integral e^(i omega_q t) <V(t)>',
         source_model='production M0 + exact Fock-projected Peierls, fixed k, 4-electron spin-conserving Slater',not_real_far_field=True), 'runs':{}}
    for N in neels:
        src=M0Source(npump,N,kfrac,nk=nk)
        st=[];inputs=[]
        for phi in phases:
            pu,nbar,tail=bsv_state(npump,mean,phi)
            st.append(np.kron(src.initial_e,pu));inputs.append({'phase':float(phi),'pump_mean_actual':nbar,'pump_tail':tail})
        psi=np.stack(st,axis=1)
        T=ncycles*2*np.pi/src.omega;dt=T/steps
        integrals=np.zeros((len(phases),2,2),complex) # [phase,pol,order 2/4]
        Jtseries=np.empty((steps,len(phases),2),complex)
        norm_err=0.;pump_edges=[]
        for it in range(steps):
            tm=(it+.5)*dt;gate=float(np.sin(np.pi*tm/T)**2)
            H,Jp,Jt=src.assemble(gate)
            psi=expm_multiply((-0.5j*dt)*H,psi)
            E=np.stack([np.sum(psi.conj()*(J@psi),axis=0) for J in (Jp,Jt)],axis=-1)
            Jtseries[it,:,:]=E
            for qi,order in enumerate((2,4)):
                integrals[:,:,qi]+=-1j*dt*np.exp(1j*order*src.omega*tm)*E
            psi=expm_multiply((-0.5j*dt)*H,psi)
            norm_err=max(norm_err,float(np.max(np.abs(np.sum(abs(psi)**2,axis=0)-1))))
        for idx,phi in enumerate(phases):
            betas=integrals[idx];
            loops=[]
            for pol in range(2):
                b2,b4=betas[pol,:]
                loops.append({'L':[float((b4/(b2*b2)).real),float((b4/(b2*b2)).imag)] if abs(b2)>1e-9 else None,
                    'phase_loop':float(np.angle(b4/(b2*b2))) if abs(b2)>1e-9 else None,
                    'beta2':[float(b2.real),float(b2.imag)],'beta4':[float(b4.real),float(b4.imag)],'abs2':float(abs(b2)),'abs4':float(abs(b4))})
            report['runs'][f'N{N}_phi{idx}']=dict(neel=N,phase=float(phi),input=inputs[idx],parallel=loops[0],perpendicular=loops[1],
                norm_error=norm_err,pump_top_final=float(np.sum(abs(psi.reshape(src.nE,npump,len(phases))[:,-1,idx])**2)))
        print('DONE neel',N,'npump',npump,'steps',steps,'time',round(time.time()-t0,1),'normerr',norm_err,'parallel:',[(round(x['parallel']['abs2'],5),round(x['parallel']['abs4'],5),round(x['parallel']['phase_loop'],5)) for x in list(report['runs'].values())[-len(phases):]],flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--npump',type=int,default=13);p.add_argument('--steps',type=int,default=64);p.add_argument('--nk',type=int,default=1);p.add_argument('--out',default='q13_result.json');p.add_argument('--phases',default='0,1.5707963267948966,3.141592653589793,4.71238898038469');p.add_argument('--neels',default='1,-1');args=p.parse_args()
    res=one_run(npump=args.npump,steps=args.steps,phases=[float(v) for v in args.phases.split(',')],neels=[int(v) for v in args.neels.split(',')],nk=args.nk)
    Path(args.out).write_text(json.dumps(res,indent=2,ensure_ascii=False))

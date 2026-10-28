#!/usr/bin/env python3
"""Q-MATH-16.22 preliminary: correct P_N exp(i η X) P_N finite-Fock Peierls.
Compare against Q16.19's exp(i η P_N X P_N), *same finite-time closed-mode model*.
No Maxwell/LSZ output. Pump self-consistent, H output first order gH.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,math,time
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.special import eval_genlaguerre,gammaln
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'q19'),str(HERE/'q19'/'reference_q167'),str(HERE/'q19'/'reference_q167'/'deps')]
import qmath_16_19_quantum_pump as old
q=old.q

def prepare_laguerre(nmax):
    d=nmax+1
    i,j=np.meshgrid(np.arange(d),np.arange(d),indexing='ij')
    ell=np.minimum(i,j);diff=np.abs(i-j)
    coeff=np.exp((gammaln(ell+1)-gammaln(ell+diff+1))/2)
    return (ell,diff,coeff)

def projected_D(eta,precomp):
    ell,diff,coeff=precomp
    return np.exp(-eta*eta/2)*coeff*(1j*eta)**diff*eval_genlaguerre(ell,diff,eta*eta)

def corrected_simulate(kfrac=(.4158,.1683),neel=1,lam=.1,delta=.8,E=.1,gp=.08,
             omega=.18,Omega=.36,gH=1e-3,T=10.,nmax=8,rtol=3e-5, classical_phase=0.):
    k,bonds,onsite,_,_,_,g,params=old.prepared(kfrac,neel,lam,delta,nmax)
    d=nmax+1
    pre=prepare_laguerre(nmax)
    es=np.array([b[0] for b in bonds]);eps=np.array([b[1] for b in bonds]);hs=np.array([b[2] for b in bonds]);Bs=np.array([b[3] for b in bonds])
    Bsdag=Bs.conj().transpose(0,2,1)
    number=np.arange(d)[:,None]
    def vmat(t):
        env=np.sin(np.pi*t/T)**2
        Ds=np.array([np.exp(1j*phase)*projected_D(gp*env*ep,pre) for phase,ep in zip(es,eps)])
        return env,Ds
    def action(psi,Ds,env,kind='h'):
        if kind=='h':
            out=np.einsum('ef,pf->pe',onsite,psi,optimize=True)
            out+=omega*number*psi
        else:
            out=np.zeros_like(psi)
        for j,D in enumerate(Ds):
            forward=np.einsum('pq,ef,qf->pe',D,Bs[j],psi,optimize=True)
            backward=np.einsum('pq,ef,qf->pe',D.conj().T,Bsdag[j],psi,optimize=True)
            if kind=='h':out+=forward+backward
            else:out+=(-1j*hs[j])*(forward-backward)
        return out
    def rhs(t,y):
        a=y[:d*36].reshape(d,36);b=y[d*36:].reshape(d,36)
        env,Ds=vmat(t)
        va=action(a,Ds,env,'h')
        vb=action(b,Ds,env,'h')+Omega*b+gH*env*action(a,Ds,env,'J')
        return (-1j*np.concatenate((va.ravel(),vb.ravel())))
    observed={}
    for kind in ('bsv','cohplus','cohminus','vac'):
        o,cut=old.optical_state(E,nmax,kind,phase=classical_phase)
        y=np.zeros((2,d,36),complex);y[0]=np.outer(o,g)
        st=time.time();sol=solve_ivp(rhs,[0,T],y.ravel(),rtol=rtol,atol=rtol*1e-2,method='DOP853')
        af,bf=sol.y[:,-1].reshape(2,d,36)
        emit=float(np.vdot(bf,bf).real)
        beta=complex(np.vdot(af,bf))
        observed[kind]={'count':emit,'beta':[beta.real,beta.imag],
          'norm_total':float(np.vdot(af,af).real+np.vdot(bf,bf).real),
          'pump_top_prob':float(np.vdot(af[-1],af[-1]).real+np.vdot(bf[-1],bf[-1]).real),
          'initial_tail_removed':cut,'seconds':time.time()-st,'nfev':sol.nfev}
    R=3*E+1;p=E/R
    mix=(1-p)*observed['vac']['count']+(p/2)*(observed['cohplus']['count']+observed['cohminus']['count'])
    observed['classical_P']={'count':mix,'weights':[1-p,p/2,p/2]}
    return {'settings':dict(kfrac=kfrac,neel=neel,lam=lam,delta=delta,E=E,gp=gp,T=T,nmax=nmax,rtol=rtol,Omega=Omega,omega=omega),
      'data':observed,'comparison':{'relative_BSV_minus_classical_count':(observed['bsv']['count']-mix)/mix}}

if __name__=='__main__':
  import argparse
  ap=argparse.ArgumentParser();ap.add_argument('--nmax',type=int,default=8);ap.add_argument('--T',type=float,default=8.)
  ap.add_argument('--rtol',type=float,default=3e-5);ap.add_argument('--out',default=str(HERE/'qmath_16_22_peierls_pair_results.json'))
  a=ap.parse_args()
  st=time.time()
  z=corrected_simulate(nmax=a.nmax,T=a.T,rtol=a.rtol)
  print('correct projector counts',z['data']['bsv']['count'],z['data']['classical_P']['count'],'seconds',time.time()-st,flush=True)
  st=time.time()
  o=old.simulate(nmax=a.nmax,T=a.T,rtol=a.rtol)
  print('original cutoff exponent counts',o['data']['bsv']['count'],o['data']['classical_P']['count'],'seconds',time.time()-st,flush=True)
  data={'scope':'finite-time single-k closed-mode, correct projected Peierls vs old truncated-X exponent; comparison NOT continuum HHG',
        'correct_projection':z,'exp_of_truncated_X':o}
  for kind in ('bsv','cohplus','cohminus','vac','classical_P'):
    aa=z['data'][kind]['count'];bb=o['data'][kind]['count']
    data.setdefault('differences',{})[kind]={'correct':aa,'old':bb,
         'relative_difference':float((aa-bb)/max(aa,bb,1e-30))}
  Path(a.out).write_text(json.dumps(data,ensure_ascii=False,indent=2))
  print('differences',data['differences'],flush=True)

#!/usr/bin/env python3
"""Q-MATH-16.19 independent finite-time M0+single quantized pump + weak output mode.
Finite-time switched pulse, periodic-k M0, fixed 36-electron spin sector. Not Maxwell LSZ HHG.
All photon quadrature Peierls phases retained, no mean-field approximation or positive-Wigner averaging.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaln
from pathlib import Path
import json,sys,math,time
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'reference_q167'
if not SRC.exists():
    SRC=Path('/mnt/data/QMATH_16_18/work/src_0')
sys.path[:0]=[str(SRC),str(SRC/'deps')]
import q167_m0_three_photon_T as q
m=q.m

def optical_state(E,nmax,kind,phase=0.):
    d=nmax+1
    out=np.zeros(d,complex)
    if kind=='bsv':
        r=np.arcsinh(np.sqrt(E));t=np.tanh(r)
        for nn in range(0,d,2):
            k=nn//2
            out[nn]=np.exp((gammaln(nn+1)-2*gammaln(k+1))/2-k*np.log(2))*(t**k)/np.sqrt(np.cosh(r))
    elif kind=='cohplus' or kind=='cohminus':
        x=math.sqrt(3*E+1)*np.exp(1j*phase)*(1 if kind=='cohplus' else -1)
        for nn in range(d):
            out[nn]=np.exp(-abs(x)**2/2+nn*np.log(abs(x))-gammaln(nn+1)/2)*(x/abs(x))**nn
    elif kind=='vac':out[0]=1
    else:raise ValueError(kind)
    missing=1-float(np.vdot(out,out).real)
    out/=np.linalg.norm(out)
    return out,missing


def prepared(kfrac=(.4158,.1683), neel=1, lam=.1, delta=.8,nmax=12):
    prm=m.M0Params(lam_so=lam,delta=delta)
    k=q.geo.k_cart(kfrac)
    X=np.diag(np.sqrt(np.arange(1,nmax+1)),1);X=X+X.T
    xval,Q=np.linalg.eigh(X)
    Nq=Q.T@np.diag(np.arange(nmax+1))@Q
    # Peierls bond electronic matrices for requested parameters, following q167 all-bond Fourier convention
    bonds=[]
    for dv,sp,kind in q.BONDS:
        if kind=='NN':sc=prm.t_h
        elif kind=='SOC':sc=prm.lam_so/m.R0.lam_so
        else:sc=(prm.u if kind=='interlayer' and abs(sp[0,4]+sp[2,6])>0 else 1) # unused default equal R0 parameters
        if kind!='SOC':sc=1
        B=q.dgamma(sp*sc)
        # for Peierls, matrix hermitian conjugate of dgamma(b) equals dgamma(b†)
        bonds.append((np.dot(dv,k),float(np.dot(dv,q.p)),float(np.dot(dv,q.h)),B))
    # exchange spin diag
    onsite=np.diag([prm.delta*neel,-prm.delta*neel,prm.delta*neel,-prm.delta*neel,
                  -prm.delta*neel,prm.delta*neel,-prm.delta*neel,prm.delta*neel])
    onsite=q.dgamma(onsite)
    baseh=m.h8(q.geo,prm,k,neel)
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    blocks=[]
    for ids in (q.UP,q.DN):
        e,U=np.linalg.eigh(baseh[np.ix_(ids,ids)])
        A=U[:,:2];v=np.array([np.linalg.det(A[[i,j],:]) for (i,j) in pairs])
        blocks.append(v)
    g=np.kron(blocks[0],blocks[1]);g/=np.linalg.norm(g)
    return k,bonds,onsite,xval,Q,Nq,g,prm


def simulate(kfrac=(.4158,.1683),neel=1,lam=.1,delta=.8,E=.1,gp=.08,
             omega=.18,Omega=.36,gH=1e-3,T=14.,nmax=12,rtol=2e-7,classical_phase=0.):
    k,bonds,onsite,xval,Q,Nq,g,prm=prepared(kfrac,neel,lam,delta,nmax)
    d=nmax+1
    BE=np.array([b[3] for b in bonds]);BC=np.conj(np.transpose(BE,(0,2,1)))
    phases=np.array([b[0] for b in bonds]);DP=np.array([b[1] for b in bonds]);DH=np.array([b[2] for b in bonds])
    def compute(t):
        # switching makes external coupling vanish at the endpoints; internal photon evolves freely
        env=np.sin(np.pi*t/T)**2
        arg=phases[:,None]+DP[:,None]*(gp*env*xval[None,:])
        z=np.exp(1j*arg)
        # combine Fourier pieces in quadrature eigenbasis; onsite included
        H=onsite[None,:,:]+np.einsum('bp,bij->pij',z,BE,optimize=True)+np.einsum('bp,bij->pij',z.conj(),BC,optimize=True)
        # J=-dGamma(dh8/dA_output); overall sign conventional
        zz=1j*DH[:,None]*z
        J=-(np.einsum('bp,bij->pij',zz,BE,optimize=True)+np.einsum('bp,bij->pij',zz.conj(),BC,optimize=True))
        return H,J,env
    def rhs(t,y):
        a=y[:36*d].reshape(d,36);b=y[36*d:].reshape(d,36)
        H,J,env=compute(t)
        va=np.einsum('pij,pj->pi',H,a,optimize=True)+omega*(Nq@a)
        vb=np.einsum('pij,pj->pi',H,b,optimize=True)+omega*(Nq@b)+Omega*b+gH*env*np.einsum('pij,pj->pi',J,a,optimize=True)
        return -1j*np.concatenate((va.ravel(),vb.ravel()))
    init={};norms={}
    for kind in ['bsv','cohplus','cohminus','vac']:
        o,missing=optical_state(E,nmax,kind,phase=classical_phase)
        y0=np.zeros((2,d,36),complex)
        y0[0]=np.outer(Q.T@o,g)
        sol=solve_ivp(rhs,[0,T],y0.ravel(),method='DOP853',rtol=rtol,atol=rtol*1e-2)
        yf=sol.y[:,-1].reshape(2,d,36)
        a,b=yf
        emit=float(np.vdot(b,b).real)
        beta=np.vdot(a,b)
        init[kind]={'count':emit,'beta':[float(beta.real),float(beta.imag)],'coherence':float(abs(beta)**2/max(emit,1e-300)),
                    'norm_total':float(np.vdot(a,a).real+emit),'n_rhs':sol.nfev,'photon_cut_probability':max(0,missing)}
    R=3*E+1;p=E/R
    mix_count=(1-p)*init['vac']['count']+(p/2)*(init['cohplus']['count']+init['cohminus']['count'])
    mb=(1-p)*complex(*init['vac']['beta'])+(p/2)*(complex(*init['cohplus']['beta'])+complex(*init['cohminus']['beta']))
    bv=complex(*init['bsv']['beta'])
    init['classical_P']={'count':mix_count,'beta':[mb.real,mb.imag], 'coherence':abs(mb)**2/max(mix_count,1e-300),
       'weights':[1-p,p/2,p/2],'E':E,'F2':3*E*E+E, 'input_mean_a':0}
    init['comparison']={'quantum_minus_classical_count':init['bsv']['count']-mix_count,
        'count_relative_diff':(init['bsv']['count']-mix_count)/max(mix_count,1e-300),
        'beta_ratio':float(abs(bv)/max(abs(mb),1e-300)),
        'bsv_coherent_fraction':init['bsv']['coherence'],'classical_coherent_fraction':init['classical_P']['coherence']}
    return {'parameters':dict(kfrac=kfrac,neel=neel,lam=lam,delta=delta,E=E,gp=gp,gH=gH,
        omega=omega,Omega=Omega,T=T,nmax=nmax,rtol=rtol,classical_phase=classical_phase), 'data':init,
        'scope':'finite-time switched single-k weak H-mode: full Peierls and quantum pump backaction; no Maxwell continuum/LSZ'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'qmath_16_19_quantized_results.json'))
    p.add_argument('--T',type=float,default=14);p.add_argument('--gp',type=float,default=.08)
    p.add_argument('--nmax',type=int,default=12);p.add_argument('--rtol',type=float,default=2e-7)
    p.add_argument('--cases',type=int,default=1)
    args=p.parse_args()
    cases=[dict(kfrac=(.4158,.1683),neel=1,lam=.1,delta=.8),dict(kfrac=(.4158,.1683),neel=1,lam=0,delta=.8),
           dict(kfrac=(.4158,.1683),neel=-1,lam=.1,delta=.8),dict(kfrac=(.4158,.1683),neel=1,lam=.1,delta=0)]
    outs=[]
    for c in cases[:args.cases]:
        st=time.time();z=simulate(T=args.T,gp=args.gp,nmax=args.nmax,rtol=args.rtol,**c);outs.append(z)
        print('case',c,'elapsed',round(time.time()-st,2),'q cl',z['data']['bsv']['count'],z['data']['classical_P']['count'],
              'diff',z['data']['comparison']['count_relative_diff'],'cutoff',z['data']['bsv']['photon_cut_probability'],flush=True)
    Path(args.out).write_text(json.dumps({'cases':outs},ensure_ascii=False,indent=2))

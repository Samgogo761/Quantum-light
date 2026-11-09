#!/usr/bin/env python3
"""Q-MATH-16.25: retarded current kernel and exact frequency-dependent downfolding.
Original M0 Peierls bonds; linear response only. No HHG/BSV emission claims.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'base'; SRC=BASE/'source'/'deps'/'q19'/'reference_q167'
sys.path[:0]=[str(BASE),str(SRC),str(SRC/'deps')]
import q167_m0_three_photon_T as q

def terms(frac, Delta=.8, lam=.1):
    k=q.geo.k_cart(frac)
    h=np.diag(np.array([Delta,-Delta,Delta,-Delta,-Delta,Delta,-Delta,Delta],float)).astype(complex)
    V=np.zeros((8,8),complex); W=np.zeros((8,8),complex)
    for dv,sp,kind in q.BONDS:
        amp=sp*(lam/.1 if kind=='SOC' else 1.0)
        z=np.exp(1j*np.dot(dv,k))*amp; d=float(np.dot(dv,q.p))
        h += z+z.conj().T
        V += 1j*d*(z-z.conj().T)
        W += -d*d*(z+z.conj().T)
    D=0.; tr=[]; mingap=1e9
    for inds in [q.UP,q.DN]:
        hb=h[np.ix_(inds,inds)]; vb=V[np.ix_(inds,inds)]; wb=W[np.ix_(inds,inds)]
        e,U=np.linalg.eigh(hb); v=U.conj().T@vb@U; wmat=U.conj().T@wb@U
        D += float(np.sum(np.diag(wmat)[:2]).real)
        mingap=min(mingap,float(e[2]-e[1]))
        for i in range(2):
            for a in range(2,4):
                gap=float(e[a]-e[i]); strength=float(abs(v[a,i])**2)
                tr.append((gap,strength))
    tr.sort(key=lambda x:x[0])
    return D,np.array([t[0] for t in tr]),np.array([t[1] for t in tr]),mingap

def kernel(D,gaps,strengths,omega,eta=5e-4,keep=None,mode="full"):
    z=complex(omega,eta)
    terms=2*gaps*strengths/(z*z-gaps*gaps)
    if keep is None: keep=len(gaps)
    out=complex(D)+np.sum(terms[:keep])
    if keep<len(gaps):
        if mode=="none":
            pass
        elif mode=="static":
            if np.any(gaps[keep:] < 1e-10):
                return complex(float('nan'),float('nan'))
            out += np.sum(-2*strengths[keep:]/gaps[keep:])
        elif mode=="dynamic":
            out += np.sum(terms[keep:])
        else:
            raise ValueError("mode must be none/static/dynamic/full")
    return complex(out)

def grid(L,D,lam,omegas,eta=5e-4,keep=2):
    arr={str(w):{'full':[],'naive':[],'static':[],'dynamic':[]} for w in omegas}; gaps_all=[]
    for i in range(L):
      for j in range(L):
        d,g,s,gmin=terms(((i+.5)/L,(j+.5)/L),Delta=D,lam=lam);gaps_all.append(gmin)
        for om in omegas:
          rec=arr[str(om)]
          rec['full'].append(kernel(d,g,s,om,eta=eta,mode='full'))
          rec['naive'].append(kernel(d,g,s,om,eta=eta,keep=keep,mode='none'))
          rec['static'].append(kernel(d,g,s,om,eta=eta,keep=keep,mode='static'))
          rec['dynamic'].append(kernel(d,g,s,om,eta=eta,keep=keep,mode='dynamic'))
    out={}
    for om,rec in arr.items():
      out[om]={}
      for typ,vals in rec.items():
        a=np.mean(vals)
        out[om][typ]={'real':float(a.real),'imag':float(a.imag)}
    return {'L':L,'Delta':D,'lambda_SO':lam,'min_grid_direct_gap':float(min(gaps_all)),'eta':eta,'keep_transitions':keep,'response':out}

def main():
    cases=[
      ('AFM_full_SOC',.8,.1,[0.,.005,.01,.02]),
      ('AFM_half_SOC',.8,.05,[0.,.005,.01,.02]),
      ('no_exchange_SOC',0.,.1,[0.,.005,.01,.02]),
      ('AFM_no_SOC_gapless',.8,0.,[.005,.01,.02,.04]),
    ]
    out={'title':'Q-MATH-16.25 retarded Ward kernel and downfolding','scope':'Original M0 linear-response kernel only. Not nonlinear HHG, not quantized BSV output.','formula':'K^R(omega)=BZavg[D_k + sum_j 2 gap_j |V_j|^2/((omega+i eta)^2-gap_j^2)]. A static counterterm replaces omitted transitions by their omega=0 value; dynamic counterterm retains frequency denominators.', 'cases':[]}
    for name,D,lam,omegas in cases:
      rows=[]
      for L in [81,121]:
        t=time.time();r=grid(L,D,lam,omegas,eta=5e-4,keep=2);r['seconds']=time.time()-t;rows.append(r)
        print('\n',name,'L',L,'gap',r['min_grid_direct_gap'],'time',r['seconds'])
        for om,v in r['response'].items():
          print(' om',om,'full',v['full'],'naive',v['naive'],'static',v['static'],'dynamic',v['dynamic'])
      out['cases'].append({'name':name,'Delta':D,'lambda_SO':lam,'omegas':omegas,'grid_results':rows})
    p=ROOT/'qmath_16_25_retarded_downfolding_results.json';p.write_text(json.dumps(out,indent=2));print('\nsaved',p)
if __name__=='__main__':main()

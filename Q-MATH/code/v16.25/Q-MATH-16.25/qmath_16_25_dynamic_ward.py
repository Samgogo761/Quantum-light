#!/usr/bin/env python3
"""Q-MATH-16.25: dynamic Ward kernel and frequency-dependent downfolding audit.
Uses original Q-MATH-16.24 M0 Peierls bonds. Linear response only; no BSV/HHG claims.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'base'
SRC=BASE/'source'/'deps'/'q19'/'reference_q167'
sys.path[:0]=[str(BASE),str(SRC),str(SRC/'deps')]
import q167_m0_three_photon_T as q

def spectral_terms(frac, Delta=.8, lam=.1):
    k=q.geo.k_cart(frac)
    h0=np.diag(np.array([Delta,-Delta,Delta,-Delta,-Delta,Delta,-Delta,Delta],float)).astype(complex)
    hp=np.zeros((8,8),complex); hpp=np.zeros((8,8),complex)
    for dv,sp,kind in q.BONDS:
        amp=sp*(lam/.1 if kind=='SOC' else 1.0)
        z=np.exp(1j*np.dot(dv,k))*amp
        d=float(np.dot(dv,q.p))
        h0 += z+z.conj().T
        hp += 1j*d*(z-z.conj().T)
        hpp += -d*d*(z+z.conj().T)
    D=0.; terms=[]; min_gap=1e9
    for inds in [q.UP,q.DN]:
        h=h0[np.ix_(inds,inds)]; V=hp[np.ix_(inds,inds)]; W=hpp[np.ix_(inds,inds)]
        e,U=np.linalg.eigh(h); v=U.conj().T@V@U; wmat=U.conj().T@W@U
        D += float(np.sum(np.diag(wmat)[:2]).real)
        min_gap=min(min_gap,float(e[2]-e[1]))
        for i in range(2):
            for a in range(2,4):
                gap=float(e[a]-e[i]); weight=2*float(abs(v[a,i])**2)/gap
                terms.append((gap,weight))
    terms.sort(key=lambda z:z[0])
    return D, np.array([x[0] for x in terms]), np.array([x[1] for x in terms]), min_gap

def response(D,gaps,weights,omega,keep=None, static_omitted=False):
    fac=1.0/(1.0-(omega/gaps)**2)
    if keep is None: keep=len(weights)
    # first keep channels ordered by transition gap
    val=D-float(np.sum(weights[:keep]*fac[:keep]))
    if keep < len(weights):
        if static_omitted:
            val-=float(np.sum(weights[keep:]))
        else:
            val-=float(np.sum(weights[keep:]*fac[keep:]))
    return val

def grid(L, omegas, keep_list=(2,4,6)):
    full=[]; naive={str(n):[] for n in keep_list}; statcorr={str(n):[] for n in keep_list}; dyncorr={str(n):[] for n in keep_list}; min_gap=1e9
    for i in range(L):
        for j in range(L):
            frac=((i+.5)/L,(j+.5)/L)
            D,g,w,gmin=spectral_terms(frac); min_gap=min(min_gap,gmin)
            for om in omegas:
                full.append((om,response(D,g,w,om)))
                for n in keep_list:
                    naive[str(n)].append((om,response(D,g,w,om,keep=n,static_omitted=False) if False else D-float(np.sum(w[:n]/(1-(om/g[:n])**2)))))
                    statcorr[str(n)].append((om,response(D,g,w,om,keep=n,static_omitted=True)))
                    dyncorr[str(n)].append((om,response(D,g,w,om,keep=n,static_omitted=False)))
    # avg each omega; `naive` means simply dropping omitted transitions, no counterterm
    def avgdict(arr):
        out={}
        for om in omegas:
            vals=[v for x,v in arr if abs(x-om)<1e-14]
            out[str(om)]=float(np.mean(vals))
        return out
    F=avgdict(full); N={k:avgdict(v) for k,v in naive.items()}; S={k:avgdict(v) for k,v in statcorr.items()}; Y={k:avgdict(v) for k,v in dyncorr.items()}
    # residuals against full response; report absolute and relative to max(|full|, 1e-12)
    errs={}
    for typ,arr in [('naive',N),('static_counterterm',S),('dynamic_counterterm',Y)]:
        errs[typ]={}
        for n,d in arr.items():
            errs[typ][n]={str(om):{'abs_error':abs(d[str(om)]-F[str(om)]),'relative_error_to_full':abs(d[str(om)]-F[str(om)])/max(abs(F[str(om)]),1e-12)} for om in omegas}
    return {'L':L,'min_direct_gap':min_gap,'omegas':list(omegas),'K_full_BZ':F,'K_truncated_naive':N,'K_static_counterterm':S,'K_dynamic_counterterm':Y,'errors':errs,
            'definitions':'K(omega)=D-sum_j w_j/[1-(omega/gap_j)^2], w_j=2|V_ji|^2/gap_j. Frequencies chosen below minimum direct gap. Static counterterm restores omitted channels only at omega=0; dynamic counterterm uses their frequency-dependent denominators.'}

def main():
    omegas=[0.0,0.01,0.02,0.04,0.06]
    out={'title':'Q-MATH-16.25 dynamic Ward and downfolding audit','scope':'Linear response of original eight-band M0 only. Not nonlinear HHG or BSV emission.','grids':[grid(L,omegas) for L in [15,25,41,61]],
         'single_k_examples':[]}
    for frac in [(0.4158,.1683),(.37,.1913),(.21,.31)]:
        D,g,w,gmin=spectral_terms(frac)
        ex=[]
        for om in [0.,.02,.04,.06]:
            ex.append({'omega':om,'K_full':response(D,g,w,om),'K_keep2_naive':D-float(np.sum(w[:2]/(1-(om/g[:2])**2))),
                       'K_keep2_static_corrected':response(D,g,w,om,keep=2,static_omitted=True),
                       'K_keep2_dynamic_corrected':response(D,g,w,om,keep=2,static_omitted=False)})
        out['single_k_examples'].append({'frac':list(frac),'min_gap':gmin,'D':D,'gaps':g.tolist(),'weights':w.tolist(),'responses':ex})
    p=ROOT/'qmath_16_25_dynamic_ward_results.json';p.write_text(json.dumps(out,indent=2))
    for gr in out['grids']:
        print('\nGRID',gr['L'],'min gap',gr['min_direct_gap'])
        for om in omegas:
            print('w=',om,'full=',gr['K_full_BZ'][str(om)],'keep2 naive=',gr['K_truncated_naive']['2'][str(om)],'keep2 static=',gr['K_static_counterterm']['2'][str(om)],'keep2 dyn=',gr['K_dynamic_counterterm']['2'][str(om)])
    print('\nSaved',p)
if __name__=='__main__': main()

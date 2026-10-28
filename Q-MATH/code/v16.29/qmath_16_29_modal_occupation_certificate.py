#!/usr/bin/env python3
"""Full two-mode GS-positive-P certificate matching each modal population n_e,n_o,
zero modal first moments and both same-state output proxy reads. Numerical convex
minimization of spectral norm along the complex null line. No physical HHG claim.
"""
import json, numpy as np
from pathlib import Path
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'qmath_16_29_results.json').read_text())
N=np.diag([.1,.09566769993281876])
D=np.diag(1/np.sqrt(np.diag(N)))
m=np.array([complex(*z) for z in r['input_m']])
def vecmat(v):return np.array([[v[0],v[2]],[v[2],v[1]]],complex)
def cert(K):
    y=K@m
    z0=np.linalg.pinv(K)@y
    _,_,Vh=np.linalg.svd(K,full_matrices=True)
    null=Vh.conj().T[:,-1]
    scale=np.linalg.norm(z0)
    def vv(x):return z0+scale*(x[0]+1j*x[1])*null
    def fun(x):return float(np.linalg.svd(D@vecmat(vv(x))@D,compute_uv=False)[0])
    solutions=[minimize(fun,g,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-14,'fatol':1e-14}) for g in ([0,0],[1,1],[-1,-1])]
    opt=min(solutions,key=lambda rr:rr.fun)
    v=vv(opt.x);M=vecmat(v)
    B=np.block([[(N.real+M.real)/2,M.imag/2],[(M.imag/2).T,(N.real-M.real)/2]])
    assert np.allclose(B,B.T,atol=1e-12)
    eig,ev=np.linalg.eigh(B)
    rec={'spectral_norm_relative_to_modal_bounds':float(opt.fun),
         'is_classical_moment_feasible':bool(opt.fun<=1+1e-9),
         'covariance_min_eigenvalue':float(eig[0]),
         'moments':[ [float(a.real),float(a.imag)] for a in v],
         'readout_error':float(np.linalg.norm(K@v-y)/np.linalg.norm(y))}
    if eig.min()>=-1e-10:
        eig=np.maximum(0,eig)
        s=float(eig.sum())
        dist=[]
        for i,lam in enumerate(eig):
            if lam<1e-13:continue
            for sg in (-1,+1):
                realvec=sg*np.sqrt(s)*ev[:,i]
                alpha=realvec[:2]+1j*realvec[2:]
                dist.append((float(lam/(2*s)),alpha))
        p=np.array([a for a,b in dist]);a=np.array([b for _,b in dist]);
        Nm=np.einsum('a,ai,aj->ij',p,a,a.conj())
        Mm=np.einsum('a,ai,aj->ij',p,a,a)
        rec['ensemble_size']=len(dist)
        rec['modal_number_error']=float(np.linalg.norm(Nm-N))
        rec['moments_error']=float(np.linalg.norm(Mm-M))
        rec['zero_mean_error']=float(np.linalg.norm(p@a))
        rec['output_error_from_explicit_P']=float(np.linalg.norm(K@np.array([Mm[0,0],Mm[1,1],Mm[0,1]])-y)/np.linalg.norm(y))
        rec['classical_P_ensemble']=[{'p':float(pp),'alpha':[[float(z.real),float(z.imag)] for z in aa]} for pp,aa in dist]
    return rec
result={'N_target':N.tolist(), 'meaning':'Classical positive GS P inputs match two individual photon occupations, zero mean field, all output pair proxy amplitudes (Omega .34,.38). A fully explicit finite +/- coherent-state ensemble is supplied where feasible.','cases':[]}
for cc in r['cases']:
 K=np.array([[complex(*zz) for zz in row] for row in cc['analysis']['K']])[[0,2]]
 rec=cert(K)
 result['cases'].append({'L':cc['L'],'eta':cc['eta'],'Delta':cc['Delta'],**rec})
 print(cc['L'],cc['eta'],cc['Delta'],'min-norm',round(rec['spectral_norm_relative_to_modal_bounds'],6),'positive',rec['is_classical_moment_feasible'],'nstate',rec.get('ensemble_size'), 'explicit_err',rec.get('output_error_from_explicit_P'))
(ROOT/'qmath_16_29_modal_occupation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

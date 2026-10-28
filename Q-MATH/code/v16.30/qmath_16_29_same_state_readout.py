#!/usr/bin/env python3
# Q-MATH-16.29 SAME-STATE multi-output classical chi2 PROXY: not quantum emission
from __future__ import annotations
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import sys, time, json, itertools, argparse, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
PREV=ROOT if (ROOT/'q16_27_reference').exists() and (ROOT/'qmath_16_28_photon_normalized_results.json').exists() else Path('/mnt/data/QMATH_16_28')
sys.path.insert(0,str(PREV/'q16_27_reference'))
from qmath_16_27_retarded_kernel import kernel

CENTER=.18; SIG=.043; BAND=(.10,.26)
OUTPUTS=[.34,.36,.38]
FSTEP=.01

def modes(h=FSTEP):
    w=np.arange(BAND[0], BAND[1]+h/3,h)
    trap=np.ones(len(w))*h;trap[[0,-1]]*=.5
    ue=np.exp(-.5*((w-CENTER)/SIG)**2);ue/=np.sqrt(np.sum(trap*ue**2))
    uo=(w-CENTER)*ue;uo/=np.sqrt(np.sum(trap*uo**2))
    f=[ue,uo]
    W=np.array([[np.sum(trap*w*a*b) for b in f] for a in f]);
    evals,U=np.linalg.eigh(W); Whalf=(U*np.sqrt(evals))@U.T
    return w,trap,np.array(f),W,Whalf

def overlap_grid(Omega,h):
    lo=max(BAND[0],Omega-BAND[1]);hi=min(BAND[1],Omega-BAND[0]);
    n=int(round((hi-lo)/h))
    w=np.linspace(lo,hi,n+1)
    trap=np.ones(len(w))*h;trap[[0,-1]]*=0.5
    return w,Omega-w,trap

def compress(z):return [float(z.real),float(z.imag)]
def decompress(z):return complex(z[0],z[1])

def row_project(omega, chi, w1, w2, trap, modes_w, modes_shape):
    ua=np.array([np.interp(w1,modes_w,sh) for sh in modes_shape])
    ub=np.array([np.interp(w2,modes_w,sh) for sh in modes_shape])
    fac=trap/np.sqrt(w1*w2)
    p0=np.sum(fac*chi*ua[0]*ub[0]);p1=np.sum(fac*chi*ua[1]*ub[1]);
    # Input M symmetric; coefficient is K_eo+K_oe (here 2K_eo).
    p2=np.sum(fac*chi*(ua[0]*ub[1]+ua[1]*ub[0]))
    return np.array([p0,p1,p2],complex)

def build_scan(L,eta,Delta,cache,force=False,quiet=False):
    mw,_,mf, W, Whalf=modes()
    rows=[]; flats=[]; deviations=[]; rows_info=[]
    for Omega in OUTPUTS:
        w1,w2,trap=overlap_grid(Omega,FSTEP)
        chi=[]
        for t, (a,b) in enumerate(zip(w1,w2)):
            # Frequencies on ascending grid; treat (a,b) and (b,a) symmetrically.
            k=f'{L}_{eta:.5f}_{Delta:.4f}_{min(a,b):.4f}_{max(a,b):.4f}'
            if k not in cache or force:
                rec=kernel(L=L,Delta=Delta,eta=eta,w1=float(a),w2=float(b))
                cache[k]=rec['chi12']
            chi.append(decompress(cache[k]))
        chi=np.array(chi)
        mid=int(np.argmin(np.abs(w1-Omega/2)))
        center_chi=chi[mid]
        material=row_project(Omega,chi,w1,w2,trap,mw,mf)
        flat=row_project(Omega,np.ones_like(chi)*center_chi,w1,w2,trap,mw,mf)
        rows.append(material);flats.append(flat);deviations.append(material-flat)
        rows_info.append({'Omega':Omega,'chi_center':compress(center_chi),'full_row':[compress(z) for z in material],
                          'flat_control_row':[compress(z) for z in flat],
                          'excess_row':[compress(z) for z in material-flat],
                          'material_fraction':float(np.linalg.norm(material-flat)/max(1e-15,np.linalg.norm(material))),
                          'cross_fraction':float(abs(material[2])/max(1e-15,np.linalg.norm(material)))})
        if not quiet:print(' L',L,'eta',eta,'D',Delta,'Omega',Omega,'row',np.round(material,4),flush=True)
    K=np.array(rows);F=np.array(flats);C=np.array(deviations)
    return K,F,C,rows_info,W,Whalf

def matrix_from_mvec(z):return np.array([[z[0],z[2]],[z[2],z[1]]],complex)
def energy_cost(z,Whalf):
    M=matrix_from_mvec(z)
    Q=Whalf@M@Whalf
    return float(np.linalg.svd(Q,compute_uv=False).sum())

def complex_null_svd(K):
    U,s,Vh=np.linalg.svd(K,full_matrices=True)
    return Vh.conj().T[:,-1],s

def min_cost_restricted(K,y,Whalf):
    # Algebraic classical minimization under complex linear measurement equations.
    p=np.linalg.pinv(K)@y
    U,s,Vh=np.linalg.svd(K,full_matrices=True)
    rank=int(np.sum(s>max(s)*1e-8))
    zdim=K.shape[1]-rank
    if zdim==0:
        return {'cost':energy_cost(p,Whalf),'m_fit':[compress(t) for t in p],
                'relative_residual':float(np.linalg.norm(K@p-y)/max(np.linalg.norm(y),1e-30)),
                'null_dimension':zdim,'method':'unique complex inverse'}
    N=Vh.conj().T[:,rank:]
    scale=max(np.linalg.norm(p),1e-8)
    def getm(uv):return p+N@(uv[:zdim]+1j*uv[zdim:])*scale
    def fun(uv):return energy_cost(getm(uv),Whalf)
    candidates=[]
    for guess in [np.zeros(2*zdim), np.ones(2*zdim)*0.1, -np.ones(2*zdim)*0.3]:
        sol=minimize(fun,guess,method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-12,'fatol':1e-12})
        candidates.append(sol)
    best=min(candidates,key=lambda x:x.fun);m=getm(best.x)
    return {'cost':float(best.fun),'m_fit':[compress(t) for t in m],
            'relative_residual':float(np.linalg.norm(K@m-y)/max(np.linalg.norm(y),1e-30)),
            'null_dimension':zdim,'method':'Nelder-Mead (convex nuclear norm objective, numerical only)',
            'optimizer_spread':float(max(x.fun for x in candidates)-best.fun)}

def state():
    _,_,_,W,Whalf=modes()
    from math import sqrt
    # match Q16.28 (two real nonzero squeezed moments, no inter-mode anomalous moment)
    me=sqrt(.11)
    old=__import__('json').loads((PREV/'qmath_16_28_photon_normalized_results.json').read_text())['input']
    mo=old['m_odd'];n1=.1;n2=old['n_odd']
    m=np.array([me,mo,0.],complex);U=float(W[0,0]*n1+W[1,1]*n2)
    return m,U,W,Whalf

def stats(K,F,m,Whalf,U):
    y=K@m
    mat_norm=float(np.linalg.svd(K,compute_uv=False).max())
    data={'K':[list(map(compress,row)) for row in K],
          'K_flat':[list(map(compress,row)) for row in F],
          'K_singular_values':np.linalg.svd(K,compute_uv=False).tolist(),
          'K_flat_singular_values':np.linalg.svd(F,compute_uv=False).tolist(),
          'K_singular_condition':float(np.linalg.cond(K)) if K.shape[0]>=3 else None,
          'K_flat_singular_condition':float(np.linalg.cond(F)) if F.shape[0]>=3 else None,
          'y_BSV':[compress(x) for x in y],
          'material_delta_overall':float(np.linalg.norm(K-F)/np.linalg.norm(K)),
          'cost_BSV_M':energy_cost(m,Whalf), 'resource_U':U,
          'BSV_cost_to_U':energy_cost(m,Whalf)/U}
    for subset,ids in [('pair_34_36',[0,1]),('pair_34_38',[0,2]),('pair_36_38',[1,2]),('triple',[0,1,2])]:
        data[subset]=min_cost_restricted(K[ids],y[ids],Whalf)
        data[subset]['cost_over_U']=data[subset]['cost']/U
        data[subset]['physically_certified']=False
    # Singular-only classical bound |lambda.y| <= U ||W^-1/2 K(lambda) W^-1/2||.
    return data

def run(args):
    cachefile=ROOT/'qmath_16_29_kernel_cache.json'
    cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
    m,U,W,Whalf=state()
    all_cases=[]
    cases=[(61,.08,.8),(101,.08,.8),(141,.08,.8),(201,.08,.8),
           (81,.03,.8),(141,.03,.8),(201,.03,.8),
           (101,.08,.4)] if not args.quick else [(41,.08,.8),(81,.08,.8),(81,.08,.4)]
    for L,eta,D in cases:
        t=time.time()
        K,F,C,rows,W,Wh=build_scan(L,eta,D,cache)
        cachefile.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
        measurements=stats(K,F,m,Wh,U)
        all_cases.append({'L':L,'eta':eta,'Delta':D,'seconds':float(time.time()-t),'rows':rows,'analysis':measurements})
        print('--- DONE',L,eta,D,'time',round(time.time()-t,2),'cost ratios:',[(x,round(measurements[x]['cost_over_U'],4)) for x in ['pair_34_38','triple']],flush=True)
    result={'identity':'Q-MATH-16.29 same M0 state, distinct output Ω, 2 squeezed inputs + unrestricted classical intermode coherence. Retarded chi2 MATERIAL PROXY only; NOT quantum radiated field, no LSZ, no Maxwell.','Omega_values':OUTPUTS,'center_frequency':CENTER,'input_band':BAND,'input_quad_step':FSTEP,
      'input_m':[compress(t) for t in m],'input_energy_W':W.tolist(),'U':U,
      'theorem':'A generic output bin omega != 2 center reads the off-diagonal anomalous moment M_eo. Two distinct output ports cannot invert 3 complex moments of arbitrary two-mode classical input. Three ports might invert, but even flat chi(Ω) produces rank by spectral kinematics: compare against the exact flat per-bin control.',
      'classical_criterion':'Classic GS-P with mean optical energy <= U iff the symmetric anomalous M has || W^(1/2) M W^(1/2) ||_* <= U (allowing vacuum and +/- coherent mixtures); measured proxy y=K M. Minimize weighted nuclear cost for a classical fit.','cases':all_cases}
    outfile=ROOT/'qmath_16_29_results.json';outfile.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT',outfile,'cases=',len(all_cases),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true')
    args=p.parse_args();run(args)

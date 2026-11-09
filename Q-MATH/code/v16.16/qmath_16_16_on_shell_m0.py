#!/usr/bin/env python3
"""Q-MATH-16.16 on-shell finite-wavepacket, fixed M0 36-d electron model.
Third-order Peierls T, energy conservation in final state, virtual-state gamma only.
No Maxwell/LSZ, no absolute photon emission normalization.  Fixed one BSV mode.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import dataclasses, sys, itertools, json, time, math
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
SRC=BASE/'src'
sys.path.insert(0,str(SRC))
import q167_m0_three_photon_T as q

WMIN,WMAX=.12,.24

def swap_soc(lam):
    q.m.R0=dataclasses.replace(q.m.R0,lam_so=float(lam)); q.BONDS=q.bonds()

def wavepack(w,centre=.18,sigma=.030):
    # normalized in finite interval
    import math
    norm2=.5*math.sqrt(math.pi)*sigma*(math.erf((WMAX-centre)/sigma)-math.erf((WMIN-centre)/sigma))
    return np.exp(-.5*((np.asarray(w)-centre)/sigma)**2)/math.sqrt(norm2)

def setup(kf,N):
    k=q.geo.k_cart(kf); v=q.vertices(k,N)
    E,U=np.linalg.eigh(v['0']); EG=E[0]
    A={nm:U.conj().T@val@U for nm,val in v.items() if nm!='0'}
    gap=E-EG
    return E,EG,A,gap

def amplitudes(model,w1,w2,Omega,gamma):
    E,EG,A,_=model
    freqs=np.array([w1,w2,-Omega]); events=(0,1,2)
    def R(inds):return 1/(EG+sum(freqs[list(inds)])-E+1j*gamma)
    def v(ids):
        ids=set(ids)
        if ids in ({0},{1}):return A['p']
        if ids=={2}:return A['h']
        if ids=={0,1}:return A['pp']
        if ids in ({0,2},{1,2}):return A['ph']
        if ids=={0,1,2}:return A['pph']
        raise ValueError(ids)
    o=A['pph'][:,0].copy()
    for r in (1,2):
        for first in itertools.combinations(events,r):
            last=tuple(x for x in events if x not in first)
            o+=v(last)@(R(first)*v(first)[:,0])
    for a,b,c in itertools.permutations(events):
        o+=v((c,))@(R((a,b))*(v((b,))@(R((a,))*v((a,))[:,0])))
    return o

def fixed_pulse(model, OMEGAS, nq=15, gamma=.25, centre=.18,sigma=.03):
    """S_f(Omega) for one fixed u and all electronic f. Vectors in electron eigenbasis."""
    E,EG,A,gaps=model
    n=len(OMEGAS); array=np.zeros((n,36),complex)
    for io,Omega in enumerate(OMEGAS):
        for f,gap in enumerate(gaps):
            S=Omega+gap
            lo=max(WMIN,S-WMAX);hi=min(WMAX,S-WMIN)
            if hi<=lo+1e-12:continue
            # Gauss-Legendre in physical frequency integral, no virtual gamma in energy delta
            x,w=np.polynomial.legendre.leggauss(nq)
            ww=lo+(x+1)*(hi-lo)/2
            other=S-ww
            factor=wavepack(ww,centre,sigma)*wavepack(other,centre,sigma)*w*(hi-lo)/2
            vals=np.array([amplitudes(model,float(w1),float(w2),float(Omega),gamma)[f] for w1,w2 in zip(ww,other)])
            array[io,f]=np.sum(factor*vals)
    return array

def run():
    start=time.perf_counter();ori=q.m.R0;origb=q.BONDS
    out={'scope':'M0 36-electron exact Peierls third order, fixed Gaussian pump packet and on-shell electronic excitations; no absolute Maxwell normalization, gamma virtual regulator','pump_band':[WMIN,WMAX], 'pump_gaussian':{'centre':.18,'sigma':.03},'frequency_modes':'output continuum sampled by quadrature'}
    try:
        OMEGAS=np.linspace(.22,.38,9)
        kfs=[(.4158,.1683),(.3703,.1913),(.2760,.4480),(.173,.271)]
        entries=[]
        for lam in [0., .1]:
            swap_soc(lam)
            for kf in kfs:
                for N in [1]:
                    model=setup(kf,N)
                    S=fixed_pulse(model,OMEGAS,nq=11,gamma=.25)
                    # one k (not BZ) elastic channel and positive inelastic spectrum
                    I_exc=np.sum(abs(S[:,1:])**2,axis=1)
                    entries.append({'lambda':lam,'kfrac':kf,'N':N,'lowest_gap':float(model[3][1]),'Omega':OMEGAS.tolist(),
                                    'elastic_amplitude':[[float(z.real),float(z.imag)] for z in S[:,0]],
                                    'excited_kernel_norm2':I_exc.tolist(),
                                    'count_integral_excited':float(np.trapezoid(I_exc,OMEGAS))})
                    print('local',lam,kf,'gap',model[3][1],'sum excited',np.trapezoid(I_exc,OMEGAS),flush=True)
        out['single_k']=entries
        # SOCless inversion pairs: exact electronic excited norm positivity, elastic average cancellation.
        for lam in [0.,.1]:
            swap_soc(lam)
            for L in (5,9,13):
                Ssum=np.zeros(len(OMEGAS),complex)
                Iin=np.zeros(len(OMEGAS),float)
                groups=0
                for i in range(L):
                    for j in range(L):
                        kf=((i+.5)/L,(j+.5)/L)
                        model=setup(kf,1)
                        S=fixed_pulse(model,OMEGAS,nq=7,gamma=.25)
                        Ssum+=S[:,0];Iin+=np.sum(abs(S[:,1:])**2,axis=1)
                        groups+=1
                Ssum/=groups;Iin/=groups
                row={'lambda':lam,'L':L,'gamma':.25,'nw':7,'integrated_inelastic_kernel_norm2':float(np.trapezoid(Iin,OMEGAS)),
                     'integrated_mean_elastic_amplitude_norm2':float(np.trapezoid(abs(Ssum)**2,OMEGAS)),
                     'elastic_amplitude_sample':[[float(z.real),float(z.imag)] for z in Ssum],
                     'positive_inelastic_spectral_density':Iin.tolist()}
                out.setdefault('bz',[]).append(row)
                print('BZ',lam,L,'inel=',row['integrated_inelastic_kernel_norm2'],'elastic=',row['integrated_mean_elastic_amplitude_norm2'],flush=True)
    finally:
        q.m.R0=ori;q.BONDS=origb
    out['runtime_seconds']=time.perf_counter()-start
    (BASE/'qmath_16_16_on_shell_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',out['runtime_seconds'])

if __name__=='__main__':run()

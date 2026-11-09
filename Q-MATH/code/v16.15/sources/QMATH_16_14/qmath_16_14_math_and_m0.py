#!/usr/bin/env python3
"""Q-MATH-16.14: rigorous moment hierarchy + M0 magnetic gate tests.

Source: original q167 production M0 imported through Q16.13 verified five-photon implementation.
T5 is the gamma-regularized BARE T coefficient (not on-shell LSZ/Maxwell),
so the 'magnetic gate' does not constitute a confirmed new HHG effect.
"""
from __future__ import annotations
import json, math, os, sys, dataclasses, itertools
from pathlib import Path
import numpy as np
from scipy.special import factorial

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'QMATH_16_13'
if not OLD.exists():
    OLD=HERE/'Q16_13_original'
sys.path.insert(0,str(OLD))
from qmath_16_13_singlemode_five_photon import calc
import q167_m0_three_photon_T as q


def c3_c51(kf,N,delta=.8,omega=.18,gamma=.25,external=False):
    before=q.m.R0
    q.m.R0=dataclasses.replace(q.m.R0,delta=float(delta))
    try:
        aa=calc(kf,N,omega,gamma,2,project_external=external)
        bb=calc(kf,N,omega,gamma,4,project_external=external)
        c3=complex(*aa['T3_scaled'])
        c50=complex(*aa['T5_scaled'])
        c51=(complex(*bb['T5_scaled'])-c50)/2.
        return np.array([c3,c50,c51],complex)
    finally:
        q.m.R0=before

def pair_and_symmetry():
    cases=[]
    for delta in (0., .2, .4, .8):
        for kf in ((.4158,.1683),(.173,.271)):
            kp=np.asarray(kf,dtype=float)
            a=c3_c51(kp,1,delta)
            b=c3_c51(-kp,-1,delta)
            c=c3_c51(-kp,1,delta)
            relinv=float(np.linalg.norm(a+b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-30))
            even_under_k=float(np.linalg.norm(a+c)/max(np.linalg.norm(a),np.linalg.norm(c),1e-30))
            cases.append(dict(delta=delta,kf=list(kf),
                              inversion_pair_residual= relinv,
                              same_N_k_pair_residual= even_under_k,
                              c3_abs=float(abs(a[0])),c51_abs=float(abs(a[2]))))
    return cases

def bz(L,delta,gamma=.25,omega=.18):
    ss=np.zeros(3,complex)
    for i in range(L):
        for j in range(L):
            kf=((i+.5)/L,(j+.5)/L)
            ss+=c3_c51(kf,1,delta,omega,gamma)
    avg=ss/L**2
    return dict(L=L,delta=delta,gamma=gamma,omega=omega,
                c3=[avg[0].real,avg[0].imag],
                c50=[avg[1].real,avg[1].imag],
                c51=[avg[2].real,avg[2].imag],
                abs_c3=float(abs(avg[0])),abs_c51=float(abs(avg[2])),
                c51_over_c3=[float((avg[2]/avg[0]).real),float((avg[2]/avg[0]).imag)]
                if abs(avg[0])>1e-10 else None)

def bs_moment(j,E):
    m=math.sqrt(E*(E+1))
    ans=0.
    for s in range(j//2+1):
        coeff=(math.factorial(j)*math.factorial(j+2)/
               (math.factorial(j-2*s)*2**(2*s+1)*math.factorial(s)*math.factorial(s+1)))
        ans+=coeff*E**(j-2*s)*(E*(E+1))**s
    return m*ans

def bsv_direct(j,E,ncut=90):
    lam=math.sqrt(E/(E+1))
    prob0=math.sqrt(1-lam**2)
    coeff=np.zeros(ncut+1,complex)
    # c_(2k) = (1-lambda^2)^1/4 sqrt((2k)!)/(2^k k!) (-lambda)^k
    for k in range((ncut//2)+1):
        ln = 0.25*math.log(1-lam**2) + .5*math.lgamma(2*k+1) - k*math.log(2) - math.lgamma(k+1)
        coeff[2*k]=math.exp(ln)*((-lam)**k)
    operator_value=0j
    for n in range(j+2,ncut+1):
        # (adag)^j a^(j+2) |n> = sqrt(n!(n-2)!)/(n-j-2)! |n-2>
        c=math.exp((math.lgamma(n+1)+math.lgamma(n-1))/2-math.lgamma(n-j-1))
        operator_value+=coeff[n-2].conjugate()*coeff[n]*c
    return abs(operator_value)

def test_moments():
    rows=[]
    for E in (.15,1/3,.7,1.5):
        f2=3*E*E+E
        m=math.sqrt(E*(E+1))
        for j in (0,1,2,3):
            ana=bs_moment(j,E)
            direct=bsv_direct(j,E,150)
            rows.append({'E':E,'j':j,'formula':ana,'direct_fock':direct,
                         'relerr':abs(ana-direct)/max(ana,1e-30),
                         'F2':f2,'classic_F2_bound_exceeded':bool(j==1 and ana>f2+1e-12)})
    return rows

def main():
    results={'scope':'formal M0 T3/T5 bare coefficient and exact single-mode Gaussian moment bounds; no LSZ, no broadband A2',
             'analytic_moments':test_moments(),
             'inversion_delta_tests':pair_and_symmetry(),
             'bz':[]}
    (HERE/'qmath_16_14_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('moment tests max relative residual',max(x['relerr'] for x in results['analytic_moments']),flush=True)
    print('inversion tests max residual',max(x['inversion_pair_residual'] for x in results['inversion_delta_tests']),flush=True)
    for delta,L in [(0,5),(0,9),(.2,5),(.4,5),(.8,5),(.4,9),(.8,9),(.8,13),(.4,13)]:
        rec=bz(L,delta)
        results['bz'].append(rec)
        (HERE/'qmath_16_14_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
        print(f'Delta={delta:.2f}, grid {L}^2, c3={rec["abs_c3"]:.3e}, c51={rec["abs_c51"]:.3e}',flush=True)
    print('DONE')
if __name__=='__main__':main()

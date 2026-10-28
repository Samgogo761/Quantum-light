#!/usr/bin/env python3
"""T5 bare coefficient SOC odd and spin inversion selection test, using Q16.14 original code.
Never mistake this for a connected LSZ scattering computation.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
Q14=B/'sources/QMATH_16_14'
S3=Q14/'Q16_13_original/src_0'
sys.path[0:0]=[str(Q14),str(S3),str(Q14/'Q16_13_original')]
from qmath_16_14_math_and_m0 import c3_c51, q
import dataclasses

def testbz(L,lam,delta=.8,gamma=.25):
    prev,prevb=q.m.R0,q.BONDS
    try:
        q.m.R0=dataclasses.replace(prev,lam_so=float(lam))
        q.BONDS=q.bonds()   # restore bond coefficients when changing SOC
        s=np.zeros(3,complex)
        for i in range(L):
            for j in range(L):
                k=((i+.5)/L,(j+.5)/L)
                s+=c3_c51(k,1,delta=delta,omega=.18,gamma=gamma)
        return s/L**2
    finally:q.m.R0=prev;q.BONDS=prevb

def main():
    rows=[]
    for L in (5,9):
        for lam in (0,.05,.1,-.1):
            t=time.time();a=testbz(L,lam)
            rec={'L':L,'lambda':lam,'Delta':.8,'gamma':.25,'c3':[float(a[0].real),float(a[0].imag)],'c50':[float(a[1].real),float(a[1].imag)],'c51':[float(a[2].real),float(a[2].imag)],'abs_c51':float(abs(a[2])),'duration_sec':round(time.time()-t,2)}
            rows.append(rec)
            print(L,lam,'c51',rec['abs_c51'],'duration',rec['duration_sec'],flush=True)
    audit=[]
    for L in (5,9):
        a=next(x for x in rows if x['L']==L and x['lambda']==.1)
        b=next(x for x in rows if x['L']==L and x['lambda']==-.1)
        c=next(x for x in rows if x['L']==L and x['lambda']==0)
        pa=complex(*a['c51']);pb=complex(*b['c51']);p0=complex(*c['c51'])
        audit.append({'L':L,'c51_soc_odd_relative_error':float(abs(pa+pb)/max(abs(pa),1e-30)),'zero_soc_residual_relative':float(abs(p0)/max(abs(pa),1e-30))})
    obj={'description':'bare gamma-regularized M0 five-photon T, comparison with original Q16.14, no LSZ','data':rows,'audit':audit}
    (B/'qmath_16_15_five_soc_results.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print('audit',audit,flush=True)
if __name__=='__main__':main()

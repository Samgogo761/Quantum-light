import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,dataclasses,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'sources/QMATH_16_14/Q16_13_original/src_0'))
import q167_m0_three_photon_T as q

def avg(L,lam,gamma=.25):
    old=q.m.R0; oldb=q.BONDS
    try:
        q.m.R0=dataclasses.replace(old,lam_so=lam,delta=.8)
        q.BONDS=q.bonds()
        tot=0j
        for i in range(L):
            for j in range(L):
                tot+=q.calc(((i+.5)/L,(j+.5)/L),N=1,w1=.16,w2=.20,gamma=gamma)['total']
        return tot/L**2
    finally:q.m.R0=old;q.BONDS=oldb

results=[]
for L in (9,15,25,35):
  arr={}
  for lam in (.005,.01,.02,.05,.1):
    arr[lam]=avg(L,lam)
  r=dict(L=L,values={str(k):[float(v.real),float(v.imag)] for k,v in arr.items()},
         slopes={str(k):[float((v/k).real),float((v/k).imag)] for k,v in arr.items()},
         slope_0p005_over_0p010_rel=float(abs(arr[.005]/.005-arr[.01]/.01)/max(abs(arr[.005]/.005),1e-30)))
  results.append(r)
  print('L',L,'slopes .005 .01 .02',[(k,abs(arr[k]/k)) for k in (.005,.01,.02)],'rel discrepancy',r['slope_0p005_over_0p010_rel'],flush=True)
(B/'qmath_16_15_soc_derivative.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

#!/usr/bin/env python3
"""Finite-frequency linear-response parameter scan for original M0.
No nonlinear HHG or BSV output claims. Frequencies are kept below each model's minimum direct gap.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import qmath_16_25_dynamic_ward as m

def calc(L,D,lam,omegas):
    vals={str(w):[] for w in omegas}; gaps=[]
    for i in range(L):
      for j in range(L):
        d,g,w,gmin=m.spectral_terms(((i+.5)/L,(j+.5)/L),Delta=D,lam=lam)
        gaps.append(gmin)
        for om in omegas:
          vals[str(om)].append(m.response(d,g,w,om))
    return {'L':L,'Delta':D,'lambda_SO':lam,'min_direct_gap':float(min(gaps)),
            'K_BZ':{k:float(np.mean(v)) for k,v in vals.items()},
            'mean_abs_local_K0':None}

def main():
    cases=[('AFM_full',.8,.1,[0.,.01,.02,.04]),('no_SOC',.8,0.,[0.,.005,.01,.02]),('no_exchange',0.,.1,[0.,.01,.02,.04]),('no_SOC_no_exchange',0.,0.,[0.,.005,.01,.02]),('weaker_exchange',.4,.1,[0.,.01,.02,.04]),('stronger_exchange',1.2,.1,[0.,.005,.01,.02])]
    out={'scope':'Linear response of original M0 Peierls Hamiltonian; BZ-averaged current kernel K(omega). Not HHG or quantized-output prediction.', 'formula':'K(omega)=<D_k - sum_j w_kj/(1-(omega/gap_kj)^2)>_BZ, w=2|V_ai|^2/gap. Frequencies below the smallest direct gap in each case.', 'results':[]}
    for name,D,lam,om in cases:
      t=time.time(); r=calc(81,D,lam,om);r['case']=name;r['seconds']=time.time()-t;out['results'].append(r)
      print(name,'gap',r['min_direct_gap'],'K',r['K_BZ'],'secs',r['seconds'])
    p=ROOT/'qmath_16_25_parameter_scan.json';p.write_text(json.dumps(out,indent=2));print('saved',p)
if __name__=='__main__':main()

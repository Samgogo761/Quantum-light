#!/usr/bin/env python3
"""Independent 36-dimensional dGamma f-sum audit of 8-orbital R0 M0 model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,numpy as np
from pathlib import Path
R=Path(__file__).parent
sys.path.insert(0,str(R));from qmath_16_24_ward_sumrule import one
S=R/'source'/'deps'/'q19'/'reference_q167';sys.path[:0]=[str(S),str(S/'deps')]
import q167_m0_three_photon_T as q
checks=[]
for frac in [(0.4158,.1683),(.37,.1913),(.21,.31)]:
    k=q.geo.k_cart(frac);v=q.vertices(k,1)
    e,U=np.linalg.eigh(v['0']);g=U[:,0];eg=e[0]
    J=U.conj().T@v['p']@U
    D=U.conj().T@v['pp']@U
    p=2*sum(abs(J[n,0])**2/(e[n]-eg) for n in range(1,len(e)) if (e[n]-eg)>1e-10)
    j=float(np.vdot(g,v['p']@g).real)
    d=float(np.vdot(g,v['pp']@g).real)
    result=one(frac)
    checks.append({'frac':list(frac), 'dim':len(e),
       'j_manybody':j,'j_single':result['j'], 'D_manybody':d,'D_single':result['D'],
       'P_manybody':float(p),'P_single':result['paramagnetic'],
       'K_manybody':float(d-p),'K_single':result['K'],
       'K_residual':float(abs((d-p)-result['K']))})
path=R/'qmath_16_24_manybody_crosscheck.json';path.write_text(json.dumps(checks,indent=2))
for row in checks:print(row)

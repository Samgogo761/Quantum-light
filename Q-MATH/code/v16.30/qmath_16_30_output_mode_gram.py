#!/usr/bin/env python3
"""Q-MATH-16.30 finite-width continuum-normalized H2 output wavepacket Gram check.
Numerical quadrature over a shared continuum [0.28,0.44], not a physical M0 emission.
"""
from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).resolve().parent
CENTERS=np.array([.32,.34,.36,.38,.40])
w=np.arange(.28,.44000001,.00005)
weights=np.ones(len(w))*.00005;weights[[0,-1]]*=.5
records=[]
for sigma in [.002,.003,.004,.005,.007,.010]:
    v=np.array([np.exp(-((w-c)**2)/(4*sigma**2)) for c in CENTERS],float).T
    for j in range(5):
        v[:,j]/=np.sqrt(np.sum(weights*v[:,j]**2))
    G=(v.T*weights)@v
    eig,U=np.linalg.eigh(G)
    v2=v@((U*(1/np.sqrt(eig)))@U.T)
    G2=(v2.T*weights)@v2
    records.append({'sigma':sigma,'adjacent_overlap':float(G[0,1]),'max_offdiag_raw':float(np.max(np.abs(G-np.eye(5)))),
        'lowdin_orthonormality_max_abs':float(np.max(np.abs(G2-np.eye(5)))),
        'raw_condition':float(np.linalg.cond(G)),
        'G_raw':G.tolist()})
result={'definition':'b_j=int dOmega v_j*(Omega) b(Om); [b_i,b_j^dagger]=int v_i*v_j.',
         'centers':CENTERS.tolist(),'integration_domain':[.28,.44],'integration_step':.00005,'results':records}
(OUT/'qmath_16_30_output_modes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for x in records:
    print('sigma=',x['sigma'],'adjacent raw overlap=',x['adjacent_overlap'],'corrected max error=',x['lowdin_orthonormality_max_abs'])

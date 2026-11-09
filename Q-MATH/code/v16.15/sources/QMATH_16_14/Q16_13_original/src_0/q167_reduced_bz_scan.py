#!/usr/bin/env python3
"""Fast BZ gamma/grid convergence of elastic T using exact 9-dim Slater-Condon projection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from pathlib import Path
import q167_slater_condon_direct as r
import q167_m0_three_photon_T as m
ROOT=Path(__file__).resolve().parent

def scan(grids=(35,65,101),gammas=(.05,.1,.25)):
    rows=[]
    for gamma in gammas:
        for g in grids:
            t=time.time();result={x:0j for x in ('direct','mixed','sequential','total')}
            for i in range(g):
                for j in range(g):
                    a=r.calc(((i+.5)/g,(j+.5)/g),1,.16,.20,gamma)
                    for x in result:result[x]+=a[x]
            for x in result:result[x]/=g*g
            rows.append({'gamma':gamma,'grid':g,'T':m.cp(result['total']), 'magnitude':float(abs(result['total'])), 'phase':float(np.angle(result['total'])),'components':{x:m.cp(z) for x,z in result.items()},'seconds':time.time()-t})
            (ROOT/"q167_bz_partial.json").write_text(json.dumps({"data":rows},indent=2))
            print('gamma',gamma,'grid',g,'T',m.cp(result['total']),'time',rows[-1]['seconds'],flush=True)
    out={'source':'M0 real production Fourier Peierls + exact one-particle-hole 9-state Slater-Condon; electronic ground elastic coefficient', 'data':rows}
    (ROOT/'q167_reduced_bz_gamma_scan.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':scan()

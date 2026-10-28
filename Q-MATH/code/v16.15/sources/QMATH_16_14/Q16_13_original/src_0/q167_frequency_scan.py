#!/usr/bin/env python3
"""BZ third-order elastic T coefficient scan over energy-matched pump pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,sys,time, numpy as np
from pathlib import Path
import q167_m0_three_photon_T as q
ROOT=Path(__file__).resolve().parent

def run(grid=45,gamma=.25):
    rows=[]
    for w1 in (.12,.14,.16,.18):
        w2=.36-w1
        ti=time.time();tot=0j;direct=0j;mixed=0j;sequential=0j
        for i in range(grid):
            for j in range(grid):
                z=q.calc(((i+.5)/grid,(j+.5)/grid),1,w1,w2,gamma)
                tot+=z['total'];direct+=z['direct'];mixed+=z['mixed'];sequential+=z['sequential']
        r={'w1':w1,'w2':w2,'T_plus':q.cp(tot/grid**2),'direct':q.cp(direct/grid**2),'mixed':q.cp(mixed/grid**2),'sequential':q.cp(sequential/grid**2),'magnitude':float(abs(tot/grid**2)), 'phase':float(np.angle(tot)),'seconds':time.time()-ti}
        rows.append(r);print('pair',w1,w2,'|T|',r['magnitude'],'phase',r['phase'], 'seconds',r['seconds'],flush=True)
    output={'grid':grid,'gamma':gamma,'scope':'q=0 plane wave, lowest 3-photon M0 elastic coefficient, not physical continuous spectral wavepacket','pairs':rows}
    (ROOT/f'q167_frequency_scan_g{gamma}.json').write_text(json.dumps(output,indent=2));return output
if __name__=='__main__':run()

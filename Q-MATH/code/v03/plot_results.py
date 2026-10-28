import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=json.loads((P/'QMATH03_RESULTS.json').read_text())
rec=r['records']
vals=[next(x['n_H2'] for x in rec if x['pump']==s and x['neel']==1 and x['detector']=='parallel' and x['npump']==25 and x['nout']==5 and x['nsteps']==160) for s in ['vacuum','coherent','squeezed']]
therm=r['thermal_positive_P']['n_H2_partial']
plt.figure(figsize=(7.0,4.4))
plt.bar(['Vacuum','Coherent','Thermal (positive P)','Squeezed vacuum'],[vals[0],vals[1],therm,vals[2]])
plt.ylabel('Mean H2 photon number')
plt.title('M0 single-k: quantum pump to H2 (finite-mode test)')
plt.xticks(rotation=12,ha='right')
plt.tight_layout();plt.savefig(P/'mean_H2_results.png',dpi=190);plt.close()
plt.figure(figsize=(6.7,4))
for s in ['coherent','squeezed']:
 xs=[];ys=[]
 for x in rec:
  if x['pump']==s and x['neel']==1 and x['detector']=='parallel' and x['npump']==25 and x['nout']==5 and x['nsteps'] in [80,160,320]:
   xs.append(x['nsteps']);ys.append(x['n_H2'])
 plt.plot(xs,ys,marker='o',label=s)
plt.xlabel('Strang time steps (2 cycles)');plt.ylabel('Mean H2 photon number')
plt.title('Time-step convergence');plt.legend();plt.tight_layout();plt.savefig(P/'timestep_convergence.png',dpi=190);plt.close()

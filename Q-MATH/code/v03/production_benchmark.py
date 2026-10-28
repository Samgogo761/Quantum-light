#!/usr/bin/env python3
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import qmath03_exact_unitary as q, q8_m0_model as m
BASE=Path(__file__).parent; geo=m.build_geometry(BASE/'CrI3_scf_soc_a1h_clean.in');basis,G=q.make_ops(); k=geo.k_cart([.4158,.1683]); rec=[]
CONF=dict(npump=25,nout=5,mean=.7,azp=.05,azout=.005,omega=.18,ncycles=2,nsteps=160)
def run(typ,N=1,pol='parallel',kwargs={}):
 d=q.simulate(geo,G,basis,k,N,typ,**(CONF|kwargs),det_polar=pol)
 rec.append(d)
 print(typ,N,pol,'n',f"{d['n_H2']:.9g}",'g2',f"{d['g2_H2']:.6g}" if d['g2_H2'] else None, 'edge',f"{d['H2_edge_probability']:.3g}", 'pumpedge',f"{d['pump_edge_probability']:.3g}", 's',d['elapsed_s'],flush=True)
 return d
for N in [+1,-1]:
 for typ in ['vacuum','coherent','squeezed']:
  run(typ,N)
for typ in ['coherent','squeezed','vacuum']:
 run(typ,1,'perpendicular')
for typ in ['coherent','squeezed']:
 for n in [80,320]:run(typ,1,'parallel',{'nsteps':n})
for nout in [4,6]:run('squeezed',1,'parallel',{'nout':nout})
# thermal is a convex mixture of Fock states, positive Glauber-Sudarshan P; phase-random coherent also Fock mixture.
thermal=[]
for fn in range(11):
 d=run('fock:'+str(fn),1)
 thermal.append(d)
nbar=.7; r=nbar/(1+nbar);wt=np.array([(1-r)*r**k for k in range(11)]);thermalP=sum(w*np.array(d['P_H2']) for w,d in zip(wt,thermal));tail=r**len(wt)
thermN=(np.arange(5)*thermalP).sum(); therG2=(np.arange(5)*(np.arange(5)-1)*thermalP).sum()/thermN**2
summary={'model':'single k: two spin-up + two spin-down M0 four electrons; pump and one H2 mode via Peierls h8(k+g(t) A_op). Two-cycle switchable coupling.', 'parameters':CONF, 'thermal_positive_P':{'P_H2':thermalP.tolist(),'n_H2_partial':float(thermN),'g2_H2_partial':float(therG2),'weight_kept':float(wt.sum()),'tail_omitted':float(tail)},'records':rec}
(BASE/'QMATH03_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('THERMAL partial',thermN,therG2,'tail',tail)
print('WROTE',BASE/'QMATH03_RESULTS.json')

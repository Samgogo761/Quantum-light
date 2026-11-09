"""Check whether SOC-free inelastic count requires Neel order, at fixed k and input packet."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import qmath_16_16_on_shell_m0 as m
import dataclasses,numpy as np,json
from pathlib import Path
org=m.q.m.R0;origb=m.q.BONDS
out=[]
try:
 for lam in (0.,.1):
  for delta in (0.,.4,.8):
   m.q.m.R0=dataclasses.replace(org,lam_so=lam,delta=delta);m.q.BONDS=m.q.bonds()
   for kf in ((.4158,.1683),(.3703,.1913),(.276,.448)):
    model=m.setup(kf,1)
    oms=np.linspace(.20,.34,15)
    S=m.fixed_pulse(model,oms,nq=17,gamma=.25)
    I0=float(np.trapezoid(abs(S[:,0])**2,oms))
    Iin=float(np.trapezoid(np.sum(abs(S[:,1:])**2,axis=1),oms))
    out.append({'lambda':lam,'delta':delta,'kfrac':kf,'direct_gap':float(model[3][1]),'elastic_local_norm2':I0,'inelastic_local_norm2':Iin})
    print(lam,delta,kf,'gap',model[3][1],'inel',Iin,flush=True)
finally:m.q.m.R0=org;m.q.BONDS=origb
Path(__file__).resolve().with_name('qmath_16_16_magnetism_null.json').write_text(json.dumps(out,indent=2))

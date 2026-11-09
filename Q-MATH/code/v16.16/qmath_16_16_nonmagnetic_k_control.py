"""Independent same-model delta=0 nonmagnetic false-positive control, k-resolved."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import qmath_16_16_on_shell_m0 as m
import dataclasses,numpy as np,json
from pathlib import Path
B=Path(__file__).resolve().parent
old=m.q.m.R0;oldb=m.q.BONDS
out=[]
try:
 for D in (0., .4, .8):
  m.q.m.R0=dataclasses.replace(old,lam_so=0.,delta=D);m.q.BONDS=m.q.bonds()
  for kf in ((1/3,1/3),(.34,.33),(.36,.31),(.42,.18)):
   model=m.setup(kf,1)
   om=np.linspace(.20,.34,15)
   S=m.fixed_pulse(model,om,11,.25)
   val=float(np.trapezoid(np.sum(abs(S[:,1:])**2,axis=1),om))
   out.append({'Delta':D,'lambda':0,'kfrac':kf,'minimum_electron_excitation':float(model[3][1]),'inelastic_positive_kernel_norm2':val})
finally:m.q.m.R0=old;m.q.BONDS=oldb
(B/'qmath_16_16_nonmagnetic_k_control.json').write_text(json.dumps({'scope':'original M0 reduced Peierls third-order T, single k, not BZ converged; false positive without Neel order','tests':out},indent=2))
for a in out:
 print(a)

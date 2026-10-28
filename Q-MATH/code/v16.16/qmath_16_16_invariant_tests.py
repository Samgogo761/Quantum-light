"""Analytical and source-to-source checks for the Q-MATH-16.16 on-shell calculation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np, dataclasses, json
from pathlib import Path
import qmath_16_16_on_shell_m0 as w
q=w.q; o=q.m.R0;ob=q.BONDS
res={'scope':'exact production q167 elastic self consistency and on-shell kinematical support; results are bare regulated T'}
try:
 elastic=[]; inv=[]; positivity=[]
 for lam in (0., .1):
  w.swap_soc(lam)
  for kf in ((.4158,.1683),(.3703,.1913),(.173,.271)):
   M=w.setup(kf,1)
   x=w.amplitudes(M,.16,.20,.36,.25)[0]
   y=q.calc(kf,1,.16,.20,.25)['total']
   elastic.append({'lam':lam,'kf':kf,'abs_error':float(abs(x-y)),'relative_error':float(abs(x-y)/max(1e-15,abs(y)))})
   # energy window validity for excited transitions
   gap=M[3]
   positive_x =float(np.max(gap[1:]))
   inv.append({'lam':lam,'kf':kf,'min_excitation':float(gap[1]),'max_excitation':positive_x})
   oms=np.linspace(.22,.38,9)
   S=w.fixed_pulse(M,oms,11,.25)
   positivity.append({'lam':lam,'kf':kf,'positive_integral':float(np.trapezoid(np.sum(abs(S[:,1:])**2,axis=1),oms)), 'all_finite':bool(np.isfinite(S).all())})
 res['q167_elastic_crosscheck']=elastic
 res['max_relative_elastic_source_error']=max(x['relative_error'] for x in elastic)
 res['positive_kernel_checks']=positivity
 print(json.dumps(res,indent=2))
finally:q.m.R0=o;q.BONDS=ob
Path(__file__).resolve().with_name('qmath_16_16_invariant_results.json').write_text(json.dumps(res,indent=2))

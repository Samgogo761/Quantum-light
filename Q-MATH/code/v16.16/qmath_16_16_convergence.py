"""Q16.16 independent quadrature and BZ resolution diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from pathlib import Path
import qmath_16_16_on_shell_m0 as m
B=Path(__file__).resolve().parent
org=m.q.m.R0; origb=m.q.BONDS
out={'scope':'gamma in virtual denominators only; on-shell final state exact; numbers not absolute rates','quadrature':[],'BZ':[],'positive_p_control':{}}
try:
 for soc in (0.0,.1):
  m.swap_soc(soc)
  kf=(.4158,.1683);model=m.setup(kf,1)
  omega=np.linspace(.21,.31,17)
  for gamma in (.25,.1):
   for nq in (7,11,17,25):
    res=m.fixed_pulse(model,omega,nq,gamma)
    val=float(np.trapezoid(np.sum(abs(res[:,1:])**2,axis=1),omega))
    out['quadrature'].append({'lambda':soc,'gamma':gamma,'nq':nq,'local_inelastic_norm2':val})
   print('quadrature',soc,gamma,'done',flush=True)
 # sparse grid strong nonconvergence warning on total nonnegative inelastic count
 om=np.linspace(.22,.34,11)
 for soc in (0.,.1):
  m.swap_soc(soc)
  for gamma in (.25,.1):
   for L in (9,13,17,25,33):
    st=time.perf_counter();acc=0.;count=0
    for i in range(L):
     for j in range(L):
      model=m.setup(((i+.5)/L,(j+.5)/L),1)
      # zero contribution if lowest excitation exceeds maximum 2wmax - Omin
      if model[3][1]>=2*m.WMAX-min(om):continue
      r=m.fixed_pulse(model,om,nq=7,gamma=gamma)
      count+=1
      acc+=float(np.trapezoid(np.sum(abs(r[:,1:])**2,axis=1),om))
    # do not average only over active points; include zeros for missing grid sample
    out['BZ'].append({'lambda':soc,'gamma':gamma,'L':L,'active_k':count,'average_positive_inelastic_norm2':acc/(L*L),'seconds':time.perf_counter()-st})
    print('BZ',soc,gamma,L,'nk',count,'avg',acc/(L*L),'sec',time.perf_counter()-st,flush=True)
 # exact P control for all bins simultaneously; only requires equality of F2
 E=.7;F2=3*E**2+E
 X=F2/E;p=E/X
 out['positive_p_control']={'E':E,'F2':F2,'P_vacuum':1-p,'P_plus_coherent':p/2,'P_minus_coherent':p/2,'coherent_amplitudes':[float(X**.5),float(-X**.5)],'E_classical':p*X,'F2_classical':p*X**2,'BSV_to_classical_count_all_bins':1.0,'note':'same fixed spectral mode u, weak 2 photon annihilation only'}
finally:
 m.q.m.R0=org;m.q.BONDS=origb
(B/'qmath_16_16_convergence_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))

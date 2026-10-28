#!/usr/bin/env python3
"""Q-MATH-16.18: retarded causal M0 cumulants and coupling-scan inverse witness.
Not LSZ/Maxwell or fully quantized BSV HHG. All estimator tests label synthetic coefficients.
"""
from pathlib import Path
import numpy as np, json, math, sys, time
from itertools import product
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import qmath_16_18_keldysh_determinant as d
q=d.q;m=d.m


def drive(steps=20,dt=.072):
 t=(np.arange(steps)+.5)*dt
 env=np.sin(np.pi*(np.arange(steps)+.5)/steps)**2
 pump=(env*np.sin(1.5*t))[:,None]*q.p[None,:]
 probes=[(env*np.sin(2.8*t))[:,None]*q.h[None,:],(env*np.cos(2.8*t))[:,None]*q.h[None,:]]
 return pump,probes


def response(L,prm,N,h,steps=20,dt=.072):
 pump,probes=drive(steps,dt)
 # compute current at five pump amplitudes, sufficient for second/fourth derivatives.
 J=np.zeros((2,5))
 for i in range(L):
  for j in range(L):
   k=q.geo.k_cart(((i+.5)/L,(j+.5)/L))
   for idx,a in enumerate([-2*h,-h,0,h,2*h]):
    for p,probe in enumerate(probes):
     J[p,idx]+=d.integrate_current(k,N,pump,probe,a,dt,prm)/L**2
 c2=(J[:,3]-2*J[:,2]+J[:,1])/h**2
 c4=(J[:,0]-4*J[:,1]+6*J[:,2]-4*J[:,3]+J[:,4])/h**4
 return {'C2':c2,'C4':c4}


def run_m0():
 out=[]; st=time.time()
 for L in (3,5,7):
  for lam,Delta in ((.1,.8),(0.,.8),(.1,0.)):
   prm=m.M0Params(lam_so=lam,delta=Delta)
   for N in (1,-1):
    r=response(L,prm,N,.08)
    out.append({'L':L,'lambda':lam,'Delta':Delta,'N':N,'h':.08,'C2':r['C2'].tolist(),'C4':r['C4'].tolist()})
   print('done',L,lam,Delta,'elapsed',round(time.time()-st,1),flush=True)
   (ROOT/'qmath_16_18_causal_partial.json').write_text(json.dumps(out,indent=2))
 # check derivative finite h at the middle grid and varying time discretization.
 checks=[]
 prm=m.M0Params(lam_so=.1,delta=.8)
 for h,steps in ((.12,20),(.08,20),(.05,20)):
  r=response(5,prm,1,h,steps=steps,dt=.072*20/steps)
  checks.append({'L':5,'h':h,'steps':steps,'C2':r['C2'].tolist(),'C4':r['C4'].tolist()})
 return out,checks


def bsv_moments(E):
 F0,F1=1.,E
 F2=3*E**2+E
 F3=15*E**3+9*E**2
 F4=105*E**4+90*E**3+9*E**2
 return {'E':E,'F2':F2,'F3':F3,'F4':F4,'witness':E*F3-F2**2}


def inverse_test(E=.1):
 b=bsv_moments(E)
 # synthetic complex coefficients, deliberately fully proportional across all spectral output modes.
 S0=1+.30j; S10=.20-.10j; S11=.35+.15j
 aa=abs(S0)**2; bb=2*(S0.conjugate()*S10).real; cc=2*(S0.conjugate()*S11).real
 # N_eta/g^4 = |S0|^2 F2 + eta*(2Re S0*S10 F2 + 2Re S0*S11 F3) + eta^2 * ...
 # exact response through fifth amplitude, including finite eta^2 terms with F4 contamination.
 def count(eta,F2,F3,F4):
  ss=S0+eta*S10
  return (abs(ss)**2*F2+2*eta*(ss.conjugate()*S11).real*F3+
    eta**2*abs(S11)**2*(F4+F3))
 eta=np.array([0.001,0.002,0.003,0.004])
 measures=np.array([count(x,b['F2'],b['F3'],b['F4']) for x in eta])
 A=np.column_stack([aa+bb*eta,cc*eta])
 est=np.linalg.lstsq(A,measures,rcond=None)[0]
 # exact polynomial fit intercept & slope in eta; absorb eta² curvature instead of forgetting it.
 poly=np.polyfit(eta,measures,2)
 F2_fit=poly[-1]/aa
 F3_fit=(poly[-2]-bb*F2_fit)/cc
 # Classical full positive P matching E,F2 has F3 >=F2²/E.
 lower_classical=b['F2']**2/E
 eta_big=np.array([.005,.01,.02])
 signal_bs=np.array([count(x,b['F2'],b['F3'],b['F4']) for x in eta_big])
 # Classical extremal 0/bright mixture matches E and F2; Fk=E*(F2/E)**(k-1)
 R=b['F2']/E
 clasF3=E*R**2;clasF4=E*R**3
 signal_cl=np.array([count(x,b['F2'],clasF3,clasF4) for x in eta_big])
 rank=np.linalg.svd(A,compute_uv=False)
 return {'scope':'synthetic coefficients, exact truncated A=eta O0+eta²(S10 O0+S11 O1); not M0 prediction',
  'phase_mode_coefficient_S0':[S0.real,S0.imag],
  'S10':[S10.real,S10.imag],'S11':[S11.real,S11.imag],
  'A0':aa,'B0':bb,'B1':cc,'eta_values':eta.tolist(),'normed_counts':measures.tolist(),
  'linearized_estimate':est.tolist(),'quadratic_fit':{'F2':F2_fit,'F3':F3_fit},
  'true':b,'classical_F3_min':lower_classical,
  'rank_2':bool(rank[-1]>1e-14),'rank_condition_number':float(rank[0]/rank[-1]),
  'classical_extremal_0_bright':{'intensity':R,'F3':clasF3,'F4':clasF4},
  'eta_larger':eta_big.tolist(),'counts_BSV':signal_bs.tolist(),'counts_classical':signal_cl.tolist(),
  'relative_differences_classical_minus_BSV':((signal_cl-signal_bs)/signal_bs).tolist()}

if __name__=='__main__':
 import sys
 if '--full' in sys.argv:
  st=time.time();calc,checks=run_m0()
  out={'scope':'finite-time connected causal susceptibility of original M0 and synthetic multi-coupling inverse; no photon LSZ',
       'M0_causal_response':calc,'discretization_checks':checks,
       'synthetic_inverse':inverse_test(), 'runtime':time.time()-st}
  (ROOT/'qmath_16_18_causal_inverse_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,
       default=lambda z:float(z) if isinstance(z,np.generic) else str(z)))
  print('FULL scan done in',round(out['runtime'],2),'seconds')
 else:
  z=inverse_test()
  print('Synthetic inverse only (run --full for the full M0 BZ/time-step scan)')
  print('BSV E=0.1: fitted',z['quadratic_fit'],'classical F3 minimum',z['classical_F3_min'])
  print('measurement condition number',z['rank_condition_number'])

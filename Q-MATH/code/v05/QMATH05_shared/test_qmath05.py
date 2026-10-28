"""Independent smoke/regression tests for Q-MATH-05.
Run OPENBLAS_NUM_THREADS=1 python test_qmath05.py
"""
import json,numpy as np
import shared
from pathlib import Path

def phase(z):return complex(*z)

def test():
  kw=dict(npump=7,nharm=2,nsteps=24)
  a=shared.calc(1,'vacuum',0.,**kw)
  b=shared.calc(-1,'vacuum',0.,**kw)
  assert abs(a['n_parallel']-b['n_parallel'])<1e-12
  assert abs(phase(a['m_cross'])-phase(b['m_cross']))<1e-12
  c=shared.calc(1,'coherent',0.,azp=0.,azh=0.,**kw)
  assert c['n_parallel']<1e-12 and c['n_perp']<1e-12
  assert abs(phase(c['m_cross']))<1e-12
  d=shared.calc(1,'seeded',np.pi/2,**kw)
  e=shared.calc(-1,'seeded',3*np.pi/2,**kw)
  inv=max(abs(d['n_parallel']-e['n_parallel']),abs(d['n_perp']-e['n_perp']),abs(phase(d['m_cross'])-phase(e['m_cross'])))
  assert inv<1e-11,inv
  print('PASS: vacuum parity, zero coupling, seeded alpha->-alpha inversion', 'max',inv)
  (Path(__file__).parent/'unit_test_output.json').write_text(json.dumps({'pass':True,'inversion_error':inv,'zero_coupling_npar':c['n_parallel']},indent=2))
if __name__=='__main__':test()

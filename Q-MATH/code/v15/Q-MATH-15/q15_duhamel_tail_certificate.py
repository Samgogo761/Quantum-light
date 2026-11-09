#!/usr/bin/env python3
"""Analytic far-Fock tail bound for displacement, beyond rectangular residual sampling.
For initial support <=N-1 and any |eta|<=eta_max:
|| Q_{>=N+M} exp(i eta(a+a†)) P_{<N} || <= sum_{l>=M+1} (2|eta|)^l/l! sqrt((N+l)!/N!).
The factorial choice is slightly looser than the exact operator-norm power bound.
"""
import numpy as np,sys,json
from pathlib import Path
from scipy.special import gammaln,logsumexp
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'QMATH13_M0_H2H4'))
from q13_M0_BSV_weak_radiation import M0Source
s=M0Source(5,1)
eta_max=max(abs(s.azp*np.dot(s.ep,np.array(d[:2]))) for d,O in s.terms)
sum2ops=sum(2*np.linalg.norm(O.toarray(),2) for d,O in s.terms)
T=4*np.pi/s.omega
out={'eta_max':eta_max,'sum_two_ops':sum2ops,'T':T,'bounds':[]}
for N in (13,17,23):
 for extra in (20,30):
  l=np.arange(extra+1,600)
  terms=l*np.log(2*eta_max)+.5*(gammaln(N+l+1)-gammaln(N+1))-gammaln(l+1)
  tail=float(np.exp(logsumexp(terms)))
  out['bounds'].append({'Np':N,'extra_levels':extra,'missing_unitary_displacement_norm_bound':tail,
         'missing_Duhamel_integral_bound':tail*sum2ops*T})
(ROOT/'q15_duhamel_tail_certificate.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

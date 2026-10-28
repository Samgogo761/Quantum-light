#!/usr/bin/env python3
"""Q-MATH-15 corrected universal phase protocol, certified finite-K classical envelope.

Run: python q15_audit_certificate.py
Requires the Q14 q13_kernel_23_96_Nplus.npz / ...Nminus.npz nearby in data/.
"""
import json
from pathlib import Path
from decimal import Decimal, localcontext
import numpy as np
from scipy.special import gammaln
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parent
D=ROOT/'data'
p=np.load(D/'q13_kernel_23_96_Nplus.npz')['kernel'][1,0]
m=np.load(D/'q13_kernel_23_96_Nminus.npz')['kernel'][1,0]
K=(p-m)/2
n=len(K);j=np.arange(n-2)
a=np.diag(K,2)*np.exp(-.5*(gammaln(j+1)+gammaln(j+3)))
c=np.abs(a)
C0=abs(a[0])
coefratios=c*np.exp(gammaln(j+1))/C0
assert np.all(coefratios<=1+1e-12), 'coefficientwise sharp bound fails'

def amp(x):return np.exp(-x)*np.polynomial.polynomial.polyval(x,a)
def bsv(nbar):
 t=np.sqrt(nbar/(1+nbar));v=np.zeros(n,dtype=complex)
 jj=np.arange((n+1)//2);ii=2*jj
 v[ii]=(-t)**jj*np.exp(.5*gammaln(ii+1)-gammaln(jj+1)-jj*np.log(2))/(1+nbar)**.25
 v/=np.linalg.norm(v)
 return sum(np.conj(v[k])*v[k+2]*K[k,k+2] for k in range(0,n-2,2)), float(np.dot(np.arange(n),abs(v)**2))

# Certified interval upper bound for every x in [0, 7]. Polynomial has positive coefficients.
# For x in [left,right]:  x exp(-x) sum c_j x^j <= right exp(-left) sum c_j right^j.
with localcontext() as ctx:
 ctx.prec=55
 relative_upper=Decimal('1.0000000000001')
 dec_coefs=[Decimal(str(float(q)))*relative_upper+Decimal('1e-40') for q in c]
 A=Decimal('0.098');B=Decimal('0.25')
 minmargin=Decimal('Inf');minindex=None
 for i in range(1400):
  lo=Decimal(i)/Decimal(200);hi=Decimal(i+1)/Decimal(200)
  poly=Decimal('0')
  for cj in reversed(dec_coefs):poly=poly*hi+cj
  up=hi*(-lo).exp()*poly
  lower=A+B*lo
  gap=lower-up
  if gap<minmargin:minmargin,minindex=gap,i
 assert minmargin>Decimal('0.005'), (minmargin,minindex)
 tailmax=float(max(abs(np.diag(K,2))))
 assert float(A+B*Decimal('7'))>tailmax

# Universal full-boson bound is obtained independently in norm_bound.py from actual M0 bonds.
universal_K_norm_bound=1760.2181543590432
# quantum-input example at weak mean if full |K02| were validated
n_ultras=[1e-8,1e-7,1e-6]
conditional=[]
import mpmath as mp
mp.mp.dps=60
for nbar in n_ultras:
 nn=mp.mpf(str(nbar))
 p0=(1+nn)**(-mp.mpf('.5'))
 p2=nn/(2*(1+nn)**(mp.mpf('1.5')))
 pm2=1-p0;pm4=1-p0-p2
 B0=mp.mpf(str(universal_K_norm_bound))
 leading=mp.mpf(str(abs(K[0,2])))*mp.sqrt(nn)/(mp.sqrt(2)*(1+nn))
 remainder=B0*mp.sqrt(pm2*pm4)
 cb=B0*nn/mp.sqrt(2)
 conditional.append({'n':nbar,'BSV_lower_if_true_K02_matches_finite':float(leading-remainder),'classic_universal_bound':float(cb),'ratio_lower':float((leading-remainder)/cb)})

scans=[]
for mean in [.001,.01,.1,.7]:
 z,actual=bsv(mean)
 scans.append({'target_mean':mean,'actual_BSV_mean':actual,'BSV_winding_abs':abs(z), 'linear_classic_tight_23Fock':float(mean*C0),'affine_classic_23Fock':.098+.25*mean})

# LP is a numerical upper-concave-envelope estimate, NOT a global proof.
xs=np.r_[np.linspace(0,2,2001),np.linspace(2.01,7,500)]
h=np.array([x*abs(amp(x)) for x in xs])
for entry in scans:
 mean=entry['target_mean']
 opt=linprog(-h,A_eq=np.vstack([np.ones(len(xs)),xs]),b_eq=[1.,mean],bounds=(0,None),method='highs')
 assert opt.success
 entry['numerical_concave_envelope_noncertified']=-opt.fun

# Wrong 2pi protocol explicitly violated by perfectly classical single coherent probe
alpha=.1;theta=np.arange(16384)*2*np.pi/16384
wrong=np.mean(np.exp(-1j*theta)*np.exp(-alpha**2)*alpha*np.exp(.5j*theta))
theta4=np.arange(16384)*4*np.pi/16384
correct=np.mean(np.exp(-1j*theta4)*np.exp(-alpha**2)*alpha*np.exp(.5j*theta4))

out={'phase_counterexample':{'classical_alpha':alpha,'2pi_complex':[wrong.real,wrong.imag], '2pi_modulus':abs(wrong),'4pi_modulus':abs(correct),'C_K_for_K01_only':0},
 'K_source':'true production M0 finite single k, 23 pump Fock, 96 steps, H2 perp magnetic odd K',
 'K02_odd':[K[0,2].real,K[0,2].imag],'finite_Cstar_exact':float(C0),'all_coefficientwise_ratios_below_one':bool(np.max(coefratios)<1.0000001),'largest_other_ratio':float(max(coefratios[1:])),
 'affine_certificate':{'A':.098,'B':.25,'interval_0_7_decimal_min_margin':str(minmargin),'interval_width':.005,'tail_operator_shift_bound':tailmax,'tail_line_at_7':1.848,'statement':'for all x>=0, |<alpha|K_plus2|alpha>|<=.098+.25*x'},
 'scans':scans,'full_infinite_Fock_operator_norm_bound':universal_K_norm_bound, 'full_infinite_Fock_Cstar_universal_bound':universal_K_norm_bound/np.sqrt(2),
 'ultralow_input_conditional':conditional,
 'limits':['The numerical kernel is finite 23 Fock single k; the decimal certificate is for its saved entries, allowing coefficient outward margin 1e-14.',
           'The norm bound for infinite boson Peierls K is rigorous for the chosen finite M0 bond model, envelope and integration time, but the actual infinite K02 has not been rigorously certified.',
           'The numerical concave envelope is grid-based and not a certified global optimum.',
           'Emission-mode normalization and physical single-sided boundary are not yet included.']}
(ROOT/'q15_certified_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({k:out[k] for k in ('K02_odd','finite_Cstar_exact','largest_other_ratio','affine_certificate','scans','ultralow_input_conditional')},ensure_ascii=False,indent=2))

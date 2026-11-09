#!/usr/bin/env python3
"""Q-MATH-16.9 reproducible math verification; NOT a recomputation of M0 scattering."""
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from pathlib import Path
OUT=Path(__file__).parent

def occupation(lam, sig):
    sig=np.array(sig,dtype=float)
    return 0.5*(lam/np.sqrt(lam*lam-sig*sig)-1)

def optimal_bsv(sig, E):
    sig=np.array(sig,dtype=float)
    assert E>0 and sig.max()>0
    nonzero=sig[sig>0]
    low=nonzero.max()*(1+1e-14)
    high=max(2*low,1.)
    while np.sum(occupation(high,nonzero))>E:
        high*=2
    lam=brentq(lambda l:np.sum(occupation(l,nonzero))-E,low,high,xtol=1e-14)
    n=occupation(lam,nonzero)
    Q=float(np.sum(nonzero*np.sqrt(n*(n+1))))
    dual=float(lam*E+0.5*np.sum(lam-np.sqrt(lam*lam-nonzero*nonzero)))
    return Q,n,lam,dual

def tail_upper(sig,E):
    s=np.max(sig)
    lam0=s*(2*E+1)/(2*np.sqrt(E*(E+1)))
    tail=np.sort(np.array(sig,float))[::-1][1:]
    # rationalized expression avoids cancellation
    return float(np.sum(tail*tail/(2*(lam0+np.sqrt(lam0*lam0-tail*tail)))))

E=0.7
p_tail=0.00028518
s0=np.sqrt(1-p_tail)
st=np.sqrt(p_tail)
sig=np.array([s0,st])
F,n,lam,dual=optimal_bsv(sig,E)
F1=s0*np.sqrt(E*(E+1))
exact_gain=F/F1-1
approx_gain=(st/s0)**2/(2*(2*E+1))
upper_gain=tail_upper(sig,E)/F1
assert abs(F-dual)<1e-10
assert exact_gain<=upper_gain+1e-11

# Second two-frequency physical-energy example (finite-dimensional counterexample)
# U=1, frequencies (omega_H=2, omega_L=1), K=diag(2, 1.1).
U=1.0
w=np.array([2.0,1.0])
k=np.array([2.0,1.1])
def bsv_amp(x,k=k):
    z=np.array([x,1-x])
    wbar=z@w
    ne=U/wbar
    return float((z@k)*np.sqrt(ne*(ne+1)))
res=minimize_scalar(lambda x:-bsv_amp(x),bounds=(0,1),method="bounded",options={"xatol":1e-14})
x_candidates=[0.,1.,float(res.x)]
q_opt=max((bsv_amp(x),x) for x in x_candidates)
c_opt=U*np.max(k/w)
assert abs(q_opt[0]-np.sqrt(3))<1e-9, q_opt
assert abs(c_opt-1.1)<1e-12
caseA={"omega":w.tolist(),"k":k.tolist(),"classical_opt":c_opt,"bsv_opt":q_opt[0],"bsv_x_high_frequency":q_opt[1],"ratio":q_opt[0]/c_opt}
caseB={"omega":w.tolist(),"k":[0.,1.1],"classical_opt":1.1,"bsv_opt":1.1*np.sqrt(2),"ratio":np.sqrt(2.)}

# Kraus rank-one vs rank-two proof examples: pair vectors, K_f = coefficients times pair operator
pair_K_rank1=np.array([[1.,0.],[2.,0.]]) # f=G,X; pair operator basis (a1²,a2²)
pair_K_rank2=np.array([[1.,0.],[0.,2.]])
rank1=np.linalg.matrix_rank(pair_K_rank1)
rank2=np.linalg.matrix_rank(pair_K_rank2)
assert (rank1,rank2)==(1,2)

# Original 16.8 dark-mode counterexample resource audit
Ed=.2
Ib=.5
Id_hi=8.6
p=Ed/Id_hi
En=Ib+p*Id_hi
F2=Ib**2+2*Ib*Ed+p*Id_hi**2
assert np.isclose(En,E) and np.isclose(F2,3*E**2+E)

# Universal photon-count bound with single wavepacket (old theorem)
universal_ratio=np.sqrt((E+1)/E)

payload={
    "caveat":"Toy and analytic numerical audit only. Original M0 raw code/complex-frequency arrays were not available as executable bytes in this session.",
    "M0_reported_41grid_second_mode_fraction":p_tail,
    "tail_test":{"E":E,"model_singular_values":sig.tolist(),"exact_relative_gain_percent":100*exact_gain,"leading_relative_gain_percent":100*approx_gain,"certified_dual_upper_percent":100*upper_gain,"lambda":lam,"primal_dual_gap":F-dual,"modes_n":n.tolist(),"previous_M0_reported_61grid_relative_gain_percent":0.005844},
    "single_mode_photon_number_universal_ratio":universal_ratio,
    "fixed_physical_energy_examples":{"A":caseA,"B":caseB},
    "electronic_channel_operator_schmidt_ranks":{"rank_one":int(rank1),"rank_two":int(rank2)},
    "classical_dark_mode_counterexample":{"E":En,"F2":F2,"coherent_fraction":1.0}
}
(OUT/'qmath_16_9_verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2))

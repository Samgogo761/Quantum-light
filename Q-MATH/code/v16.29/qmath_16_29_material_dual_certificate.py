#!/usr/bin/env python3
"""Numerical dual witnesses for the SAME-state 5-bin material-only null,
within the retarded CLASSICAL chi2 PROXY. Output is NOT physical HHG.
"""
from pathlib import Path
import json,numpy as np
from scipy.optimize import minimize
from qmath_16_29_same_state_readout import state,min_cost_restricted
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'qmath_16_29_material_null_results.json').read_text())
data=json.loads((ROOT/'qmath_16_29_results.json').read_text())
W=np.array(data['input_energy_W']); el,ev=np.linalg.eigh(W); Wmh=(ev*(1/np.sqrt(el)))@ev.T
m=np.array([complex(*v) for v in data['input_m']]);Uin=float(data['U'])
def mkmatrix(v):return np.array([[v[0],v[2]/2],[v[2]/2,v[1]]],complex)
def ratio(lam,Q,y):
    KK=np.conj(lam)@Q
    eff=Wmh@mkmatrix(KK)@Wmh
    bound=Uin*np.linalg.svd(eff,compute_uv=False)[0]
    return float(abs(np.vdot(lam,y))/bound)
for c in base['cases']:
    K=np.array([[complex(*v) for v in row['material_row']] for row in c['rows']]);F=np.array([[complex(*v) for v in row['frequency_flat_control_row']] for row in c['rows']]);
    UF,s,Vh=np.linalg.svd(F,full_matrices=True)
    Q=UF[:,3:].conj().T@K;y=Q@m
    if 'projected_pair_min_classical_cost_over_U' not in c:
        _,_,_,Wh=state()
        fit=min_cost_restricted(Q,y,Wh)
        c['projected_pair_min_classical_cost_over_U']=float(fit['cost']/Uin)
    def objective(x):return -ratio(np.array([1.,x[0]+1j*x[1]]),Q,y)
    seeds=[[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,-1],[3,-2],[-4,3]]
    sols=[minimize(objective,x,method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-12,'fatol':1e-12}) for x in seeds]
    best=min(sols,key=lambda z:z.fun)
    lam=np.array([1,best.x[0]+1j*best.x[1]])
    phase=np.angle(np.vdot(lam,y));lam*=np.exp(1j*phase)
    rv=ratio(lam,Q,y)
    ctl=np.conj(lam)@UF[:,3:].conj().T@F
    c['dual_certificate']={'ratio_BSV_to_classical_positive_P_energy_bound':float(rv),
         'dual_observer_coefficients':[ [float(z.real),float(z.imag)] for z in lam],
         'dual_numerator':float(abs(np.vdot(lam,y))),
         'dual_classical_energy_bound':float(abs(np.vdot(lam,y))/rv),
         'frequency_flat_null_response_abs':float(np.linalg.norm(ctl)),
         'dual_vs_primal_gap':float(abs(rv-c['projected_pair_min_classical_cost_over_U']))}
    print(c['L'],c['eta'],c['Delta'],'dual',rv,'primal',c['projected_pair_min_classical_cost_over_U'], 'gap',c['dual_certificate']['dual_vs_primal_gap'],'flat0',c['dual_certificate']['frequency_flat_null_response_abs'])
(ROOT/'qmath_16_29_material_dual_results.json').write_text(json.dumps(base,ensure_ascii=False,indent=2))

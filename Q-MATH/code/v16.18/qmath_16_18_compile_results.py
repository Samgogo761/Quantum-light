#!/usr/bin/env python3
"""Combine Q18 deterministic tests, build summary and plots, no heavy M0 rerun."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import qmath_16_18_causal_rank_and_inverse as qr
old=json.loads((ROOT/'qmath_16_18_results.json').read_text())
scan=json.loads((ROOT/'qmath_16_18_causal_partial.json').read_text())
checks=[
 {'h':.12,'L':5,'steps':20,'C2':[-.08314888,.19341859],'C4':[5.65874007,-14.93702861]},
 {'h':.08,'L':5,'steps':20,'C2':[-.08758657,.20539489],'C4':[6.42198337,-17.25149634]},
 {'h':.05,'L':5,'steps':20,'C2':[-.08983537,.21149330],'C4':[6.83906735,-18.51985870]},
]
for row in scan:
 if row['N']!=1 or row['lambda']!=.1 or row['Delta']!=.8: continue
 M=np.column_stack([row['C2'],row['C4']]); s=np.linalg.svd(M,compute_uv=False)
 row['response_2x2_det']=float(np.linalg.det(M))
 row['normalized_angle_sine']=float(abs(np.linalg.det(M))/(np.linalg.norm(M[:,0])*np.linalg.norm(M[:,1])))
 row['raw_condition_number']=float(s[0]/s[1])
for row in checks:
 M=np.column_stack([row['C2'],row['C4']]);row['normalized_angle_sine']=float(abs(np.linalg.det(M))/(np.linalg.norm(M[:,0])*np.linalg.norm(M[:,1])))
inv=qr.inverse_test()
# NumPy scalar to plain JSON conversion.
inv=json.loads(json.dumps(inv,default=lambda o:o.item() if isinstance(o,np.generic) else o.tolist() if isinstance(o,np.ndarray) else str(o)))
summary={
 'stage':'Q-MATH-16.18','date':'2026-10-24',
 'status':'M0 exact free-fermion Keldysh identity + finite-time causal current candidate + strict coupling-scan inverse theorem. Not connected photon S matrix.',
 'full_36_vs_determinant':old['exact_determinant_checks'],
 'max_absolute_determinant_residual':max(z['residual'] for z in old['exact_determinant_checks']),
 'self_branch_residual_max':max(z['same_branch_residual'] for z in old['exact_determinant_checks']),
 'retarded_discretization':{'Nsteps':[35,70,140],
 'error_current_leftpoint_vs_exact_Keldysh':[.02619664479747552,.013063383536376705,.006518982503853188],
 'interpretation':'Left-point current quadrature has O(dt) time-discretization error; exact Keldysh determinant is the reference.'},
 'production_M0_finite_time_second_fourth_current_response':scan,
 'finite_difference_step_sensitivity_L5':checks,
 'synthetic_coupling_inverse_proof_of_concept':inv,
 'unresolved':['physical connected five-photon S matrix photon external legs','BSV quantum Keldysh contour integration and output statistics','BZ convergence and fourth derivative extrapolation','Maxwell port calibration','noise interval certification']
}
(ROOT/'qmath_16_18_final_results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
fig,axes=plt.subplots(1,2,figsize=(11.5,4.0))
for col,key in enumerate(('C2','C4')):
 for N in (+1,-1):
  subset=[x for x in scan if x['N']==N and x['lambda']==.1 and x['Delta']==.8]
  subset.sort(key=lambda x:x['L']);x=[v['L'] for v in subset]
  axes[col].plot(x,[r[key][0] for r in subset],marker='o',label=f'N={N:+d}, probe sine')
  axes[col].plot(x,[r[key][1] for r in subset],marker='s',linestyle='--',label=f'N={N:+d}, probe cosine')
 axes[col].set_xlabel('BZ mesh L x L')
 axes[col].set_ylabel('Finite-time response derivative (arb. units)')
 axes[col].set_title(f'{key}: fixed input pulse, two output probes')
 axes[col].legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'qmath_16_18_causal_M0_BZ.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.5,4.0))
E=.1;m=inv['true'];etas=np.linspace(0,.022,100)
S0=1+.3j;S10=.2-.1j;S11=.35+.15j
def count(eta,F2,F3,F4):
 ss=S0+eta*S10
 return abs(ss)**2*F2+2*eta*np.real(np.conj(ss)*S11)*F3+eta**2*abs(S11)**2*(F3+F4)
axes[0].plot(etas,[count(z,m['F2'],m['F3'],m['F4']) for z in etas],label='Pure BSV E=0.1')
F3=inv['classical_extremal_0_bright']['F3'];F4=inv['classical_extremal_0_bright']['F4']
axes[0].plot(etas,[count(z,m['F2'],F3,F4) for z in etas],label='Positive-P classical (same E,F2)')
axes[0].set_xlabel('Adjustable squared interaction strength eta')
axes[0].set_ylabel('Photon count / leading eta² prefactor')
axes[0].set_title('Synthetic coefficients: not M0 prediction')
axes[0].legend(fontsize=8)
axes[1].plot([x['h'] for x in checks],[x['normalized_angle_sine'] for x in checks],marker='o')
axes[1].set_xlabel('Finite-difference pump amplitude step h')
axes[1].set_ylabel('sin(angle) of finite-time C2 and C4 response')
axes[1].set_title('M0: L=5, same pump and output-probe pair')
fig.tight_layout();fig.savefig(ROOT/'qmath_16_18_inverse_and_rank.png',dpi=180);plt.close(fig)
print('det residual',summary['max_absolute_determinant_residual'])
print('rows',len(scan),'synthetic F3 fit',inv['quadratic_fit'],'cond',inv['rank_condition_number'])
print('saved',ROOT)

#!/usr/bin/env python3
"""Reproduce Q-MATH-16.26 limited-pulse nonlinear Peierls response audit."""
import os
for s in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[s]='1'
from pathlib import Path
import json, time, numpy as np
from qmath_16_26_cubic_audit import propagate, bond_data, derivatives, q
ROOT=Path(__file__).resolve().parent

def vertex_check():
    fr,cart,base,bonds=bond_data(3,1)
    H=derivatives(base,bonds,4)
    eps=8e-4
    def shifted(delta):
        return derivatives(base,[(phase*np.exp(1j*dot*delta),dot,Bs) for phase,dot,Bs in bonds],2)[2]
    H4fd=(shifted(eps)-2*shifted(0)+shifted(-eps))/(eps**2)
    return {'h4_abs_max_fd_error':float(np.max(abs(H4fd-H[4]))),
            'h4_abs_max':float(np.max(abs(H[4]))),
            'h4_relative_fd_error':float(np.max(abs(H4fd-H[4]))/max(np.max(abs(H[4])),1e-12)),
            'exact_peierls_first_three_vertices': q.validate_bonds()}

configs=[]
for L in (3,5,9,13,17):configs.append(dict(L=L,Delta=.8,N=1,lam=.1,steps=360,amplitude=.0003))
configs += [dict(L=9,Delta=.8,N=-1,lam=.1,steps=360,amplitude=.0003),
            dict(L=9,Delta=.4,N=1,lam=.1,steps=360,amplitude=.0003),
            dict(L=9,Delta=0,N=1,lam=.1,steps=360,amplitude=.0003),
            dict(L=9,Delta=.8,N=1,lam=0.,steps=360,amplitude=.0003),
            dict(L=9,Delta=.8,N=1,lam=.1,steps=720,amplitude=.0003)]
results=[]
for c in configs:
    tic=time.monotonic()
    res,t,cubic,fd,Jc,quad,qfd,Qt=propagate(**c)
    res['runtime_s']=round(time.monotonic()-tic,3)
    results.append(res)
    if c['L']==9 and c['Delta']==.8 and c['N']==1 and c['lam']==.1 and c['steps']==360:
        np.savez_compressed(ROOT/'qmath_16_26_primary_timeseries.npz',t=t,chi3_t=cubic,fd3_t=fd,
                            chi3_components=Jc,chi2_t=quad,fd2_t=qfd,chi2_components=Qt)
    print('L=',c['L'],'Delta=',c['Delta'],'N=',c['N'],'SOC=',c['lam'],
          'chi2=',res['transverse_second_harmonic'],'chi3=',res['sum_cubic_components'],
          '3rdFDerr=',res['fourier_fd_error_relative'],flush=True)
# Symmetry and independent-implementation checks.
base=results[2]; rev=results[5]; no_m=results[7]; no_soc=results[8];stepfine=results[9]
def z(d,key):return complex(*d[key])
checks={
    'opposite_neel_chi2_even_sum':abs(z(base,'transverse_second_harmonic')+z(rev,'transverse_second_harmonic')),
    'opposite_neel_chi3_difference':abs(z(base,'sum_cubic_components')-z(rev,'sum_cubic_components')),
    'zero_delta_chi2_abs':abs(z(no_m,'transverse_second_harmonic')),
    'zero_soc_chi2_abs':abs(z(no_soc,'transverse_second_harmonic')),
    'dt_refinement_chi2_relative':abs(z(stepfine,'transverse_second_harmonic')-z(base,'transverse_second_harmonic'))/abs(z(stepfine,'transverse_second_harmonic')),
    'dt_refinement_chi3_relative':abs(z(stepfine,'sum_cubic_components')-z(base,'sum_cubic_components'))/abs(z(stepfine,'sum_cubic_components')),
    'fourth_contact_term_nonzero':abs(z(base,'fourth_vertex_term'))>1e-6,
    'third_order_direct_vs_tangent_relative':base['fourier_fd_error_relative'],
    'second_order_direct_vs_tangent_relative':base['transverse_second_fd_rel']
}
out={'scope':'Classical prescribed finite pump vector potential, exact eight-band M0, full Peierls current. NOT a monochromatic causal chi3 tensor, NOT quantized BSV, NOT physical output photon count.',
     'peierls_derivative_check':vertex_check(),'checks':checks,'configurations':results,
     'continuation':'Q-MATH-16.27: full multitone causal current kernel + Ward tests in dressed input-output quantum optics.'}
(ROOT/'qmath_16_26_verified_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('checks',json.dumps(checks,indent=2))

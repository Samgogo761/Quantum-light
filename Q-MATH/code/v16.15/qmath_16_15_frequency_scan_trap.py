#!/usr/bin/env python3
"""Demonstrate misleading spectral rank across *different monochromatic experiments*.
All T amplitudes are bare M0, not a broadband fixed-mode S matrix.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
Q14=B/'sources/QMATH_16_14'
sys.path[0:0]=[str(Q14),str(Q14/'Q16_13_original/src_0'),str(Q14/'Q16_13_original')]
from qmath_16_14_math_and_m0 import c3_c51

def scan(L=9,gamma=.25):
    records=[]
    for omega in np.linspace(.14,.22,11):
        tot=np.zeros(3,complex)
        for i in range(L):
            for j in range(L):
                tot+=c3_c51(((i+.5)/L,(j+.5)/L),1,delta=.8,omega=float(omega),gamma=gamma)
        tot/=L**2
        records.append(dict(pump_omega=float(omega),output_omega=float(2*omega),c3=[float(tot[0].real),float(tot[0].imag)],c51=[float(tot[2].real),float(tot[2].imag)]))
        print('pump',round(float(omega),3),'|c3|',round(float(abs(tot[0])),4),'|c51|',round(float(abs(tot[2])),2),flush=True)
    a=np.array([complex(*r['c3']) for r in records]);b=np.array([complex(*r['c51']) for r in records]);
    overlap=abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))
    ss=np.linalg.svd(np.stack([a,b]),compute_uv=False)
    return dict(scope='rank of a collection of separate monochromatic experiments: NOT a fixed-broadband-pump spectral Gram',L=L,gamma=gamma,records=records,apparent_normalized_overlap=float(overlap),apparent_sin_angle=float(np.sqrt(max(0,1-overlap**2))),apparent_s2_over_s1=float(ss[1]/ss[0]),physical_fixed_mono_output_dim_per_experiment=1,physical_fixed_mono_Gram_det_per_experiment=0)
if __name__=='__main__':
    data=scan()
    (B/'qmath_16_15_frequency_scan_trap.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('apparent sin',data['apparent_sin_angle'],'apparent s2/s1',data['apparent_s2_over_s1'])

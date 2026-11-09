#!/usr/bin/env python3
"""Static-Ward-subtracted retarded kernel: stable BZ quadrature and downfolding test.
Only for insulating M0 cases with a nonzero direct gap; not HHG or BSV output.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from qmath_16_25_retarded_downfolding import terms

def delta_kernel(gaps,strengths,omega,eta=5e-4,keep=None):
    if keep is None: keep=len(gaps)
    if omega==0: return 0j
    z=complex(omega,eta)
    g=gaps[:keep];s=strengths[:keep]
    return np.sum(2*s/g * (z*z)/(z*z-g*g))

def grid(L,D,lam,omegas,eta=5e-4,keep=2):
    out={str(w):{'full':[],'keep2':[],'omitted':[]} for w in omegas}; gaps=[]
    for i in range(L):
      for j in range(L):
        _,g,s,gmin=terms(((i+.5)/L,(j+.5)/L),Delta=D,lam=lam);gaps.append(gmin)
        for om in omegas:
          full=delta_kernel(g,s,om,eta=eta)
          low=delta_kernel(g,s,om,eta=eta,keep=keep)
          out[str(om)]['full'].append(full)
          out[str(om)]['keep2'].append(low)
          out[str(om)]['omitted'].append(full-low)
    r={}
    for om,vals in out.items():
      r[om]={}
      for typ,arr in vals.items():
        x=np.mean(arr)
        r[om][typ]={'real':float(x.real),'imag':float(x.imag)}
    return {'L':L,'Delta':D,'lambda_SO':lam,'min_grid_gap':float(min(gaps)),'eta':eta,'response_delta_K':r}

def main():
    # Fast, reproducible set. Near-small-gap half-SOC and gapless SOC=0 controls are deliberately excluded from this insulating formula.
    cases=[('AFM_full_SOC',.8,.1,[41,61,81,101,121,151]),('weaker_exchange',.4,.1,[41,61,81]),('no_exchange_SOC',0.,.1,[41,61,81])]
    omegas=[0.,.005,.01,.02]
    out={'title':'Q-MATH-16.25 static-Ward-subtracted dynamic kernel','scope':'Insulating M0 parameter sets only. The SOC=0 control is gapless and excluded from this static-subtracted formula. No nonlinear HHG or quantized BSV prediction.',
      'formula':'Because the full BZ integral of K_k(0)=D_k-sum_j 2|V_j|^2/gap_j vanishes, the physical finite-frequency kernel can be evaluated as DeltaK(omega)=BZavg[K_k(omega)-K_k(0)]=BZavg sum_j (2|V_j|^2/gap_j)*(z^2/(z^2-gap_j^2)), z=omega+i eta. Keeping only two virtual transitions omits the corresponding Q-space dynamic memory term.',
      'cases':[]}
    for name,D,lam,grid_sizes in cases:
      rows=[]
      for L in grid_sizes:
        t=time.time();r=grid(L,D,lam,omegas,eta=5e-4,keep=2);r['seconds']=time.time()-t;rows.append(r)
        print('\n',name,'L',L,'gap',r['min_grid_gap'],'sec',r['seconds'])
        for om in [0.,.005,.01,.02]:
          d=r['response_delta_K'][str(om)]
          print('omega',om,'full',d['full'],'keep2',d['keep2'],'omitted',d['omitted'])
      out['cases'].append({'name':name,'Delta':D,'lambda_SO':lam,'grid_results':rows})
    p=ROOT/'qmath_16_25_static_subtracted_results.json';p.write_text(json.dumps(out,indent=2));print('\nsaved',p)
if __name__=='__main__':main()

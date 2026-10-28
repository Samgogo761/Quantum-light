#!/usr/bin/env python3
"""Same-state M0 material-curvature null against output-bin-dependent frequency-flat chi.
Retarded CLASSICAL chi2 proxy only. Not an emitted quantum-light observable.
"""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from qmath_16_29_same_state_readout import modes,overlap_grid,row_project,state,compress,decompress,OUTPUTS
from qmath_16_27_retarded_kernel import kernel
BINS=[.32,.34,.36,.38,.40]
cachepath=ROOT/'qmath_16_29_kernel_cache.json'
cache=json.loads(cachepath.read_text())
Wgrid,_,shapes,_,_=modes()
m,U,W,Wh=state()
def compute(L,eta,D):
    K=[];F=[];rows=[]
    for om in BINS:
        w1,w2,wt=overlap_grid(om,.01)
        chi=[]
        for a,b in zip(w1,w2):
            key=f'{L}_{eta:.5f}_{D:.4f}_{min(a,b):.4f}_{max(a,b):.4f}'
            if key not in cache:
                cache[key]=kernel(L=L,Delta=D,eta=eta,w1=float(a),w2=float(b))['chi12']
            chi.append(decompress(cache[key]))
        chi=np.array(chi)
        c=chi[np.argmin(abs(w1-om/2))]
        km=row_project(om,chi,w1,w2,wt,Wgrid,shapes)
        kf=row_project(om,np.ones_like(chi)*c,w1,w2,wt,Wgrid,shapes)
        K.append(km);F.append(kf)
        rows.append({'Omega':om,'material_row':[compress(z) for z in km],
                     'frequency_flat_control_row':[compress(z) for z in kf]})
        print('BINS',L,eta,D,om,flush=True)
    cachepath.write_text(json.dumps(cache),encoding='utf-8')
    K=np.array(K);F=np.array(F)
    U0,s0,_=np.linalg.svd(F,full_matrices=True)
    Pperp=U0[:,3:]@U0[:,3:].conj().T
    y=K@m;yf=F@m
    excess=Pperp@y
    # Material-only output combination (flat killed regardless of M, due to P F=0)
    return {'L':L,'eta':eta,'Delta':D,'rows':rows,'rank_material':int(np.linalg.matrix_rank(K)),
            'rank_flat_control':int(np.linalg.matrix_rank(F)),
            'null_control_error':float(np.linalg.norm(Pperp@F)/np.linalg.norm(F)),
            'projected_signal_abs':float(np.linalg.norm(excess)),
            'projected_signal_over_total':float(np.linalg.norm(excess)/np.linalg.norm(y)),
            'direct_material_minus_control_frac':float(np.linalg.norm(K-F)/np.linalg.norm(K)),
            'flat_singular_values':s0.tolist(),'material_singular_values':np.linalg.svd(K,compute_uv=False).tolist(),
            'projected_output':[compress(z) for z in excess],
            'BSV_output':[compress(z) for z in y], 'flat_BSV_output':[compress(z) for z in yf],
            'projected_rank':int(np.linalg.matrix_rank(Pperp@K,tol=1e-8)),
            'normalized_projected_material_K':float(np.linalg.norm(Pperp@K)/np.linalg.norm(K))}

if __name__=='__main__':
    d=[]
    for L,eta,D in [(101,.08,.8),(201,.08,.8),(101,.08,.4),(101,.03,.8),(201,.03,.8)]:
        t=time.time();r=compute(L,eta,D);d.append(r)
        print('DONE',L,eta,D,'projected=',r['projected_signal_over_total'],'time=',round(time.time()-t,2),flush=True)
    (ROOT/'qmath_16_29_material_null_results.json').write_text(json.dumps({
        'identity':'Five-output SAME-STATE material-only null projection, 3 input symmetric moment dims, flat conditional frequency-response subspace removed. CLASSICAL chi2 PROXY, not quantum output.',
        'output_Omega':BINS,'cases':d},indent=2,ensure_ascii=False))

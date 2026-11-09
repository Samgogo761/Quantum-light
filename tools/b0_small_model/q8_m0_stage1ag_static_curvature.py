#!/usr/bin/env python3
"""Static gauge-curvature diagnostic for the frozen insulating M0 model.

No propagation, counterterm, initial-state change, or BZ production run.
Compare D - 2 Re sum(X_a* X_b / gap) with the curvature of the occupied
energy, then inspect ordinary static BZ quadrature.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from q8_m0_stage1ag_acceptance_audit import (
    REPO, KF, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases, parallel_perp_basis)

def evaluate(geo,kf,fd=False):
    kc=geo.k_cart(kf)
    vv,vc,en=occup_empty_bases(h8(geo,R0,kc,1),MU)
    x=x_from_bases(current_ops(geo,R0,kc,1),vv,vc)
    gap=en[:,4:,None]-en[:,None,:4]
    d=np.mean(mean_d_from_bases(d_ops(geo,R0,kc,1),vv),axis=0)
    chi=2*np.einsum('kacv,kbcv,kcv->ab',x.conj(),x,1/gap).real/len(kc)
    m=d-chi
    ep,et=parallel_perp_basis(geo)
    result={'n_k':len(kc),'D':d.tolist(),'static_paramagnetic':chi.tolist(),
            'occupied_energy_curvature':m.tolist(),'curvature_eigenvalues':np.linalg.eigvalsh(m).tolist(),
            'parallel_curvature':float(ep@m@ep),'perp_curvature':float(et@m@et)}
    if fd:
        eps=1e-4
        def energy(shift):
            return float(np.mean(np.sum(np.linalg.eigvalsh(h8(geo,R0,kc+shift,1))[:,:4],axis=1)))
        z=np.zeros(2);hess=np.zeros((2,2));e0=energy(z)
        for a in range(2):
            da=np.eye(2)[a]*eps
            hess[a,a]=(energy(da)-2*e0+energy(-da))/eps**2
        dx,dy=np.eye(2)*eps
        hess[0,1]=hess[1,0]=(energy(dx+dy)-energy(dx-dy)-energy(-dx+dy)+energy(-dx-dy))/(4*eps**2)
        result['finite_difference_curvature']=hess.tolist()
        result['finite_difference_maxabs_error']=float(np.max(abs(m-hess)))
    return result

GRID_SIZES = (12, 24, 48, 96, 128, 192)


def main():
    geo=build_geometry()
    rows=[{'grid':'two_k_probe',**evaluate(geo,KF,True)}]
    for nk in GRID_SIZES:
        f=np.stack(np.meshgrid(np.arange(nk)/nk,np.arange(nk)/nk,indexing='ij'),axis=-1).reshape(-1,2)
        # The two curvature paths use the same frozen Hamiltonian but
        # independent derivatives: interband Kubo versus shifted occupied energy.
        row={'grid':f'Gamma_{nk}x{nk}',**evaluate(geo,f,True)}
        rows.append(row)
        print(row['grid'],row['curvature_eigenvalues'],row['perp_curvature'],flush=True)
    out=REPO/'results/q_line/stage1ah_static_bz_qualification_20260924/static_curvature.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    payload={
        'status':'static_diagnostic_no_model_changes',
        'qualification':'P1=0 uniform BZ gauge-curvature audit only',
        'grid_sizes':list(GRID_SIZES),
        'finite_difference_eps':1e-4,
        'finite_difference_on_all_uniform_grids':True,
        'rows':rows,
    }
    out.write_text(json.dumps(payload,indent=2),encoding='utf-8')

if __name__=='__main__':main()

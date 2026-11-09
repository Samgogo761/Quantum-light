#!/usr/bin/env python3
"""Read-only audit: autonomous zero-drive reference for the existing Gaussian model.

No physical input is changed. At A_cl=0 the existing interaction-picture
vertices have exactly known free phases. Undoing that rotation gives a
constant generator, so one matrix exponential is an independent reference.
This is a finite-quadrature, two-k diagnostic, not a continuum qualification.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from scipy.linalg import expm, eigvals, eigvalsh
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_classical import p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_model import R0, build_geometry, h8, current_ops
from q8_m0_response import d_ops, mean_d_from_bases
from q8_m0_mode_io import eta2_ncell
from q8_m0_phgauss import linear_K, sqrt_w, symplectic_J, pauli_from_phi
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_stage1s_continuum import open_grid
from q8_m0_stage1t_continuum_pauli import gauss_segment

REPO = Path(__file__).resolve().parents[2]
OUT = REPO/'results/q_line/stage1ag_acceptance_audit_20260924'
KF = np.array([[0.,0.],[.17,.41]])
MU = -0.009010620372198276

def rotation(freq, t):
    c, s = np.diag(np.cos(freq*t)), np.diag(np.sin(freq*t))
    return np.block([[c,s],[-s,c]])

def run(cutoff, nlow, nuv, pol):
    start=time.perf_counter()
    geo=build_geometry(); prot=p1_a2_protocol(0.)
    kc=geo.k_cart(KF); vv,vc,en=occup_empty_bases(h8(geo,R0,kc,1),MU)
    x0=x_from_bases(current_ops(geo,R0,kc,1),vv,vc)
    gap=en[:,4:,None]-en[:,None,:4]
    w=np.full(2,.5); sw=sqrt_w(w)
    d=np.einsum('k,kab->ab',w,mean_d_from_bases(d_ops(geo,R0,kc,1),vv))
    ep,et=parallel_perp_basis(geo); pols=(et,) if pol=='perp' else (ep,et)
    om,wt=open_grid(8*prot.omega0,nlow)
    if cutoff>8:
        ou,wu=gauss_segment(8*prot.omega0,cutoff*prot.omega0,nuv)
        om,wt=np.r_[om,ou],np.r_[wt,wu]
    of=np.tile(np.repeat(om,2),len(pols))
    f0=np.concatenate([np.repeat(np.sqrt(wt/om)[:,None]*e[None,:],2,axis=0) for e in pols])
    nph=len(of); nb=gap.size; freq=np.r_[of,gap.ravel()]
    n=len(freq); j=symplectic_J(n); eta=float(np.sqrt(eta2_ncell(geo)))
    k0=linear_K(nph,sw,x0,f0,np.zeros_like(f0),d,eta)
    free=j@np.diag(np.r_[freq,freq]); ks=free+k0
    residual=[]
    for t in (0.37,17.3,0.73*prot.t_pulse):
        rt=rotation(freq,t)
        xt=x0*np.exp(1j*gap[:,None,:,:]*t)
        ft=f0*np.exp(1j*of[:,None]*t)
        kt=linear_K(nph,sw,xt,ft.real,ft.imag,d,eta)
        residual.append(float(np.max(abs(kt-rt.T@k0@rt))))
    hs=-j@ks
    eig=eigvals(ks)
    grow=float(np.max(eig.real))
    phi=rotation(freq,-prot.t_pulse)@expm(prot.t_pulse*ks)
    p=pauli_from_phi(phi,nph,nb,w)
    sym=float(np.max(abs(phi.T@j@phi-j)))
    result={'a_star':0.,'polarization':pol,'cutoff_over_omega0':cutoff,
            'n_low':nlow,'n_uv':nuv,'n_frequency':len(om),'dimension':2*n,
            'rotation_identity_maxabs':max(residual),
            'autonomous_generator_max_real_eigenvalue':grow,
            'autonomous_hessian_min_eigenvalue':float(eigvalsh((hs+hs.T)/2,subset_by_index=[0,0])[0]),
            'C_P_final':p['Ncell_times_eps_Pauli'],
            'C_C_final':p['Ncell_times_C_minus_I'],
            'Phi_maxabs':float(np.max(abs(phi))),
            'symplectic_abs':sym,
            'wall_seconds':time.perf_counter()-start,
            'qualification':'fixed finite photon quadrature and polarization only'}
    # Saving Phi permits direct, same-representation future comparisons.
    tag=f'a0_uv{cutoff:g}_low{nlow}_uvnodes{nuv}_{pol}'
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f'{tag}.npz',Phi=phi,freq=freq)
    (OUT/f'{tag}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cutoff',type=float,default=8.)
    ap.add_argument('--nlow',type=int,default=96);ap.add_argument('--nuv',type=int,default=0)
    ap.add_argument('--pol',choices=['perp','both'],default='perp')
    a=ap.parse_args();run(a.cutoff,a.nlow,a.nuv,a.pol)

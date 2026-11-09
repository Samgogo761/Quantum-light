#!/usr/bin/env python3
"""Reconstruct acceptance evidence from existing files; no joint propagation."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from scipy.special import sici
from scipy.signal import fftconvolve
sys.path.insert(0,str(Path(__file__).resolve().parent))
from q8_m0_stage1ae_source_qual import load_xref,strongest_channel,channel_current,uv_grid
from q8_m0_stage1s_continuum import photon_retarded_kernel,open_grid,compare
from q8_m0_model import build_geometry
from q8_m0_s1_kernel import parallel_perp_basis
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'results/q_line/stage1ag_acceptance_audit_20260924/evidence.json'

def conv(cur,k,dt):
    return dt*np.stack([fftconvolve(cur[:,a],k)[:len(cur)] for a in range(2)],axis=1)

def main():
    ref=load_xref(.8,2048);_,eperp=parallel_perp_basis(build_geometry())
    ch=strongest_channel(ref['X_full']);cur=channel_current(ref['X_full'],ch,eperp)
    dt=ref['dt'];w0=ref['omega0'];lag=np.arange(len(cur))*dt
    out={'source_channel':ch,'exact_cutoff_checks':[]}
    for cutoff,nlow in ((8,96),(64,96),(64,192),(64,384)):
        if cutoff==8:om,wt=open_grid(8*w0,nlow)
        else:om,wt=uv_grid(w0,cutoff,nlow,384)
        k=np.concatenate([photon_retarded_kernel(chunk,om,wt,1.) for chunk in np.array_split(lag,32)])
        kexact=2/np.pi*sici(cutoff*w0*lag)[0]
        aq=conv(cur,k,dt);ae=conv(cur,kexact,dt)
        target=conv(cur,np.r_[0.,np.ones(len(cur)-1)],dt)
        out['exact_cutoff_checks'].append({'cutoff':cutoff,'n_low':nlow,'n_frequency':len(om),
            'kernel_maxabs_error':float(np.max(abs(k-kexact))),
            'field_rel_max_vs_exact_cutoff':float(np.max(abs(aq-ae))/np.max(abs(ae))),
            'exact_cutoff_vs_Maxwell':compare(ae,target,cur,dt,1.)})
    # A retarded kernel has a removable low-frequency limit, while F vacuum
    # variance uses weight/omega. This records the different convergence axes.
    out['low_frequency_moments']=[]
    for n in (48,96,192,384):
        om,wt=open_grid(8*w0,n)
        out['low_frequency_moments'].append({'n':n,'omega_min':float(om[0]),
                                            'sum_weight_over_omega':float(np.sum(wt/om))})
    p=REPO/'results/q_line/stage1af_joint_time_20260923/stage1af_a0_n96_uv8_perp_ntpc1024_sub8.json'
    a,b=json.loads(p.read_text())['runs']
    aa,bb=np.array(a['C_P_t']),np.array(b['C_P_t'])
    out['zero_drive_1024_method_comparison']={
        'CP_relative_difference_to_CF4':float(abs(aa[-1]-bb[-1])/bb[-1]),
        'max_trajectory_relative_difference_to_CF4_excluding_initial':float(np.max(abs(aa[1:]-bb[1:])/bb[1:])),
        'Phi_relative_difference_reported':a['Phi_rel_fro_diff_between_methods']}
    root=REPO/'results/q_line/stage1ad_xref_20260923'
    out['D_reference_convergence']=[]
    for a,b in ((256,512),(512,1024),(1024,2048)):
        ca=np.load(root/f'xref_a0.8_ntpc{a}.npz')['mean_D_t']
        fb=np.load(root/f'xref_a0.8_ntpc{b}.npz')['mean_D_t'][::2]
        out['D_reference_convergence'].append({'pair':[a,b],'maxrel':float(np.max(abs(ca-fb))/np.max(abs(fb)))})
    # Check the analytic zero-drive phase against the actual stage cache.
    from q8_m0_stage1ag_acceptance_audit import KF,MU,R0,h8,current_ops,occup_empty_bases,x_from_bases
    geo=build_geometry();kc=geo.k_cart(KF)
    vv,vc,en=occup_empty_bases(h8(geo,R0,kc,1),MU)
    x0=x_from_bases(current_ops(geo,R0,kc,1),vv,vc)
    gap=en[:,4:,None]-en[:,None,:4]
    z=np.load(REPO/'results/q_line/stage1af_electron_stage_cache/estage_a0_ntpc128_sub8.npz')
    ts=z['times_stage'];exact=x0[None,None]*np.exp(1j*gap[None,None,:,None]*ts[:,:,None,None,None,None])
    out['zero_drive_analytic_phase_vs_stage_cache_maxrel']=float(np.max(abs(exact-z['X_stage']))/np.max(abs(exact)))
    # Direct replay of the small CF4 order check without overwriting the source result.
    from q8_m0_stage1ac_mms import make_generator,reference,propagate
    gen,j,_=make_generator(4);tf=2*np.pi;r=reference(gen,8,tf)
    out['mms_replay']={}
    for method in ('cayley_midpoint','cf4'):
        hs=[];es=[]
        for m in (7,8,9,10):
            h=tf/2**m;p=propagate(gen,8,h,tf,method)
            hs.append(h);es.append(float(np.linalg.norm(p-r)/np.linalg.norm(r)))
        out['mms_replay'][method]={'errors':es,'order':float(np.polyfit(np.log(hs),np.log(es),1)[0])}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()

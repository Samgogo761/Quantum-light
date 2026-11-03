"""One explicit driven or no-field run; no automatic production scans.

Right emission candidate = P(x>detector,t)-P(x>detector,t0)+right CAP loss.
An independent accumulation of the signed FD current at that face must agree.
This finite-time observable needs detector/time/CAP/grid convergence before use.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
import numpy as np
import scipy
from ql_tip_1d.core import (A0_NM,EH_EV,TAU_FS,well_grid,surface_potential,
                          bound_states,CrankNicolson,norm,pulse,integrated_near_field)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path,obj):
    s=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,
                 default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    with path.open('x',encoding='utf-8') as f:
        f.write(s+'\n')


def csv_write(path,names,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(names);w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=BASE/'configs/stage2_baseline.json')
    ap.add_argument('--run-id',required=True)
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['bias_V_nm']!=0:
        raise ValueError('Only explicitly agreed zero-bias model implemented')
    raw=BASE/'data/raw'/args.run_id
    origin=BASE/'data/origin'/args.run_id
    results=BASE/'results'/args.run_id
    for p in [raw,origin,results]:
        p.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    g=well_grid(cfg['well_cells'],cfg['xmin_nm'],cfg['xmax_nm'],cfg['width_nm'])
    v=surface_potential(g.x_au,cfg['width_nm'],cfg['depth_eV'])
    en,states=bound_states(g,v,cfg['depth_eV'])
    if len(en)!=1:
        raise RuntimeError('Exactly one initial bound state required')
    initial=states[:,0].astype(complex)
    psi=initial.copy()
    x=g.x_nm
    cap_left=cfg['cap_strength_au']*(np.maximum(cfg['left_cap_start_nm']-x,0)/
                                  cfg.get('left_cap_width_nm',cfg['left_cap_start_nm']-cfg['xmin_nm']))**cfg['cap_power']
    cap_right=cfg['cap_strength_au']*(np.maximum(x-cfg['right_cap_start_nm'],0)/
                                   cfg.get('right_cap_width_nm',cfg['xmax_nm']-cfg['right_cap_start_nm']))**cfg['cap_power']
    cap=cap_left+cap_right
    static=v-1j*cap
    spatial=integrated_near_field(g.x_au,cfg['beta'],cfg['ell_nm'])
    idx=int(np.searchsorted(x,cfg['detector_nm'],side='right'))
    if not (0<idx<len(x)) or np.any(cap_left[idx:]) or np.any(cap_right[:idx]):
        raise ValueError('Detector must separate left/right CAPs')
    pr0=float(np.sum(abs(initial[idx:])**2)*g.dx_au)
    t0=cfg['start_fs']/TAU_FS
    duration=(cfg['end_fs']-cfg['start_fs'])/TAU_FS
    nsteps=int(np.ceil(duration/cfg['dt_au']))
    dt=duration/nsteps
    readout=max(1,round(cfg['readout_interval_fs']/(dt*TAU_FS)))
    cn=CrankNicolson(g,dt,static)
    source=[Path(__file__),BASE/'src/ql_tip_1d/core.py',BASE/'src/ql_tip_1d/__init__.py']
    metadata={'run_id':args.run_id,'time_iso8601':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'config':cfg,'config_sha256':sha(args.config),
              'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
              'source_sha256':{str(p.relative_to(BASE)).replace('\\','/'):sha(p) for p in source},
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'platform':platform.platform(),'device':'CPU','dtype':'complex128',
              'nx':len(x),'dx_nm':g.dx_au*A0_NM,'actual_dt_au':dt,'steps':nsteps,
              'energy_eV':float(en[0]*EH_EV),
              'actual_detector_face_nm':float((x[idx-1]+x[idx])/2),
              'initial_right_tail':pr0,'initial_x_gt_zero_probability':float(np.sum(abs(initial[x>0])**2)*g.dx_au),
              'renormalization_during_or_after_propagation':False,
              'observable':'signed_net_right_flux = P_right(t)-P_right(t0)+right_CAP_loss',
              'CAP_loss_rule':'2 dt dx sum W |(psi_next+psi_old)/2|^2',
              'current_rule':'Im(conj(psi_mid[j-1])*psi_mid[j])/dx at detector face',
              'no_claims':['No asymptotic spectrum','No photon statistics','No absolute electron count']}
    write_json(results/'metadata.json',metadata)
    rows=[]
    loss_l=loss_r=flux=0.
    max_budget=max_flow_budget=0.
    propagation_start=time.perf_counter()
    for step in range(nsteps+1):
        if step%readout==0 or step==nsteps:
            t=t0+step*dt
            nrm=norm(psi,g.dx_au)
            pr=float(np.sum(abs(psi[idx:])**2)*g.dx_au)
            yr=pr-pr0+loss_r
            budget=nrm+loss_l+loss_r-1
            flowerr=yr-flux
            max_budget=max(max_budget,abs(budget))
            max_flow_budget=max(max_flow_budget,abs(flowerr))
            survival=abs(np.vdot(initial,psi)*g.dx_au)**2
            rows.append([t*TAU_FS,nrm,pr,loss_r,loss_l,yr,flux,budget,flowerr,float(survival)])
        if step<nsteps:
            tmid=t0+(step+.5)*dt
            e=pulse(tmid,cfg['intensity_W_cm2'],cfg['wavelength_nm'],cfg['tau_intensity_fs'],cfg['cep_rad'])
            after=cn.step(psi,static+e*spatial)
            mid=(psi+after)*.5
            rho=abs(mid)**2
            loss_l += float(2*dt*g.dx_au*np.dot(cap_left,rho))
            loss_r += float(2*dt*g.dx_au*np.dot(cap_right,rho))
            flux += float(dt*np.imag(np.conj(mid[idx-1])*mid[idx])/g.dx_au)
            psi=after
    propagation_seconds=time.perf_counter()-propagation_start
    csv_write(origin/'probability_history.csv',
              ['time_fs','norm','P_right_retained','right_CAP_loss','left_CAP_loss','net_right_yield',
               'signed_right_flux','total_budget_error','right_budget_error','bound_survival'],rows)
    # Split checkpoint arrays rather than writing a multi-MiB single compressed file.
    for part,sl in enumerate(np.array_split(np.arange(len(x)),max(1,int(np.ceil(len(x)/12000))))):
        np.savez_compressed(raw/f'wavefunction_part{part:02d}.npz',
                            x_au=g.x_au[sl],psi_initial=initial[sl],psi_final=psi[sl],
                            surface_au=v[sl],left_CAP_au=cap_left[sl],right_CAP_au=cap_right[sl],
                            index_start=sl[0],index_stop=sl[-1]+1)
    # Full density remains available as split CSV, with no lossy decimation.
    for part,sl in enumerate(np.array_split(np.arange(len(x)),max(1,int(np.ceil(len(x)/12000))))):
        csv_write(origin/f'final_density_part{part:02d}.csv',
                  ['x_nm','psi_initial_abs2_per_nm','psi_final_abs2_per_nm'],
                  zip(x[sl],abs(initial[sl])**2/A0_NM,abs(psi[sl])**2/A0_NM))
    plateau=[r[5] for r in rows if r[0]>=cfg['plateau_start_fs']-1e-10]
    plateau_abs=float(max(plateau)-min(plateau))
    yield_r=rows[-1][5]
    floor=cfg['probability_absolute_tolerance']
    plateau_rel=plateau_abs/max(abs(yield_r),floor)
    summary={'run_id':args.run_id,'energy_eV':metadata['energy_eV'],'nx':len(x),
             'dx_nm':g.dx_au*A0_NM,'actual_dt_au':dt,'steps':nsteps,
             'initial_right_tail':pr0,'yield_right':yield_r,'right_flux':flux,
             'final_right_retained':rows[-1][2],'right_CAP_loss':loss_r,'left_CAP_loss':loss_l,
             'final_norm':rows[-1][1],'bound_survival':rows[-1][9],
             'max_total_budget_error':max_budget,'max_right_budget_error':max_flow_budget,
             'plateau_interval_fs':[cfg['plateau_start_fs'],cfg['end_fs']],
             'plateau_absolute_range':plateau_abs,'plateau_relative_range':plateau_rel,
             'numeric_sanity_pass': bool(-floor<=yield_r<=1+floor and max_budget<1e-8 and max_flow_budget<1e-8),
             'plateau_pass':bool(plateau_abs<floor or plateau_rel<cfg['relative_convergence_tolerance']),
             'gate_b_pass':None,'gate_b_note':'Requires comparison with dx, dt, box, CAP and no-field controls.',
             'propagation_seconds':propagation_seconds,'elapsed_seconds':time.perf_counter()-start,
             'algorithm':'second-order centered FD + midpoint CN; cubic left/right CAP',
             'no_renormalization':True}
    write_json(results/'summary.json',summary)
    artifacts=[]
    for directory in [raw,origin,results]:
        for p in sorted(directory.iterdir()):
            artifacts.append([args.run_id,metadata['time_iso8601'],metadata['config_sha256'],metadata['base_git_commit'],
                              str(p.relative_to(BASE)).replace('\\','/'),sha(p),p.stat().st_size,
                              'single_run_completed_not_gate_b'])
    csv_write(results/'manifest.csv',['run_id','time_iso8601','config_sha256','base_git_commit','path','sha256','bytes','status'],artifacts)
    print(json.dumps(summary,indent=2))
    return 0 if summary['numeric_sanity_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())

"""Run only Stage 1. Outputs are immutable per run ID; no scanning of light fields."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
import numpy as np
import scipy
from scipy.integrate import quad
from ql_tip_1d.core import (A0_NM, EH_EV, TAU_FS, Grid, analytic_well,
    analytic_wavefunction, bound_states, well_grid, surface_potential,
    CrankNicolson, norm, fft_free_step, enhancement)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, names, rows):
    with path.open('x', encoding='utf-8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(names)
        writer.writerows(rows)


def json_write(path, obj):
    with path.open('x', encoding='utf-8') as out:
        json.dump(obj, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=BASE/'configs/stage1_baseline.json')
    ap.add_argument('--run-id', default='stage1_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    runid = args.run_id
    raw = BASE/'data/raw'/runid
    origin = BASE/'data/origin'/runid
    results = BASE/'results'/runid
    for dest in [raw, origin, results]:
        dest.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    prefix = {'run_id': runid, 'started_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'config': cfg, 'config_sha256': sha(args.config),
              'base_git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=BASE, text=True).strip(),
              'source_sha256': {str(p.relative_to(BASE)):sha(p) for p in
                   [Path(__file__), BASE/'src/ql_tip_1d/core.py', BASE/'src/ql_tip_1d/__init__.py']},
              'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'platform': platform.platform(), 'device': 'CPU',
              'dtype': 'float64 eigenvectors; complex128 propagation',
              'renormalization_after_initialization': False,
              'units': {'x':'nm in CSV; bohr in NPZ x_au', 'energy':'eV',
                        'time':'fs and atomic units', 'density':'1/nm in CSV'},
              'constants': {'bohr_nm': A0_NM, 'hartree_eV': EH_EV, 'time_au_fs': TAU_FS}}
    json_write(results/'metadata.json', prefix)
    a, v = cfg['width_nm'], cfg['depth_eV']
    analytic = analytic_well(a, v)
    e_exact = analytic['energy_au'] * EH_EV
    # Real-space adaptive integral is independent of finite-difference diagonalization.
    anorm, qerr = quad(lambda x: float(analytic_wavefunction(x, a, v)**2),
                       -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11, limit=300)
    analytic.update({'energy_eV':e_exact, 'normalization_integral_quad':anorm,
                     'normalization_quadrature_error':qerr})
    convergence = []
    for cells in cfg['well_cells_scan']:
        g = well_grid(cells, cfg['xmin_nm'], cfg['xmax_nm'], a)
        pot = surface_potential(g.x_au, a, v)
        energies, states = bound_states(g, pot, v)
        psi = states[:, 0]
        ana = analytic_wavefunction(g.x_au, a, v)
        overlap = abs(np.vdot(ana, psi)*g.dx_au)**2/(norm(ana,g.dx_au)*norm(psi,g.dx_au))
        convergence.append([cells,len(psi),g.dx_au*A0_NM,float(energies[0]*EH_EV),
                            float(energies[0]*EH_EV-e_exact),len(energies),norm(psi,g.dx_au),float(overlap)])
        np.savez_compressed(raw/f'bound_cells{cells}.npz',x_au=g.x_au,psi0=psi,
                            potential_au=pot,energies_au=energies,analytic_psi=ana)
    csv_write(origin/'bound_convergence.csv',
              ['well_cells','nx','dx_nm','energy_eV','energy_error_eV','n_bound','norm','analytic_fidelity'],convergence)
    # Intentionally naive endpoint representations expose geometry bias; not used for production.
    endpoint_rows=[]
    for cells in [12,24,48,96]:
        dx=a/cells
        n=int(np.ceil(2/dx))
        indices=np.arange(-n,n+1)
        g=Grid(indices*dx/A0_NM,dx/A0_NM,cells)
        for convention in ['inclusive_nodes','half_weight_interfaces']:
            pot=np.zeros(len(indices))
            pot[(indices>=-cells)&(indices<=0)]=-v/EH_EV
            if convention=='half_weight_interfaces':
                pot[(indices==-cells)|(indices==0)] *= .5
            en,_=bound_states(g,pot,v)
            endpoint_rows.append([cells,dx,convention,float(-np.sum(pot)*EH_EV*dx),float(en[0]*EH_EV),float(en[0]*EH_EV-e_exact)])
    csv_write(origin/'endpoint_audit.csv',['well_cells','dx_nm','representation','integrated_depth_eV_nm','energy_eV','energy_error_eV'],endpoint_rows)
    g=well_grid(cfg['no_field_well_cells'],cfg['xmin_nm'],cfg['xmax_nm'],a)
    pot=surface_potential(g.x_au,a,v)
    energy,states=bound_states(g,pot,v)
    psi0=states[:,0].astype(complex)
    rho0=abs(psi0)**2
    right=g.x_nm>cfg['detector_nm']
    p_right0=float(np.sum(rho0[right])*g.dx_au)
    nofield=[]
    for dt_requested in cfg['no_field_dt_au']:
        duration_au=cfg['no_field_duration_fs']/TAU_FS
        nsteps=int(np.ceil(duration_au/dt_requested))
        dt=duration_au/nsteps
        cn=CrankNicolson(g,dt,pot)
        psi=psi0.copy()
        timeline=[]
        t0=time.perf_counter()
        sample=max(1,nsteps//100)
        max_norm=max_density=max_right=0.
        for step in range(nsteps+1):
            if step%sample==0 or step==nsteps:
                t=step*dt
                density=abs(psi)**2
                nrm=norm(psi,g.dx_au)
                dens=float(np.sum(abs(density-rho0))*g.dx_au)
                pr=float(np.sum(density[right])*g.dx_au)
                overlap=np.vdot(psi0,psi)*g.dx_au
                phaseerr=float(np.angle(overlap*np.exp(1j*energy[0]*t)))
                max_norm=max(max_norm,abs(nrm-1))
                max_density=max(max_density,dens)
                max_right=max(max_right,abs(pr-p_right0))
                timeline.append([t*TAU_FS,t,nrm,dens,pr,pr-p_right0,phaseerr])
            if step<nsteps:
                psi=cn.step(psi)
        elapsed=time.perf_counter()-t0
        expected_phase=(energy[0]*dt-2*np.arctan(energy[0]*dt/2))*nsteps
        observed=timeline[-1][-1]
        nofield.append({'requested_dt_au':dt_requested,'actual_dt_au':dt,'steps':nsteps,
                        'duration_fs':cfg['no_field_duration_fs'],'norm_drift_max':max_norm,
                        'density_L1_drift_max':max_density,'right_probability_initial':p_right0,
                        'right_probability_change_max':max_right,'phase_error_rad':observed,
                        'cn_predicted_phase_error_rad':float(expected_phase),
                        'phase_prediction_residual_rad':float(observed-expected_phase),
                        'elapsed_seconds':elapsed})
        tag=f'nofield_dt{dt_requested:g}'
        csv_write(origin/(tag+'.csv'),['time_fs','time_au','norm','density_L1_drift','P_x_gt_detector','delta_P_x_gt_detector','phase_error_rad'],timeline)
        np.savez_compressed(raw/(tag+'.npz'),x_au=g.x_au,psi0=psi0,psi_final=psi,
                            energy_au=energy[0],actual_dt_au=dt,steps=nsteps)
    csv_write(origin/'Fig1_potential.csv',['x_nm','V_surface_eV','psi0_abs2_per_nm','E_enhancement','analytic_psi_abs2_per_nm'],
              zip(g.x_nm,pot*EH_EV,rho0/A0_NM,enhancement(g.x_au),abs(analytic_wavefunction(g.x_au,a,v))**2/A0_NM))
    fp=cfg['free_packet']
    gaussian=[]
    for method in ['fft','cn']:
        if method=='fft':
            x=np.linspace(fp['xmin_nm'],fp['xmax_nm'],fp['nx_fft'],endpoint=False)/A0_NM
            fg=Grid(x,x[1]-x[0],0)
        else:
            fg=well_grid(48,fp['xmin_nm'],fp['xmax_nm'],a)
            x=fg.x_au
        x0,sigma,p0=fp['x0_nm']/A0_NM,fp['density_sigma_nm']/A0_NM,fp['momentum_au']
        psi=(2*np.pi*sigma*sigma)**(-.25)*np.exp(-(x-x0)**2/(4*sigma*sigma)+1j*p0*x)
        initial=psi.copy()
        duration=fp['duration_fs']/TAU_FS
        nsteps=int(np.ceil(duration/fp['dt_au']))
        dt=duration/nsteps
        if method=='cn':
            cn=CrankNicolson(fg,dt,np.zeros(len(x)))
        k=2*np.pi*np.fft.fftfreq(len(x),d=fg.dx_au)
        phi=np.fft.fft(psi,norm='ortho')
        mean_p=float(np.sum(k*abs(phi)**2)*fg.dx_au)
        timeline=[]
        t0=time.perf_counter()
        sample=max(1,nsteps//100)
        for step in range(nsteps+1):
            if step%sample==0 or step==nsteps:
                t=step*dt
                density=abs(psi)**2
                nrm=norm(psi,fg.dx_au)
                mean=float(np.sum(x*density)*fg.dx_au/nrm)
                sd=float(np.sqrt(np.sum((x-mean)**2*density)*fg.dx_au/nrm))
                exact_mean=x0+p0*t
                exact_sd=sigma*np.sqrt(1+(t/(2*sigma*sigma))**2)
                timeline.append([t*TAU_FS,nrm,mean*A0_NM,exact_mean*A0_NM,(mean-exact_mean)*A0_NM,
                                 sd*A0_NM,exact_sd*A0_NM,(sd-exact_sd)*A0_NM])
            if step<nsteps:
                psi=fft_free_step(psi,fg.dx_au,dt) if method=='fft' else cn.step(psi)
        gaussian.append({'method':method,'nx':len(x),'dx_nm':fg.dx_au*A0_NM,'dt_au':dt,
                         'steps':nsteps,'final_centroid_error_nm':timeline[-1][4],
                         'final_sigma_error_nm':timeline[-1][7],
                         'norm_drift_max':max(abs(row[1]-1) for row in timeline),
                         'initial_momentum_au':mean_p,'momentum_error_au':mean_p-p0,
                         'elapsed_seconds':time.perf_counter()-t0})
        csv_write(origin/f'free_gaussian_{method}.csv',['time_fs','norm','centroid_nm','analytic_centroid_nm','centroid_error_nm','density_sigma_nm','analytic_sigma_nm','sigma_error_nm'],timeline)
        np.savez_compressed(raw/f'free_gaussian_{method}.npz',x_au=x,psi_initial=initial,psi_final=psi,k_au=k)
    tol=cfg['gate_a_tolerances']
    checks={
        'one_bound_state_all_grids':all(row[5]==1 for row in convergence) and analytic['bound_state_count']==1,
        'analytic_normalization':abs(anorm-1)<1e-9,
        'finest_energy':abs(convergence[-1][4])<tol['finest_energy_absolute_error_eV'],
        'no_field_norm':max(r['norm_drift_max'] for r in nofield)<tol['maximum_norm_drift'],
        'no_field_density':max(r['density_L1_drift_max'] for r in nofield)<tol['maximum_density_L1_drift'],
        'no_field_right_probability':max(r['right_probability_change_max'] for r in nofield)<tol['maximum_no_field_right_probability_change'],
        'no_field_finest_phase':abs(nofield[-1]['phase_error_rad'])<tol['finest_phase_absolute_error_rad'],
        'fft_free_centroid':abs(gaussian[0]['final_centroid_error_nm'])<tol['fft_free_centroid_absolute_error_nm'],
        'fft_free_width':abs(gaussian[0]['final_sigma_error_nm'])<tol['fft_free_sigma_absolute_error_nm'],
        'cn_free_centroid':abs(gaussian[1]['final_centroid_error_nm'])<tol['cn_free_centroid_absolute_error_nm'],
        'cn_free_width':abs(gaussian[1]['final_sigma_error_nm'])<tol['cn_free_sigma_absolute_error_nm'],
    }
    summary={'run_id':runid,'analytic':analytic,'bound_convergence_rows':convergence,
             'nofield':nofield,'free_gaussian':gaussian,'gate_a_checks':checks,
             'gate_a_pass':all(checks.values()),'elapsed_seconds':time.perf_counter()-start,
             'limitations':['No CAP or driven emission validated in Stage1.',
                            'CN phase error is reported, not removed from propagated state.',
                            'FD initial state is not interchangeable with a spectral-kinetic FFT initial state.',
                            'Two-sided finite well is an effective one-electron model, not a metal Fermi sea.']}
    json_write(results/'summary.json',summary)
    artifacts=[]
    for directory in [raw,origin,results]:
        for p in sorted(directory.iterdir()):
            artifacts.append([runid,prefix['started_at'],prefix['config_sha256'],prefix['base_git_commit'],
                              str(p.relative_to(BASE)).replace('\\','/'),sha(p),p.stat().st_size,
                              'PASS' if summary['gate_a_pass'] else 'FAIL'])
    csv_write(results/'manifest.csv',['run_id','time_iso8601','config_sha256','base_git_commit','path','sha256','bytes','status'],artifacts)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return 0 if summary['gate_a_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())

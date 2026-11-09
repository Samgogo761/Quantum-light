#!/usr/bin/env python3
"""One approved GH5 repeat, retaining endpoints. No QED evolution or scans.

The first trajectory's first 64 steps are a reusable cost probe, not an extra
trajectory. A single 240 s budget covers preparation, propagation and analysis.
Completed trajectories survive timeout; no automatic resume or second run.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

import q8_m0_qinput_shared_kk_errdecomp_20261111 as ed
import q8_m0_qinput_shared_kk_mtef_compare_20261109 as cmp
import q8_m0_qinput_shared_kk_mtef_map_20261108 as mp
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import build_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/qinput_a_shared_kk_endpoint_20261112"
OLD_GH5 = ROOT / "results/q_line/qinput_a_shared_kk_mtef_gh5_20261110"
CAP = 240.0
ORDER = 5
N_STEPS = cmp.N_STEPS
PROBE_STEPS = 64


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(serializable(data), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def serializable(data):
    if isinstance(data, np.ndarray):
        return serializable(data.tolist())
    if isinstance(data, dict):
        return {str(k): serializable(v) for k,v in data.items()}
    if isinstance(data, (tuple,list)):
        return [serializable(v) for v in data]
    if isinstance(data, (complex,np.complexfloating)):
        return {"re":float(data.real),"im":float(data.imag)}
    if isinstance(data,np.generic):
        return data.item()
    return data


class Budget:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start

    def check(self):
        # Leave time for a small final timeout report and hash checks.
        if self.elapsed() >= CAP - 2:
            raise TimeoutError("238 s work deadline; 240 s total cap")


def new_state(ek, emk, node):
    return {"ek": ek.copy(), "emk": emk.copy(), "X": float(node["X"]),
            "P": float(node["P"]), "next_j": 0, "rows": [], "times": [],
            "max_norm_error": 0.0}


def advance(state, model, env, end_step, budget, sample_steps, n_steps=N_STEPS):
    """Same calls/order/endpoint sampling as archived cmp.one_traj."""
    dt = env.t_ramp / n_steps
    for j in range(state["next_j"], end_step):
        if budget is not None:
            budget.check()
        ek, emk, X, P, _ = mp.step_ehrenfest(
            state["ek"], state["emk"], state["X"], state["P"],
            model, env.s((j + 0.5)*dt), dt, kick=True)
        state.update(ek=ek, emk=emk, X=X, P=P, next_j=j+1)
        state["max_norm_error"] = max(state["max_norm_error"],
            abs(np.linalg.norm(ek)-1), abs(np.linalg.norm(emk)-1))
        if j in sample_steps:
            t = (j+1)*dt
            state["times"].append(t)
            state["rows"].append(mp.j_total(ek, emk, model, env.s(t), X))
    return state


def electronic_symbols(model, ek, emk):
    nk, nm = np.vdot(ek, ek).real, np.vdot(emk, emk).real
    return np.array([[np.vdot(ek, Jk[a]@ek)/nk + np.vdot(emk, Jm[a]@emk)/nm
                      for a in range(2)] for Jk, Jm in zip(model.j_k, model.j_mk)])


def ensemble_decomposition(jq, dq, weights):
    """jq[r,q,a], dq[r,q]. Positive normalized GH weights counted once."""
    w = np.asarray(weights, float)
    if np.any(w <= 0) or abs(w.sum()-1) > 1e-12:
        raise ValueError("invalid GH weights")
    w = w/w.sum()
    total = np.einsum('r,rqa,rq->a', w, jq, dq)
    j_mean = np.einsum('r,rqa->qa', w, jq)
    d_mean = np.einsum('r,rq->q', w, dq)
    product = np.einsum('qa,q->a', j_mean, d_mean)
    # Independent centered-covariance formula, not total-product by definition.
    corr = np.einsum('r,rqa,rq->a', w, jq-j_mean, dq-d_mean)
    return {"total": total, "product": product, "correlation": corr,
            "reconstruction": total-product-corr, "j_mean_q": j_mean, "d_mean_q": d_mean}


def qed_decomposition(model, psi):
    """Reduced-state and direct centered joint expectation, independently."""
    V = psi.reshape(sh.EL36, sh.EL36, model.d1)
    n2 = np.vdot(psi, psi).real
    rk = np.tensordot(V, V.conj(), axes=([1, 2], [1, 2])) / n2
    rm = np.tensordot(V, V.conj(), axes=([0, 2], [0, 2])) / n2
    rp = np.tensordot(V, V.conj(), axes=([0, 1], [0, 1])) / n2
    product = np.zeros(2, complex)
    corr = np.zeros(2, complex)
    eye = np.eye(model.d1)
    for Jk, Jm, D in zip(model.j_k, model.j_mk,
                         sh.dyn.peierls_at(model.q_units, 1.0, model.n_max)):
        dj = np.trace(rp@D.T)
        # <(Jq-<Jq>)(Dq-<Dq>)>, photon matrix is symmetric.
        vd = np.tensordot(V, D-dj*eye, axes=(2, 0))
        for a in range(2):
            jm = np.trace(rk@Jk[a])+np.trace(rm@Jm[a])
            product[a] += jm*dj
            acted = np.einsum('ai,ijn->ajn', Jk[a], vd, optimize=True)
            acted += np.einsum('bj,ijn->ibn', Jm[a], vd, optimize=True)
            acted -= jm*vd
            corr[a] += np.vdot(V, acted)/n2
    total = np.array([np.vdot(psi, v)/n2 for v in sh.apply_current(model, psi, 1.0)])
    return {"total": total, "product": product, "correlation": corr,
            "reconstruction": total-product-corr,
            "product_vs_previous_formula": product-ed.factorized_current(model, psi)}


def combinations(jobs):
    out = {tag: {k: jobs[tag][k] for k in ("total", "product", "correlation")}
           for tag, _, _ in cmp.JOBS}
    for name, p, m in (("J_odd_C", "coh_p", "coh_m"), ("J_odd_DS", "ds_p", "ds_m")):
        out[name] = {k: (out[p][k]-out[m][k])/2 for k in out[p]}
    out["Delta_J_odd"] = {k: out["J_odd_DS"][k]-out["J_odd_C"][k] for k in out["J_odd_C"]}
    out["common_odd"] = {k: (out["J_odd_DS"][k]+out["J_odd_C"][k])/2 for k in out["J_odd_C"]}
    return out


def error_decomposition(m, q):
    out = {}
    for name in m:
        row = {k: m[name][k]-q[name][k] for k in ("total", "product", "correlation")}
        row["reconstruction"] = row["total"]-row["product"]-row["correlation"]
        row["sum_magnitudes_over_net"] = ((np.abs(row["product"])+np.abs(row["correlation"]))
                                            / np.maximum(np.abs(row["total"]), 1e-30))
        out[name] = row
    return out


def protected_paths():
    paths = set(ed.OLD_ARCHIVES)
    paths.add(ed.OUT_DIR / "errdecomp.json")
    paths.update(OLD_GH5.glob("*.json"))
    for _, folder, tag, _, _, _, _ in ed.CKPT_SPEC:
        paths.update(ed.dth.ckpt_paths(folder, tag))
    # Include the actual implementation modules used for this run.
    paths.update(Path(mod.__file__) for mod in (ed, cmp, mp, sh, sh.dyn))
    return sorted(paths)


def write_table(path, m, q, errors):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['quantity','axis','QED_total_re','QED_total_im','QED_product_re','QED_product_im',
                    'QED_correlation_re','QED_correlation_im','MTEF_total_re','MTEF_total_im',
                    'MTEF_product_re','MTEF_product_im','MTEF_correlation_re','MTEF_correlation_im',
                    'error_total_re','error_total_im','error_product_re','error_product_im',
                    'error_correlation_re','error_correlation_im','reconstruction_abs'])
        for name in m:
            for a, axis in enumerate(('x','y')):
                vals = [name,axis]
                for block in (q[name],m[name],errors[name]):
                    for key in ('total','product','correlation'):
                        z = block[key][a]
                        vals += [f'{z.real:.16e}',f'{z.imag:.16e}']
                vals += [f'{abs(errors[name]["reconstruction"][a]):.16e}']
                w.writerow(vals)


def run():
    if OUT.exists():
        raise FileExistsError(f"one-shot run refuses an existing directory: {OUT}")
    budget = Budget()
    OUT.mkdir(parents=True)
    traj_dir = OUT/'trajectories'
    traj_dir.mkdir()
    log = OUT/'progress.txt'
    def progress(msg):
        mp._progress(log, msg)
        print(msg, flush=True)
    paths = protected_paths()
    before = {str(p): mp.sha256_file(p) for p in paths}
    report = {"task_label":"20261112", "executed_at_utc":datetime.now(timezone.utc).isoformat(),
              "source_sha256":mp.sha256_file(Path(__file__)), "wall_cap_s":CAP,
              "new_QED_evolutions":0, "GH_order":ORDER, "n_steps":N_STEPS,
              "status":"running", "completed_trajectories":0,
              "old_hashes_before":before, "not_HHG_not_causal_attribution":True}
    try:
        geo = build_geometry()
        models, grounds, env = cmp.build_bundle(geo, geo.k_cart(sh.K_FRAC))
        js = set(cmp.sample_js())
        first_node = mp.gauss_hermite_nodes('coherent', ORDER)['nodes'][0]
        ek, emk, _ = grounds[1]
        first = new_state(ek, emk, first_node)
        t0 = time.perf_counter()
        advance(first, models[1], env, PROBE_STEPS, budget, js)
        probe_s = time.perf_counter()-t0
        estimate = probe_s/PROBE_STEPS*N_STEPS*100*1.2
        report['cost_preflight'] = {'probe_steps':PROBE_STEPS,'probe_s':probe_s,
            'trajectory_count':100,'factor':1.2,'estimated_propagation_s':estimate,
            'first_probe_reused':True,'projected_total_s':budget.elapsed()+estimate+3}
        atomic_json(OUT/'preflight.json', report['cost_preflight'])
        progress(f"cost estimate={estimate:.1f}s cap={CAP:.0f}s; first 64 steps reused")
        if budget.elapsed()+estimate+3 >= CAP:
            report['status']='cost_gate_stop'
        else:
            mjobs, time_series, regressions = {}, {}, {}
            for tag, kind, neel in cmp.JOBS:
                nodes = mp.gauss_hermite_nodes(kind, ORDER)['nodes']
                ek0, emk0, _ = grounds[neel]
                jq_list, dq_list, weights, rows_list = [], [], [], []
                for i,node in enumerate(nodes):
                    state = first if tag=='coh_p' and i==0 else new_state(ek0,emk0,node)
                    advance(state, models[neel], env, N_STEPS, budget, js)
                    jq = electronic_symbols(models[neel], state['ek'], state['emk'])
                    dq = np.exp(1j*models[neel].q_units*state['X'])
                    fp = traj_dir/f'{tag}_{i:02d}.npz'
                    tmp = fp.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,ek=state['ek'],emk=state['emk'],X=state['X'],P=state['P'],
                        weight=node['weight'],initial_X=node['X'],initial_P=node['P'],
                        tag=tag,kind=kind,neel=neel,next_j=state['next_j'],
                        t=np.asarray(state['times']),J=np.asarray(state['rows']),
                        j_q=jq,d_q=dq,source_sha256=report['source_sha256'])
                    os.replace(tmp,fp)
                    report['completed_trajectories'] += 1
                    jq_list.append(jq); dq_list.append(dq); weights.append(node['weight'])
                    rows_list.append(state['rows'])
                    if (i+1)%5==0:
                        progress(f"{tag} {i+1}/25; total={report['completed_trajectories']} wall={budget.elapsed():.2f}s")
                w = np.asarray(weights); w = w/w.sum()
                avg = np.einsum('r,rta->ta', w, np.asarray(rows_list))
                old = json.loads((OLD_GH5/f'job_{tag}.json').read_text(encoding='utf-8'))
                old_j = cmp.unpack_rows(old['J'])
                if not np.allclose(old['t'],state['times'],atol=1e-12,rtol=0):
                    raise RuntimeError(f'time mismatch {tag}')
                maxerr = float(np.max(np.abs(avg-old_j)))
                regressions[tag] = {'all_42_samples_max_abs':maxerr,
                                    'endpoint_max_abs':float(np.max(np.abs(avg[-1]-old_j[-1])))}
                if maxerr > 1e-12:
                    raise RuntimeError(f'old GH5 current mismatch {tag}: {maxerr}')
                # Only assemble the physical split after successful regression.
                mjobs[tag] = ensemble_decomposition(np.asarray(jq_list),np.asarray(dq_list),np.asarray(weights))
                mjobs[tag]['endpoint_vs_sample'] = mjobs[tag]['total']-avg[-1]
                time_series[tag] = cmp.pack_j(avg)
                atomic_json(OUT/f'job_{tag}.json',{'decomposition':mjobs[tag],'regression':regressions[tag]})
            budget.check()
            qmodels = ed.build_models({(8,1),(8,-1)})
            qjobs = {}
            for tag,kind,neel in cmp.JOBS:
                folder = ed.DIR_COH8 if kind=='coherent' else ed.DIR_DS8
                psi,nnext,samples = ed.dth.load_ckpt(folder,tag)
                if nnext!=2570 or abs(samples[-1]['t']-env.t_ramp)>1e-10:
                    raise RuntimeError(f'incompatible QED checkpoint {tag}')
                qjobs[tag] = qed_decomposition(qmodels[(8,neel)],psi)
                qjobs[tag]['vs_saved_J'] = qjobs[tag]['total']-np.array([
                    complex(z['re'],z['im']) for z in samples[-1]['mean_J']])
            m,q = combinations(mjobs), combinations(qjobs)
            errors = error_decomposition(m,q)
            reconstruction = max(float(np.max(np.abs(d['reconstruction'])))
                for d in list(mjobs.values())+list(qjobs.values())+list(errors.values()))
            report.update(status='ok',regressions=regressions,MTEF_jobs=mjobs,QED_jobs=qjobs,
                MTEF=m,QED=q,method_errors=errors,max_reconstruction_abs=reconstruction,
                qed_reference='nmax8 dt/2 endpoints; MTEF remains frozen 1285 steps',
                sample_current_series=time_series)
            report['all_numerical_checks_pass'] = bool(reconstruction<1e-12
                and all(np.max(np.abs(d['endpoint_vs_sample']))<1e-12 for d in mjobs.values())
                and all(np.max(np.abs(d['vs_saved_J']))<1e-12 for d in qjobs.values())
                and all(np.max(np.abs(d['product_vs_previous_formula']))<1e-12 for d in qjobs.values()))
            if not report['all_numerical_checks_pass']:
                report['status']='check_failed'
            write_table(OUT/'endpoint_table.tsv',m,q,errors)
    except TimeoutError as exc:
        report.update(status='timeout_partial',stop_reason=str(exc))
        progress(f"STOP: {exc}; completed endpoints retained, no automatic continuation")
    except Exception as exc:
        report.update(status='error_stopped',stop_reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['old_hashes_unchanged'] = all(mp.sha256_file(Path(p))==h for p,h in before.items())
        report['trajectory_hashes'] = {p.name:mp.sha256_file(p) for p in traj_dir.glob('*.npz')}
        report['wall_s_before_final_save'] = budget.elapsed()
        report['branch_closed_no_auto_expansion'] = True
        atomic_json(OUT/'endpoint.json',report)
        progress(f"status={report['status']} completed={report['completed_trajectories']} wall={budget.elapsed():.2f}s")
    return report


if __name__=='__main__':
    result = run()
    raise SystemExit(0 if result['status']=='ok' else 2)

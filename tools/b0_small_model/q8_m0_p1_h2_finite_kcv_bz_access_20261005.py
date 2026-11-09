#!/usr/bin/env python3
"""BZ access: generic frequency partition, static M, two-k regression, Γ12.

Does not overwrite two-k qualify/tails JSON.
Does not start half-shift 8x8 P1.
Does not change D, gamma, Hamiltonian, P1, or the packet.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("Q8_STAGE", "probe")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_continuum_ground_bridge_20261003 import ContinuumGroundBridge
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import dark_subspace
from q8_m0_k1_chi_interface import CHI_STATUS
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_caller_fix_20261005 import TWO_K, WEIGHTS, FREQ, m0_sample, sha
from q8_m0_s1_kernel import parallel_perp_basis
from q8_m0_stage1ap_h2_window_quadrature_20261004 import inversion_ok

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/p1_h2_finite_kcv_bz_access_20261005"
PLUS_JSON = REPO / "results/q_line/p1_h2_finite_kcv_nm_qualify_20261005/nm_qualify_plus.json"
TAILS_JSON = REPO / "results/q_line/p1_h2_finite_kcv_nm_tails_20261005/nm_tails.json"
PROTECTED = [
    PLUS_JSON,
    TAILS_JSON,
    q.OLD_U0_H2,
    q.OLD_INPUT_CLOSE,
    q.EIGHT_CYCLE,
]
q.OUT = OUT
H2_LO, H2_HI = 0.02, 0.30


def progress(msg):
    q.progress(msg)


def static_m(sample, gamma):
    br = ContinuumGroundBridge(sample["gap"], sample["u"], sample["D"], gamma)
    m = np.array(br.raw_static_residual(), float)
    m = 0.5 * (m + m.T)
    ev = np.linalg.eigvalsh(m)
    return {
        "min_direct_gap": float(sample["min_direct_gap"]),
        "max_direct_gap": float(sample["max_direct_gap"]),
        "n_el": int(sample["n_el"]),
        "n_k": int(sample["n_k"]),
        "M_eigs": [float(ev[0]), float(ev[1])],
        "M_min": float(ev[0]),
        "negative_curvature": bool(ev[0] < -1e-8),
        "weight_sum": float(np.sum(sample["weights"])),
    }


def growth_root(sample, gamma, s_grid=None):
    br = ContinuumGroundBridge(sample["gap"], sample["u"], sample["D"], gamma)
    if s_grid is None:
        s_grid = np.geomspace(1e-4, 1.0, 80)
    rec = []
    for s in s_grid:
        z = 1j * float(s)
        k = br.electronic_kernel(z)
        mat = float(s) * np.eye(2, dtype=complex) + float(gamma) * k
        sv = float(np.linalg.svd(mat, compute_uv=False)[-1])
        rec.append((float(s), sv))
    s_best, sv_best = min(rec, key=lambda t: t[1])
    period = 2.0 * np.pi / 0.1
    return {
        "s_at_min_svd": s_best,
        "min_svd": sv_best,
        "e_fold_time_over_p1_period": float((1.0 / max(s_best, 1e-30)) / period),
    }


def quadrature_health(oms, wts, gaps, u, gamma):
    oms = np.asarray(oms, float)
    wts = np.asarray(wts, float)
    bright, widths, groups = q.grouped_bright_gaps(gaps, u, gamma)
    dist_w = []
    for dg, w in zip(bright, widths):
        dist_w.append(float(np.min(np.abs(oms - dg)) / max(float(w), 1e-16)))
    in_h2 = (bright > H2_LO) & (bright < H2_HI)
    h2_w = []
    for dg, w in zip(bright[in_h2], widths[in_h2]):
        h2_w.append(float(np.min(np.abs(oms - dg)) / max(float(w), 1e-16)))
    inverted_cuts = False
    try:
        _b, _w, groups = q.grouped_bright_gaps(gaps, u, gamma)
        g_left = np.array([float(_b[g][0]) for g in groups], float)
        if g_left.size and g_left[0] < H2_HI:
            # old bug signature: forcing cell_lo=0.30 > first pole
            inverted_cuts = True
    except Exception:
        inverted_cuts = False
    return {
        "n_nodes": int(oms.size),
        "sum_dnu": float(np.sum(wts)),
        "omega_min": float(oms[0]) if oms.size else None,
        "omega_max": float(oms[-1]) if oms.size else None,
        "negative_weights": bool(np.any(wts < 0)),
        "unsorted": bool(oms.size and np.any(np.diff(oms) < 0)),
        "n_bright_poles": int(bright.size),
        "n_poles_in_h2_window": int(np.count_nonzero(in_h2)),
        "max_pole_distance_in_widths": float(max(dist_w) if dist_w else 0.0),
        "min_h2_pole_distance_in_widths": float(min(h2_w) if h2_w else 0.0),
        "h2_is_not_electron_free": True,
        "first_bright_below_h2_hi": bool(bright.size and bright[0] < H2_HI),
        "old_cut0_would_invert": bool(bright.size and bright[0] < H2_HI),
    }


def grid_sample(geo, nk, shift, neel=+1):
    kf, _kc, w = k_grid(geo, nk, shift=float(shift))
    sample = m0_sample(geo, kf, w, neel)
    inv = inversion_ok(kf, w)
    return sample, {
        "missing_partners": int(inv["missing_partners"]),
        "max_weight_mismatch": float(inv["max_weight_mismatch"]),
    }


def electronic_measure_batched(sample, gamma, oms, wts, batch=32):
    """Same CCR objects as electronic_measure_report; BLAS-batched outers."""
    gaps, u, d = sample["gap"], sample["u"], sample["D"]
    eta = q.eta_of(gamma)
    basis, info, near = dark_subspace(gaps, u)
    p_dark = q.dark_projector(basis)
    n = len(gaps)
    acc_bb = np.zeros((n, n), complex)
    acc_dd = np.zeros((n, n), complex)
    acc_bd = np.zeros((n, n), complex)
    acc_db = np.zeros((n, n), complex)
    skipped = 0
    buf_b, buf_d = [], []

    def flush():
        nonlocal acc_bb, acc_dd, acc_bd, acc_db, buf_b, buf_d
        if not buf_b:
            return
        B = np.concatenate(buf_b, axis=1)
        Dcol = np.concatenate(buf_d, axis=1)
        acc_bb += B @ B.conj().T
        acc_dd += Dcol.conj() @ Dcol.T
        acc_bd += B @ Dcol.conj().T
        acc_db += Dcol.conj() @ B.T
        buf_b, buf_d = [], []

    for om, dw in zip(oms, wts):
        try:
            b, bd, a = q.even_bd(gaps, u, d, gamma, eta, float(om))
        except Exception:
            skipped += 1
            continue
        s = np.sqrt(float(dw))
        buf_b.append(s * b)
        buf_d.append(s * bd)
        if len(buf_b) >= int(batch):
            flush()
    flush()
    ccr_n = acc_bb - acc_dd + p_dark - np.eye(n)
    ccr_a = acc_bd - acc_db
    herm = 0.5 * (ccr_n + ccr_n.conj().T)
    evals = np.linalg.eigvalsh(herm)
    return {
        "n_nodes": int(np.asarray(oms).size),
        "sum_dnu": float(np.sum(wts)),
        "omega_min": float(oms[0]),
        "omega_max": float(oms[-1]),
        "covers_min_gap": bool(oms[0] < sample["min_direct_gap"] < oms[-1]),
        "covers_max_gap": bool(oms[0] < sample["max_direct_gap"] < oms[-1]),
        "skipped_poles": int(skipped),
        "n_dark": int(basis.shape[1]),
        "n_el": int(n),
        "trace_photon_bb": float(np.trace(acc_bb).real),
        "trace_dd": float(np.trace(acc_dd).real),
        "trace_dark": float(np.trace(p_dark).real),
        "electronic_ccr_normal_maxabs": q.maxabs(ccr_n),
        "electronic_ccr_anomalous_maxabs": q.maxabs(ccr_a),
        "electronic_ccr_min_eig": float(evals[0]),
        "batched_outers": True,
    }
    mods = {}
    for name in ("torch", "cupy", "numba"):
        try:
            __import__(name)
            mods[name] = True
        except Exception:
            mods[name] = False
    gpu_name = None
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        gpu_name = r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    except Exception as exc:
        gpu_name = f"nvidia-smi_failed:{exc}"
    return {
        "python": sys.executable,
        "modules": mods,
        "nvidia_smi": gpu_name,
        "did_not_install_packages": True,
        "gpu_backend_in_this_env": False if not any(mods.values()) else True,
    }


def env_probe():
    mods = {}
    for name in ("torch", "cupy", "numba"):
        try:
            __import__(name)
            mods[name] = True
        except Exception:
            mods[name] = False
    gpu_name = None
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        gpu_name = r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    except Exception as exc:
        gpu_name = f"nvidia-smi_failed:{exc}"
    return {
        "python": sys.executable,
        "modules": mods,
        "nvidia_smi": gpu_name,
        "did_not_install_packages": True,
        "gpu_backend_in_this_env": bool(any(mods.values())),
    }


def write_json(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    for p in PROTECTED:
        if not p.exists():
            raise FileNotFoundError(p)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    eta = q.eta_of(gamma)
    prot = p1_a2_protocol(q.A_STAR)
    progress(f"bz access start gamma={gamma:.12g}")

    static_rows = {}
    for tag, nk, sh in (
        ("gamma8", 8, 0.0),
        ("gamma10", 10, 0.0),
        ("gamma12", 12, 0.0),
        ("half8", 8, 0.5),
        ("two_k", None, None),
    ):
        if tag == "two_k":
            sample = m0_sample(geo, TWO_K, WEIGHTS, +1)
            inv = {"missing_partners": None, "two_k_not_bz": True}
        else:
            sample, inv = grid_sample(geo, nk, sh)
        rec = static_m(sample, gamma)
        rec["inversion"] = inv
        rec["nk"] = nk
        rec["shift"] = sh
        if rec["negative_curvature"]:
            rec["growth"] = growth_root(sample, gamma)
        static_rows[tag] = rec
        progress(
            f"static {tag} gap={rec['min_direct_gap']:.6g} Mmin={rec['M_min']:.6g} "
            f"neg={rec['negative_curvature']}"
        )

    two = m0_sample(geo, TWO_K, WEIGHTS, +1)
    oms2, wts2 = q.input_quadrature(two["gap"], two["u"], gamma)
    h2 = quadrature_health(oms2, wts2, two["gap"], two["u"], gamma)
    meas2 = q.electronic_measure_report(two, gamma, oms2, wts2)
    oms2f, wts2f = q.input_quadrature(two["gap"], two["u"], gamma, n_panel=q.N_PANEL_FINE, n_h2=q.N_H2_FINE)
    meas2f = q.electronic_measure_report(two, gamma, oms2f, wts2f)
    res2 = q.resolvent_from_measure(two, gamma, oms2, wts2, q.COMPLEX_Z)
    gate2 = q.measure_gate(meas2, meas2f, res2)
    pol = np.vstack(parallel_perp_basis(geo))
    kernel = q.detector_kernel_scalar(gamma, prot, 64, prot.period / q.NTPC)
    nrm = kernel["normalizer"]

    def phi_of(nu):
        return complex(np.asarray(q.phi_at(nu, prot, nrm)).reshape(-1)[0])

    u0_2 = q.u0_packet_check(two, gamma, oms2, wts2, pol, eta, phi_of, prot, nrm)
    progress(
        f"two-k nodes={h2['n_nodes']} CCR_n={meas2['electronic_ccr_normal_maxabs']:.4g} "
        f"dressed={gate2['resolvent_dressed']:.4g} gate={gate2['ok']} U0={u0_2['ok']}"
    )

    g12, inv12 = grid_sample(geo, 12, 0.0)
    oms12, wts12 = q.input_quadrature(g12["gap"], g12["u"], gamma)
    h12 = quadrature_health(oms12, wts12, g12["gap"], g12["u"], gamma)
    progress(f"gamma12 nodes={h12['n_nodes']} poles_in_h2={h12['n_poles_in_h2_window']} "
             f"max_width_dist={h12['max_pole_distance_in_widths']:.4g}")
    t_m = time.perf_counter()
    meas12 = electronic_measure_batched(g12, gamma, oms12, wts12)
    progress(f"gamma12 CCR_n={meas12['electronic_ccr_normal_maxabs']:.4g} n_el={g12['n_el']}")
    if meas12["electronic_ccr_normal_maxabs"] > 0.2:
        meas12f = None
        refine12 = None
        progress("gamma12 fine panel not_run; coarse CCR already above 0.2")
    else:
        oms12f, wts12f = q.input_quadrature(g12["gap"], g12["u"], gamma, n_panel=q.N_PANEL_FINE)
        meas12f = electronic_measure_batched(g12, gamma, oms12f, wts12f)
        refine12 = abs(meas12["trace_photon_bb"] - meas12f["trace_photon_bb"]) / max(
            abs(meas12f["trace_photon_bb"]), 1e-30
        )
    cover12 = bool(meas12["covers_min_gap"] and meas12["covers_max_gap"])
    analytic_interface_available = True
    projected_resolvent_checked = False
    gate12 = {
        "ok": bool(
            cover12
            and meas12["electronic_ccr_normal_maxabs"] < 0.05
            and meas12["electronic_ccr_anomalous_maxabs"] < 0.05
            and refine12 is not None
            and refine12 < 0.05
            and int(meas12.get("skipped_poles", 0)) == 0
        ),
        "ccr_normal": float(meas12["electronic_ccr_normal_maxabs"]),
        "ccr_anomalous": float(meas12["electronic_ccr_anomalous_maxabs"]),
        "refine": None if refine12 is None else float(refine12),
        "refine_status": "not_run" if refine12 is None else "ran",
        "resolvent_dressed": None,
        "coordinate_stieltjes_skipped_n_el": int(g12["n_el"]),
        "coordinate_stieltjes_too_large_for_2n_matrix": True,
        "analytic_interface_available": analytic_interface_available,
        "projected_resolvent_checked": projected_resolvent_checked,
        "analytic_dressed_bridge_used": False,
    }
    br12 = ContinuumGroundBridge(g12["gap"], g12["u"], g12["D"], gamma)
    dressed_probe = []
    if meas12["electronic_ccr_normal_maxabs"] > 0.2:
        dressed_probe.append({
            "skipped": True,
            "reason": "ccr_not_closed",
            "projected_resolvent_checked": False,
            "check_passed": False,
        })
    else:
        for z in q.COMPLEX_Z:
            rd, rb = q.dressed_electron_resolvent(br12, z)
            dressed_probe.append({
                "z": {"real": float(complex(z).real), "imag": float(complex(z).imag)},
                "dressed_maxabs": q.maxabs(rd),
                "dressed_vs_bare_maxabs": q.maxabs(rd - rb),
            })
    progress(
        f"gamma12 CCR_n={meas12['electronic_ccr_normal_maxabs']:.4g} "
        f"CCR_a={meas12['electronic_ccr_anomalous_maxabs']:.4g} "
        f"gate={gate12['ok']} meas_s={time.perf_counter()-t_m:.2f}"
    )

    h8s, _ = grid_sample(geo, 8, 0.5)
    oms8, wts8 = q.input_quadrature(h8s["gap"], h8s["u"], gamma)
    h_half = quadrature_health(oms8, wts8, h8s["gap"], h8s["u"], gamma)
    basis12, info12, _ = dark_subspace(g12["gap"], g12["u"])
    n_ph12 = 4 * int(oms12.size)
    ncol12 = n_ph12 + int(basis12.shape[1])

    progress("cpu short prefix gamma12, 2 wiring freqs, 16 steps")
    t_pref = time.perf_counter()
    prefix_err = None
    try:
        q.dt_halving_check(
            geo, g12, prot, gamma, FREQ,
            lambda ns, hh: q.detector_kernel_scalar(gamma, prot, ns, hh)["k_scalar"],
            16, prot.period / q.NTPC,
        )
    except Exception as exc:
        prefix_err = repr(exc)
        progress(f"prefix failed {prefix_err}")
    prefix_s = time.perf_counter() - t_pref
    two_k_cycle = 9.0
    scale = (ncol12 / 2424.0) * (g12["n_el"] / 32.0)
    est_full_s = float(2 * 9 * two_k_cycle * scale)
    env = env_probe()
    start_plus = bool(
        gate2["ok"] and u0_2["ok"] and gate12["ok"] and not static_rows["gamma12"]["negative_curvature"]
        and est_full_s < 3 * 3600
        and env["gpu_backend_in_this_env"]
    )
    stop_reason = None
    if static_rows["half8"]["negative_curvature"]:
        stop_reason = "half-shift 8x8 static negative curvature; not used"
    if not gate12["ok"]:
        stop_reason = "gamma12 input-measure gate failed"
    if est_full_s >= 3 * 3600 and not env["gpu_backend_in_this_env"]:
        stop_reason = (
            f"CPU +N gamma12 estimate {est_full_s/3600:.1f} h and no GPU backend "
            "in this Python; not starting unbudgeted long job"
        )

    payload = {
        "status": "p1_h2_bz_access_generic_quadrature_not_domain",
        "two_k_archived": True,
        "half8_not_a_pulse_candidate": True,
        "gamma12_is_access_candidate_not_bz_converged": True,
        "static": static_rows,
        "quadrature": {
            "two_k": h2,
            "gamma12": h12,
            "half8": h_half,
            "generic_no_h2_as_electron_free": True,
            "not_a_new_physical_packet": True,
        },
        "two_k_regression": {
            "n_nodes": h2["n_nodes"],
            "old_n_nodes_601": 601,
            "measure_gate": gate2,
            "u0_packet": {"ok": u0_2["ok"], "ccr": u0_2["U0_packet_ccr"], "mass": u0_2["packet_mass_dnu"]},
            "electronic_ccr_normal": meas2["electronic_ccr_normal_maxabs"],
            "electronic_ccr_anomalous": meas2["electronic_ccr_anomalous_maxabs"],
            "trace_photon_bb": meas2["trace_photon_bb"],
        },
        "gamma12_access": {
            "n_nodes": h12["n_nodes"],
            "n_el": g12["n_el"],
            "n_dark": int(basis12.shape[1]),
            "n_photon_columns": n_ph12,
            "ncol": ncol12,
            "measure_gate": gate12,
            "electronic_ccr_normal": meas12["electronic_ccr_normal_maxabs"],
            "electronic_ccr_anomalous": meas12["electronic_ccr_anomalous_maxabs"],
            "trace_photon_bb": meas12["trace_photon_bb"],
            "dressed": gate12["resolvent_dressed"],
            "dressed_analytic_probe": dressed_probe,
            "kept_all_kcv_and_dark": True,
            "no_second_bright_vacuum": True,
        },
        "short_test": {
            "cpu_gamma12_16step_2freq_seconds": prefix_s,
            "prefix_error": prefix_err,
            "est_full_plusN_seconds_from_two_k_scale": est_full_s,
            "est_full_plusN_hours": est_full_s / 3600.0,
            "scale_ncol_nel": scale,
            "not_half8_budget": True,
            "gpu_env": env,
            "did_not_start_full_pulse": not start_plus,
        },
        "start_plusN": start_plus,
        "stop_reason": stop_reason,
        "chi_rad_placeholder_status": CHI_STATUS,
        "old_json_not_overwritten": [str(p.relative_to(REPO)) for p in PROTECTED],
        "plus_sha256": sha(PLUS_JSON) if PLUS_JSON.exists() else None,
        "tails_sha256": sha(TAILS_JSON) if TAILS_JSON.exists() else None,
        "elapsed_s": time.perf_counter() - t0,
        "production": False,
        "n_m_claimed_as_domain_signal": False,
    }
    out = write_json("bz_access.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "two_k_gate": gate2["ok"],
        "two_k_nodes": h2["n_nodes"],
        "gamma12_gate": gate12["ok"],
        "gamma12_nodes": h12["n_nodes"],
        "gamma12_CCR_n": meas12["electronic_ccr_normal_maxabs"],
        "half8_Mmin": static_rows["half8"]["M_min"],
        "gamma12_Mmin": static_rows["gamma12"]["M_min"],
        "est_hours": est_full_s / 3600.0,
        "start_plusN": start_plus,
        "stop_reason": stop_reason,
        "elapsed_s": payload["elapsed_s"],
        "out": str(out),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

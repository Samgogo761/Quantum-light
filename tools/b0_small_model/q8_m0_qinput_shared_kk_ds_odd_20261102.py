#!/usr/bin/env python3
"""Shared-mode first ΔJ_odd compare: coherent vs D(α)S(ξ) at θ=0, ±N.

Not HHG, not output squeezing, not quantum advantage. One-cycle rising
edge only. Does not rewrite old archives. No G1/H2, BZ, extra θ, Cn,
nmax=12, or Wigner trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry, h8, op_P

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "M0_shared_mode_k_minus_k_ds_odd_compare_v1"
RECORD_DATE = "2026-11-02"
WALL_CAP_S = 720.0
NMAX = 8
NMAX_REF = 80
ALPHA = 0.5 + 0j
R_SQ = 0.3
THETA = 0.0
SAMPLE_EVERY = 32
ORDER = pf.ORDER
TAYLOR_HDT_MAX = pf.TAYLOR_HDT_MAX
OMIT_GATE = 1e-5
EXT_P_GT8 = {0.0: 2.63e-6, 0.5 * np.pi: 2.31e-5, np.pi: 5.40e-5}
JOBS = (
    ("coh_p", "coherent", 1),
    ("coh_m", "coherent", -1),
    ("ds_p", "ds", 1),
    ("ds_m", "ds", -1),
)
OLD_ARCHIVES = tuple(
    ROOT / p for p in (
        "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
        "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json",
        "results/q_line/qinput_a_j0_drive_diag_20261028/diag.json",
        "results/q_line/qinput_a_shared_kk_static_20261029/static.json",
        "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json",
        "results/q_line/qinput_a_shared_kk_squeeze_diag_20261031/diag.json",
        "results/q_line/qinput_a_shared_kk_vacuum_switch_20261101/switch.json",
    )
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progress(progress, msg):
    if progress is None:
        return
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, complex):
        return {"re": float(x.real), "im": float(x.imag)}
    return x


def photon_definition():
    return {
        "state": "D(alpha) S(xi) |0>, xi = r exp(i theta)",
        "S_on_vacuum": (
            "c_0 = sech(r)^{1/2}; c_n = -exp(i theta) tanh(r) "
            "sqrt((n-1)/n) c_{n-2} for even n. Same as q8_m0_qinput_minimal."
        ),
        "D": (
            "Galerkin <m|D_inf(alpha)|n>, not exp(alpha a_P^dagger - alpha* a_P). "
            "Finite-nmax code applies this matrix to the already-truncated SV."
        ),
        "order": "D then S, not S then D",
        "squeeze_axis": (
            "ellipse orientation is theta/2. theta=0 vs pi/2 differ by 45 deg, "
            "not orthogonal. Orthogonal squeeze axes differ by theta=pi."
        ),
        "omitted_mass_code_trunc": (
            "1 - ||P D_inf P S_trunc|0>||^2 on nmax=8. Not equal to P(n>8) "
            "of the infinite DS state, because high-n SV amplitude that D "
            "would map back into n<=8 is missing."
        ),
        "omitted_P_n_gt_8": (
            "Build D S |0> at nmax_ref=80 (Galerkin = expm of truncated "
            "generators to overlap 1), then sum p_n for n>8 after normalize."
        ),
        "alpha": {"re": float(ALPHA.real), "im": float(ALPHA.imag)},
        "r": R_SQ,
        "theta_first_batch": THETA,
        "n_max": NMAX,
        "n_max_ref": NMAX_REF,
    }


def photon_raw(kind: str, n_max: int, theta: float = THETA):
    if kind == "coherent":
        return qm.coherent_coefficients(ALPHA, n_max)
    if kind == "ds":
        return dyn.displaced_squeezed_coefficients(ALPHA, R_SQ, theta, n_max)
    raise ValueError(kind)


def photon_moments(c):
    c = np.asarray(c, complex)
    mass = float(np.vdot(c, c).real)
    cn = c / np.sqrt(mass)
    n = np.arange(len(c), dtype=float)
    a = np.diag(np.sqrt(np.arange(1, len(c), dtype=float)), 1).astype(complex)
    mean_a = complex(np.vdot(cn, a @ cn))
    mean_n = float(np.vdot(cn, n * cn).real)
    mean_n2 = float(np.vdot(cn, n * n * cn).real)
    mean_aa = complex(np.vdot(cn, a @ a @ cn))
    return {
        "unnormalized_mass": mass,
        "omitted_unnormalized": max(0.0, 1.0 - mass),
        "mean_a": pf.cpack(mean_a),
        "mean_n": mean_n,
        "var_n": mean_n2 - mean_n ** 2,
        "mean_aa": pf.cpack(mean_aa),
        "top_two_mass": float(np.sum(np.abs(cn[-2:]) ** 2)),
        "p_n": [float(x) for x in (np.abs(cn) ** 2)],
    }


def p_n_gt(c, ncut: int):
    p = np.abs(np.asarray(c, complex)) ** 2
    p = p / p.sum()
    return float(p[ncut + 1:].sum())


def photon_audit():
    rows = []
    for th in (0.0, 0.5 * np.pi, np.pi):
        c8 = photon_raw("ds", NMAX, th)
        c_ref = photon_raw("ds", NMAX_REF, th)
        m8 = photon_moments(c8)
        p_gt8 = p_n_gt(c_ref, NMAX)
        ext = EXT_P_GT8[th]
        rows.append({
            "theta": float(th),
            "axis_deg": float(0.5 * th * 180.0 / np.pi),
            "code_trunc8_omitted_unnormalized": m8["omitted_unnormalized"],
            "large80_P_n_gt_8": p_gt8,
            "external_P_n_gt_8": ext,
            "rel_diff_to_external": abs(p_gt8 - ext) / ext,
            "code8_normalized_moments": m8,
        })
    th0 = rows[0]
    coh = photon_moments(photon_raw("coherent", NMAX))
    ds_norm = photon_raw("ds", NMAX, THETA)
    nrm = float(np.linalg.norm(ds_norm))
    ds_used = ds_norm / nrm
    return {
        "definition": photon_definition(),
        "theta_scan": rows,
        "coherent_nmax8_normalized": coh,
        "ds_theta0_raw_norm": nrm,
        "ds_theta0_normalized_moments": photon_moments(ds_used),
        "label_correction": (
            "Do not call theta=0 and theta=pi/2 orthogonal. First batch uses "
            "theta=0 only."
        ),
        "gate_ds_theta0_large_P_gt8_lt_1e-5": bool(th0["large80_P_n_gt_8"] < OMIT_GATE),
        "one_pager_error": (
            "20261102 one-pager quoted ~4e-6 for both theta=0 and pi/2. That "
            "was the nmax=8 Galerkin-on-truncated omitted mass at theta=0 only; "
            "theta=pi/2 was not computed. Large-space P(n>8) matches the "
            "external independent numbers."
        ),
    }


def ground_pair_neel(geo, k, neel: int):
    ek, mass_k, leak_k = sh.slater36(dyn.electron_ground_state(geo, R0, k, neel))
    emk, mass_mk, leak_mk = sh.slater36(dyn.electron_ground_state(geo, R0, -k, neel))
    return ek, emk, {
        "neel": int(neel),
        "mass_k": mass_k, "mass_mk": mass_mk,
        "leak_k": leak_k, "leak_mk": leak_mk,
        "norm_k": float(np.linalg.norm(ek)),
        "norm_mk": float(np.linalg.norm(emk)),
    }


def electron_map_audit(geo, k):
    wp = sh.wedge_p36()
    p8 = op_P()
    ek_p, emk_p, mass_p = ground_pair_neel(geo, k, 1)
    ek_m, emk_m, mass_m = ground_pair_neel(geo, k, -1)
    ov_k = complex(np.vdot(emk_m, wp @ ek_p))
    ov_mk = complex(np.vdot(ek_m, wp @ emk_p))
    mapped = np.kron(wp @ emk_p, wp @ ek_p)
    target = np.kron(ek_m, emk_m)
    ov_prod = complex(np.vdot(target, mapped))
    hres_k = float(np.max(np.abs(p8 @ h8(geo, R0, k, 1) @ p8.conj().T - h8(geo, R0, -k, -1))))
    hres_mk = float(np.max(np.abs(p8 @ h8(geo, R0, -k, 1) @ p8.conj().T - h8(geo, R0, k, -1))))
    abs_prod = abs(ov_prod)
    return {
        "statement": (
            "P |e(k,+N)> = e^{i phi} |e(-k,-N)> and P |e(-k,+N)> = e^{i psi} "
            "|e(k,-N)>. Then Swap (P⊗P) |e_k(+N) e_{-k}(+N)> equals the -N "
            "product up to a global phase. Current relation J_{-N}(t; alpha, xi) "
            "= -J_{+N}(t; -alpha, xi) needs this electronic map plus photon "
            "parity on alpha. This round does not propagate -alpha."
        ),
        "plus": mass_p,
        "minus": mass_m,
        "overlap_P_ek_plus_vs_emk_minus": pf.cpack(ov_k),
        "abs_P_ek_plus_vs_emk_minus": abs(ov_k),
        "overlap_P_emk_plus_vs_ek_minus": pf.cpack(ov_mk),
        "abs_P_emk_plus_vs_ek_minus": abs(ov_mk),
        "overlap_Swap_PoxP_product": pf.cpack(ov_prod),
        "abs_Swap_PoxP_product": abs_prod,
        "h8_P_k_plus_vs_mk_minus": hres_k,
        "h8_P_mk_plus_vs_k_minus": hres_mk,
        "phase_not_gauged_to_1": True,
        "gate_abs_product_gt_1_minus_1e12": bool(abs_prod > 1.0 - 1e-12),
    }


def input_gate(geo, k):
    ph = photon_audit()
    el = electron_map_audit(geo, k)
    ok = bool(ph["gate_ds_theta0_large_P_gt8_lt_1e-5"] and el["gate_abs_product_gt_1_minus_1e12"])
    return ok, {"photon": ph, "electron": el, "pass": ok}


def product_normalized(ek, emk, photon):
    psi = sh.product_state(ek, emk, photon)
    nrm = float(np.linalg.norm(psi))
    return psi / nrm, nrm


def j_series(samples):
    t = np.array([s["t"] for s in samples], float)
    j = np.array(
        [[s["mean_J"][a]["re"] + 1j * s["mean_J"][a]["im"] for a in range(2)]
         for s in samples], complex)
    return t, j


def even_odd(jp, jm):
    return 0.5 * (jp + jm), 0.5 * (jp - jm)


def hann_jq(t, j, omega0, qmax=8):
    t = np.asarray(t, float)
    j = np.asarray(j, complex)
    T = float(t[-1] - t[0])
    if T <= 0.0:
        raise ValueError("empty window")
    w = np.sin(np.pi * (t - t[0]) / T) ** 2
    out = []
    for q in range(int(qmax) + 1):
        ker = w * np.exp(1j * q * omega0 * t)
        integ = np.trapezoid
        slot = np.array([integ(ker * j[:, a], t) for a in range(j.shape[1])])
        out.append({"q": q, "Jx": pf.cpack(slot[0]), "Jy": pf.cpack(slot[1])})
    return {
        "convention": "int_0^T w(t) exp(+i q omega0 t) J(t) dt, Hann w, q>=0",
        "not_certified_H2": True,
        "slots": out,
    }


def compare_block(runs):
    t_c, jp_c = j_series(runs["coh_p"]["samples"])
    t_m, jm_c = j_series(runs["coh_m"]["samples"])
    t_d, jp_d = j_series(runs["ds_p"]["samples"])
    t_e, jm_d = j_series(runs["ds_m"]["samples"])
    if not (np.allclose(t_c, t_m) and np.allclose(t_c, t_d) and np.allclose(t_c, t_e)):
        raise RuntimeError("time grids of the four jobs do not match")
    even_c, odd_c = even_odd(jp_c, jm_c)
    even_d, odd_d = even_odd(jp_d, jm_d)
    d_odd = odd_d - odd_c
    d_even = even_d - even_c
    imag_floor = max(
        float(np.max(np.abs(jp_c.imag))), float(np.max(np.abs(jm_c.imag))),
        float(np.max(np.abs(jp_d.imag))), float(np.max(np.abs(jm_d.imag))),
    )
    max_abs = {
        "dJ_odd_x": float(np.max(np.abs(d_odd[:, 0]))),
        "dJ_odd_y": float(np.max(np.abs(d_odd[:, 1]))),
        "dJ_even_x": float(np.max(np.abs(d_even[:, 0]))),
        "dJ_even_y": float(np.max(np.abs(d_even[:, 1]))),
        "J_odd_coh_x": float(np.max(np.abs(odd_c[:, 0]))),
        "J_odd_coh_y": float(np.max(np.abs(odd_c[:, 1]))),
        "J_odd_ds_x": float(np.max(np.abs(odd_d[:, 0]))),
        "J_odd_ds_y": float(np.max(np.abs(odd_d[:, 1]))),
        "imag_J_floor": imag_floor,
    }
    scale = max(max_abs["J_odd_coh_x"], max_abs["J_odd_coh_y"],
                max_abs["J_odd_ds_x"], max_abs["J_odd_ds_y"], 1e-30)
    return {
        "t": t_c.tolist(),
        "J_even_coh": [[pf.cpack(z) for z in row] for row in even_c],
        "J_odd_coh": [[pf.cpack(z) for z in row] for row in odd_c],
        "J_even_ds": [[pf.cpack(z) for z in row] for row in even_d],
        "J_odd_ds": [[pf.cpack(z) for z in row] for row in odd_d],
        "dJ_odd": [[pf.cpack(z) for z in row] for row in d_odd],
        "dJ_even": [[pf.cpack(z) for z in row] for row in d_even],
        "max_abs": max_abs,
        "dJ_odd_over_J_odd_scale": {
            "x": max_abs["dJ_odd_x"] / scale,
            "y": max_abs["dJ_odd_y"] / scale,
            "scale": scale,
        },
        "hann_J_odd_coh": hann_jq(t_c, odd_c, pf.OMEGA0),
        "hann_J_odd_ds": hann_jq(t_c, odd_d, pf.OMEGA0),
        "hann_dJ_odd": hann_jq(t_c, d_odd, pf.OMEGA0),
        "not_certified_harmonic_spectrum": True,
        "no_quantum_advantage_claim": True,
        "Cn_and_Wigner_not_run": True,
    }, t_c, even_c, odd_c, even_d, odd_d, d_odd


def write_plots(out_dir: Path, t, even_c, odd_c, even_d, odd_d, d_odd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    series = (
        (axes[0, 0], odd_c[:, 0].real, odd_d[:, 0].real, d_odd[:, 0].real, r"$J_{\mathrm{odd},x}$"),
        (axes[0, 1], odd_c[:, 1].real, odd_d[:, 1].real, d_odd[:, 1].real, r"$J_{\mathrm{odd},y}$"),
        (axes[1, 0], even_c[:, 0].real, even_d[:, 0].real, (even_d - even_c)[:, 0].real, r"$J_{\mathrm{even},x}$"),
        (axes[1, 1], even_c[:, 1].real, even_d[:, 1].real, (even_d - even_c)[:, 1].real, r"$J_{\mathrm{even},y}$"),
    )
    for ax, coh, ds, delta, title in series:
        ax.plot(t, coh, color="#4c6b8a", lw=1.4, label="coherent")
        ax.plot(t, ds, color="#c45c26", lw=1.4, label=r"$D(\alpha)S(r,\theta{=}0)$")
        ax.plot(t, delta, color="#2f9e44", lw=1.1, ls="--", label=r"$\Delta$")
        ax.set_title(title)
        ax.axhline(0.0, color="#888", lw=0.6)
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    fig.suptitle(
        r"shared $k,-k$: $J_{\mathrm{even/odd}}=[J_{+N}\pm J_{-N}]/2$"
        "\nnot HHG, not quantum advantage, one-cycle rising edge",
        fontsize=10)
    fig.tight_layout()
    png = out_dir / "j_even_odd.png"
    svg = out_dir / "j_even_odd.svg"
    fig.savefig(png, dpi=140)
    fig.savefig(svg)
    plt.close(fig)
    return [str(png.as_posix()), str(svg.as_posix())]


def run_job(geo, k, env, kind, neel, n_steps, clock, progress, tag):
    ek, emk, masses = ground_pair_neel(geo, k, neel)
    raw = photon_raw(kind, NMAX, THETA)
    raw_diag = photon_moments(raw)
    c = raw / np.sqrt(raw_diag["unnormalized_mass"])
    model = sh.build_shared(geo, R0, k, neel, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    psi, prod_n = product_normalized(ek, emk, c)
    s0 = pf.snapshot(model, psi, 0.0)
    _progress(progress, f"{tag} start neel={neel} kind={kind} dim={model.dim} steps={n_steps}")
    psi, led = pf.propagate(model, psi, 0.0, env.t_ramp, env, n_steps, clock, SAMPLE_EVERY)
    send = pf.snapshot(model, psi, 1.0)
    packed = pf.pack_samples(led["samples"])
    top = [s["top_fock_mass"] for s in led["samples"]]
    norms = [s["norm"] for s in led["samples"]]
    out = {
        "tag": tag,
        "kind": kind,
        "neel": int(neel),
        "electron": masses,
        "photon_unnormalized": raw_diag,
        "photon_used_normalized": True,
        "product_norm_before_renorm": prod_n,
        "initial": s0,
        "after_ramp": send,
        "delta_energy": send["energy"] - s0["energy"],
        "work_trap_dHds_dsdt": led["work_trap_dHds_dsdt"],
        "energy_minus_work": (send["energy"] - s0["energy"]) - led["work_trap_dHds_dsdt"],
        "ledger": {k: led[k] for k in ("steps", "dt", "order", "max_last_taylor_term")},
        "max_top_fock": float(np.max(top) if top else send["top_fock_mass"]),
        "max_abs_norm_minus_1": float(np.max(np.abs(np.asarray(norms) - 1.0))) if norms else abs(send["norm"] - 1.0),
        "time_series": packed,
        "samples": led["samples"],
        "wall_seconds": clock.elapsed(),
        "top_two_alarm_0p05_is_not_convergence": True,
    }
    _progress(progress, f"{tag} done wall={clock.elapsed():.1f}s dE={out['delta_energy']:.3e}")
    return out


def _parallel_job(tag, kind, neel, n_steps, progress_s):
    progress = Path(progress_s) if progress_s else None
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    clock = dyn.WallClock(WALL_CAP_S)
    return run_job(geo, k, env, kind, neel, n_steps, clock, progress, tag)


def run(progress=None):
    hashes_before = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    ok, gate = input_gate(geo, k)
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "ds_vs_coherent_J_odd_method_baseline_not_HHG_not_advantage",
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite trial.json", "rewrite 20261027 preflight.json",
            "rewrite 20261028 diag.json", "rewrite static.json",
            "rewrite 20261030 preflight.json", "rewrite 20261031 diag.json",
            "rewrite 20261101 switch.json", "G1", "H2", "HHG spectra",
            "theta=pi/2", "theta=pi", "Cn", "nmax=12",
            "Wigner trajectories", "full BZ", "output photon statistics",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "envelope": "1-cycle sin^2 rising edge, stop at s=1, no plateau",
            "n_max": NMAX, "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": 0.5, "r": R_SQ, "theta": THETA,
            "jobs": [list(j) for j in JOBS],
            "k_fractional": list(sh.K_FRAC),
            "sample_every": SAMPLE_EVERY,
            "parallel_jobs": 4,
        },
        "input_gate": gate,
        "joint_inversion_current_interface": {
            "formula": "J_{-N}(t; alpha, xi) = -J_{+N}(t; -alpha, xi)",
            "electronic_initial_states_mapped": gate["electron"]["gate_abs_product_gt_1_minus_1e12"],
            "minus_alpha_not_propagated": True,
        },
        "old_archive_hashes_before": hashes_before,
    }
    if not ok:
        report["status"] = "gate_failed_no_propagation"
        report["stopped"] = True
        return report

    rng = np.random.default_rng(20261102)
    model_probe = sh.build_shared(geo, R0, k, 1, n_max=NMAX, omega0=pf.OMEGA0, g=pf.G_ABS)
    rho, _ = pf.spectral_radius_power(model_probe, 1.0, rng)
    n_steps = dyn.n_steps_for(env.t_ramp, rho, requested=32, hdt_max=TAYLOR_HDT_MAX)
    dt = env.t_ramp / n_steps
    v = rng.normal(size=model_probe.dim) + 1j * rng.normal(size=model_probe.dim)
    v = v / np.linalg.norm(v)
    t0 = time.perf_counter()
    pf.taylor_step(model_probe, v, 0.4, dt)
    t_step = time.perf_counter() - t0
    est_one_raw = t_step * n_steps
    est_one = 3.0 * est_one_raw
    est_serial = 4.0 * est_one
    est_wall = est_one
    report["cost"] = {
        "dim": model_probe.dim, "n_steps": n_steps, "dt": dt,
        "timed_taylor_step_seconds": t_step,
        "sample_every": SAMPLE_EVERY,
        "estimated_one_job_raw_seconds": est_one_raw,
        "estimated_one_job_seconds": est_one,
        "estimated_serial_seconds": est_serial,
        "estimated_wall_parallel_seconds": est_wall,
        "wall_cap_s": WALL_CAP_S,
        "exceeds_12min": bool(est_wall > WALL_CAP_S),
        "serial_four_jobs_exceed_12min": bool(est_serial > WALL_CAP_S),
        "note": "timed Taylor undercounts; x3 from sample_every=8 job ~241s vs ~85s raw. Four jobs run in parallel.",
    }
    _progress(progress, f"gate pass est_wall={est_wall:.1f}s serial={est_serial:.1f}s steps={n_steps} dt={dt}")
    if est_wall > WALL_CAP_S:
        report["status"] = "cost_gate_failed_no_propagation"
        report["stopped"] = True
        return report

    t_par0 = time.perf_counter()
    progress_s = str(progress) if progress is not None else ""
    runs = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(_parallel_job, tag, kind, neel, n_steps, progress_s): tag
            for tag, kind, neel in JOBS
        }
        for fut in futs:
            tag = futs[fut]
            runs[tag] = fut.result()
    clock_elapsed = time.perf_counter() - t_par0
    slim = {}
    for tag, job in runs.items():
        slim[tag] = {k: job[k] for k in job if k != "samples"}
    cmp_block, t, even_c, odd_c, even_d, odd_d, d_odd = compare_block(runs)
    report["jobs"] = slim
    report["compare"] = {k: cmp_block[k] for k in cmp_block if k not in (
        "J_even_coh", "J_odd_coh", "J_even_ds", "J_odd_ds", "dJ_odd", "dJ_even")}
    report["compare"]["series"] = {
        "t": cmp_block["t"],
        "J_odd_coh": cmp_block["J_odd_coh"],
        "J_odd_ds": cmp_block["J_odd_ds"],
        "dJ_odd": cmp_block["dJ_odd"],
        "J_even_coh": cmp_block["J_even_coh"],
        "J_even_ds": cmp_block["J_even_ds"],
        "dJ_even": cmp_block["dJ_even"],
    }
    report["numerical_floor"] = {
        "max_last_taylor": {tag: runs[tag]["ledger"]["max_last_taylor_term"] for tag in runs},
        "max_abs_norm_minus_1": {tag: runs[tag]["max_abs_norm_minus_1"] for tag in runs},
        "max_top_fock": {tag: runs[tag]["max_top_fock"] for tag in runs},
        "imag_J_floor": cmp_block["max_abs"]["imag_J_floor"],
        "no_dt_refinement_this_round": True,
        "vacuum_dt_Wsq_rel_was_2e-7_not_a_J_errorbar": True,
    }
    report["wall_seconds_total"] = clock_elapsed
    report["status"] = "ok"
    report["stopped"] = True
    report["_plot_arrays"] = (t, even_c, odd_c, even_d, odd_d, d_odd)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path,
                    default=ROOT / "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json")
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    if args.report.exists():
        print(f"report already exists: {args.report}")
        return 2
    if args.check_only:
        geo = build_geometry()
        k = geo.k_cart(sh.K_FRAC)
        ok, gate = input_gate(geo, k)
        payload = {"pass": ok, "gate": gate}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as fh:
            json.dump(jsonable(payload), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(json.dumps({"pass": ok}, ensure_ascii=False))
        return 0 if ok else 3
    t0 = time.perf_counter()
    report = run(progress=args.progress)
    plot_arrays = report.pop("_plot_arrays", None)
    report["wall_seconds_process"] = time.perf_counter() - t0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if plot_arrays is not None:
        report["figures"] = write_plots(args.report.parent, *plot_arrays)
    hashes_after = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report["old_archive_hashes_unchanged"] = hashes_after == report.get("old_archive_hashes_before", hashes_after)
    with args.report.open("x", encoding="utf-8") as fh:
        json.dump(jsonable(report), fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    summary = {
        "status": report.get("status"),
        "gate": report["input_gate"]["pass"],
        "wall": report.get("wall_seconds_total"),
        "max_abs": None if "compare" not in report else report["compare"].get("max_abs"),
        "old_untouched": report.get("old_archive_hashes_unchanged"),
    }
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    if report.get("status") == "gate_failed_no_propagation":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

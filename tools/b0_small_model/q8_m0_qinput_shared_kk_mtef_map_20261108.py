#!/usr/bin/env python3
"""Wigner/MTEF map of the frozen shared-mode Hamiltonian.

Equations, static consistency, one Ehrenfest rising-edge, cost estimate.
Does not launch a quadrature ensemble. Not HHG, not quantum advantage.
Does not rewrite old archives. No G1/H2, vacuum, nmax, Cn, squeeze angle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm

import q8_m0_qinput_dynamics_20261026 as dyn
import q8_m0_qinput_minimal_20261020 as qm
import q8_m0_qinput_shared_kk_ds_odd_20261102 as ds
import q8_m0_qinput_shared_kk_dyn_preflight_20261030 as pf
import q8_m0_qinput_shared_kk_static_20261029 as sh
from q8_m0_model import R0, build_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/q_line/qinput_a_shared_kk_mtef_map_20261108"
MODEL_ID = "M0_shared_mode_k_minus_k_mtef_map_v1"
RECORD_DATE = "2026-11-08"
WALL_CAP_S = 120.0
ALPHA = ds.ALPHA
R_SQ = ds.R_SQ
THETA = ds.THETA
N_STEPS_LOCK = 1285
SAMPLE_EVERY = 32
NMAX_FOCK_REF = 80
COST_PROBE_STEPS = 40
GH_ORDERS_STATIC = (1, 2, 3, 5)
OLD_ARCHIVES = tuple(
    ROOT / p for p in (
        "results/q_line/qinput_a_dynamics_trial_20261026/trial.json",
        "results/q_line/qinput_a_truncation_preflight_20261027/preflight.json",
        "results/q_line/qinput_a_j0_drive_diag_20261028/diag.json",
        "results/q_line/qinput_a_shared_kk_static_20261029/static.json",
        "results/q_line/qinput_a_shared_kk_dyn_preflight_20261030/preflight.json",
        "results/q_line/qinput_a_shared_kk_squeeze_diag_20261031/diag.json",
        "results/q_line/qinput_a_shared_kk_vacuum_switch_20261101/switch.json",
        "results/q_line/qinput_a_shared_kk_ds_odd_20261102/compare.json",
        "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/trunc.json",
        "results/q_line/qinput_a_shared_kk_ds_nmax12_20261103/perf.json",
        "results/q_line/qinput_a_shared_kk_ds_dt_20261104/dt.json",
        "results/q_line/qinput_a_shared_kk_coh_dt_20261105/coh_dt.json",
        "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106/trunc10.json",
        "results/q_line/qinput_a_shared_kk_ds_nmax10_20261106/trunc10_resume.json",
    )
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _progress(progress, msg):
    if progress is None:
        return
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


# ---------------------------------------------------------------- phase space
def analytic_xp(kind: str):
    """Peierls X=a+a†, P=-i(a-a†). Vacuum Var(X)=1, not the W_sq /2 quadrature."""
    a = complex(ALPHA)
    mx, mp = 2.0 * a.real, 2.0 * a.imag
    if kind == "coherent":
        vx, vp, r = 1.0, 1.0, 0.0
    elif kind == "ds":
        if abs(THETA) > 1e-15:
            raise ValueError("this map only ships theta=0")
        vx, vp, r = float(np.exp(-2.0 * R_SQ)), float(np.exp(2.0 * R_SQ)), R_SQ
    else:
        raise ValueError(kind)
    nbar = abs(a) ** 2 + np.sinh(r) ** 2
    return {
        "kind": kind, "alpha": pf.cpack(a), "r": r, "theta": THETA,
        "mean_X": mx, "mean_P": mp, "var_X": vx, "var_P": vp,
        "mean_n_Weyl": 0.25 * (mx * mx + mp * mp + vx + vp) - 0.5,
        "mean_n_analytic": float(nbar),
        "var_X_equals_e_minus_2r": bool(kind == "ds"),
    }


def fock_xp_moments(c):
    c = np.asarray(c, complex)
    mass = float(np.vdot(c, c).real)
    cn = c / np.sqrt(mass)
    n = np.arange(len(cn), dtype=float)
    mean_n = float(np.vdot(cn, n * cn).real)
    mean_a = 0.0j
    mean_aa = 0.0j
    for i in range(len(cn) - 1):
        mean_a += np.conj(cn[i]) * cn[i + 1] * np.sqrt(i + 1)
    for i in range(len(cn) - 2):
        mean_aa += np.conj(cn[i]) * cn[i + 2] * np.sqrt((i + 1) * (i + 2))
    mean_X = 2.0 * mean_a.real
    mean_P = 2.0 * mean_a.imag
    mean_X2 = 2.0 * mean_aa.real + 2.0 * mean_n + 1.0
    mean_P2 = -2.0 * mean_aa.real + 2.0 * mean_n + 1.0
    return {
        "unnormalized_mass": mass,
        "mean_a": pf.cpack(mean_a),
        "mean_n": mean_n,
        "mean_X": float(mean_X),
        "mean_P": float(mean_P),
        "var_X": float(mean_X2 - mean_X ** 2),
        "var_P": float(mean_P2 - mean_P ** 2),
        "mean_n_Weyl_from_XP": float(0.25 * (mean_X2 + mean_P2) - 0.5),
    }


def gauss_hermite_nodes(kind: str, order: int):
    """Product GH in the standardized plane, affine-mapped to each Wigner.

    Same (xi, eta) for coherent and DS: common nodes / CRN. Mean-field is order 1.
    """
    if order < 1:
        raise ValueError("GH order >= 1")
    xi, w1 = hermgauss(order)
    mom = analytic_xp(kind)
    sx, sp = np.sqrt(mom["var_X"]), np.sqrt(mom["var_P"])
    nodes = []
    for i, x in enumerate(xi):
        for j, y in enumerate(xi):
            wt = float(w1[i] * w1[j] / np.pi)
            nodes.append({
                "xi": float(x), "eta": float(y), "weight": wt,
                "X": mom["mean_X"] + sx * np.sqrt(2.0) * float(x),
                "P": mom["mean_P"] + sp * np.sqrt(2.0) * float(y),
            })
    wsum = float(sum(n["weight"] for n in nodes))
    mx = sum(n["weight"] * n["X"] for n in nodes)
    mp = sum(n["weight"] * n["P"] for n in nodes)
    vx = sum(n["weight"] * n["X"] ** 2 for n in nodes) - mx * mx
    vp = sum(n["weight"] * n["P"] ** 2 for n in nodes) - mp * mp
    return {
        "kind": kind, "order": int(order), "n_nodes": len(nodes),
        "weight_sum": wsum,
        "quad_mean_X": float(mx), "quad_mean_P": float(mp),
        "quad_var_X": float(vx), "quad_var_P": float(vp),
        "target_mean_X": mom["mean_X"], "target_var_X": mom["var_X"],
        "target_var_P": mom["var_P"],
        "nodes": nodes,
        "mean_field_is_order_1": bool(order == 1),
    }


# ---------------------------------------------------------------- electronic + field
def h_el(onsite, f_groups, q_units, s: float, X: float):
    h = np.diag(np.asarray(onsite, complex))
    for F, q in zip(f_groups, q_units):
        h = h + F * np.exp(1j * float(q) * float(s) * float(X))
    return h


def j_el(psi, j_groups, q_units, s: float, X: float):
    j = np.zeros(2, complex)
    nrm2 = float(np.vdot(psi, psi).real) + 1e-30
    for Jq, q in zip(j_groups, q_units):
        phase = np.exp(1j * float(q) * float(s) * float(X))
        for a in range(2):
            j[a] += np.vdot(psi, Jq[a] @ psi) * phase / nrm2
    return j


def j_total(ek, emk, model, s: float, X: float):
    return (j_el(ek, model.j_k, model.q_units, s, X)
            + j_el(emk, model.j_mk, model.q_units, s, X))


def rotate_xp(X, P, dt: float, omega0: float):
    c, s_ = np.cos(omega0 * dt), np.sin(omega0 * dt)
    return float(X * c + P * s_), float(P * c - X * s_)


def kick_p(P, s: float, j, g: float, e, dt: float):
    return float(P + dt * 2.0 * float(s) * float(g) * float(np.dot(e, j.real)))


def electron_step(ek, emk, model, s: float, X: float, dt: float):
    uk = expm(-1j * dt * h_el(model.onsite_k, model.f_k, model.q_units, s, X))
    um = expm(-1j * dt * h_el(model.onsite_mk, model.f_mk, model.q_units, s, X))
    return uk @ ek, um @ emk


def step_ehrenfest(ek, emk, X, P, model, s: float, dt: float, kick: bool = True):
    X, P = rotate_xp(X, P, 0.5 * dt, model.omega0)
    ek, emk = electron_step(ek, emk, model, s, X, dt)
    j = j_total(ek, emk, model, s, X)
    if kick:
        P = kick_p(P, s, j, model.g, model.e, dt)
    X, P = rotate_xp(X, P, 0.5 * dt, model.omega0)
    return ek, emk, X, P, j


def snapshot_traj(ek, emk, X, P, model, s: float, t: float):
    j = j_total(ek, emk, model, s, X)
    nk = float(np.linalg.norm(ek))
    nm = float(np.linalg.norm(emk))
    e_el = sh.classical_x_energy(model, ek, emk, float(s) * float(X))
    n_w = 0.25 * (X * X + P * P) - 0.5
    return {
        "t": float(t), "s": float(s), "X": float(X), "P": float(P),
        "n_Weyl": float(n_w),
        "norm_k": nk, "norm_mk": nm,
        "mean_J": [pf.cpack(z) for z in j],
        "F_linear": float(-model.g * np.dot(model.e, j.real)),
        "energy_el_plus_peierls": float(np.real(e_el)),
        "energy_ph_Weyl": float(model.omega0 * n_w),
    }


# ---------------------------------------------------------------- static checks
def static_correspondence():
    n_max = 8
    a, ad, _ = qm.boson_ops(n_max)
    X = a + ad
    P = -1j * (a - ad)
    comm = X @ P - P @ X
    return {
        "X_def": "a + a_dagger",
        "P_def": "-i (a - a_dagger)",
        "commutator_XP_minus_2i_aa_dag_nmax8": float(np.max(np.abs(comm - 2j * (a @ ad - ad @ a)))),
        "interior_XP_minus_2iI_nmax8": float(np.max(np.abs((comm - 2j * np.eye(n_max + 1))[:-1, :-1]))),
        "note_cutoff": (
            "On infinite space [X,P]=2i. Finite Fock only fails on the last "
            "number state. MTEF uses the infinite-space Poisson bracket {X,P}=2."
        ),
        "Weyl_n": "(X^2 + P^2 - 2)/4",
        "eom_free": "dX/dt = omega0 P, dP/dt = -omega0 X",
        "eom_backaction": "dP/dt += 2 s g e·<J>",
        "factor_2_from": "[X,P]=2i, not an arbitrary scale",
        "not_Wsq_quadrature": (
            "Do not use X_theta=(a e^{-i theta}+h.c.)/2 from the 20261030 W_sq "
            "page. Vacuum Var(X_peierls)=1, vacuum Var(X_theta)=1/4."
        ),
        "peierls_classical": "exp(i q s X_cl), Weyl symbol of exp(i q s X_hat)",
        "mean_field_cannot_split_Calpha_and_DS": True,
    }


def static_wigner_and_quadrature():
    rows = {}
    for kind in ("coherent", "ds"):
        ana = analytic_xp(kind)
        raw = ds.photon_raw(kind, NMAX_FOCK_REF, THETA)
        fock = fock_xp_moments(raw)
        gh = {m: gauss_hermite_nodes(kind, m) for m in GH_ORDERS_STATIC}
        gh_slim = {}
        for m, g in gh.items():
            gh_slim[str(m)] = {k: g[k] for k in g if k != "nodes"}
            gh_slim[str(m)]["abs_d_mean_X"] = abs(g["quad_mean_X"] - ana["mean_X"])
            gh_slim[str(m)]["abs_d_var_X"] = abs(g["quad_var_X"] - ana["var_X"])
            gh_slim[str(m)]["abs_d_var_P"] = abs(g["quad_var_P"] - ana["var_P"])
        rows[kind] = {
            "analytic": ana,
            "fock_nmax80": fock,
            "abs_d_var_X_fock_minus_Wigner": abs(fock["var_X"] - ana["var_X"]),
            "abs_d_mean_n_fock_minus_analytic": abs(fock["mean_n"] - ana["mean_n_analytic"]),
            "gauss_hermite": gh_slim,
        }
    coh1 = gauss_hermite_nodes("coherent", 1)["nodes"][0]
    ds1 = gauss_hermite_nodes("ds", 1)["nodes"][0]
    rows["mean_field_shared_XP"] = {
        "coherent": {"X": coh1["X"], "P": coh1["P"]},
        "ds": {"X": ds1["X"], "P": ds1["P"]},
        "same_initial_XP": bool(abs(coh1["X"] - ds1["X"]) < 1e-15 and abs(coh1["P"] - ds1["P"]) < 1e-15),
        "consequence": "order-1 Ehrenfest Delta J_odd must vanish identically",
    }
    coh_xi = [(n["xi"], n["eta"]) for n in gauss_hermite_nodes("coherent", 3)["nodes"]]
    ds_xi = [(n["xi"], n["eta"]) for n in gauss_hermite_nodes("ds", 3)["nodes"]]
    rows["crn_order3_same_standard_plane"] = bool(coh_xi == ds_xi)
    return rows


def static_peierls_weyl(model):
    """Wigner average of e^{i q X} for a coherent Gaussian equals the quantum displacement."""
    q = float(model.q_units[np.argmax(np.abs(model.q_units))])
    ana = analytic_xp("coherent")
    # infinite-space coherent: <e^{i q X}> = exp(i q <X> - q^2 Var(X)/2)
    quantum = np.exp(1j * q * ana["mean_X"] - 0.5 * q * q * ana["var_X"])
    gh3 = gauss_hermite_nodes("coherent", 3)
    quad = sum(n["weight"] * np.exp(1j * q * n["X"]) for n in gh3["nodes"])
    return {
        "q_probe": q,
        "analytic_Weyl_coherent": pf.cpack(quantum),
        "gh3_average": pf.cpack(quad),
        "abs_diff_gh3_minus_analytic": float(abs(quad - quantum)),
        "note": (
            "Classical average of exp(i q X) against the Wigner is the Weyl "
            "symbol route. Finite-nmax Galerkin in the quantum code is a "
            "different cutoff of the same infinite-space operator."
        ),
    }


def static_force_identity(geo, k):
    model = sh.build_shared(geo, R0, k, 1, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
    ek, emk, masses = ds.ground_pair_neel(geo, k, 1)
    j0 = j_total(ek, emk, model, 0.0, 0.0)
    fk_groups = sh.fk_from_groups(model.f_k, model.q_units, ek)
    fmk_groups = sh.fk_from_groups(model.f_mk, model.q_units, emk)
    f_from_j = -model.g * float(np.dot(model.e, j0.real))
    f_from_f = complex(fk_groups + fmk_groups)
    return {
        "electron": masses,
        "J_s0": [pf.cpack(z) for z in j0],
        "F_from_minus_g_e_dot_J": float(f_from_j),
        "F_from_i_q_Fgroups": pf.cpack(f_from_f),
        "abs_F_minus_i_q_sum": float(abs(f_from_j - f_from_f)),
        "el_dim_each": 36,
        "product_not_1296": True,
        "joint_quantum_dim_nmax8": 11664,
        "n_group": int(model.q_units.size),
    }


def two_step_backaction(geo, k):
    """Two midpoint steps: kick vs frozen-X. Proves back-action is installed."""
    model = sh.build_shared(geo, R0, k, 1, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
    ek0, emk0, _ = ds.ground_pair_neel(geo, k, 1)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    dt = env.t_ramp / N_STEPS_LOCK
    X0, P0 = analytic_xp("coherent")["mean_X"], analytic_xp("coherent")["mean_P"]
    s1 = env.s(0.5 * dt)
    ek_k, emk_k, Xk, Pk, jk = step_ehrenfest(ek0.copy(), emk0.copy(), X0, P0, model, s1, dt, kick=True)
    s2 = env.s(1.5 * dt)
    ek_k, emk_k, Xk, Pk, jk = step_ehrenfest(ek_k, emk_k, Xk, Pk, model, s2, dt, kick=True)
    ek_f, emk_f, Xf, Pf, jf = step_ehrenfest(ek0.copy(), emk0.copy(), X0, P0, model, s1, dt, kick=False)
    ek_f, emk_f, Xf, Pf, jf = step_ehrenfest(ek_f, emk_f, Xf, Pf, model, s2, dt, kick=False)
    return {
        "dt": dt,
        "kick": {"X": Xk, "P": Pk, "J": [pf.cpack(z) for z in jk],
                 "norm_k": float(np.linalg.norm(ek_k)),
                 "norm_mk": float(np.linalg.norm(emk_k))},
        "frozen_no_kick": {"X": Xf, "P": Pf, "J": [pf.cpack(z) for z in jf],
                           "norm_k": float(np.linalg.norm(ek_f)),
                           "norm_mk": float(np.linalg.norm(emk_f))},
        "abs_dP_kick_minus_frozen": abs(Pk - Pf),
        "abs_dX_kick_minus_frozen": abs(Xk - Xf),
        "norms_stay_one_kick": abs(np.linalg.norm(ek_k) - 1.0) + abs(np.linalg.norm(emk_k) - 1.0),
    }


# ---------------------------------------------------------------- one trajectory + cost
def one_ehrenfest_ramp(geo, k, clock, progress, n_steps: int, sample_every: int):
    model = sh.build_shared(geo, R0, k, 1, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
    ek, emk, masses = ds.ground_pair_neel(geo, k, 1)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    dt = env.t_ramp / n_steps
    X, P = analytic_xp("coherent")["mean_X"], analytic_xp("coherent")["mean_P"]
    samples = [snapshot_traj(ek, emk, X, P, model, 0.0, 0.0)]
    _progress(progress, f"single_traj start steps={n_steps} dt={dt:.6g} dim_el=36+36")
    max_last = 0.0
    for j in range(n_steps):
        if clock is not None:
            clock.check()
        s = env.s((j + 0.5) * dt)
        ek, emk, X, P, jvec = step_ehrenfest(ek, emk, X, P, model, s, dt, kick=True)
        if (j % sample_every == 0) or j == n_steps - 1:
            snap = snapshot_traj(ek, emk, X, P, model, env.s((j + 1) * dt), (j + 1) * dt)
            samples.append(snap)
        max_last = max(max_last, abs(np.linalg.norm(ek) - 1.0), abs(np.linalg.norm(emk) - 1.0))
    end = samples[-1]
    _progress(progress, f"single_traj done wall={clock.elapsed():.3f}s X={X:.6g} nW={end['n_Weyl']:.6g}")
    return {
        "kind": "coherent_mean_field_order1",
        "neel": 1,
        "electron": masses,
        "n_steps": n_steps,
        "dt": dt,
        "t_ramp": env.t_ramp,
        "kick": True,
        "n_samples": len(samples),
        "initial": samples[0],
        "after_ramp": end,
        "max_abs_norm_minus_1": float(max_last),
        "delta_X": float(end["X"] - samples[0]["X"]),
        "delta_P": float(end["P"] - samples[0]["P"]),
        "delta_n_Weyl": float(end["n_Weyl"] - samples[0]["n_Weyl"]),
        "time_series": samples,
        "wall_seconds": clock.elapsed() if clock is not None else None,
        "not_a_Calpha_vs_DS_compare": True,
        "not_J_odd": True,
    }


def cost_estimate(geo, k, n_probe: int = COST_PROBE_STEPS):
    model = sh.build_shared(geo, R0, k, 1, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
    ek, emk, _ = ds.ground_pair_neel(geo, k, 1)
    env = dyn.Envelope(pf.OMEGA0, ramp_cycles=1.0, flat_cycles=0.0)
    dt = env.t_ramp / N_STEPS_LOCK
    X, P = analytic_xp("coherent")["mean_X"], analytic_xp("coherent")["mean_P"]
    t0 = time.perf_counter()
    for j in range(n_probe):
        s = env.s((j + 0.5) * dt)
        ek, emk, X, P, _ = step_ehrenfest(ek, emk, X, P, model, s, dt, kick=True)
    elapsed = time.perf_counter() - t0
    step = elapsed / max(n_probe, 1)
    per_ramp = step * N_STEPS_LOCK
    table = []
    for m in (1, 3, 5, 7):
        n_node = m * m
        # 2 kinds (Cα, DS) × 2 Néel, paired nodes, no extra RN.
        n_traj = 2 * 2 * n_node
        table.append({
            "gh_order": m,
            "n_nodes_per_kind": n_node,
            "n_ehrenfest_ramps": n_traj,
            "est_serial_s": per_ramp * n_traj,
            "est_4way_s": per_ramp * n_traj / 4.0,
            "role": {1: "mean-field negative control",
                     3: "first fluctuation compare",
                     5: "quadrature convergence",
                     7: "quadrature convergence"}[m],
        })
    return {
        "probe_steps": n_probe,
        "probe_seconds": elapsed,
        "seconds_per_step": step,
        "seconds_per_rising_edge": per_ramp,
        "locked_n_steps": N_STEPS_LOCK,
        "dt": dt,
        "quantum_nmax8_parallel_class_s": 180.0,
        "quantum_dim": 11664,
        "mtef_el_dim": "36+36",
        "later_ensemble_table": table,
        "this_round_runs_ensemble": False,
        "Cn_this_round": False,
    }


def run(progress=None):
    hashes_before = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    geo = build_geometry()
    k = geo.k_cart(sh.K_FRAC)
    clock = dyn.WallClock(WALL_CAP_S)
    _progress(progress, "gate static start")
    corr = static_correspondence()
    wig = static_wigner_and_quadrature()
    model0 = sh.build_shared(geo, R0, k, 1, n_max=0, omega0=pf.OMEGA0, g=pf.G_ABS)
    pei = static_peierls_weyl(model0)
    force = static_force_identity(geo, k)
    two = two_step_backaction(geo, k)
    cost = cost_estimate(geo, k)
    _progress(progress, f"static done; cost_step={cost['seconds_per_step']:.4e}s")
    traj = one_ehrenfest_ramp(geo, k, clock, progress, N_STEPS_LOCK, SAMPLE_EVERY)
    hashes_after = {str(p.as_posix()): sha256_file(p) for p in OLD_ARCHIVES if p.is_file()}
    report = {
        "record_date": RECORD_DATE,
        "model_id": MODEL_ID,
        "claim_level": "wigner_mtef_mapping_static_and_one_traj_not_ensemble_not_HHG",
        "odd_current_precision_campaign_closed": True,
        "vacuum_squeeze_branch_closed": True,
        "did_not": [
            "rewrite any 20261026-20261107 json",
            "G1", "H2", "HHG spectra", "vacuum",
            "theta=pi/2", "theta=pi", "Cn",
            "nmax scan", "full GH ensemble", "output photon statistics",
        ],
        "definition": {
            "H": "omega0 n + He(k, X; N) + He(-k, X; N)",
            "X": "a + a_dagger",
            "P": "-i (a - a_dagger)",
            "envelope": "1-cycle sin^2 rising edge, stop at s=1, no plateau",
            "g": pf.G_ABS, "omega0": pf.OMEGA0,
            "alpha": 0.5, "r": R_SQ, "theta": THETA,
            "k_fractional": list(sh.K_FRAC),
            "n_steps_locked_to_20261102": N_STEPS_LOCK,
        },
        "correspondence": corr,
        "wigner_quadrature": wig,
        "peierls_weyl": pei,
        "force_identity": force,
        "two_step_backaction": two,
        "single_trajectory": {k: traj[k] for k in traj if k != "time_series"},
        "single_trajectory_time_series": traj["time_series"],
        "cost": cost,
        "later_compare_protocol": {
            "pairing": "same (xi,eta) for Calpha, DS, +N, -N",
            "J_odd": "(J_+N - J_-N)/2",
            "Delta_J_odd": "J_odd_DS - J_odd_Calpha",
            "order_1_must_give_zero_Delta": True,
            "first_fair_order": 3,
            "if_mismatch_check_first": [
                "quadrature order", "CRN nodes", "X vs W_sq convention",
                "back-action kick on/off", "time-grid alignment with 20261102",
            ],
            "do_not_call_mismatch_quantum_advantage": True,
        },
        "old_archive_hashes_before": hashes_before,
        "old_archive_hashes_unchanged": bool(hashes_before == hashes_after),
        "wall_seconds": clock.elapsed(),
        "status": "ok",
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress")
    ap.add_argument("--out", default=str(OUT_DIR / "map.json"))
    a = ap.parse_args()
    progress = Path(a.progress) if a.progress else OUT_DIR / "progress.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run(progress)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(jsonable(report), fh, ensure_ascii=False, indent=1)
    _progress(progress, f"wrote {out} status={report['status']}")
    print(f"status={report['status']} wall={report['wall_seconds']:.3f}s out={out}")


if __name__ == "__main__":
    main()

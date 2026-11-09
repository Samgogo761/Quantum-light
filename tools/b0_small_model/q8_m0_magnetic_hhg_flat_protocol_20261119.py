#!/usr/bin/env python3
"""20261119: flat 2+8+2 drive protocol identities and cost quote. No electronic propagate."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
from q8_m0_model import R0, build_geometry, sha256_file

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/magnetic_hhg_flat_protocol_20261119"
MU = -0.009010620372198276
NTPC = 400
A_STAR = 0.8
# 20261116 measured classical Γ48 eight-cycle jobs (sheet current only).
WALL_8CYC_S = {"mean": 72.43, "jobs": [72.64, 72.35, 72.30], "source": "gamma96_fill_20261116"}
HIST_CJJ_8CYC_S = 140.5


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_sin2_kappa(t, geo, prot, sign=+1):
    t = np.asarray(t, dtype=float)
    a0 = prot.a0_dimless / geo.a_nn_mean_ang
    inside = (t >= 0.0) & (t <= prot.t_pulse)
    env = np.where(inside, np.sin(np.pi * t / prot.t_pulse) ** 2, 0.0)
    scal = sign * a0 * env * np.cos(prot.omega0 * t + prot.cep)
    return scal[..., None] * cl.pol_vector(prot, geo)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def rel_res(num, den):
    d = rms(den)
    n = rms(num)
    return n / d if d > 0 else None


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    old = cl.p1_a2_protocol(A_STAR)
    flat = cl.p1_a2_flat_282_protocol(A_STAR)
    dt = flat.period / NTPC
    nt = flat.n_cycles * NTPC
    times = np.arange(nt + 1) * dt
    t0p = flat.period

    t_old = np.linspace(0.0, old.t_pulse, old.n_cycles * NTPC + 1)
    k_old = cl.kappa_shift(t_old, geo, old, sign=+1)
    k_ref = independent_sin2_kappa(t_old, geo, old, sign=+1)
    old_maxabs = float(np.max(np.abs(k_old - k_ref)))

    env, denv = cl.envelope_and_derivative(times, flat)
    a = cl.kappa_shift(times, geo, flat, sign=+1)
    e = cl.e_field(times, geo, flat, sign=+1)
    i2 = int(round(2.0 * t0p / dt))
    i10 = int(round(10.0 * t0p / dt))
    n_half = int(round(0.5 * t0p / dt))
    if abs(n_half * dt - 0.5 * t0p) > 1e-12:
        raise RuntimeError("half period is not an integer number of samples")

    c1 = {
        "env_at_0": float(env[0]),
        "env_at_2T": float(env[i2]),
        "env_at_10T": float(env[i10]),
        "env_at_end": float(env[-1]),
        "denv_at_0": float(denv[0]),
        "denv_at_2T": float(denv[i2]),
        "denv_at_10T": float(denv[i10]),
        "denv_at_end": float(denv[-1]),
        "max_abs_denv_on_plateau": float(np.max(np.abs(denv[i2 : i10 + 1]))),
        "max_abs_env_minus_1_on_plateau": float(np.max(np.abs(env[i2 : i10 + 1] - 1.0))),
    }

    sl = slice(i2, i10 - n_half + 1)
    sls = slice(i2 + n_half, i10 + 1)
    rA = a[sls] + a[sl]
    plateau_rel = rel_res(np.linalg.norm(rA, axis=1), np.linalg.norm(a[sl], axis=1))
    plateau_maxabs = float(np.max(np.abs(rA)))

    # Numerical C¹: one-sided derivatives of env at the two junctions.
    d_left_2 = (env[i2] - env[i2 - 1]) / dt
    d_right_2 = (env[i2 + 1] - env[i2]) / dt
    d_left_10 = (env[i10] - env[i10 - 1]) / dt
    d_right_10 = (env[i10 + 1] - env[i10]) / dt

    e_jump_2 = float(np.max(np.abs(e[i2] - e[i2 - 1])))
    e_jump_10 = float(np.max(np.abs(e[i10] - e[i10 - 1])))

    # Catch a wrong analytic d(env)/dt against a centered difference of env.
    denv_fd = np.empty_like(env)
    denv_fd[0] = (env[1] - env[0]) / dt
    denv_fd[-1] = (env[-1] - env[-2]) / dt
    denv_fd[1:-1] = (env[2:] - env[:-2]) / (2.0 * dt)
    denv_err = np.abs(denv_fd - denv)
    # f is C¹ but not C²: centered FD straddling u=2 and u=10 is O(dt).
    # One-sided FD at the ends is O(dt) because f ~ u² near 0 and 12.
    interior = np.zeros(env.shape, dtype=bool)
    interior[2 : max(2, i2 - 2)] = True
    interior[i2 + 2 : max(i2 + 2, i10 - 2)] = True
    interior[i10 + 2 : max(i10 + 2, len(env) - 2)] = True
    denv_fd_maxabs = float(np.max(denv_err))
    denv_fd_maxabs_smooth = float(np.max(denv_err[interior])) if np.any(interior) else denv_fd_maxabs
    denv_fd_rel = denv_fd_maxabs / max(float(np.max(np.abs(denv))), 1e-30)
    denv_fd_rel_smooth = denv_fd_maxabs_smooth / max(float(np.max(np.abs(denv))), 1e-30)

    scale = 12.0 / 8.0
    per_job = WALL_8CYC_S["mean"] * scale
    quote = {
        "ntpc": NTPC,
        "n_cycles": 12,
        "n_time_steps": int(nt),
        "n_samples": int(nt + 1),
        "dt": float(dt),
        "period": float(t0p),
        "t_pulse": float(flat.t_pulse),
        "nk": 48,
        "n_k": 48 * 48,
        "mu": MU,
        "params_label": "R0",
        "a_star": A_STAR,
        "scale_from_8cycle": scale,
        "anchor_8cycle_s": WALL_8CYC_S,
        "per_job_classical_s": per_job,
        "stage_D1_Gamma48_pmN_s": 2.0 * per_job,
        "stage_D1_plus_save_headroom_s": 2.0 * per_job + 30.0,
        "stage_D2_shift48_extra_s": 2.0 * per_job,
        "stage_D1_plus_D2_s": 4.0 * per_job,
        "hist_cjj_per_job_s": HIST_CJJ_8CYC_S * scale,
        "local_10min_budget_s": 600.0,
        "D1_under_10min": (2.0 * per_job + 30.0) < 600.0,
        "D1_D2_under_10min": (4.0 * per_job + 60.0) < 600.0,
        "server_required_for_D1": False,
        "includes_cjj": False,
        "includes_qed": False,
        "propagate_now": False,
    }

    payload = {
        "task": "20261119_flat_2_8_2_protocol_cost",
        "propagate_now": False,
        "old_p1_kappa_maxabs_vs_independent_sin2": old_maxabs,
        "old_protocol": old.__dict__,
        "flat_protocol": flat.__dict__,
        "protocol_id": flat.protocol_id,
        "c1_envelope": c1,
        "one_sided_denv": {
            "u2_left": float(d_left_2),
            "u2_right": float(d_right_2),
            "u10_left": float(d_left_10),
            "u10_right": float(d_right_10),
        },
        "e_field_sample_jumps": {"u2": e_jump_2, "u10": e_jump_10},
        "denv_vs_env_fd": {
            "maxabs": denv_fd_maxabs,
            "rel": denv_fd_rel,
            "maxabs_excluding_c2_kinks": denv_fd_maxabs_smooth,
            "rel_excluding_c2_kinks": denv_fd_rel_smooth,
        },
        "plateau_A_halfcycle": {
            "n_half_samples": n_half,
            "rel_rms": plateau_rel,
            "maxabs": plateau_maxabs,
            "window": "[2 T0, 10 T0]",
        },
        "main_fourier_window": {
            "kind": "rectangular_integer_cycles_on_plateau_center",
            "M": 6,
            "t_start_cycles": 3.0,
            "t_end_cycles": 9.0,
            "formula": "J_q = 1/(M T0) ∫_{ta}^{ta+M T0} J(t) exp(i q ω0 t) dt",
            "not_the_old_sin2_full_pulse_window": True,
        },
        "control_windows": [
            {"M": 8, "t_start_cycles": 2.0, "t_end_cycles": 10.0, "kind": "full_plateau_rect"},
            {"kind": "smoothed_sensitivity_only", "not_acceptance": True},
        ],
        "cost": quote,
        "frozen": {
            "hamiltonian": "h8/R0",
            "mu": MU,
            "polarization": "a2",
            "cep": 0.0,
            "ntpc": NTPC,
            "field_sign": 1,
            "neel": [1, -1],
            "no_cjj": True,
            "no_qed": True,
            "no_gamma192": True,
        },
        "source_hashes": {
            "classical": sha256_file(cl.CLASSICAL_MODULE),
            "script": sha256_path(Path(__file__)),
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    gates = {
        "old_p1_unchanged": old_maxabs < 1e-15,
        "env_c1_endpoints": max(abs(c1["env_at_0"]), abs(c1["env_at_end"]), abs(c1["denv_at_0"]), abs(c1["denv_at_end"])) < 1e-12,
        "env_unit_plateau": c1["max_abs_env_minus_1_on_plateau"] < 1e-12,
        "denv_zero_plateau": c1["max_abs_denv_on_plateau"] < 1e-12,
        "A_halfcycle_rel": plateau_rel is not None and plateau_rel < 1e-12,
        "A_halfcycle_maxabs": plateau_maxabs < 1e-14,
        "denv_matches_env_fd": denv_fd_maxabs_smooth < 1e-6,
    }
    payload["gates"] = gates
    payload["all_gates"] = all(gates.values())
    (OUT / "protocol_cost.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"all_gates": payload["all_gates"], "gates": gates, "cost": quote, "out": str(OUT)}, indent=2))
    if not payload["all_gates"]:
        raise SystemExit("flat-protocol identity gates failed")


if __name__ == "__main__":
    main()

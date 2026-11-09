#!/usr/bin/env python3
"""档 1：同协议经典 P1 ±N，H2 两偏振电流源投影 β_src。

硬边界
------
- 不改 Hamiltonian / P1 / 初态 / φ / γ / η / D / k 网格。
- n_cycles 保持 8；探测记录到 9 周期。第九周期外加驱动必须严格为零。
- β 标为 beta_src，不是完整 ⟨b_out⟩，禁止 |β_src|² 与量子 n 比绝对大小。
- 仅两条经典轨道 + H2 后处理。不跑档 2 / K' / 新 U,V。
- 不覆盖归档 JSON。production_G1_pass=false。G1 冻结。准玻色 pending。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from q8_m0_classical import (
    e_field,
    initial_rho,
    k_grid,
    kappa_shift,
    p1_a2_protocol,
    pol_vector,
    trace_current,
)
from q8_m0_gsheet import gamma_self
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import R0, build_geometry, current_ops, h8, h8_and_current
from q8_m0_s1_kernel import parallel_perp_basis

OUTDIR = ROOT / "results/q_line/h2_class_pmN_run_20261025"
OUTJSON = OUTDIR / "h2_class_pmN.json"
PROGRESS = OUTDIR / "progress.txt"
NPZ = OUTDIR / "h2_class_pmN_traj.npz"

A_STAR = 0.8
NTPC = 1024
N_TAIL_CYCLES = 1
MU = -0.009010620372198276
FROZEN_GAMMA = 0.001080789276
NK = 12
GRID_SHIFT = 0.0
FIELD_SIGN = +1
N_DIAG = 64
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def cpack(z):
    z = complex(z)
    return {
        "re": float(z.real),
        "im": float(z.imag),
        "abs": float(abs(z)),
        "arg": float(np.angle(z)),
    }


def eta_of(gamma):
    return float(np.sqrt(float(gamma) / (2.0 * np.pi)))


def kernel_at_times(gamma, prot, t_mid):
    """K(t)=i η ∫ dω φ(ω)/√ω exp(+iω t), η=sqrt(γ/2π). 与量子探测核同一公式。"""
    eta = eta_of(gamma)
    w0 = float(prot.omega0)
    tp = float(prot.t_pulse)
    om = omega_grid(w0, tp)
    om = om[om > 0.0]
    raw = phi_unnormalized(om, 2.0 * w0, tp)
    phi, nrm = normalize_phi(om, raw)
    t_mid = np.asarray(t_mid, float)
    weight = phi / np.sqrt(om)
    k = np.empty(t_mid.size, complex)
    batch = 64
    for s in range(0, t_mid.size, batch):
        sl = slice(s, min(s + batch, t_mid.size))
        phase = np.exp(1j * om[:, None] * t_mid[sl][None, :])
        k[sl] = 1j * eta * np.trapezoid(weight[:, None] * phase, om, axis=0)
    return k, float(nrm), eta, om, phi


def preflight(geo, prot, gamma, e_par, e_perp):
    """三项启动前检查。失败则中止，不传播。"""
    dt = prot.period / NTPC
    n_pulse = int(prot.n_cycles) * NTPC
    nstep = n_pulse + N_TAIL_CYCLES * NTPC
    t_mid = (np.arange(nstep) + 0.5) * dt
    t_nodes = np.arange(nstep + 1) * dt
    pol = pol_vector(prot, geo)

    if int(prot.n_cycles) != 8:
        raise RuntimeError(f"n_cycles must stay 8, got {prot.n_cycles}")
    if abs(float(prot.a0_dimless) - A_STAR) > 1e-15:
        raise RuntimeError("A* mismatch")
    if prot.polarization != "a2":
        raise RuntimeError("drive is not p1_a2")
    if abs(float(gamma) - FROZEN_GAMMA) > 1e-12:
        raise RuntimeError(f"gamma_self {gamma} != frozen {FROZEN_GAMMA}")

    tail = t_nodes > prot.t_pulse + 1e-15 * prot.period
    if not np.any(tail):
        raise RuntimeError("no tail samples after t_pulse")
    kap_tail = kappa_shift(t_nodes[tail], geo, prot, FIELD_SIGN)
    e_tail = e_field(t_nodes[tail], geo, prot, FIELD_SIGN)
    kap_max = float(np.max(np.abs(kap_tail)))
    e_max = float(np.max(np.abs(e_tail)))
    if kap_max > 1e-15 or e_max > 1e-15:
        raise RuntimeError(f"drive not off after cycle 8: |A|={kap_max}, |E|={e_max}")

    t_dense = np.linspace(prot.t_pulse, t_nodes[-1], 4001)
    t_after = t_dense[t_dense > prot.t_pulse]
    kap_d = float(np.max(np.abs(kappa_shift(t_after, geo, prot, FIELD_SIGN))))
    e_d = float(np.max(np.abs(e_field(t_after, geo, prot, FIELD_SIGN))))
    if kap_d > 1e-15 or e_d > 1e-15:
        raise RuntimeError(f"dense tail drive leak |A|={kap_d} |E|={e_d}")

    k, nrm, eta, om, phi = kernel_at_times(gamma, prot, t_mid)
    a_nodes = kappa_shift(t_nodes, geo, prot, FIELD_SIGN)
    e_nodes = e_field(t_nodes, geo, prot, FIELD_SIGN)
    a_mid = 0.5 * (a_nodes[:-1] + a_nodes[1:])
    e_mid = 0.5 * (e_nodes[:-1] + e_nodes[1:])
    leak_a = {
        "parallel": cpack(np.sum(k * (a_mid @ e_par) * dt)),
        "perp": cpack(np.sum(k * (a_mid @ e_perp) * dt)),
    }
    leak_e = {
        "parallel": cpack(np.sum(k * (e_mid @ e_par) * dt)),
        "perp": cpack(np.sum(k * (e_mid @ e_perp) * dt)),
    }
    i0_w = np.argmin(np.abs(om - prot.omega0))
    i2_w = np.argmin(np.abs(om - 2.0 * prot.omega0))

    kf, kc, w = k_grid(geo, NK, shift=GRID_SHIFT)
    i_eq = {}
    for neel in (+1, -1):
        rho, _ = initial_rho(geo, R0, kc, neel, MU)
        j = current_ops(geo, R0, kc, neel)
        i_cell = w @ trace_current(rho, j)
        i_eq[neel] = {
            "cartesian": [float(i_cell[0]), float(i_cell[1])],
            "parallel": float(i_cell @ e_par),
            "perp": float(i_cell @ e_perp),
        }
    k_int = complex(np.sum(k * dt))
    beta_eq = {}
    for neel in (+1, -1):
        beta_eq[str(neel)] = {
            "parallel": cpack(i_eq[neel]["parallel"] * k_int),
            "perp": cpack(i_eq[neel]["perp"] * k_int),
        }

    checks = {
        "check1_beta_is_src_not_bout": {
            "object": "beta_src = int K(t) e_mu · I_cell(t) dt on midpoint grid",
            "not_object": "full <b_out>; |beta_src|^2 vs quantum n",
            "direct_free_field": (
                "classical P1 is local Peierls A_cl at omega0, not an incoming H2 packet; "
                "quantum <a_in>=0 so U0 transmission of vacuum does not give a classical H2 displacement"
            ),
            "pump_off": (
                "A*=0 leaves rho at the occupied projector; I_cell is constant I_eq; "
                "beta_eq = I_eq_mu * int K dt is the no-pump reference, not a second 9-cycle run"
            ),
            "incident_H2_leakage": (
                "drive is along e_parallel; geometric overlap onto e_perp is ~0; "
                "H2 packet is centered at 2*omega0 so parallel leakage is a diagnostic overlap, not subtracted"
            ),
            "do_not_subtract_leak_from_beta_src": True,
            "leak_A_same_units_as_beta": False,
            "leak_E_same_units_as_beta": False,
            "leak_A": leak_a,
            "leak_E": leak_e,
            "abs_phi_at_omega0": float(np.abs(phi[i0_w])),
            "abs_phi_at_2omega0": float(np.abs(phi[i2_w])),
            "I_eq": {str(k): v for k, v in i_eq.items()},
            "beta_eq_src": beta_eq,
            "pass": True,
        },
        "check2_ninth_cycle_drive_off": {
            "n_cycles": int(prot.n_cycles),
            "n_tail_cycles": int(N_TAIL_CYCLES),
            "nstep": int(nstep),
            "t_pulse": float(prot.t_pulse),
            "T_win": float(nstep * dt),
            "dt": float(dt),
            "max_abs_kappa_tail_nodes": kap_max,
            "max_abs_E_tail_nodes": e_max,
            "max_abs_kappa_tail_dense": kap_d,
            "max_abs_E_tail_dense": e_d,
            "electronic_free_evolution_kept": True,
            "pass": True,
        },
        "check3_full_complex": {
            "will_record": [
                "beta_src_parallel",
                "beta_src_perp",
                "abs",
                "arg",
                "Delta",
                "beta_even",
                "beta_odd",
            ],
            "no_modulus_only": True,
            "no_preset_odd_flip": True,
            "pass": True,
        },
        "kernel": {
            "formula": "K(t)=i*eta*int d omega phi(omega)/sqrt(omega)*exp(+i*omega*t)",
            "eta": float(eta),
            "normalizer": float(nrm),
            "n_omega": int(om.size),
            "n_t_mid": int(t_mid.size),
        },
        "geometry": {
            "e_par": e_par.tolist(),
            "e_perp": e_perp.tolist(),
            "drive_pol": pol.tolist(),
            "par_dot_perp": float(e_par @ e_perp),
            "par_dot_drive": float(e_par @ pol),
            "perp_dot_drive": float(e_perp @ pol),
        },
    }
    if abs(e_par @ e_perp) > 1e-12:
        raise RuntimeError("polarization basis not orthogonal")
    if abs(e_par @ pol - 1.0) > 1e-12:
        raise RuntimeError("drive is not along e_parallel")
    if abs(e_perp @ pol) > 1e-12:
        raise RuntimeError("drive leaks into e_perp geometrically")
    return t_mid, dt, nstep, n_pulse, k, checks


def propagate_record(geo, prot, *, neel, nstep, dt):
    kf, kc, w = k_grid(geo, NK, shift=GRID_SHIFT)
    rho, e0 = initial_rho(geo, R0, kc, neel, MU)
    nt = int(nstep)
    times = np.arange(nt + 1) * dt
    i_t = np.zeros((nt + 1, 2))
    diag_every = max(1, nt // N_DIAG)
    t_wall = time.perf_counter()
    for n in range(nt + 1):
        t = times[n]
        kap = kc + kappa_shift(t, geo, prot, FIELD_SIGN)
        need_h = n % diag_every == 0 or n == nt
        if need_h:
            _h, j = h8_and_current(geo, R0, kap, neel)
        else:
            j = current_ops(geo, R0, kap, neel)
        i_t[n] = w @ trace_current(rho, j)
        if n == nt:
            break
        kmid = kc + kappa_shift(t + 0.5 * dt, geo, prot, FIELD_SIGN)
        hm = h8(geo, R0, kmid, neel)
        em, wm = np.linalg.eigh(hm)
        ph = np.exp(-1j * em * dt)
        u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
        if n % NTPC == 0:
            prog(f"neel={neel:+d} step {n}/{nt} t={t:.4f} elapsed {time.perf_counter()-t_wall:.1f}s")
    wall = time.perf_counter() - t_wall
    return {
        "times": times,
        "I_cell": i_t,
        "wall_seconds": wall,
        "e0_band_ranges": [[float(e0[:, n].min()), float(e0[:, n].max())] for n in range(8)],
        "kf": kf,
        "w_k": w,
    }


def beta_from_I(k, i_t, e_par, e_perp, dt, n_pulse):
    i_mid = 0.5 * (i_t[:-1] + i_t[1:])
    i_par = i_mid @ e_par
    i_perp = i_mid @ e_perp
    b_par = complex(np.sum(k * i_par * dt))
    b_perp = complex(np.sum(k * i_perp * dt))
    pulse = slice(0, n_pulse)
    tail = slice(n_pulse, None)
    return {
        "parallel": cpack(b_par),
        "perp": cpack(b_perp),
        "parallel_pulse_only": cpack(complex(np.sum(k[pulse] * i_par[pulse] * dt))),
        "perp_pulse_only": cpack(complex(np.sum(k[pulse] * i_perp[pulse] * dt))),
        "parallel_tail_only": cpack(complex(np.sum(k[tail] * i_par[tail] * dt))),
        "perp_tail_only": cpack(complex(np.sum(k[tail] * i_perp[tail] * dt))),
        "I_cell_t0_par": float(i_t[0] @ e_par),
        "I_cell_t0_perp": float(i_t[0] @ e_perp),
        "I_cell_t_pulse_par": float(i_t[n_pulse] @ e_par),
        "I_cell_t_pulse_perp": float(i_t[n_pulse] @ e_perp),
        "I_cell_T_par": float(i_t[-1] @ e_par),
        "I_cell_T_perp": float(i_t[-1] @ e_perp),
    }


def combine_pm(bp, bm):
    out = {}
    for key in ("parallel", "perp"):
        zp = complex(bp[key]["re"], bp[key]["im"])
        zm = complex(bm[key]["re"], bm[key]["im"])
        even = 0.5 * (zp + zm)
        odd = 0.5 * (zp - zm)
        ratio = None
        if abs(zm) > 0.0:
            ratio = cpack(zp / zm)
        out[key] = {
            "plus": bp[key],
            "minus": bm[key],
            "delta": cpack(zp - zm),
            "beta_even": cpack(even),
            "beta_odd": cpack(odd),
            "plus_over_minus": ratio,
            "odd_over_even_abs": float(abs(odd) / abs(even)) if abs(even) > 0.0 else None,
        }
    return out


def main():
    if OUTJSON.exists():
        raise FileExistsError(f"Refuse overwrite of {OUTJSON}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    prog("start preflight")
    geo = build_geometry()
    prot = p1_a2_protocol(A_STAR)
    gamma = float(gamma_self(geo))
    e_par, e_perp = parallel_perp_basis(geo)
    t_mid, dt, nstep, n_pulse, k, checks = preflight(geo, prot, gamma, e_par, e_perp)
    prog("preflight pass; start +N")
    rec_p = propagate_record(geo, prot, neel=+1, nstep=nstep, dt=dt)
    prog(f"+N done wall={rec_p['wall_seconds']:.1f}s; start -N")
    rec_m = propagate_record(geo, prot, neel=-1, nstep=nstep, dt=dt)
    prog(f"-N done wall={rec_m['wall_seconds']:.1f}s; project")
    beta_p = beta_from_I(k, rec_p["I_cell"], e_par, e_perp, dt, n_pulse)
    beta_m = beta_from_I(k, rec_m["I_cell"], e_par, e_perp, dt, n_pulse)
    comb = combine_pm(beta_p, beta_m)
    payload = {
        "task": "H2 same-protocol classical +/-N source projection",
        "date": "2026-10-25",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "g1_frozen": True,
        "no_Kprime": True,
        "no_tranche2": True,
        "no_H3": True,
        "no_two_color": True,
        "not_full_b_out": True,
        "not_photon_number": True,
        "do_not_compare_abs2_to_quantum_n": True,
        "no_preset_SHG_sign_flip": True,
        "protocol": {
            "drive": "p1_a2_protocol(0.8)",
            "A_star": A_STAR,
            "hbar_omega0": float(prot.hbar_omega0),
            "n_cycles_envelope": int(prot.n_cycles),
            "tail_cycles": N_TAIL_CYCLES,
            "ntpc": NTPC,
            "nk": NK,
            "grid_shift": GRID_SHIFT,
            "mu": MU,
            "gamma_self": gamma,
            "eta": eta_of(gamma),
            "field_sign": FIELD_SIGN,
            "nstep": nstep,
            "dt": dt,
            "t_pulse": float(prot.t_pulse),
            "T_win": float(nstep * dt),
        },
        "preflight": checks,
        "beta_src_plus": beta_p,
        "beta_src_minus": beta_m,
        "beta_src_even_odd": comb,
        "wall_seconds": {
            "plus": rec_p["wall_seconds"],
            "minus": rec_m["wall_seconds"],
            "total": elapsed(),
        },
        "e0_band_ranges": {
            "plus": rec_p["e0_band_ranges"],
            "minus": rec_m["e0_band_ranges"],
        },
        "stop": "pause for review; G1 remains frozen; tranche 2 not authorized",
    }
    np.savez_compressed(
        NPZ,
        t_mid=t_mid,
        K=k,
        times_plus=rec_p["times"],
        I_cell_plus=rec_p["I_cell"],
        times_minus=rec_m["times"],
        I_cell_minus=rec_m["I_cell"],
        e_par=e_par,
        e_perp=e_perp,
    )
    OUTJSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    prog(f"wrote {OUTJSON.name} and {NPZ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

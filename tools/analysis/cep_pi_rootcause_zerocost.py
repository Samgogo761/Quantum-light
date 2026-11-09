#!/usr/bin/env python3
"""Zero-cost CEP-π absolute-residual root-cause diagnostics (no V2.1 retune).

Checks (independent of new SBE runs):
  1) manifest antipode residuals: α+ᾱ, I, w, φ+π
  2) production-formula E(t): max_t |E_α(t)+E_{-α}(t)|
  3) disk ±N TB: write-roundtrip vs float64 reconstruct + k-op conjugation
  4) R_J(t) vs dephasing ticks — requires Jt.dat (report UNAVAILABLE if missing)
  5) η_contam = R_CEP,n / |Jo_n|

Does NOT create CEP V2.1. Does NOT retune τ_abs from measured 1e-8.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from audit_full112_gh3_candidates import (
    HARD_ORDERS,
    INTENSITY_RTOL,
    PHASE_TOL,
    TWOPI,
    WEIGHT_TOL,
    amp_xy,
    load_manifest,
    load_node_modes,
    pair_nodes_by_neg_alpha,
    phase_diff,
)
from construct_p_conjugate_tb import (
    build_minus_hr,
    build_minus_rr,
    build_permutation,
    build_wf_metadata,
    dipole_k,
    eigh_at,
    interband_strength,
    neg_R_index,
    parse_final_centers,
    parse_lattice_ang,
    read_tb_blocks,
)


def jones_norm(jx: complex, jy: complex) -> float:
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def wrap_phase(phi: float) -> float:
    return float(phi) % TWOPI


def phase_equals_plus_pi(phi_a: float, phi_b: float, tol: float = PHASE_TOL) -> bool:
    target = wrap_phase(wrap_phase(phi_a) + math.pi)
    return phase_diff(wrap_phase(phi_b), target) <= tol


def jones_residual(
    jx_a: complex, jy_a: complex, jx_b: complex, jy_b: complex
) -> float:
    return jones_norm(jx_a + jx_b, jy_a + jy_b)

# Match src/mod_params.f90
PI = math.pi
C_AU = 137.035999084
ANG_TO_BOHR = 1.0 / 0.529177210903
NM_TO_BOHR = 10.0 * ANG_TO_BOHR
WCM2_TO_AU = 1.0 / 3.50944758e16


def envelope(t: float, t_mid: float, t_pulse: float, etype: int) -> float:
    if etype == 1:
        return math.exp(-2.0 * math.log(2.0) * ((t - t_mid) / (t_pulse * 0.35)) ** 2)
    if etype == 2:
        return math.cos(PI * (t - t_mid) / t_pulse) ** 2 if abs(t - t_mid) <= 0.5 * t_pulse else 0.0
    if etype == 3:
        return math.cos(PI * (t - t_mid) / t_pulse) ** 4 if abs(t - t_mid) <= 0.5 * t_pulse else 0.0
    return 1.0


def generate_field_sample(
    e_peak: float,
    phi0: float,
    *,
    omega0: float,
    dt: float,
    nt: int,
    t_total_1: float,
    env_type: int,
    ellipticity: float,
    theta: float,
    delta_phase: float = 0.0,
) -> np.ndarray:
    """Reproduce mod_laser.generate_field_sample primary pulse (Et only)."""
    # Fortran: t_mid = T_total * 0.5 with T_total == T_total_1 for single pulse.
    t_mid = 0.5 * t_total_1
    t_pulse = t_total_1
    ex = np.array([math.cos(theta), math.sin(theta), 0.0])
    ey = np.array([-ex[1], ex[0], 0.0])
    et = np.zeros((nt, 3), dtype=np.float64)
    for it in range(nt):
        t = it * dt
        f_env = envelope(t, t_mid, t_pulse, env_type)
        ex_t = e_peak * f_env * math.sin(omega0 * t + phi0)
        ey_t = e_peak * ellipticity * f_env * math.sin(omega0 * t + phi0 + delta_phase)
        et[it] = ex_t * ex + ey_t * ey
    return et


def parse_laser_from_nml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {
        "wvl_nm": 3200.0,
        "theta_deg": 0.0,
        "phi_cep_deg": 90.0,
        "ncyc": 4.0,
        "env_type": 2,
        "ellipticity": 0.0,
        "dt": 0.35,
        "delta_phase_deg": 0.0,
        "n_dt_deph": 5,
        "T2_cycles": 0.5,
    }
    for raw in text.splitlines():
        line = raw.split("!")[0].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().rstrip(",").strip()
        if key in out:
            if key in ("env_type", "n_dt_deph"):
                out[key] = int(float(val))
            else:
                out[key] = float(val)
    wvl = float(out["wvl_nm"])
    omega0 = TWOPI * C_AU / (wvl * NM_TO_BOHR)
    t_cycle = TWOPI / omega0
    t_total_1 = float(out["ncyc"]) * t_cycle
    dt = float(out["dt"])
    nt = int(math.ceil(t_total_1 / dt)) + 1
    out.update(
        {
            "omega0": omega0,
            "t_cycle": t_cycle,
            "t_total_1": t_total_1,
            "nt": nt,
            "theta": float(out["theta_deg"]) * PI / 180.0,
            "delta_phase": float(out["delta_phase_deg"]) * PI / 180.0,
        }
    )
    return out


def diagnose_manifest(manifest: list[dict], pairs: list[dict]) -> dict:
    by_id = {int(r["id"]): r for r in manifest}
    rows = []
    max_alpha = max_w = max_i = max_phi = 0.0
    for pr in pairs:
        ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
        a, b = by_id[ia], by_id[ib]
        if pr["kind"] == "self_alpha0":
            d_alpha = abs(complex(float(a["re"]), float(a["im"])))
            d_w = 0.0
            d_i = 0.0
            d_phi = 0.0  # self; φ free
            phi_ok = True
        else:
            d_alpha = abs(
                complex(float(a["re"]) + float(b["re"]), float(a["im"]) + float(b["im"]))
            )
            d_w = abs(float(a["weight"]) - float(b["weight"]))
            ia_, ib_ = float(a["intensity"]), float(b["intensity"])
            d_i = abs(ia_ - ib_) / max(ia_, ib_, 1.0e-300)
            d_phi = phase_diff(
                wrap_phase(float(b["phase"])),
                wrap_phase(wrap_phase(float(a["phase"])) + math.pi),
            )
            # also 0≡2π already via wrap
            phi_ok = phase_equals_plus_pi(float(a["phase"]), float(b["phase"]))
        max_alpha = max(max_alpha, d_alpha)
        max_w = max(max_w, d_w)
        max_i = max(max_i, d_i)
        max_phi = max(max_phi, d_phi)
        rows.append(
            {
                "kind": pr["kind"],
                "id_a": ia,
                "id_b": ib,
                "d_alpha": d_alpha,
                "d_weight": d_w,
                "d_intensity_rel": d_i,
                "d_phi_mod2pi": d_phi,
                "phi_plus_pi_ok": phi_ok,
            }
        )
    status = "PASS"
    fails = []
    if max_w > WEIGHT_TOL:
        status = "FAIL"
        fails.append(f"weight residual {max_w} > {WEIGHT_TOL}")
    if max_i > INTENSITY_RTOL:
        status = "FAIL"
        fails.append(f"intensity rel residual {max_i} > {INTENSITY_RTOL}")
    if max_phi > PHASE_TOL:
        status = "FAIL"
        fails.append(f"phase+π residual {max_phi} > {PHASE_TOL}")
    return {
        "status": status,
        "fails": fails,
        "max_d_alpha": max_alpha,
        "max_d_weight": max_w,
        "max_d_intensity_rel": max_i,
        "max_d_phi_mod2pi": max_phi,
        "pairs": rows,
    }


def diagnose_field(manifest: list[dict], pairs: list[dict], laser: dict) -> dict:
    by_id = {int(r["id"]): r for r in manifest}
    per_pair = []
    max_sum = 0.0
    for pr in pairs:
        ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
        a, b = by_id[ia], by_id[ib]
        ea = math.sqrt(float(a["intensity"]) * WCM2_TO_AU)
        eb = math.sqrt(float(b["intensity"]) * WCM2_TO_AU)
        et_a = generate_field_sample(
            ea,
            float(a["phase"]),
            omega0=laser["omega0"],
            dt=laser["dt"],
            nt=laser["nt"],
            t_total_1=laser["t_total_1"],
            env_type=laser["env_type"],
            ellipticity=laser["ellipticity"],
            theta=laser["theta"],
            delta_phase=laser["delta_phase"],
        )
        et_b = generate_field_sample(
            eb,
            float(b["phase"]),
            omega0=laser["omega0"],
            dt=laser["dt"],
            nt=laser["nt"],
            t_total_1=laser["t_total_1"],
            env_type=laser["env_type"],
            ellipticity=laser["ellipticity"],
            theta=laser["theta"],
            delta_phase=laser["delta_phase"],
        )
        # For self α=0: I=0 → E=0; sum is 0. For antipodes: expect E_a + E_b = 0
        s = np.linalg.norm(et_a + et_b, axis=1)
        m = float(np.max(s))
        ref = float(max(np.max(np.linalg.norm(et_a, axis=1)), np.max(np.linalg.norm(et_b, axis=1)), 1e-300))
        max_sum = max(max_sum, m)
        per_pair.append(
            {
                "kind": pr["kind"],
                "id_a": ia,
                "id_b": ib,
                "max_abs_Esum": m,
                "max_abs_Esum_over_Eref": m / ref,
                "E_peak_a": ea,
                "E_peak_b": eb,
            }
        )
    refs = [max(p["E_peak_a"], p["E_peak_b"]) for p in per_pair]
    eref = max(refs) if refs else 1.0
    rel = max_sum / max(eref, 1e-300)
    # Manifest φ is ~1e-8–1e-9 digits; Esum residual tracking dφ is expected.
    # Only FAIL if far above manifest phase digitization (~1e-8).
    status = "PASS" if rel <= 1.0e-8 else "FAIL"
    return {
        "status": status,
        "max_t_abs_Esum": max_sum,
        "max_t_abs_Esum_over_Eref": rel,
        "tau_field_proxy": rel,
        "nt": laser["nt"],
        "n_dt_deph": laser["n_dt_deph"],
        "pairs": per_pair,
        "note": (
            "Field from production formula; A(t) DC correction does not alter E(t). "
            "Residual should track manifest φ digitization (~1e-9)."
        ),
    }


def hk(irvec, ndegen, h_r, kfrac):
    phase = np.exp(2j * np.pi * (irvec @ np.array(kfrac, dtype=float))) / ndegen
    mat = np.tensordot(phase, h_r, axes=(0, 0))
    return 0.5 * (mat + mat.conj().T)


def vk_from_h(irvec, ndegen, h_r, kfrac, a_cart_bohr):
    """v_a = dH/dk_a using Cartesian derivative via fractional R · a_vec."""
    # k_cart = kfrac · b; ∂/∂k_cart_i H = sum_R i R_cart_i e^{ik·R} H(R)/ndegen
    # R_cart = R_frac · A (A rows = lattice vectors in bohr)
    r_cart = irvec.astype(float) @ a_cart_bohr  # (nr, 3)
    phase = np.exp(2j * np.pi * (irvec @ np.array(kfrac, dtype=float))) / ndegen
    out = []
    for a in range(3):
        mat = np.tensordot(1j * r_cart[:, a] * phase, h_r, axes=(0, 0))
        out.append(0.5 * (mat + mat.conj().T))
    return out


def apply_U(mat, perm, parity):
    """U_{a,P(a)} = eta_a ;  U M U† with indices in WF basis."""
    # (U M U†)_{ab} = eta_a eta_b M_{P(a) P(b)}
    sign = np.outer(parity.astype(float), parity.astype(float))
    return sign * mat[np.ix_(perm, perm)]


def diagnose_tb(
    plus_tb: Path,
    minus_tb: Path,
    plus_win: Path,
    plus_wout: Path,
    *,
    center: np.ndarray,
    nk: int,
    k_stride: int,
    max_k_points: int,
) -> dict:
    lattice = parse_lattice_ang(plus_win)
    c_cart = center @ lattice
    centers_cart = parse_final_centers(plus_wout)
    centers_frac = (centers_cart @ np.linalg.inv(lattice)) % 1.0
    nwann = centers_frac.shape[0]
    meta = build_wf_metadata(nwann)
    perm, diag = build_permutation(centers_frac, meta, center)
    if diag["duplicate_targets"] or diag["non_involutive"]:
        raise SystemExit("Permutation not involution; abort TB diagnose.")

    print("[TB] reading +N ...", flush=True)
    ir_p, nd_p, h_p, r_p = read_tb_blocks(plus_tb, True)
    print("[TB] reading -N disk ...", flush=True)
    ir_m, nd_m, h_m_disk, r_m_disk = read_tb_blocks(minus_tb, True)
    if not np.array_equal(ir_p, ir_m):
        return {"status": "FAIL", "error": "irvec mismatch ±N"}
    if np.max(np.abs(nd_p - nd_m)) > 0:
        return {"status": "FAIL", "error": "ndegen mismatch ±N"}

    neg_idx = neg_R_index(ir_p)
    h_m_exact = build_minus_hr(h_p, perm, meta["parity"], neg_idx)
    r_m_exact = build_minus_rr(r_p, perm, meta["parity"], neg_idx, ir_p, c_cart)

    dH = float(np.max(np.abs(h_m_disk - h_m_exact)))
    dR = float(np.max(np.abs(r_m_disk - r_m_exact)))
    # Typical hop scale for relative view
    h_scale = float(np.max(np.abs(h_m_exact)))
    r_scale = float(np.max(np.abs(r_m_exact)))

    # Sampled E18.8 round-trip floor (full matrix Python format loop is too slow).
    rng = np.random.default_rng(0)
    n_sample = min(20000, h_m_exact.size)

    def sample_e188_err(exact: np.ndarray, disk: np.ndarray) -> dict:
        flat_e = exact.ravel()
        flat_d = disk.ravel()
        idx = rng.choice(flat_e.size, size=n_sample, replace=False)
        err_exact_round = []
        err_disk_round = []
        for i in idx:
            c = flat_e[i]
            re = float(f"{c.real:18.8E}")
            im = float(f"{c.imag:18.8E}")
            rounded = re + 1j * im
            err_exact_round.append(abs(c - rounded))
            err_disk_round.append(abs(flat_d[i] - rounded))
        return {
            "n_sample": n_sample,
            "max_abs_exact_vs_roundtrip": float(max(err_exact_round)),
            "max_abs_disk_vs_roundtrip": float(max(err_disk_round)),
            "median_abs_exact_vs_roundtrip": float(np.median(err_exact_round)),
        }

    h_e188 = sample_e188_err(h_m_exact, h_m_disk)
    r_e188 = sample_e188_err(r_m_exact, r_m_disk)
    dH_e188 = h_e188["max_abs_exact_vs_roundtrip"]
    dR_e188 = r_e188["max_abs_exact_vs_roundtrip"]
    dH_disk_vs_round = h_e188["max_abs_disk_vs_roundtrip"]
    dR_disk_vs_round = r_e188["max_abs_disk_vs_roundtrip"]

    # k-grid operator conjugation using *disk* matrices (solver view)
    a_bohr = lattice * ANG_TO_BOHR
    ks = []
    for ikx in range(0, nk, k_stride):
        for iky in range(0, nk, k_stride):
            ks.append((ikx / nk, iky / nk, 0.0))
            if len(ks) >= max_k_points:
                break
        if len(ks) >= max_k_points:
            break

    max_dH_k = max_dE = max_dR_k = max_dv = 0.0
    max_dS = 0.0
    for kf in ks:
        km = tuple((-x) % 1.0 for x in kf)
        hm = hk(ir_m, nd_m, h_m_disk, kf)
        hp = hk(ir_p, nd_p, h_p, km)
        hp_mapped = apply_U(hp, perm, meta["parity"])
        max_dH_k = max(max_dH_k, float(np.max(np.abs(hm - hp_mapped))))
        em, _ = eigh_at(ir_m, nd_m, h_m_disk, kf)
        ep, _ = eigh_at(ir_p, nd_p, h_p, km)
        max_dE = max(max_dE, float(np.max(np.abs(em - ep))))

        rm = dipole_k(ir_m, nd_m, r_m_disk, kf)
        rp = dipole_k(ir_p, nd_p, r_p, km)
        for a in range(3):
            rp_map = apply_U(rp[:, :, a], perm, meta["parity"])
            # polar: r_{-N}(k) ≈ − U r_{+N}(−k) U†  (+ gauge from 2c on diagonal)
            max_dR_k = max(max_dR_k, float(np.max(np.abs(rm[:, :, a] + rp_map))))

        vm = vk_from_h(ir_m, nd_m, h_m_disk, kf, a_bohr)
        vp = vk_from_h(ir_p, nd_p, h_p, km, a_bohr)
        for a in range(3):
            vp_map = apply_U(vp[a], perm, meta["parity"])
            # v_{-N}(k) + U v_{+N}(−k) U† ≈ 0
            max_dv = max(max_dv, float(np.max(np.abs(vm[a] + vp_map))))

        # gauge-invariant interband dipole strength
        _, vm_e = eigh_at(ir_m, nd_m, h_m_disk, kf)
        _, vp_e = eigh_at(ir_p, nd_p, h_p, km)
        sm = interband_strength(vm_e, rm)
        sp = interband_strength(vp_e, rp)
        denom = np.maximum(sp, 1e-6)
        max_dS = max(max_dS, float(np.max(np.abs(sm - sp) / denom)))

    return {
        "status": "INFO",
        "nwann": nwann,
        "nrpts": int(ir_p.shape[0]),
        "n_k_checked": len(ks),
        "k_stride": k_stride,
        "nk": nk,
        "disk_vs_float64_reconstruct": {
            "max_abs_dH": dH,
            "max_abs_dR": dR,
            "H_scale": h_scale,
            "R_scale": r_scale,
            "max_rel_dH": dH / max(h_scale, 1e-300),
            "max_rel_dR": dR / max(r_scale, 1e-300),
        },
        "E18_8_truncation_floor": {
            "H": h_e188,
            "R": r_e188,
            "max_abs_dH_exact_vs_roundtrip": dH_e188,
            "max_abs_dR_exact_vs_roundtrip": dR_e188,
            "max_abs_dH_disk_vs_roundtrip": dH_disk_vs_round,
            "max_abs_dR_disk_vs_roundtrip": dR_disk_vs_round,
            "note": "If disk≈roundtrip and exact-roundtrip~1e-8·scale, write fmt is the floor.",
        },
        "k_operator_conjugation_disk": {
            "max_abs_H_minus_UHpU": max_dH_k,
            "max_abs_dE_eV": max_dE,
            "max_abs_r_plus_UrU": max_dR_k,
            "max_abs_r_plus_UrU_note": (
                "Large absolute value is expected: includes 2c home-cell gauge on "
                "diagonal; use max_rel_interband_dipole_S for physical dipole check."
            ),
            "max_abs_v_plus_UvU": max_dv,
            "max_rel_interband_dipole_S": max_dS,
        },
        "interpretation_hint": (
            "If disk H matches float64 reconstruct and k-H/v are ~1e-15 while "
            "CEP e_abs~1e-8, TB write truncation is NOT the dominant source; "
            "next suspect is instantaneous-basis T2 dephasing (needs Jt dump or "
            "dephasing-off trajectories)."
        ),
    }


def diagnose_jt(
    plus_dir: Path,
    minus_dir: Path,
    pairs: list[dict],
    laser: dict,
) -> dict:
    """Look for per-node Jt; 27992 BSV path does not save them."""
    candidates = list(plus_dir.glob("Jt*.dat")) + list(plus_dir.glob("**/Jt_node*.dat"))
    if not candidates:
        return {
            "status": "UNAVAILABLE",
            "reason": (
                "No Jt.dat / Jt_node*.dat in 27992 BSV outputs. "
                "main.f90 quantum-light path keeps Jt_sample in memory only; "
                "only HHG_nodes_modes (+ optional HHG_complex_node_0001) are written. "
                "Cannot align R_J(t) with n_dt_deph ticks without a new dump or dephasing-off run."
            ),
            "n_dt_deph": laser["n_dt_deph"],
            "dephasing_tick_predicate": "mod(it, n_dt_deph)==0 in propagate_lg_covariant",
            "next_step_if_needed": (
                "Four-trajectory dephasing-off vs T2=0.5T on one strong antipode pair."
            ),
        }
    return {"status": "FOUND", "files": [str(p) for p in candidates]}


def diagnose_contam(
    modes_p: dict,
    modes_m: dict,
    manifest: list[dict],
    pairs: list[dict],
) -> dict:
    out_h = []
    for h in HARD_ORDERS:
        # A_n and residuals
        amps = []
        residuals = []
        for pr in pairs:
            ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
            mp, mm = modes_p[(ia, h)], modes_m[(ib, h)]
            d = jones_residual(mp["jx"], mp["jy"], mm["jx"], mm["jy"])
            a = max(amp_xy(mp["jx"], mp["jy"]), amp_xy(mm["jx"], mm["jy"]))
            amps.append(a)
            residuals.append(d)
            if pr["kind"] != "self_alpha0":
                mp2, mm2 = modes_p[(ib, h)], modes_m[(ia, h)]
                d2 = jones_residual(mp2["jx"], mp2["jy"], mm2["jx"], mm2["jy"])
                a2 = max(amp_xy(mp2["jx"], mp2["jy"]), amp_xy(mm2["jx"], mm2["jy"]))
                amps.append(a2)
                residuals.append(d2)
        A_n = max(amps) if amps else 0.0
        R_cep = max(residuals) if residuals else 0.0

        # Jo = sum_w 1/2 (J+ - J-) over nodes (same id)
        jox = joy = 0.0 + 0.0j
        for row in manifest:
            nid = int(row["id"])
            w = float(row["weight"])
            mp, mm = modes_p[(nid, h)], modes_m[(nid, h)]
            jox += w * 0.5 * (mp["jx"] - mm["jx"])
            joy += w * 0.5 * (mp["jy"] - mm["jy"])
        jo = jones_norm(jox, joy)
        eta = R_cep / max(jo, 1e-300)
        out_h.append(
            {
                "harmonic": h,
                "R_CEP": R_cep,
                "A_n": A_n,
                "e_abs_max": R_cep / max(A_n, 1e-300),
                "Jo_amp": jo,
                "eta_contam": eta,
                "contam_gate_0p01": eta <= 0.01,
            }
        )
    return {
        "status": "INFO",
        "definition": "eta = max_pair ||J++J-|| / |Jo|; Jo=sum_w 1/2(J+-J-)",
        "harmonics": out_h,
        "any_eta_gt_0p01": any(not h["contam_gate_0p01"] for h in out_h),
    }


def run_state(outroot: Path, stem: str, tb_cfg: dict | None) -> dict:
    plus = outroot / f"{stem}_plusN"
    minus = outroot / f"{stem}_minusN"
    man = load_manifest(plus / "nodes_manifest.input.dat")
    pairs = pair_nodes_by_neg_alpha(man)
    modes_p = load_node_modes(plus / "HHG_nodes_modes.dat")
    modes_m = load_node_modes(minus / "HHG_nodes_modes.dat")
    laser = parse_laser_from_nml(plus / "input.nml")

    rep = {
        "state": stem,
        "manifest": diagnose_manifest(man, pairs),
        "field_Esum": diagnose_field(man, pairs, laser),
        "jt_time_domain": diagnose_jt(plus, minus, pairs, laser),
        "contam": diagnose_contam(modes_p, modes_m, man, pairs),
    }
    if tb_cfg is not None and stem == tb_cfg.get("run_on_state"):
        rep["tb"] = diagnose_tb(
            Path(tb_cfg["plus_tb"]),
            Path(tb_cfg["minus_tb"]),
            Path(tb_cfg["plus_win"]),
            Path(tb_cfg["plus_wout"]),
            center=np.array(tb_cfg["center"], dtype=float),
            nk=int(tb_cfg["nk"]),
            k_stride=int(tb_cfg["k_stride"]),
            max_k_points=int(tb_cfg["max_k_points"]),
        )
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outroot", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument(
        "--states",
        nargs="+",
        default=["sv_r2p5_th000", "sv_r2p5_th180", "sv_r0p5_th180"],
    )
    ap.add_argument("--plus-tb", type=Path, default=None)
    ap.add_argument("--minus-tb", type=Path, default=None)
    ap.add_argument("--plus-win", type=Path, default=None)
    ap.add_argument("--plus-wout", type=Path, default=None)
    ap.add_argument("--center", default="0.499998583,0.333333067,0.372004766")
    ap.add_argument("--nk", type=int, default=20)
    ap.add_argument("--k-stride", type=int, default=5, help="subsample k-grid for speed")
    ap.add_argument("--max-k-points", type=int, default=16)
    ap.add_argument("--tb-on-state", default="sv_r2p5_th000")
    ap.add_argument("--skip-tb", action="store_true")
    args = ap.parse_args()

    tb_cfg = None
    if not args.skip_tb and args.plus_tb and args.minus_tb and args.plus_win and args.plus_wout:
        tb_cfg = {
            "plus_tb": args.plus_tb,
            "minus_tb": args.minus_tb,
            "plus_win": args.plus_win,
            "plus_wout": args.plus_wout,
            "center": [float(x) for x in args.center.split(",")],
            "nk": args.nk,
            "k_stride": args.k_stride,
            "max_k_points": args.max_k_points,
            "run_on_state": args.tb_on_state,
        }

    states = []
    for stem in args.states:
        print(f"=== {stem} ===", flush=True)
        states.append(run_state(args.outroot, stem, tb_cfg))

    report = {
        "gate_note": (
            "CEP_RELATIVE_COVARIANCE=PASS; CEP_ABSOLUTE_EXACTNESS=FAIL. "
            "τ_abs=1e-12 is a preset acceptance threshold, not the measured chunk floor "
            "(chunk V3.1 max_e_abs_vec≈4.8e-15). Residual ~1e-8 is numerical covariance "
            "error (implementation/discretization/dephasing), NOT a physical tolerance."
        ),
        "no_v21": True,
        "gh5": "HOLD",
        "states": states,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""1600 nm / 25 fs M0 BZ statistical production.

Stage 1 accepts the physical pulse, mode normalization and 1D GH quadrature.
Stage 2 compares coherent / BSV / displaced-squeezed inputs at screening
intensities. The kernel stage is a classical two-color mixed derivative of
Néel-odd H2, not a quantum output kernel.

Not QED, not g^(2), not CrI3 unless a later adapter reuses the same field.
Does not modify q8_m0_model.py. Does not start nmax=8/10 Fock QED.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.polynomial.hermite import hermgauss

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from q8_m0_classical import initial_rho, k_grid_xy, pol_vector, trace_current
from q8_m0_model import R0, build_geometry, h8, h8_and_current, sha256_file
from qinput_spectrum_reanalysis_20261028 import packed, spectrum

ROOT = HERE.parents[1]
OUT = ROOT / "results/q_line/q1600_bz_stat_production_20261028"
DOC = ROOT / "docs/04_BSV量子光"

HBAR_EV_FS = 0.6582119568564649
HC_EV_NM = 1239.8419843320025
LAMBDA_NM = 1600.0
PHOTON_EV = HC_EV_NM / LAMBDA_NM
FWHM_I_FS = 25.0
T0_FS = 2.0 * np.pi * HBAR_EV_FS / PHOTON_EV
I0_REF = 1.0e11
NBAR_REF = 75756144588.05614
EPEAK_REF_V_M = 868021098.4381311
QPEAK_REF_AINV = 0.11201699692798193
ALPHA_COH_REF = 275238.34142076963
ALPHA_DS_REF = 246180.65657245475
R_DS_REF = 12.413820921719454
R_BSV_REF = 13.218539877923305
W0_M = 15.0e-6


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cpack(z) -> dict:
    return packed(z)


def protocol(i0_w_cm2: float) -> dict:
    scale = float(i0_w_cm2) / I0_REF
    nbar = NBAR_REF * scale
    epeak = EPEAK_REF_V_M * np.sqrt(scale)
    qpeak = QPEAK_REF_AINV * np.sqrt(scale)
    alpha_coh = np.sqrt(nbar)
    alpha_ds = np.sqrt(0.8 * nbar)
    r_ds = float(np.arcsinh(np.sqrt(0.2 * nbar)))
    r_bsv = float(np.arcsinh(np.sqrt(nbar)))
    e1 = epeak / (2.0 * alpha_coh)
    return {
        "lambda_nm": LAMBDA_NM,
        "photon_eV": float(PHOTON_EV),
        "T0_fs": float(T0_FS),
        "H2_eV": float(2.0 * PHOTON_EV),
        "fwhm_I_fs": FWHM_I_FS,
        "t0_fs": 2.5 * FWHM_I_FS,
        "t_max_fs": 5.0 * FWHM_I_FS,
        "envelope": "gaussian_intensity_FWHM_25fs_no_plateau",
        "polarization": "a2",
        "cep": 0.0,
        "I0_W_cm2": float(i0_w_cm2),
        "nbar": float(nbar),
        "E_peak_coh_V_m": float(epeak),
        "q_peak_coh_Ainv_narrowband": float(qpeak),
        "E1_V_m": float(e1),
        "alpha_coh": float(alpha_coh),
        "alpha_ds": float(alpha_ds),
        "r_ds": r_ds,
        "r_bsv": r_bsv,
        "w0_m": W0_M,
        "note": (
            "E1 is the single-photon field of the defined macroscopic mode and "
            "is independent of I0. DS 80/20 is a theoretical candidate, not an "
            "implemented laboratory protocol. I0 values are screening strengths, "
            "not CrI3 damage thresholds."
        ),
        "ref_check": {
            "nbar_ref_rel": abs(nbar / scale - NBAR_REF) / NBAR_REF if scale else None,
            "alpha_coh_ref_rel": abs(alpha_coh / np.sqrt(scale) - ALPHA_COH_REF) / ALPHA_COH_REF if scale else None,
            "alpha_ds_ref_rel": abs(alpha_ds / np.sqrt(scale) - ALPHA_DS_REF) / ALPHA_DS_REF if scale else None,
            "r_ds_ref_abs": abs(r_ds - R_DS_REF) if abs(scale - 1.0) < 1e-12 else None,
            "r_bsv_ref_abs": abs(r_bsv - R_BSV_REF) if abs(scale - 1.0) < 1e-12 else None,
        },
    }


def gaussian_env(t_fs: np.ndarray, t0_fs: float) -> np.ndarray:
    sigma_i = FWHM_I_FS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_e = sigma_i * np.sqrt(2.0)
    return np.exp(-0.5 * ((t_fs - t0_fs) / sigma_e) ** 2)


def build_times(nt_per_cycle: int) -> dict:
    omega = PHOTON_EV
    t_unit_fs = HBAR_EV_FS
    t0_fs = 2.5 * FWHM_I_FS
    t_max_fs = 5.0 * FWHM_I_FS
    n_cycles = t_max_fs / (2.0 * np.pi * t_unit_fs / omega)
    nt = int(np.ceil(n_cycles * nt_per_cycle))
    t = np.linspace(0.0, t_max_fs / t_unit_fs, nt + 1)
    t_fs = t * t_unit_fs
    return {
        "omega": omega,
        "t_unit_fs": t_unit_fs,
        "t0_fs": t0_fs,
        "t_max_fs": t_max_fs,
        "times": t,
        "times_fs": t_fs,
        "env": gaussian_env(t_fs, t0_fs),
        "nt": nt,
        "dt": float(t[1] - t[0]),
        "nt_per_cycle": int(nt_per_cycle),
        "n_cycles": float(n_cycles),
    }


def xp_state(kind: str, proto: dict, theta: float = 0.0) -> dict:
    kind = str(kind)
    if kind == "coherent":
        alpha = complex(proto["alpha_coh"])
        r = 0.0
        th = 0.0
    elif kind == "bsv":
        alpha = 0.0 + 0.0j
        r = float(proto["r_bsv"])
        th = float(theta)
    elif kind == "ds":
        alpha = complex(proto["alpha_ds"])
        r = float(proto["r_ds"])
        th = float(theta)
    else:
        raise ValueError(kind)
    mean_x = 2.0 * alpha.real
    mean_p = 2.0 * alpha.imag
    # θ=0 squeezes X and anti-squeezes P; θ=π squeezes P and anti-squeezes X.
    c2, s2 = np.cos(th), np.sin(th)
    var_x = np.exp(-2.0 * r) * c2 * c2 + np.exp(2.0 * r) * s2 * s2
    var_p = np.exp(2.0 * r) * c2 * c2 + np.exp(-2.0 * r) * s2 * s2
    if abs(th) < 1e-12:
        anti = "P"
        squeezed = "X"
        sigma_anti = float(np.exp(r))
    elif abs(abs(th) - np.pi) < 1e-12:
        anti = "X"
        squeezed = "P"
        sigma_anti = float(np.exp(r))
    else:
        raise ValueError("this production quadrature only ships theta=0 or pi")
    nbar = abs(alpha) ** 2 + np.sinh(r) ** 2
    return {
        "kind": kind,
        "theta": float(th),
        "alpha_re": float(alpha.real),
        "alpha_im": float(alpha.imag),
        "r": float(r),
        "mean_X": float(mean_x),
        "mean_P": float(mean_p),
        "var_X": float(var_x),
        "var_P": float(var_p),
        "anti_axis": anti,
        "squeezed_axis": squeezed,
        "sigma_anti": sigma_anti,
        "mean_n": float(nbar),
        "vacuum_var_convention": "Var(X)=Var(P)=1 for unsqueezed vacuum",
    }


def gh1d_nodes(state: dict, order: int) -> dict:
    if order < 1:
        raise ValueError("GH order >= 1")
    if state["kind"] == "coherent":
        nodes = [{
            "xi": 0.0,
            "weight": 1.0,
            "X": state["mean_X"],
            "P": state["mean_P"],
            "mean_field": True,
        }]
        return {
            "kind": state["kind"],
            "order": 1,
            "n_nodes": 1,
            "weight_sum": 1.0,
            "quad_mean_X": state["mean_X"],
            "quad_mean_P": state["mean_P"],
            "nodes": nodes,
            "note": "macroscopic coherent: vacuum fluctuations omitted",
        }
    xi, w1 = hermgauss(int(order))
    wt = w1 / np.sqrt(np.pi)
    nodes = []
    mx = mp = 0.0
    for x, w in zip(xi, wt):
        if state["anti_axis"] == "P":
            X = state["mean_X"]
            P = state["mean_P"] + state["sigma_anti"] * np.sqrt(2.0) * float(x)
        else:
            X = state["mean_X"] + state["sigma_anti"] * np.sqrt(2.0) * float(x)
            P = state["mean_P"]
        nodes.append({"xi": float(x), "weight": float(w), "X": float(X), "P": float(P), "mean_field": False})
        mx += float(w) * X
        mp += float(w) * P
    vx = sum(n["weight"] * n["X"] ** 2 for n in nodes) - mx * mx
    vp = sum(n["weight"] * n["P"] ** 2 for n in nodes) - mp * mp
    return {
        "kind": state["kind"],
        "order": int(order),
        "n_nodes": len(nodes),
        "weight_sum": float(np.sum(wt)),
        "quad_mean_X": float(mx),
        "quad_mean_P": float(mp),
        "quad_var_X": float(vx),
        "quad_var_P": float(vp),
        "target_mean_X": state["mean_X"],
        "target_mean_P": state["mean_P"],
        "target_var_X": state["var_X"],
        "target_var_P": state["var_P"],
        "nodes": nodes,
        "note": "1D GH along the anti-squeezed axis only; squeezed-axis vacuum dropped",
    }


def field_from_xp(meta: dict, proto: dict, X: float, P: float) -> dict:
    t = meta["times"]
    env = meta["env"]
    omega = meta["omega"]
    t0 = meta["t0_fs"] / meta["t_unit_fs"]
    tau = t - t0
    e1_va = proto["E1_V_m"] * 1.0e-10
    e_mag = e1_va * env * (X * np.cos(omega * tau) + P * np.sin(omega * tau))
    da = np.zeros_like(e_mag)
    if t.size > 1:
        da[1:] = -0.5 * np.diff(t) * (e_mag[1:] + e_mag[:-1])
    a_mag = np.cumsum(da)
    return {"E_VA": e_mag, "a_mag": a_mag}


def a_vector(meta: dict, proto: dict, geo, X: float, P: float) -> np.ndarray:
    pol = pol_vector(SimpleNamespace(polarization="a2"), geo)
    fld = field_from_xp(meta, proto, X, P)
    return fld["a_mag"][:, None] * pol[None, :], fld


def propagate_a(geo, p, kc, w, neel: int, a_t: np.ndarray, times: np.ndarray, rho0=None) -> dict:
    rho = rho0.copy() if rho0 is not None else initial_rho(geo, p, kc, neel, mu=0.0)[0]
    nt = times.size - 1
    j_t = np.zeros((nt + 1, 2))
    t0 = time.perf_counter()
    for n in range(nt + 1):
        _h, jop = h8_and_current(geo, p, kc + a_t[n], neel)
        j_t[n] = w @ trace_current(rho, jop)
        if n == nt:
            break
        dt = times[n + 1] - times[n]
        amid = 0.5 * (a_t[n] + a_t[n + 1])
        hm = h8(geo, p, kc + amid, neel)
        em, wm = np.linalg.eigh(hm)
        u = (wm * np.exp(-1j * em * dt)[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
    return {
        "J": j_t,
        "wall_s": time.perf_counter() - t0,
        "herm": float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, 1, 2))))),
    }


def project_qs(times, jxy, omega, qs, lo, hi) -> np.ndarray:
    return spectrum(times, np.asarray(jxy, complex), lo, hi, qs, method="linear_exact", omega=omega)


def analyzer_stats(z_plus: np.ndarray, z_minus: np.ndarray) -> dict:
    e = 0.5 * (z_plus + z_minus)
    o = 0.5 * (z_plus - z_minus)
    out = {
        "plus": cpack(z_plus),
        "minus": cpack(z_minus),
        "even": cpack(e),
        "odd": cpack(o),
        "odd_norm2D": float(np.linalg.norm(o)),
        "even_norm2D": float(np.linalg.norm(e)),
        "odd_over_even": float(np.linalg.norm(o) / np.linalg.norm(e)) if np.linalg.norm(e) > 0 else None,
        "pol_odd_Jy_over_Jx": cpack(o[1] / o[0]) if abs(o[0]) > 0 else None,
    }
    stats = {}
    for name, vec in (
        ("x", np.array([1.0, 0.0])),
        ("y", np.array([0.0, 1.0])),
        ("45deg", np.ones(2) / np.sqrt(2.0)),
        ("135deg", np.array([1.0, -1.0]) / np.sqrt(2.0)),
    ):
        pp, mm, ee, oo = vec @ z_plus, vec @ z_minus, vec @ e, vec @ o
        ip, im = abs(pp) ** 2, abs(mm) ** 2
        stats[name] = {
            "Pplus_coh": float(ip),
            "Pminus_coh": float(im),
            "contrast_coh": float((ip - im) / (ip + im)) if ip + im > 0 else None,
            "interference_4Re": float(4.0 * np.real(np.conjugate(ee) * oo)),
            "phase_odd_minus_even_deg": float(np.degrees(np.angle(oo / ee))) if abs(ee) > 0 else None,
        }
    out["analyzers"] = stats
    return out


def inversion_closed(kf: np.ndarray) -> bool:
    wrap = np.round((-kf) % 1.0, 12)
    return bool(np.allclose(sorted(map(tuple, np.round(kf, 12))), sorted(map(tuple, wrap))))


def fwhm_intensity_fs() -> float:
    t = np.linspace(0.0, 5.0 * FWHM_I_FS, 20001)
    env = gaussian_env(t, 2.5 * FWHM_I_FS)
    inten = env ** 2
    half = 0.5 * inten.max()
    above = np.where(inten >= half)[0]
    return float(t[above[-1]] - t[above[0]])


def stage1(geo, p, *, nt_per_cycle: int = 32) -> dict:
    proto = protocol(1.0e10)
    meta = build_times(nt_per_cycle)
    pol = pol_vector(SimpleNamespace(polarization="a2"), geo)
    fwhm = fwhm_intensity_fs()
    st_coh = xp_state("coherent", proto)
    st_bsv = xp_state("bsv", proto, 0.0)
    st_ds0 = xp_state("ds", proto, 0.0)
    st_ds_pi = xp_state("ds", proto, np.pi)
    gh_bsv = gh1d_nodes(st_bsv, 5)
    gh_ds0 = gh1d_nodes(st_ds0, 5)
    a_coh, fld_coh = a_vector(meta, proto, geo, st_coh["mean_X"], st_coh["mean_P"])
    e_coh_si = fld_coh["E_VA"] / 1.0e-10
    epeak_num = float(np.max(np.abs(e_coh_si)))
    qpeak_num = float(np.max(np.abs(fld_coh["a_mag"])))
    e_bsv_mean = np.zeros_like(meta["times"])
    for n in gh_bsv["nodes"]:
        e_bsv_mean += n["weight"] * field_from_xp(meta, proto, n["X"], n["P"])["E_VA"]
    kf, kc, w = k_grid_xy(geo, 2, 0.5, 0.5)
    closed = inversion_closed(kf)
    acc_p = np.zeros((meta["times"].size, 2))
    acc_m = np.zeros((meta["times"].size, 2))
    walls = []
    rho_p, _ = initial_rho(geo, p, kc, +1, mu=0.0)
    rho_m, _ = initial_rho(geo, p, kc, -1, mu=0.0)
    for n in gh_bsv["nodes"]:
        a_t, _fld = a_vector(meta, proto, geo, n["X"], n["P"])
        rp = propagate_a(geo, p, kc, w, +1, a_t, meta["times"], rho_p)
        rm = propagate_a(geo, p, kc, w, -1, a_t, meta["times"], rho_m)
        acc_p += n["weight"] * rp["J"]
        acc_m += n["weight"] * rm["J"]
        walls.append(rp["wall_s"] + rm["wall_s"])
    je = 0.5 * (acc_p + acc_m)
    jo = 0.5 * (acc_p - acc_m)
    lo, hi = float(meta["times"][0]), float(meta["times"][-1])
    h2e = project_qs(meta["times"], je, meta["omega"], [2], lo, hi)[0]
    h2o = project_qs(meta["times"], jo, meta["omega"], [2], lo, hi)[0]
    even_over_odd = float(np.linalg.norm(h2e) / np.linalg.norm(h2o)) if np.linalg.norm(h2o) else None
    checks = {
        "fwhm_I_fs": fwhm,
        "fwhm_rel_err": abs(fwhm - FWHM_I_FS) / FWHM_I_FS,
        "coherent_Epeak_V_m": epeak_num,
        "coherent_Epeak_rel_err": abs(epeak_num - proto["E_peak_coh_V_m"]) / proto["E_peak_coh_V_m"],
        "coherent_qpeak_Ainv": qpeak_num,
        "narrowband_qpeak_Ainv": proto["q_peak_coh_Ainv_narrowband"],
        "qpeak_vs_narrowband_rel": abs(qpeak_num - proto["q_peak_coh_Ainv_narrowband"]) / proto["q_peak_coh_Ainv_narrowband"],
        "bsv_mean_E_over_coh_peak": float(np.max(np.abs(e_bsv_mean)) / (proto["E_peak_coh_V_m"] * 1e-10)),
        "gh5_weight_sum": gh_bsv["weight_sum"],
        "gh5_mean_X": gh_bsv["quad_mean_X"],
        "gh5_mean_P": gh_bsv["quad_mean_P"],
        "nk2_inversion_closed": closed,
        "bsv_H2_even_over_odd": even_over_odd,
        "bsv_H2_odd_norm2D": float(np.linalg.norm(h2o)),
        "pol_a2": pol.tolist(),
        "ds_theta0_anti_axis": st_ds0["anti_axis"],
        "ds_theta_pi_anti_axis": st_ds_pi["anti_axis"],
    }
    checks["pass_fwhm"] = bool(checks["fwhm_rel_err"] < 0.04)
    checks["pass_epeak"] = bool(checks["coherent_Epeak_rel_err"] < 0.08)
    checks["pass_bsv_mean"] = bool(checks["bsv_mean_E_over_coh_peak"] < 1e-10)
    checks["pass_gh5"] = bool(
        abs(gh_bsv["weight_sum"] - 1.0) < 1e-12
        and abs(gh_bsv["quad_mean_X"]) < 1e-8
        and abs(gh_bsv["quad_mean_P"]) < 1e-6 * st_bsv["sigma_anti"]
    )
    checks["pass_inversion"] = bool(closed)
    checks["pass_bsv_parity"] = bool(even_over_odd is not None and even_over_odd < 0.05)
    checks["pass"] = all(
        checks[k] for k in (
            "pass_fwhm", "pass_epeak", "pass_bsv_mean", "pass_gh5",
            "pass_inversion", "pass_bsv_parity",
        )
    )
    rec = {
        "status": "stage1",
        "protocol_0p01TWcm2": proto,
        "time_grid": {k: (v if not isinstance(v, np.ndarray) else None) for k, v in meta.items() if k not in ("times", "times_fs", "env")},
        "nt": meta["nt"],
        "dt_model": meta["dt"],
        "states": {"coherent": st_coh, "bsv": st_bsv, "ds0": st_ds0, "ds_pi": st_ds_pi},
        "gh_bsv": {k: v for k, v in gh_bsv.items() if k != "nodes"},
        "gh_ds0": {k: v for k, v in gh_ds0.items() if k != "nodes"},
        "checks": checks,
        "nk2_wall_s": float(np.sum(walls)),
        "H2_even": cpack(h2e),
        "H2_odd": cpack(h2o),
        "source_sha256": {
            "this": sha256_file(Path(__file__)),
            "q8_m0_model": sha256_file(HERE / "q8_m0_model.py"),
            "q8_m0_classical": sha256_file(HERE / "q8_m0_classical.py"),
        },
        "not_a_physics_result": True,
        "evidence_grade": "input_acceptance_only",
    }
    dump(OUT / "stage1.json", rec)
    return rec


def _rho_cache(geo, p, kc):
    return {+1: initial_rho(geo, p, kc, +1, mu=0.0)[0], -1: initial_rho(geo, p, kc, -1, mu=0.0)[0]}


def accumulate_input(geo, p, kc, w, meta, proto, state, gh, rhos) -> dict:
    qs = [1, 2, 3, 4, 5, 6]
    lo, hi = float(meta["times"][0]), float(meta["times"][-1])
    mid_lo = float(meta["times"][0] + 0.08 * (meta["times"][-1] - meta["times"][0]))
    mid_hi = float(meta["times"][-1] - 0.08 * (meta["times"][-1] - meta["times"][0]))
    mean_jp = np.zeros((meta["times"].size, 2))
    mean_jm = np.zeros((meta["times"].size, 2))
    h2_plus = []
    h2_minus = []
    h3_plus = []
    h3_minus = []
    walls = []
    herm = []
    for n in gh["nodes"]:
        a_t, _fld = a_vector(meta, proto, geo, n["X"], n["P"])
        rp = propagate_a(geo, p, kc, w, +1, a_t, meta["times"], rhos[+1])
        rm = propagate_a(geo, p, kc, w, -1, a_t, meta["times"], rhos[-1])
        wt = n["weight"]
        mean_jp += wt * rp["J"]
        mean_jm += wt * rm["J"]
        zp = project_qs(meta["times"], rp["J"], meta["omega"], qs, lo, hi)
        zm = project_qs(meta["times"], rm["J"], meta["omega"], qs, lo, hi)
        h2_plus.append((wt, zp[1]))
        h2_minus.append((wt, zm[1]))
        h3_plus.append((wt, zp[2]))
        h3_minus.append((wt, zm[2]))
        walls.append(rp["wall_s"] + rm["wall_s"])
        herm.extend([rp["herm"], rm["herm"]])
    z_mean_p = project_qs(meta["times"], mean_jp, meta["omega"], qs, lo, hi)
    z_mean_m = project_qs(meta["times"], mean_jm, meta["omega"], qs, lo, hi)
    z_mean_p_mid = project_qs(meta["times"], mean_jp, meta["omega"], [2], mid_lo, mid_hi)[0]
    z_mean_m_mid = project_qs(meta["times"], mean_jm, meta["omega"], [2], mid_lo, mid_hi)[0]
    h2_stats = analyzer_stats(z_mean_p[1], z_mean_m[1])
    def mean_abs2(pairs):
        return sum(wt * np.abs(z) ** 2 for wt, z in pairs)
    pstat_plus = mean_abs2(h2_plus)
    pstat_minus = mean_abs2(h2_minus)
    pcoh_plus = np.abs(z_mean_p[1]) ** 2
    pcoh_minus = np.abs(z_mean_m[1]) ** 2
    cov_h2h3_plus = sum(wt * np.conjugate(a[1]) * b[1] for (wt, a), (_, b) in zip(h2_plus, h3_plus))
    return {
        "kind": state["kind"],
        "theta": state["theta"],
        "n_nodes": gh["n_nodes"],
        "harmonics_mean": {
            str(q): {
                "plus": cpack(z_mean_p[i]),
                "minus": cpack(z_mean_m[i]),
                "even": cpack(0.5 * (z_mean_p[i] + z_mean_m[i])),
                "odd": cpack(0.5 * (z_mean_p[i] - z_mean_m[i])),
            }
            for i, q in enumerate(qs)
        },
        "H2": h2_stats,
        "H2_window_trim_16pct": analyzer_stats(z_mean_p_mid, z_mean_m_mid),
        "Pcoh_H2": {"plus": pcoh_plus.tolist(), "minus": pcoh_minus.tolist()},
        "Pstat_H2": {"plus": pstat_plus.tolist(), "minus": pstat_minus.tolist()},
        "Pstat_minus_Pcoh_H2_plus": (pstat_plus - pcoh_plus).tolist(),
        "sample_H2H3_cov_plus": cpack(cov_h2h3_plus),
        "sample_H2H3_note": "trajectory covariance of frequency projections; not g^(2)",
        "wall_s": float(np.sum(walls)),
        "max_herm": float(np.max(herm)),
    }


def physics(geo, p, *, i0: float, nk: int, gh: int, nt_per_cycle: int) -> dict:
    proto = protocol(i0)
    meta = build_times(nt_per_cycle)
    kf, kc, w = k_grid_xy(geo, nk, 0.5, 0.5)
    rhos = _rho_cache(geo, p, kc)
    inputs = [
        ("coherent", 0.0),
        ("bsv", 0.0),
        ("ds", 0.0),
        ("ds", np.pi),
    ]
    rows = []
    t_all = time.perf_counter()
    for kind, th in inputs:
        st = xp_state(kind, proto, th)
        ghn = gh1d_nodes(st, gh)
        rec = accumulate_input(geo, p, kc, w, meta, proto, st, ghn, rhos)
        rec["state"] = {k: v for k, v in st.items()}
        rec["gh_summary"] = {k: v for k, v in ghn.items() if k != "nodes"}
        rows.append(rec)
        dump(OUT / f"physics_{kind}_th{int(round(th*100))}_i{i0:.0e}_nk{nk}_gh{gh}.json", rec)
        print(f"done {kind} theta={th:.3f} wall={rec['wall_s']:.1f}s oddH2={rec['H2']['odd_norm2D']:.4e}", flush=True)
    out = {
        "status": "physics",
        "i0_W_cm2": i0,
        "nk": nk,
        "n_kpts": int(kc.shape[0]),
        "inversion_closed": inversion_closed(kf),
        "gh_order": gh,
        "nt_per_cycle": nt_per_cycle,
        "nt": meta["nt"],
        "protocol": proto,
        "rows": rows,
        "wall_s_total": time.perf_counter() - t_all,
        "evidence_grade": "statistical_SBE_mean_current_M0_not_CrI3_not_QED",
        "observables_are": "|<J>|^2 coherent-source proxy and <|J|^2> sample-power proxy",
        "source_sha256": sha256_file(Path(__file__)),
    }
    dump(OUT / f"physics_i{i0:.0e}_nk{nk}_gh{gh}.json", out)
    return out


def kernel(geo, p, *, nk: int, nt_per_cycle: int = 32, e_peak_v_m: float = 5.0e7) -> dict:
    """Classical two-color mixed derivative of Néel-odd H2. Not a quantum kernel."""
    meta = build_times(nt_per_cycle)
    proto_dummy = protocol(1.0e10)
    kf, kc, w = k_grid_xy(geo, nk, 0.5, 0.5)
    rhos = _rho_cache(geo, p, kc)
    pol = pol_vector(SimpleNamespace(polarization="a2"), geo)
    omega = meta["omega"]
    t0 = meta["t0_fs"] / meta["t_unit_fs"]
    tau = meta["times"] - t0
    env = meta["env"]
    e_va = e_peak_v_m * 1.0e-10

    def drive(w1, w2, s1, s2):
        e_mag = e_va * env * (s1 * np.cos(w1 * tau) + s2 * np.cos(w2 * tau))
        da = np.zeros_like(e_mag)
        da[1:] = -0.5 * np.diff(meta["times"]) * (e_mag[1:] + e_mag[:-1])
        a_mag = np.cumsum(da)
        return a_mag[:, None] * pol[None, :]

    def odd_h2(a_t):
        jp = propagate_a(geo, p, kc, w, +1, a_t, meta["times"], rhos[+1])["J"]
        jm = propagate_a(geo, p, kc, w, -1, a_t, meta["times"], rhos[-1])["J"]
        jo = 0.5 * (jp - jm)
        lo, hi = float(meta["times"][0]), float(meta["times"][-1])
        return project_qs(meta["times"], jo, omega, [2], lo, hi)[0]

    pairs = {
        "equal_ww": (omega, omega),
        "split_0p7_1p3": (0.7 * omega, 1.3 * omega),
    }
    rec = {
        "status": "kernel",
        "nk": nk,
        "e_peak_V_m": e_peak_v_m,
        "note": "finite-difference mixed derivative; classical causal proxy, not LSZ output",
        "pairs": {},
    }
    t0w = time.perf_counter()
    for name, (w1, w2) in pairs.items():
        z11 = odd_h2(drive(w1, w2, 1, 1))
        z10 = odd_h2(drive(w1, w2, 1, 0))
        z01 = odd_h2(drive(w1, w2, 0, 1))
        z00 = odd_h2(drive(w1, w2, 0, 0))
        mixed = (z11 - z10 - z01 + z00) / (e_va * e_va)
        rec["pairs"][name] = {
            "w1_eV": float(w1),
            "w2_eV": float(w2),
            "sum_eV": float(w1 + w2),
            "odd_H2_11": cpack(z11),
            "odd_H2_mixed_over_E2": cpack(mixed),
            "odd_norm": float(np.linalg.norm(mixed)),
        }
        print(f"kernel {name} odd||K||={np.linalg.norm(mixed):.4e}", flush=True)
    ke = rec["pairs"]["equal_ww"]["odd_norm"]
    ks = rec["pairs"]["split_0p7_1p3"]["odd_norm"]
    rec["R_mat_equal_over_split"] = float(ke / ks) if ks else None
    rec["wall_s"] = time.perf_counter() - t0w
    rec["protocol_dummy_I0_only_for_units"] = proto_dummy["I0_W_cm2"]
    dump(OUT / f"kernel_nk{nk}.json", rec)
    return rec


def candidate_verdict(phys: dict) -> dict:
    rows = {f"{r['kind']}_th{int(round(r['theta']*100))}": r for r in phys["rows"]}
    coh = rows["coherent_th0"]
    bsv = rows["bsv_th0"]
    ds0 = rows["ds_th0"]
    dsp = rows["ds_th314"]
    bsv_even_over_odd = bsv["H2"]["odd_over_even"]
    # For BSV, even should vanish so odd_over_even is huge; use even/odd instead.
    bsv_even = bsv["H2"]["even_norm2D"]
    bsv_odd = bsv["H2"]["odd_norm2D"]
    ratio_even_odd = bsv_even / bsv_odd if bsv_odd else None
    ph0 = np.angle(complex(ds0["H2"]["odd"][0]["re"], ds0["H2"]["odd"][0]["im"])) if isinstance(ds0["H2"]["odd"], list) else None
    # packed() on ndarray becomes list of dicts
    def odd_jx(row):
        odd = row["H2"]["odd"]
        if isinstance(odd, list):
            return complex(odd[0]["re"], odd[0]["im"])
        return complex(odd["re"], odd["im"]) if "re" in odd else complex(odd[0]["re"], odd[0]["im"])

    z_ds0 = odd_jx(ds0)
    z_dsp = odd_jx(dsp)
    z_coh = odd_jx(coh)
    z_bsv = odd_jx(bsv)
    dphi = float(np.degrees(np.angle(z_dsp / z_ds0))) if abs(z_ds0) else None
    amp_ratio = float(abs(z_dsp) / abs(z_ds0)) if abs(z_ds0) else None
    coh_vs_bsv = float(abs(z_bsv) / abs(z_coh)) if abs(z_coh) else None
    trim0 = ds0["H2_window_trim_16pct"]["odd_norm2D"]
    full0 = ds0["H2"]["odd_norm2D"]
    window_rel = abs(trim0 - full0) / full0 if full0 else None
    magnetic = bool(
        ratio_even_odd is not None and ratio_even_odd < 0.08
        and bsv_odd > 0
        and (dphi is not None and abs(abs(dphi) - 0) > 8.0 or (amp_ratio is not None and abs(amp_ratio - 1) > 0.08))
    )
    stable_enough = bool(window_rel is not None and window_rel < 0.35)
    return {
        "bsv_even_over_odd": ratio_even_odd,
        "ds_theta_pi_over_0_amp": amp_ratio,
        "ds_theta_phase_diff_deg": dphi,
        "bsv_over_coherent_odd_amp": coh_vs_bsv,
        "ds0_window_trim_rel": window_rel,
        "magnetic_candidate": magnetic,
        "window_stable_enough": stable_enough,
        "continue_to_denser_grid": bool(magnetic and stable_enough),
        "note": "candidate means Néel-odd H2 differs by statistics/phase, not a yield-enhancement claim",
    }


def write_report(stage: dict | None, kern: dict | None, phys_list: list) -> Path:
    DOC.mkdir(parents=True, exist_ok=True)
    path = DOC / "线Q_1600nm_BZ统计首批物理结论_20261028.md"
    lines = []
    lines.append("# 线 Q：1600 nm / 25 fs 首批物理结论（2026-10-28）\n\n")
    lines.append("统计电子动力学，不是精确 QED，不是光子 g^{(2)}，不是 CrI₃ 定量预测。")
    lines.append("观测量区分 |⟨J⟩|²（相干源代理）与 ⟨|J|²⟩（样本功率代理）。")
    lines.append("纯 BSV 在现有反演协议下两畴相干源强度必须相同，这不是发现。\n\n")
    if stage:
        c = stage["checks"]
        lines.append("## 阶段 1 输入验收\n\n")
        lines.append(f"- 通过：{'是' if c['pass'] else '否'}\n")
        lines.append(f"- 强度 FWHM {c['fwhm_I_fs']:.3f} fs（相对误差 {100*c['fwhm_rel_err']:.2f}%）\n")
        lines.append(f"- 相干 Epeak 相对误差 {100*c['coherent_Epeak_rel_err']:.2f}%\n")
        lines.append(f"- BSV ⟨E⟩/相干峰 {c['bsv_mean_E_over_coh_peak']:.3e}\n")
        lines.append(f"- nk=2 BSV H2 even/odd {c['bsv_H2_even_over_odd']}\n\n")
    if kern:
        lines.append("## 经典双频核代理（M 与 K_N 的材料入口）\n\n")
        lines.append(f"- equal (ω,ω) ||K_odd|| = {kern['pairs']['equal_ww']['odd_norm']:.4e}\n")
        lines.append(f"- split (0.7ω,1.3ω) ||K_odd|| = {kern['pairs']['split_0p7_1p3']['odd_norm']:.4e}\n")
        lines.append(f"- R_mat = equal/split = {kern.get('R_mat_equal_over_split')}\n")
        lines.append("- 这是经典因果混合导数，不是量子输出核。\n\n")
    for phys in phys_list:
        v = candidate_verdict(phys)
        lines.append(f"## 物理传播 I0={phys['i0_W_cm2']:.2e} W/cm²，nk={phys['nk']}\n\n")
        lines.append(f"- 网格反演封闭：{phys['inversion_closed']}；总墙时 {phys['wall_s_total']:.1f} s\n")
        for r in phys["rows"]:
            h2 = r["H2"]
            lines.append(
                f"- {r['kind']} θ={r['theta']:.3f}：H2 odd={h2['odd_norm2D']:.4e}，"
                f"even={h2['even_norm2D']:.4e}，"
                f"x-contrast={h2['analyzers']['x']['contrast_coh']}\n"
            )
        lines.append(f"- BSV even/odd = {v['bsv_even_over_odd']}\n")
        lines.append(f"- DS θ=π/θ=0 奇振幅比 = {v['ds_theta_pi_over_0_amp']}，相位差 {v['ds_theta_phase_diff_deg']}°\n")
        lines.append(f"- BSV/相干 奇振幅比 = {v['bsv_over_coherent_odd_amp']}\n")
        lines.append(f"- 去边窗相对变化 = {v['ds0_window_trim_rel']}\n")
        lines.append(f"- 磁性统计候选：{v['magnetic_candidate']}；窗口尚可：{v['window_stable_enough']}\n\n")
    lines.append("## 五个问题的当前答案\n\n")
    if phys_list:
        v = candidate_verdict(phys_list[-1])
        lines.append(f"1. 稳定 Néel×统计 H2：候选={v['magnetic_candidate']}，窗口稳定={v['window_stable_enough']}。尚未认证谐波产额。\n")
        lines.append("2. 相干光有平均场；纯 BSV 平均场为 0、靠涨落驱动，且 even 均值电流应被反演禁止；DS 保留相位参考，θ=0/π 对照压缩轴。\n")
        lines.append("3. 目前只有 M0。不能推广为双层 CrI₃，除非同一输入协议下 Wannier/SBE 复现。\n")
        if kern and kern.get("R_mat_equal_over_split") is not None:
            lines.append(f"4. 机制入口：最低阶 J ~ K_N · M。本轮经典核 R_mat={kern['R_mat_equal_over_split']:.4f}。强场统计超出该式。\n")
        else:
            lines.append("4. 机制入口仍是 M 与 K_N；完整电子跃迁分解未完成。\n")
        lines.append("5. 下一步：若候选成立，加密 BZ 并做少量真实材料 k 点；若消失，收口并改观测量，而不是扩扫描。两时关联/输出统计不是本批前提。\n")
    else:
        lines.append("物理传播尚未完成。\n")
    lines.append("\n详细 JSON：`" + str(OUT) + "`\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "physics", "kernel", "all", "report"])
    ap.add_argument("--i0", type=float, default=1.0e10)
    ap.add_argument("--nk", type=int, default=6)
    ap.add_argument("--gh", type=int, default=5)
    ap.add_argument("--nt-per-cycle", type=int, default=32)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    p = R0
    if args.cmd == "stage1":
        rec = stage1(geo, p, nt_per_cycle=args.nt_per_cycle)
        print(json.dumps(rec["checks"], indent=2, ensure_ascii=False))
        return 0 if rec["checks"]["pass"] else 1
    if args.cmd == "kernel":
        rec = kernel(geo, p, nk=args.nk, nt_per_cycle=args.nt_per_cycle)
        print("R_mat", rec.get("R_mat_equal_over_split"))
        return 0
    if args.cmd == "physics":
        rec = physics(geo, p, i0=args.i0, nk=args.nk, gh=args.gh, nt_per_cycle=args.nt_per_cycle)
        print(json.dumps(candidate_verdict(rec), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "report":
        st = json.loads((OUT / "stage1.json").read_text(encoding="utf-8")) if (OUT / "stage1.json").exists() else None
        kn = None
        phys = []
        for fn in sorted(OUT.glob("physics_i*_nk*_gh*.json")):
            phys.append(json.loads(fn.read_text(encoding="utf-8")))
        kns = sorted(OUT.glob("kernel_nk*.json"))
        if kns:
            kn = json.loads(kns[-1].read_text(encoding="utf-8"))
        path = write_report(st, kn, phys)
        print(path)
        return 0
    st = stage1(geo, p, nt_per_cycle=args.nt_per_cycle)
    print("STAGE1", st["checks"]["pass"], flush=True)
    if not st["checks"]["pass"]:
        write_report(st, None, [])
        return 1
    kn = kernel(geo, p, nk=6, nt_per_cycle=args.nt_per_cycle)
    ph = physics(geo, p, i0=1.0e10, nk=6, gh=5, nt_per_cycle=args.nt_per_cycle)
    verd = candidate_verdict(ph)
    phys_list = [ph]
    if verd["continue_to_denser_grid"]:
        print("candidate: denser nk=12 at 0.01 TW", flush=True)
        ph2 = physics(geo, p, i0=1.0e10, nk=12, gh=5, nt_per_cycle=args.nt_per_cycle)
        phys_list.append(ph2)
        verd2 = candidate_verdict(ph2)
        if verd2["continue_to_denser_grid"]:
            print("still candidate: 0.1 TW nk=6", flush=True)
            ph3 = physics(geo, p, i0=1.0e11, nk=6, gh=5, nt_per_cycle=args.nt_per_cycle)
            phys_list.append(ph3)
    else:
        print("no denser scan: no stable magnetic candidate at nk=6 / 0.01 TW", flush=True)
    path = write_report(st, kn, phys_list)
    print("REPORT", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

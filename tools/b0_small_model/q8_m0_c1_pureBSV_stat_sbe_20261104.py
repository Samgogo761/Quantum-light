#!/usr/bin/env python3
"""C1 minimal statistical SBE: 1600 nm / 25 fs ideal pure BSV vs equal-energy coherent.

Authorized scope (external review 2026-11-04, protocol v2 frozen):
- traveling-wave Gaussian envelope; no 1+4+1 switch; no abstract g=0.005
- M0 few k/-k; not CrI3
- ideal single-mode pure BSV: alpha=0, |M|=sqrt(n_s(n_s+1))
- stable squeezed covariance; never 2 n_s + 1 - 2|M|
- save both Néel domains' complex H2, then even/odd and intensity proxies
- |<J2>|^2 is not <|J2|^2
- Wigner SBE is an approximation, not joint QED

Not C2, not extra Fock kernels, not new QED, does not modify frozen fits.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from q8_m0_classical import initial_rho, k_grid_xy, pol_vector, trace_current
from q8_m0_model import R0, build_geometry, h8, h8_and_current, sha256_file
from qinput_spectrum_reanalysis_20261028 import clean, packed, spectrum

ROOT = HERE.parents[1]
OUT = ROOT / "results/q_line/q1600_c1_pureBSV_stat_sbe_20261104"
DOC = ROOT / "docs/04_BSV量子光"
PROTOCOL_V2 = DOC / "ds_experimental_protocol_20261104_v2.json"

HBAR_EV_FS = 0.6582119568564649
HC_EV_NM = 1239.8419843320025
LAMBDA_NM = 1600.0
PHOTON_EV = HC_EV_NM / LAMBDA_NM
FWHM_I_FS = 25.0
T0_FS = 2.0 * np.pi * HBAR_EV_FS / PHOTON_EV
I0_DESIGN_W_CM2 = 1.0e11
NBAR_V2_0P1TW = 75756144634.47339
W0_M = 15.0e-6
C_SI = 299792458.0
EPS0_SI = 8.8541878128e-12
HBAR_SI = 1.054571817e-34
E_SI = 1.602176634e-19
FROZEN_FIT_SHA = "17a8e37ec3e0ab54096ff4edd1440c5a3f0412b09450c2fd8960cbc00d8e69db"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cpack(z):
    return packed(z)


def unpack(z) -> complex:
    if isinstance(z, dict):
        return complex(z["re"], z["im"])
    return complex(z)


def inversion_closed(kf: np.ndarray) -> bool:
    wrap = np.round((-kf) % 1.0, 12)
    return bool(np.allclose(sorted(map(tuple, np.round(kf, 12))), sorted(map(tuple, wrap))))


def gaussian_env(t_fs: np.ndarray, t0_fs: float) -> np.ndarray:
    sigma_i = FWHM_I_FS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_e = sigma_i * np.sqrt(2.0)
    return np.exp(-0.5 * ((t_fs - t0_fs) / sigma_e) ** 2)


def tau_eff_s(n: int = 20001) -> float:
    t = np.linspace(0.0, 5.0 * FWHM_I_FS, n)
    env = gaussian_env(t, 2.5 * FWHM_I_FS)
    return float(np.trapezoid(env ** 2, t) * 1.0e-15)


def stable_pure_bsv(n_s: float) -> dict:
    """Ideal pure BSV with vacuum Var(X)=Var(P)=1, X=a+a†.

    Forbidden: evaluating 2 n_s + 1 - 2|M| for the squeezed variance.
    Use σ_anti = √(n_s+1)+√(n_s) and σ_sq = 1/σ_anti, which is the
    stable rewrite of (√(n_s+1)-√(n_s)).
    """
    n_s = float(n_s)
    if n_s < 0.0:
        raise ValueError("n_s >= 0")
    sqrt_ns = np.sqrt(n_s)
    sqrt_nsp1 = np.sqrt(n_s + 1.0)
    sigma_anti = float(sqrt_nsp1 + sqrt_ns)  # e^{+r}, well conditioned
    sigma_sq = float(1.0 / sigma_anti)  # e^{-r}; never √(n+1)-√n at large n_s
    abs_M = float(sqrt_ns * sqrt_nsp1)
    r = float(np.arcsinh(sqrt_ns))
    return {
        "alpha": 0.0,
        "n_s": n_s,
        "abs_M": abs_M,
        "M": -abs_M,  # theta=0 squeezes X
        "r_internal": r,
        "sigma_sq": sigma_sq,
        "sigma_anti": sigma_anti,
        "var_X": sigma_sq * sigma_sq,
        "var_P": sigma_anti * sigma_anti,
        "theta": 0.0,
        "purity_bound_saturated": True,
        "vacuum_var_convention": "Var(X)=Var(P)=1, X=a+a^dagger",
        "parameterization": "sigma_anti=sqrt(n_s+1)+sqrt(n_s), sigma_sq=1/sigma_anti",
    }


def forbidden_squeezed_var(n_s: float) -> float:
    m = np.sqrt(n_s * (n_s + 1.0))
    return float((2.0 * n_s + 1.0) - 2.0 * m)


def traveling_protocol(i0_w_cm2: float) -> dict:
    i0_w_cm2 = float(i0_w_cm2)
    scale = i0_w_cm2 / I0_DESIGN_W_CM2
    a_eff = 0.5 * np.pi * W0_M * W0_M
    i0_si = i0_w_cm2 * 1.0e4
    e_peak = float(np.sqrt(2.0 * i0_si / (C_SI * EPS0_SI)))
    t_eff = tau_eff_s()
    u_j = float(i0_si * a_eff * t_eff)
    hbar_omega = HBAR_SI * (PHOTON_EV * E_SI / HBAR_SI)
    nbar = u_j / hbar_omega
    alpha_coh = float(np.sqrt(nbar))
    bsv = stable_pure_bsv(nbar)
    e1 = e_peak / (2.0 * alpha_coh) if alpha_coh > 0 else 0.0
    q_narrow = (e_peak * 1.0e-10) / PHOTON_EV
    return {
        "lambda_nm": LAMBDA_NM,
        "photon_eV": float(PHOTON_EV),
        "T0_fs": float(T0_FS),
        "fwhm_I_fs": FWHM_I_FS,
        "envelope": "gaussian_intensity_FWHM_25fs_no_plateau",
        "polarization": "a2",
        "cep": 0.0,
        "w0_m": W0_M,
        "A_eff_m2": float(a_eff),
        "tau_eff_s": t_eff,
        "I0_W_cm2": i0_w_cm2,
        "I0_is_literature_2TWcm2": bool(abs(i0_w_cm2 - 2.0e12) < 1.0),
        "I0_note": (
            "Design-point traveling-wave intensity for the first C1 method test. "
            "Not the Rasputnyi ~2 TW/cm2 comparison point unless I0 is set to 2e12."
        ),
        "U_J": u_j,
        "U_nJ": u_j * 1.0e9,
        "nbar": float(nbar),
        "nbar_v2_scaled": float(NBAR_V2_0P1TW * scale),
        "nbar_vs_v2_rel": abs(nbar - NBAR_V2_0P1TW * scale) / (NBAR_V2_0P1TW * scale),
        "E_peak_coh_V_m": e_peak,
        "q_peak_coh_Ainv_narrowband": float(q_narrow),
        "E1_V_m": float(e1),
        "alpha_coh": alpha_coh,
        "coherent_mean_X": float(2.0 * alpha_coh),
        "coherent_mean_P": 0.0,
        "bsv": bsv,
        "did_not_reuse_abstract_g": True,
        "did_not_use_1plus4plus1": True,
        "equal_energy": "coherent |alpha|^2 = BSV n_s = nbar",
    }


def build_times(nt_per_cycle: int) -> dict:
    omega = PHOTON_EV
    t_unit_fs = HBAR_EV_FS
    t0_fs = 2.5 * FWHM_I_FS
    t_max_fs = 5.0 * FWHM_I_FS
    n_cycles = t_max_fs / T0_FS
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


def sample_pure_bsv(state: dict, n: int, rng: np.random.Generator, theta: float = 0.0) -> np.ndarray:
    """Return (n,2) array of (X,P). theta=0 squeezes X; theta=pi squeezes P."""
    g = rng.standard_normal((int(n), 2))
    xs = state["sigma_sq"] * g[:, 0]
    pa = state["sigma_anti"] * g[:, 1]
    if abs(theta) < 1e-12:
        xp = np.column_stack([xs, pa])
    elif abs(abs(theta) - np.pi) < 1e-12:
        xp = np.column_stack([pa, xs])
    else:
        c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
        rot = np.array([[c, -s], [s, c]])
        xp = (rot @ np.column_stack([xs, pa]).T).T
    return xp


def xp_moments(xp: np.ndarray) -> dict:
    x = xp[:, 0]
    p = xp[:, 1]
    mx, mp = float(x.mean()), float(p.mean())
    vx, vp = float(x.var(ddof=1)), float(p.var(ddof=1))
    c_xp = float(np.mean((x - mx) * (p - mp)))
    n_from_wigner = 0.25 * (vx + vp) - 0.5
    m_from_wigner = 0.25 * ((vx - vp) + 2.0j * c_xp)
    return {
        "n": int(xp.shape[0]),
        "mean_X": mx,
        "mean_P": mp,
        "var_X": vx,
        "var_P": vp,
        "cov_XP": c_xp,
        "n_s_from_samples": float(n_from_wigner),
        "M_from_samples_re": float(m_from_wigner.real),
        "M_from_samples_im": float(m_from_wigner.imag),
        "abs_M_from_samples": float(abs(m_from_wigner)),
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


def a_vector(meta: dict, proto: dict, geo, X: float, P: float):
    pol = pol_vector(SimpleNamespace(polarization="a2"), geo)
    fld = field_from_xp(meta, proto, X, P)
    return fld["a_mag"][:, None] * pol[None, :], fld


def fourier_selftest() -> dict:
    t = np.linspace(2.0, 4.0, 21)
    omega = 2.0 * np.pi
    const = spectrum(t, np.ones_like(t), 2.0, 4.0, [0], method="linear_exact", omega=omega)[0, 0]
    # q=0 mean of 1 on [2,4] is 1; multiply by window length for the integral.
    const_int = const * 2.0
    t_dense = np.linspace(0.0, 4.0 * np.pi, 8001)
    y_dense = 2.0 * np.exp(-1j * 2.0 * t_dense)
    dense = spectrum(t_dense, y_dense, 0.0, 4.0 * np.pi, [2], method="linear_exact", omega=1.0)[0, 0]
    return {
        "constant_integral": {"value": float(const_int.real), "target": 2.0, "abs_err": abs(const_int - 2.0)},
        "dense_tone_mean": {"re": float(dense.real), "im": float(dense.imag), "target": 2.0, "abs_err": abs(dense - 2.0)},
        "pass": bool(abs(const_int - 2.0) < 1e-12 and abs(dense - 2.0) < 2e-6),
    }


def sampling_gates(proto: dict, n_samples: int, seed: int) -> dict:
    bsv = proto["bsv"]
    n_s = bsv["n_s"]
    unstable = forbidden_squeezed_var(n_s)
    stable = bsv["var_X"]
    product = bsv["sigma_sq"] * bsv["sigma_anti"]
    rng = np.random.default_rng(seed)
    xp = sample_pure_bsv(bsv, n_samples, rng, theta=0.0)
    mom = xp_moments(xp)
    rel_var_x = abs(mom["var_X"] - bsv["var_X"]) / bsv["var_X"]
    rel_var_p = abs(mom["var_P"] - bsv["var_P"]) / bsv["var_P"]
    rel_ns = abs(mom["n_s_from_samples"] - n_s) / n_s
    rel_m = abs(mom["abs_M_from_samples"] - bsv["abs_M"]) / bsv["abs_M"]
    unstable_rel = abs(unstable - stable) / stable if stable > 0 else np.inf
    unstable_unusable = (
        (not np.isfinite(unstable))
        or (unstable <= 0.0)
        or (unstable_rel > 0.1)
    )
    checks = {
        "sigma_product_is_1": abs(product - 1.0) < 1e-12,
        "unstable_formula_is_garbage": bool(unstable_unusable),
        "stable_var_X_positive": bsv["var_X"] > 0.0,
        "sample_mean_X_small": abs(mom["mean_X"]) < 8.0 * bsv["sigma_sq"] / np.sqrt(n_samples),
        "sample_mean_P_small": abs(mom["mean_P"]) < 8.0 * bsv["sigma_anti"] / np.sqrt(n_samples),
        "sample_var_X_ok": rel_var_x < 0.15,
        "sample_var_P_ok": rel_var_p < 0.15,
        "sample_n_s_ok": rel_ns < 0.15,
        "sample_abs_M_ok": rel_m < 0.15,
        "did_not_use_unstable_var": True,
    }
    checks["pass"] = all(bool(v) for k, v in checks.items() if k != "pass")
    return {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "stable_state": bsv,
        "forbidden_var_X": unstable,
        "stable_var_X": stable,
        "unstable_over_stable": float(unstable_rel),
        "sigma_product": float(product),
        "sample_moments": mom,
        "rel_err": {
            "var_X": float(rel_var_x),
            "var_P": float(rel_var_p),
            "n_s": float(rel_ns),
            "abs_M": float(rel_m),
        },
        "checks": checks,
    }


def field_gates(geo, proto: dict, meta: dict) -> dict:
    a_coh, fld_coh = a_vector(meta, proto, geo, proto["coherent_mean_X"], proto["coherent_mean_P"])
    e_si = fld_coh["E_VA"] / 1.0e-10
    epeak_num = float(np.max(np.abs(e_si)))
    qpeak_num = float(np.max(np.abs(fld_coh["a_mag"])))
    rng = np.random.default_rng(202611041)
    xp = sample_pure_bsv(proto["bsv"], 4096, rng, 0.0)
    e_mean = np.zeros_like(meta["times"])
    for X, P in xp:
        e_mean += field_from_xp(meta, proto, float(X), float(P))["E_VA"]
    e_mean /= xp.shape[0]
    e_mean_over_peak = float(np.max(np.abs(e_mean)) / (proto["E_peak_coh_V_m"] * 1e-10))
    fwhm_t = np.linspace(0.0, 5.0 * FWHM_I_FS, 20001)
    inten = gaussian_env(fwhm_t, 2.5 * FWHM_I_FS) ** 2
    half = 0.5 * inten.max()
    above = np.where(inten >= half)[0]
    fwhm = float(fwhm_t[above[-1]] - fwhm_t[above[0]])
    checks = {
        "fwhm_rel_err": abs(fwhm - FWHM_I_FS) / FWHM_I_FS,
        "coherent_Epeak_rel_err": abs(epeak_num - proto["E_peak_coh_V_m"]) / proto["E_peak_coh_V_m"],
        "qpeak_vs_narrowband_rel": abs(qpeak_num - proto["q_peak_coh_Ainv_narrowband"]) / proto["q_peak_coh_Ainv_narrowband"],
        "bsv_mean_E_over_coh_peak": e_mean_over_peak,
        "nbar_vs_v2_rel": float(proto["nbar_vs_v2_rel"]),
    }
    flags = {
        "pass_fwhm": checks["fwhm_rel_err"] < 0.04,
        "pass_epeak": checks["coherent_Epeak_rel_err"] < 0.08,
        "pass_bsv_mean_E": e_mean_over_peak < 0.05,
        "pass_nbar_v2": proto["nbar_vs_v2_rel"] < 0.02,
    }
    flags["pass"] = all(flags.values())
    return {
        "fwhm_I_fs": fwhm,
        "coherent_Epeak_V_m": epeak_num,
        "coherent_qpeak_Ainv": qpeak_num,
        "narrowband_qpeak_Ainv": proto["q_peak_coh_Ainv_narrowband"],
        "checks": checks,
        "flags": flags,
        "a_coh_shape": list(a_coh.shape),
    }


def propagate_a(geo, p, kc, w, neel: int, a_t: np.ndarray, times: np.ndarray, rho0) -> dict:
    rho = rho0.copy()
    nt = times.size - 1
    j_t = np.zeros((nt + 1, 2))
    t0 = time.perf_counter()
    herm = 0.0
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
    herm = float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, 1, 2)))))
    return {"J": j_t, "wall_s": time.perf_counter() - t0, "herm": herm}


def project_h2(times, jxy, omega) -> np.ndarray:
    lo, hi = float(times[0]), float(times[-1])
    return spectrum(times, np.asarray(jxy, complex), lo, hi, [2], method="linear_exact", omega=omega)[0]


def domain_bundle(z_plus: np.ndarray, z_minus: np.ndarray) -> dict:
    e = 0.5 * (z_plus + z_minus)
    o = 0.5 * (z_plus - z_minus)
    s_plus = np.abs(z_plus) ** 2
    s_minus = np.abs(z_minus) ** 2
    inter = 4.0 * np.real(np.conjugate(e) * o)
    ident = s_plus - s_minus
    out = {
        "plus": cpack(z_plus),
        "minus": cpack(z_minus),
        "even": cpack(e),
        "odd": cpack(o),
        "odd_norm2D": float(np.linalg.norm(o)),
        "even_norm2D": float(np.linalg.norm(e)),
        "S_meanabs2_plus": s_plus.tolist(),
        "S_meanabs2_minus": s_minus.tolist(),
        "contrast_meanabs2": [
            float((a - b) / (a + b)) if (a + b) > 0 else None for a, b in zip(s_plus, s_minus)
        ],
        "interference_4Re_even_odd": inter.tolist(),
        "identity_Splus_minus_Sminus_minus_4Re": (ident - inter).tolist(),
        "note": (
            "S here is |<J>|^2 from the averaged current. Sample-averaged <|J|^2> "
            "is stored separately. Nonzero odd current does not imply nonzero domain contrast."
        ),
    }
    return out


def running_convergence(values: np.ndarray) -> dict:
    """values shape (N, ...) real or complex. Report split-half relative change of the mean."""
    arr = np.asarray(values)
    n = arr.shape[0]
    if n < 4:
        return {"n": n, "pass": False, "reason": "too_few_samples"}
    half = n // 2
    m_full = arr.mean(axis=0)
    m_half = arr[:half].mean(axis=0)
    denom = np.maximum(np.abs(m_full), 1.0e-30)
    rel = np.abs(m_full - m_half) / denom
    rel_max = float(np.max(rel))
    return {
        "n": n,
        "n_half": half,
        "rel_max_full_vs_first_half": rel_max,
        "pass": bool(rel_max < 0.35),
        "threshold": 0.35,
    }


def analyze_samples(h2_plus: np.ndarray, h2_minus: np.ndarray) -> dict:
    """h2_* shape (N, 2) complex per-sample domain currents."""
    mean_p = h2_plus.mean(axis=0)
    mean_m = h2_minus.mean(axis=0)
    mean_bundle = domain_bundle(mean_p, mean_m)
    stat_p = np.mean(np.abs(h2_plus) ** 2, axis=0)
    stat_m = np.mean(np.abs(h2_minus) ** 2, axis=0)
    pcoh_p = np.abs(mean_p) ** 2
    pcoh_m = np.abs(mean_m) ** 2
    conv_mean = running_convergence(h2_plus)
    conv_stat = running_convergence(np.abs(h2_plus) ** 2)
    contrast_stat = [
        float((a - b) / (a + b)) if (a + b) > 0 else None for a, b in zip(stat_p, stat_m)
    ]
    contrast_coh = [
        float((a - b) / (a + b)) if (a + b) > 0 else None for a, b in zip(pcoh_p, pcoh_m)
    ]
    return {
        "mean_current": mean_bundle,
        "Pcoh_abs_mean_sq": {"plus": pcoh_p.tolist(), "minus": pcoh_m.tolist(), "contrast": contrast_coh},
        "Pstat_mean_abs_sq": {"plus": stat_p.tolist(), "minus": stat_m.tolist(), "contrast": contrast_stat},
        "not_equivalent": True,
        "mean_abs_sq_minus_abs_mean_sq_plus": (stat_p - pcoh_p).tolist(),
        "convergence_mean_H2_plus": conv_mean,
        "convergence_Pstat_plus": conv_stat,
        "gates": {
            "mean_current_converged": conv_mean["pass"],
            "intensity_proxy_converged": conv_stat["pass"],
            "pass": bool(conv_mean["pass"] and conv_stat["pass"]),
        },
    }


def run_input(geo, p, kc, w, meta, proto, kind: str, n_samples: int, seed: int, theta: float = 0.0) -> dict:
    rhos = {
        +1: initial_rho(geo, p, kc, +1, mu=0.0)[0],
        -1: initial_rho(geo, p, kc, -1, mu=0.0)[0],
    }
    h2_plus = []
    h2_minus = []
    walls = []
    herms = []
    if kind == "coherent":
        pairs = [(proto["coherent_mean_X"], proto["coherent_mean_P"])]
        weights = [1.0]
        sampled = False
    elif kind == "bsv":
        rng = np.random.default_rng(seed)
        xp = sample_pure_bsv(proto["bsv"], n_samples, rng, theta=theta)
        pairs = [(float(X), float(P)) for X, P in xp]
        weights = [1.0 / len(pairs)] * len(pairs)
        sampled = True
    else:
        raise ValueError(kind)
    t0 = time.perf_counter()
    for X, P in pairs:
        a_t, _fld = a_vector(meta, proto, geo, X, P)
        rp = propagate_a(geo, p, kc, w, +1, a_t, meta["times"], rhos[+1])
        rm = propagate_a(geo, p, kc, w, -1, a_t, meta["times"], rhos[-1])
        zp = project_h2(meta["times"], rp["J"], meta["omega"])
        zm = project_h2(meta["times"], rm["J"], meta["omega"])
        h2_plus.append(zp)
        h2_minus.append(zm)
        walls.append(rp["wall_s"] + rm["wall_s"])
        herms.extend([rp["herm"], rm["herm"]])
    h2_plus = np.asarray(h2_plus, complex)
    h2_minus = np.asarray(h2_minus, complex)
    ana = analyze_samples(h2_plus, h2_minus)
    rec = {
        "kind": kind,
        "theta": float(theta),
        "sampled": sampled,
        "n_trajectories": int(h2_plus.shape[0]),
        "weight_sum": float(np.sum(weights)),
        "wall_s": time.perf_counter() - t0,
        "mean_job_s": float(np.mean(walls)) if walls else None,
        "max_herm": float(np.max(herms)) if herms else None,
        "analysis": ana,
        "per_sample_H2_plus": [cpack(z) for z in h2_plus],
        "per_sample_H2_minus": [cpack(z) for z in h2_minus],
    }
    return rec


def compare_coherent_bsv(coh: dict, bsv: dict) -> dict:
    def odd(row):
        return np.array([unpack(z) for z in row["analysis"]["mean_current"]["odd"]], complex)

    def even(row):
        return np.array([unpack(z) for z in row["analysis"]["mean_current"]["even"]], complex)

    o_c, o_b = odd(coh), odd(bsv)
    e_c, e_b = even(coh), even(bsv)
    pstat_c = np.array(coh["analysis"]["Pstat_mean_abs_sq"]["plus"])
    pstat_b = np.array(bsv["analysis"]["Pstat_mean_abs_sq"]["plus"])
    return {
        "odd_H2_norm_coherent": float(np.linalg.norm(o_c)),
        "odd_H2_norm_bsv": float(np.linalg.norm(o_b)),
        "odd_ratio_bsv_over_coh": float(np.linalg.norm(o_b) / np.linalg.norm(o_c)) if np.linalg.norm(o_c) else None,
        "even_H2_norm_coherent": float(np.linalg.norm(e_c)),
        "even_H2_norm_bsv": float(np.linalg.norm(e_b)),
        "Pstat_plus_ratio_bsv_over_coh": (pstat_b / np.maximum(pstat_c, 1e-30)).tolist(),
        "bsv_odd_over_even": float(np.linalg.norm(o_b) / np.linalg.norm(e_b)) if np.linalg.norm(e_b) else None,
        "coh_odd_over_even": float(np.linalg.norm(o_c) / np.linalg.norm(e_c)) if np.linalg.norm(e_c) else None,
        "warning": (
            "Wigner-SBE mean current and sample intensity proxy are not a quantum-advantage claim. "
            "A large odd current still needs 4 Re(Je* Jo) before any domain-contrast statement."
        ),
    }


def write_report(rec: dict) -> Path:
    lines = []
    lines.append("# 1600 nm C1 最小统计 SBE（理想纯 BSV，2026-11-04）\n\n")
    lines.append("协议 v2 冻结。本轮只做 C1 方法验证：行波 25 fs 高斯包络、M0 少量 k/-k、")
    lines.append("同能量相干光对照。**不是 CrI₃，不是联合 QED，不是 C2，未改冻结拟合。**\n\n")
    proto = rec["protocol"]
    lines.append("## 冻结输入\n\n")
    lines.append("- 1600 nm，25 fs，w0 = 15 um\n")
    lines.append(
        f"- I0 = {proto['I0_W_cm2']:.3e} W/cm^2（设计点 0.1 TW/cm^2，不是文献 2 TW 比较点）\n"
    )
    lines.append(f"- nbar = {proto['nbar']:.6e}，相干 alpha = {proto['alpha_coh']:.6e}\n")
    lines.append("- 纯 BSV：alpha=0，n_s=nbar，|M|=sqrt(n_s(n_s+1))，压缩轴 theta=0（压 X）\n")
    lines.append("- 不用 g=0.005，不用 1+4+1，不用 GH5\n\n")
    g = rec["gates"]
    lines.append("## 前置闸门\n\n")
    lines.append(f"- Fourier 自检：{'通过' if g['fourier']['pass'] else '失败'}\n")
    lines.append(f"- 稳定协方差：{'通过' if g['sampling']['checks']['pass'] else '失败'}\n")
    lines.append(
        f"- 禁止公式 2 n_s + 1 - 2|M| 相对误差 {g['sampling']['unstable_over_stable']:.3e}（必须是垃圾）\n"
    )
    lines.append(f"- 行波场映射：{'通过' if g['field']['flags']['pass'] else '失败'}\n")
    lines.append(f"- 反演封闭 k 网格：{rec['grid']['inversion_closed']}\n\n")
    if rec.get("status") == "gates_failed":
        lines.append("**闸门失败，未启动 M0 传播。**\n")
        path = OUT / "C1_report.md"
        path.write_text("".join(lines), encoding="utf-8")
        return path
    if rec.get("status") == "gates_only":
        lines.append("本文件只跑了闸门，尚未做 M0 传播。\n")
        path = OUT / "C1_report.md"
        path.write_text("".join(lines), encoding="utf-8")
        return path
    lines.append("## M0 传播\n\n")
    lines.append(f"- nk={rec['grid']['nk']}，k 点数 {rec['grid']['n_kpts']}，nt/cycle={rec['time']['nt_per_cycle']}\n")
    coh = rec["coherent"]
    bsv = rec["bsv"]
    lines.append(f"- 相干：1 条平均场轨迹，墙时 {coh['wall_s']:.2f} s\n")
    lines.append(f"- BSV：{bsv['n_trajectories']} 条 Wigner 样本，墙时 {bsv['wall_s']:.2f} s\n")
    lines.append(f"- 最大非厄米 {max(coh['max_herm'], bsv['max_herm']):.3e}\n\n")
    ac = bsv["analysis"]["gates"]
    lines.append("## 统计收敛\n\n")
    lines.append(
        f"- 平均复 H2：{'通过' if ac['mean_current_converged'] else '未通过'} "
        f"（半样本相对变化 {bsv['analysis']['convergence_mean_H2_plus']['rel_max_full_vs_first_half']:.3f}）\n"
    )
    lines.append(
        f"- 强度代理 < |J2|^2 >：{'通过' if ac['intensity_proxy_converged'] else '未通过'} "
        f"（半样本相对变化 {bsv['analysis']['convergence_Pstat_plus']['rel_max_full_vs_first_half']:.3f}）\n\n"
    )
    if not ac["pass"]:
        lines.append("**统计平均未收敛。按契约收口，不调参数寻找现象。**\n\n")
    cmp_ = rec["comparison"]
    mc_b = bsv["analysis"]["mean_current"]
    mc_c = coh["analysis"]["mean_current"]
    lines.append("## 观测量（有限时间电流代理）\n\n")
    lines.append("| 量 | 相干 | 纯 BSV |\n|---|---|---|\n")
    lines.append(f"| Néel-odd ||H2|| | {mc_c['odd_norm2D']:.6e} | {mc_b['odd_norm2D']:.6e} |\n")
    lines.append(f"| Néel-even ||H2|| | {mc_c['even_norm2D']:.6e} | {mc_b['even_norm2D']:.6e} |\n")
    lines.append(
        f"| odd/even | {cmp_['coh_odd_over_even']} | {cmp_['bsv_odd_over_even']} |\n"
    )
    lines.append(
        f"| C from |<J>|^2 (x) | {coh['analysis']['Pcoh_abs_mean_sq']['contrast'][0]} | {bsv['analysis']['Pcoh_abs_mean_sq']['contrast'][0]} |\n"
    )
    lines.append(
        f"| C from <|J|^2> (x) | {coh['analysis']['Pstat_mean_abs_sq']['contrast'][0]} | {bsv['analysis']['Pstat_mean_abs_sq']['contrast'][0]} |\n\n"
    )
    ident = np.max(np.abs(mc_b["identity_Splus_minus_Sminus_minus_4Re"]))
    lines.append(
        f"- BSV 平均电流恒等式 |J+|^2 - |J-|^2 - 4 Re(Je* Jo) 最大残差 {ident:.3e}\n\n"
    )
    lines.append("## 判断\n\n")
    if ac["pass"]:
        lines.append(
            f"- 在当前 M0、nk={rec['grid']['nk']}、I0=0.1 TW/cm^2 下，"
            f"BSV/相干的 odd-H2 范数比为 {cmp_['odd_ratio_bsv_over_coh']}。\n"
        )
        lines.append("- 这不是量子优势，也不是真实材料结论。Wigner 随机场 SBE ≠ 精确 QED。\n")
        lines.append("- 未启动 C2 / CrI₃ / 新 QED。\n")
    else:
        mean_ok = ac["mean_current_converged"]
        stat_ok = ac["intensity_proxy_converged"]
        lines.append(
            f"- 统计闸门未过：平均复 H2 {'通过' if mean_ok else '未过'}，"
            f"强度代理 {'通过' if stat_ok else '未过'}。保持停机，不调 I0/g。\n"
        )
    lines.append("\nJSON：`" + str(OUT / "c1.json") + "`\n")
    path = OUT / "C1_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    doc_copy = DOC / "线Q_C1最小统计SBE_20261104.md"
    doc_copy.write_text("".join(lines), encoding="utf-8")
    return path


def run_gates(i0: float, n_gate: int, seed: int, nt_per_cycle: int) -> dict:
    geo = build_geometry()
    proto = traveling_protocol(i0)
    meta = build_times(nt_per_cycle)
    fourier = fourier_selftest()
    sampling = sampling_gates(proto, n_gate, seed)
    field = field_gates(geo, proto, meta)
    kf, kc, w = k_grid_xy(geo, 2, 0.5, 0.5)
    closed = inversion_closed(kf)
    ok = bool(fourier["pass"] and sampling["checks"]["pass"] and field["flags"]["pass"] and closed)
    return {
        "protocol": proto,
        "time": {k: v for k, v in meta.items() if k not in ("times", "times_fs", "env")},
        "meta": meta,
        "geo": geo,
        "grid_preview": {
            "nk": 2,
            "n_kpts": int(kc.shape[0]),
            "inversion_closed": closed,
            "kfrac": kf.tolist(),
        },
        "gates": {"fourier": fourier, "sampling": sampling, "field": field, "pass": ok},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C1 minimal pure-BSV statistical SBE")
    ap.add_argument("cmd", choices=["gates", "physics", "all"], nargs="?", default="all")
    ap.add_argument("--i0", type=float, default=I0_DESIGN_W_CM2)
    ap.add_argument("--nk", type=int, default=2)
    ap.add_argument("--nt-per-cycle", type=int, default=32)
    ap.add_argument("--n-gate-samples", type=int, default=20000)
    ap.add_argument("--n-physics-samples", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    packed_gates = run_gates(args.i0, args.n_gate_samples, args.seed, args.nt_per_cycle)
    geo = packed_gates.pop("geo")
    meta = packed_gates.pop("meta")
    proto = packed_gates["protocol"]
    rec = {
        "status": "gates_only" if args.cmd == "gates" else "running",
        "date": "2026-11-04",
        "not_CrI3": True,
        "not_QED": True,
        "not_C2": True,
        "did_not_modify_frozen_fit": True,
        "frozen_fit_sha256": FROZEN_FIT_SHA,
        "protocol_v2": str(PROTOCOL_V2),
        "I0_W_cm2": args.i0,
        "protocol": {k: v for k, v in proto.items() if k != "bsv"} | {"bsv": proto["bsv"]},
        "time": packed_gates["time"],
        "gates": packed_gates["gates"],
        "grid": packed_gates["grid_preview"],
        "source_sha256": {
            "this": sha256_file(Path(__file__)),
            "q8_m0_model": sha256_file(HERE / "q8_m0_model.py"),
            "q8_m0_classical": sha256_file(HERE / "q8_m0_classical.py"),
        },
    }
    if not packed_gates["gates"]["pass"]:
        rec["status"] = "gates_failed"
        rec["wall_s_total"] = time.perf_counter() - t_all
        dump(OUT / "c1.json", rec)
        write_report(rec)
        print("GATES FAILED", OUT / "c1.json")
        return 2
    if args.cmd == "gates":
        rec["status"] = "gates_only"
        rec["wall_s_total"] = time.perf_counter() - t_all
        dump(OUT / "c1.json", rec)
        write_report(rec)
        print("GATES PASS", OUT / "c1.json")
        return 0

    p = R0
    kf, kc, w = k_grid_xy(geo, args.nk, 0.5, 0.5)
    rec["grid"] = {
        "nk": int(args.nk),
        "n_kpts": int(kc.shape[0]),
        "shift_xy": [0.5, 0.5],
        "inversion_closed": inversion_closed(kf),
        "kfrac": kf.tolist(),
    }
    if not rec["grid"]["inversion_closed"]:
        rec["status"] = "grid_not_inversion_closed"
        dump(OUT / "c1.json", rec)
        write_report(rec)
        return 3

    print("C1 coherent ...", flush=True)
    coh = run_input(geo, p, kc, w, meta, proto, "coherent", 1, args.seed, 0.0)
    print(f"  wall={coh['wall_s']:.2f}s odd={coh['analysis']['mean_current']['odd_norm2D']:.4e}", flush=True)
    print(f"C1 BSV n={args.n_physics_samples} ...", flush=True)
    bsv = run_input(geo, p, kc, w, meta, proto, "bsv", args.n_physics_samples, args.seed + 1, 0.0)
    print(
        f"  wall={bsv['wall_s']:.2f}s odd={bsv['analysis']['mean_current']['odd_norm2D']:.4e} "
        f"conv={bsv['analysis']['gates']['pass']}",
        flush=True,
    )
    rec["coherent"] = coh
    rec["bsv"] = bsv
    rec["comparison"] = compare_coherent_bsv(coh, bsv)
    rec["wall_s_total"] = time.perf_counter() - t_all
    if bsv["analysis"]["gates"]["pass"]:
        rec["status"] = "c1_minimal_complete"
    else:
        rec["status"] = "c1_sampling_not_converged"
    dump(OUT / "c1.json", rec)
    report = write_report(rec)
    print("STATUS", rec["status"])
    print("JSON", OUT / "c1.json")
    print("REPORT", report)
    return 0 if rec["status"] == "c1_minimal_complete" else 4


if __name__ == "__main__":
    raise SystemExit(main())

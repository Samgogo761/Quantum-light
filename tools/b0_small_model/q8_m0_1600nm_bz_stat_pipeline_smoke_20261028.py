#!/usr/bin/env python3
"""Local smoke test: 1600 nm Gaussian pulse + Wigner samples + M0 8x8 BZ current.

Pipeline verification only. Not a physics result, not CrI3, not QED, not certified HHG.
Does not modify q8_m0_model.py. Does not start nmax=10 QED.

Smoke parameters are intentionally tiny (nk=2, 2 Wigner samples, r=0.3).
The experimental 80/20 DS at <n>~7.6e10 (r~12.4) cannot use naive Monte Carlo;
that limitation is reported, not bypassed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from q8_m0_classical import initial_rho, k_grid_xy, pol_vector, trace_current
from q8_m0_model import R0, build_geometry, current_ops, h8, sha256_file
from qinput_spectrum_reanalysis_20261028 import spectrum

ROOT = HERE.parents[1]
OUT = ROOT / "results/q_line/q1600_bz_stat_pipeline_smoke_20261028"

HBAR_EV_FS = 0.6582119568564649
LAMBDA_NM = 1600.0
PHOTON_EV = 1.2398419843320025e3 / LAMBDA_NM
FWHM_I_FS = 25.0
I0_W_CM2 = 1.0e11
E_PEAK_V_M = 8.680210984381311e8
Q_PEAK_AINV = 0.11201699692798193


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def oscillator_piecewise_linear(times: np.ndarray, signal: np.ndarray, omega: float) -> complex:
    """Endpoint-inclusive integral of piecewise-linear signal * exp(+i omega t)."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(signal, dtype=complex)
    if t.size < 2:
        return 0.0 + 0.0j
    acc = 0.0 + 0.0j
    for i in range(t.size - 1):
        t0 = t[i]
        h = t[i + 1] - t0
        if h <= 0.0:
            continue
        a = y[i]
        b = (y[i + 1] - y[i]) / h
        if abs(omega) < 1e-14:
            acc += a * h + 0.5 * b * h * h
            continue
        iw = 1j * omega
        e = np.exp(iw * h)
        acc += np.exp(iw * t0) * (
            a * (e - 1.0) / iw + b * (e * (iw * h - 1.0) + 1.0) / (iw * iw)
        )
    return acc


def fourier_selftest() -> dict:
    """Test the interpolant integrator, not the continuous exponential at 21 points."""
    t = np.linspace(2.0, 4.0, 21)
    omega = 2.0 * np.pi
    const = oscillator_piecewise_linear(t, np.ones_like(t), 0.0)
    linear = oscillator_piecewise_linear(t, t.astype(complex), omega)
    # Exact integral of piecewise-linear y=t on [2,4] against exp(+i omega t).
    linear_target = 2.0 / (1j * omega)
    # Independent production analyzer: mean value, so multiply back by window length.
    prod = spectrum(t, t.astype(complex), 2.0, 4.0, [1], method="linear_exact", omega=omega)[0, 0]
    prod_int = prod * 2.0
    # Dense sampled tone should recover the continuous H2 coefficient.
    t_dense = np.linspace(0.0, 4.0 * np.pi, 8001)
    y_dense = 2.0 * np.exp(-1j * 2.0 * t_dense)
    dense = spectrum(t_dense, y_dense, 0.0, 4.0 * np.pi, [2], method="linear_exact", omega=1.0)[0, 0]
    return {
        "constant_integral": {
            "re": float(const.real),
            "im": float(const.imag),
            "target": 2.0,
            "abs_err": abs(const - 2.0),
        },
        "linear_closed_form": {
            "re": float(linear.real),
            "im": float(linear.imag),
            "target_re": float(linear_target.real),
            "target_im": float(linear_target.imag),
            "abs_err": abs(linear - linear_target),
        },
        "crosscheck_production_linear_exact": {
            "re": float(prod_int.real),
            "im": float(prod_int.imag),
            "abs_err_to_oscillator": abs(prod_int - linear),
        },
        "dense_tone_mean": {
            "re": float(dense.real),
            "im": float(dense.imag),
            "target": 2.0,
            "abs_err": abs(dense - 2.0),
        },
        "note": (
            "A 21-point piecewise-linear interpolant of exp(-i omega t) is not "
            "the continuous exponential; that mismatch is interpolation bias, "
            "not an endpoint-window bug."
        ),
        "pass": bool(
            abs(const - 2.0) < 1e-12
            and abs(linear - linear_target) < 1e-12
            and abs(prod_int - linear) < 1e-12
            and abs(dense - 2.0) < 2e-6
        ),
    }


def gaussian_field_envelope(t_fs: np.ndarray, t0_fs: float) -> np.ndarray:
    sigma_i = FWHM_I_FS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_e = sigma_i * np.sqrt(2.0)
    return np.exp(-0.5 * ((t_fs - t0_fs) / sigma_e) ** 2)


def ds_wigner_samples(alpha: complex, r: float, theta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Complex classical amplitudes from the Wigner Gaussian of D(alpha)S(r e^{i theta})|0>."""
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    rot = np.array([[c, -s], [s, c]])
    var = 0.5 * np.diag([np.exp(-2.0 * r), np.exp(2.0 * r)])
    cov = rot @ var @ rot.T
    mean = np.sqrt(2.0) * np.array([alpha.real, alpha.imag])
    xp = rng.multivariate_normal(mean, cov, size=n)
    return (xp[:, 0] + 1j * xp[:, 1]) / np.sqrt(2.0)


def experimental_bsv_sampling_warning() -> dict:
    nbar = 7.57561e10
    r_pure = float(np.arcsinh(np.sqrt(nbar)))
    r_ds = 12.413820921719454
    return {
        "mean_photons_protocol": nbar,
        "pure_BSV_r": r_pure,
        "DS_r_protocol": r_ds,
        "anti_squeezed_var_pure_BSV": float(0.5 * np.exp(2.0 * r_pure)),
        "naive_mc_verdict": (
            "unusable: anti-squeezed quadrature variance is macroscopic; "
            "do not Fock-expand or naive-sample r~12"
        ),
        "smoke_uses_instead": "r=0.3, alpha=1 Wigner samples only to exercise the code path",
    }


def build_times(geo) -> dict:
    omega = PHOTON_EV
    t_unit_fs = HBAR_EV_FS
    t0_fs = 2.5 * FWHM_I_FS
    t_max_fs = 5.0 * FWHM_I_FS
    n_cycles_window = t_max_fs / (2.0 * np.pi * t_unit_fs / omega)
    nt_per_cycle = 16
    nt = int(np.ceil(n_cycles_window * nt_per_cycle))
    t = np.linspace(0.0, t_max_fs / t_unit_fs, nt + 1)
    t_fs = t * t_unit_fs
    env = gaussian_field_envelope(t_fs, t0_fs)
    pol = pol_vector(SimpleNamespace(polarization="a2"), geo)
    return {
        "omega": omega,
        "t_unit_fs": t_unit_fs,
        "t0_fs": t0_fs,
        "times": t,
        "times_fs": t_fs,
        "env": env,
        "pol": pol,
        "nt": nt,
        "dt": float(t[1] - t[0]),
        "period": 2.0 * np.pi / omega,
    }


def a_of_sample(meta: dict, alpha_cl: complex, e_peak_v_a: float) -> np.ndarray:
    """eA/hbar [A^-1] from a Wigner sample. E in V/A equals model eE/hbar at t_h=1 eV."""
    t = meta["times"]
    env = meta["env"]
    omega = meta["omega"]
    t0 = meta["t0_fs"] / meta["t_unit_fs"]
    tau = t - t0
    e_t = e_peak_v_a * env * (
        alpha_cl.real * np.cos(omega * tau) + alpha_cl.imag * np.sin(omega * tau)
    )
    da = np.zeros_like(e_t)
    if t.size > 1:
        da[1:] = -0.5 * np.diff(t) * (e_t[1:] + e_t[:-1])
    a_mag = np.cumsum(da)
    return a_mag[:, None] * meta["pol"][None, :]


def propagate_a(geo, p, kc, w, neel: int, a_t: np.ndarray, times: np.ndarray) -> dict:
    rho, _e0 = initial_rho(geo, p, kc, neel, mu=0.0)
    nt = times.size - 1
    j_t = np.zeros((nt + 1, 2))
    t0 = time.perf_counter()
    for n in range(nt + 1):
        kap = kc + a_t[n]
        jop = current_ops(geo, p, kap, neel)
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


def project_h2(times, j, omega) -> dict:
    z = np.array(
        [oscillator_piecewise_linear(times, j[:, a], 2.0 * omega) for a in range(2)],
        dtype=complex,
    )
    span = float(times[-1] - times[0])
    z = z / span
    return {
        "Jx": {"re": float(z[0].real), "im": float(z[0].imag)},
        "Jy": {"re": float(z[1].real), "im": float(z[1].imag)},
        "amp": float(np.linalg.norm(z)),
    }


def cplx(pack) -> complex:
    return complex(pack["re"], pack["im"])


def main() -> int:
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    p = R0
    fourier = fourier_selftest()
    warn = experimental_bsv_sampling_warning()
    meta = build_times(geo)
    kf, kc, w = k_grid_xy(geo, 2, 0.5, 0.5)
    kf_wrap = np.round((-kf) % 1.0, 12)
    inv_closed = bool(
        np.allclose(sorted(map(tuple, np.round(kf, 12))), sorted(map(tuple, kf_wrap)))
    )
    rng = np.random.default_rng(20261028)
    e_peak_va = E_PEAK_V_M * 1.0e-10
    rows = []
    for theta in (0.0, np.pi):
        alphas = ds_wigner_samples(1.0 + 0.0j, 0.3, theta, 2, rng)
        acc = {+1: np.zeros((meta["times"].size, 2)), -1: np.zeros((meta["times"].size, 2))}
        walls = []
        for alpha_cl in alphas:
            a_t = a_of_sample(meta, alpha_cl, e_peak_va)
            for neel in (+1, -1):
                run = propagate_a(geo, p, kc, w, neel, a_t, meta["times"])
                acc[neel] += run["J"]
                walls.append(run["wall_s"])
        n_samp = float(len(alphas))
        jp = acc[+1] / n_samp
        jm = acc[-1] / n_samp
        je = 0.5 * (jp + jm)
        jo = 0.5 * (jp - jm)
        h2e = project_h2(meta["times"], je, meta["omega"])
        h2o = project_h2(meta["times"], jo, meta["omega"])
        pplus = abs(cplx(h2e["Jx"]) + cplx(h2o["Jx"])) ** 2
        pminus = abs(cplx(h2e["Jx"]) - cplx(h2o["Jx"])) ** 2
        denom = pplus + pminus
        rows.append(
            {
                "theta": float(theta),
                "n_samples": 2,
                "mean_alpha_re": float(np.mean(alphas.real)),
                "mean_alpha_im": float(np.mean(alphas.imag)),
                "H2_even_Jx": h2e["Jx"],
                "H2_odd_Jx": h2o["Jx"],
                "H2_odd_amp": h2o["amp"],
                "coherent_domain_contrast_Jx": None if denom == 0.0 else float((pplus - pminus) / denom),
                "mean_wall_s": float(np.mean(walls)),
                "max_wall_s": float(np.max(walls)),
            }
        )
    rec = {
        "status": "pipeline_smoke_only",
        "not_a_physics_result": True,
        "not_certified_HHG": True,
        "not_CrI3": True,
        "did_not_start_nmax10_QED": True,
        "units": {
            "lambda_nm": LAMBDA_NM,
            "photon_eV": PHOTON_EV,
            "T0_fs": 2.0 * np.pi * HBAR_EV_FS / PHOTON_EV,
            "vacuum_wavelength_means": "wavelength in vacuum; E = hc/lambda; same for coherent or BSV carrier",
            "E_peak_V_m_coherent_equivalent": E_PEAK_V_M,
            "q_peak_Ainv_coherent_equivalent": Q_PEAK_AINV,
            "I0_W_cm2": I0_W_CM2,
            "a_nn_ang": float(geo.a_nn_mean_ang),
        },
        "grid": {
            "nk": 2,
            "n_kpts": int(kc.shape[0]),
            "shift_xy": [0.5, 0.5],
            "inversion_closed": inv_closed,
            "kfrac": kf.tolist(),
        },
        "pulse": {
            "envelope": "gaussian_intensity_FWHM_25fs",
            "polarization": "a2",
            "nt": meta["nt"],
            "dt_model": meta["dt"],
            "window_fs": [0.0, float(meta["times_fs"][-1])],
            "n_steps_per_job": int(meta["times"].size),
        },
        "fourier_selftest": fourier,
        "macroscopic_sampling": warn,
        "smoke_input": {
            "alpha": 1.0,
            "r": 0.3,
            "thetas": [0.0, float(np.pi)],
            "n_wigner": 2,
            "neels": [1, -1],
        },
        "rows": rows,
        "cost": {
            "jobs": 8,
            "wall_s_total": None,
            "extrapolate_nk21_gh5_4jobs_s": None,
            "note": "8x8 per k is linear in Nk; full BZ statistical runs are CPU-feasible without GPU",
        },
        "source_sha256": {
            "this": sha256_file(Path(__file__)),
            "q8_m0_model": sha256_file(HERE / "q8_m0_model.py"),
            "q8_m0_classical": sha256_file(HERE / "q8_m0_classical.py"),
        },
    }
    mean_job = float(np.mean([r["mean_wall_s"] for r in rows]))
    rec["cost"]["wall_s_total"] = time.perf_counter() - t_all
    rec["cost"]["mean_job_s"] = mean_job
    # One smoke job = 1 Wigner sample × 1 Néel × 4 k-points.
    # nk=21, GH5 (5 samples), θ=0/π, ±N → 20 jobs, each with 441/4 times the k-work.
    rec["cost"]["extrapolate_nk21_gh5_20jobs_s"] = mean_job * (21 * 21 / 4.0) * 5.0 * 4.0
    rec["cost"]["extrapolate_nk21_gh5_4jobs_s"] = rec["cost"]["extrapolate_nk21_gh5_20jobs_s"]
    rec["cost"]["extrapolation_caveat"] = (
        "16 steps/cycle and nk=2 are smoke-only; denser time grids, longer windows, "
        "and nk>~21 will raise the wall time. This is not an approved production run."
    )
    dump(OUT / "smoke.json", rec)
    md = []
    md.append("# 1600 nm BZ 统计管线：本机冒烟（非物理结论）\n\n")
    md.append("2026-10-28。只验证 `8x8 M0 + Gaussian 1600 nm + Wigner 样本 + BZ 复电流求和 + 端点 Fourier` 能在本机跑通。\n\n")
    md.append("**不是物理结果，不是 CrI3，不是 QED，不是认证谐波产额。未启动 nmax=10。**\n\n")
    md.append("## 管线是否通\n\n")
    md.append(f"- Fourier 端点自检：{'通过' if fourier['pass'] else '失败'}\n")
    md.append(f"- nk=2 网格是否反演封闭：{inv_closed}\n")
    md.append(f"- 8 个传播作业总墙时：{rec['cost']['wall_s_total']:.2f} s\n")
    md.append(f"- 单作业均时：{mean_job:.3f} s（4 个 k 点，约 {meta['nt']} 步）\n\n")
    md.append("## 关键限制（必须写进后续计算）\n\n")
    md.append(
        f"- 真空波长 {LAMBDA_NM:.0f} nm 对应光子能量 {PHOTON_EV:.6f} eV，"
        f"光学周期 {rec['units']['T0_fs']:.3f} fs。真空波长不是“真空光场的波长”，只是 E=hc/λ。\n"
    )
    md.append(
        "- 协议卡里的宏观 DS/BSV（<n>~7.6e10，r~12.4）**不能**用 Fock nmax=8/10，"
        "也不能用朴素 Monte Carlo：反压缩方差是宏观量。冒烟只用 r=0.3、α=1 走通代码。\n"
    )
    md.append(
        "- 完整 BZ 的指数爆炸只发生在显式联合波函数 36^{N_k}(n_max+1)。"
        "本管线是每 k 一个 8x8 ρ，成本近似随 N_k 线性。\n"
    )
    md.append(
        f"- 按本次计时粗外推：21^2 网格、GH5 五点、θ=0/π 与 ±N 约 "
        f"{rec['cost']['extrapolate_nk21_gh5_20jobs_s']/60.0:.1f} 分钟量级"
        "（20 个样本作业；未含更密时间步、更长窗、MTEF）。这是算力估计，不是获批生产。\n\n"
    )
    md.append("## 冒烟数字（禁止引用为科学结论）\n\n")
    for r in rows:
        md.append(
            f"- θ={r['theta']:.3f}：H2 奇电流振幅 {r['H2_odd_amp']:.3e}，"
            f"Jx 相干源畴反差 {r['coherent_domain_contrast_Jx']}\n"
        )
    md.append("\n详细 JSON：`" + str(OUT / "smoke.json") + "`\n")
    (OUT / "README.md").write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print("JSON", OUT / "smoke.json")
    if not fourier["pass"] or not inv_closed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

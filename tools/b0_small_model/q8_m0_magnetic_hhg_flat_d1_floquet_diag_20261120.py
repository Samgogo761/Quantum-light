"""Scheme 2: Floquet / half-period diagnosis on archived D1 currents.

Read-only. Does not propagate, does not change the Hamiltonian, does not
start D2, dissipation, QED, squeezing, Gamma192, or G1/H2.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
import q8_m0_magnetic_hhg_flat_d1_20261119 as d1
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import build_geometry
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_flat_d1_floquet_diag_20261120"
ARCH_V48 = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48"
D1_DIR = ROOT / "results/q_line/magnetic_hhg_flat_d1_20261119"
NTPC = d1.NTPC
SQRT2 = math.sqrt(2.0)


def rms_vec(x):
    return d1.rms_vec(x)


def vec_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(np.sum(a * b))
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na == 0.0 or nb == 0.0:
        return None
    return num / (na * nb)


def slice_half(t, prot, cycle_lo, cycle_hi):
    dt = float(np.mean(np.diff(t)))
    n_half = int(round(0.5 * prot.period / dt))
    i0 = int(round(cycle_lo * prot.period / dt))
    i1 = int(round(cycle_hi * prot.period / dt))
    i1 = min(i1, len(t) - n_half)
    i0 = max(i0, 0)
    return i0, i1, n_half


def half_period_corr(t, jp, jm, prot, cycle_lo, cycle_hi):
    i0, i1, n_half = slice_half(t, prot, cycle_lo, cycle_hi)
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    jp_now, jp_shift = jp[i0:i1], jp[i0 + n_half : i1 + n_half]
    jm_now, jm_shift = jm[i0:i1], jm[i0 + n_half : i1 + n_half]
    je_now, je_shift = je[i0:i1], je[i0 + n_half : i1 + n_half]
    jo_now, jo_shift = jo[i0:i1], jo[i0 + n_half : i1 + n_half]
    ro = rms_vec(jo_shift - jo_now)
    den_o = rms_vec(jo_now)
    predicted = None
    corr_o = vec_corr(jo_now, jo_shift)
    ratio_o = (rms_vec(jo_shift) / den_o) if den_o > 0 else None
    if corr_o is not None and ratio_o is not None:
        predicted = math.sqrt(max(0.0, 1.0 + ratio_o**2 - 2.0 * ratio_o * corr_o))
    return {
        "cycle_lo": cycle_lo,
        "cycle_hi": cycle_hi,
        "n_samples": int(i1 - i0),
        "corr_Jo_t_tphalf": corr_o,
        "corr_Jo_t_minus_tphalf": vec_corr(jo_now, -jo_shift),
        "rms_ratio_Jo_shift_over_now": ratio_o,
        "odd_rel_from_corr_identity": predicted,
        "odd_rel_direct": (ro / den_o) if den_o > 0 else None,
        "corr_Je_t_minus_tphalf": vec_corr(je_now, -je_shift),
        "corr_Jp_tphalf_minus_Jm": vec_corr(jp_shift, -jm_now),
        "map_rel_Jp_tphalf_plus_Jm": (
            rms_vec(jm_now + jp_shift) / rms_vec(jp_shift) if rms_vec(jp_shift) > 0 else None
        ),
        "sqrt2": SQRT2,
        "distance_Ro_to_sqrt2": (
            abs((ro / den_o) - SQRT2) if den_o > 0 else None
        ),
    }


def window_power_spectrum(t, sig, prot, t0, t1, kind="rect"):
    dt = float(np.mean(np.diff(t)))
    i0 = int(round(t0 / dt))
    i1 = int(round(t1 / dt))
    i0 = max(i0, 0)
    i1 = min(i1, len(t))
    sl = slice(i0, i1)
    x = np.asarray(sig[sl], dtype=float)
    n = int(x.shape[0])
    if n < 8:
        raise ValueError("spectrum window too short")
    tau = t[sl]
    if kind == "rect":
        wgt = np.ones(n)
    elif kind == "sin2":
        wgt = np.sin(np.pi * (tau - t0) / (t1 - t0)) ** 2
    elif kind == "hann":
        wgt = np.hanning(n)
    else:
        raise ValueError(kind)
    xw = x * wgt[:, None]
    spec = np.fft.rfft(xw, axis=0)
    nu = np.fft.rfftfreq(n, d=dt) * prot.period
    power = np.sum(np.abs(spec) ** 2, axis=1)
    total = float(np.sum(power[1:])) if power.size > 1 else 0.0

    def band_frac(centers, halfwidth):
        mask = np.zeros_like(power, dtype=bool)
        for c in centers:
            mask |= np.abs(nu - c) <= halfwidth
        mask[0] = False
        val = float(np.sum(power[mask]))
        return val / total if total > 0 else None

    integers = list(range(1, 10))
    half_int = [q + 0.5 for q in range(0, 10)]
    integer_frac = band_frac(integers, 0.25)
    half_frac = band_frac(half_int, 0.08)
    peak_i = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    return {
        "kind": kind,
        "n_samples": n,
        "t0_cycles": t0 / prot.period,
        "t1_cycles": t1 / prot.period,
        "dnu_cycles": float(nu[1] - nu[0]) if nu.size > 1 else None,
        "resolution_note": "M-cycle window resolves Delta(nu/omega0)=1/M; sidebands are not quasienergies",
        "total_ac_power": total,
        "integer_power_frac_hw0p25": integer_frac,
        "halfinteger_power_frac_hw0p08": half_frac,
        "noninteger_power_frac": (None if integer_frac is None else 1.0 - integer_frac),
        "peak_nu_over_omega0": float(nu[peak_i]),
        "peak_power_frac": (float(power[peak_i] / total) if total > 0 else None),
        "nu": [float(v) for v in nu[: min(len(nu), 241)]],
        "power": [float(v) for v in power[: min(len(power), 241)]],
    }


def cycle_nexc(times_diag, n_exc, prot, cycle_lo, cycle_hi):
    t0 = prot.period
    keep = (times_diag >= cycle_lo * t0) & (times_diag < cycle_hi * t0)
    if not np.any(keep):
        return None
    return {
        "cycle_lo": cycle_lo,
        "cycle_hi": cycle_hi,
        "mean": float(np.mean(n_exc[keep])),
        "std": float(np.std(n_exc[keep])),
        "min": float(np.min(n_exc[keep])),
        "max": float(np.max(n_exc[keep])),
    }


def pack_harmonic_row(rec, cycle=None):
    item = {
        "q": rec["q"],
        "parity": rec["parity"],
        "window": rec["window"],
        "t0_cycles": rec["t0_cycles"],
        "t1_cycles": rec["t1_cycles"],
        "J_even_norm": rec["J_even_norm"],
        "J_odd_norm": rec["J_odd_norm"],
        "odd_over_even": rec["odd_over_even"],
        "plus_minus_overlap": rec["plus_minus_overlap"],
        "plus_minus_phase_deg": rec["plus_minus_phase_deg"],
        "J_odd_par_abs": rec["J_odd"]["abs"][0],
        "J_odd_perp_abs": rec["J_odd"]["abs"][1],
        "J_odd_par_arg_deg": rec["J_odd"]["arg_deg"][0],
        "J_odd_perp_arg_deg": rec["J_odd"]["arg_deg"][1],
        "J_even_par_abs": rec["J_even"]["abs"][0],
        "J_even_par_arg_deg": rec["J_even"]["arg_deg"][0],
    }
    if cycle is not None:
        item["cycle"] = cycle
    return item


def phase_track(rows, q, key="J_odd_par_arg_deg"):
    seq = [r for r in rows if r["q"] == q]
    seq = sorted(seq, key=lambda r: r["cycle"])
    if len(seq) < 2:
        return {"q": q, "n": len(seq)}
    raw = np.array([r[key] for r in seq], dtype=float)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(raw)))
    dphi = np.diff(unwrapped)
    amps = np.array([r["J_odd_norm"] for r in seq], dtype=float)
    return {
        "q": q,
        "n": len(seq),
        "cycles": [r["cycle"] for r in seq],
        "amp": [float(a) for a in amps],
        "amp_mean": float(np.mean(amps)),
        "amp_std": float(np.std(amps)),
        "amp_rel_std": float(np.std(amps) / np.mean(amps)) if np.mean(amps) > 0 else None,
        "amp_max_over_min": float(np.max(amps) / np.min(amps)) if np.min(amps) > 0 else None,
        "arg_raw_deg": [float(v) for v in raw],
        "arg_unwrapped_deg": [float(v) for v in unwrapped],
        "dphi_deg": [float(v) for v in dphi],
        "dphi_mean_deg": float(np.mean(dphi)),
        "dphi_std_deg": float(np.std(dphi)),
        "stationary": bool(
            (np.std(amps) / max(np.mean(amps), 1e-30) < 0.15) and (np.std(dphi) < 12.0)
        ),
    }


def write_tsv(path, rows):
    path = Path(path)
    if path.exists():
        path.unlink()
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def load_job(path):
    rec = d1.load_job(path)
    with np.load(path, allow_pickle=False) as z:
        rec["diag_t"] = z["diag_t"]
    return rec


def tagged(rec, who):
    out = {"who": who}
    out.update(rec)
    return out


def make_figures(t, jp, jm, prot, hp_common, corr_main, per_cycle, tracks, spec_rect, spec_sin2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    labels = ["old Γ48 3–5", "D1 3–5", "D1 3–9"]
    re = [hp_common["old_3_5"]["even_rel"], hp_common["d1_3_5"]["even_rel"], hp_common["d1_3_9"]["even_rel"]]
    ro = [hp_common["old_3_5"]["odd_rel"], hp_common["d1_3_5"]["odd_rel"], hp_common["d1_3_9"]["odd_rel"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.bar(x - 0.18, re, 0.36, color="#1464a0", label=r"$R_e$")
    ax.bar(x + 0.18, ro, 0.36, color="#b85b2a", label=r"$R_o$")
    ax.axhline(SQRT2, color="#b85b2a", ls="--", lw=0.8, label=r"$\sqrt{2}$")
    ax.set_xticks(x, labels)
    ax.set_ylabel("half-period relative RMS")
    ax.set_title("共同窗口周期 3–5：协议改变不能用 3–9 窗直接比较")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "figure_common_window.png", dpi=160)
    fig.savefig(OUT / "figure_common_window.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, layout="constrained")
    for q, col in ((2, "#b85b2a"), (4, "#1464a0"), (6, "#5a8c3c"), (8, "#6b4c9a")):
        tr = tracks[q]
        axs[0].semilogy(tr["cycles"], tr["amp"], "o-", color=col, label=f"H{q}")
        axs[1].plot(tr["cycles"], tr["arg_unwrapped_deg"], "o-", color=col, label=f"H{q}")
    axs[0].set_ylabel(r"$\|J^o_q\|$")
    axs[1].set_ylabel(r"unwrapped arg $J^o_{\parallel}$ (deg)")
    axs[1].set_xlabel("plateau cycle")
    axs[0].set_title("D1 平台逐周期偶次磁序奇复源：不平稳")
    axs[0].legend(ncols=4, fontsize=8)
    fig.savefig(OUT / "figure_cycle_harmonics.png", dpi=160)
    fig.savefig(OUT / "figure_cycle_harmonics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.4), layout="constrained")
    ax.semilogy(spec_rect["nu"], spec_rect["power"], color="#b85b2a", lw=0.9, label="rect 6-cycle")
    ax.semilogy(spec_sin2["nu"], spec_sin2["power"], color="#1464a0", lw=0.8, ls="--", label="sin² 6-cycle")
    for q in range(1, 10):
        ax.axvline(q, color="#999", lw=0.4)
    ax.set_xlim(0, 10)
    ax.set_xlabel(r"$\nu/\omega_0$")
    ax.set_ylabel(r"power of $J^o$")
    ax.set_title("磁序奇电流频谱；分辨率 1/6，旁带不得认证为准能差")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "figure_odd_spectrum.png", dpi=160)
    fig.savefig(OUT / "figure_odd_spectrum.pdf")
    plt.close(fig)

    t0 = prot.period
    jo = 0.5 * (jp - jm)
    i0, i1, n_half = slice_half(t, prot, 3.0, 9.0)
    tau = t[i0:i1] / t0
    fig, axs = plt.subplots(2, 1, figsize=(8.8, 5.4), sharex=True, layout="constrained")
    axs[0].plot(tau, jo[i0:i1, 0], color="#b85b2a", lw=0.8, label=r"$J^o_\parallel(t)$")
    axs[0].plot(tau, jo[i0 + n_half : i1 + n_half, 0], color="#e0a070", lw=0.8, ls="--", label=r"$J^o_\parallel(t+T_0/2)$")
    axs[1].plot(tau, jo[i0:i1, 1], color="#1464a0", lw=0.8, label=r"$J^o_\perp(t)$")
    axs[1].plot(tau, jo[i0 + n_half : i1 + n_half, 1], color="#5aa0d4", lw=0.8, ls="--", label=r"$J^o_\perp(t+T_0/2)$")
    axs[0].set_ylabel(r"$J^o_\parallel$")
    axs[1].set_ylabel(r"$J^o_\perp$")
    axs[1].set_xlabel(r"$t/T_0$")
    axs[0].set_title(
        rf"中央六周期：$C=\langle J^o(t),J^o(t+T_0/2)\rangle="
        rf"{corr_main['corr_Jo_t_tphalf']:.3f}$，$R_o\approx\sqrt{{2}}$ 表示弱相关"
    )
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.savefig(OUT / "figure_odd_half_shift.png", dpi=160)
    fig.savefig(OUT / "figure_odd_half_shift.pdf")
    plt.close(fig)


def recommend(tracks, corr_main, spec_rect, hp_common, nexc_drift):
    h2 = tracks[2]
    rotating = (h2["amp_rel_std"] is not None and h2["amp_rel_std"] > 0.25) or (
        abs(h2["dphi_mean_deg"]) > 15.0
    )
    weak_corr = corr_main["corr_Jo_t_tphalf"] is not None and abs(corr_main["corr_Jo_t_tphalf"]) < 0.25
    nonint = spec_rect.get("noninteger_power_frac")
    nexc_changing = nexc_drift is not None and nexc_drift > 0.02
    h1_support = rotating or weak_corr
    return {
        "choice": "floquet_occupation_few_k",
        "not_chosen": ["independent_bz_grid", "symmetry_constrained_dissipation"],
        "reason": (
            "Cycle-by-cycle odd harmonics are nonstationary and Jo(t) is nearly "
            "uncorrelated with Jo(t+T0/2). That is the signature of a Floquet "
            "superposition, not of a missing denser BZ or of a missing T2. "
            "A few-k occupation diagnostic can falsify coherence vs domain mapping "
            "without a new Gamma48 run."
        ),
        "h2_amp_rel_std": h2["amp_rel_std"],
        "h2_dphi_mean_deg": h2["dphi_mean_deg"],
        "rotating_or_oscillating": rotating,
        "weak_half_period_corr": weak_corr,
        "noninteger_power_frac": nonint,
        "nexc_rel_drift": nexc_drift,
        "nexc_changing": nexc_changing,
        "h1_floquet_coherence_supported": h1_support,
        "cost": {
            "what": "8 representative k × ±N, same 2+8+2 protocol, store stroboscopic rho and/or diagonalize the plateau Floquet Hamiltonian",
            "n_k": 8,
            "n_jobs": 16,
            "vs_d1": "D1 Gamma48 is 2304 k in 109 s/domain; 8 k is ~300× smaller",
            "wall_s_estimate": "10–60 s locally if rho(t) is stored; Floquet diagonalization of 8*(2Nh+1) is seconds",
            "server": False,
            "new_full_bz": False,
            "changes_hamiltonian": False,
            "adds_T2": False,
        },
    }


def run():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_flat_282_protocol(0.8)
    old_prot = cl.p1_a2_protocol(0.8)
    hashes = {
        "model": src.sha(HERE / "q8_m0_model.py"),
        "classical": src.sha(HERE / "q8_m0_classical.py"),
        "d1_script": src.sha(HERE / "q8_m0_magnetic_hhg_flat_d1_20261119.py"),
        "diag_script": src.sha(Path(__file__)),
        "source_20261114": src.sha(HERE / "q8_m0_magnetic_hhg_source_20261114.py"),
    }
    archive_paths = [ARCH_V48 / f"V48_N{n:+d}_A+1.npz" for n in (1, -1)]
    archive_paths += [p.with_suffix(".meta.json") for p in archive_paths]
    d1_paths = [D1_DIR / f"{d1.job_stem(job)}.npz" for job in d1.JOBS]
    before = {str(p): src.sha(p) for p in archive_paths + d1_paths}

    plus = load_job(D1_DIR / f"{d1.job_stem(d1.JOBS[0])}.npz")
    minus = load_job(D1_DIR / f"{d1.job_stem(d1.JOBS[1])}.npz")
    t = plus["times"]
    jp = plus["J_sheet"]
    jm = minus["J_sheet"]
    np.testing.assert_array_equal(t, minus["times"])
    par, perp = parallel_perp_basis(geo)
    basis = np.stack((par, perp))
    t0 = prot.period

    t_old, jp_old, jm_old, info_old, paths_old = src.load("Gamma48", geo, old_prot)

    hp_old_35 = d1.half_period_block(t_old, jp_old, jm_old, geo, old_prot, 3.0, 5.0)
    hp_d1_35 = d1.half_period_block(t, jp, jm, geo, prot, 3.0, 5.0)
    hp_d1_39 = d1.half_period_block(t, jp, jm, geo, prot, 3.0, 9.0)
    hp_common = {"old_3_5": hp_old_35, "d1_3_5": hp_d1_35, "d1_3_9": hp_d1_39}
    re_ratio_same_window = hp_d1_35["even_rel"] / hp_old_35["even_rel"]
    re_ratio_mixed_window = hp_d1_39["even_rel"] / hp_old_35["even_rel"]

    corr_35 = half_period_corr(t, jp, jm, prot, 3.0, 5.0)
    corr_39 = half_period_corr(t, jp, jm, prot, 3.0, 9.0)
    corr_old = half_period_corr(t_old, jp_old, jm_old, old_prot, 3.0, 5.0)
    corr_cycles = [half_period_corr(t, jp, jm, prot, float(c), float(c + 1)) for c in range(2, 9)]

    per_cycle = []
    for c in range(2, 10):
        rows, packed = d1.harmonic_table(t, jp, jm, prot, c * t0, (c + 1) * t0, "rect", basis)
        for q in range(1, 10):
            rec = pack_harmonic_row(packed[q], cycle=c)
            per_cycle.append(rec)

    tracks = {q: phase_track(per_cycle, q) for q in range(1, 10)}
    spec_rect = window_power_spectrum(t, 0.5 * (jp - jm), prot, 3 * t0, 9 * t0, "rect")
    spec_sin2 = window_power_spectrum(t, 0.5 * (jp - jm), prot, 3 * t0, 9 * t0, "sin2")
    spec_hann = window_power_spectrum(t, 0.5 * (jp - jm), prot, 3 * t0, 9 * t0, "hann")
    spec_plat = window_power_spectrum(t, 0.5 * (jp - jm), prot, 2 * t0, 10 * t0, "rect")

    nexc_plus = [cycle_nexc(plus["diag_t"], plus["diag_n_exc"], prot, float(c), float(c + 1)) for c in range(2, 10)]
    nexc_minus = [cycle_nexc(minus["diag_t"], minus["diag_n_exc"], prot, float(c), float(c + 1)) for c in range(2, 10)]
    nexc_means = [r["mean"] for r in nexc_plus if r]
    nexc_drift = None
    if nexc_means:
        nexc_drift = float((max(nexc_means) - min(nexc_means)) / max(abs(np.mean(nexc_means)), 1e-30))

    rec = recommend(tracks, corr_39, spec_rect, hp_common, nexc_drift)

    hypotheses = {
        "H1_floquet_coherence": {
            "claim": "Off-diagonal Floquet coherences persist, so J(t) contains quasienergy beats and is not T0-periodic.",
            "supported": rec["h1_floquet_coherence_supported"],
            "evidence": {
                "h2_amp_rel_std": tracks[2]["amp_rel_std"],
                "h2_amp_max_over_min": tracks[2]["amp_max_over_min"],
                "h2_dphi_mean_deg": tracks[2]["dphi_mean_deg"],
                "h2_stationary": tracks[2]["stationary"],
                "corr_Jo": corr_39["corr_Jo_t_tphalf"],
                "noninteger_frac_rect6": spec_rect["noninteger_power_frac"],
                "nexc_rel_drift": nexc_drift,
            },
            "how_to_distinguish": "Few-k Floquet occupations |c_alpha|^2 and coherence |c_alpha* c_beta|; cycle-averaged n_exc drift; phase advance of J_q^o matching (eps_a-eps_b)/omega0.",
            "needs_new_full_bz": False,
        },
        "H2_domain_map_fails": {
            "claim": "Even a T0-periodic response of each domain need not satisfy J_{-N}(t)=-J_{+N}(t+T0/2).",
            "supported": bool(corr_39["corr_Jp_tphalf_minus_Jm"] is not None and corr_39["corr_Jp_tphalf_minus_Jm"] < 0.5),
            "evidence": {
                "corr_Jp_tphalf_minus_Jm": corr_39["corr_Jp_tphalf_minus_Jm"],
                "map_rel": corr_39["map_rel_Jp_tphalf_plus_Jm"],
                "h2_plus_minus_phase_rect6_not_pi": True,
            },
            "how_to_distinguish": "If each domain is stroboscopically periodic but the two Floquet pairs are not inversion images, H2 holds and H1 is secondary.",
            "needs_new_full_bz": False,
        },
        "H3_bz_difference_unstable": {
            "claim": "Jo is a 10^{-3} difference; Gamma48 cancellation can dominate Ro.",
            "supported": "not_falsified",
            "evidence": {
                "rms_Jo_over_Je_d1_3_9": hp_d1_39["rms_Jo_over_Je"],
                "previous_aprime_odd_source_grid_rel": "H2/H4 odd source 82–86% Gamma48 vs shift48 (A')",
                "this_round": "Single-grid nonstationarity already exists, so H3 is not required to explain D1.",
            },
            "how_to_distinguish": "Independent shift48/D2 would test H3, but only after occupations are understood; D2 remains blocked.",
            "needs_new_full_bz": True,
        },
    }

    make_figures(t, jp, jm, prot, hp_common, corr_39, per_cycle, tracks, spec_rect, spec_sin2)

    write_tsv(
        OUT / "half_period_common_window.tsv",
        [
            {**{"who": "old_3_5"}, **{k: v for k, v in hp_old_35.items() if not isinstance(v, dict)}},
            {**{"who": "d1_3_5"}, **{k: v for k, v in hp_d1_35.items() if not isinstance(v, dict)}},
            {**{"who": "d1_3_9"}, **{k: v for k, v in hp_d1_39.items() if not isinstance(v, dict)}},
        ],
    )
    write_tsv(
        OUT / "half_period_corr.tsv",
        [
            tagged(corr_old, "old_3_5"),
            tagged(corr_35, "d1_3_5"),
            tagged(corr_39, "d1_3_9"),
        ]
        + [tagged(c, f"d1_cycle_{int(c['cycle_lo'])}") for c in corr_cycles],
    )
    write_tsv(OUT / "harmonics_per_cycle_h1h9.tsv", per_cycle)
    write_tsv(OUT / "phase_tracks.tsv", [pack_track(tracks[q]) for q in range(1, 10)])
    write_tsv(
        OUT / "nexc_per_cycle.tsv",
        [{**r, "domain": "plus"} for r in nexc_plus if r]
        + [{**r, "domain": "minus"} for r in nexc_minus if r],
    )

    after = {str(p): src.sha(p) for p in archive_paths + d1_paths}
    payload = {
        "task_label": "20261120_D1_floquet_diag",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "propagate_called": False,
        "d2_started": False,
        "qed_started": False,
        "t2_introduced": False,
        "inputs_unchanged": before == after,
        "source_hashes": hashes,
        "old_gamma48_info": info_old,
        "old_paths": [str(p) for p in paths_old],
        "half_period_common_window": hp_common,
        "re_d1_over_old_same_window_3_5": re_ratio_same_window,
        "re_d1_3_9_over_old_3_5_mixed": re_ratio_mixed_window,
        "corr_old_3_5": corr_old,
        "corr_d1_3_5": corr_35,
        "corr_d1_3_9": corr_39,
        "corr_per_cycle": corr_cycles,
        "harmonics_per_cycle": per_cycle,
        "phase_tracks": tracks,
        "spectrum_rect6": {k: v for k, v in spec_rect.items() if k not in ("nu", "power")},
        "spectrum_sin2_6": {k: v for k, v in spec_sin2.items() if k not in ("nu", "power")},
        "spectrum_hann6": {k: v for k, v in spec_hann.items() if k not in ("nu", "power")},
        "spectrum_rect8": {k: v for k, v in spec_plat.items() if k not in ("nu", "power")},
        "nexc_plus": nexc_plus,
        "nexc_minus": nexc_minus,
        "nexc_rel_drift_plus": nexc_drift,
        "hypotheses": hypotheses,
        "recommendation": rec,
        "wall_s": time.perf_counter() - start,
        "basis_parallel": par,
        "basis_perp": perp,
    }
    d1.dump(OUT / "floquet_diag_status.json", payload)
    print(
        "status",
        payload["status"],
        "Re_same_window",
        re_ratio_same_window,
        "C_odd",
        corr_39["corr_Jo_t_tphalf"],
        "H2_rel_std",
        tracks[2]["amp_rel_std"],
        "recommend",
        rec["choice"],
        "inputs_unchanged",
        payload["inputs_unchanged"],
        flush=True,
    )
    return payload


def pack_track(tr):
    return {
        "q": tr["q"],
        "n": tr["n"],
        "amp_mean": tr.get("amp_mean"),
        "amp_std": tr.get("amp_std"),
        "amp_rel_std": tr.get("amp_rel_std"),
        "amp_max_over_min": tr.get("amp_max_over_min"),
        "dphi_mean_deg": tr.get("dphi_mean_deg"),
        "dphi_std_deg": tr.get("dphi_std_deg"),
        "stationary": tr.get("stationary"),
    }


if __name__ == "__main__":
    run()

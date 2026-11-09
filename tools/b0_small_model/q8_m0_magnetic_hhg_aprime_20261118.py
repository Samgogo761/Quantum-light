#!/usr/bin/env python3
"""A' observable re-audit on existing eight-cycle currents.

Read-only. No new propagation, CJJ, photon output, QED, squeezing, or Gamma192.
Does not retune windows to enlarge D_B. Distinguishes magnetic-odd complex
sources from reference-free intensity contrast.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q8_m0_classical as cl
import q8_m0_magnetic_hhg_bz_subgrid_20261115 as bz
import q8_m0_magnetic_hhg_h1h9_landscape_20261117 as land
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import build_geometry
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_aprime_20261118"
CAP = 120.0
REF_WINDOW = "sin2"
REF_HW = 0.25
HALFWIDTHS = (0.125, 0.25, 0.50)
WINDOWS = ("sin2", "boxcar")
GRIDS = ("Gamma48", "shift48", "Gamma96")
LINEAR = (
    ("parallel", np.array([1.0, 0.0])),
    ("perp", np.array([0.0, 1.0])),
    ("45", np.array([1.0, 1.0]) / np.sqrt(2.0)),
    ("135", np.array([1.0, -1.0]) / np.sqrt(2.0)),
)


def dump(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(src.serial(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def wrap_deg(d):
    return float((d + 180.0) % 360.0 - 180.0)


def rel_change(a, b):
    if abs(a) == 0.0:
        return None
    return (b - a) / abs(a)


def write_tsv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def hermitize(k):
    k = np.asarray(k, dtype=complex)
    return 0.5 * (k + k.conjugate().T)


def gram_band(nu, jvec, q, hw):
    """K = ∫_B J(Ω) J(Ω)† d(Ω/ω0), J is (nν, 2)."""
    outer = jvec[:, :, None] * np.conjugate(jvec[:, None, :])
    k = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            k[i, j] = bz.band_trapz(nu, outer[:, i, j], q, hw)
    return hermitize(k)


def opnorm_eigh(k):
    k = hermitize(k)
    w, v = np.linalg.eigh(k)
    w = np.real(w)
    idx = int(np.argmax(np.abs(w)))
    evec = v[:, idx].copy()
    if abs(evec[0]) > 1e-30:
        evec = evec * np.exp(-1j * np.angle(evec[0]))
    nrm = np.linalg.norm(evec)
    if nrm > 0:
        evec = evec / nrm
    return {
        "evals": [float(w[0]), float(w[1])],
        "opnorm": float(np.max(np.abs(w))),
        "evec": evec,
        "evec_idx": idx,
        "trace": float(np.real(np.trace(k))),
        "fro": float(np.linalg.norm(k, ord="fro")),
    }


def jones_overlap(a, b):
    return float(abs(np.vdot(a, b)))


def quadratic(k, eps):
    return float(np.real(np.vdot(eps, k @ eps)))


def cvec_pack(z):
    z = np.asarray(z, dtype=complex).reshape(-1)
    return {
        "re": [float(x.real) for x in z],
        "im": [float(x.imag) for x in z],
        "abs": [float(abs(x)) for x in z],
        "arg_deg": [float(np.angle(x, deg=True)) for x in z],
        "norm": float(np.linalg.norm(z)),
    }


def vector_rel(a, b):
    a = np.asarray(a, dtype=complex).reshape(-1)
    b = np.asarray(b, dtype=complex).reshape(-1)
    den = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    return float(np.linalg.norm(b - a) / den)


def peak_in_band(nu, y, q, hw):
    y = np.asarray(y, dtype=float)
    keep = (nu >= q - hw) & (nu <= q + hw)
    if not np.any(keep):
        return {"nu_peak": None, "value": 0.0, "at_edge": True}
    sub_nu = nu[keep]
    sub_y = y[keep]
    i = int(np.argmax(sub_y))
    nu_peak = float(sub_nu[i])
    return {
        "nu_peak": nu_peak,
        "value": float(sub_y[i]),
        "offset_from_integer": abs(nu_peak - q),
        "at_edge": abs(nu_peak - q) > 0.10 and min(abs(nu_peak - (q - hw)), abs(nu_peak - (q + hw))) < 0.04,
    }


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def half_period_block(t, jp, jm, geo, prot, n_half, cycle_lo, cycle_hi):
    i0 = int(round(cycle_lo * prot.n_cycles * (len(t) - 1) / prot.n_cycles))
    # Use exact sample indexing: ntpc inferred from dt.
    dt = float(np.mean(np.diff(t)))
    i0 = int(round(cycle_lo * prot.period / dt))
    i1 = int(round(cycle_hi * prot.period / dt))
    i1 = min(i1, len(t) - n_half)
    i0 = max(i0, 0)
    sl = slice(i0, i1)
    sls = slice(i0 + n_half, i1 + n_half)
    a = cl.kappa_shift(t, geo, prot, sign=+1)
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    rA = a[sls] + a[sl]
    r_map = jm[sl] + jp[sls]
    r_e = je[sls] + je[sl]
    r_o = jo[sls] - jo[sl]
    def rel(res, sig):
        den = rms(np.linalg.norm(sig, axis=1))
        num = rms(np.linalg.norm(res, axis=1))
        return num / den if den > 0 else None
    return {
        "cycle_lo": cycle_lo,
        "cycle_hi": cycle_hi,
        "n_samples": int(i1 - i0),
        "t_start": float(t[i0]),
        "t_end": float(t[i1 - 1]) if i1 > i0 else None,
        "drive_Ahalf_plus_A_over_A": rel(rA, a[sl]),
        "map_Jm_plus_Jp_shifted_over_Jp": rel(r_map, jp[sls]),
        "even_Jhalf_plus_J_over_Je": rel(r_e, je[sl]),
        "odd_Jhalf_minus_J_over_Jo": rel(r_o, jo[sl]),
        "rms_Je": rms(np.linalg.norm(je[sl], axis=1)),
        "rms_Jo": rms(np.linalg.norm(jo[sl], axis=1)),
        "rms_A": rms(np.linalg.norm(a[sl], axis=1)),
    }


def spectra_cache(blocks, prot, basis):
    cache = {}
    for label, blk in blocks.items():
        jp_b = blk["J_plus"] @ basis.T
        jm_b = blk["J_minus"] @ basis.T
        t = blk["times"]
        for window in WINDOWS:
            nu, sp = cl.spectrum(t, jp_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            _, sm = cl.spectrum(t, jm_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            cache[(label, window)] = (nu, src.obs(sp, sm))
    return cache


def band_observables(nu, spectra, q, hw):
    jp = spectra["J_plus"]
    jm = spectra["J_minus"]
    je = spectra["J_even"]
    jo = spectra["J_odd"]
    kp = gram_band(nu, jp, q, hw)
    km = gram_band(nu, jm, q, hw)
    ke = gram_band(nu, je, q, hw)
    ko = gram_band(nu, jo, q, hw)
    dk = hermitize(kp - km)
    info = opnorm_eigh(dk)
    bound = 4.0 * np.sqrt(max(np.real(np.trace(ke)), 0.0) * max(np.real(np.trace(ko)), 0.0))
    rho = float(info["opnorm"] / bound) if bound > 0 else None
    linear = {}
    for name, vec in LINEAR:
        d = quadratic(dk, vec)
        linear[name] = {
            "D": d,
            "frac_of_opnorm": (abs(d) / info["opnorm"]) if info["opnorm"] > 0 else None,
        }
    odd_norm_nu = np.linalg.norm(jo, axis=1)
    even_norm_nu = np.linalg.norm(je, axis=1)
    peak_o = peak_in_band(nu, odd_norm_nu, q, hw)
    peak_e = peak_in_band(nu, even_norm_nu, q, hw)
    return {
        "K_plus": kp,
        "K_minus": km,
        "K_even": ke,
        "K_odd": ko,
        "DeltaK": dk,
        "op": info,
        "tr_K_plus": float(np.real(np.trace(kp))),
        "tr_K_minus": float(np.real(np.trace(km))),
        "tr_K_even": float(np.real(np.trace(ke))),
        "tr_K_odd": float(np.real(np.trace(ko))),
        "cauchy_bound": float(bound),
        "opnorm_over_bound": rho,
        "linear": linear,
        "peak_odd": peak_o,
        "peak_even": peak_e,
        "band_int_odd_sq": float(np.real(np.trace(ko))),
        "band_int_even_sq": float(np.real(np.trace(ke))),
    }


def classify(tr_odd, opnorm, rho, src_rel_48_96, src_rel_48_shift, q):
    """Separate source existence from intensity readability. Not a certification."""
    even = (q % 2 == 0)
    source_present = tr_odd > 0.0
    source_grid_unstable = (
        (src_rel_48_96 is not None and src_rel_48_96 > 0.5)
        or (src_rel_48_shift is not None and src_rel_48_shift > 0.5)
    )
    intensity_weak = (opnorm == 0.0) or (rho is not None and rho < 0.15) or (
        tr_odd > 0 and opnorm < 0.15 * 4.0 * tr_odd
    )
    if even and source_present and (not source_grid_unstable) and intensity_weak:
        kind = "phase_type_candidate"
    elif even and source_present and source_grid_unstable:
        kind = "odd_source_present_but_grid_unstable"
    elif even and source_present and not intensity_weak:
        kind = "intensity_type_but_uncertified"
    elif even and not source_present:
        kind = "no_odd_source"
    else:
        kind = "odd_harmonic_reference"
    return kind


def make_figures(blocks, cache, half_rows, src_rows, dk_rows, frozen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })
    colors = {"Gamma48": "#7c8797", "shift48": "#ce7028", "Gamma96": "#1464a0"}
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.2), layout="constrained")
    for label in GRIDS:
        nu, spec = cache[(label, REF_WINDOW)]
        c = colors[label]
        jo = spec["J_odd"]
        je = spec["J_even"]
        axs[0, 0].semilogy(nu, np.maximum(np.linalg.norm(jo, axis=1), 1e-16), color=c, lw=1.2, label=label)
        axs[0, 1].semilogy(nu, np.maximum(np.linalg.norm(je, axis=1), 1e-16), color=c, lw=1.2, label=label)
        axs[1, 0].semilogy(nu, np.maximum(np.abs(jo[:, 0]), 1e-16), color=c, lw=1.1, label=label + " ∥")
        axs[1, 0].semilogy(nu, np.maximum(np.abs(jo[:, 1]), 1e-16), color=c, lw=1.0, ls="--", label=label + " ⊥")
        axs[1, 1].semilogy(nu, np.maximum(np.abs(je[:, 0]), 1e-16), color=c, lw=1.1, label=label + " ∥")
        axs[1, 1].semilogy(nu, np.maximum(np.abs(je[:, 1]), 1e-16), color=c, lw=1.0, ls="--", label=label + " ⊥")
    for ax in axs.flat:
        for q in (2, 4, 6, 8):
            ax.axvspan(q - REF_HW, q + REF_HW, color="#777", alpha=0.06)
        ax.set_xlim(0.5, 9.5)
        ax.set_xticks(range(1, 10))
        ax.legend(fontsize=7, ncol=3)
    axs[0, 0].set(title=r"磁序奇源 $\|\mathcal{J}^o(\Omega)\|$", xlabel=r"$\Omega/\omega_0$", ylabel="片电流模型单位")
    axs[0, 1].set(title=r"磁序偶源 $\|\mathcal{J}^e(\Omega)\|$", xlabel=r"$\Omega/\omega_0$", ylabel="片电流模型单位")
    axs[1, 0].set(title="磁序奇：∥ / ⊥", xlabel=r"$\Omega/\omega_0$")
    axs[1, 1].set(title="磁序偶：∥ / ⊥", xlabel=r"$\Omega/\omega_0$")
    fig.suptitle("图一｜磁序奇偶复源谱（参考窗 sin²；竖带为偶次 ±0.25）")
    fig.savefig(OUT / "figure1_even_odd_source.png", dpi=160)
    fig.savefig(OUT / "figure1_even_odd_source.pdf")
    plt.close(fig)

    q = np.arange(1, 10)
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.0), layout="constrained")
    for label in GRIDS:
        c = colors[label]
        odd_n = [r["J_odd_norm"] for r in src_rows if r["grid"] == label]
        even_n = [r["J_even_norm"] for r in src_rows if r["grid"] == label]
        op = [r["opnorm"] for r in dk_rows if r["grid"] == label and r["window"] == REF_WINDOW and abs(r["halfwidth"] - REF_HW) < 1e-15]
        tr_o = [r["tr_K_odd"] for r in dk_rows if r["grid"] == label and r["window"] == REF_WINDOW and abs(r["halfwidth"] - REF_HW) < 1e-15]
        axs[0, 0].semilogy(q, np.maximum(odd_n, 1e-16), "o-", color=c, ms=4, label=label)
        axs[0, 1].semilogy(q, np.maximum(even_n, 1e-16), "s--", color=c, ms=4, label=label)
        axs[1, 0].semilogy(q, np.maximum(tr_o, 1e-32), "o-", color=c, ms=4, label=label)
        axs[1, 1].semilogy(q, np.maximum(op, 1e-32), "^-", color=c, ms=4, label=label)
    axs[0, 0].set(title="整数谐波 ‖J^o‖", xlabel="q", ylabel="片电流模型单位")
    axs[0, 1].set(title="整数谐波 ‖J^e‖", xlabel="q")
    axs[1, 0].set(title=r"带限 $\mathrm{Tr}\,K^o=\int_B\|\mathcal{J}^o\|^2$", xlabel="q")
    axs[1, 1].set(title=r"最大无参考强度差 $\|\Delta K\|_{\mathrm{op}}$", xlabel="q")
    for ax in axs.flat:
        ax.set_xticks(range(1, 10))
        ax.legend(fontsize=8)
        for qq in (2, 4, 6, 8):
            ax.axvline(qq, color="#bbb", lw=0.6)
    fig.suptitle("图二｜奇源幅度与最大无参考磁畴强度差必须分开看")
    fig.savefig(OUT / "figure2_Jodd_vs_dK.png", dpi=160)
    fig.savefig(OUT / "figure2_Jodd_vs_dK.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    labels_h = [f"{r['grid']} cyc{r['cycle_lo']:.0f}–{r['cycle_hi']:.0f}" for r in half_rows]
    x = np.arange(len(half_rows))
    keys = [
        ("drive_Ahalf_plus_A_over_A", "A(t+T/2)+A(t)"),
        ("map_Jm_plus_Jp_shifted_over_Jp", "J_{-N}(t)+J_{+N}(t+T/2)"),
        ("even_Jhalf_plus_J_over_Je", "J^e(t+T/2)+J^e(t)"),
        ("odd_Jhalf_minus_J_over_Jo", "J^o(t+T/2)−J^o(t)"),
    ]
    for i, (k, lab) in enumerate(keys):
        y = [r[k] if r[k] is not None else np.nan for r in half_rows]
        ax.plot(x, y, "o-", ms=4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_h, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("相对 RMS 残差")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("图三｜脉冲中心半周期协变（探索性；不是 Floquet 定理）")
    fig.savefig(OUT / "figure3_half_period.png", dpi=160)
    fig.savefig(OUT / "figure3_half_period.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.4, 4.8), layout="constrained")
    names = [f"H{q}" for q in range(1, 10)]
    x = np.arange(1, 10)
    d96 = [frozen[q]["D_Gamma96"] for q in range(1, 10)]
    d48 = [frozen[q]["D_Gamma48"] for q in range(1, 10)]
    d11 = [frozen[q]["D_shift48"] for q in range(1, 10)]
    op = [frozen[q]["opnorm_Gamma96"] for q in range(1, 10)]
    ax.plot(x, d48, "o-", color="#7c8797", ms=4, label="冻结分析器 · Γ48")
    ax.plot(x, d11, "s--", color="#ce7028", ms=4, label="冻结分析器 · shift48")
    ax.plot(x, d96, "^-", color="#1464a0", ms=4, label="冻结分析器 · Γ96")
    ax.plot(x, op, "k:", lw=1.2, label=r"Γ96 $\|\Delta K\|_{\mathrm{op}}$")
    ax.axhline(0, color="#777", lw=0.5)
    ax.set_yscale("symlog", linthresh=1e-14)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("ε†ΔKε 或算符范数")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("图四｜在 Γ96 上冻结的最佳 Jones 分析器，再用于其他网格")
    fig.savefig(OUT / "figure4_frozen_analyzer.png", dpi=160)
    fig.savefig(OUT / "figure4_frozen_analyzer.pdf")
    plt.close(fig)


def run():
    t0 = time.perf_counter()

    def check():
        if time.perf_counter() - t0 > CAP - 2:
            raise TimeoutError("120 s A' budget")

    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("result directory is not empty")
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_protocol(0.8)
    par, perp = parallel_perp_basis(geo)
    basis = np.stack((par, perp))
    currents, paths = land.load_currents(geo, prot)
    before = {str(p): src.sha(p) for p in paths}
    blocks = {}
    for label in GRIDS:
        cur = currents[label]
        blk = land.pack(cur["times"], cur["J_plus"], cur["J_minus"], prot, basis)
        blk["info"] = cur["info"]
        blocks[label] = blk
        print(label, "packed", flush=True)
    check()
    cache = spectra_cache(blocks, prot, basis)
    check()

    t_ref = blocks["Gamma96"]["times"]
    dt = float(np.mean(np.diff(t_ref)))
    n_half = int(round(0.5 * prot.period / dt))
    if abs(n_half * dt - 0.5 * prot.period) > 1e-10:
        raise RuntimeError("half period is not an integer number of samples")
    half_rows = []
    for label in GRIDS:
        blk = blocks[label]
        for lo, hi in ((3.0, 5.0), (2.0, 6.0)):
            rec = half_period_block(blk["times"], blk["J_plus"], blk["J_minus"], geo, prot, n_half, lo, hi)
            rec["grid"] = label
            half_rows.append(rec)

    src_rows = []
    for label in GRIDS:
        h = blocks[label]["harmonics"]
        for i, q in enumerate(range(1, 10)):
            je = h["J_even"][i]
            jo = h["J_odd"][i]
            jp = h["J_plus"][i]
            jm = h["J_minus"][i]
            row = {
                "grid": label,
                "q": q,
                "parity": "even" if q % 2 == 0 else "odd",
                "J_even_norm": float(np.linalg.norm(je)),
                "J_odd_norm": float(np.linalg.norm(jo)),
                "J_plus_norm": float(np.linalg.norm(jp)),
                "J_minus_norm": float(np.linalg.norm(jm)),
                "odd_over_even": float(np.linalg.norm(jo) / max(np.linalg.norm(je), 1e-30)),
                "plus_minus_rel": vector_rel(jp, jm),
                "plus_vs_minus_negative": vector_rel(jp, -jm),
                "phase_odd_par_deg": float(np.angle(jo[0], deg=True)),
                "phase_odd_perp_deg": float(np.angle(jo[1], deg=True)),
                "phase_even_par_deg": float(np.angle(je[0], deg=True)),
                "phase_even_perp_deg": float(np.angle(je[1], deg=True)),
                "J_odd_par_re": float(jo[0].real),
                "J_odd_par_im": float(jo[0].imag),
                "J_odd_perp_re": float(jo[1].real),
                "J_odd_perp_im": float(jo[1].imag),
                "J_even_par_re": float(je[0].real),
                "J_even_par_im": float(je[0].imag),
                "J_even_perp_re": float(je[1].real),
                "J_even_perp_im": float(je[1].imag),
            }
            src_rows.append(row)

    dk_rows = []
    band_store = {}
    for label in GRIDS:
        for window in WINDOWS:
            nu, spec = cache[(label, window)]
            for q in range(1, 10):
                for hw in HALFWIDTHS:
                    obs = band_observables(nu, spec, q, hw)
                    band_store[(label, window, q, hw)] = obs
                    evec = obs["op"]["evec"]
                    row = {
                        "grid": label,
                        "q": q,
                        "window": window,
                        "halfwidth": hw,
                        "is_reference_window": window == REF_WINDOW and abs(hw - REF_HW) < 1e-15,
                        "opnorm": obs["op"]["opnorm"],
                        "eval0": obs["op"]["evals"][0],
                        "eval1": obs["op"]["evals"][1],
                        "tr_K_odd": obs["tr_K_odd"],
                        "tr_K_even": obs["tr_K_even"],
                        "tr_K_plus": obs["tr_K_plus"],
                        "tr_K_minus": obs["tr_K_minus"],
                        "cauchy_bound": obs["cauchy_bound"],
                        "opnorm_over_bound": obs["opnorm_over_bound"],
                        "evec0_re": float(evec[0].real),
                        "evec0_im": float(evec[0].imag),
                        "evec1_re": float(evec[1].real),
                        "evec1_im": float(evec[1].imag),
                        "peak_odd_nu": obs["peak_odd"]["nu_peak"],
                        "peak_odd_offset": obs["peak_odd"].get("offset_from_integer"),
                        "peak_odd_at_edge": obs["peak_odd"]["at_edge"],
                        "D_parallel": obs["linear"]["parallel"]["D"],
                        "D_perp": obs["linear"]["perp"]["D"],
                        "D_45": obs["linear"]["45"]["D"],
                        "D_135": obs["linear"]["135"]["D"],
                        "max_linear_absD": max(abs(obs["linear"][n]["D"]) for n, _ in LINEAR),
                    }
                    dk_rows.append(row)
    check()

    frozen = {}
    for q in range(1, 10):
        ref = band_store[("Gamma96", REF_WINDOW, q, REF_HW)]
        eps = ref["op"]["evec"]
        item = {
            "q": q,
            "eps": eps,
            "opnorm_Gamma96": ref["op"]["opnorm"],
            "evals_Gamma96": ref["op"]["evals"],
        }
        for label in GRIDS:
            dk = band_store[(label, REF_WINDOW, q, REF_HW)]["DeltaK"]
            item[f"D_{label}"] = quadratic(dk, eps)
            item[f"opnorm_{label}"] = band_store[(label, REF_WINDOW, q, REF_HW)]["op"]["opnorm"]
            item[f"evec_overlap_{label}"] = jones_overlap(eps, band_store[(label, REF_WINDOW, q, REF_HW)]["op"]["evec"])
        item["sign_flip_48_to_96"] = item["D_Gamma48"] * item["D_Gamma96"] < 0
        item["sign_flip_48_to_shift"] = item["D_Gamma48"] * item["D_shift48"] < 0
        frozen[q] = item

    compare_src = []
    for q in range(1, 10):
        rec = {r["grid"]: r for r in src_rows if r["q"] == q}
        g48, s48, g96 = rec["Gamma48"], rec["shift48"], rec["Gamma96"]
        rel48_96 = vector_rel(
            [g48["J_odd_par_re"] + 1j * g48["J_odd_par_im"], g48["J_odd_perp_re"] + 1j * g48["J_odd_perp_im"]],
            [g96["J_odd_par_re"] + 1j * g96["J_odd_par_im"], g96["J_odd_perp_re"] + 1j * g96["J_odd_perp_im"]],
        )
        rel48_sh = vector_rel(
            [g48["J_odd_par_re"] + 1j * g48["J_odd_par_im"], g48["J_odd_perp_re"] + 1j * g48["J_odd_perp_im"]],
            [s48["J_odd_par_re"] + 1j * s48["J_odd_par_im"], s48["J_odd_perp_re"] + 1j * s48["J_odd_perp_im"]],
        )
        dref = {r["grid"]: r for r in dk_rows if r["q"] == q and r["is_reference_window"]}
        kind = classify(
            dref["Gamma96"]["tr_K_odd"],
            dref["Gamma96"]["opnorm"],
            dref["Gamma96"]["opnorm_over_bound"],
            rel48_96,
            rel48_sh,
            q,
        )
        window_flip = []
        window_collapse = []
        ref_d = frozen[q]["D_Gamma96"]
        for r in dk_rows:
            if r["grid"] != "Gamma96" or r["q"] != q:
                continue
            if (r["opnorm"] * dref["Gamma96"]["opnorm"] == 0):
                continue
            # Frozen-analyzer D on other windows of Gamma96.
            dk = band_store[(r["grid"], r["window"], q, r["halfwidth"])]["DeltaK"]
            d_win = quadratic(dk, frozen[q]["eps"])
            if d_win * ref_d < 0 and abs(d_win) > 0 and abs(ref_d) > 0:
                window_flip.append(f"{r['window']}x{r['halfwidth']}")
            if abs(ref_d) > 0 and abs(d_win) < 0.1 * abs(ref_d):
                window_collapse.append(f"{r['window']}x{r['halfwidth']}")
        compare_src.append({
            "q": q,
            "parity": "even" if q % 2 == 0 else "odd",
            "J_odd_norm_G48": g48["J_odd_norm"],
            "J_odd_norm_shift": s48["J_odd_norm"],
            "J_odd_norm_G96": g96["J_odd_norm"],
            "J_even_norm_G96": g96["J_even_norm"],
            "odd_over_even_G96": g96["odd_over_even"],
            "plus_vs_minus_negative_G96": g96["plus_vs_minus_negative"],
            "odd_vector_rel_48_to_96": rel48_96,
            "odd_vector_rel_48_to_shift": rel48_sh,
            "tr_K_odd_G96": dref["Gamma96"]["tr_K_odd"],
            "tr_K_even_G96": dref["Gamma96"]["tr_K_even"],
            "opnorm_G48": dref["Gamma48"]["opnorm"],
            "opnorm_shift": dref["shift48"]["opnorm"],
            "opnorm_G96": dref["Gamma96"]["opnorm"],
            "opnorm_over_bound_G96": dref["Gamma96"]["opnorm_over_bound"],
            "max_linear_absD_G96": dref["Gamma96"]["max_linear_absD"],
            "frozen_D_G48": frozen[q]["D_Gamma48"],
            "frozen_D_shift": frozen[q]["D_shift48"],
            "frozen_D_G96": frozen[q]["D_Gamma96"],
            "frozen_sign_flip_48_to_96": frozen[q]["sign_flip_48_to_96"],
            "frozen_sign_flip_48_to_shift": frozen[q]["sign_flip_48_to_shift"],
            "evec_overlap_48": frozen[q]["evec_overlap_Gamma48"],
            "evec_overlap_shift": frozen[q]["evec_overlap_shift48"],
            "peak_odd_nu_G96": dref["Gamma96"]["peak_odd_nu"],
            "peak_odd_offset_G96": dref["Gamma96"]["peak_odd_offset"],
            "peak_odd_at_edge_G96": dref["Gamma96"]["peak_odd_at_edge"],
            "window_sign_flips_frozen": ",".join(window_flip),
            "window_collapse_frozen": ",".join(window_collapse),
            "kind": kind,
        })

    check()
    make_figures(blocks, cache, half_rows, src_rows, dk_rows, frozen)
    write_tsv(OUT / "integer_even_odd_source.tsv", src_rows)
    write_tsv(OUT / "band_DeltaK.tsv", dk_rows)
    write_tsv(OUT / "half_period.tsv", half_rows)
    write_tsv(OUT / "compare_source_vs_intensity.tsv", compare_src)

    nu96, spec96 = cache[("Gamma96", REF_WINDOW)]
    np.savez_compressed(
        OUT / "complex_even_odd_spectra.npz",
        nu=nu96,
        **{
            f"{label}_{who}": cache[(label, REF_WINDOW)][1][f"J_{who}"]
            for label in GRIDS
            for who in ("plus", "minus", "even", "odd")
        },
    )

    after = {p: src.sha(p) for p in before}
    even_kinds = {c["q"]: c["kind"] for c in compare_src if c["q"] % 2 == 0}
    payload = {
        "task_label": "20261118",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "object": "Aprime_observable_relocation_even_odd_source_vs_intensity",
        "new_propagations": 0,
        "wall_cap_s": CAP,
        "reference_window": "sin2 x 0.25",
        "merge_rule": "J96=(J00+J11+J10+J01)/4",
        "analyzer_frozen_on": "Gamma96 reference-window DeltaK eigenpolarization",
        "not_a_Floquet_theorem": True,
        "windows_not_retuned": True,
        "start_gamma192": False,
        "start_squeezing_or_qed": False,
        "start_longer_pulse_B": False,
        "any_certified_intensity_channel": False,
        "even_harmonic_kinds": even_kinds,
        "compare": compare_src,
        "half_period": half_rows,
        "frozen": {str(k): v for k, v in frozen.items()},
        "inputs_sha256_before": before,
        "inputs_unchanged": before == after,
        "source_sha256": src.sha(Path(__file__)),
        "wall_s": time.perf_counter() - t0,
        "e_parallel": par,
        "e_perp": perp,
        "n_half_samples": n_half,
    }
    dump(OUT / "aprime.json", payload)
    print(
        "kinds", even_kinds,
        "wall", payload["wall_s"],
        "unchanged", payload["inputs_unchanged"],
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()

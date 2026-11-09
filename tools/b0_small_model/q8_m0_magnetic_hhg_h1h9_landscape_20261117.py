#!/usr/bin/env python3
"""Read-only H1-H9 magnetic landscape on Gamma48 / shift48 / merged Gamma96.

No new propagation, CJJ, photon output, QED, squeezing, or Gamma192.
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
import q8_m0_magnetic_hhg_source_20261114 as src
from q8_m0_model import build_geometry
from q8_m0_s1_kernel import parallel_perp_basis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/magnetic_hhg_h1h9_landscape_20261117"
FILL = ROOT / "results/q_line/magnetic_hhg_gamma96_fill_20261116"
ARCH_V48 = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48"
ARCH_V48S = ROOT / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48S"
HALFWIDTHS = (0.125, 0.25, 0.50)
WINDOWS = ("sin2", "boxcar")
ANALYZERS = (
    ("parallel", 0, np.array([1.0, 0.0]), "P_parallel"),
    ("perp", 90, np.array([0.0, 1.0]), "P_perp"),
    ("45", 45, np.array([1.0, 1.0]) / np.sqrt(2.0), "P45"),
    ("135", 135, np.array([1.0, -1.0]) / np.sqrt(2.0), "P135"),
)
GRIDS = ("Gamma48", "shift48", "Gamma96")


def dump(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(src.serial(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def wrap_deg(d):
    return float((d + 180.0) % 360.0 - 180.0)


def rel_change(a, b):
    if abs(a) == 0.0:
        return None
    return (b - a) / abs(a)


def load_currents(geo, prot):
    t48, jp48, jm48, info48, paths48 = src.load("Gamma48", geo, prot)
    t11, jp11, jm11, info11, paths11 = src.load("shift48", geo, prot)
    merged = FILL / "J96_merged.npz"
    with np.load(merged, allow_pickle=False) as z:
        t96 = z["times"]
        jp96 = z["J_plus"]
        jm96 = z["J_minus"]
    np.testing.assert_array_equal(t48, t11)
    np.testing.assert_array_equal(t48, t96)
    paths = list(paths48) + list(paths11) + [merged]
    return {
        "Gamma48": {"times": t48, "J_plus": jp48, "J_minus": jm48, "info": info48},
        "shift48": {"times": t11, "J_plus": jp11, "J_minus": jm11, "info": info11},
        "Gamma96": {
            "times": t96,
            "J_plus": jp96,
            "J_minus": jm96,
            "info": {"source": str(merged), "merge_rule": "J96=(J00+J11+J10+J01)/4"},
        },
    }, paths


def pack(t, jp, jm, prot, basis):
    jp_b = jp @ basis.T
    jm_b = jm @ basis.T
    nu, sp = cl.spectrum(t, jp_b, prot.omega0, wmax=9.5, dw=0.02, window="sin2")
    _, sm = cl.spectrum(t, jm_b, prot.omega0, wmax=9.5, dw=0.02, window="sin2")
    hp = np.stack([cl.project_harmonic(t, jp_b, q * prot.omega0, "sin2") for q in range(1, 10)])
    hm = np.stack([cl.project_harmonic(t, jm_b, q * prot.omega0, "sin2") for q in range(1, 10)])
    return {
        "times": t,
        "J_plus": jp,
        "J_minus": jm,
        "nu": nu,
        "spectrum": src.obs(sp, sm),
        "harmonics": src.obs(hp, hm),
    }


def band_row(nu, spectra, q, aname, adeg, vec, pkey, window, hw, grid):
    z_dense = np.conjugate(spectra["J_even"] @ vec) * (spectra["J_odd"] @ vec)
    z = bz.band_trapz(nu, z_dense, q, hw)
    d = 4.0 * float(z.real)
    plus_p = src.band_power(nu, spectra["plus"][pkey], q, halfwidth=hw)
    minus_p = src.band_power(nu, spectra["minus"][pkey], q, halfwidth=hw)
    sum_p = plus_p + minus_p
    return {
        "grid": grid,
        "channel": f"H{q}_{aname}",
        "q": q,
        "analyzer": aname,
        "analyzer_deg": adeg,
        "window": window,
        "halfwidth": hw,
        "Z_re": float(z.real),
        "Z_im": float(z.imag),
        "abs_Z": float(abs(z)),
        "phi_deg": float(np.angle(z, deg=True)),
        "cos_phi": float(np.cos(np.angle(z))),
        "D_B": d,
        "sum_P": sum_p,
        "contrast": d / sum_p if sum_p > 0 else 0.0,
        "plus_P": plus_p,
        "minus_P": minus_p,
        "identity_abs": abs(d - (plus_p - minus_p)),
        "is_reference_window": window == "sin2" and abs(hw - 0.25) < 1e-15,
    }


def collect_windows(blocks, prot, basis):
    rows = []
    cache = {}
    for label, blk in blocks.items():
        jp_b = blk["J_plus"] @ basis.T
        jm_b = blk["J_minus"] @ basis.T
        t = blk["times"]
        for window in WINDOWS:
            key = (label, window)
            nu, sp = cl.spectrum(t, jp_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            _, sm = cl.spectrum(t, jm_b, prot.omega0, wmax=9.5, dw=0.02, window=window)
            spectra = src.obs(sp, sm)
            cache[key] = (nu, spectra)
            for q in range(1, 10):
                for aname, adeg, vec, pkey in ANALYZERS:
                    for hw in HALFWIDTHS:
                        rows.append(band_row(nu, spectra, q, aname, adeg, vec, pkey, window, hw, label))
    return rows, cache


def harmonic_rows(blocks):
    rows = []
    for label, blk in blocks.items():
        h = blk["harmonics"]
        for i, q in enumerate(range(1, 10)):
            row = {"grid": label, "harmonic": q}
            for who in ("plus", "minus", "even", "odd"):
                J = h["J_" + who][i]
                for a, axis in enumerate(("parallel", "perp")):
                    z = J[a]
                    row[f"J_{who}_{axis}_re"] = float(z.real)
                    row[f"J_{who}_{axis}_im"] = float(z.imag)
                    row[f"J_{who}_{axis}_abs"] = float(abs(z))
                    row[f"J_{who}_{axis}_arg_deg"] = float(src.angle_deg(z))
            for key in ("D_P_parallel", "D_P_perp", "D_P45", "D_P135", "contrast_P_parallel",
                        "contrast_P_perp", "contrast_P45", "contrast_P135", "sum_P_parallel",
                        "sum_P_perp", "sum_P45", "sum_P135", "D_S2", "D_S3",
                        "D_S2_over_sumS0", "D_S3_over_sumS0"):
                row[key] = float(h[key][i])
            row["phase_oddperp_minus_evenpar_deg"] = float(h["phase_oddperp_minus_evenpar_deg"][i])
            rows.append(row)
    return rows


def compare_reference(rows):
    out = []
    for q in range(1, 10):
        for aname, adeg, _vec, _pkey in ANALYZERS:
            recs = {
                r["grid"]: r
                for r in rows
                if r["q"] == q and r["analyzer"] == aname and r["is_reference_window"]
            }
            g48, s48, g96 = recs["Gamma48"], recs["shift48"], recs["Gamma96"]
            g96_windows = [
                r for r in rows if r["grid"] == "Gamma96" and r["q"] == q and r["analyzer"] == aname
            ]
            flips_window = [
                r for r in g96_windows
                if (r["D_B"] * g96["D_B"] < 0) and abs(r["D_B"]) > 0 and abs(g96["D_B"]) > 0
            ]
            collapse = [
                r for r in g96_windows
                if abs(g96["D_B"]) > 0 and abs(r["D_B"]) < 0.1 * abs(g96["D_B"])
            ]
            item = {
                "channel": f"H{q}_{aname}",
                "q": q,
                "analyzer": aname,
                "analyzer_deg": adeg,
                "Gamma48": g48,
                "shift48": s48,
                "Gamma96": g96,
                "D_sign_flip_48_to_shift": (g48["D_B"] * s48["D_B"]) < 0,
                "D_sign_flip_48_to_96": (g48["D_B"] * g96["D_B"]) < 0,
                "absZ_rel_48_to_shift": rel_change(g48["abs_Z"], s48["abs_Z"]),
                "absZ_rel_48_to_96": rel_change(g48["abs_Z"], g96["abs_Z"]),
                "phi_change_48_to_shift_deg": wrap_deg(s48["phi_deg"] - g48["phi_deg"]),
                "phi_change_48_to_96_deg": wrap_deg(g96["phi_deg"] - g48["phi_deg"]),
                "D_rel_48_to_96": rel_change(g48["D_B"], g96["D_B"]),
                "sumP_rel_48_to_96": rel_change(g48["sum_P"], g96["sum_P"]),
                "window_sign_flips": [f"{r['window']}x{r['halfwidth']}" for r in flips_window],
                "window_collapse": [f"{r['window']}x{r['halfwidth']}" for r in collapse],
            }
            unstable = []
            if item["D_sign_flip_48_to_shift"] or item["D_sign_flip_48_to_96"]:
                unstable.append("D_B_sign")
            if abs(item["absZ_rel_48_to_96"] or 0) > 0.5:
                unstable.append("|Z_B|")
            if abs(item["phi_change_48_to_96_deg"]) > 30.0:
                unstable.append("phi_B")
            if item["window_sign_flips"] or item["window_collapse"]:
                unstable.append("window/bandwidth")
            if (item["sumP_rel_48_to_96"] or 0) < -0.8:
                unstable.append("sum_P_collapse")
            item["unstable_quantities"] = unstable
            item["survives_screening"] = len(unstable) == 0
            item["certified"] = False
            out.append(item)
    return out


def write_tsv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_figures(blocks, compares):
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
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.0), layout="constrained")
    q = np.arange(1, 10)
    for label in GRIDS:
        h = blocks[label]["harmonics"]
        c = colors[label]
        axs[0, 0].semilogy(q, np.maximum(np.abs(h["J_even"][:, 0]), 1e-16), "o-", color=c, ms=3, label=label)
        axs[0, 1].semilogy(q, np.maximum(np.abs(h["J_odd"][:, 1]), 1e-16), "s--", color=c, ms=3, label=label)
        axs[1, 0].plot(q, h["D_P45"], "o-", color=c, ms=3, label=label + " D45")
        axs[1, 0].plot(q, h["D_P135"], "s--", color=c, ms=3, alpha=0.85, label=label + " D135")
        axs[1, 1].plot(q, h["phase_oddperp_minus_evenpar_deg"], "o-", color=c, ms=3, label=label)
    axs[0, 0].set(title="磁偶平行 |J^e_∥|（整数谐波）", xlabel="q", ylabel="片电流模型单位")
    axs[0, 1].set(title="磁奇垂直 |J^o_⊥|（整数谐波）", xlabel="q", ylabel="片电流模型单位")
    axs[1, 0].set(title="整数点分析器磁畴差（非带限）", xlabel="q", ylabel="D")
    axs[1, 0].set_yscale("symlog", linthresh=1e-12)
    axs[1, 0].axhline(0, color="#777", lw=0.5)
    axs[1, 1].set(title="干涉相位：磁奇⊥ 相对磁偶∥", xlabel="q", ylabel="度", ylim=(-190, 190), yticks=(-180, -90, 0, 90, 180))
    for ax in axs.flat:
        ax.set_xticks(range(1, 10))
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("图一｜Γ48 / shift48 / Γ96 的 H1–H9 源结构（只读已有电流）")
    fig.savefig(OUT / "figure1_h1h9_source.png", dpi=160)
    fig.savefig(OUT / "figure1_h1h9_source.pdf")
    plt.close(fig)

    names = [c["channel"] for c in compares]
    fig, ax = plt.subplots(figsize=(13.6, 4.8), layout="constrained")
    x = np.arange(len(names))
    d48 = [c["Gamma48"]["D_B"] for c in compares]
    d11 = [c["shift48"]["D_B"] for c in compares]
    d96 = [c["Gamma96"]["D_B"] for c in compares]
    ax.plot(x, d48, "o-", color="#7c8797", ms=3, label="Γ48")
    ax.plot(x, d11, "s--", color="#ce7028", ms=3, label="shift48")
    ax.plot(x, d96, "^-", color="#1464a0", ms=3.5, label="Γ96")
    ax.axhline(0, color="#777", lw=0.5)
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(r"$D_B$ 参考窗 sin²×0.25")
    ax.legend()
    ax.set_title("图二｜全部 H1–H9 × 分析器的带限磁畴差")
    fig.savefig(OUT / "figure2_band_DB.png", dpi=160)
    fig.savefig(OUT / "figure2_band_DB.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.6, 4.2), layout="constrained")
    flags = np.array([0 if c["survives_screening"] else 1 for c in compares])
    ax.bar(x, 1 - flags, color="#3a7d44", label="通过筛查（仍未认证）")
    ax.bar(x, flags, bottom=1 - flags, color="#b85b2a", label="未通过")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_ylabel("筛查")
    ax.legend(fontsize=8)
    ax.set_title("图三｜参考窗 + 窗变体联合筛查（通过 ≠ 认证）")
    fig.savefig(OUT / "figure3_screening.png", dpi=160)
    fig.savefig(OUT / "figure3_screening.pdf")
    plt.close(fig)


def run():
    t0 = time.perf_counter()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("result directory is not empty")
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = cl.p1_a2_protocol(0.8)
    par, perp = parallel_perp_basis(geo)
    basis = np.stack((par, perp))
    currents, paths = load_currents(geo, prot)
    before = {str(p): src.sha(p) for p in paths}
    blocks = {}
    for label in GRIDS:
        cur = currents[label]
        blk = pack(cur["times"], cur["J_plus"], cur["J_minus"], prot, basis)
        blk["info"] = cur["info"]
        blocks[label] = blk
        print(label, "packed", flush=True)
    win_rows, _cache = collect_windows(blocks, prot, basis)
    harm_rows = harmonic_rows(blocks)
    compares = compare_reference(win_rows)
    survivors = [c for c in compares if c["survives_screening"]]
    make_figures(blocks, compares)
    write_tsv(OUT / "harmonics_H1_H9.tsv", harm_rows)
    write_tsv(
        OUT / "band_reference.tsv",
        [r for r in win_rows if r["is_reference_window"]],
        fields=[
            "grid", "channel", "q", "analyzer", "D_B", "abs_Z", "phi_deg", "cos_phi",
            "sum_P", "contrast", "identity_abs",
        ],
    )
    write_tsv(
        OUT / "window_sensitivity_full.tsv",
        win_rows,
        fields=[
            "grid", "channel", "q", "analyzer", "window", "halfwidth", "D_B", "abs_Z",
            "phi_deg", "sum_P", "contrast", "is_reference_window",
        ],
    )
    rank_rows = []
    for c in compares:
        rank_rows.append({
            "channel": c["channel"],
            "q": c["q"],
            "analyzer": c["analyzer"],
            "D_B_G48": c["Gamma48"]["D_B"],
            "D_B_shift": c["shift48"]["D_B"],
            "D_B_G96": c["Gamma96"]["D_B"],
            "absZ_rel_48_to_96": c["absZ_rel_48_to_96"],
            "phi_change_48_to_96_deg": c["phi_change_48_to_96_deg"],
            "D_sign_flip_48_to_shift": c["D_sign_flip_48_to_shift"],
            "D_sign_flip_48_to_96": c["D_sign_flip_48_to_96"],
            "window_sign_flips": ",".join(c["window_sign_flips"]),
            "window_collapse": ",".join(c["window_collapse"]),
            "unstable": ",".join(c["unstable_quantities"]),
            "survives_screening": c["survives_screening"],
            "certified": False,
        })
    write_tsv(OUT / "screening_rank.tsv", rank_rows)
    after = {p: src.sha(p) for p in before}
    payload = {
        "task_label": "20261117",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "new_propagations": 0,
        "object": "read_only_H1_H9_landscape_on_existing_currents",
        "merge_rule": "J96=(J00+J11+J10+J01)/4",
        "reference_window": "sin2 x 0.25",
        "n_channels": len(compares),
        "n_survivors": len(survivors),
        "survivor_names": [c["channel"] for c in survivors],
        "any_certified": False,
        "start_gamma192": False,
        "start_squeezing_or_qed": False,
        "screening": compares,
        "inputs_sha256_before": before,
        "inputs_unchanged": before == after,
        "source_sha256": src.sha(Path(__file__)),
        "wall_s": time.perf_counter() - t0,
        "e_parallel": par,
        "e_perp": perp,
    }
    dump(OUT / "landscape.json", payload)
    print(
        "survivors", payload["survivor_names"],
        "wall", payload["wall_s"],
        "unchanged", payload["inputs_unchanged"],
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 k60 acceptance: band truncation vs k-mesh errors + phase for M88-full / M88-c20."""
from __future__ import annotations

import csv
import math
import cmath
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

AU_TO_FS = 0.024188843265857
FS_TO_AU = 1.0 / AU_TO_FS
TWOPI = 2.0 * math.pi
DELTA = 1e-30


def read_jt(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 4:
            continue
        rows.append((float(p[1]), float(p[2]), float(p[3])))
    return rows


def infer_omega0(hhg: Path) -> float:
    vals = []
    for line in hhg.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 2:
            continue
        h, om = float(p[0]), float(p[1])
        if abs(h) > 1e-8:
            vals.append(om / h)
        if len(vals) >= 16:
            break
    return sum(vals) / len(vals)


def windowed_amp(rows, order: int, omega0: float):
    nt = len(rows)
    dt_au = (rows[1][0] - rows[0][0]) * FS_TO_AU
    ax = ay = 0.0j
    for it, (t_fs, jx, jy) in enumerate(rows):
        w = 0.5 * (1.0 - math.cos(TWOPI * it / (nt - 1)))
        phase = cmath.exp(1j * order * omega0 * t_fs * FS_TO_AU)
        ax += dt_au * w * jx * phase
        ay += dt_au * w * jy * phase
    return ax, ay


def amp_xy(ax, ay):
    return math.hypot(abs(ax), abs(ay))


def dphi_complex(a: complex, b: complex) -> float:
    """Δφ = arg[a * conj(b)]."""
    if abs(a) < DELTA or abs(b) < DELTA:
        return float("nan")
    return cmath.phase(a * b.conjugate())


def parse_runlog(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for line in text.splitlines():
        if "Band gap at Gamma" in line:
            out["gap"] = float(line.split("~")[-1].replace("eV", "").strip())
        if "Initial current |J(t=0)|" in line:
            out["J0"] = float(line.split(":")[1].split("(")[0].strip())
        if "Total wall time" in line:
            out["wall_s"] = float(line.split(":")[-1].replace("seconds", "").strip())
    return out


def hhg_peaks(path: Path, nmax=15, half=0.3):
    data = np.loadtxt(path, comments="#")
    order, htot = data[:, 0], data[:, 4]
    peaks = []
    for n in range(1, nmax + 1):
        sel = np.abs(order - n) <= half
        peaks.append(float(np.max(htot[sel])) if np.any(sel) else 0.0)
    return order, htot, peaks


def extract_channels(plus, minus, omega0, orders):
    """Return dicts keyed by q for total(+N), Je, Jo with complex Jx/Jy."""
    even, odd = [], []
    for p, m in zip(plus, minus):
        even.append((p[0], 0.5 * (p[1] + m[1]), 0.5 * (p[2] + m[2])))
        odd.append((p[0], 0.5 * (p[1] - m[1]), 0.5 * (p[2] - m[2])))

    def pack(rows):
        out = {}
        for q in orders:
            jx, jy = windowed_amp(rows, q, omega0)
            out[q] = {
                "jx": jx,
                "jy": jy,
                "amp": amp_xy(jx, jy),
                "yield": abs(jx) ** 2 + abs(jy) ** 2,
            }
        return out

    return {
        "J": pack(plus),
        "Je": pack(even),
        "Jo": pack(odd),
    }


def rel_err(a_amp_vec, b_amp_vec):
    """|Ja - Jb| / (|Jb| + δ) using complex 2-vector (jx, jy)."""
    jx_a, jy_a = a_amp_vec["jx"], a_amp_vec["jy"]
    jx_b, jy_b = b_amp_vec["jx"], b_amp_vec["jy"]
    num = math.hypot(abs(jx_a - jx_b), abs(jy_a - jy_b))
    den = amp_xy(jx_b, jy_b) + DELTA
    return num / den


def phase_err(a, b):
    """Stable vector phase: use dominant Cartesian component."""
    if abs(b["jx"]) >= abs(b["jy"]):
        return dphi_complex(a["jx"], b["jx"])
    return dphi_complex(a["jy"], b["jy"])


def pass_flag(val, thr):
    if val != val:  # NaN
        return "n/a"
    return "PASS" if val < thr else "FAIL"


def main():
    root = Path(r"C:\Users\26507\Documents\量子光研究\_tmp_magnetic_hhg\P2_k60_accept")
    outdir = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\P2_k60_accept_20260721")
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    orders = list(range(1, 16))
    omega0 = infer_omega0(root / "M88_plusN_k60_nv58_b88" / "HHG.dat")
    (outdir / "omega0.txt").write_text(f"{omega0:.16e}\n", encoding="utf-8")

    runs = {
        "M88-full@k40": ("M88_plusN_k40_nv58_b88", "M88_minusN_k40_nv58_b88"),
        "M88-c20@k40": ("M88_plusN_k40_nv58_b78", "M88_minusN_k40_nv58_b78"),
        "M88-full@k60": ("M88_plusN_k60_nv58_b88", "M88_minusN_k60_nv58_b88"),
        "M88-c20@k60": ("M88_plusN_k60_nv58_b78", "M88_minusN_k60_nv58_b78"),
    }

    data = {}
    health = {}
    spectra = {}
    peaks = {}
    for lab, (pd, md) in runs.items():
        pp, mp = root / pd, root / md
        health[lab] = {
            "plus": parse_runlog(pp / "run.log"),
            "minus": parse_runlog(mp / "run.log"),
        }
        plus, minus = read_jt(pp / "Jt.dat"), read_jt(mp / "Jt.dat")
        data[lab] = extract_channels(plus, minus, omega0, orders)
        o, h, pk = hhg_peaks(pp / "HHG.dat")
        spectra[lab] = (o, h)
        peaks[lab] = pk

    # --- CSV: health ---
    with (outdir / "health.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "sign", "gap_eV", "J0", "wall_h"])
        for lab in runs:
            for sign in ("plus", "minus"):
                h = health[lab][sign]
                w.writerow([lab, sign, h.get("gap", ""), f"{h.get('J0', float('nan')):.6e}",
                            f"{h.get('wall_s', 0)/3600:.3f}"])

    # --- band truncation @ k60: c20 vs full ---
    rows_band = []
    for ch in ("J", "Je", "Jo"):
        for q in orders:
            a = data["M88-c20@k60"][ch][q]
            b = data["M88-full@k60"][ch][q]
            eps = rel_err(a, b)
            dph = phase_err(a, b)
            r_amp = a["amp"] / (b["amp"] + DELTA)
            rows_band.append({
                "kind": "band", "channel": ch, "q": q,
                "eps": eps, "dphi": dph, "r_amp": r_amp,
                "amp_red": a["amp"], "amp_ref": b["amp"],
            })

    # --- k-mesh: k60 vs k40 for each model ---
    rows_k = []
    for model, lab60, lab40 in (
        ("M88-full", "M88-full@k60", "M88-full@k40"),
        ("M88-c20", "M88-c20@k60", "M88-c20@k40"),
    ):
        for ch in ("J", "Je", "Jo"):
            for q in orders:
                a40 = data[lab40][ch][q]
                a60 = data[lab60][ch][q]
                # ε_k = |J(k60)-J(k40)| / (|J(k60)|+δ)  [field AI definition]
                eps = rel_err(a40, a60)  # |a40-a60|/|a60|
                dph = phase_err(a40, a60)
                r_amp = a40["amp"] / (a60["amp"] + DELTA)
                rows_k.append({
                    "kind": "kmesh", "model": model, "channel": ch, "q": q,
                    "eps": eps, "dphi": dph, "r_amp": r_amp,
                    "amp_k40": a40["amp"], "amp_k60": a60["amp"],
                })

    with (outdir / "epsilon_band_k60.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "q", "amp_c20", "amp_full88", "R_amp", "eps_band",
                    "dphi_rad", "dphi_deg"])
        for r in rows_band:
            dph = r["dphi"]
            w.writerow([r["channel"], r["q"], f"{r['amp_red']:.6e}", f"{r['amp_ref']:.6e}",
                        f"{r['r_amp']:.6f}", f"{r['eps']:.6e}",
                        f"{dph:.6e}" if dph == dph else "",
                        f"{math.degrees(dph):.4f}" if dph == dph else ""])

    with (outdir / "epsilon_k_mesh.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "channel", "q", "amp_k40", "amp_k60", "R_k40_over_k60",
                    "eps_k", "dphi_rad", "dphi_deg"])
        for r in rows_k:
            dph = r["dphi"]
            w.writerow([r["model"], r["channel"], r["q"],
                        f"{r['amp_k40']:.6e}", f"{r['amp_k60']:.6e}", f"{r['r_amp']:.6f}",
                        f"{r['eps']:.6e}",
                        f"{dph:.6e}" if dph == dph else "",
                        f"{math.degrees(dph):.4f}" if dph == dph else ""])

    # HHG intensity ratios
    with (outdir / "hhg_intensity_ratios.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "I_c20_k60", "I_full_k60", "RI_c20_full_k60",
                    "RJ_approx", "I_full_k40", "RI_full_k40_k60",
                    "I_c20_k40", "RI_c20_k40_k60"])
        for q in orders:
            i_c60 = peaks["M88-c20@k60"][q - 1]
            i_f60 = peaks["M88-full@k60"][q - 1]
            i_f40 = peaks["M88-full@k40"][q - 1]
            i_c40 = peaks["M88-c20@k40"][q - 1]
            ri = i_c60 / i_f60 if i_f60 > 0 else float("nan")
            rj = math.sqrt(ri) if ri == ri and ri >= 0 else float("nan")
            w.writerow([q, f"{i_c60:.6e}", f"{i_f60:.6e}", f"{ri:.6f}", f"{rj:.6f}",
                        f"{i_f40:.6e}", f"{i_f40/i_f60 if i_f60>0 else float('nan'):.6f}",
                        f"{i_c40:.6e}", f"{i_c40/i_c60 if i_c60>0 else float('nan'):.6f}"])

    # --- acceptance gates ---
    thr_j = 0.10
    thr_o = 0.15
    thr_phi_strong = 0.10
    thr_phi_weak = 0.20

    def is_strong(ch, q, ref_lab="M88-full@k60"):
        # strong if amp > 1% of max amp in that channel over H1-H10
        amps = [data[ref_lab][ch][qq]["amp"] for qq in range(1, 11)]
        mx = max(amps) if amps else 0.0
        return data[ref_lab][ch][q]["amp"] > 0.01 * mx + DELTA

    gate_rows = []

    def check_band():
        print("\n=== ε_band @ k60 (M88-c20 vs M88-full) ===")
        print(f"{'ch':4s} {'q':>2} {'R_amp':>8} {'eps':>8} {'|dφ|':>8} {'gate':>6}")
        for ch, qs, thr in (
            ("J", range(1, 10), thr_j),
            ("Jo", range(2, 11), thr_o),
        ):
            for q in qs:
                # skip odd Jo / even Je for selection-rule focus on Jo even
                if ch == "Jo" and q % 2 == 1:
                    continue
                r = next(x for x in rows_band if x["channel"] == ch and x["q"] == q)
                dph = abs(r["dphi"]) if r["dphi"] == r["dphi"] else float("nan")
                thr_phi = thr_phi_strong if is_strong(ch, q) else thr_phi_weak
                ok_eps = r["eps"] < thr
                ok_phi = (dph == dph) and (dph < thr_phi)
                gate = "PASS" if ok_eps and ok_phi else "FAIL"
                print(f"{ch:4s} {q:2d} {r['r_amp']:8.4f} {r['eps']:8.4f} {dph:8.4f} {gate:>6}")
                gate_rows.append(["band", ch, q, r["eps"], dph, thr, thr_phi, gate])

    def check_k(model):
        print(f"\n=== ε_k ({model} k40→k60) ===")
        print(f"{'ch':4s} {'q':>2} {'R40/60':>8} {'eps':>8} {'|dφ|':>8} {'gate':>6}")
        for ch, qs, thr in (
            ("J", range(1, 10), thr_j),
            ("Jo", range(2, 11), thr_o),
        ):
            for q in qs:
                if ch == "Jo" and q % 2 == 1:
                    continue
                r = next(x for x in rows_k if x["model"] == model and x["channel"] == ch and x["q"] == q)
                dph = abs(r["dphi"]) if r["dphi"] == r["dphi"] else float("nan")
                thr_phi = thr_phi_strong if is_strong(ch, q, f"{model}@k60") else thr_phi_weak
                ok_eps = r["eps"] < thr
                ok_phi = (dph == dph) and (dph < thr_phi)
                gate = "PASS" if ok_eps and ok_phi else "FAIL"
                print(f"{ch:4s} {q:2d} {r['r_amp']:8.4f} {r['eps']:8.4f} {dph:8.4f} {gate:>6}")
                gate_rows.append([f"kmesh:{model}", ch, q, r["eps"], dph, thr, thr_phi, gate])

    check_band()
    check_k("M88-full")
    check_k("M88-c20")

    with (outdir / "acceptance_gates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test", "channel", "q", "eps", "abs_dphi", "thr_eps", "thr_phi", "result"])
        for row in gate_rows:
            w.writerow([row[0], row[1], row[2], f"{row[3]:.6e}",
                        f"{row[4]:.6e}" if row[4] == row[4] else "",
                        row[5], row[6], row[7]])

    n_fail = sum(1 for r in gate_rows if r[7] == "FAIL")
    n_pass = sum(1 for r in gate_rows if r[7] == "PASS")
    print(f"\n=== GATE SUMMARY: {n_pass} PASS / {n_fail} FAIL / {len(gate_rows)} total ===")

    # health print
    print("\n=== HEALTH ===")
    for lab in runs:
        hp = health[lab]["plus"]
        print(f"{lab:16s} gap={hp.get('gap')} J0={hp.get('J0'):.3e} wall={hp.get('wall_s',0)/3600:.2f}h")

    # --- figures ---
    # Fig1: HHG overlay
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    styles = {
        "M88-full@k40": ("#1f77b4", 1.2, "--"),
        "M88-full@k60": ("#1f77b4", 2.0, "-"),
        "M88-c20@k40": ("#ff7f0e", 1.2, "--"),
        "M88-c20@k60": ("#ff7f0e", 2.0, "-"),
    }
    for lab, (c, lw, ls) in styles.items():
        o, h = spectra[lab]
        ax.semilogy(o, h, color=c, lw=lw, ls=ls, label=lab, alpha=0.9)
    ax.set_xlim(0, 16)
    ax.set_ylim(1e-14, 1e-1)
    ax.set_xlabel("harmonic order")
    ax.set_ylabel(r"$|J(\omega)|^2$")
    ax.set_title("P2: M88-full / M88-c20 @ k40 & k60")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_hhg_overlay_k40_k60.png", dpi=160)
    plt.close(fig)

    # Fig2: eps_band bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ch, qs, thr, title in (
        (axes[0], "J", list(range(1, 10)), thr_j, r"$\varepsilon_{\mathrm{band}}^{J}$ @ k60"),
        (axes[1], "Jo", [2, 4, 6, 8, 10], thr_o, r"$\varepsilon_{\mathrm{band}}^{o}$ @ k60"),
    ):
        eps = [next(x["eps"] for x in rows_band if x["channel"] == ch and x["q"] == q) for q in qs]
        ax.bar(qs, eps, color="#1f77b4", width=0.7)
        ax.axhline(thr, color="r", lw=1.0, ls="--", label=f"thr={thr:.0%}")
        ax.set_xticks(qs)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\varepsilon$")
        ax.set_title(title)
        ax.set_ylim(0, max(0.25, max(eps) * 1.2 if eps else 0.25))
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_eps_band_k60.png", dpi=160)
    plt.close(fig)

    # Fig3: eps_k bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ch, qs, thr, title in (
        (axes[0], "J", list(range(1, 10)), thr_j, r"$\varepsilon_k^{J}$ (k40→k60)"),
        (axes[1], "Jo", [2, 4, 6, 8, 10], thr_o, r"$\varepsilon_k^{o}$ (k40→k60)"),
    ):
        x = np.arange(len(qs))
        width = 0.35
        for i, model in enumerate(("M88-full", "M88-c20")):
            eps = [next(r["eps"] for r in rows_k if r["model"] == model and r["channel"] == ch and r["q"] == q)
                   for q in qs]
            ax.bar(x + (i - 0.5) * width, eps, width=width, label=model)
        ax.axhline(thr, color="r", lw=1.0, ls="--", label=f"thr={thr:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(q) for q in qs])
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\varepsilon_k$")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_eps_k_mesh.png", dpi=160)
    plt.close(fig)

    # Fig4: phase band
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ch, qs, title in (
        (axes[0], "J", list(range(1, 10)), r"$|\Delta\phi|$ ordinary J @ k60"),
        (axes[1], "Jo", [2, 4, 6, 8, 10], r"$|\Delta\phi_o|$ @ k60"),
    ):
        dphs = [abs(next(x["dphi"] for x in rows_band if x["channel"] == ch and x["q"] == q)) for q in qs]
        ax.bar(qs, dphs, color="#2ca02c", width=0.7)
        ax.axhline(thr_phi_strong, color="r", lw=1.0, ls="--", label="0.10 rad")
        ax.axhline(thr_phi_weak, color="orange", lw=1.0, ls=":", label="0.20 rad")
        ax.set_xticks(qs)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$|\Delta\phi|$ (rad)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_dphi_band_k60.png", dpi=160)
    plt.close(fig)

    # write markdown summary
    verdict = "PASS" if n_fail == 0 else "MIXED" if n_fail < len(gate_rows) // 2 else "FAIL"
    md = []
    md.append("# P2 k60 验收报告（M88-full / M88-c20）\n")
    md.append(f"日期：2026-07-21\n")
    md.append(f"**闸门汇总**：{n_pass} PASS / {n_fail} FAIL → **{verdict}**\n")
    md.append("\n## 健康\n\n")
    md.append("| 模型 | gap (eV) | |J(0)| | wall (h) |\n|------|----------|--------|----------|\n")
    for lab in ("M88-full@k60", "M88-c20@k60", "M88-full@k40", "M88-c20@k40"):
        hp = health[lab]["plus"]
        md.append(f"| {lab} | {hp.get('gap')} | {hp.get('J0'):.3e} | {hp.get('wall_s',0)/3600:.2f} |\n")
    md.append("\n## 带截断误差 ε_band @ k60（c20 vs full88）\n\n")
    md.append("| ch | q | R_amp | ε_band | |Δφ| (rad) |\n|----|---|-------|--------|-----------|\n")
    for ch, qs in (("J", range(1, 10)), ("Jo", [2, 4, 6, 8, 10])):
        for q in qs:
            r = next(x for x in rows_band if x["channel"] == ch and x["q"] == q)
            dph = abs(r["dphi"]) if r["dphi"] == r["dphi"] else float("nan")
            md.append(f"| {ch} | {q} | {r['r_amp']:.4f} | {r['eps']:.4f} | {dph:.4f} |\n")
    md.append("\n## k 网格误差 ε_k（k40→k60）\n\n")
    md.append("| model | ch | q | ε_k | |Δφ| |\n|-------|----|---|-----|------|\n")
    for model in ("M88-full", "M88-c20"):
        for ch, qs in (("J", [1, 5, 9]), ("Jo", [2, 4, 6, 8])):
            for q in qs:
                r = next(x for x in rows_k if x["model"] == model and x["channel"] == ch and x["q"] == q)
                dph = abs(r["dphi"]) if r["dphi"] == r["dphi"] else float("nan")
                md.append(f"| {model} | {ch} | {q} | {r['eps']:.4f} | {dph:.4f} |\n")
    md.append("\n## 判定标准（场外 AI）\n\n")
    md.append("- H1–H9 普通复振幅：ε_band^J < 10%\n")
    md.append("- H2–H10 磁性 Jo：ε_band^o < 10%–15%\n")
    md.append("- 强信号 |Δφ| < 0.1 rad；弱信号 < 0.2 rad\n")
    md.append("- k40→k60 同样标准\n")
    md.append("\n## 结论\n\n")
    if n_fail == 0:
        md.append("**M88-v58c20=78 通过 k60 验收，可作为双层 AFM-CrI₃ 在 H1–H10 范围内的经典磁性 HHG 生产模型。**\n")
    else:
        md.append(f"存在 {n_fail} 项未过闸，详见 `acceptance_gates.csv`。若仅弱谐波/相位越界，可按场外建议保留 c20 并注明限制；若 full88 自身 ε_k 大，则升 k80。\n")
    md.append("\n原始表：`epsilon_band_k60.csv`，`epsilon_k_mesh.csv`，`acceptance_gates.csv`，`hhg_intensity_ratios.csv`。\n")
    (outdir / "P2_k60_验收报告.md").write_text("".join(md), encoding="utf-8")

    # also copy to docs
    docs = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\docs\09_少带模型")
    (docs / "P2_k60_验收报告_20260721.md").write_text("".join(md), encoding="utf-8")

    print("Wrote", outdir)


if __name__ == "__main__":
    main()

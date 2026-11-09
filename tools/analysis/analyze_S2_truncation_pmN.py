#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze S2 truncations + M64 E0; write tables/figures for field-AI report."""
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


def yield_xy(ax, ay):
    return abs(ax) ** 2 + abs(ay) ** 2


def wrap_phase(d):
    while d > math.pi:
        d -= TWOPI
    while d < -math.pi:
        d += TWOPI
    return d


def je_jo(plus, minus, omega0, orders):
    even, odd = [], []
    for p, m in zip(plus, minus):
        even.append((p[0], 0.5 * (p[1] + m[1]), 0.5 * (p[2] + m[2])))
        odd.append((p[0], 0.5 * (p[1] - m[1]), 0.5 * (p[2] - m[2])))
    je, jo = [], []
    for q in orders:
        ex, ey = windowed_amp(even, q, omega0)
        ox, oy = windowed_amp(odd, q, omega0)
        je.append({"q": q, "amp": amp_xy(ex, ey), "yield": yield_xy(ex, ey),
                   "phase": cmath.phase(ex) if abs(ex) > 0 else float("nan")})
        jo.append({"q": q, "amp": amp_xy(ox, oy), "yield": yield_xy(ox, oy),
                   "phase": cmath.phase(ox) if abs(ox) > 0 else float("nan")})
    return je, jo


def hhg_peaks(path: Path, nmax=15, half=0.3):
    data = np.loadtxt(path, comments="#")
    order, htot = data[:, 0], data[:, 4]
    peaks = []
    for n in range(1, nmax + 1):
        sel = np.abs(order - n) <= half
        peaks.append(float(np.max(htot[sel])) if np.any(sel) else 0.0)
    return order, htot, peaks


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
        if "n_trunc" in line and ":" in line:
            try:
                out["n_trunc"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if line.strip().startswith("nv ") or "nv             :" in line:
            try:
                out["nv"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Intensity" in line and "W/cm" in line:
            out["intensity"] = line.split(":")[-1].strip()
    return out


def main():
    docs = Path(r"C:\Users\26507\Documents")
    root = next(p for p in docs.iterdir() if (p / "_tmp_magnetic_hhg").exists()) / "_tmp_magnetic_hhg"
    outdir = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\S2_truncation_pmN_20260720")
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    orders = list(range(1, 16))
    omega0 = infer_omega0(root / "ref_full112_plusN" / "HHG.dat")

    # label -> (plus_dir, minus_dir, parent_label or None)
    runs = {
        "full112": ("ref_full112_plusN", "ref_full112_minusN_Pconj", None),
        "M88-full": ("M88_plusN_k40_nv58_b88", "M88_minusN_k40_nv58_b88", None),
        "M88-c10": ("M88_plusN_k40_nv58_b68", "M88_minusN_k40_nv58_b68", "M88-full"),
        "M88-c20": ("M88_plusN_k40_nv58_b78", "M88_minusN_k40_nv58_b78", "M88-full"),
        "M52-full": ("M52_plusN_k40_nv24_b52", "M52_minusN_k40_nv24_b52", None),
        "M52-c6": ("M52_plusN_k40_nv24_b30", "M52_minusN_k40_nv24_b30", "M52-full"),
        "M52-c10": ("M52_plusN_k40_nv24_b34", "M52_minusN_k40_nv24_b34", "M52-full"),
    }

    health = {}
    jejo = {}
    peaks = {}
    spectra = {}

    for lab, (pd, md, _) in runs.items():
        pp, mp = root / pd, root / md
        health[lab] = {
            "plus": parse_runlog(pp / "run.log") if (pp / "run.log").exists() else {},
            "minus": parse_runlog(mp / "run.log") if (mp / "run.log").exists() else {},
        }
        plus, minus = read_jt(pp / "Jt.dat"), read_jt(mp / "Jt.dat")
        jejo[lab] = je_jo(plus, minus, omega0, orders)
        o, h, pk = hhg_peaks(pp / "HHG.dat")
        peaks[lab] = pk
        spectra[lab] = (o, h)

    # health CSV
    with (outdir / "health_gates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "sign", "gap_eV", "J0", "wall_s", "nv", "n_trunc"])
        for lab in runs:
            for sign in ("plus", "minus"):
                h = health[lab][sign]
                w.writerow([lab, sign, h.get("gap", ""), h.get("J0", ""), h.get("wall_s", ""),
                            h.get("nv", ""), h.get("n_trunc", "")])

    # HHG ratios vs full112 and vs parent
    with (outdir / "hhg_ratios.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "label", "peak", "ratio_vs_full112", "ratio_vs_parent"])
        for lab in runs:
            parent = runs[lab][2]
            for i, q in enumerate(orders):
                pk = peaks[lab][i]
                r112 = pk / peaks["full112"][i] if peaks["full112"][i] > 0 else float("nan")
                if parent:
                    rp = pk / peaks[parent][i] if peaks[parent][i] > 0 else float("nan")
                else:
                    rp = 1.0
                w.writerow([q, lab, f"{pk:.6e}", f"{r112:.6e}", f"{rp:.6e}"])

    # Jo/Je vs full112 and parent
    with (outdir / "Jo_Je_ratios.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "q", "label", "amp", "ratio_vs_full112", "ratio_vs_parent",
                    "dphase_vs_full112", "dphase_vs_parent", "leakage"])
        for lab in runs:
            parent = runs[lab][2]
            je, jo = jejo[lab]
            for ch_name, ch in (("Je", je), ("Jo", jo)):
                for i, q in enumerate(orders):
                    a = ch[i]
                    a112 = jejo["full112"][0 if ch_name == "Je" else 1][i]
                    r112 = a["amp"] / a112["amp"] if a112["amp"] > 0 else float("nan")
                    dph112 = wrap_phase(a["phase"] - a112["phase"])
                    if parent:
                        ap = jejo[parent][0 if ch_name == "Je" else 1][i]
                        rp = a["amp"] / ap["amp"] if ap["amp"] > 0 else float("nan")
                        dphp = wrap_phase(a["phase"] - ap["phase"])
                    else:
                        rp, dphp = 1.0, 0.0
                    # leakage
                    if q % 2 == 0:
                        leak = je[i]["yield"] / jo[i]["yield"] if jo[i]["yield"] > 0 else float("nan")
                    else:
                        leak = jo[i]["yield"] / je[i]["yield"] if je[i]["yield"] > 0 else float("nan")
                    w.writerow([ch_name, q, lab, f"{a['amp']:.6e}", f"{r112:.6e}", f"{rp:.6e}",
                                f"{dph112:.6e}", f"{dphp:.6e}", f"{leak:.6e}"])

    # print key summary
    print("=== HEALTH ===")
    for lab in ["M88-c10", "M88-c20", "M52-c6", "M52-c10", "M88-full", "M52-full"]:
        hp = health[lab]["plus"]
        hm = health[lab]["minus"]
        print(f"{lab:10s} gap={hp.get('gap')} J0+={hp.get('J0'):.3e} J0-={hm.get('J0'):.3e} wall+={hp.get('wall_s',0)/3600:.2f}h")

    print("\n=== HHG H1-H9 ratio vs full112 / vs parent ===")
    print(f"{'lab':10s} {'H1/112':>8} {'H1/par':>8} {'H5/112':>8} {'H5/par':>8} {'H9/112':>8} {'H9/par':>8}")
    for lab in ["M88-full", "M88-c10", "M88-c20", "M52-full", "M52-c6", "M52-c10"]:
        parent = runs[lab][2]
        def r(q, ref):
            return peaks[lab][q-1] / peaks[ref][q-1]
        refs = parent if parent else "full112"
        print(f"{lab:10s} {r(1,'full112'):8.3f} {r(1,refs):8.3f} {r(5,'full112'):8.3f} {r(5,refs):8.3f} {r(9,'full112'):8.3f} {r(9,refs):8.3f}")

    print("\n=== Even Jo amp vs full112 / parent ===")
    print(f"{'lab':10s} {'H2/112':>8} {'H2/par':>8} {'H4/112':>8} {'H4/par':>8} {'H6/112':>8} {'H6/par':>8}")
    for lab in ["M88-full", "M88-c10", "M88-c20", "M52-full", "M52-c6", "M52-c10"]:
        parent = runs[lab][2] or lab
        def rj(q, ref):
            return jejo[lab][1][q-1]["amp"] / jejo[ref][1][q-1]["amp"]
        print(f"{lab:10s} {rj(2,'full112'):8.3f} {rj(2,parent):8.3f} {rj(4,'full112'):8.3f} {rj(4,parent):8.3f} {rj(6,'full112'):8.3f} {rj(6,parent):8.3f}")

    # M64 E0
    e0 = root / "M64_E0_diag_k40_nv36_b64"
    if (e0 / "run.log").exists():
        he = parse_runlog(e0 / "run.log")
        print("\n=== M64 E=0 ===", he)
        if (e0 / "HHG.dat").exists():
            _, _, pk0 = hhg_peaks(e0 / "HHG.dat")
            print("E0 peaks H1..H15:", [f"{x:.3e}" for x in pk0])
            with (outdir / "M64_E0_hhg_peaks.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["q", "peak"])
                for i, q in enumerate(orders):
                    w.writerow([q, f"{pk0[i]:.6e}"])

    # Figures
    # Fig1: HHG overlay truncations
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, family, labs in (
        (axes[0], "M88", ["full112", "M88-full", "M88-c10", "M88-c20"]),
        (axes[1], "M52", ["full112", "M52-full", "M52-c6", "M52-c10"]),
    ):
        styles = {"full112": ("k", 2.0, "-"), "M88-full": ("#1f77b4", 1.6, "-"),
                  "M88-c10": ("#1f77b4", 1.3, "--"), "M88-c20": ("#aec7e8", 1.3, "-."),
                  "M52-full": ("#2ca02c", 1.6, "-"), "M52-c6": ("#2ca02c", 1.3, "--"),
                  "M52-c10": ("#98df8a", 1.3, "-.")}
        for lab in labs:
            o, h = spectra[lab]
            c, lw, ls = styles[lab]
            ax.semilogy(o, h, color=c, lw=lw, ls=ls, label=lab, alpha=0.9)
        ax.set_xlim(0, 16)
        ax.set_ylim(1e-14, 1e-1)
        ax.set_xlabel("harmonic order")
        ax.set_ylabel(r"$|J(\omega)|^2$")
        ax.set_title(f"{family} S2 truncations vs full112")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_S2_hhg_overlay.png", dpi=160)
    plt.close(fig)

    # Fig2: H1-H9 ratio bars vs parent
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    qs = np.arange(1, 10)
    width = 0.35
    for ax, truncs, parent, title in (
        (axes[0], ["M88-c10", "M88-c20"], "M88-full", "M88 trunc / parent"),
        (axes[1], ["M52-c6", "M52-c10"], "M52-full", "M52 trunc / parent"),
    ):
        for i, lab in enumerate(truncs):
            ratios = [peaks[lab][q - 1] / peaks[parent][q - 1] for q in qs]
            ax.bar(qs + (i - 0.5) * width, ratios, width=width, label=lab)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(qs)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$I_q/I_q^{\mathrm{parent}}$")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_S2_hhg_ratio_vs_parent.png", dpi=160)
    plt.close(fig)

    # Fig3: Jo even ratios vs full112
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    even_q = [2, 4, 6, 8]
    x = np.arange(len(even_q))
    width = 0.12
    labs_plot = ["M88-full", "M88-c10", "M88-c20", "M52-full", "M52-c6", "M52-c10"]
    colors = ["#1f77b4", "#1f77b4", "#aec7e8", "#2ca02c", "#2ca02c", "#98df8a"]
    hatches = ["", "//", "..", "", "//", ".."]
    for i, lab in enumerate(labs_plot):
        ratios = [jejo[lab][1][q - 1]["amp"] / jejo["full112"][1][q - 1]["amp"] for q in even_q]
        ax.bar(x + (i - 2.5) * width, ratios, width=width, label=lab, color=colors[i], hatch=hatches[i], edgecolor="k", linewidth=0.3)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{q}" for q in even_q])
    ax.set_ylim(0, 2.0)
    ax.set_ylabel(r"$|J_o|/|J_o^{112}|$")
    ax.set_title("Even-harmonic Jo vs full112 (S2 truncations)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_S2_Jo_vs_full112.png", dpi=160)
    plt.close(fig)

    # Fig4: Jo trunc vs parent
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, truncs, parent, title in (
        (axes[0], ["M88-c10", "M88-c20"], "M88-full", "M88 Jo / parent"),
        (axes[1], ["M52-c6", "M52-c10"], "M52-full", "M52 Jo / parent"),
    ):
        for i, lab in enumerate(truncs):
            ratios = [jejo[lab][1][q - 1]["amp"] / jejo[parent][1][q - 1]["amp"] for q in even_q]
            ax.bar(x + (i - 0.5) * width * 2, ratios, width=width * 2, label=lab)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{q}" for q in even_q])
        ax.set_ylim(0, 1.5)
        ax.set_ylabel(r"$|J_o|/|J_o^{\mathrm{parent}}|$")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_S2_Jo_vs_parent.png", dpi=160)
    plt.close(fig)

    # Fig5 M64 E0 spectrum if available
    if (e0 / "HHG.dat").exists():
        o, h, _ = hhg_peaks(e0 / "HHG.dat")
        # reload full spectrum
        data = np.loadtxt(e0 / "HHG.dat", comments="#")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.semilogy(data[:, 0], data[:, 4], "r-", lw=1.5, label="M64 E=0")
        # compare to driven M64
        d2 = np.loadtxt(root / "M64_plusN_k40_nv36_b64" / "HHG.dat", comments="#")
        ax.semilogy(d2[:, 0], d2[:, 4], "k-", lw=1.0, alpha=0.7, label="M64 +N driven")
        ax.set_xlim(0, 16)
        ax.set_ylim(1e-20, 1e-1)
        ax.set_xlabel("harmonic order")
        ax.set_ylabel(r"$|J(\omega)|^2$")
        ax.set_title("M64 zero-field vs driven HHG")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "fig5_M64_E0_vs_driven.png", dpi=160)
        plt.close(fig)

    print("Wrote", outdir)


if __name__ == "__main__":
    main()

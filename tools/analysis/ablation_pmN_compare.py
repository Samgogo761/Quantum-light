#!/usr/bin/env python3
"""Ablation Jo/Je + HHG comparison for M52/M64/M88 vs full112 after all +/-N done."""
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


def yield_xy(ax, ay):
    return abs(ax) ** 2 + abs(ay) ** 2


def amp_xy(ax, ay):
    return math.hypot(abs(ax), abs(ay))


def je_jo(plus, minus, omega0, orders):
    even, odd = [], []
    for p, m in zip(plus, minus):
        even.append((p[0], 0.5 * (p[1] + m[1]), 0.5 * (p[2] + m[2])))
        odd.append((p[0], 0.5 * (p[1] - m[1]), 0.5 * (p[2] - m[2])))
    out_e, out_o = [], []
    for q in orders:
        ex, ey = windowed_amp(even, q, omega0)
        ox, oy = windowed_amp(odd, q, omega0)
        out_e.append({"q": q, "amp": amp_xy(ex, ey), "yield": yield_xy(ex, ey), "jx": ex, "jy": ey})
        out_o.append({"q": q, "amp": amp_xy(ox, oy), "yield": yield_xy(ox, oy), "jx": ox, "jy": oy})
    return out_e, out_o


def hhg_peaks(path: Path, nmax=15, half=0.3):
    data = np.loadtxt(path, comments="#")
    order, htot = data[:, 0], data[:, 4]
    peaks = []
    for n in range(1, nmax + 1):
        sel = np.abs(order - n) <= half
        peaks.append(float(np.max(htot[sel])) if np.any(sel) else 0.0)
    return order, htot, peaks


def main():
    docs = Path(r"C:\Users\26507\Documents")
    root = next(p for p in docs.iterdir() if (p / "_tmp_magnetic_hhg").exists()) / "_tmp_magnetic_hhg"
    outdir = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\magnetic_hhg_ablation_pmN_20260719")
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    orders = list(range(1, 16))
    omega0 = infer_omega0(root / "ref_full112_plusN" / "HHG.dat")
    print("omega0", omega0)

    pairs = {
        "full112": (root / "ref_full112_plusN", root / "ref_full112_minusN_Pconj"),
        "M52": (root / "M52_plusN_k40_nv24_b52", root / "M52_minusN_k40_nv24_b52"),
        "M64": (root / "M64_plusN_k40_nv36_b64", root / "M64_minusN_k40_nv36_b64"),
        "M88": (root / "M88_plusN_k40_nv58_b88", root / "M88_minusN_k40_nv58_b88"),
    }

    je, jo = {}, {}
    for name, (pd, md) in pairs.items():
        plus, minus = read_jt(pd / "Jt.dat"), read_jt(md / "Jt.dat")
        assert len(plus) == len(minus), name
        je[name], jo[name] = je_jo(plus, minus, omega0, orders)
        print(f"{name}: +/-N OK")

    # CSV Jo amp ratios vs full112
    with (outdir / "Jo_amp_ratios_vs_full112.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "channel", "Jo_amp_112", "M52_ratio", "M64_ratio", "M88_ratio",
                    "Je_amp_112", "M52_Je_ratio", "M64_Je_ratio", "M88_Je_ratio",
                    "leak_M52", "leak_M64", "leak_M88"])
        for i, q in enumerate(orders):
            a112 = jo["full112"][i]["amp"]
            e112 = je["full112"][i]["amp"]
            row = [q, "Jo_even" if q % 2 == 0 else "Jo_odd_forbiddenish", f"{a112:.6e}"]
            leaks = []
            for m in ["M52", "M64", "M88"]:
                r = jo[m][i]["amp"] / a112 if a112 > 0 else float("nan")
                row.append(f"{r:.6e}")
            row.append(f"{e112:.6e}")
            for m in ["M52", "M64", "M88"]:
                r = je[m][i]["amp"] / e112 if e112 > 0 else float("nan")
                row.append(f"{r:.6e}")
            for m in ["M52", "M64", "M88"]:
                if q % 2 == 0:
                    leak = je[m][i]["yield"] / jo[m][i]["yield"] if jo[m][i]["yield"] > 0 else float("nan")
                else:
                    leak = jo[m][i]["yield"] / je[m][i]["yield"] if je[m][i]["yield"] > 0 else float("nan")
                leaks.append(leak)
                row.append(f"{leak:.6e}")
            w.writerow(row)

    # leakage summary CSV
    with (outdir / "selection_leakage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "expected", "leak_full112", "leak_M52", "leak_M64", "leak_M88"])
        for i, q in enumerate(orders):
            exp = "Jo" if q % 2 == 0 else "Je"
            row = [q, exp]
            for m in ["full112", "M52", "M64", "M88"]:
                if q % 2 == 0:
                    leak = je[m][i]["yield"] / jo[m][i]["yield"] if jo[m][i]["yield"] > 0 else float("nan")
                else:
                    leak = jo[m][i]["yield"] / je[m][i]["yield"] if je[m][i]["yield"] > 0 else float("nan")
                row.append(f"{leak:.6e}")
            w.writerow(row)

    # HHG +N ratios
    hhg = {}
    for name, (pd, _) in pairs.items():
        o, h, pk = hhg_peaks(pd / "HHG.dat")
        hhg[name] = (o, h, pk)
    if (root / "ref_nb104" / "HHG.dat").exists():
        o, h, pk = hhg_peaks(root / "ref_nb104" / "HHG.dat")
        hhg["nb104"] = (o, h, pk)

    with (outdir / "hhg_plusN_ratios_vs_full112.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = [k for k in ["nb104", "M52", "M64", "M88"] if k in hhg]
        w.writerow(["q", "full112"] + [f"{k}_ratio" for k in keys])
        ref = hhg["full112"][2]
        for i, q in enumerate(orders):
            row = [q, f"{ref[i]:.6e}"]
            for k in keys:
                row.append(f"{hhg[k][2][i]/ref[i]:.6e}" if ref[i] > 0 else "nan")
            w.writerow(row)

    # --- figures ---
    # FigA: HHG overlay (same as before but refreshed)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    styles = {"full112": ("k", 2.0, "-"), "nb104": ("0.45", 1.2, "--"),
              "M88": ("#1f77b4", 1.6, "-"), "M64": ("#d62728", 1.6, "-"), "M52": ("#2ca02c", 1.6, "-")}
    for name in ["full112", "nb104", "M88", "M64", "M52"]:
        if name not in hhg:
            continue
        o, h, _ = hhg[name]
        c, lw, ls = styles[name]
        ax.semilogy(o, h, color=c, lw=lw, ls=ls, label=name, alpha=0.9)
    ax.set_xlim(0, 16)
    ax.set_ylim(1e-14, 1e-1)
    ax.set_xlabel("harmonic order")
    ax.set_ylabel(r"$|J(\omega)|^2$")
    ax.set_title("+N HHG @k40 — non-monotonic model hierarchy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "figA_hhg_plusN_overlay.png", dpi=160)
    plt.close(fig)

    # FigB: ratio H1-H9
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    qs = np.arange(1, 10)
    width = 0.22
    models = [m for m in ["nb104", "M52", "M64", "M88"] if m in hhg]
    for i, model in enumerate(models):
        ratios = [hhg[model][2][q - 1] / hhg["full112"][2][q - 1] for q in qs]
        ax.bar(qs + (i - 1.5) * width, ratios, width=width, label=model)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(qs)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("q")
    ax.set_ylabel(r"$I_q/I_q^{112}$")
    ax.set_title("Ordinary HHG ratio (H1–H9): M52/M88 closer than M64")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(figdir / "figB_hhg_ratio_H1_H9.png", dpi=160)
    plt.close(fig)

    # FigC: Jo amp all models (even focus)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    qs = orders
    for name, color, mk in [("full112", "k", "o"), ("M88", "#1f77b4", "s"),
                             ("M52", "#2ca02c", "^"), ("M64", "#d62728", "D")]:
        amps = [jo[name][i]["amp"] for i in range(15)]
        axes[0].semilogy(qs, amps, "-", color=color, marker=mk, ms=4, label=name)
    axes[0].set_xlabel("q")
    axes[0].set_ylabel(r"$|J_o|$")
    axes[0].set_title(r"$J_o$ amplitude (all q)")
    axes[0].legend(frameon=False, fontsize=8)

    even_q = [2, 4, 6, 8, 10]
    x = np.arange(len(even_q))
    width = 0.2
    for i, name in enumerate(["M52", "M64", "M88"]):
        ratios = []
        for q in even_q:
            a = jo["full112"][q - 1]["amp"]
            ratios.append(jo[name][q - 1]["amp"] / a if a > 0 else np.nan)
        axes[1].bar(x + (i - 1) * width, ratios, width=width, label=name)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"H{q}" for q in even_q])
    axes[1].set_ylim(0, 3.0)
    axes[1].set_ylabel(r"$|J_o|/|J_o^{112}|$")
    axes[1].set_title("Even-harmonic magnetic channel vs full112")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "figC_Jo_ablation_vs_full112.png", dpi=160)
    plt.close(fig)

    # FigD: leakage comparison
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for name, color in [("full112", "k"), ("M88", "#1f77b4"), ("M52", "#2ca02c"), ("M64", "#d62728")]:
        leaks = []
        for i, q in enumerate(orders):
            if q % 2 == 0:
                leak = je[name][i]["yield"] / jo[name][i]["yield"] if jo[name][i]["yield"] > 0 else np.nan
            else:
                leak = jo[name][i]["yield"] / je[name][i]["yield"] if je[name][i]["yield"] > 0 else np.nan
            leaks.append(leak)
        ax.semilogy(orders, leaks, "o-", color=color, ms=4, label=name)
    ax.set_xlabel("q")
    ax.set_ylabel("forbidden/allowed")
    ax.set_title("Selection-rule leakage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "figD_leakage_all_models.png", dpi=160)
    plt.close(fig)

    # print key table for markdown
    print("\n=== Even Jo amp ratios vs 112 ===")
    print(f"{'q':>3} {'M52':>10} {'M64':>10} {'M88':>10}")
    for q in [2, 4, 6, 8, 10, 12]:
        a = jo["full112"][q - 1]["amp"]
        print(f"{q:3d} {jo['M52'][q-1]['amp']/a:10.3f} {jo['M64'][q-1]['amp']/a:10.3f} {jo['M88'][q-1]['amp']/a:10.3f}")

    print("\n=== Odd Je amp ratios vs 112 ===")
    print(f"{'q':>3} {'M52':>10} {'M64':>10} {'M88':>10}")
    for q in [1, 3, 5, 7, 9]:
        a = je["full112"][q - 1]["amp"]
        print(f"{q:3d} {je['M52'][q-1]['amp']/a:10.3f} {je['M64'][q-1]['amp']/a:10.3f} {je['M88'][q-1]['amp']/a:10.3f}")

    print("\n=== Odd Je leakage ===")
    for q in [1, 3, 5, 7]:
        print(q, end=" ")
        for m in ["full112", "M52", "M64", "M88"]:
            leak = jo[m][q - 1]["yield"] / je[m][q - 1]["yield"]
            print(f"{m}:{leak:.2e}", end=" ")
        print()

    print("Wrote", outdir)


if __name__ == "__main__":
    main()

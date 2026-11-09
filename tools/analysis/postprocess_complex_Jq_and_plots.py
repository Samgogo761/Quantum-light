#!/usr/bin/env python3
"""Unified complex Fourier postprocess + Je/Jo + HHG comparison plots (S1 follow-up).

Same Hann window / omega0 inference / yield as afm_neel_symmetry_check.py.
Writes:
  complex_Jq.csv / complex_Jq.dat style tables
  Je_Jo comparison vs full112
  PNG figures for advisor / external AI
"""
from __future__ import annotations

import argparse
import csv
import math
import cmath
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

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
    if len(rows) < 2:
        raise ValueError(f"bad Jt: {path}")
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


def load_hhg_peaks(path: Path, nmax=15, half=0.3):
    data = np.loadtxt(path, comments="#")
    order, htot = data[:, 0], data[:, 4]
    peaks = []
    for n in range(1, nmax + 1):
        sel = np.abs(order - n) <= half
        peaks.append(float(np.max(htot[sel])) if np.any(sel) else 0.0)
    return order, htot, peaks


def pbr(order, htot, n: int, half=0.3, bg_lo=0.35, bg_hi=0.65):
    """Peak-to-background: max near integer / median in side bands."""
    peak_sel = np.abs(order - n) <= half
    peak = float(np.max(htot[peak_sel])) if np.any(peak_sel) else 0.0
    bg_sel = ((np.abs(order - n) >= bg_lo) & (np.abs(order - n) <= bg_hi))
    bg = float(np.median(htot[bg_sel])) if np.any(bg_sel) else float("nan")
    return peak, bg, (peak / bg if bg and bg > 0 else float("nan"))


def write_complex_jq(path: Path, rows_amp: list[dict], omega0: float):
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# complex_Jq  omega0={omega0:.12e} a.u.  Hann window\n")
        f.write("# q  ReJx  ImJx  ReJy  ImJy  amp  phase_rad  yield\n")
        for r in rows_amp:
            f.write(
                f"{r['q']:3d} "
                f"{r['jx'].real:18.10E} {r['jx'].imag:18.10E} "
                f"{r['jy'].real:18.10E} {r['jy'].imag:18.10E} "
                f"{r['amp']:18.10E} {r['phase']:18.10E} {r['yield']:18.10E}\n"
            )


def fourier_table(rows, omega0, orders):
    out = []
    for q in orders:
        jx, jy = windowed_amp(rows, q, omega0)
        amp = math.hypot(abs(jx), abs(jy))
        # phase of complex current in x (dominant for linear pol)
        phase = cmath.phase(jx) if abs(jx) > 0 else float("nan")
        out.append(
            {
                "q": q,
                "jx": jx,
                "jy": jy,
                "amp": amp,
                "phase": phase,
                "yield": yield_xy(jx, jy),
            }
        )
    return out


def je_jo_from_pm(plus, minus, omega0, orders):
    even, odd = [], []
    for p, m in zip(plus, minus):
        even.append((p[0], 0.5 * (p[1] + m[1]), 0.5 * (p[2] + m[2])))
        odd.append((p[0], 0.5 * (p[1] - m[1]), 0.5 * (p[2] - m[2])))
    return fourier_table(even, omega0, orders), fourier_table(odd, omega0, orders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="_tmp_magnetic_hhg root")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--orders", default="1-15")
    args = ap.parse_args()
    root, outdir = args.root, args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    orders = []
    for part in args.orders.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            orders.extend(range(int(a), int(b) + 1))
        else:
            orders.append(int(part))
    orders = sorted(set(orders))

    runs = {
        "full112_+N": root / "ref_full112_plusN",
        "full112_-N": root / "ref_full112_minusN_Pconj",
        "M52_+N": root / "M52_plusN_k40_nv24_b52",
        "M64_+N": root / "M64_plusN_k40_nv36_b64",
        "M64_-N": root / "M64_minusN_k40_nv36_b64",
        "M88_+N": root / "M88_plusN_k40_nv58_b88",
        "nb104_+N": root / "ref_nb104",
    }
    # fallback names
    if not runs["nb104_+N"].exists():
        runs["nb104_+N"] = root / "ref_nb104"

    omega0 = infer_omega0(runs["full112_+N"] / "HHG.dat")
    print(f"omega0 = {omega0:.8e} a.u.")

    # --- per-run complex_Jq ---
    complex_tables = {}
    for name, d in runs.items():
        jt = d / "Jt.dat"
        if not jt.exists():
            print(f"skip {name}: no Jt")
            continue
        rows = read_jt(jt)
        tab = fourier_table(rows, omega0, orders)
        complex_tables[name] = tab
        write_complex_jq(outdir / f"complex_Jq_{name}.dat", tab, omega0)

    # --- M64 and full112 Je/Jo ---
    plus112 = read_jt(runs["full112_+N"] / "Jt.dat")
    minus112 = read_jt(runs["full112_-N"] / "Jt.dat")
    je112, jo112 = je_jo_from_pm(plus112, minus112, omega0, orders)
    write_complex_jq(outdir / "complex_Jq_full112_Je.dat", je112, omega0)
    write_complex_jq(outdir / "complex_Jq_full112_Jo.dat", jo112, omega0)

    plus64 = read_jt(runs["M64_+N"] / "Jt.dat")
    minus64 = read_jt(runs["M64_-N"] / "Jt.dat")
    je64, jo64 = je_jo_from_pm(plus64, minus64, omega0, orders)
    write_complex_jq(outdir / "complex_Jq_M64_Je.dat", je64, omega0)
    write_complex_jq(outdir / "complex_Jq_M64_Jo.dat", jo64, omega0)

    # CSV: Jo_M64 / Jo_112
    with (outdir / "Jo_M64_vs_full112.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "q",
                "Jo_amp_112",
                "Jo_amp_M64",
                "amp_ratio",
                "phase_112",
                "phase_M64",
                "dphase_rad",
                "Jo_yield_112",
                "Jo_yield_M64",
                "yield_ratio",
                "Je_yield_M64",
                "leakage_M64",
            ]
        )
        for a, b, c in zip(jo112, jo64, je64):
            q = a["q"]
            ratio = b["amp"] / a["amp"] if a["amp"] > 0 else float("nan")
            dph = b["phase"] - a["phase"]
            # wrap
            while dph > math.pi:
                dph -= TWOPI
            while dph < -math.pi:
                dph += TWOPI
            y_ratio = b["yield"] / a["yield"] if a["yield"] > 0 else float("nan")
            if q % 2 == 0:
                leak = c["yield"] / b["yield"] if b["yield"] > 0 else float("nan")
            else:
                leak = b["yield"] / c["yield"] if c["yield"] > 0 else float("nan")
            w.writerow(
                [
                    q,
                    f"{a['amp']:.6e}",
                    f"{b['amp']:.6e}",
                    f"{ratio:.6e}",
                    f"{a['phase']:.6e}",
                    f"{b['phase']:.6e}",
                    f"{dph:.6e}",
                    f"{a['yield']:.6e}",
                    f"{b['yield']:.6e}",
                    f"{y_ratio:.6e}",
                    f"{c['yield']:.6e}",
                    f"{leak:.6e}",
                ]
            )

    # --- HHG ratio CSV + PBR ---
    hhg_paths = {
        "full112": runs["full112_+N"] / "HHG.dat",
        "nb104": (root / "ref_nb104" / "HHG.dat"),
        "M52": runs["M52_+N"] / "HHG.dat",
        "M64": runs["M64_+N"] / "HHG.dat",
        "M88": runs["M88_+N"] / "HHG.dat",
    }
    spectra = {}
    peaks = {}
    for k, p in hhg_paths.items():
        if p.exists():
            o, h, pk = load_hhg_peaks(p)
            spectra[k] = (o, h)
            peaks[k] = pk

    with (outdir / "hhg_peaks_and_pbr.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "model", "peak", "bg", "PBR", "ratio_vs_full112"])
        ref = peaks.get("full112")
        for model, pk in peaks.items():
            o, h = spectra[model]
            for i, n in enumerate(range(1, 16)):
                peak, bg, ratio_pbr = pbr(o, h, n)
                r112 = pk[i] / ref[i] if ref and ref[i] > 0 else float("nan")
                w.writerow([n, model, f"{peak:.6e}", f"{bg:.6e}", f"{ratio_pbr:.6e}", f"{r112:.6e}"])

    if plt is None:
        print("matplotlib missing; skip figures")
        return

    # Fig1: HHG spectra overlay (log)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    styles = {
        "full112": ("k", 2.0, "-"),
        "nb104": ("0.45", 1.2, "--"),
        "M88": ("#1f77b4", 1.6, "-"),
        "M64": ("#d62728", 1.6, "-"),
        "M52": ("#2ca02c", 1.6, "-"),
    }
    for name, (o, h) in spectra.items():
        c, lw, ls = styles.get(name, ("C0", 1.2, "-"))
        ax.semilogy(o, h, color=c, lw=lw, ls=ls, label=name, alpha=0.9)
    ax.set_xlim(0, 16)
    ax.set_ylim(1e-14, 1e-1)
    ax.set_xlabel("harmonic order")
    ax.set_ylabel(r"$|J(\omega)|^2$ (a.u.)")
    ax.set_title("HHG intensity @k40 (lg_cov, T2=0.5 cyc)")
    ax.legend(frameon=False, loc="upper right")
    for n in range(1, 16):
        ax.axvline(n, color="0.85", lw=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_hhg_spectra_overlay.png", dpi=160)
    plt.close(fig)

    # Fig2: ratio bars H1-H9
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    qs = np.arange(1, 10)
    width = 0.22
    for i, model in enumerate(["nb104", "M52", "M64", "M88"]):
        if model not in peaks:
            continue
        ratios = [peaks[model][q - 1] / peaks["full112"][q - 1] for q in qs]
        ax.bar(qs + (i - 1.5) * width, ratios, width=width, label=model)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(qs)
    ax.set_xlabel("harmonic order")
    ax.set_ylabel(r"$I_q / I_q^{\mathrm{full112}}$")
    ax.set_title("Ordinary HHG ratio vs full112 (H1–H9)")
    ax.set_ylim(0, 1.6)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_hhg_ratio_H1_H9.png", dpi=160)
    plt.close(fig)

    # Fig3: M64 PBR vs full112
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for model, color in [("full112", "k"), ("M64", "#d62728"), ("M88", "#1f77b4"), ("M52", "#2ca02c")]:
        if model not in spectra:
            continue
        o, h = spectra[model]
        pbrs = [pbr(o, h, n)[2] for n in range(1, 16)]
        ax.semilogy(range(1, 16), pbrs, "o-", color=color, label=model, ms=4)
    ax.set_xlabel("harmonic order")
    ax.set_ylabel("peak / sideband median (PBR)")
    ax.set_title("Peak-to-background ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_pbr_H1_H15.png", dpi=160)
    plt.close(fig)

    # Fig4: Jo amp ratio M64/112
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    qs = [r["q"] for r in jo112]
    amp_r = [
        (b["amp"] / a["amp"] if a["amp"] > 0 else np.nan) for a, b in zip(jo112, jo64)
    ]
    y_r = [
        (b["yield"] / a["yield"] if a["yield"] > 0 else np.nan) for a, b in zip(jo112, jo64)
    ]
    axes[0].semilogy(qs, [a["amp"] for a in jo112], "k-o", label="full112 Jo", ms=4)
    axes[0].semilogy(qs, [b["amp"] for b in jo64], "r-s", label="M64 Jo", ms=4)
    axes[0].set_xlabel("q")
    axes[0].set_ylabel(r"$|J_o|$ amplitude")
    axes[0].set_title(r"$J_o$ amplitude (N-odd channel)")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].plot(qs, amp_r, "o-", color="#d62728", label="amp ratio")
    axes[1].plot(qs, y_r, "s--", color="#ff7f0e", label="yield ratio")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("M64 / full112")
    axes[1].set_title(r"$J_o$ M64 vs full112")
    axes[1].set_ylim(0, 5)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_Jo_M64_vs_full112.png", dpi=160)
    plt.close(fig)

    # Fig5: leakage M64 odd/even
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    leak = []
    for je, jo in zip(je64, jo64):
        q = je["q"]
        if q % 2 == 0:
            leak.append(je["yield"] / jo["yield"] if jo["yield"] > 0 else np.nan)
        else:
            leak.append(jo["yield"] / je["yield"] if je["yield"] > 0 else np.nan)
    colors = ["#1f77b4" if q % 2 else "#d62728" for q in qs]
    ax.bar(qs, leak, color=colors)
    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("forbidden / allowed")
    ax.set_title("M64 selection-rule leakage (blue=odd/Je, red=even/Jo)")
    fig.tight_layout()
    fig.savefig(figdir / "fig5_M64_leakage.png", dpi=160)
    plt.close(fig)

    # Fig6: walltime bar
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    labels = ["M52 +N", "M64 +N", "M64 −N", "M88 +N"]
    hours = [1.242, 2.176, 2.132, 4.908]
    ax.bar(labels, hours, color=["#2ca02c", "#d62728", "#ff9896", "#1f77b4"])
    ax.set_ylabel("wall time (h)  part_2 8c")
    ax.set_title("S1 cost @k40")
    for i, h in enumerate(hours):
        ax.text(i, h + 0.08, f"{h:.2f}h", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig6_walltime.png", dpi=160)
    plt.close(fig)

    print(f"Wrote figures to {figdir}")
    print(f"Wrote tables to {outdir}")


if __name__ == "__main__":
    main()

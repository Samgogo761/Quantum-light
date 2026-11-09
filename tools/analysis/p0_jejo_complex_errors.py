#!/usr/bin/env python3
"""P0: Je/Jo complex tables + M88/M52 vs full112 amp/phase error figures."""
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


def wrap_phase(d):
    while d > math.pi:
        d -= TWOPI
    while d < -math.pi:
        d += TWOPI
    return d


def channel_tables(plus, minus, omega0, orders):
    even, odd = [], []
    for p, m in zip(plus, minus):
        even.append((p[0], 0.5 * (p[1] + m[1]), 0.5 * (p[2] + m[2])))
        odd.append((p[0], 0.5 * (p[1] - m[1]), 0.5 * (p[2] - m[2])))
    je, jo = [], []
    for q in orders:
        for rows, out in ((even, je), (odd, jo)):
            jx, jy = windowed_amp(rows, q, omega0)
            amp = math.hypot(abs(jx), abs(jy))
            ph = cmath.phase(jx) if abs(jx) > 0 else float("nan")
            out.append(
                {
                    "q": q,
                    "jx_re": jx.real,
                    "jx_im": jx.imag,
                    "jy_re": jy.real,
                    "jy_im": jy.imag,
                    "amp": amp,
                    "phase": ph,
                    "yield": abs(jx) ** 2 + abs(jy) ** 2,
                }
            )
    return je, jo


def write_channel_csv(path: Path, rows, tag: str):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "q", "Jx_re", "Jx_im", "Jy_re", "Jy_im", "amp", "phase_rad", "yield"])
        for r in rows:
            w.writerow(
                [
                    tag,
                    r["q"],
                    f"{r['jx_re']:.12e}",
                    f"{r['jx_im']:.12e}",
                    f"{r['jy_re']:.12e}",
                    f"{r['jy_im']:.12e}",
                    f"{r['amp']:.12e}",
                    f"{r['phase']:.12e}",
                    f"{r['yield']:.12e}",
                ]
            )


def main():
    docs = Path(r"C:\Users\26507\Documents")
    root = next(p for p in docs.iterdir() if (p / "_tmp_magnetic_hhg").exists()) / "_tmp_magnetic_hhg"
    outdir = Path(r"C:\Users\26507\Documents\New_SBEs\Quantum-light\results\magnetic_hhg_P0_JeJo_complex_20260719")
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    orders = list(range(1, 16))
    omega0 = infer_omega0(root / "ref_full112_plusN" / "HHG.dat")
    (outdir / "omega0.txt").write_text(f"{omega0:.16e}\n", encoding="utf-8")

    pairs = {
        "full112": (root / "ref_full112_plusN", root / "ref_full112_minusN_Pconj"),
        "M52": (root / "M52_plusN_k40_nv24_b52", root / "M52_minusN_k40_nv24_b52"),
        "M64": (root / "M64_plusN_k40_nv36_b64", root / "M64_minusN_k40_nv36_b64"),
        "M88": (root / "M88_plusN_k40_nv58_b88", root / "M88_minusN_k40_nv58_b88"),
    }
    # nb104: only +N usually — skip Je/Jo if no -N
    nb104_plus = root / "ref_nb104_plusN"
    if not (nb104_plus / "Jt.dat").exists():
        nb104_plus = root / "ref_nb104"

    data = {}
    for name, (pd, md) in pairs.items():
        plus, minus = read_jt(pd / "Jt.dat"), read_jt(md / "Jt.dat")
        je, jo = channel_tables(plus, minus, omega0, orders)
        data[name] = {"Je": je, "Jo": jo}
        write_channel_csv(outdir / f"{name}_Je_q.csv", je, "Je")
        write_channel_csv(outdir / f"{name}_Jo_q.csv", jo, "Jo")

    # combined error table M52/M88 vs 112
    with (outdir / "M52_M88_vs_full112_amp_phase_errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "channel",
                "q",
                "amp_112",
                "amp_M52",
                "amp_M88",
                "amp_err_M52",
                "amp_err_M88",
                "amp_ratio_M52",
                "amp_ratio_M88",
                "phase_112",
                "dphase_M52",
                "dphase_M88",
            ]
        )
        for ch in ("Je", "Jo"):
            for i, q in enumerate(orders):
                a0 = data["full112"][ch][i]
                for _ in (0,):
                    pass
                row_base = [ch, q, f"{a0['amp']:.12e}"]
                amps = []
                dphs = []
                ratios = []
                for m in ("M52", "M88"):
                    a = data[m][ch][i]
                    amps.append(a["amp"])
                    ratios.append(a["amp"] / a0["amp"] if a0["amp"] > 0 else float("nan"))
                    dphs.append(wrap_phase(a["phase"] - a0["phase"]))
                w.writerow(
                    [
                        ch,
                        q,
                        f"{a0['amp']:.12e}",
                        f"{amps[0]:.12e}",
                        f"{amps[1]:.12e}",
                        f"{amps[0]-a0['amp']:.12e}",
                        f"{amps[1]-a0['amp']:.12e}",
                        f"{ratios[0]:.12e}",
                        f"{ratios[1]:.12e}",
                        f"{a0['phase']:.12e}",
                        f"{dphs[0]:.12e}",
                        f"{dphs[1]:.12e}",
                    ]
                )

    # Figures: amp ratio + phase error for Je (odd) and Jo (even)
    for ch, qs, title in (
        ("Je", [1, 3, 5, 7, 9, 11, 13, 15], r"$J_e$ (odd / N-even)"),
        ("Jo", [2, 4, 6, 8, 10, 12, 14], r"$J_o$ (even / N-odd)"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        x = np.arange(len(qs))
        width = 0.35
        for i, m in enumerate(("M52", "M88")):
            ratios, dph = [], []
            for q in qs:
                a0 = data["full112"][ch][q - 1]
                a = data[m][ch][q - 1]
                ratios.append(a["amp"] / a0["amp"] if a0["amp"] > 0 else np.nan)
                dph.append(wrap_phase(a["phase"] - a0["phase"]))
            axes[0].bar(x + (i - 0.5) * width, ratios, width=width, label=m)
            axes[1].bar(x + (i - 0.5) * width, dph, width=width, label=m)
        axes[0].axhline(1.0, color="k", lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"H{q}" for q in qs])
        axes[0].set_ylabel(r"$|J|/|J^{112}|$")
        axes[0].set_title(f"{title}: amplitude ratio")
        axes[0].legend(frameon=False)
        axes[0].set_ylim(0, 2.0)
        axes[1].axhline(0.0, color="k", lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"H{q}" for q in qs])
        axes[1].set_ylabel(r"$\Delta\mathrm{phase}$ (rad)")
        axes[1].set_title(f"{title}: phase error vs full112")
        axes[1].legend(frameon=False)
        axes[1].set_ylim(-math.pi, math.pi)
        fig.tight_layout()
        fig.savefig(figdir / f"figP0_{ch}_amp_phase_error_M52_M88.png", dpi=160)
        plt.close(fig)

    # overview: all models Jo amp
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for name, color, mk in [
        ("full112", "k", "o"),
        ("M88", "#1f77b4", "s"),
        ("M52", "#2ca02c", "^"),
        ("M64", "#d62728", "D"),
    ]:
        ax.semilogy(
            orders,
            [data[name]["Jo"][i]["amp"] for i in range(15)],
            "-",
            color=color,
            marker=mk,
            ms=4,
            label=name,
        )
    ax.set_xlabel("q")
    ax.set_ylabel(r"$|J_o|$")
    ax.set_title(r"$J_o$ amplitude (unified Hann FFT)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "figP0_Jo_amp_all_models.png", dpi=160)
    plt.close(fig)

    print("Wrote", outdir)


if __name__ == "__main__":
    main()

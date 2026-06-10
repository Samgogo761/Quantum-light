"""Plot / diagnose an HHG.dat spectrum and explain its peak width.

Why HHG peaks look broad and "continuous" (the common complaint):
  * Frequency-grid step in harmonic-order units = 1 / ncyc  (FFT bin = 1/T_total).
    A 4-cycle pulse => 0.25-order grid. More cycles => finer grid, sharper peaks.
  * Homogeneous (dephasing) width: for an exp(-t/T2) coherence the Lorentzian
    FWHM in harmonic-order units is ~ 2 / (T2 * omega0). Short T2 => broad peaks.
  * The Hann window adds ~1.5 bins of main-lobe broadening.

This tool infers the grid step (=> ncyc), reports the T2-limited FWHM if you give
--t2-cycles, extracts the integer-harmonic peak heights, and (if matplotlib is
present) draws a log-scale spectrum with integer-harmonic gridlines. Without
matplotlib it writes a clean 2-column file you can plot anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_hhg(path: Path):
    data = np.loadtxt(path, comments="#")
    order = data[:, 0]
    omega = data[:, 1]
    # columns: order omega HHG_x HHG_y HHG_total
    hx, hy, htot = data[:, 2], data[:, 3], data[:, 4]
    return order, omega, hx, hy, htot


def peak_heights(order, y, nmax, half=0.3):
    out = []
    for n in range(1, nmax + 1):
        sel = np.abs(order - n) <= half
        out.append((n, float(np.max(y[sel])) if np.any(sel) else 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--t2-cycles", type=float, default=None,
                    help="T2 in optical cycles, to report the dephasing-limited FWHM")
    ap.add_argument("--nmax", type=int, default=15)
    ap.add_argument("--component", choices=["total", "x", "y"], default="total")
    ap.add_argument("--png", type=Path, default=None, help="Save a log-scale plot here")
    ap.add_argument("--clean-out", type=Path, default=None,
                    help="Write a 2-column (order, value) file for external plotting")
    args = ap.parse_args()

    order, omega, hx, hy, htot = load_hhg(args.file)
    y = {"total": htot, "x": hx, "y": hy}[args.component]

    # --- grid step => effective ncyc ---
    dstep = float(np.median(np.diff(order[order <= 5])))
    ncyc_eff = 1.0 / dstep if dstep > 0 else float("nan")

    print("=" * 66)
    print(f"HHG spectrum: {args.file.name}   (component={args.component})")
    print("=" * 66)
    print(f"  harmonic-order grid step      = {dstep:.4f}  => ncyc ~ {ncyc_eff:.1f}")
    print(f"  => spectral resolution is {dstep:.3f} order; use more cycles to refine.")
    if args.t2_cycles is not None and args.t2_cycles > 0:
        # FWHM_order = 2/(T2*omega0); T2 = t2_cycles * T_cycle = t2_cycles * 2pi/omega0
        # => T2*omega0 = t2_cycles*2pi ; FWHM_order = 2/(t2_cycles*2pi) = 1/(pi*t2_cycles)
        fwhm = 1.0 / (np.pi * args.t2_cycles)
        print(f"  dephasing-limited peak FWHM   = {fwhm:.3f} order  (T2={args.t2_cycles} cycle)")
        if fwhm > 0.5:
            print("  !! FWHM > 0.5 order: peaks overlap into a quasi-continuum.")
            print("     For sharp, separated peaks use T2_cycles >= ~1-2 (or no dephasing)")
            print("     AND ncyc >= ~8.  Short T2 broadens peaks; it does NOT sharpen them.")

    print("\n  Integer-harmonic peak heights:")
    peaks = peak_heights(order, y, args.nmax)
    ref = max((v for _, v in peaks), default=1.0) or 1.0
    print(f"  {'H':>3} {'value':>14} {'rel(dB)':>10} {'parity':>7}")
    for n, v in peaks:
        db = 10.0 * np.log10(v / ref) if v > 0 else float("-inf")
        print(f"  {n:>3} {v:>14.4e} {db:>10.1f} {'odd' if n % 2 else 'even':>7}")

    if args.clean_out is not None:
        np.savetxt(args.clean_out, np.column_stack([order, y]),
                   header="harmonic_order  HHG_value", fmt="%.6e")
        print(f"\n  [written] clean 2-column spectrum -> {args.clean_out}")

    if args.png is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n  matplotlib not available; skipped --png (use --clean-out instead).")
            return
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.semilogy(order, np.maximum(y, 1e-300), lw=0.9)
        for n in range(1, args.nmax + 1):
            ax.axvline(n, color="0.85", lw=0.6, zorder=0)
        ax.set_xlim(0, args.nmax)
        pos = y[y > 0]
        if pos.size:
            ax.set_ylim(pos.max() * 1e-8, pos.max() * 3)
        ax.set_xlabel("Harmonic order")
        ax.set_ylabel(f"HHG intensity ({args.component})")
        ax.set_title(f"{args.file.parent.name}  (ncyc~{ncyc_eff:.0f})")
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print(f"\n  [written] log-scale plot -> {args.png}")


if __name__ == "__main__":
    main()

"""Harmonic-resolved intraband/interband decomposition of the HHG current.

Zero-cost analysis on existing `Jt_decomposed.dat` (columns:
it, time_fs, Jx_intra, Jy_intra, Jx_inter, Jy_inter, Jx_tot, Jy_tot).

Goal
----
The CrI3-AFM paper argues that even harmonics are carried by the interband
(polarization) channel rather than the intraband (band-velocity) channel. This
script Fourier-analyses each channel with the SAME Hann window + FFT convention
as the solver's HHG.dat (mod_hhg.f90) and reports, per harmonic order, the
intraband vs interband magnitudes and which channel dominates.

Caveat (honesty)
----------------
The solver outputs only intra + inter. The Yue-Gaarde anomalous current j_anom
is a *sub-part* of j_inter and is NOT separately written, so a literal
"|j_anom| << |j_pol|" check is not possible from this file alone -- that needs
a small solver addition (isolate the Berry-curvature term). What is rigorously
checkable here: (1) decomposition closure J_tot = J_intra + J_inter, and
(2) which channel (intra vs inter) carries each harmonic, in both the parallel
(Jx, drive direction) and transverse (Jy) components.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def infer_omega0_au(hhg_path: Path) -> float:
    """omega0 in a.u. from HHG.dat (omega / harmonic_order over clean rows)."""
    vals = []
    for line in hhg_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        order, omega = float(p[0]), float(p[1])
        if order > 0.5:
            vals.append(omega / order)
    if not vals:
        raise ValueError(f"Could not infer omega0 from {hhg_path}")
    return float(np.median(vals))


def load_decomposed(path: Path):
    data = np.loadtxt(path, comments="#")
    t_fs = data[:, 1]
    chans = {
        "x_intra": data[:, 2], "y_intra": data[:, 3],
        "x_inter": data[:, 4], "y_inter": data[:, 5],
        "x_tot":   data[:, 6], "y_tot":   data[:, 7],
    }
    return t_fs, chans


def hann_fft(signal: np.ndarray):
    nt = signal.size
    it = np.arange(nt)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * it / (nt - 1)))
    return np.fft.rfft(signal * w)


def harmonic_peak(mag: np.ndarray, omega_grid: np.ndarray, n: float,
                  omega0: float, half_width: float = 0.4) -> float:
    """Max |.| within +/- half_width harmonics around order n."""
    lo, hi = (n - half_width) * omega0, (n + half_width) * omega0
    sel = (omega_grid >= lo) & (omega_grid <= hi)
    return float(mag[sel].max()) if np.any(sel) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path,
                    help="Output dir containing Jt_decomposed.dat and HHG.dat")
    ap.add_argument("--dt-au", type=float, default=0.35)
    ap.add_argument("--orders", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    jt_path = args.dir / "Jt_decomposed.dat"
    hhg_path = args.dir / "HHG.dat"
    omega0 = infer_omega0_au(hhg_path)
    t_fs, chans = load_decomposed(jt_path)
    nt = t_fs.size

    # FFT frequency grid in a.u. (dt given in a.u.)
    omega_grid = np.fft.rfftfreq(nt, d=args.dt_au) * 2.0 * np.pi

    spec = {k: np.abs(hann_fft(v)) for k, v in chans.items()}

    # Closure check: J_tot vs J_intra + J_inter (time domain, relative L2)
    closure = {}
    for comp in ("x", "y"):
        tot = chans[f"{comp}_tot"]
        recon = chans[f"{comp}_intra"] + chans[f"{comp}_inter"]
        num = np.linalg.norm(tot - recon)
        den = max(np.linalg.norm(tot), 1e-300)
        closure[comp] = num / den

    print("=" * 78)
    print(f"Harmonic-resolved intra/inter decomposition : {args.dir.name}")
    print(f"omega0 = {omega0:.6e} a.u.  (lambda~3200 nm => ~0.387 eV),  nt={nt}")
    print("=" * 78)
    print(f"[closure]  ||J_tot-(J_intra+J_inter)||/||J_tot||:  "
          f"x={closure['x']:.2e}   y={closure['y']:.2e}   (should be ~1e-12)")
    print()

    header = (f"{'n':>3} {'parity':>6} | "
              f"{'Jx_intra':>11} {'Jx_inter':>11} {'inter/tot_x':>11} | "
              f"{'Jy_intra':>11} {'Jy_inter':>11} {'inter/tot_y':>11}")
    print(header)
    print("-" * len(header))

    rows = []
    for n in range(1, args.orders + 1):
        parity = "odd" if n % 2 == 1 else "EVEN"
        xi = harmonic_peak(spec["x_intra"], omega_grid, n, omega0)
        xe = harmonic_peak(spec["x_inter"], omega_grid, n, omega0)
        yi = harmonic_peak(spec["y_intra"], omega_grid, n, omega0)
        ye = harmonic_peak(spec["y_inter"], omega_grid, n, omega0)
        rx = xe / max(xi + xe, 1e-300)
        ry = ye / max(yi + ye, 1e-300)
        rows.append((n, parity, xi, xe, rx, yi, ye, ry))
        print(f"{n:>3} {parity:>6} | {xi:11.3e} {xe:11.3e} {rx:11.3f} | "
              f"{yi:11.3e} {ye:11.3e} {ry:11.3f}")

    print()
    print("inter/tot ~ 1  => harmonic carried by INTERBAND (polarization) channel")
    print("inter/tot ~ 0  => harmonic carried by INTRABAND (band-velocity) channel")
    print("Note: j_anom is a sub-part of j_inter, not separable from this file.")

    if args.out:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            f.write("order,parity,Jx_intra,Jx_inter,inter_frac_x,"
                    "Jy_intra,Jy_inter,inter_frac_y\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.6e},{r[3]:.6e},{r[4]:.6f},"
                        f"{r[5]:.6e},{r[6]:.6e},{r[7]:.6f}\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()

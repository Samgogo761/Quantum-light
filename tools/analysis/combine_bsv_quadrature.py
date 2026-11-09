"""Combine deterministic intensity-quadrature runs into the BSV-averaged HHG.

    HHG_bsv(w) = int_0^inf e^{-x} S(w; I0*x) dx  ~=  sum_i  w_i S(w; I_i)

Inputs: node_XX/HHG.dat produced by deploy/run_bsv_quadrature.sh and the node
manifest deploy/bsv_quad_nodes.dat (idx, x, weight, I).

Outputs / checks:
  1. HHG_bsv_quad.dat -- the noise-free BSV average (x, y, total columns).
  2. Truncation sensitivity: relative change when dropping the top 1 / 2 nodes
     (bounds the contribution of the strongest-intensity tail).
  3. Cross-validation vs the 51-sample MC pilot (--mc HHG_bsv_pooled.dat):
     quadrature should sit within ~2 stderr of MC where MC is reliable
     (low harmonics). This simultaneously validates the fixed-CEP assumption
     (MC also averaged the random phase; the power spectrum's phi-dependence
     is weak for a 4-cycle pulse -- discrepancy here would expose it).
  4. Enhancement table vs a classical run at I = I0 (--classical HHG.dat).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_nodes(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return rows


def peak(o, y, n, h=0.35):
    s = np.abs(o - n) <= h
    return float(np.max(y[s])) if s.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quad-root", required=True, type=Path)
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--classical", type=Path, default=None)
    ap.add_argument("--mc", type=Path, default=None,
                    help="pooled MC spectrum (order omega avg stderr) for cross-check")
    ap.add_argument("--nmax", type=int, default=13)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    nodes = load_nodes(args.nodes)
    spectra, weights, missing = [], [], []
    for idx, x, w, I in nodes:
        f = args.quad_root / f"node_{idx:02d}" / "HHG.dat"
        if not f.exists():
            missing.append(idx)
            continue
        d = np.loadtxt(f)
        spectra.append(d)
        weights.append(w)
    if missing:
        print(f"WARNING: missing nodes {missing} (their weight "
              f"{sum(n[2] for n in nodes if n[0] in missing):.3e} treated as 0)")
    if not spectra:
        raise SystemExit("no node spectra found")

    order, omega = spectra[0][:, 0], spectra[0][:, 1]
    W = np.array(weights)
    Sx = np.array([s[:, 2] for s in spectra])
    Sy = np.array([s[:, 3] for s in spectra])
    St = np.array([s[:, 4] for s in spectra])

    bsv_x = W @ Sx
    bsv_y = W @ Sy
    bsv_t = W @ St

    # truncation sensitivity: drop top-1 / top-2 intensity nodes
    sort_by_I = np.argsort([n[1] for n in nodes if n[0] not in missing])
    keep1 = sort_by_I[:-1]
    keep2 = sort_by_I[:-2]
    bsv_t1 = W[keep1] @ St[keep1]
    bsv_t2 = W[keep2] @ St[keep2]

    print(f"BSV quadrature combination: {len(weights)} nodes, sum(w) = {W.sum():.8f}")
    cl = None
    if args.classical is not None:
        c = np.loadtxt(args.classical)
        cl = (c[:, 0], c[:, 4])
    mc = None
    if args.mc is not None:
        m = np.loadtxt(args.mc)
        mc = (m[:, 0], m[:, 2], m[:, 3])

    hdr = f"{'H':>3} {'BSV_quad':>12} {'drop1':>8} {'drop2':>8}"
    if cl is not None:
        hdr += f" {'BSV/class':>10}"
    if mc is not None:
        hdr += f" {'MC_avg':>12} {'(quad-MC)/se':>12}"
    hdr += f" {'parity':>6}"
    print(hdr)
    for n in range(1, args.nmax + 1):
        b = peak(order, bsv_t, n)
        d1 = abs(peak(order, bsv_t1, n) / b - 1) if b > 0 else 0
        d2 = abs(peak(order, bsv_t2, n) / b - 1) if b > 0 else 0
        line = f"{n:>3} {b:>12.4e} {d1:>8.1%} {d2:>8.1%}"
        if cl is not None:
            cv = peak(cl[0], cl[1], n)
            line += f" {(b/cv if cv > 0 else 0):>10.2f}"
        if mc is not None:
            mv = peak(mc[0], mc[1], n)
            se = peak(mc[0], mc[2], n)
            z = (b - mv) / se if se > 0 else 0.0
            line += f" {mv:>12.4e} {z:>12.2f}"
        line += f" {'odd' if n % 2 else 'EVEN':>6}"
        print(line)

    print("\n  drop1/drop2 = relative change without the top 1/2 intensity nodes")
    print("  (small => the [0,18]*I0 window captures the integral; large => extend x_max)")
    if mc is not None:
        print("  |(quad-MC)/se| <~ 2 at low harmonics validates quadrature AND the")
        print("  fixed-CEP assumption against the phase-averaged MC pilot.")

    out = args.out or (args.quad_root / "HHG_bsv_quad.dat")
    np.savetxt(out, np.column_stack([order, omega, bsv_x, bsv_y, bsv_t]),
               header="deterministic BSV intensity quadrature\n"
                      "harmonic_order omega(a.u.) HHG_x HHG_y HHG_total",
               fmt="%.8e")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()

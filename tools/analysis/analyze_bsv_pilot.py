"""Pool BSV array-task HHG_bsv.dat files into one 51-sample spectrum and (if a
classical baseline is given) report the BSV/classical enhancement per harmonic.

Each task's HHG_bsv.dat has columns: order omega HHG_bsv_avg HHG_bsv_stderr,
with stderr = sqrt((<x^2>-<x>^2)/N_task). We pool tasks of equal N_task into a
single mean and a correct pooled stderr over all samples (not the coarse
across-task stderr that combine_bsv.sh computes).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bsv-dir", required=True, type=Path, help="output_bsv/ with task_*/")
    ap.add_argument("--n-per-task", type=int, default=17)
    ap.add_argument("--classical", type=Path, default=None,
                    help="classical HHG.dat at I=<I> for the enhancement ratio")
    ap.add_argument("--nmax", type=int, default=13)
    ap.add_argument("--out", type=Path, default=None, help="write pooled spectrum here")
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.bsv_dir / "task_*" / "HHG_bsv.dat")))
    if not files:
        raise SystemExit(f"no task_*/HHG_bsv.dat under {args.bsv_dir}")
    data = [np.loadtxt(f) for f in files]
    order, omega = data[0][:, 0], data[0][:, 1]
    means = np.array([d[:, 2] for d in data])
    sterr = np.array([d[:, 3] for d in data])
    N = args.n_per_task
    ntot = N * len(files)

    mean_pool = means.mean(axis=0)
    Ex2 = ((sterr ** 2 * N) + means ** 2).mean(axis=0)   # pooled <x^2>
    var_pool = np.maximum(Ex2 - mean_pool ** 2, 0.0)
    stderr_pool = np.sqrt(var_pool / ntot)

    def peak(o, y, n, h=0.35):
        s = np.abs(o - n) <= h
        return float(np.max(y[s])) if s.any() else 0.0

    cl = None
    if args.classical is not None:
        c = np.loadtxt(args.classical)
        cl = (c[:, 0], c[:, 4])

    print(f"BSV pooled spectrum: {len(files)} tasks x {N} = {ntot} samples")
    hdr = f"{'H':>3} {'BSV_avg':>12} {'rel_stderr':>11}"
    if cl is not None:
        hdr += f" {'classical':>12} {'BSV/class':>10}"
    hdr += f" {'parity':>6}"
    print(hdr)
    for n in range(1, args.nmax + 1):
        b = peak(order, mean_pool, n)
        be = peak(order, stderr_pool, n)
        line = f"{n:>3} {b:>12.4e} {(be/b if b>0 else 0):>11.1%}"
        if cl is not None:
            cval = peak(cl[0], cl[1], n)
            line += f" {cval:>12.4e} {(b/cval if cval>0 else 0):>10.2f}"
        line += f" {'odd' if n % 2 else 'EVEN':>6}"
        print(line)

    if args.out is not None:
        np.savetxt(args.out, np.column_stack([order, omega, mean_pool, stderr_pool]),
                   header=f"pooled BSV {ntot} samples\norder omega HHG_bsv_avg HHG_bsv_stderr",
                   fmt="%.6e")
        print(f"\n[written] {args.out}")

    # crude N-for-target-precision estimate using the worst harmonic 1..nmax
    rels = []
    for n in range(1, args.nmax + 1):
        b = peak(order, mean_pool, n)
        be = peak(order, stderr_pool, n)
        if b > 0:
            rels.append(be / b)
    if rels:
        worst = max(rels)
        for target in (0.10, 0.05):
            need = int(np.ceil(ntot * (worst / target) ** 2))
            print(f"  to reach {target:.0%} rel-stderr on the worst harmonic (now {worst:.0%}): "
                  f"~{need} samples")


if __name__ == "__main__":
    main()

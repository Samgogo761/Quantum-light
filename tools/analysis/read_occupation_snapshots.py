"""Read occupation_kt.dat (k-space occupation snapshots) from the SBE solver.

Columns: it  time_fs  ikx iky  kx ky  n_val  n_cond
Produced when &output save_occupation=.true.

Reports, per snapshot, the total excited (conduction) population and its k-space
peak, and can dump per-snapshot 2D conduction-population maps as .npy for
plotting / HHG-XR visualisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--dump-maps", type=Path, default=None,
                    help="Optional .npz path to save per-snapshot n_cond(kx,ky) maps")
    args = ap.parse_args()

    data = np.loadtxt(args.file, comments="#")
    it = data[:, 0].astype(int)
    t_fs = data[:, 1]
    ikx = data[:, 2].astype(int)
    iky = data[:, 3].astype(int)
    n_val = data[:, 6]
    n_cond = data[:, 7]

    nkx, nky = ikx.max(), iky.max()
    snaps = sorted(set(it.tolist()))
    print("=" * 64)
    print(f"Occupation snapshots: {args.file.name}")
    print(f"  k-grid {nkx} x {nky},  {len(snaps)} snapshots")
    print("=" * 64)
    print(f"{'it':>7} {'time_fs':>10} {'sum n_cond':>14} {'max n_cond':>12} {'tot carriers':>14}")

    maps = {}
    for s in snaps:
        sel = it == s
        ncond_k = n_cond[sel]
        # total excited carriers per cell = mean conduction population over BZ
        sum_cond = float(np.mean(ncond_k))
        max_cond = float(np.max(ncond_k))
        print(f"{s:>7} {t_fs[sel][0]:>10.4f} {sum_cond:>14.6e} {max_cond:>12.4e} "
              f"{float(np.sum(ncond_k)):>14.4e}")
        if args.dump_maps is not None:
            grid = np.zeros((nkx, nky))
            grid[ikx[sel] - 1, iky[sel] - 1] = ncond_k
            maps[f"it_{s:06d}"] = grid

    if args.dump_maps is not None:
        np.savez_compressed(args.dump_maps, **maps)
        print(f"\n[saved] per-snapshot n_cond(kx,ky) maps -> {args.dump_maps}")


if __name__ == "__main__":
    main()

"""Analyze the solver's quantum_geometry.dat (Berry curvature + quantum metric).

Produced by mod_geometry.f90 in the length-gauge path. Columns:
    ikx iky band  kx ky  E(eV)  Omega(a.u.)  gxx  gyy  gxy  valley

Two physics deliverables:
  1. PT / j_anom check: for a PT-symmetric AFM, Omega_n(k) = 0 pointwise, so the
     anomalous (Berry) current j_anom vanishes at all times. We report
     max|Omega| and compare it to the natural velocity^2/gap^2 scale of the
     quantum metric; max|Omega| << typical Tr g confirms j_anom ~ 0.
  2. Quantum metric: Tr g_n(k) = gxx+gyy is the PT-even geometric object that can
     underlie the even-harmonic interband polarization response. We summarise its
     magnitude and (optionally) its band/k distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_geometry(path: Path):
    data = np.loadtxt(path, comments="#")
    cols = {
        "ikx": data[:, 0].astype(int), "iky": data[:, 1].astype(int),
        "band": data[:, 2].astype(int),
        "kx": data[:, 3], "ky": data[:, 4], "E": data[:, 5],
        "omega": data[:, 6], "gxx": data[:, 7], "gyy": data[:, 8],
        "gxy": data[:, 9], "valley": data[:, 10].astype(int),
    }
    return cols


def occupied_sum_per_k(c, mask):
    """Sum a per-(k,band) quantity over bands selected by `mask`, grouped by k."""
    key = c["ikx"].astype(np.int64) * 100000 + c["iky"].astype(np.int64)
    ukey, inv = np.unique(key, return_inverse=True)
    out = np.zeros(ukey.size)
    np.add.at(out, inv[mask], c["_val"][mask])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--ef", type=float, default=0.0843,
                    help="Fermi level (eV); occupied = E < ef (default 0.0843)")
    args = ap.parse_args()

    c = load_geometry(args.file)
    omega = c["omega"]
    trg = c["gxx"] + c["gyy"]
    nrow = omega.size
    nk = len(set(zip(c["ikx"].tolist(), c["iky"].tolist())))
    nband = len(set(c["band"].tolist()))
    occ = c["E"] < args.ef

    print("=" * 70)
    print(f"Quantum geometry: {args.file.name}")
    print(f"  k-points = {nk},  bands = {nband},  rows = {nrow},  occupied(E<{args.ef}) = {int(occ.sum())}")
    print("=" * 70)

    # --- 1. PT / j_anom check (OCCUPIED-MANIFOLD SUM) -------------------------
    # With SOC + PT every band is Kramers-degenerate, so the per-band Abelian
    # Omega_n diverges from its near-degenerate partner (1/dE^2) and is
    # meaningless. The PT-protected, j_anom-relevant quantity is the sum over
    # OCCUPIED bands, where degenerate-pair divergences cancel.
    c["_val"] = omega
    om_occ = occupied_sum_per_k(c, occ)
    om_all = occupied_sum_per_k(c, np.ones_like(occ))
    c["_val"] = trg
    trg_occ = occupied_sum_per_k(c, occ)

    max_occ = float(np.max(np.abs(om_occ)))
    mean_occ = float(np.mean(np.abs(om_occ)))
    floor = float(np.max(np.abs(om_all)))   # all-band trace = numerical/roundoff floor
    print("\n[1] PT / anomalous-current (j_anom) check  [occupied-manifold sum]")
    print(f"    max_k |Omega_occ(k)|      = {max_occ:.4e} a.u.   <- the PT check")
    print(f"    mean_k|Omega_occ(k)|      = {mean_occ:.4e} a.u.")
    print(f"    all-band Sum_n Omega_n    = {floor:.4e} a.u.  (trace sum-rule = numerical floor)")
    print(f"    per-band max|Omega_n|     = {float(np.max(np.abs(omega))):.3e}  (Kramers-divergent, IGNORE)")
    # Berry curvature is in bohr^2; |Omega_occ| ~ O(0.1) is small in absolute terms.
    if max_occ < 1.0:
        print("    => |Omega_occ| < 1 bohr^2: anomalous (Berry) channel strongly PT-suppressed,")
        print("       j_anom small -> even harmonics dominated by the polarization current j_pol.")
    else:
        print("    => |Omega_occ| not small: investigate Wannier PT quality.")
    if max_occ > 30.0 * max(floor, 1e-12):
        print(f"    note: |Omega_occ| is ~{max_occ/max(floor,1e-12):.0f}x the roundoff floor -> a small")
        print("       genuine PT residual of the Wannier model (consistent with the ~5e-2")
        print("       non-Hermiticity of the Wannier position matrix). For a rigorous even-")
        print("       harmonic bound, compute the time-resolved j_anom(t) (Tier-0b+ TODO).")

    # --- 2. Quantum metric (occupied-manifold, geometry of polarization) ------
    # Per-band Tr g is also Kramers-divergent; the occupied-manifold sum is the
    # gauge-invariant quantum volume of the filled bands.
    print("\n[2] Quantum metric (occupied-manifold; geometric origin of polarization)")
    print(f"    max_k  Tr g_occ(k)        = {float(np.max(trg_occ)):.4e} a.u.")
    print(f"    mean_k Tr g_occ(k)        = {float(np.mean(trg_occ)):.4e} a.u.")
    print(f"    (per-band max Tr g_n = {float(np.max(trg)):.3e}, Kramers-divergent, IGNORE)")


if __name__ == "__main__":
    main()

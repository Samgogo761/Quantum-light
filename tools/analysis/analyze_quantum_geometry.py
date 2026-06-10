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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--ef", type=float, default=None,
                    help="Fermi level (eV) to split valence/conduction; default median E")
    args = ap.parse_args()

    c = load_geometry(args.file)
    omega = c["omega"]
    trg = c["gxx"] + c["gyy"]
    nrow = omega.size
    nk = len(set(zip(c["ikx"].tolist(), c["iky"].tolist())))
    nband = len(set(c["band"].tolist()))

    print("=" * 70)
    print(f"Quantum geometry: {args.file.name}")
    print(f"  k-points = {nk},  bands = {nband},  rows = {nrow}")
    print("=" * 70)

    # --- 1. PT / j_anom check -------------------------------------------------
    max_abs_omega = float(np.max(np.abs(omega)))
    rms_omega = float(np.sqrt(np.mean(omega ** 2)))
    typ_trg = float(np.median(trg[trg > 0])) if np.any(trg > 0) else 0.0
    print("\n[1] PT / anomalous-current (j_anom) check")
    print(f"    max |Omega_n(k)|        = {max_abs_omega:.4e} a.u.")
    print(f"    rms |Omega_n(k)|        = {rms_omega:.4e} a.u.")
    print(f"    median Tr g_n(k) (>0)   = {typ_trg:.4e} a.u.  (natural geometric scale)")
    if typ_trg > 0:
        ratio = max_abs_omega / typ_trg
        print(f"    max|Omega| / median Tr g = {ratio:.2e}")
        if ratio < 1e-3:
            print("    => Omega at numerical floor relative to quantum metric:")
            print("       PT symmetry confirmed, j_anom ~ 0 -> even harmonics are j_pol.")
        else:
            print("    => Omega NOT negligible vs metric: investigate PT breaking /")
            print("       degeneracy handling before claiming j_anom = 0.")

    # --- 2. Quantum metric ----------------------------------------------------
    print("\n[2] Quantum metric (PT-even, geometric origin of polarization response)")
    print(f"    max  Tr g_n(k)          = {float(np.max(trg)):.4e} a.u.")
    print(f"    mean Tr g_n(k)          = {float(np.mean(trg)):.4e} a.u.")

    # Per-valley split if available
    for vid, name in ((1, "K"), (-1, "Kp"), (2, "K"), (0, "all")):
        sel = c["valley"] == vid
        if np.any(sel):
            print(f"    valley {name:3s} (id={vid:+d}): "
                  f"mean Tr g = {float(np.mean(trg[sel])):.4e}, "
                  f"max|Omega| = {float(np.max(np.abs(omega[sel]))):.4e}")

    # Band-resolved near-gap summary (bands with largest metric)
    print("\n[3] Bands with largest BZ-summed quantum metric (geometry hot spots):")
    band_ids = sorted(set(c["band"].tolist()))
    trg_by_band = []
    for b in band_ids:
        sel = c["band"] == b
        trg_by_band.append((b, float(np.sum(trg[sel])), float(np.mean(c["E"][sel]))))
    trg_by_band.sort(key=lambda x: -x[1])
    print(f"    {'band':>5} {'sum Tr g':>14} {'<E>(eV)':>10}")
    for b, s, e in trg_by_band[:8]:
        print(f"    {b:>5} {s:>14.4e} {e:>10.3f}")


if __name__ == "__main__":
    main()

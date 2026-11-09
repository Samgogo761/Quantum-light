#!/usr/bin/env python3
"""Export a teaching-only 3D geometry JSON for simple T_perp.

Uses production SCF fractional coordinates and the already-audited nearest
interlayer Cr-Cr bonds (corrected R). Hexagonalizes the in-plane Bravais
metric (numerical C2/m shear only). Does not read tb.dat, does not fit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from q8_tperp_static_audit import (
    INV_C,
    OUT,
    SCF,
    bond_vector,
    cr_sites,
    integer_R,
    parse_scf,
    verify_delta_identity,
    wrap_frac,
)

OUT_JSON = OUT / "q8_geometry_for_visualization.json"

ORBIT = {
    ("tA", "bA"): "AA_BB",
    ("tB", "bB"): "AA_BB",
    ("tA", "bB"): "AB_BA",
    ("tB", "bA"): "AB_BA",
}
BOND_ORDER = (("tB", "bB"), ("tA", "bA"), ("tA", "bB"), ("tB", "bA"))


def _lst(x) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=float)]


def symmetry_cleaned_lattice(lat_prod: np.ndarray) -> dict:
    """Hexagonal in-plane metric; keep production |a1| and c.

    Production a1_y ~ -0.0012 Å and |a1|≠|a2| are numerical C2/m shear.
    HT stacking lives in the fractional Cr coordinates, not in this cleanup.
    """
    a = float(np.linalg.norm(lat_prod[0]))
    c = float(lat_prod[2, 2])
    lat = np.array(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )
    return {
        "lattice_vectors_ang": lat.tolist(),
        "a_ang": a,
        "c_ang": c,
        "rule": (
            "a=|a1_production|; a1=(a,0,0); a2=(-a/2, a√3/2, 0); a3=(0,0,c). "
            "Removes in-plane shear only; fractional coordinates unchanged."
        ),
        "production_lattice_ang": lat_prod.tolist(),
        "lat": lat,
    }


def iodine_sites(atoms) -> list[np.ndarray]:
    return [f for n, f in atoms if n == "I"]


def simple_tperp_bonds(lat: np.ndarray, sites: dict) -> list[dict]:
    bonds = []
    for top, bot in BOND_ORDER:
        rvec = integer_R(sites[top], sites[bot])
        residual = verify_delta_identity(lat, sites[top], sites[bot], rvec)
        delta_frac = rvec.astype(float) + sites[bot] - sites[top]
        wrapped = wrap_frac(sites[bot] - sites[top])
        if np.max(np.abs(delta_frac - wrapped)) > 1.0e-12:
            raise RuntimeError(f"delta != R+tau_to-tau_from for {top}->{bot}")
        vec = bond_vector(lat, sites[top], sites[bot], rvec)
        bonds.append(
            {
                "from": top,
                "to": bot,
                "from_sublattice": top[1],
                "to_sublattice": bot[1],
                "channel": top[1] + bot[1],
                "orbit": ORBIT[(top, bot)],
                "R": [int(x) for x in rvec],
                "delta_frac": _lst(delta_frac),
                "delta_cart_ang": _lst(vec),
                "dxy_ang": float(np.linalg.norm(vec[:2])),
                "d3_ang": float(np.linalg.norm(vec)),
                "delta_identity_residual": residual,
            }
        )
    return bonds


def interface_iodine(lat: np.ndarray, atoms) -> dict:
    iod = iodine_sites(atoms)
    if len(iod) != 12:
        raise RuntimeError(f"expected 12 I, got {len(iod)}")
    # SCF order: 0-2 interface top, 3-5 outer top, 6-8 outer bot, 9-11 interface bot.
    top = iod[0:3]
    bot = iod[9:12]
    def pack(fracs, layer: str, index0: int):
        rows = []
        for i, frac in enumerate(fracs):
            rows.append(
                {
                    "label": f"I_{layer}_{i}",
                    "qe_index": index0 + i,
                    "frac": _lst(frac),
                    "cart_ang": _lst(frac @ lat),
                }
            )
        return rows
    return {"top": pack(top, "top", 0), "bottom": pack(bot, "bot", 9)}


def build_export() -> dict:
    lat_prod, atoms = parse_scf(SCF)
    cleaned = symmetry_cleaned_lattice(lat_prod)
    lat = cleaned["lat"]
    sites = cr_sites(atoms)
    site_block = {}
    for name in ("tA", "tB", "bA", "bB"):
        frac = sites[name]
        site_block[name] = {"frac": _lst(frac), "cart_ang": _lst(frac @ lat)}
    inv_frac = np.asarray(INV_C, dtype=float)
    out = {
        "purpose": "3D teaching model for simple T_perp. Not a new crystal, not a Wannier fit.",
        "units": "angstrom",
        "lattice_vectors_ang": cleaned["lattice_vectors_ang"],
        "lattice_cleanup": {
            "a_ang": cleaned["a_ang"],
            "c_ang": cleaned["c_ang"],
            "rule": cleaned["rule"],
            "production_lattice_ang": cleaned["production_lattice_ang"],
        },
        "sites": site_block,
        "inversion_center": {
            "frac": _lst(inv_frac),
            "cart_ang": _lst(inv_frac @ lat),
            "maps": {"tA": "bB", "tB": "bA", "bA": "tB", "bB": "tA"},
        },
        "simple_T_perp_bonds": simple_tperp_bonds(lat, sites),
        "interface_I": interface_iodine(lat, atoms),
        "notes": {
            "simple_T_perp": (
                "Four nearest interlayer Cr-Cr vectors used by GEO_EQUAL_AMP; "
                "equal-amp illustration only. AA_BB and AB_BA are inversion orbits, "
                "not fitted shells."
            ),
            "coordinates": (
                "Fractional positions are production SCF. Cartesian uses the "
                "cleaned lattice. bA is not the 3.5e-5 Å inversion-paired proposal."
            ),
            "excluded": "tb.dat, hopping amplitudes, fit parameters",
        },
    }
    return out


def main() -> int:
    data = build_export()
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

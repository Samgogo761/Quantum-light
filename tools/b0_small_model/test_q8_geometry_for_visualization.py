#!/usr/bin/env python3
"""Gates for the teaching-geometry JSON. No tb.dat."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_geometry_for_visualization import build_export, simple_tperp_bonds, symmetry_cleaned_lattice
from q8_tperp_static_audit import SCF, cr_sites, parse_scf


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_export() -> None:
    data = build_export()
    lat = np.array(data["lattice_vectors_ang"], dtype=float)
    a1, a2, a3 = lat
    require(abs(a1[1]) < 1.0e-15 and abs(a1[2]) < 1.0e-15, "a1 on x")
    require(abs(a3[0]) < 1.0e-15 and abs(a3[1]) < 1.0e-15, "a3 on z")
    require(abs(float(np.linalg.norm(a1)) - float(np.linalg.norm(a2))) < 1.0e-12, "|a1|=|a2|")
    cosang = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))
    require(abs(math.degrees(math.acos(np.clip(cosang, -1, 1))) - 120.0) < 1.0e-8, "120 deg")
    require(set(data["sites"]) == {"tA", "tB", "bA", "bB"}, data["sites"].keys())
    for name, site in data["sites"].items():
        frac = np.array(site["frac"])
        cart = np.array(site["cart_ang"])
        require(np.allclose(cart, frac @ lat, atol=1.0e-12), name)
    require(len(data["simple_T_perp_bonds"]) == 4, "four bonds")
    orbits = {b["orbit"] for b in data["simple_T_perp_bonds"]}
    require(orbits == {"AA_BB", "AB_BA"}, orbits)
    ab = next(b for b in data["simple_T_perp_bonds"] if b["from"] == "tA" and b["to"] == "bB")
    require(ab["R"] == [0, 1, 0], ab["R"])
    for b in data["simple_T_perp_bonds"]:
        recon = np.array(b["R"], dtype=float) + np.array(data["sites"][b["to"]]["frac"]) - np.array(
            data["sites"][b["from"]]["frac"]
        )
        require(np.allclose(recon, b["delta_frac"], atol=1.0e-12), f"identity {b['from']}->{b['to']}")
        require(b["delta_identity_residual"] < 1.0e-12, b)
    require(len(data["interface_I"]["top"]) == 3, "3 top I")
    require(len(data["interface_I"]["bottom"]) == 3, "3 bot I")
    require(data["interface_I"]["top"][0]["qe_index"] == 0, "top I index")
    require(data["interface_I"]["bottom"][0]["qe_index"] == 9, "bot I index")
    blob = str(data)
    require("ndegen" not in blob, "wannier stats leaked")
    require("sum_abs" not in blob, "hopping stats leaked")
    require("C3_CARTOON" not in blob, "cartoon params leaked")
    lat_prod, atoms = parse_scf(SCF)
    cleaned = symmetry_cleaned_lattice(lat_prod)
    bonds = simple_tperp_bonds(cleaned["lat"], cr_sites(atoms))
    require(len(bonds) == 4, "helper bonds")


def main() -> int:
    test_export()
    print("test_q8_geometry_for_visualization: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

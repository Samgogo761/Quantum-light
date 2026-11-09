#!/usr/bin/env python3
"""Tests for expanded-shell trial nnkp builder."""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from build_shell_trial_nnkp import (
    build_nnkp,
    build_projections,
    expected_counts,
    parse_scf_atoms,
)


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_counts() -> None:
    root = Path(__file__).resolve().parents[2]
    scf = root / "deploy/a0_layerA_m88full/a1_fewk_nscf/CrI3_scf_soc_a1h_clean.in"
    template = root / "deploy/a0_layerA_m88full/a1_fewk_nscf/CrI3_a1o.nnkp"
    atoms = parse_scf_atoms(scf)
    proj, catalog = build_projections(atoms)
    expect = expected_counts()
    require(catalog["n_spinor"] == 288, catalog["n_spinor"])
    by_name = {s["shell"]: s["n_spinor"] for s in catalog["shells"]}
    for key, n in expect.items():
        if key == "total":
            continue
        require(by_name[key] == n, (key, by_name[key], n))
    nnkp = build_nnkp(template.read_text(encoding="utf-8"), proj)
    require("begin spinor_projections\n   288\n" in nnkp, "header")
    require(nnkp.count("begin kpoints") == 1, "kpoints")
    require("    0.0000000000    0.0000000000    0.0000000000" in nnkp, "gamma")
    require(nnkp.count("begin nnkpts") == 1, "nnkpts")
    require("\r" not in nnkp, "crlf")
    require(catalog["rvalue"] == 1, catalog["rvalue"])
    require(catalog["zona_bohr_inv"] == 1.0, catalog["zona_bohr_inv"])
    fams = [
        {"tag": "r1_z0p5", "rvalue": 1, "zona": 0.5},
        {"tag": "r1_z2p0", "rvalue": 1, "zona": 2.0},
        {"tag": "r2_z1p0", "rvalue": 2, "zona": 1.0},
    ]
    proj3, cat3 = build_projections(atoms, fams)
    require(cat3["n_spinor"] == 864, cat3["n_spinor"])
    require(len(cat3["families"]) == 3, len(cat3["families"]))
    nnkp3 = build_nnkp(template.read_text(encoding="utf-8"), proj3)
    require("begin spinor_projections\n   864\n" in nnkp3, "header864")
    require("0.50" in nnkp3, "zona0.5")
    require(any(row.split()[-1] == "2" and row.split()[-2] == "1" for row in nnkp3.splitlines() if len(row.split()) == 6), "rvalue2")


def main() -> int:
    test_counts()
    print("test_build_shell_trial_nnkp: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

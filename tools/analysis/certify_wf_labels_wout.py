#!/usr/bin/env python3
"""Certify production 112 WF labels from CrI3.wout Final State + CrI3.nnkp.

Certified (hard):
  atom / species / bilayer / I vacuum-vs-interface, from Final State centre
  vs win atoms (periodic nearest image).

Initial-projection tags (not a final Ylm dump):
  l, mr, cubic name, spinor sign from CrI3.nnkp spinor_projections.
  Wannierisation can mix same-atom p (or d) channels; keep that status.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

WF_RE = re.compile(
    r"WF centre and spread\s+(\d+)\s+\(\s*([-\d.Ee+]+),\s*([-\d.Ee+]+),\s*([-\d.Ee+]+)\s*\)\s+([-\d.Ee+]+)"
)
L2 = {1: "dz2", 2: "dxz", 3: "dyz", 4: "dx2-y2", 5: "dxy"}
L1 = {1: "pz", 2: "px", 3: "py"}
# Cr planes from win; I between them = interlayer-facing.
Z_CR1 = 0.2388666379
Z_CR2 = 0.5051428940


def parse_win_atoms(win: Path) -> tuple[np.ndarray, list[dict]]:
    text = win.read_text(encoding="utf-8", errors="replace").splitlines()
    cell = np.zeros((3, 3))
    atoms: list[dict] = []
    mode = None
    i_cell = 0
    for line in text:
        s = line.strip()
        low = s.lower()
        if low.startswith("begin unit_cell"):
            mode = "cell"
            continue
        if low.startswith("end unit_cell"):
            mode = None
            continue
        if low.startswith("begin atoms_frac"):
            mode = "atoms"
            continue
        if low.startswith("end atoms_frac"):
            mode = None
            continue
        if mode == "cell":
            if not s or s.lower() == "angstrom":
                continue
            cell[i_cell] = [float(x) for x in s.split()[:3]]
            i_cell += 1
        elif mode == "atoms":
            p = s.split()
            frac = [float(p[1]), float(p[2]), float(p[3])]
            z = frac[2]
            species = "Cr" if p[0].startswith("Cr") else "I"
            layer = 1 if z < 0.40 else 2
            if species == "I":
                site = "interface" if (Z_CR1 < z < Z_CR2) else "vacuum"
            else:
                site = "metal"
            atoms.append(
                {
                    "index": len(atoms),
                    "label": p[0],
                    "frac": frac,
                    "species": species,
                    "layer": layer,
                    "site": site,
                }
            )
    xyz = np.array([a["frac"] for a in atoms]) @ cell
    for a, r in zip(atoms, xyz):
        a["cart"] = r.tolist()
    return cell, atoms


def parse_nnkp_spinor_projections(nnkp: Path) -> list[dict]:
    text = nnkp.read_text(encoding="utf-8", errors="replace").splitlines()
    i0 = next(i for i, ln in enumerate(text) if ln.strip() == "begin spinor_projections")
    n = int(text[i0 + 1].split()[0])
    rows = []
    i = i0 + 2
    while len(rows) < n:
        p = text[i].split()
        l = int(p[3])
        mr = int(p[4])
        spin_line = text[i + 2].split()
        spin = int(float(spin_line[0]))
        if l == 2:
            orb = L2[mr]
        elif l == 1:
            orb = L1[mr]
        else:
            orb = f"l{l}_mr{mr}"
        rows.append(
            {
                "wf": len(rows) + 1,
                "frac_proj": [float(p[0]), float(p[1]), float(p[2])],
                "l": l,
                "mr": mr,
                "spinor_sign": spin,
                "orbital_nnkp": orb,
                "orbital_status": "NNKP_INITIAL_PROJECTION",
            }
        )
        i += 3
    return rows


def parse_final_state(wout: Path) -> list[dict]:
    text = wout.read_text(encoding="utf-8", errors="replace")
    idx = text.rfind(" Final State")
    if idx < 0:
        raise RuntimeError("no Final State in wout")
    chunk = text[idx : idx + 80000]
    rows = []
    for m in WF_RE.finditer(chunk):
        i = int(m.group(1))
        rows.append(
            {
                "wf": i,
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
                "spread": float(m.group(5)),
            }
        )
        if i >= 112:
            break
    if len(rows) != 112:
        raise RuntimeError(f"expected 112 Final State centres, got {len(rows)}")
    return rows


def nearest_atom(cart: np.ndarray, atoms: list[dict], cell: np.ndarray) -> tuple[int, float]:
    best_i, best_d = 0, 1e9
    inv = np.linalg.inv(cell)
    for i, a in enumerate(atoms):
        dcart = cart - np.array(a["cart"])
        df = dcart @ inv
        df -= np.round(df)
        d = float(np.linalg.norm(df @ cell))
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wout", type=Path, required=True)
    ap.add_argument("--win", type=Path, required=True)
    ap.add_argument("--nnkp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-atom-dist", type=float, default=0.6)
    args = ap.parse_args()
    cell, atoms = parse_win_atoms(args.win)
    wfs = parse_final_state(args.wout)
    proj = parse_nnkp_spinor_projections(args.nnkp)
    if len(proj) != 112:
        raise RuntimeError(f"nnkp projections {len(proj)} != 112")
    table = []
    n_far = 0
    n_species_mismatch = 0
    for row, pr in zip(wfs, proj):
        if row["wf"] != pr["wf"]:
            raise RuntimeError("wf index mismatch")
        cart = np.array([row["x"], row["y"], row["z"]])
        ia, dist = nearest_atom(cart, atoms, cell)
        atom = atoms[ia]
        rec = {
            **row,
            **pr,
            "atom_index": ia,
            "atom_label": atom["label"],
            "species": atom["species"],
            "layer": atom["layer"],
            "site": atom["site"],
            "atom_dist_ang": dist,
            "species_status": "NEAREST_ATOM_FINAL_STATE",
            "far": dist > args.max_atom_dist,
        }
        expected = "Cr" if pr["l"] == 2 else "I"
        rec["nnkp_vs_atom_species_ok"] = expected == atom["species"]
        if not rec["nnkp_vs_atom_species_ok"]:
            n_species_mismatch += 1
        if rec["far"]:
            n_far += 1
        table.append(rec)
    cr_wfs = sum(1 for r in table if r["species"] == "Cr")
    i_wfs = sum(1 for r in table if r["species"] == "I")
    out = {
        "source_wout": str(args.wout),
        "source_win": str(args.win),
        "source_nnkp": str(args.nnkp),
        "n_wf": len(table),
        "n_cr_wf": cr_wfs,
        "n_i_wf": i_wfs,
        "expected": {"n_cr_wf": 40, "n_i_wf": 72},
        "counts_match": cr_wfs == 40 and i_wfs == 72,
        "n_far_gt_max_atom_dist": n_far,
        "n_nnkp_vs_atom_species_mismatch": n_species_mismatch,
        "layer_rule": "frac_z < 0.40 -> layer 1 else layer 2",
        "site_rule": "I with Z_CR1 < z < Z_CR2 -> interface, else vacuum",
        "orbital_note": (
            "orbital_nnkp is the initial spinor projection (Wannier90 l,mr). "
            "Final MLWFs may mix same-atom p or d. Species/layer/site are certified."
        ),
        "atoms": atoms,
        "wf": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "counts_match": out["counts_match"],
                "n_cr_wf": cr_wfs,
                "n_i_wf": i_wfs,
                "n_far": n_far,
                "n_nnkp_vs_atom_species_mismatch": n_species_mismatch,
                "layer1_cr": sum(1 for r in table if r["species"] == "Cr" and r["layer"] == 1),
                "layer2_cr": sum(1 for r in table if r["species"] == "Cr" and r["layer"] == 2),
                "layer1_i_interface": sum(
                    1 for r in table if r["species"] == "I" and r["layer"] == 1 and r["site"] == "interface"
                ),
                "layer1_i_vacuum": sum(
                    1 for r in table if r["species"] == "I" and r["layer"] == 1 and r["site"] == "vacuum"
                ),
                "layer2_i_interface": sum(
                    1 for r in table if r["species"] == "I" and r["layer"] == 2 and r["site"] == "interface"
                ),
                "layer2_i_vacuum": sum(
                    1 for r in table if r["species"] == "I" and r["layer"] == 2 and r["site"] == "vacuum"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an AMN-only nnkp with Cr-s/p/d and I-s/p/d hydrogenic spinor trials.

Does not wannierise. Radial part is QE 7.2 pw2wannier90 `radialpart`:
rvalue=1, zona=alpha=1.0 bohr^{-1}:
    R(r) = 2 alpha^{3/2} exp(-alpha r)
This is NOT a spectroscopic I 5d or Cr 4p atomic radial from the UPF.
Keep the A1-O 52-k mesh so AMN can be combined with the existing MMN.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ZAXIS = (0.0, 0.0, 1.0)
XAXIS = (1.0, 0.0, 0.0)
SCHEMA = "shell_trial_nnkp_v2"
DEFAULT_FAMILIES = [{"tag": "r1_z1p0", "rvalue": 1, "zona": 1.0}]
RADIAL_NOTE = {
    1: "R(r)=2*alpha**(3/2)*exp(-alpha*r)",
    2: "R(r)=1/sqrt(8)*alpha**(3/2)*(2-alpha*r)*exp(-alpha*r/2)",
    3: "R(r)=sqrt(4/27)*alpha**(3/2)*(1-2*alpha*r/3+2*(alpha*r)**2/27)*exp(-alpha*r/3)",
}

SHELLS: list[tuple[str, str, int, int]] = [
    ("Cr_s", "Cr", 0, 1),
    ("Cr_p", "Cr", 1, 3),
    ("Cr_d", "Cr", 2, 5),
    ("I_s", "I", 0, 1),
    ("I_p", "I", 1, 3),
    ("I_d", "I", 2, 5),
]

CR_NAMES = {"Cr", "Cr1", "Cr2"}
I_NAMES = {"I"}


def parse_scf_atoms(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    text = path.read_text(encoding="utf-8")
    m = re.search(
        r"ATOMIC_POSITIONS\s*\([^)]+\)\s*(.*?)(?:\n\s*\n|HUBBARD|\Z)",
        text,
        re.S | re.I,
    )
    if m is None:
        raise RuntimeError(f"no ATOMIC_POSITIONS in {path}")
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        if name not in CR_NAMES | I_NAMES:
            continue
        xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
        species = "Cr" if name in CR_NAMES else "I"
        atoms.append((species, xyz))
    if len(atoms) != 16:
        raise RuntimeError(f"expected 16 atoms, got {len(atoms)}")
    n_cr = sum(1 for s, _ in atoms if s == "Cr")
    n_i = sum(1 for s, _ in atoms if s == "I")
    if n_cr != 4 or n_i != 12:
        raise RuntimeError(f"expected 4 Cr + 12 I, got {n_cr} Cr + {n_i} I")
    return atoms


def nnkp_block(text: str, name: str) -> str:
    m = re.search(rf"begin {name}\n(.*?)end {name}\n", text, re.S)
    if m is None:
        raise RuntimeError(f"missing nnkp block {name}")
    return m.group(0)


def fmt_site(
    xyz: tuple[float, float, float], l: int, mr: int, rvalue: int, zona: float
) -> str:
    x, y, z = xyz
    line1 = f"   {x:8.5f}    {y:8.5f}    {z:8.5f}     {l}  {mr}  {rvalue}\n"
    line2 = (
        f"    {ZAXIS[0]:.7f}  {ZAXIS[1]:.7f}  {ZAXIS[2]:.7f}   "
        f"{XAXIS[0]:.7f}  {XAXIS[1]:.7f}  {XAXIS[2]:.7f}    {zona:.2f}\n"
    )
    return line1 + line2


def parse_family(text: str) -> dict:
    tag, r_s, z_s = [p.strip() for p in text.split(",")]
    rvalue = int(r_s)
    zona = float(z_s)
    if rvalue not in RADIAL_NOTE:
        raise RuntimeError(f"unsupported rvalue {rvalue}")
    if zona <= 0:
        raise RuntimeError(f"zona must be positive, got {zona}")
    return {"tag": tag, "rvalue": rvalue, "zona": zona}


def build_projections(
    atoms: list[tuple[str, tuple[float, float, float]]],
    families: list[dict] | None = None,
) -> tuple[str, dict]:
    fams = list(families) if families else list(DEFAULT_FAMILIES)
    chunks: list[str] = []
    family_out: list[dict] = []
    idx0 = 0
    n_proj = 0
    first_shells: list[dict] = []
    for fam in fams:
        rvalue = int(fam["rvalue"])
        zona = float(fam["zona"])
        tag = str(fam["tag"])
        shells: list[dict] = []
        fam_start = idx0
        for name, species, l, n_mr in SHELLS:
            n_here = 0
            for spec, xyz in atoms:
                if spec != species:
                    continue
                for mr in range(1, n_mr + 1):
                    for spin in (1, -1):
                        chunks.append(fmt_site(xyz, l, mr, rvalue, zona))
                        chunks.append(
                            f"    {spin:d}   {ZAXIS[0]:.7f}  {ZAXIS[1]:.7f}  {ZAXIS[2]:.7f}\n"
                        )
                        n_here += 1
                        n_proj += 1
            shells.append(
                {
                    "shell": name,
                    "species": species,
                    "l": l,
                    "n_mr": n_mr,
                    "index0": idx0,
                    "n_spinor": n_here,
                    "family": tag,
                }
            )
            idx0 += n_here
        family_out.append(
            {
                "tag": tag,
                "rvalue": rvalue,
                "zona": zona,
                "radial": RADIAL_NOTE[rvalue],
                "index0": fam_start,
                "n_spinor": idx0 - fam_start,
                "shells": shells,
            }
        )
        if not first_shells:
            first_shells = shells
    body = "".join(chunks)
    header = f"begin spinor_projections\n   {n_proj}\n"
    first = fams[0]
    return header + body + "end spinor_projections\n", {
        "schema": SCHEMA,
        "n_spinor": n_proj,
        "rvalue": int(first["rvalue"]),
        "zona_bohr_inv": float(first["zona"]),
        "radial_rvalue_1": RADIAL_NOTE[1] + "  (QE 7.2 pw2wannier90 radialpart)",
        "not": [
            "spectroscopic I 5d atomic radial",
            "UPF PP_CHI",
            "new Wannier parent",
            "chemical I-d 144",
            "232-band parent",
        ],
        "shells": first_shells if len(fams) == 1 else [s for fam in family_out for s in fam["shells"]],
        "families": family_out,
    }


def build_nnkp(template: str, proj_block: str) -> str:
    keep = [
        template.split("begin spinor_projections")[0].rstrip() + "\n\n",
        proj_block + "\n",
        nnkp_block(template, "nnkpts") + "\n",
        nnkp_block(template, "exclude_bands"),
    ]
    text = "".join(keep)
    if not text.endswith("\n"):
        text += "\n"
    return text.replace("\r\n", "\n")


def expected_counts() -> dict[str, int]:
    return {
        "Cr_s": 4 * 1 * 2,
        "Cr_p": 4 * 3 * 2,
        "Cr_d": 4 * 5 * 2,
        "I_s": 12 * 1 * 2,
        "I_p": 12 * 3 * 2,
        "I_d": 12 * 5 * 2,
        "total": 8 + 24 + 40 + 24 + 72 + 120,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-nnkp", type=Path, required=True)
    ap.add_argument("--scf", type=Path, required=True)
    ap.add_argument("--out-nnkp", type=Path, required=True)
    ap.add_argument("--out-map", type=Path, required=True)
    ap.add_argument(
        "--family",
        action="append",
        default=[],
        help="tag,rvalue,zona  e.g. r1_z0p5,1,0.5  (repeatable)",
    )
    args = ap.parse_args()
    atoms = parse_scf_atoms(args.scf)
    families = [parse_family(x) for x in args.family] if args.family else None
    proj, catalog = build_projections(atoms, families)
    expect = expected_counts()
    n_fam = max(1, len(catalog.get("families") or [0]))
    if catalog["n_spinor"] != expect["total"] * n_fam:
        raise RuntimeError(catalog["n_spinor"])
    for fam in catalog["families"]:
        for sh in fam["shells"]:
            if sh["n_spinor"] != expect[sh["shell"]]:
                raise RuntimeError(sh)
    nnkp = build_nnkp(args.template_nnkp.read_text(encoding="utf-8"), proj)
    args.out_nnkp.write_text(nnkp, encoding="utf-8", newline="\n")
    catalog["source_scf"] = str(args.scf)
    catalog["template_nnkp"] = str(args.template_nnkp)
    args.out_map.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"n_spinor={catalog['n_spinor']} wrote {args.out_nnkp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

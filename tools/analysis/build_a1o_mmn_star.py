#!/usr/bin/env python3
"""Write A1-O NSCF + hand nnkp + pw2wannier90 inputs for the MMN/Berry star."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1o_star_geometry import NNB_CENTER, build_star

PW2WAN = """&INPUTPP
    outdir = './temp_a1h_scf_clean/'
    prefix = 'CrI3_scf_soc'
    seedname = 'CrI3_a1o'
    spin_component = 'none'
    wan_mode = 'standalone'
    write_mmn = .true.
    write_amn = .false.
    write_eig = .true.
    write_unk = .false.
    write_spn = .false.
    regular_mesh = .true.
/
"""


def extract_block(lines: list[str], name: str) -> list[str]:
    out: list[str] = []
    grab = False
    token = f"begin {name}"
    end = f"end {name}"
    for ln in lines:
        if ln.strip() == token:
            grab = True
            out.append(ln.rstrip("\r"))
            continue
        if grab:
            out.append(ln.rstrip("\r"))
            if ln.strip() == end:
                return out
    raise RuntimeError(f"missing nnkp block {name}")


def replace_nscf(template: str, kpoints: list[list[float]]) -> str:
    text = template.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("disk_io = 'low'", "disk_io = 'high'")
    text = text.replace('disk_io = "low"', 'disk_io = "high"')
    marker = "K_POINTS crystal"
    i = text.find(marker)
    if i < 0:
        raise RuntimeError("no K_POINTS crystal in nscf template")
    rest = text[i + len(marker) :]
    hub = rest.find("HUBBARD")
    if hub < 0:
        raise RuntimeError("no HUBBARD after K_POINTS")
    tail = rest[hub:]
    head = text[:i]
    lines = [marker, str(len(kpoints))]
    for kx, ky, kz in kpoints:
        lines.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  1.0")
    return head + "\n".join(lines) + "\n" + tail.lstrip("\n")


def write_nnkp(template_nnkp: Path, star: dict, out_path: Path) -> None:
    src = template_nnkp.read_text(encoding="utf-8", errors="replace").splitlines()
    real = extract_block(src, "real_lattice")
    recip = extract_block(src, "recip_lattice")
    spinor = extract_block(src, "spinor_projections")
    kpts = star["kpoints"]
    chunks = [
        f"File written on A1-O MMN star {date.today().isoformat()}",
        "",
        "calc_only_A  :  F",
        "",
        *real,
        "",
        *recip,
        "",
        "begin kpoints",
        f"   {star['n_k']}",
    ]
    for kx, ky, kz in kpts:
        chunks.append(f"    {kx:.10f}    {ky:.10f}    {kz:.10f}")
    chunks.extend(["end kpoints", "", *spinor, "", "begin nnkpts", f"   {star['nnb']}"])
    for ik, jk, g1, g2, g3 in star["nnkpts"]:
        chunks.append(f"     {ik:3d}   {jk:3d}      {g1}   {g2}   {g3}")
    chunks.extend(["end nnkpts", "", "begin exclude_bands", "   0", "end exclude_bands", ""])
    out_path.write_text("\n".join(chunks), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nscf-template", type=Path, required=True)
    ap.add_argument("--nnkp-template", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    star = build_star()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    nscf = replace_nscf(args.nscf_template.read_text(encoding="utf-8", errors="replace"), star["kpoints"])
    (out / "CrI3_nscf_soc_a1o_star.in").write_text(nscf, encoding="utf-8", newline="\n")
    (out / "CrI3_a1o.pw2wan").write_text(PW2WAN, encoding="utf-8", newline="\n")
    write_nnkp(args.nnkp_template, star, out / "CrI3_a1o.nnkp")
    slim = {k: v for k, v in star.items() if k != "nnkpts"}
    slim["nnb"] = NNB_CENTER
    (out / "a1o_star_map.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_k": star["n_k"],
                "n_centers": star["n_centers"],
                "nnb": star["nnb"],
                "out": str(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

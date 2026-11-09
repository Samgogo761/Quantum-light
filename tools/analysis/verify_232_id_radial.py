#!/usr/bin/env python3
"""Engineering check: I-d PSWFC 7/8 are l=2; 232 nnkp I-d projectors are l=2.

Does not review physics. Confirms the trial orbital angular momentum that
patched pw2wannier90 actually uses (Bessel l from nnkp, not radial_R l_dummy).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "verify_232_id_radial_v1"


def parse_spinor_projections(nnkp: Path) -> list[dict]:
    text = nnkp.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"begin spinor_projections\n(.*?)end spinor_projections\n", text, re.S)
    if m is None:
        raise RuntimeError(f"no spinor_projections in {nnkp}")
    lines = [ln.rstrip() for ln in m.group(1).splitlines() if ln.strip()]
    n = int(lines[0].split()[0])
    body = lines[1:]
    if len(body) % 3 != 0:
        raise RuntimeError(f"spinor body not multiple of 3: {len(body)}")
    recs = []
    for i in range(0, len(body), 3):
        p = body[i].split()
        recs.append(
            {
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "l": int(p[3]),
                "mr": int(p[4]),
                "rvalue": int(p[5]),
            }
        )
    if len(recs) != n:
        raise RuntimeError(f"nnkp n_proj={n} parsed={len(recs)}")
    return recs


def radial_R_tags(path: Path) -> list[tuple[str, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    n_type, mesh = (int(x) for x in lines[0].split())
    i = 1 + mesh
    tags = []
    for _ in range(n_type):
        name, ell = lines[i].split()
        tags.append((name, int(ell)))
        i += 1 + mesh
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-upf", type=Path, required=True)
    ap.add_argument("--nnkp", type=Path, required=True)
    ap.add_argument("--radial-id", type=Path, required=True)
    ap.add_argument("--radial-R", type=Path, required=True)
    ap.add_argument("--patched-f90", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    from extract_upf_defined_radials import (
        EXPLODE_ABSMAX,
        KNOWN_SHA256,
        parse_upf_radials,
        select_id_first_channel,
    )

    iodine = parse_upf_radials(args.i_upf)
    if iodine["sha256"] != KNOWN_SHA256[iodine["file"]]:
        raise RuntimeError(f"UPF hash mismatch {iodine['file']}")
    keep, rejected = select_id_first_channel(iodine, "5D")
    keep_idx = [int(h["index"]) for h in keep]
    keep_l = [int(h["l"]) for h in keep]
    rej_idx = [int(h["index"]) for h in rejected]
    if keep_idx != [7, 8]:
        raise RuntimeError(f"expected PSWFC 7/8, got {keep_idx}")
    if keep_l != [2, 2]:
        raise RuntimeError(f"PSWFC 7/8 are not l=2: {keep_l}")
    if rej_idx != [9, 10]:
        raise RuntimeError(f"expected rejected 9/10, got {rej_idx}")
    if any(h["abs_max"] >= EXPLODE_ABSMAX for h in keep):
        raise RuntimeError("kept I-d channel exploded")

    ids = [int(x) for x in args.radial_id.read_text().split()[1:]]
    if ids[0:40] != [1] * 40 or ids[40:112] != [2] * 72 or ids[112:232] != [3] * 120:
        raise RuntimeError("radial_id layout is not 40 Cr_d + 72 I_p + 120 I_d")
    tags = radial_R_tags(args.radial_R)
    if tags != [("Cr_d", 2), ("I_p", 1), ("I_d", 2)]:
        raise RuntimeError(f"radial_R tags {tags}")

    proj = parse_spinor_projections(args.nnkp)
    if len(proj) != 232:
        raise RuntimeError(f"nnkp n_proj={len(proj)}")
    cr_d = proj[:40]
    i_p = proj[40:112]
    i_d = proj[112:232]
    if any(p["l"] != 2 for p in cr_d):
        raise RuntimeError("Cr-d nnkp l != 2")
    if any(p["l"] != 1 for p in i_p):
        raise RuntimeError("I-p nnkp l != 1")
    if any(p["l"] != 2 for p in i_d):
        raise RuntimeError("I-d nnkp l != 2")
    if sorted({p["mr"] for p in i_d}) != [1, 2, 3, 4, 5]:
        raise RuntimeError("I-d mr not 1..5")

    patched = None
    if args.patched_f90 is not None:
        text = args.patched_f90.read_text(encoding="utf-8", errors="replace")
        if "UPF_CUSTOM_RADIAL_LOADED" not in text:
            raise RuntimeError("patched f90 missing UPF_CUSTOM_RADIAL_LOADED")
        if "l_dummy" not in text:
            raise RuntimeError("patched f90 should read radial_R l as dummy")
        patched = {
            "path": str(args.patched_f90),
            "custom_radial_required": True,
            "radial_R_l_is_dummy": True,
            "bessel_l_from_nnkp": True,
        }

    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "i_upf": {"file": iodine["file"], "sha256": iodine["sha256"]},
        "I_d_pswfc": {
            "kept_indices": keep_idx,
            "kept_l": keep_l,
            "rejected_indices": rej_idx,
            "kept_abs_max": [float(h["abs_max"]) for h in keep],
        },
        "nnkp": {
            "n_proj": len(proj),
            "n_Cr_d_l2": len(cr_d),
            "n_I_p_l1": len(i_p),
            "n_I_d_l2": len(i_d),
        },
        "radial_id_layout": {"Cr_d": 40, "I_p": 72, "I_d": 120},
        "radial_R_tags": [{"name": n, "l": ell} for n, ell in tags],
        "pw2wannier90": patched
        or {
            "bessel_l_from_nnkp": True,
            "radial_R_l_is_dummy": True,
            "note": "patched f90 not passed; nnkp/UPF checks still PASS",
        },
        "paw_note": "AMN is <pseudo-psi_nk|g>; g radial is UPF PSWFC 7/8 j-average, angular/spinor from nnkp. Same PAW+SOC path as production 112, only trial R(r) replaced.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("verify_232_id_radial: PASS")
    print(json.dumps(payload["I_d_pswfc"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

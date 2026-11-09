#!/usr/bin/env python3
"""Build one physically defined radial per trial shell from the production UPF.

Not a zona scan. CHI is used where the PP has atomic wfc; I-d uses the first
well-behaved 5D PAW pseudo partial-wave pair (PSWFC 7/8), never the exploding
high-energy 5D channels. Output is for a patched QE 7.2 pw2wannier90 radialpart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from build_shell_trial_nnkp import expected_counts

SCHEMA = "upf_defined_radials_v1"
XMIN = -6.0
DX = 0.025
RMAX = 10.0
EXPLODE_ABSMAX = 1.0e3
KNOWN_SHA256 = {
    "Cr.rel-pbe.upf": "be756eda2d6d7f07b1f19f6c4ba216cb2cd37615671bf98c872d4904bfaf2e6a",
    "I.rel-pbe.upf": "c7ce679d2426cea1b7508a63104b6a8492529af66860ce7f5afba00f3cc49123",
}
SHELL_ORDER = ["Cr_s", "Cr_p", "Cr_d", "I_s", "I_p", "I_d"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attr(blob: str, key: str) -> str | None:
    m = re.search(rf"""(?:^|[\s]){key}\s*=\s*['"]([^'"]+)['"]""", blob, re.I)
    return m.group(1) if m else None


def _floats(body: str) -> np.ndarray:
    vals: list[float] = []
    for tok in body.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return np.asarray(vals, dtype=float)


def qe_radialpart_mesh(xmin: float = XMIN, dx: float = DX, rmax: float = RMAX) -> np.ndarray:
    """Match QE 7.2 radialpart with alfa=1: r(ir)=exp(xmin+(ir-1)*dx), rmax=10 bohr."""
    mesh_r = int(round((math.log(rmax) - xmin) / dx + 1))
    ir = np.arange(1, mesh_r + 1, dtype=float)
    return np.exp(xmin + (ir - 1.0) * dx)


def _blocks(text: str, tag: str) -> list[dict]:
    out = []
    for m in re.finditer(rf"<{tag}\.(\d+)([^>]*)>(.*?)</{tag}\.\1>", text, re.S | re.I):
        attrs = m.group(2)
        chi = _floats(m.group(3))
        out.append(
            {
                "index": int(m.group(1)),
                "label": _attr(attrs, "label"),
                "l": _attr(attrs, "l") or _attr(attrs, "angular_momentum"),
                "n": _attr(attrs, "n"),
                "occupation": _attr(attrs, "occupation"),
                "chi": chi,
                "abs_max": float(np.max(np.abs(chi))) if chi.size else 0.0,
            }
        )
    return out


def parse_upf_radials(path: Path) -> dict:
    text = path.read_text(errors="replace")
    r_m = re.search(r"<PP_R\b[^>]*>(.*?)</PP_R>", text, re.S | re.I)
    if r_m is None:
        raise RuntimeError(f"no PP_R in {path}")
    r = _floats(r_m.group(1))
    return {
        "file": path.name,
        "path": str(path),
        "sha256": sha256(path),
        "r": r,
        "chi": _blocks(text, "PP_CHI"),
        "pswfc": _blocks(text, "PP_PSWFC"),
        "relbeta": [
            {
                "index": int(m.group(1)),
                "l": _attr(m.group(2), "lll"),
                "j": _attr(m.group(2), "jjj"),
            }
            for m in re.finditer(r"<PP_RELBETA\.(\d+)([^>]*)>", text)
        ],
    }


def _interp_chi(r_src: np.ndarray, chi_src: np.ndarray, r_dst: np.ndarray) -> np.ndarray:
    if r_src.size != chi_src.size:
        raise RuntimeError(f"mesh mismatch {r_src.size} vs {chi_src.size}")
    order = np.argsort(r_src)
    return np.interp(r_dst, r_src[order], chi_src[order], left=0.0, right=0.0)


def chi_to_R(r: np.ndarray, chi: np.ndarray) -> np.ndarray:
    R = np.zeros_like(chi)
    mask = r > 1.0e-12
    R[mask] = chi[mask] / r[mask]
    return R


def normalize_R(r: np.ndarray, R: np.ndarray, dx: float = DX) -> tuple[np.ndarray, float]:
    """Log-mesh Simpson weight of QE radialpart: rij = dx * r, integrand R^2 r^2."""
    w = dx * r
    nrm = float(np.sum((R * r) ** 2 * w))
    if nrm <= 0:
        raise RuntimeError("radial norm is zero")
    return R / math.sqrt(nrm), nrm


def mean_r(r: np.ndarray, R: np.ndarray, dx: float = DX) -> float:
    w = dx * r
    dens = (R * r) ** 2 * w
    return float(np.sum(r * dens) / np.sum(dens))


def j_average_chi(r: np.ndarray, chis: list[np.ndarray], r_dst: np.ndarray) -> np.ndarray:
    stacked = np.vstack([_interp_chi(r, chi, r_dst) for chi in chis])
    return np.mean(stacked, axis=0)


def select_chi(upf: dict, label: str) -> list[dict]:
    hits = [c for c in upf["chi"] if str(c.get("label") or "").upper() == label.upper()]
    if not hits:
        raise RuntimeError(f"{upf['file']}: no PP_CHI label {label}")
    return hits


def select_id_first_channel(upf: dict, label: str = "5D") -> tuple[list[dict], list[dict]]:
    """Keep the lowest-index finite PSWFC pair; reject exploding high-energy channels."""
    hits = [
        p
        for p in upf["pswfc"]
        if str(p.get("label") or "").upper() == label.upper() and str(p.get("l")) == "2"
    ]
    if not hits:
        hits = [p for p in upf["pswfc"] if str(p.get("label") or "").upper() == label.upper()]
    finite = [p for p in hits if p["abs_max"] < EXPLODE_ABSMAX]
    rejected = [p for p in hits if p["abs_max"] >= EXPLODE_ABSMAX]
    if not finite:
        raise RuntimeError(f"{upf['file']}: no finite {label} PSWFC (all exploded)")
    finite = sorted(finite, key=lambda p: int(p["index"]))
    # First SO pair: two lowest finite indices if present, else the single finite channel.
    keep = finite[:2] if len(finite) >= 2 else finite[:1]
    return keep, rejected


def build_shell(upf: dict, r_dst: np.ndarray, label: str, source: str) -> dict:
    if source == "chi":
        hits = select_chi(upf, label)
        chi = j_average_chi(upf["r"], [h["chi"] for h in hits], r_dst)
        used = hits
        rejected: list[dict] = []
    elif source == "pswfc_d_first_channel":
        used, rejected = select_id_first_channel(upf, label)
        chi = j_average_chi(upf["r"], [h["chi"] for h in used], r_dst)
    else:
        raise RuntimeError(source)
    R_raw = chi_to_R(r_dst, chi)
    R, nrm0 = normalize_R(r_dst, R_raw)
    return {
        "source": source,
        "label": label,
        "n_channels_averaged": len(used),
        "channel_indices": [int(h["index"]) for h in used],
        "channel_abs_max": [float(h["abs_max"]) for h in used],
        "rejected_exploding_indices": [int(h["index"]) for h in rejected],
        "norm_before": nrm0,
        "mean_r_bohr": mean_r(r_dst, R),
        "R": R,
        "used": [
            {k: v for k, v in h.items() if k != "chi"}
            for h in used
        ],
    }


def build_all(cr: dict, iodine: dict, r_dst: np.ndarray) -> dict[str, dict]:
    return {
        "Cr_s": build_shell(cr, r_dst, "4S", "chi"),
        "Cr_p": build_shell(cr, r_dst, "3P", "chi"),
        "Cr_d": build_shell(cr, r_dst, "3D", "chi"),
        "I_s": build_shell(iodine, r_dst, "5S", "chi"),
        "I_p": build_shell(iodine, r_dst, "5P", "chi"),
        "I_d": build_shell(iodine, r_dst, "5D", "pswfc_d_first_channel"),
    }


def write_fortran_tables(
    r: np.ndarray,
    shells: dict[str, dict],
    n_spinor_by_shell: dict[str, int],
    radial_id: Path,
    radial_R: Path,
) -> None:
    ids: list[int] = []
    for i, name in enumerate(SHELL_ORDER, start=1):
        ids.extend([i] * int(n_spinor_by_shell[name]))
    radial_id.write_text(str(len(ids)) + "\n" + "\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
    lines = [f"{len(SHELL_ORDER)} {r.size}"]
    lines.extend(f"{x:.16e}" for x in r)
    l_of = {"Cr_s": 0, "Cr_p": 1, "Cr_d": 2, "I_s": 0, "I_p": 1, "I_d": 2}
    for name in SHELL_ORDER:
        lines.append(f"{name} {l_of[name]}")
        lines.extend(f"{x:.16e}" for x in shells[name]["R"])
    radial_R.write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonable_shells(shells: dict[str, dict]) -> dict:
    out = {}
    for name, rec in shells.items():
        out[name] = {
            k: v
            for k, v in rec.items()
            if k not in {"R", "used"}
        }
        out[name]["used"] = rec["used"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cr-upf", type=Path, required=True)
    ap.add_argument("--i-upf", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--radial-id", type=Path, required=True)
    ap.add_argument("--radial-R", type=Path, required=True)
    ap.add_argument("--shell-map", type=Path, default=None)
    ap.add_argument("--require-known-hash", action="store_true")
    args = ap.parse_args()
    cr = parse_upf_radials(args.cr_upf)
    iodine = parse_upf_radials(args.i_upf)
    if args.require_known_hash:
        for rec in (cr, iodine):
            expect = KNOWN_SHA256[rec["file"]]
            if rec["sha256"] != expect:
                raise RuntimeError(f"UPF hash mismatch {rec['file']}: {rec['sha256']} != {expect}")
    r_dst = qe_radialpart_mesh()
    shells = build_all(cr, iodine, r_dst)
    counts = expected_counts()
    n_by = {k: counts[k] for k in SHELL_ORDER}
    if sum(n_by.values()) != counts["total"]:
        raise RuntimeError("shell counts do not sum to total")
    write_fortran_tables(r_dst, shells, n_by, args.radial_id, args.radial_R)
    if args.shell_map is not None:
        idx0 = 0
        map_shells = []
        meta = [
            ("Cr_s", "Cr", 0, 1),
            ("Cr_p", "Cr", 1, 3),
            ("Cr_d", "Cr", 2, 5),
            ("I_s", "I", 0, 1),
            ("I_p", "I", 1, 3),
            ("I_d", "I", 2, 5),
        ]
        for name, species, l, n_mr in meta:
            n_here = n_by[name]
            map_shells.append(
                {
                    "shell": name,
                    "species": species,
                    "l": l,
                    "n_mr": n_mr,
                    "index0": idx0,
                    "n_spinor": n_here,
                    "family": "upf_defined",
                }
            )
            idx0 += n_here
        shell_map = {
            "schema": "shell_trial_nnkp_v2",
            "n_spinor": counts["total"],
            "rvalue": None,
            "zona_bohr_inv": None,
            "radial_rvalue_1": "UPF-defined; see extract_upf_defined_radials.py",
            "trial_note": (
                "UPF CHI / first finite I-5D PAW PSWFC pair. "
                "trial |A|^2 is a screening metric, not an orthogonal percent."
            ),
            "not": [
                "zona scan",
                "spectroscopic I 5d bound orbital",
                "new Wannier parent",
                "chemical I-d 144",
                "232-band parent",
            ],
            "shells": map_shells,
            "families": [
                {
                    "tag": "upf_defined",
                    "rvalue": None,
                    "zona": None,
                    "radial": "UPF CHI + first finite I-5D PSWFC pair",
                    "index0": 0,
                    "n_spinor": counts["total"],
                    "shells": map_shells,
                }
            ],
        }
        args.shell_map.write_text(json.dumps(shell_map, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.shell_map}")
    report = {
        "schema": SCHEMA,
        "NOT": [
            "zona scan",
            "spectroscopic I 5d bound orbital",
            "232-band parent",
            "SCDM",
            "new Wannier parent",
        ],
        "definition": {
            "Cr_s": "PP_CHI 4S (valence s; 3S semicore excluded)",
            "Cr_p": "j-average of PP_CHI 3P SO pair",
            "Cr_d": "j-average of PP_CHI 3D SO pair",
            "I_s": "PP_CHI 5S",
            "I_p": "j-average of PP_CHI 5P SO pair",
            "I_d": (
                "j-average of the first finite I 5D PAW pseudo partial waves "
                "(PP_PSWFC indices 7 and 8; high-energy 9/10 rejected)"
            ),
        },
        "mesh": {
            "xmin": XMIN,
            "dx": DX,
            "rmax_bohr": RMAX,
            "n": int(r_dst.size),
            "note": "QE 7.2 radialpart mesh with alfa=1",
        },
        "upf": {
            "Cr": {"file": cr["file"], "sha256": cr["sha256"]},
            "I": {"file": iodine["file"], "sha256": iodine["sha256"]},
        },
        "n_spinor": counts["total"],
        "shells": jsonable_shells(shells),
        "mean_r_bohr": {name: shells[name]["mean_r_bohr"] for name in SHELL_ORDER},
    }
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.radial_id}")
    print(f"wrote {args.radial_R}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

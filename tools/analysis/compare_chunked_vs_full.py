#!/usr/bin/env python3
"""Compare merged chunked ensemble outputs against a full (unchunked) run.

Requires merged dir to contain:
  - nodes_manifest.input.dat
  - run_metadata.txt (or merge_provenance.txt)
  - HHG_nodes_modes.dat, HHG_ics_cs.dat

Vs Job 27992:
  - hard physics equality: TB/nodes/nk/dt/T2/r/θ/Ibar/harmonics/model
  - build provenance (source/binary/template SHA) is recorded only, not forced equal
  - continuum rows aligned by FFT index iw; allow F10.4/ES16.8 historical rounding
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# Hard-equal when comparing chunked merge to a historical full run.
PHYSICS_KEYS = (
    "tb_sha256",
    "nodes_sha256",
    "nk",
    "model",
    "wvl_nm",
    "dt",
    "T2_cycles",
    "squeeze_r",
    "squeeze_theta_deg",
    "I_bar",
    "harmonics",
)
# Implementation provenance: record, do not force equal to Job 27992.
BUILD_KEYS = ("source_sha256", "binary_sha256", "template_sha256")

# Historical HHG_ics_cs.dat uses F10.4 / ES16.8; new merge uses ES25.17.
H_ORDER_ATOL = 5.1e-5
OMEGA_ATOL = 5.1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata: {path}")
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_modes(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10:
            raise ValueError(f"{path}: expected 10 columns")
        key = (int(f[0]), int(f[4]))
        if key in out:
            raise ValueError(f"{path}: duplicate mode key {key}")
        vals = [float(x) for x in f[1:4] + f[5:]]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError(f"{path}: NaN/Inf at node/order {key}")
        out[key] = {
            "weight": float(f[1]),
            "intensity": float(f[2]),
            "phase": float(f[3]),
            "jx": complex(float(f[5]), float(f[6])),
            "jy": complex(float(f[7]), float(f[8])),
            "power": float(f[9]),
        }
    if not out:
        raise ValueError(f"{path}: no mode rows")
    return out


def load_ics_cs_continuum(path: Path) -> list[dict]:
    """Load continuum rows in file order = FFT bin order (iw = 1..n)."""
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4:
            raise ValueError(f"{path}: expected >=4 columns in ICS/CS continuum")
        h, omega, ics, cs = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
        if not all(math.isfinite(x) for x in (h, omega, ics, cs)):
            raise ValueError(f"{path}: NaN/Inf in continuum row")
        iw = len(rows) + 1
        rows.append({"iw": iw, "h_order": h, "omega": omega, "ICS": ics, "CS": cs})
    if not rows:
        raise ValueError(f"{path}: empty continuum")
    return rows


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def cplx_rel(a: complex, b: complex) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--rtol-modes", type=float, default=1.0e-12)
    ap.add_argument("--rtol-continuum", type=float, default=2.0e-7)
    ap.add_argument("--h-atol", type=float, default=H_ORDER_ATOL)
    ap.add_argument("--omega-atol", type=float, default=OMEGA_ATOL)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    ok = True
    errors: list[str] = []

    man_merged = args.merged / "nodes_manifest.input.dat"
    man_full = args.full_run_dir / "nodes_manifest.input.dat"
    if not man_merged.is_file():
        raise SystemExit(f"FAIL: merged missing required manifest snapshot: {man_merged}")
    if not man_full.is_file():
        raise SystemExit(f"FAIL: full-run missing manifest: {man_full}")
    if sha256(man_merged) != sha256(man_full):
        ok = False
        errors.append("manifest SHA mismatch between merged and full")

    meta_merged_path = args.merged / "run_metadata.txt"
    if not meta_merged_path.is_file():
        meta_merged_path = args.merged / "merge_provenance.txt"
    if not meta_merged_path.is_file():
        raise SystemExit(
            "FAIL: merged missing required run_metadata.txt / merge_provenance.txt"
        )
    meta_full_path = args.full_run_dir / "run_metadata.txt"
    if not meta_full_path.is_file():
        raise SystemExit(f"FAIL: full-run missing run_metadata.txt: {meta_full_path}")

    meta_merged = read_metadata(meta_merged_path)
    meta_full = read_metadata(meta_full_path)

    provenance_cmp: dict[str, dict[str, str]] = {}
    for key in PHYSICS_KEYS:
        a, b = meta_merged.get(key, ""), meta_full.get(key, "")
        provenance_cmp[key] = {"merged": a, "full": b, "required_equal": True}
        if not a or not b:
            ok = False
            errors.append(f"missing physics provenance key {key}")
        elif a != b:
            ok = False
            errors.append(f"physics provenance mismatch {key}: {a!r} vs {b!r}")

    build_note: dict[str, dict[str, object]] = {}
    for key in BUILD_KEYS:
        a, b = meta_merged.get(key, ""), meta_full.get(key, "")
        build_note[key] = {
            "merged": a,
            "full": b,
            "required_equal": False,
            "equal": bool(a) and bool(b) and a == b,
        }

    modes_m = load_modes(args.merged / "HHG_nodes_modes.dat")
    modes_f = load_modes(args.full_run_dir / "HHG_nodes_modes.dat")
    if set(modes_m.keys()) != set(modes_f.keys()):
        ok = False
        errors.append("mode key coverage mismatch between merged and full")

    mode_diffs: dict[str, float] = {}
    for key in sorted(set(modes_m.keys()) & set(modes_f.keys())):
        mm, mf = modes_m[key], modes_f[key]
        for field in ("weight", "intensity", "phase", "power"):
            if field == "phase":
                e = abs(mm[field] - mf[field])
                e = min(e, 2.0 * math.pi - e)
            else:
                e = rel_err(mm[field], mf[field])
            tag = f"node{key[0]}_H{key[1]}_{field}"
            mode_diffs[tag] = e
            if e > args.rtol_modes:
                ok = False
                errors.append(f"{tag} rel_err={e}")
        for field in ("jx", "jy"):
            e = cplx_rel(mm[field], mf[field])
            tag = f"node{key[0]}_H{key[1]}_{field}"
            mode_diffs[tag] = e
            if e > args.rtol_modes:
                ok = False
                errors.append(f"{tag} rel_err={e}")

    cont_m = load_ics_cs_continuum(args.merged / "HHG_ics_cs.dat")
    cont_f = load_ics_cs_continuum(args.full_run_dir / "HHG_ics_cs.dat")
    if len(cont_m) != len(cont_f):
        ok = False
        errors.append(f"continuum length mismatch merged={len(cont_m)} full={len(cont_f)}")

    continuum_stats = {
        "n_omega_merged": len(cont_m),
        "n_omega_full": len(cont_f),
        "align_by": "iw",
        "h_atol": args.h_atol,
        "omega_atol": args.omega_atol,
        "max_ICS_rel_err": 0.0,
        "max_CS_rel_err": 0.0,
        "max_h_abs_err": 0.0,
        "max_omega_abs_err": 0.0,
    }
    n_cmp = min(len(cont_m), len(cont_f))
    for i in range(n_cmp):
        a, b = cont_m[i], cont_f[i]
        if a["iw"] != b["iw"]:
            ok = False
            errors.append(f"continuum iw mismatch at row {i}: {a['iw']} vs {b['iw']}")
            continue
        dh = abs(a["h_order"] - b["h_order"])
        dw = abs(a["omega"] - b["omega"])
        continuum_stats["max_h_abs_err"] = max(continuum_stats["max_h_abs_err"], dh)
        continuum_stats["max_omega_abs_err"] = max(continuum_stats["max_omega_abs_err"], dw)
        if dh > args.h_atol or dw > args.omega_atol:
            ok = False
            errors.append(
                f"continuum iw={a['iw']} grid roundoff too large "
                f"dh={dh:.3e} dw={dw:.3e} (atol h={args.h_atol}, omega={args.omega_atol})"
            )
        e_ics = rel_err(a["ICS"], b["ICS"])
        e_cs = rel_err(a["CS"], b["CS"])
        continuum_stats["max_ICS_rel_err"] = max(continuum_stats["max_ICS_rel_err"], e_ics)
        continuum_stats["max_CS_rel_err"] = max(continuum_stats["max_CS_rel_err"], e_cs)
        if e_ics > args.rtol_continuum or e_cs > args.rtol_continuum:
            ok = False
            errors.append(f"continuum iw={a['iw']} ICS/CS rel_err ICS={e_ics} CS={e_cs}")

    report = {
        "status": "PASS" if ok else "FAIL",
        "merged": str(args.merged.resolve()),
        "full_run_dir": str(args.full_run_dir.resolve()),
        "max_mode_rel_err": max(mode_diffs.values()) if mode_diffs else None,
        "continuum": continuum_stats,
        "physics_provenance": provenance_cmp,
        "build_provenance_informational": build_note,
        "errors": errors[:50],
        "n_errors": len(errors),
        "note": (
            "Aligned by FFT iw; h/omega atol allow F10.4/ES16.8 vs ES25.17. "
            "template/source/binary SHA are informational vs historical jobs."
        ),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

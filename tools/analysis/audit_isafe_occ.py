#!/usr/bin/env python3
"""Audit 29452 classical I-scan occupation. Does not name a global I_safe.

Gates are reported separately:
  POSITIVITY_GATE: rho_nn in [-1e-8, 1+1e-8]
  TOP_BUFFER_GATE: global_edge_max <= 1e-3 AND max(edge/excitation) <= 1%
Overall tested-safe only if both pass. No interpolation; discrete brackets by CEP.
Does not approve k40 / k50 / full GH7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gh7_tail_edge_band_audit import audit_occupation

OCC_LO = -1.0e-8
OCC_HI = 1.0 + 1.0e-8
TRACE_RTOL = 1.0e-8
EDGE_ABS = 1.0e-3
EDGE_REL = 0.01
EXC_FLOOR = 1.0e-6
NV = 84
NB = 112

AUDIT_VERSION = "isafe_occ_bracket_v1_20260819"
LABEL = "FULL112_K20_SAFE_INTENSITY_BRACKET_BY_CEP"
PINNED_BINARY = "34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a"
PINNED_TB = "66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18"
EXPECTED_HEAD = "d1a14318015378df32fad6bbd7e3c95445263bb3"
NK = 20
FAIL_ANCHOR_29403 = {
    "job": "29403",
    "I_Wcm2": 2.813159615258948e12,
    "phi_cep_deg": 265.307396651,
    "cep_family": "GH7c",
    "POSITIVITY_GATE": "FAIL",
    "TOP_BUFFER_GATE": "FAIL",
    "overall": "FAIL",
    "note": "GH7 corner +N occupation; reused as upper FAIL anchor, not rerun",
}

CASES = [
    {
        "dir": "isafe_I1p00e12_cepGH7c_k20_plusN",
        "I_Wcm2": 1.00e12,
        "phi_cep_deg": 265.307396651,
        "cep_family": "GH7c",
        "role": "lower_anchor",
    },
    {
        "dir": "isafe_I1p63e12_cepGH7c_k20_plusN",
        "I_Wcm2": 1.63245553e12,
        "phi_cep_deg": 265.307396651,
        "cep_family": "GH7c",
        "role": "GH5_Imax_GH7_corner_CEP",
    },
    {
        "dir": "isafe_I1p90e12_cepGH7c_k20_plusN",
        "I_Wcm2": 1.90e12,
        "phi_cep_deg": 265.307396651,
        "cep_family": "GH7c",
        "role": "interp",
    },
    {
        "dir": "isafe_I2p20e12_cepGH7c_k20_plusN",
        "I_Wcm2": 2.20e12,
        "phi_cep_deg": 265.307396651,
        "cep_family": "GH7c",
        "role": "interp",
    },
    {
        "dir": "isafe_I2p50e12_cepGH7c_k20_plusN",
        "I_Wcm2": 2.50e12,
        "phi_cep_deg": 265.307396651,
        "cep_family": "GH7c",
        "role": "below_29403_FAIL",
    },
    {
        "dir": "isafe_I1p63e12_cepGH5th0_k20_plusN",
        "I_Wcm2": 1.63245553e12,
        "phi_cep_deg": 85.3073968,
        "cep_family": "GH5th0",
        "role": "antiphase_of_GH7c_at_GH5_Imax",
    },
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _finite_scan(_path: Path) -> dict:
    return {"finite_ok": True, "checked": "math.isfinite on every occupation value"}


def audit_grid_trace_first_violation(path: Path, *, nk: int = NK, nb: int = NB, nv: int = NV) -> dict:
    """Grid uniqueness/completeness and trace; positivity is recorded, not a parse abort."""
    expected_nk = nk * nk
    expected_trace0 = float(nv * expected_nk)
    expected_per_snap = expected_nk * nb
    snaps: dict[int, dict] = {}
    first_hi = None
    first_lo = None
    n_gt_hi = 0
    n_lt_lo = 0
    n_rows = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"{path}:{lineno}: expected 6 columns")
            it = int(fields[0])
            t_fs = float(fields[1])
            ikx = int(fields[2])
            iky = int(fields[3])
            band = int(fields[4])
            occ = float(fields[5])
            n_rows += 1
            if not math.isfinite(occ):
                raise ValueError(f"non-finite occupation in {path}:{lineno}")
            rec = snaps.setdefault(
                it,
                {"t_fs": t_fs, "trace": 0.0, "keys": set(), "n_rows": 0, "exc": 0.0, "edge": 0.0},
            )
            key = (ikx, iky, band)
            if key in rec["keys"]:
                raise ValueError(f"duplicate (ikx,iky,band)={key} at it={it}")
            rec["keys"].add(key)
            rec["n_rows"] += 1
            rec["trace"] += occ
            if band > nv:
                rec["exc"] += occ if occ > 0.0 else 0.0
            if band >= nb - 7:
                rec["edge"] += occ if occ > 0.0 else 0.0
            loc = {"it": it, "t_fs": t_fs, "ikx": ikx, "iky": iky, "band": band, "occ": occ}
            if occ > OCC_HI:
                n_gt_hi += 1
                if first_hi is None or (t_fs, it) < (first_hi["t_fs"], first_hi["it"]):
                    first_hi = loc
            if occ < OCC_LO:
                n_lt_lo += 1
                if first_lo is None or (t_fs, it) < (first_lo["t_fs"], first_lo["it"]):
                    first_lo = loc

    if not snaps:
        raise ValueError(f"no occupation rows in {path}")
    want = {(ix, iy, b) for ix in range(1, nk + 1) for iy in range(1, nk + 1) for b in range(1, nb + 1)}
    for it, rec in snaps.items():
        missing = want - rec["keys"]
        extra = rec["keys"] - want
        if missing or extra or rec["n_rows"] != expected_per_snap:
            raise ValueError(
                f"it={it}: grid incomplete rows={rec['n_rows']} expected={expected_per_snap} "
                f"missing={len(missing)} extra={len(extra)}"
            )
        rec.pop("keys")

    its = sorted(snaps)
    if len(its) < 2:
        raise ValueError(f"{path}: need first and last snapshots, got {its}")
    tr0 = snaps[its[0]]["trace"]
    max_drift = 0.0
    max_edge_rel = 0.0
    for it in its:
        rec = snaps[it]
        drift = abs(rec["trace"] - tr0) / max(abs(tr0), 1.0e-300)
        max_drift = max(max_drift, drift)
        if rec["exc"] >= EXC_FLOOR:
            max_edge_rel = max(max_edge_rel, rec["edge"] / rec["exc"])
    return {
        "n_rows": n_rows,
        "n_snapshots": len(its),
        "it_first": its[0],
        "it_last": its[-1],
        "t_first_fs": snaps[its[0]]["t_fs"],
        "t_last_fs": snaps[its[-1]]["t_fs"],
        "rows_per_snapshot": expected_per_snap,
        "unique_complete_grid": True,
        "trace0": tr0,
        "expected_trace0": expected_trace0,
        "pass_trace0": abs(tr0 - expected_trace0) / max(expected_trace0, 1.0e-300) <= TRACE_RTOL,
        "max_trace_rel_drift": max_drift,
        "pass_trace": max_drift <= TRACE_RTOL,
        "max_edge_rel_from_grid": max_edge_rel,
        "n_gt_occ_hi": n_gt_hi,
        "n_lt_occ_lo": n_lt_lo,
        "first_overshoot": first_hi,
        "first_undershoot": first_lo,
        "finite_scan": _finite_scan(path),
    }


def _gate(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def bracket_for_family(rows: list[dict], family: str, fail_anchor: dict | None) -> dict:
    pts = [r for r in rows if r["cep_family"] == family]
    pts = sorted(pts, key=lambda r: r["I_Wcm2"])
    if fail_anchor is not None and fail_anchor.get("cep_family") == family:
        pts = pts + [
            {
                "I_Wcm2": fail_anchor["I_Wcm2"],
                "overall": fail_anchor["overall"],
                "source": "job_29403_anchor",
            }
        ]
        pts = sorted(pts, key=lambda r: r["I_Wcm2"])
    overall = [(p["I_Wcm2"], p["overall"]) for p in pts]
    pass_is = [p["I_Wcm2"] for p in pts if p["overall"] == "PASS"]
    fail_is = [p["I_Wcm2"] for p in pts if p["overall"] == "FAIL"]
    monotonic_ok = True
    seen_fail = False
    for _i, status in overall:
        if status == "FAIL":
            seen_fail = True
        elif status == "PASS" and seen_fail:
            monotonic_ok = False
    if not pass_is or not fail_is or not monotonic_ok:
        return {
            "cep_family": family,
            "valid_bracket": False,
            "reason": "no_valid_PASS_FAIL_bracket" if monotonic_ok else "non_monotonic_PASS_FAIL",
            "I_P": max(pass_is) if pass_is else None,
            "I_F": min(fail_is) if fail_is else None,
            "sequence": overall,
        }
    i_p = max(pass_is)
    i_f = min(fail_is)
    if not (i_p < i_f):
        return {
            "cep_family": family,
            "valid_bracket": False,
            "reason": "I_P_not_below_I_F",
            "I_P": i_p,
            "I_F": i_f,
            "sequence": overall,
        }
    return {
        "cep_family": family,
        "valid_bracket": True,
        "I_P": i_p,
        "I_F": i_f,
        "interval": [i_p, i_f],
        "label": LABEL,
        "note": "highest tested PASS and lowest tested FAIL at this CEP; not a fitted threshold",
        "sequence": overall,
    }


def audit_case(root: Path, spec: dict) -> dict:
    outdir = root / spec["dir"]
    required = (
        "SUCCESS",
        "run_status.txt",
        "run_metadata.txt",
        "input.nml",
        "occupation_band_kt.dat",
        "occupation_kt.dat",
        "output_sha256.txt",
        "Jt.dat",
        "Et.dat",
        "HHG.dat",
    )
    missing = [name for name in required if not (outdir / name).is_file()]
    status = _kv(outdir / "run_status.txt")
    meta = _kv(outdir / "run_metadata.txt")
    exec_pass = (
        not missing
        and status.get("status") == "PASS"
        and (outdir / "SUCCESS").is_file()
        and meta.get("binary_sha256") == PINNED_BINARY
        and meta.get("tb_sha256") == PINNED_TB
        and meta.get("campaign_actual_head") == EXPECTED_HEAD
        and meta.get("bsv_enabled") == "false"
    )
    occ_path = outdir / "occupation_band_kt.dat"
    occ = audit_occupation(occ_path)
    grid = audit_grid_trace_first_violation(occ_path)
    min_f = occ["global_min"]["occ"]
    max_f = occ["global_max"]["occ"]
    pos_ok = (
        occ["n_gt_occ_hi"] == 0
        and occ["n_lt_occ_lo"] == 0
        and min_f >= OCC_LO
        and max_f <= OCC_HI
    )
    edge_max = occ["global_edge_max"]["occ"]
    edge_rel = occ["edge_peak"]["edge_over_cond"]
    # campaign-max edge/exc from time series
    rels = [row["edge_over_cond"] for row in occ["time_series"] if row["edge_over_cond"] is not None]
    max_edge_rel = max(rels) if rels else 0.0
    band_ok = edge_max <= EDGE_ABS and max_edge_rel <= EDGE_REL
    first_init = next((row for row in occ["time_series"] if row["it"] == occ["it_first"]), None)
    last = next((row for row in occ["time_series"] if row["it"] == occ["it_last"]), None)
    peak = occ["edge_peak"]
    positivity = _gate(pos_ok)
    top_buffer = _gate(band_ok)
    overall = "PASS" if (exec_pass and pos_ok and band_ok and grid["pass_trace"] and grid["pass_trace0"]) else "FAIL"
    bands = []
    for item in occ["bands"]:
        bands.append(
            {
                "band": item["band"],
                "max_occ": item["max_occ"],
                "max_loc": item["max_loc"],
                "max_pos_sum": item["max_pos_sum"],
                "at_edge_peak_max_f": item["at_edge_peak"]["max_f"],
                "at_edge_peak_pos_sum": item["at_edge_peak"]["pos_sum"],
            }
        )
    return {
        "dir": spec["dir"],
        "role": spec["role"],
        "cep_family": spec["cep_family"],
        "I_Wcm2": spec["I_Wcm2"],
        "phi_cep_deg": spec["phi_cep_deg"],
        "execution": "PASS" if exec_pass else "FAIL",
        "missing_files": missing,
        "POSITIVITY_GATE": positivity,
        "TOP_BUFFER_GATE": top_buffer,
        "TRACE_GATE": _gate(grid["pass_trace"] and grid["pass_trace0"]),
        "overall": overall,
        "min_f": min_f,
        "max_f": max_f,
        "min_loc": occ["global_min"],
        "max_loc": occ["global_max"],
        "first_overshoot": grid["first_overshoot"],
        "first_undershoot": grid["first_undershoot"],
        "n_gt_occ_hi": occ["n_gt_occ_hi"],
        "n_lt_occ_lo": occ["n_lt_occ_lo"],
        "worst_overshoot": occ["worst_overshoot"],
        "worst_undershoot": occ["worst_undershoot"],
        "max_trace_rel_drift": grid["max_trace_rel_drift"],
        "trace0": grid["trace0"],
        "global_edge_max": occ["global_edge_max"],
        "edge_over_exc_at_edge_peak": edge_rel,
        "max_edge_over_exc": max_edge_rel,
        "bands_105_112": bands,
        "decay": occ["decay"],
        "t_init_fs": grid["t_first_fs"],
        "t_final_fs": grid["t_last_fs"],
        "edge_pop_init": None if first_init is None else first_init["edge_pos"],
        "edge_pop_final": None if last is None else last["edge_pos"],
        "edge_pop_peak": peak["edge_pos"],
        "edge_peak_t_fs": peak["t_fs"],
        "n_rows": grid["n_rows"],
        "n_snapshots": grid["n_snapshots"],
        "unique_complete_grid": grid["unique_complete_grid"],
        "finite_ok": grid["finite_scan"]["finite_ok"],
        "provenance": {
            "slurm_job_id": meta.get("slurm_job_id"),
            "campaign_actual_head": meta.get("campaign_actual_head"),
            "binary_sha256": meta.get("binary_sha256"),
            "tb_sha256": meta.get("tb_sha256"),
            "input_sha256_meta": meta.get("input_sha256"),
            "input_sha256_file": sha256_file(outdir / "input.nml") if (outdir / "input.nml").is_file() else None,
            "run_metadata_sha256": sha256_file(outdir / "run_metadata.txt") if (outdir / "run_metadata.txt").is_file() else None,
            "run_status_sha256": sha256_file(outdir / "run_status.txt") if (outdir / "run_status.txt").is_file() else None,
            "output_sha256_txt": sha256_file(outdir / "output_sha256.txt") if (outdir / "output_sha256.txt").is_file() else None,
            "occupation_band_sha256": sha256_file(occ_path),
            "intensity_Wcm2_meta": meta.get("intensity_Wcm2"),
            "phi_cep_deg_meta": meta.get("phi_cep_deg"),
            "omp": meta.get("omp_num_threads"),
            "mkl": meta.get("mkl_num_threads"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--sacct", type=Path, default=None)
    parser.add_argument("--sbatch-sha256", default=None)
    parser.add_argument("--submit-sha256", default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = [audit_case(root, spec) for spec in CASES]
    gh7c = bracket_for_family(rows, "GH7c", FAIL_ANCHOR_29403)
    gh5 = bracket_for_family(rows, "GH5th0", None)
    i163 = [r for r in rows if abs(r["I_Wcm2"] - 1.63245553e12) < 1.0]
    antiphase = {
        "same_I": 1.63245553e12,
        "n_points": len(i163),
        "overalls": {r["cep_family"]: r["overall"] for r in i163},
        "positivity": {r["cep_family"]: r["POSITIVITY_GATE"] for r in i163},
        "top_buffer": {r["cep_family"]: r["TOP_BUFFER_GATE"] for r in i163},
        "note": "85.307 deg and 265.307 deg differ by pi; this is antiphase, not CEP coverage",
    }
    report = {
        "audit": AUDIT_VERSION,
        "label": LABEL,
        "auto_chain": False,
        "approve_k40": False,
        "approve_k50": False,
        "approve_full_gh7": False,
        "cep_coverage": "antiphase_pair_only",
        "note": (
            "Discrete tested-safe intensity bracket by CEP at full112/k20/dt=0.35/T2=0.5. "
            "Not a global I_safe. POSITIVITY_GATE and TOP_BUFFER_GATE are reported separately. "
            "Do not interpolate a threshold. Do not auto-submit k40/k50/full GH7."
        ),
        "root": str(root),
        "slurm_array_job_id": "29452",
        "fail_anchor_29403": FAIL_ANCHOR_29403,
        "gates": {
            "OCC_LO": OCC_LO,
            "OCC_HI": OCC_HI,
            "EDGE_ABS": EDGE_ABS,
            "EDGE_REL": EDGE_REL,
            "TRACE_RTOL": TRACE_RTOL,
        },
        "provenance": {
            "sbatch_sha256": args.sbatch_sha256,
            "submit_sha256": args.submit_sha256,
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
            "pinned_binary_sha256": PINNED_BINARY,
            "pinned_tb_sha256": PINNED_TB,
            "expected_head": EXPECTED_HEAD,
            "sacct_file": str(args.sacct) if args.sacct else None,
            "sacct_sha256": sha256_file(args.sacct) if args.sacct and args.sacct.is_file() else None,
        },
        "cases": rows,
        "brackets": {"GH7c": gh7c, "GH5th0": gh5},
        "antiphase_at_GH5_Imax": antiphase,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    print(f"label={LABEL}")
    for row in rows:
        print(
            f"{row['dir']} I={row['I_Wcm2']:.6e} POS={row['POSITIVITY_GATE']} "
            f"BUF={row['TOP_BUFFER_GATE']} OVERALL={row['overall']} "
            f"edge_max={row['global_edge_max']['occ']:.6e} "
            f"edge/exc={row['max_edge_over_exc']:.6e}"
        )
    print("GH7c_bracket=" + json.dumps(gh7c, default=str))
    print("GH5th0_bracket=" + json.dumps(gh5, default=str))
    any_fail_exec = any(r["execution"] != "PASS" for r in rows)
    return 1 if any_fail_exec else 0


if __name__ == "__main__":
    raise SystemExit(main())

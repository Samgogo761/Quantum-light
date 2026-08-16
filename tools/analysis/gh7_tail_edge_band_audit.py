#!/usr/bin/env python3
"""Zero-cost GH7 tail edge-band audit. Does not relax gates or approve k40.

Reads occupation_band_kt.dat (equilibrium-band rho_nn) and HHG_nodes_modes.dat.
Reports bands 105-112 occupancy, f overshoot locations, and w|J|^2 / w|J|^4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO = Path(os.environ["REPO"]) if os.environ.get("REPO") else _HERE.parents[1]
sys.path.insert(0, str(_REPO / "tools" / "analysis"))
sys.path.insert(0, str(_REPO / "deploy" / "a0_layerA_m88full" / "gh7_tail"))

from gh7_nodes import case_name
from rank_full112_gh3_candidates import amp, load_node_modes

EDGE_LO = 105
EDGE_HI = 112
NV = 84
HARMONICS = (2, 5, 7, 9, 10)
OCC_LO = -1.0e-8
OCC_HI = 1.0 + 1.0e-8
TOP_F_THRESH = 1.0e-3
AUDIT_VERSION = "gh7_tail_edge_band_v2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _loc(it: int, t_fs: float, ikx: int, iky: int, band: int, occ: float) -> dict:
    return {"it": it, "t_fs": t_fs, "ikx": ikx, "iky": iky, "band": band, "occ": occ}


def probe_weight(freeze: dict, probe: dict) -> float:
    man = freeze["manifests"][probe["manifest"]]
    block = man["corner_pair"] if probe["kind"] == "corner" else man["wide_axis_pair"]
    ids = [int(x) for x in block["ids"]]
    weights = [float(x) for x in block["weights"]]
    return weights[ids.index(int(probe["id"]))]


def jones_weighted(modes_path: Path, node_id: int, harmonics: tuple[int, ...] = HARMONICS) -> dict:
    modes = load_node_modes(modes_path)
    out = {}
    for order in harmonics:
        key = (int(node_id), int(order))
        if key not in modes:
            raise ValueError(f"{modes_path}: missing node {node_id} harmonic {order}")
        jx, jy, w = modes[key]
        j2 = abs(jx) ** 2 + abs(jy) ** 2
        out[str(order)] = {
            "w": w,
            "jx_re": jx.real,
            "jx_im": jx.imag,
            "jy_re": jy.real,
            "jy_im": jy.imag,
            "amp": amp(jx, jy),
            "j2": j2,
            "w_j2": w * j2,
            "w_j4": w * (j2 ** 2),
        }
    return out


def audit_occupation(path: Path, *, edge_lo: int = EDGE_LO, edge_hi: int = EDGE_HI) -> dict:
    edge_bands = list(range(edge_lo, edge_hi + 1))
    per_band = {
        b: {
            "max_occ": -1.0e300,
            "max_loc": None,
            "min_occ": 1.0e300,
            "min_loc": None,
            "max_pos_sum": 0.0,
            "max_pos_sum_it": None,
        }
        for b in edge_bands
    }
    # it -> {t_fs, pos_sum[b], max_f[b], cond_pos, edge_pos}
    snaps: dict[int, dict] = {}
    worst_hi = None
    worst_lo = None
    n_gt_hi = 0
    n_lt_lo = 0
    n_rows = 0
    global_max = None
    global_min = None
    global_edge_max = None

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            it = int(f[0])
            t_fs = float(f[1])
            ikx = int(f[2])
            iky = int(f[3])
            band = int(f[4])
            occ = float(f[5])
            n_rows += 1
            loc = _loc(it, t_fs, ikx, iky, band, occ)
            if global_max is None or occ > global_max["occ"]:
                global_max = loc
            if global_min is None or occ < global_min["occ"]:
                global_min = loc
            if occ > OCC_HI:
                n_gt_hi += 1
                if worst_hi is None or occ > worst_hi["occ"]:
                    worst_hi = loc
            if occ < OCC_LO:
                n_lt_lo += 1
                if worst_lo is None or occ < worst_lo["occ"]:
                    worst_lo = loc
            rec = snaps.setdefault(
                it,
                {
                    "t_fs": t_fs,
                    "pos_sum": {b: 0.0 for b in edge_bands},
                    "max_f": {b: -1.0e300 for b in edge_bands},
                    "cond_pos": 0.0,
                    "edge_pos": 0.0,
                    "edge_max": -1.0e300,
                    "edge_max_loc": None,
                },
            )
            pos = occ if occ > 0.0 else 0.0
            if band > NV:
                rec["cond_pos"] += pos
            if edge_lo <= band <= edge_hi:
                rec["pos_sum"][band] += pos
                rec["edge_pos"] += pos
                if occ > rec["max_f"][band]:
                    rec["max_f"][band] = occ
                if occ > rec["edge_max"]:
                    rec["edge_max"] = occ
                    rec["edge_max_loc"] = loc
                if global_edge_max is None or occ > global_edge_max["occ"]:
                    global_edge_max = loc
                st = per_band[band]
                if occ > st["max_occ"]:
                    st["max_occ"] = occ
                    st["max_loc"] = loc
                if occ < st["min_occ"]:
                    st["min_occ"] = occ
                    st["min_loc"] = loc

    if not snaps:
        raise ValueError(f"no occupation rows in {path}")

    for it, rec in snaps.items():
        for b in edge_bands:
            ps = rec["pos_sum"][b]
            st = per_band[b]
            if ps >= st["max_pos_sum"]:
                st["max_pos_sum"] = ps
                st["max_pos_sum_it"] = {"it": it, "t_fs": rec["t_fs"]}

    its = sorted(snaps)
    peak_it = max(its, key=lambda it: snaps[it]["edge_pos"])
    peak = snaps[peak_it]
    ladder = []
    for b in edge_bands:
        ladder.append(
            {
                "band": b,
                "max_occ": per_band[b]["max_occ"],
                "max_loc": per_band[b]["max_loc"],
                "min_occ": per_band[b]["min_occ"],
                "max_pos_sum": per_band[b]["max_pos_sum"],
                "max_pos_sum_it": per_band[b]["max_pos_sum_it"],
                "at_edge_peak": {
                    "max_f": peak["max_f"][b],
                    "pos_sum": peak["pos_sum"][b],
                },
            }
        )
    s105 = peak["pos_sum"].get(edge_lo, 0.0)
    s111 = peak["pos_sum"].get(111, 0.0) if 111 in peak["pos_sum"] else None
    s112 = peak["pos_sum"].get(edge_hi, 0.0)
    m105 = peak["max_f"].get(edge_lo, 0.0)
    m111 = peak["max_f"].get(111, 0.0) if 111 in peak["max_f"] else None
    m112 = peak["max_f"].get(edge_hi, 0.0)
    decay = {
        "edge_peak_it": peak_it,
        "edge_peak_t_fs": peak["t_fs"],
        "pos_sum_105": s105,
        "pos_sum_111": s111,
        "pos_sum_112": s112,
        "max_f_105": m105,
        "max_f_111": m111,
        "max_f_112": m112,
        "pos_sum_112_over_105": (s112 / s105) if s105 > 0.0 else None,
        "max_f_112_over_105": (m112 / m105) if m105 > 0.0 else None,
        "top_two_still_occupied": bool(
            m111 is not None and m111 > TOP_F_THRESH and m112 > TOP_F_THRESH
        ),
        "top_f_thresh": TOP_F_THRESH,
        "top_two_rule": "AND: band111 and band112 max_f both exceed top_f_thresh at edge-sum peak",
    }
    time_series = []
    for it in its:
        rec = snaps[it]
        time_series.append(
            {
                "it": it,
                "t_fs": rec["t_fs"],
                "edge_pos": rec["edge_pos"],
                "cond_pos": rec["cond_pos"],
                "edge_over_cond": (rec["edge_pos"] / rec["cond_pos"]) if rec["cond_pos"] > 0.0 else None,
                "edge_max": rec["edge_max"],
                "bands": {
                    str(b): {"max_f": rec["max_f"][b], "pos_sum": rec["pos_sum"][b]}
                    for b in edge_bands
                },
            }
        )
    return {
        "n_rows": n_rows,
        "n_snapshots": len(its),
        "it_first": its[0],
        "it_last": its[-1],
        "global_max": global_max,
        "global_min": global_min,
        "global_edge_max": global_edge_max,
        "n_gt_occ_hi": n_gt_hi,
        "n_lt_occ_lo": n_lt_lo,
        "worst_overshoot": worst_hi,
        "worst_undershoot": worst_lo,
        "edge_peak": {
            "it": peak_it,
            "t_fs": peak["t_fs"],
            "edge_pos": peak["edge_pos"],
            "cond_pos": peak["cond_pos"],
            "edge_over_cond": (peak["edge_pos"] / peak["cond_pos"]) if peak["cond_pos"] > 0.0 else None,
            "edge_max_at_edge_sum_peak": peak["edge_max"],
            "edge_max_loc": peak["edge_max_loc"],
        },
        "bands": ladder,
        "decay": decay,
        "time_series": time_series,
    }


def _kv_file(path: Path) -> dict[str, str]:
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


def case_product_sha(outdir: Path) -> dict:
    names = (
        "input.nml",
        "occupation_band_kt.dat",
        "HHG_nodes_modes.dat",
        "run_metadata.txt",
        "run_status.txt",
        "chunk_info.txt",
        "SUCCESS",
    )
    shas = {}
    for name in names:
        path = outdir / name
        shas[name] = sha256_file(path) if path.is_file() else None
    meta = _kv_file(outdir / "run_metadata.txt")
    status = _kv_file(outdir / "run_status.txt")
    return {
        "files": shas,
        "slurm_job_id": meta.get("slurm_job_id") or status.get("slurm_job_id"),
        "campaign_actual_head": meta.get("campaign_actual_head"),
        "freeze_pin_base_head": meta.get("freeze_pin_base_head"),
        "binary_sha256": meta.get("binary_sha256"),
        "tb_sha256": meta.get("tb_sha256"),
        "input_sha256_meta": meta.get("input_sha256"),
        "freeze_sha256_meta": meta.get("freeze_sha256"),
    }


def audit_campaign(
    freeze: dict,
    k20_root: Path,
    *,
    nk: int = 20,
    freeze_path: Path | None = None,
    audit_script_sha256: str | None = None,
    slurm_array_job_id: str | None = None,
    omit_time_series: bool = False,
) -> dict:
    probes = freeze["plusN_probes"]
    cases = []
    product_sha: dict[str, dict] = {}
    heads: set[str] = set()
    for probe in probes:
        name = case_name(probe, nk)
        outdir = k20_root / name
        weight = probe_weight(freeze, probe)
        occ = audit_occupation(outdir / "occupation_band_kt.dat")
        if omit_time_series:
            occ.pop("time_series", None)
        jones = jones_weighted(outdir / "HHG_nodes_modes.dat", int(probe["id"]))
        prod = case_product_sha(outdir)
        product_sha[name] = prod
        if prod.get("campaign_actual_head"):
            heads.add(prod["campaign_actual_head"])
        cases.append(
            {
                "case": name,
                "kind": probe["kind"],
                "id": int(probe["id"]),
                "theta_deg": int(probe["theta_deg"]),
                "I_over_Imax": probe.get("I_over_Imax"),
                "quadrature_weight": weight,
                "occupation": occ,
                "jones": jones,
            }
        )

    def _group(kind: str) -> dict:
        rows = [c for c in cases if c["kind"] == kind]
        return {
            "n": len(rows),
            "cases": [c["case"] for c in rows],
            "global_edge_max": max(c["occupation"]["global_edge_max"]["occ"] for c in rows),
            "edge_max_at_edge_sum_peak": max(
                c["occupation"]["edge_peak"]["edge_max_at_edge_sum_peak"] for c in rows
            ),
            "max_edge_over_cond": max(
                c["occupation"]["edge_peak"]["edge_over_cond"] or 0.0 for c in rows
            ),
            "max_f_112": max(c["occupation"]["decay"]["max_f_112"] for c in rows),
            "max_pos_sum_112": max(c["occupation"]["decay"]["pos_sum_112"] for c in rows),
            "top_two_still_occupied": all(
                c["occupation"]["decay"]["top_two_still_occupied"] for c in rows
            ),
            "w_j2": {
                str(h): sum(c["jones"][str(h)]["w_j2"] for c in rows) for h in HARMONICS
            },
            "w_j4": {
                str(h): sum(c["jones"][str(h)]["w_j4"] for c in rows) for h in HARMONICS
            },
        }

    job_txt = k20_root.parent / "job_id.txt"
    gate_report = k20_root / "GH7_TAIL_K20.json"
    freeze_sha = sha256_file(freeze_path) if freeze_path is not None else None
    sidecar = None
    if freeze_path is not None:
        side_path = freeze_path.with_name("FREEZE.sha256")
        if side_path.is_file():
            sidecar = side_path.read_text(encoding="utf-8").strip().split()[0]
    return {
        "audit": AUDIT_VERSION,
        "auto_chain": False,
        "approve_k40": False,
        "cep_pi_tested": False,
        "gates_unchanged": True,
        "dt2_status": "POSITIVITY_DIAGNOSTIC_ONLY",
        "note": (
            "Diagnostic only. Does not relax f in [0,1] or edge thresholds. "
            "Does not quantify HHG percent error from missing bands. "
            "global_edge_max is the max rho_nn over bands 105-112, any k, any snapshot; "
            "edge_max_at_edge_sum_peak is the max at the snapshot of maximum top-8 positive sum. "
            "Neither is the fraction of electrons that left the model. "
            "full112 already uses the entire 112-dimensional TB. "
            "top_two_still_occupied is AND: both band111 and band112 exceed top_f_thresh."
        ),
        "edge_bands": [EDGE_LO, EDGE_HI],
        "n_cases": len(cases),
        "provenance": {
            "slurm_array_job_id": slurm_array_job_id
            or (job_txt.read_text(encoding="utf-8").strip() if job_txt.is_file() else None),
            "campaign_actual_heads": sorted(heads),
            "k20_root": str(k20_root),
            "freeze_path": str(freeze_path) if freeze_path is not None else None,
            "freeze_sha256": freeze_sha,
            "freeze_sidecar_sha256": sidecar,
            "audit_script_sha256": audit_script_sha256,
            "gate_report_sha256": sha256_file(gate_report) if gate_report.is_file() else None,
            "cases": product_sha,
        },
        "cases": cases,
        "grouped": {"corner": _group("corner"), "wide": _group("wide")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--k20-root", type=Path, required=True)
    ap.add_argument("--nk", type=int, default=20)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--slurm-array-job-id")
    ap.add_argument("--audit-script-sha256")
    ap.add_argument("--omit-time-series", action="store_true")
    args = ap.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    script_sha = args.audit_script_sha256
    if script_sha is None:
        here = Path(__file__) if "__file__" in globals() else None
        if here is not None and here.is_file():
            script_sha = sha256_file(here)
    report = audit_campaign(
        freeze,
        args.k20_root,
        nk=args.nk,
        freeze_path=args.freeze,
        audit_script_sha256=script_sha,
        slurm_array_job_id=args.slurm_array_job_id,
        omit_time_series=args.omit_time_series,
    )
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

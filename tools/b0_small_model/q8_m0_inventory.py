#!/usr/bin/env python3
"""Stock CJJ / classical dump inventory. No new propagation.

Labels unresolved magnetic channels as not-signed training targets.
nk is not N_cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from q8_m0_cjj import dump

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/q_line"
OUT = ROOT / "response_prep_20260921"

CJJ_CAMPAIGNS = (
    "cjj_m0_p1_a2_s1_20260917",
    "cjj_m0_p1_validation_20260916",
    "cjj_m0_p1_qualification_v2_20260917",
)


def _meta_next_to(npz: Path) -> dict | None:
    meta = npz.with_suffix("").name
    # foo.npz -> foo.meta.json lives beside it as foo.meta.json
    p = npz.parent / (npz.name.replace(".npz", ".meta.json"))
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def classify_npz(path: Path) -> dict:
    z = np.load(path, mmap_mode="r", allow_pickle=True)
    files = set(z.files)
    rec = {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "keys": sorted(files),
        "has_X_full": "X_full" in files,
        "has_Y": "Y" in files,
        "has_snapshot_X": "snapshot_X" in files,
        "has_mu_per_cell": "mu_per_cell" in files,
        "n_snapshots": int(np.asarray(z["snapshot_index"]).size) if "snapshot_index" in files else 0,
        "nk_points": int(np.asarray(z["n_k"]).reshape(-1)[0]) if "n_k" in files else (
            int(np.asarray(z["kf"]).shape[0]) if "kf" in files else None
        ),
        "nt": int(np.asarray(z["times"]).size) if "times" in files else None,
        "nt_per_cycle": int(np.asarray(z["nt_per_cycle"])) if "nt_per_cycle" in files else None,
        "wall_seconds": float(np.asarray(z["wall_seconds"])) if "wall_seconds" in files else None,
        "cjj_definition": str(z["cjj_definition"]) if "cjj_definition" in files else None,
    }
    if rec["has_X_full"]:
        rec["time_content"] = "full_X_plus_Y_plus_snapshots"
        rec["causal_R_from_disk"] = "prefix_and_two_time_possible_on_this_k_set"
    elif rec["has_Y"] and rec["has_snapshot_X"]:
        rec["time_content"] = "Y_harmonics_plus_sparse_snapshots"
        rec["causal_R_from_disk"] = "not_available_do_not_interpolate_Y"
    elif rec["has_Y"]:
        rec["time_content"] = "Y_only"
        rec["causal_R_from_disk"] = "not_available_do_not_interpolate_Y"
    else:
        rec["time_content"] = "not_cjj_Y_schema"
        rec["causal_R_from_disk"] = "not_cjj"
    rec["nk_is_not_n_cell"] = True
    meta = _meta_next_to(path)
    if meta:
        rec["protocol_id"] = meta.get("protocol_id")
        rec["neel"] = meta.get("neel")
        rec["grid_shift"] = meta.get("grid_shift")
        rec["polarization"] = (meta.get("protocol") or {}).get("polarization")
        rec["label"] = (meta.get("protocol") or {}).get("label")
        rec["quantum_input"] = meta.get("quantum_input", "not_run")
        rec["photon_output"] = meta.get("photon_output", "not_run")
    return rec


def campaign_rows(name: str) -> list[dict]:
    folder = ROOT / name
    rows = []
    if not folder.is_dir():
        return rows
    for npz in sorted(folder.rglob("*.npz")):
        try:
            rows.append(classify_npz(npz))
        except Exception as exc:  # noqa: BLE001 — inventory must not die on one odd dump
            rows.append({"path": str(npz.relative_to(REPO)).replace("\\", "/"), "error": str(exc)})
    return rows


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    campaigns = {name: campaign_rows(name) for name in CJJ_CAMPAIGNS}
    a2_qual = load_json(ROOT / "cjj_m0_p1_a2_s1_20260917/qualification_physics3_execution2.json")
    v2_cost = load_json(ROOT / "cjj_m0_p1_qualification_v2_20260917/cost_estimate.json")
    v1_cost = load_json(ROOT / "cjj_m0_p1_validation_20260916/cost_estimate.json")
    a2_sum = load_json(ROOT / "cjj_m0_p1_a2_s1_20260917/summary.json")
    n_full_x = sum(1 for rows in campaigns.values() for r in rows if r.get("has_X_full"))
    n_y_only = sum(
        1 for rows in campaigns.values() for r in rows
        if r.get("has_Y") and not r.get("has_X_full")
    )
    out = {
        "kind": "stock_inventory_not_training_set",
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "nk_is_not_n_cell": True,
        "do_not_interpolate_Y_to_causal_R": True,
        "unresolved_must_not_be_signed_AI_labels": True,
        "campaigns": campaigns,
        "counts": {
            "npz_with_full_X": n_full_x,
            "npz_with_Y_without_full_X": n_y_only,
            "distinct_frozen_pump_points_used_for_CJJ": [
                "P1_x_A0.8_8cycle (validation + v2 96; laboratory x)",
                "P1_a2_A0.8_8cycle (sixk + V48 + V48S only; V96 skipped)",
            ],
            "not_a_7D_parameter_scan": True,
        },
        "qualification_a2": a2_qual,
        "cost_validation_20260916": v1_cost,
        "cost_qualification_v2_20260917": v2_cost,
        "a2_campaign_job_walls": {
            "B0_sixk_pair_s": (a2_sum or {}).get("jobs", {}).get("B0_sixk", {}).get("wall_seconds_pair"),
            "V48_pair_s": (a2_sum or {}).get("jobs", {}).get("V48", {}).get("wall_seconds_pair"),
            "V48S_pair_s": (a2_sum or {}).get("jobs", {}).get("V48S", {}).get("wall_seconds_pair"),
        },
        "ai_label_policy": {
            "d_parperp_odd": "unresolved_magnetic_candidate_not_a_signed_label",
            "Jo_perp_H2": "unresolved_not_a_signed_label",
            "d_perpperp_H2_even": "algorithm_baseline_not_a_magnetic_discovery",
            "approximate_mirror": "do_not_hardcode_exact_zero",
            "P_or_T_augmentation": "not_allowed_without_protocol_map",
        },
    }
    dump(OUT / "stock_inventory.json", out)
    print(json.dumps({
        "wrote": str(OUT / "stock_inventory.json"),
        "full_X": n_full_x,
        "Y_without_full_X": n_y_only,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

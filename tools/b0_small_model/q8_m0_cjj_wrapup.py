#!/usr/bin/env python3
"""Post-process existing M0 CJJ NPZ: floors, signed undriven diffs, H1–H9 inventory.

Does not propagate the eight trajectories. Does not run quantum pump.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import (  # noqa: E402
    CJJ_DEFINITION,
    CJJ_MODULE,
    DEFINITION_REVIEW,
    ORDERS,
    cjj_fingerprint,
    d_from_y,
    domain_combine,
    dump,
    undriven_y,
)
from q8_m0_cjj_validate import (  # noqa: E402
    CJJ_ABS_FLOOR_RATIONALE,
    CJJ_ABS_REPORT_RESOLUTION,
    CJJ_ABS_UNIT,
    CONFIGS,
    CROSS_WINDOWS,
    OUT,
    _load,
    cache_reuse_decision,
    check_sixk,
    classify_d,
    d_catalog,
    mu_from_static,
    signed_kernel_diff,
)
from q8_m0_classical import p1_protocol  # noqa: E402
from q8_m0_model import R0, build_geometry, sha256_file  # noqa: E402

SPOTLIGHT = (
    "d_H2x,H2x",
    "d_H2y,H2y",
    "d_H3x,H3x",
    "d_H3y,H3y",
    "d_H4x,H4x",
    "d_H5y,H5y",
    "d_H6y,H6y",
    "d_H7x,H7x",
    "d_H2x,H3x",
    "d_H3x,H4x",
    "d_H2x,H6y",
)


def load_pair(name: str) -> dict:
    plus = _load(OUT / name, f"{name}_N+1_A+1")
    minus = _load(OUT / name, f"{name}_N-1_A+1")
    if plus is None or minus is None:
        raise FileNotFoundError(name)
    comb = domain_combine(np.asarray(plus["d_namb"]), np.asarray(minus["d_namb"]))
    return {"+1": plus, "-1": minus, **comb}


def y_rebuild_residual(r: dict) -> float:
    w = np.asarray(r["w_k"], dtype=float)
    rec = d_from_y(np.asarray(r["Y"]), w)
    return float(np.max(np.abs(rec - np.asarray(r["d_namb"]))))


def audit_one(geo, prot, mu, name: str, nk: int, ntpc: int, shift: float, payload: dict, neel: int, expected_kf=None) -> dict:
    tag = f"{name}_N{neel:+d}_A+1"
    stored = json.loads((OUT / name / f"{tag}.meta.json").read_text(encoding="utf-8"))
    kf_kind = "custom" if expected_kf is not None else f"mp_nk{nk}_shift{shift}"
    ident = cjj_fingerprint(
        geo, R0, prot, nk=(len(expected_kf) if expected_kf is not None else nk),
        nt_per_cycle=ntpc, mu=mu, grid_shift=0.0 if expected_kf is not None else shift,
        neel=neel, sign=+1, kf_kind=kf_kind,
    )
    decision = cache_reuse_decision(stored, ident, payload, name=name, expected_kf=expected_kf)
    return {
        "tag": tag,
        "stored_module": stored.get("cjj_module_sha256"),
        "current_module": ident.get("cjj_module_sha256"),
        "stored_identity": stored.get("cjj_identity_sha256"),
        "current_identity": ident.get("cjj_identity_sha256"),
        **decision,
        "n_k": int(np.asarray(payload["n_k"])) if "n_k" in payload else int(np.asarray(payload["kf"]).shape[0]),
        "times": int(np.asarray(payload["times"]).size),
        "y_rebuild_maxabs": y_rebuild_residual(payload),
    }


def classify_catalog(jobs_cat: dict) -> dict:
    return classify_d(jobs_cat, abs_floor=CJJ_ABS_REPORT_RESOLUTION)


def status_index(conv: dict) -> dict:
    out = {}
    for side, table in conv.items():
        buckets = {"qualified_preliminary": [], "trend_not_qualified": [], "unresolved": [], "below_reporting_resolution": []}
        for key, row in table.items():
            buckets.setdefault(row["status"], []).append(key)
        out[side] = {k: sorted(set(v)) for k, v in buckets.items()}
    return out


def compare_old_labels(new_conv: dict) -> dict:
    old_path = OUT / "cjj_validation_summary.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))["source_observable_convergence"]
    changed = []
    unchanged = []
    for side in old:
        if side not in new_conv:
            continue
        for key, row in old[side].items():
            if key not in new_conv[side]:
                continue
            nxt = new_conv[side][key]["status"]
            rec = {"side": side, "key": key, "old": row["status"], "new": nxt}
            if row["status"] != nxt:
                changed.append(rec)
            else:
                unchanged.append(rec)
    return {"n_compared": len(changed) + len(unchanged), "changed": changed, "n_unchanged": len(unchanged)}


def spotlight_table(conv: dict) -> dict:
    out = {}
    for key in SPOTLIGHT:
        out[key] = {}
        for side in ("d_plus", "d_minus", "d_dom_e", "d_dom_o"):
            if key in conv[side]:
                row = conv[side][key]
                z = row["z_V48"]
                out[key][side] = {
                    "re": z["re"],
                    "im": z["im"],
                    "abs": z["abs"],
                    "rel_V24": row["rel_V24"],
                    "rel_shift": row["rel_shift"],
                    "rel_dt": row["rel_dt"],
                    "status": row["status"],
                }
    return out


def main() -> int:
    geo = build_geometry()
    prot = p1_protocol(0.8)
    mu = mu_from_static()
    current_mod = sha256_file(CJJ_MODULE)
    probe = json.loads((OUT / "probe_k.json").read_text(encoding="utf-8"))
    kf6 = np.asarray(probe["kf"], dtype=float)

    cache_audit = []
    b0_algebra = {}
    for neel in (+1, -1):
        r = _load(OUT / "B0_sixk", f"B0_sixk_N{neel:+d}_A+1")
        cache_audit.append(audit_one(geo, prot, mu, "B0_sixk", 6, 400, 0.0, r, neel, expected_kf=kf6))
        b0_algebra[str(neel)] = check_sixk(r, geo, prot)

    jobs_raw = {}
    for name, nk, ntpc, shift in CONFIGS:
        pair = load_pair(name)
        jobs_raw[name] = pair
        for neel in (+1, -1):
            cache_audit.append(audit_one(geo, prot, mu, name, nk, ntpc, shift, pair[f"{neel:+d}"], neel))

    jobs_cat = {}
    signed_h2xx = {}
    signed_diag = {}
    undriven_note = "analytic undriven_y on saved kf; not a new driven trajectory"
    for name, nk, ntpc, shift in CONFIGS:
        pair = jobs_raw[name]
        und = {}
        for neel in (+1, -1):
            kf = np.asarray(pair[f"{neel:+d}"]["kf"])
            und[neel] = undriven_y(geo, R0, prot, kf, mu, neel)
        jobs_cat[name] = {
            "d_plus": d_catalog(pair["+1"]["d_namb"]),
            "d_minus": d_catalog(pair["-1"]["d_namb"]),
            "d_dom_e": d_catalog(pair["d_dom_e"]),
            "d_dom_o": d_catalog(pair["d_dom_o"]),
        }
        signed_h2xx[name] = {
            "+1": signed_kernel_diff(pair["+1"]["d_namb"][1, 0, 1, 0], und[+1]["d_namb"][1, 0, 1, 0]),
            "-1": signed_kernel_diff(pair["-1"]["d_namb"][1, 0, 1, 0], und[-1]["d_namb"][1, 0, 1, 0]),
        }
        signed_diag[name] = {}
        for n in ORDERS:
            for a, ax in ((0, "x"), (1, "y")):
                key = f"d_H{n}{ax},H{n}{ax}"
                signed_diag[name][key] = {
                    "+1": signed_kernel_diff(pair["+1"]["d_namb"][n - 1, a, n - 1, a], und[+1]["d_namb"][n - 1, a, n - 1, a]),
                    "-1": signed_kernel_diff(pair["-1"]["d_namb"][n - 1, a, n - 1, a], und[-1]["d_namb"][n - 1, a, n - 1, a]),
                }

    conv = classify_catalog(jobs_cat)
    labels = compare_old_labels(conv)
    idx = status_index(conv)
    cross_keys = [f"d_{a},{b}" for _, _, a in CROSS_WINDOWS for _, _, b in CROSS_WINDOWS]
    cross_status = {key: {side: conv[side][key]["status"] for side in conv if key in conv[side]} for key in cross_keys}

    header_path = OUT / "campaign_header.json"
    launch_path = OUT / "campaign_launch.json"
    if header_path.exists() and not launch_path.exists():
        launch_path.write_text(header_path.read_text(encoding="utf-8"), encoding="utf-8")
    status = {
        "role": "canonical_status",
        "m0_cjj_status": "implementation_verified_small_bz_complete",
        "wrapup": "cjj_wrapup_inventory.json",
        "cjj_definition": CJJ_DEFINITION,
        "definition_review": DEFINITION_REVIEW,
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
        "did_not_rerun_eight_trajectories": True,
    }
    dump(OUT / "campaign_status.json", status)
    hdr = json.loads(header_path.read_text(encoding="utf-8")) if header_path.exists() else {}
    hdr.update({
        "role": "launch_record",
        "m0_cjj_status_at_launch": hdr.get("m0_cjj_status_at_launch", "running"),
        "m0_cjj_status": "implementation_verified_small_bz_complete",
        "canonical_status_file": "campaign_status.json",
        "note": "launch record; running flag is historical. Canonical status is campaign_status.json.",
    })
    dump(header_path, hdr)

    rec = {
        "postprocess": "existing_npz_no_new_propagation",
        "cjj_definition": CJJ_DEFINITION,
        "definition_review": DEFINITION_REVIEW,
        "current_cjj_module_sha256": current_mod,
        "abs_floor": CJJ_ABS_REPORT_RESOLUTION,
        "abs_unit": CJJ_ABS_UNIT,
        "abs_floor_rationale": CJJ_ABS_FLOOR_RATIONALE,
        "array_fix_note": (
            "first auto run died at V24: snapshot_X allocated with linear nk=24 "
            "instead of n_k=576; propagate_cjj now uses kc.shape[0]; eight BZ "
            "NPZ were written after that fix and were not rerun in this wrapup"
        ),
        "b0_algebra_recheck": {
            str(k): {
                "pass": v["pass"],
                "q4_minus_q8_max": max(max(row["q4_minus_q8"].values()) for row in v["pairs"]),
                "herm_Cxy_vs_Cyx_max": max(row["herm_Cxy_vs_Cyx"] for row in v["pairs"]),
            }
            for k, v in b0_algebra.items()
        },
        "cache_audit": cache_audit,
        "y_rebuild_maxabs": {row["tag"]: row["y_rebuild_maxabs"] for row in cache_audit},
        "label_change_vs_campaign_summary": labels,
        "status_index": idx,
        "spotlight": spotlight_table(conv),
        "six_window_cross_status": cross_status,
        "pumped_minus_undriven_H2xx_signed": signed_h2xx,
        "pumped_minus_undriven_diagonals_signed": signed_diag,
        "undriven_note": undriven_note,
        "source_observable_convergence": conv,
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
        "interpretation_guards": {
            "domain_odd": "not_a_resolved_magnetic_contrast",
            "pairing_residual": "do_not_translate_0.13_0.18_as_eighty_percent_paired",
            "scheme_B": "empirical_two-grid_spread_not_a_guaranteed_integral_error_bar",
            "cross_source": "not_optical_entanglement",
        },
    }
    dump(OUT / "cjj_wrapup_inventory.json", rec)
    slim = {
        "label_change_vs_campaign_summary": labels,
        "status_index": {side: {st: keys for st, keys in buckets.items() if keys} for side, buckets in idx.items()},
        "spotlight": rec["spotlight"],
        "pumped_minus_undriven_H2xx_signed": signed_h2xx,
        "b0_algebra_recheck": rec["b0_algebra_recheck"],
        "cache_reuse_all": all(row["reuse"] for row in cache_audit),
        "quantum_input": "not_run",
    }
    dump(OUT / "cjj_wrapup_slim.json", slim)
    print(json.dumps({
        "wrote": str(OUT / "cjj_wrapup_inventory.json"),
        "label_changes": labels["changed"],
        "n_unchanged_old_keys": labels["n_unchanged"],
        "cache_reuse_all": slim["cache_reuse_all"],
        "b0_pass": {k: v["pass"] for k, v in rec["b0_algebra_recheck"].items()},
        "H2xx_plus_status": conv["d_plus"]["d_H2x,H2x"]["status"],
        "H2xx_dom_o_status": conv["d_dom_o"]["d_H2x,H2x"]["status"],
        "H6yy_plus_status": conv["d_plus"]["d_H6y,H6y"]["status"],
        "H5yy_plus_status": conv["d_plus"]["d_H5y,H5y"]["status"],
        "H7xx_plus_status": conv["d_plus"]["d_H7x,H7x"]["status"],
        "H2xx_dom_o_V48_re": conv["d_dom_o"]["d_H2x,H2x"]["z_V48"]["re"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

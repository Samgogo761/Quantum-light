#!/usr/bin/env python3
"""Generate, validate, and freeze GH7 ES25.17 tail-gate manifests. Does not submit jobs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "analysis"))

from gh7_nodes import (  # noqa: E402
    I_BAR,
    SQUEEZE_R,
    make_sv_nodes_gh7,
    select_wide_axis_antinode_pair,
    sha256_file,
    write_manifest,
)
from validate_qlight_nodes_production import load_manifest  # noqa: E402
from validate_qlight_nodes_strict import validate_sv  # noqa: E402

OUTDIR = REPO / "deploy" / "a0_layerA_m88full" / "nodes_tbv2_es25" / "gh7"
THETAS = (0.0, 180.0)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifests = {}
    probes = []
    for th in THETAS:
        rows = make_sv_nodes_gh7(SQUEEZE_R, th, I_BAR)
        name = f"nodes_sv_r2p5_th{int(th)}_gh7.dat"
        path = OUTDIR / name
        write_manifest(path, rows, SQUEEZE_R, th, I_BAR)
        q0 = validate_sv(load_manifest(path), SQUEEZE_R, th, I_BAR, 2.0e-6)
        if q0["status"] != "PASS":
            raise SystemExit(f"Q0 FAIL for {name}")
        if abs(float(q0["weight_sum"]) - 1.0) > 1.0e-14:
            raise SystemExit(f"weight sum not 1: {q0['weight_sum']}")
        if int(q0["n_nodes"]) != 49:
            raise SystemExit(f"{name}: expected 49 nodes")
        pair = select_wide_axis_antinode_pair(rows, SQUEEZE_R, th)
        sha = sha256_file(path)
        manifests[name] = {
            "path": str(path.relative_to(REPO).as_posix()),
            "sha256": sha,
            "q0": q0,
            "wide_axis_pair": pair,
        }
        for nid in pair["ids"]:
            probes.append(
                {
                    "theta_deg": int(th),
                    "id": nid,
                    "I_max": pair["I_max"],
                    "manifest": name,
                    "manifest_sha256": sha,
                }
            )
        print(f"WROTE {path} sha={sha} I_max={pair['I_max']:.6e} ids={pair['ids']}")

    if len(probes) != 4:
        raise SystemExit(f"expected 4 +N probes, got {len(probes)}")
    freeze = {
        "campaign": "full112_gh7_tail_k20",
        "do_not_submit": True,
        "do_not_blind_submit_full_gh7": True,
        "policy": "Pinned 29175 binary; TB-v2; no rebuild. Staged k20 only until occupation gate PASS.",
        "gh_order": 7,
        "n_nodes": 49,
        "precision": "ES25.17",
        "I_bar": I_BAR,
        "squeeze_r": SQUEEZE_R,
        "thetas_deg": [0, 180],
        "pinned_binary_sha256": "34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a",
        "pinned_binary_source_sha256": "c6372e17457cb45d38233d81149b43fc07768978e5870c3e040d13205000110f",
        "tb_plus_sha256": "66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18",
        "tb_minus_v2_sha256": "d61034e18b551dbb33b064c0c311d4ba0ef13be1c8f4a11a88ddbe1e95b66246",
        "manifests": manifests,
        "plusN_probes": probes,
        "stages": ["k20_tail", "occupation_post", "approve_k40", "then_decide_full_gh7"],
        "renormalize_subset_weights": False,
        "use_full_manifest_moment_check": True,
        "propagate_ids_only": True,
        "occ_stride": 336,
        "n_occupation_snapshots_approx": 16,
    }
    dest = HERE / "FREEZE.json"
    dest.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print("WROTE", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

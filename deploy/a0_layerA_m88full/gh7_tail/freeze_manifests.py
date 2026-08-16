#!/usr/bin/env python3
"""Generate, validate, and freeze GH7 ES25.17 tail-gate manifests. Does not submit jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    select_corner_antinode_pair,
    select_wide_axis_antinode_pair,
    sha256_file,
    sha256_text_file,
    source_tree_digest,
    write_manifest,
)
from validate_qlight_nodes_production import load_manifest  # noqa: E402
from validate_qlight_nodes_strict import validate_sv  # noqa: E402

OUTDIR = REPO / "deploy" / "a0_layerA_m88full" / "nodes_tbv2_es25" / "gh7"
THETAS = (0.0, 180.0)
PIN_PLACEHOLDER = "TO_BE_PINNED"
SCRIPT_RELS = {
    "template_sha256": "deploy/a0_layerA_m88full/input_template_full112_gh7_tail.nml",
    "sbatch_sha256": "deploy/a0_layerA_m88full/sbatch_full112_gh7_tail_k20.sh",
    "submit_sha256": "deploy/a0_layerA_m88full/submit_gh7_tail_k20.sh",
    "validator_run_sha256": "tools/analysis/validate_a0_run_strict.py",
    "validator_node_sha256": "tools/analysis/validate_qlight_nodes_production.py",
    "validator_node_core_sha256": "tools/analysis/validate_qlight_nodes_strict.py",
    "gate_sha256": "tools/analysis/gh7_tail_model_gate.py",
}


def _script_pins() -> dict[str, str]:
    pins = {key: sha256_text_file(REPO / rel) for key, rel in SCRIPT_RELS.items()}
    pins["worktree_source_sha256"] = source_tree_digest(REPO, src_from_git_head=True)
    return pins


def _write_freeze(payload: dict) -> None:
    dest = HERE / "FREEZE.json"
    text = json.dumps(payload, indent=2) + "\n"
    dest.write_bytes(text.encode("utf-8"))
    sha_path = HERE / "FREEZE.sha256"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sha_path.write_bytes((digest + "\n").encode("ascii"))
    print("WROTE", dest)
    print("WROTE", sha_path, digest)


def pin_head() -> int:
    dest = HERE / "FREEZE.json"
    payload = json.loads(dest.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    payload["git_head"] = head
    _write_freeze(payload)
    print("PINNED git_head", head)
    return 0


def build_freeze(git_head: str) -> dict:
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
        corner = select_corner_antinode_pair(rows, SQUEEZE_R, th)
        wide = select_wide_axis_antinode_pair(rows, SQUEEZE_R, th)
        if abs(float(corner["axis_dot"])) > 1.0e-12 or abs(float(wide["axis_dot"])) > 1.0e-12:
            raise SystemExit(f"{name}: major/minor axes are not orthogonal")
        sha = sha256_file(path)
        manifests[name] = {
            "path": str(path.relative_to(REPO).as_posix()),
            "sha256": sha,
            "q0": q0,
            "corner_pair": corner,
            "wide_axis_pair": wide,
        }
        for pair in (corner, wide):
            for nid in pair["ids"]:
                probes.append(
                    {
                        "theta_deg": int(th),
                        "id": nid,
                        "kind": pair["kind"],
                        "I_max": pair["I_max"],
                        "I_over_Imax": pair["I_over_Imax"],
                        "manifest": name,
                        "manifest_sha256": sha,
                    }
                )
        print(
            f"WROTE {path} sha={sha} corner={corner['ids']} wide={wide['ids']} "
            f"I_max={corner['I_max']:.6e}"
        )

    if len(probes) != 8:
        raise SystemExit(f"expected 8 +N probes, got {len(probes)}")
    freeze = {
        "campaign": "full112_gh7_tail_k20",
        "n_probes": 8,
        "do_not_submit": True,
        "do_not_blind_submit_full_gh7": True,
        "cep_pi_tested": False,
        "cep_pi_note": "+N-only probes do not test GH7 CEP-π covariance",
        "dt2_stage": "not_implemented",
        "policy": (
            "Pinned 29175 binary; TB-v2; no rebuild. "
            "Eight k20 probes (corner + pure wide-axis). Staged until occupation gate PASS."
        ),
        "gh_order": 7,
        "n_nodes": 49,
        "precision": "ES25.17",
        "I_bar": I_BAR,
        "squeeze_r": SQUEEZE_R,
        "thetas_deg": [0, 180],
        "git_head": git_head,
        "pinned_binary_sha256": "34edab1dbc6f7033b73e4feed85d96c1f36e71b781dd810ba6ff9391db82a67a",
        "pinned_binary_source_sha256": "c6372e17457cb45d38233d81149b43fc07768978e5870c3e040d13205000110f",
        "tb_plus_sha256": "66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18",
        "tb_minus_v2_sha256": "d61034e18b551dbb33b064c0c311d4ba0ef13be1c8f4a11a88ddbe1e95b66246",
        **_script_pins(),
        "manifests": manifests,
        "plusN_probes": probes,
        "stages": ["k20_tail", "occupation_post", "approve_k40", "then_decide_full_gh7"],
        "renormalize_subset_weights": False,
        "use_full_manifest_moment_check": True,
        "propagate_ids_only": True,
        "occ_stride": 336,
        "n_occupation_snapshots_approx": 16,
        "nk_valence": 84,
        "nk_bands": 112,
    }
    return freeze


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pin-head",
        action="store_true",
        help="Rewrite FREEZE.git_head to the current Git HEAD after the checkpoint commit.",
    )
    args = ap.parse_args()
    if args.pin_head:
        return pin_head()
    freeze = build_freeze(PIN_PLACEHOLDER)
    _write_freeze(freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebuild the single authoritative E0 acceptance ledger for job 27674.

This overwrites stale FAIL summaries. It does not resubmit or recompute SBE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import combine_pmN_nodes_strict as strict
from validate_a0_run_strict import validate as validate_run


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def th_nodes(thpad: str) -> str:
    return {"000": "0", "090": "90", "180": "180", "270": "270"}[thpad]


def summary_metrics(summary_path: Path) -> dict[str, float]:
    data = np.loadtxt(summary_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    by = {int(r[0]): r for r in data}
    out: dict[str, float] = {}
    for order in (2, 5, 7, 9, 10):
        row = by[order]
        jo = complex(row[15], row[16]), complex(row[17], row[18])
        # |Jo_bar|^2 == trajectory_odd_CS (coherent sum of odd channel)
        out[f"trajectory_odd_CS_H{order}"] = abs(jo[0]) ** 2 + abs(jo[1]) ** 2
        out[f"trajectory_odd_ICS_H{order}"] = float(row[28])
        out[f"domain_plus_ICS_H{order}"] = float(row[19])
        out[f"domain_plus_CS_H{order}"] = float(row[20])
        out[f"domain_plus_classical_trajectory_variance_H{order}"] = float(row[21])
        out[f"Jo_bar_x_re_H{order}"] = float(row[15])
        out[f"Jo_bar_x_im_H{order}"] = float(row[16])
        out[f"Jo_bar_y_re_H{order}"] = float(row[17])
        out[f"Jo_bar_y_im_H{order}"] = float(row[18])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".tmp" / "e0_27674" / "output",
    )
    ap.add_argument(
        "--nodes-dir",
        type=Path,
        default=ROOT / ".tmp" / "e0_27674" / "nodes",
    )
    ap.add_argument(
        "--post-dir",
        type=Path,
        default=ROOT / ".tmp" / "e0_27674" / "post",
    )
    ap.add_argument("--harmonics", default="2,5,7,9,10")
    args = ap.parse_args()

    harmonics = [int(x) for x in args.harmonics.split(",") if x.strip()]
    post = args.post_dir
    post.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name for p in args.output_root.iterdir() if p.is_dir())
    run_results = []
    fail = 0
    for case in cases:
        run_dir = args.output_root / case
        man = next(run_dir.glob("nodes_sv_r*_gh3.dat"))
        try:
            result = validate_run(run_dir, man, harmonics, rtol=2.0e-7)
            status = "PASS"
        except (OSError, ValueError) as exc:
            status = "FAIL"
            result = {"error": str(exc)}
            fail += 1
        run_results.append(
            {
                "case": case,
                "status": status,
                "manifest": str(man),
                "manifest_sha256": sha256_file(man),
                "modes_sha256": sha256_file(run_dir / "HHG_nodes_modes.dat"),
                "ics_cs_sha256": sha256_file(run_dir / "HHG_ics_cs.dat"),
                "detail": result,
            }
        )

    pair_results = []
    for rtag in ("0p5", "1p5", "2p5"):
        for thpad in ("000", "090", "180", "270"):
            tag = f"sv_r{rtag}_th{thpad}"
            plus_dir = args.output_root / f"{tag}_plusN"
            minus_dir = args.output_root / f"{tag}_minusN"
            plus = plus_dir / "HHG_nodes_modes.dat"
            minus = minus_dir / "HHG_nodes_modes.dat"
            # Prefer the archived copies inside each ±N run directory so the
            # SHA gate compares the real pair, not a file with itself.
            plus_nodes = next(plus_dir.glob("nodes_sv_r*_gh3.dat"), None)
            minus_nodes = next(minus_dir.glob("nodes_sv_r*_gh3.dat"), None)
            shared = args.nodes_dir / f"nodes_sv_r{rtag}_th{th_nodes(thpad)}_gh3.dat"
            if plus_nodes is None:
                plus_nodes = shared
            if minus_nodes is None:
                minus_nodes = shared
            outj = post / f"{tag}_JeJo.dat"
            outs = post / f"{tag}_summary.dat"
            try:
                nodes_sha = strict.require_identical_node_manifests(
                    plus_nodes, minus_nodes
                )
                manifest = strict.load_node_manifest(plus_nodes)
                plus_arr = strict.load_modes(plus)
                minus_arr = strict.load_modes(minus)
                strict.crosscheck_modes_against_manifest(
                    plus_arr, plus, manifest,
                    rtol=1.0e-12, atol=1.0e-12, phase_atol=1.0e-6,
                )
                strict.crosscheck_modes_against_manifest(
                    minus_arr, minus, manifest,
                    rtol=1.0e-12, atol=1.0e-12, phase_atol=1.0e-6,
                )
                aligned = strict.align_and_validate_pair(
                    plus_arr, minus_arr, plus, minus, rtol=1.0e-12, atol=1.0e-12
                )
                jxp, jyp = strict._complex_components(plus_arr)
                jxm, jym = strict._complex_components(aligned)
                jxe, jye = 0.5 * (jxp + jxm), 0.5 * (jyp + jym)
                jxo, jyo = 0.5 * (jxp - jxm), 0.5 * (jyp - jym)
                strict.write_node_output(outj, plus_arr, jxe, jye, jxo, jyo)
                strict.write_summary(outs, plus_arr, jxp, jyp, jxm, jym, jxe, jye, jxo, jyo)
                metrics = summary_metrics(outs)
                status = "PASS"
            except (OSError, ValueError) as exc:
                status = "FAIL"
                nodes_sha = None
                metrics = {"error": str(exc)}
                fail += 1
            pair_results.append(
                {
                    "tag": tag,
                    "status": status,
                    "nodes_sha256": nodes_sha,
                    "metrics": metrics,
                }
            )

    # Authoritative text ledger
    ledger = post / "E0_ACCEPTANCE.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# E0_ACCEPTANCE — job 27674 (authoritative)",
        "",
        f"- generated_at_utc: {now}",
        "- job: 27674 (GH3@k20 explore, M88-full)",
        "- status_class: **integrity-accepted exploration data; NOT production provenance**",
        f"- per_run_pass: {sum(1 for r in run_results if r['status']=='PASS')}/{len(run_results)}",
        f"- pmN_pair_pass: {sum(1 for r in pair_results if r['status']=='PASS')}/{len(pair_results)}",
        f"- overall: {'PASS' if fail == 0 else 'FAIL'}",
        "",
        "## Metric naming (mandatory)",
        "",
        "- `trajectory_odd_CS_Hn` = `|Jo_bar_n|^2` where "
        "`Jo_bar = sum_s w_s (J(+N)-J(-N))/2`.",
        "- `trajectory_odd_ICS_Hn` = `sum_s w_s |Jo_s,n|^2`.",
        "- `domain_plus_classical_trajectory_variance_Hn` = ICS−CS on **+N only**.",
        "- None of the above is quantum g^(2), squeezing, or entanglement.",
        "- `r=0.5` is a low-r SV contrast, **not** a classical/fixed-field baseline.",
        "",
        "## Per-run validation",
        "",
    ]
    for r in run_results:
        lines.append(f"- {r['status']} `{r['case']}` modes={r['modes_sha256'][:12]}…")
    lines += ["", "## ±N production pairing", ""]
    for r in pair_results:
        m = r["metrics"]
        if r["status"] == "PASS":
            lines.append(
                f"- PASS `{r['tag']}` "
                f"oddCS_H2={m['trajectory_odd_CS_H2']:.6e} "
                f"oddCS_H9={m['trajectory_odd_CS_H9']:.6e} "
                f"plusVAR_H2={m['domain_plus_classical_trajectory_variance_H2']:.6e}"
            )
        else:
            lines.append(f"- FAIL `{r['tag']}` {m}")
    lines += [
        "",
        "## Recommended E2 candidates (do not submit until gates ready)",
        "",
        "- `(r,θ)=(2.5,0°)` main magnetic-odd",
        "- `(2.5,180°)` squeeze-angle contrast",
        "- `(0.5,180°)` low-r contrast (not classical baseline)",
        "",
        "Stale file note: any older `validate_runs.txt` listing 24 FAIL entries is "
        "**superseded** by this ledger.",
        "",
    ]
    ledger.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "generated_at_utc": now,
        "job": 27674,
        "overall": "PASS" if fail == 0 else "FAIL",
        "runs": run_results,
        "pairs": pair_results,
    }
    (post / "E0_ACCEPTANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Replace contradictory validate_runs.txt
    (post / "validate_runs.txt").write_text(
        "\n".join(f"{r['status']} {r['case']}" for r in run_results) + "\n",
        encoding="utf-8",
    )
    (post / "combine.txt").write_text(
        "\n".join(f"{r['status']} {r['tag']}" for r in pair_results) + "\n",
        encoding="utf-8",
    )

    print(f"E0_ACCEPTANCE={'PASS' if fail == 0 else 'FAIL'}")
    print(f"wrote {ledger}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Zero-cost C8↔C36 OpenMP topology audit (Jones vector).

Same binary / TB / nodes / MKL=1; only OMP 8 vs 36 (jobs 29002 vs 28861).
This is NOT a retune of chunk V3.1 (τ_abs=1e-12). Preset topology gate: 1e-4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_chunked_vs_full_v2 import load_modes
from compare_chunked_vs_full_v3 import jones_diff_norm, jones_norm
from compare_chunked_vs_full_v3_1 import evaluate_jones_bar_v31, evaluate_jones_modes_v31

TOPOLOGY_ATOL = 1.0e-4
HARD_ORDERS = (2, 5, 7, 9)
ALL_ORDERS = (2, 5, 7, 9, 10)

DEFAULT_C8 = Path(
    "/public/home/wangjs/project/New_SBEs/Quantum-light-wt-689d9a3/"
    "output_a0_full112_k20_gh3_chunk_c8_29002/merged_plusN"
)
DEFAULT_C36 = Path(
    "/public/home/wangjs/project/New_SBEs/Quantum-light/"
    "runs/a0_full112_gh3_chunk_smoke_689d9a3/"
    "output_a0_full112_k20_gh3_chunk_smoke_28861/merged_plusN"
)

PINNED = {
    "c8_job": 29002,
    "c36_job": 28861,
    "binary_sha256": "589da472e6ec60cfa6eef41dd1e0bd9678f8647815f7d0ea2636ba0d8c96bf42",
    "source_sha256": "3c9a3d308402b07041c7fb28997030ef8f7372b1234279e81f3fbc487caee3e2",
    "tb_sha256": "66382a51a976ea86e15ceb719121dd681bac8e64e7cda702c982921cd1bfda18",
    "nodes_sha256": "290dd4a3b53402b94294246e3ee1354edd7b3994050f865d021c6f66c04b6f4b",
    "c8_omp": 8,
    "c36_omp": 36,
    "mkl_num_threads": 1,
}


def audit_modes(c8_dir: Path, c36_dir: Path, *, atol: float = TOPOLOGY_ATOL) -> dict:
    modes_c8 = load_modes(c8_dir / "HHG_nodes_modes.dat")
    modes_c36 = load_modes(c36_dir / "HHG_nodes_modes.dat")
    jones = evaluate_jones_modes_v31(
        modes_c8,
        modes_c36,
        abs_atol=atol,
        rel_atol=atol,
        harmonics=ALL_ORDERS,
    )
    bar = evaluate_jones_bar_v31(
        modes_c8,
        modes_c36,
        abs_atol=atol,
        rel_atol=atol,
        harmonics=ALL_ORDERS,
    )
    hard_abs = 0.0
    hard_rel = 0.0
    by_order: dict[str, dict] = {}
    for order in ALL_ORDERS:
        A = 0.0
        for (nid, h), row in modes_c36.items():
            if h == order:
                A = max(A, jones_norm(row["jx"], row["jy"]))
        A = A if A > 0 else 1.0e-300
        max_e_abs = 0.0
        max_e_rel = 0.0
        n_strong = 0
        for (nid, h), r36 in modes_c36.items():
            if h != order:
                continue
            r8 = modes_c8[(nid, h)]
            d = jones_diff_norm(r8["jx"], r8["jy"], r36["jx"], r36["jy"])
            nref = jones_norm(r36["jx"], r36["jy"])
            max_e_abs = max(max_e_abs, d / A)
            if nref >= 1.0e-10 * A:
                max_e_rel = max(max_e_rel, d / nref)
                n_strong += 1
        by_order[str(order)] = {
            "A_n": A,
            "max_e_abs_vec": max_e_abs,
            "max_e_rel_vec_strong": max_e_rel,
            "n_strong": n_strong,
            "hard": order in HARD_ORDERS,
        }
        if order in HARD_ORDERS:
            hard_abs = max(hard_abs, max_e_abs)
            hard_rel = max(hard_rel, max_e_rel)

    passed = (hard_abs <= atol) and (hard_rel <= atol)
    return {
        "gate": "C8_C36_TOPOLOGY_JONES",
        "status": "PASS" if passed else "FAIL",
        "abs_atol": atol,
        "rel_atol": atol,
        "note": (
            "Preset topology gate 1e-4; not a V3.1 retune. "
            "Same binary/TB/nodes/MKL=1; OMP 8 vs 36 only."
        ),
        "pinned_provenance": PINNED,
        "c8_dir": str(c8_dir),
        "c36_dir": str(c36_dir),
        "max_e_abs_vec_hard": hard_abs,
        "max_e_rel_vec_strong_hard": hard_rel,
        "by_order": by_order,
        "jones_v31_status_at_topology_atol": jones.get("status"),
        "barJ_v31_status_at_topology_atol": bar.get("status"),
        "barJ_max_e_abs_vec": bar.get("max_e_abs_vec"),
        "pass": passed,
        "recommendation": (
            "GH5-v2 prefer part_1/OMP=36" if passed else "fall back to part_2/OMP=8"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c8", type=Path, default=DEFAULT_C8)
    ap.add_argument("--c36", type=Path, default=DEFAULT_C36)
    ap.add_argument("--atol", type=float, default=TOPOLOGY_ATOL)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    report = audit_modes(args.c8, args.c36, atol=args.atol)
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

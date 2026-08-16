#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "a0_layerA_m88full" / "gh7_tail"))

from gh7_nodes import EXPECTED_I_MAX, make_sv_nodes_gh7, select_wide_axis_antinode_pair
from gh7_tail_model_gate import parse_band_occupation


def test_wide_axis_i_max() -> None:
    for th in (0.0, 180.0):
        rows = make_sv_nodes_gh7(2.5, th, 1.0e11)
        assert len(rows) == 49
        assert abs(sum(r[1] for r in rows) - 1.0) < 1e-14
        pair = select_wide_axis_antinode_pair(rows, 2.5, th)
        assert len(pair["ids"]) == 2
        assert abs(pair["I_max"] - EXPECTED_I_MAX) / EXPECTED_I_MAX < 5e-3
        # antipodes: alphas sum to ~0
        a0, a1 = pair["alphas"]
        assert abs(a0[0] + a1[0]) < 1e-12
        assert abs(a0[1] + a1[1]) < 1e-12


def test_occupation_pass_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "good.dat"
        # two snapshots, 2 k-points, bands 1..112; valence=1, cond=0, edge=0
        lines = ["# it time ikx iky band occ\n"]
        for it, t in ((1, 0.0), (2, 1.0)):
            for ik in ((1, 1), (1, 2)):
                for band in range(1, 113):
                    occ = 1.0 if band <= 84 else 0.0
                    lines.append(f"{it} {t} {ik[0]} {ik[1]} {band} {occ}\n")
        good.write_text("".join(lines), encoding="utf-8")
        rec = parse_band_occupation(good)
        assert rec["pass_trace"] and rec["pass_edge_abs"] and rec["pass_edge_rel"]

        bad = Path(td) / "bad.dat"
        lines = ["# it time ikx iky band occ\n"]
        for band in range(1, 113):
            occ = 0.5 if band >= 105 else (1.0 if band <= 84 else 0.02)
            lines.append(f"1 0.0 1 1 {band} {occ}\n")
        bad.write_text("".join(lines), encoding="utf-8")
        rec2 = parse_band_occupation(bad)
        assert rec2["pass_edge_rel"] is False or rec2["pass_edge_abs"] is False


if __name__ == "__main__":
    test_wide_axis_i_max()
    test_occupation_pass_fail()
    print("test_gh7_tail_model_gate: PASS")

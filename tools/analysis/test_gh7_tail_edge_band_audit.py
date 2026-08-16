#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from gh7_tail_edge_band_audit import audit_occupation, jones_weighted, probe_weight


def _occ_line(it: int, t: float, ikx: int, iky: int, band: int, occ: float) -> str:
    return f"{it:7d} {t:12.6E} {ikx:4d} {iky:4d} {band:5d} {occ:16.8E}\n"


def test_ladder_and_overshoot() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "occupation_band_kt.dat"
        lines = ["# band-resolved occupation\n"]
        # snapshot 1: only band 105 occupied in the edge window
        for ikx, iky in ((1, 1), (1, 2)):
            lines.append(_occ_line(1, 0.0, ikx, iky, 85, 0.1))
            lines.append(_occ_line(1, 0.0, ikx, iky, 105, 0.04))
            for b in range(106, 113):
                lines.append(_occ_line(1, 0.0, ikx, iky, b, 1.0e-8))
        # snapshot 2: population still on 105-107, 111/112 remain occupied
        lines.append(_occ_line(2, 1.0, 1, 1, 85, 0.2))
        lines.append(_occ_line(2, 1.0, 1, 1, 105, 0.05))
        lines.append(_occ_line(2, 1.0, 1, 1, 106, 0.03))
        lines.append(_occ_line(2, 1.0, 1, 1, 107, 0.02))
        for b in range(108, 111):
            lines.append(_occ_line(2, 1.0, 1, 1, b, 0.005))
        lines.append(_occ_line(2, 1.0, 1, 1, 111, 0.004))
        lines.append(_occ_line(2, 1.0, 1, 1, 112, 0.003))
        lines.append(_occ_line(2, 1.0, 1, 2, 85, 0.0))
        for b in range(105, 113):
            lines.append(_occ_line(2, 1.0, 1, 2, b, 0.0))
        # overshoot / undershoot
        lines.append(_occ_line(2, 1.0, 1, 1, 3, 1.00003))
        lines.append(_occ_line(2, 1.0, 1, 1, 4, -1.5e-5))
        path.write_text("".join(lines), encoding="utf-8")
        out = audit_occupation(path)
        assert out["n_gt_occ_hi"] == 1
        assert out["n_lt_occ_lo"] == 1
        assert abs(out["worst_overshoot"]["occ"] - 1.00003) < 1e-12
        assert out["worst_overshoot"]["band"] == 3
        assert out["decay"]["top_two_still_occupied"] is True
        assert out["edge_peak"]["it"] == 2
        assert abs(out["edge_peak"]["edge_max_at_edge_sum_peak"] - 0.05) < 1e-12
        assert abs(out["global_edge_max"]["occ"] - 0.05) < 1e-12
        assert out["global_edge_max"]["band"] == 105
        b112 = next(x for x in out["bands"] if x["band"] == 112)
        assert abs(b112["at_edge_peak"]["pos_sum"] - 0.003) < 1e-12
        assert abs(out["decay"]["pos_sum_112_over_105"] - 0.003 / 0.05) < 1e-12


def test_top_two_requires_and() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "occupation_band_kt.dat"
        lines = ["# band-resolved occupation\n"]
        for b in range(105, 113):
            occ = 0.01 if b == 111 else 1.0e-6
            lines.append(_occ_line(1, 0.0, 1, 1, b, occ))
        path.write_text("".join(lines), encoding="utf-8")
        out = audit_occupation(path)
        assert out["decay"]["max_f_111"] > 1.0e-3
        assert out["decay"]["max_f_112"] < 1.0e-3
        assert out["decay"]["top_two_still_occupied"] is False


def test_global_edge_max_vs_sum_peak() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "occupation_band_kt.dat"
        lines = ["# band-resolved occupation\n"]
        # t=0: small total edge population, one large single-band spike
        lines.append(_occ_line(1, 0.0, 1, 1, 106, 0.07))
        for b in range(105, 113):
            if b != 106:
                lines.append(_occ_line(1, 0.0, 1, 1, b, 0.0))
        # t=1: larger total edge population, smaller peak
        for b in range(105, 113):
            lines.append(_occ_line(2, 1.0, 1, 1, b, 0.02))
        path.write_text("".join(lines), encoding="utf-8")
        out = audit_occupation(path)
        assert abs(out["global_edge_max"]["occ"] - 0.07) < 1e-12
        assert out["global_edge_max"]["it"] == 1
        assert out["edge_peak"]["it"] == 2
        assert abs(out["edge_peak"]["edge_max_at_edge_sum_peak"] - 0.02) < 1e-12


def test_jones_weight() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "HHG_nodes_modes.dat"
        w = 2.5e-4
        # nid w unused unused order jx_re jx_im jy_re jy_im unused
        rows = []
        for order, amp in ((2, 1.0), (5, 0.5), (7, 0.2), (9, 0.1), (10, 0.05)):
            rows.append(f"22 {w:.16e} 0 0 {order} {amp:.16e} 0 0 0 0\n")
        path.write_text("".join(rows), encoding="utf-8")
        got = jones_weighted(path, 22)
        assert abs(got["2"]["j2"] - 1.0) < 1e-12
        assert abs(got["2"]["w_j2"] - w) < 1e-18
        assert abs(got["2"]["w_j4"] - w) < 1e-18
        assert abs(got["5"]["w_j2"] - w * 0.25) < 1e-18


def test_probe_weight() -> None:
    freeze = {
        "manifests": {
            "nodes_sv_r2p5_th0_gh7.dat": {
                "wide_axis_pair": {"ids": [22, 28], "weights": [0.00025, 0.00025]},
                "corner_pair": {"ids": [1, 49], "weights": [3.0e-7, 3.0e-7]},
            }
        }
    }
    w = probe_weight(freeze, {"manifest": "nodes_sv_r2p5_th0_gh7.dat", "kind": "wide", "id": 22})
    assert abs(w - 0.00025) < 1e-18


if __name__ == "__main__":
    test_ladder_and_overshoot()
    test_top_two_requires_and()
    test_global_edge_max_vs_sum_peak()
    test_jones_weight()
    test_probe_weight()
    print("test_gh7_tail_edge_band_audit: PASS")

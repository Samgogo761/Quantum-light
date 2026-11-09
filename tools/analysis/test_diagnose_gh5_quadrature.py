#!/usr/bin/env python3
"""Unit + small end-to-end fixture for GH5 quadrature diagnostics."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_gh5_quadrature import (
    diagnose_state,
    jones_decomp,
    load_v1_report,
    outer_ring_ids,
)
from gh3_gh5_quadrature_gate import GATE_VERSION


def test_outer_ring_weight_count() -> None:
    ids = outer_ring_ids(5)
    assert len(ids) == 16
    assert 1 in ids and 13 not in ids and 25 in ids


def test_jones_amp_vs_vec() -> None:
    d = jones_decomp(1 + 0j, 0j, 0 + 1j, 0j)
    assert d["amp_rel"] < 1e-15
    assert abs(d["vec_rel"] - 2.0 ** 0.5) < 1e-12
    assert abs(d["phase_rad"] - 0.5 * 3.141592653589793) < 1e-12
    d2 = jones_decomp(1 + 0j, 0j, 1.03 + 0j, 0j)
    assert abs(d2["amp_rel"] - 0.03) < 1e-12
    assert d2["vec_rel"] < 0.04


def test_v1_report_must_be_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gh3, gh5 = root / "gh3", root / "gh5"
        gh3.mkdir()
        gh5.mkdir()
        bad = root / "v1_pass.json"
        bad.write_text(
            json.dumps(
                {
                    "gate": GATE_VERSION,
                    "status": "PASS",
                    "gh3_root": str(gh3),
                    "gh5_root": str(gh5),
                }
            ),
            encoding="utf-8",
        )
        try:
            load_v1_report(bad, gh3, gh5)
        except ValueError as exc:
            assert "FAIL" in str(exc)
        else:
            raise AssertionError("PASS v1 report must be rejected")

        ok = root / "v1_fail.json"
        ok.write_text(
            json.dumps(
                {
                    "gate": GATE_VERSION,
                    "status": "FAIL",
                    "gh3_root": str(gh3),
                    "gh5_root": str(gh5),
                }
            ),
            encoding="utf-8",
        )
        verified = load_v1_report(ok, gh3, gh5)
        assert verified["status"] == "FAIL"
        assert verified["roots_match"] is True


def _write_nml(path: Path) -> None:
    path.write_text(
        "&laser\n  wvl_nm = 3200.0\n/\n&timestep\n  dt = 0.35\n/\n  ncyc = 4.0\n",
        encoding="utf-8",
    )


def _write_ics(path: Path, ics: float) -> None:
    lines = ["# harmonic_order  omega  ICS  CS  var\n"]
    for h in (0.0, 2.0, 5.0, 7.0, 9.0, 10.0):
        lines.append(f"{h:.4f}  0.0  {ics:.8e}  {0.1 * ics:.8e}  0.0\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_manifest(path: Path, n_side: int, weight: float) -> list[dict]:
    rows = []
    lines = ["# sample_id  weight  Re(alpha)  Im(alpha)  I_drive  phi_drive\n"]
    nid = 0
    for ix in range(1, n_side + 1):
        for iy in range(1, n_side + 1):
            nid += 1
            re_a = float(ix - (n_side + 1) / 2.0)
            im_a = float(iy - (n_side + 1) / 2.0)
            intensity = re_a * re_a + im_a * im_a
            phi = 0.0 if intensity == 0.0 else (math.atan2(im_a, re_a) % (2.0 * math.pi))
            rows.append(
                {
                    "id": nid,
                    "weight": weight,
                    "re": re_a,
                    "im": im_a,
                    "intensity": intensity,
                    "phase": phi,
                }
            )
            lines.append(
                f"{nid:8d}{weight:25.17E}{re_a:25.17E}{im_a:25.17E}"
                f"{intensity:25.17E}{phi:25.17E}\n"
            )
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return rows


def _write_modes(
    path: Path,
    n_side: int,
    weight: float,
    amp_fn,
    *,
    minus: bool,
) -> None:
    lines = []
    nid = 0
    for ix in range(1, n_side + 1):
        for iy in range(1, n_side + 1):
            nid += 1
            jo = float(amp_fn(nid, ix, iy))
            sign = -1.0 if minus else 1.0
            jx = sign * jo
            jy = 0.0
            power = jo * jo
            re_a = float(ix - (n_side + 1) / 2.0)
            im_a = float(iy - (n_side + 1) / 2.0)
            intensity = re_a * re_a + im_a * im_a
            phi = 0.0 if intensity == 0.0 else (math.atan2(im_a, re_a) % (2.0 * math.pi))
            for order in (2, 5, 7, 9, 10):
                lines.append(
                    f"{nid} {weight:.17e} {intensity:.17e} {phi:.17e} {order} "
                    f"{jx:.17e} 0.0 {jy:.17e} 0.0 {power:.17e}\n"
                )
    path.write_text("".join(lines), encoding="utf-8")


def test_e2e_moments_outer_cep() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gh3 = root / "gh3"
        gh5 = root / "gh5"
        nodes = root / "nodes"
        for stem in ("sv_r2p5_th000_plusN", "sv_r2p5_th000_minusN"):
            (gh3 / stem).mkdir(parents=True)
            (gh5 / stem).mkdir(parents=True)
        nodes.mkdir()

        w3, w5 = 1.0 / 9.0, 1.0 / 25.0
        man5 = nodes / "nodes_sv_r2p5_th0_gh5.dat"
        _write_manifest(man5, 5, w5)

        def amp3(_nid: int, _ix: int, _iy: int) -> float:
            return 1.0

        def amp5(_nid: int, ix: int, iy: int) -> float:
            return 0.5 if ix in (1, 5) or iy in (1, 5) else 0.1

        _write_nml(gh3 / "sv_r2p5_th000_plusN" / "input.nml")
        _write_ics(gh3 / "sv_r2p5_th000_plusN" / "HHG_ics_cs.dat", 1.0)
        _write_ics(gh5 / "sv_r2p5_th000_plusN" / "HHG_ics_cs.dat", 2.0)
        _write_modes(gh3 / "sv_r2p5_th000_plusN" / "HHG_nodes_modes.dat", 3, w3, amp3, minus=False)
        _write_modes(gh3 / "sv_r2p5_th000_minusN" / "HHG_nodes_modes.dat", 3, w3, amp3, minus=True)
        _write_modes(gh5 / "sv_r2p5_th000_plusN" / "HHG_nodes_modes.dat", 5, w5, amp5, minus=False)
        _write_modes(gh5 / "sv_r2p5_th000_minusN" / "HHG_nodes_modes.dat", 5, w5, amp5, minus=True)

        out = diagnose_state("sv_r2p5_th000", gh3, gh5, man5)
        m2 = out["a1_moments"]["2"]
        assert abs(m2["A22o_gh3"] - 1.0) < 1e-12
        assert abs(m2["F22o_gh3"] - 1.0) < 1e-12
        a22_5 = 16.0 * w5 * 0.25 + 9.0 * w5 * 0.01
        f22_5 = 16.0 * w5 * (0.5**4) + 9.0 * w5 * (0.1**4)
        assert abs(m2["A22o_gh5"] - a22_5) < 1e-12
        assert abs(m2["F22o_gh5"] - f22_5) < 1e-12
        share = 16.0 * w5 * 0.25 / a22_5
        assert abs(out["tail_ics"]["2"]["outer_ics_share"] - share) < 1e-12
        assert abs(out["outer_weight_frac"] - 16.0 * w5) < 1e-12
        assert out["cep_pi_v2_gh5"]["pass"] is True
        assert out["cep_pi_v2_gh5"]["max_e_abs_hard"] < 1e-15
        assert out["cep_pi_v2_gh5"]["metadata_provenance_rechecked"] is False


if __name__ == "__main__":
    test_outer_ring_weight_count()
    test_jones_amp_vs_vec()
    test_v1_report_must_be_fail()
    test_e2e_moments_outer_cep()
    print("test_diagnose_gh5_quadrature: PASS")

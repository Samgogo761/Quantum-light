#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "a0_layerA_m88full" / "gh7_tail"))

from gh7_nodes import (
    EXPECTED_I_MAX,
    e2_axes,
    expected_case_names,
    make_sv_nodes_gh7,
    peer_k20_name,
    select_corner_antinode_pair,
    select_wide_axis_antinode_pair,
)
from gh7_tail_model_gate import (
    compare_jones,
    gate_exit_code,
    parse_band_occupation,
)

GATE = Path(__file__).resolve().parent / "gh7_tail_model_gate.py"


def test_axes_orthogonal() -> None:
    for th in (0.0, 180.0):
        major, minor = e2_axes(2.5, math.radians(th))
        assert abs(major[0] * minor[0] + major[1] * minor[1]) < 1e-12
        assert abs(math.hypot(*major) - 1.0) < 1e-12
        assert abs(math.hypot(*minor) - 1.0) < 1e-12


def test_corner_and_wide_ids() -> None:
    expect = {
        0.0: {"corner": [1, 49], "wide": [22, 28]},
        180.0: {"corner": [1, 49], "wide": [4, 46]},
    }
    for th, want in expect.items():
        rows = make_sv_nodes_gh7(2.5, th, 1.0e11)
        assert len(rows) == 49
        assert abs(sum(r[1] for r in rows) - 1.0) < 1e-14
        corner = select_corner_antinode_pair(rows, 2.5, th)
        wide = select_wide_axis_antinode_pair(rows, 2.5, th)
        assert corner["ids"] == want["corner"]
        assert wide["ids"] == want["wide"]
        assert abs(corner["I_max"] - EXPECTED_I_MAX) / EXPECTED_I_MAX < 5e-3
        assert wide["I_over_Imax"] > 0.99
        assert wide["weights"][0] / corner["weights"][0] > 100.0
        for pair in (corner, wide):
            a0, a1 = pair["alphas"]
            assert abs(a0[0] + a1[0]) < 1e-12
            assert abs(a0[1] + a1[1]) < 1e-12
            assert abs(pair["axis_dot"]) < 1e-12


def _write_occ(path: Path, nk: int, its: tuple[int, ...], *, drop: tuple | None = None, dup: bool = False) -> None:
    lines = ["# it time ikx iky band occ\n"]
    for it in its:
        for ikx in range(1, nk + 1):
            for iky in range(1, nk + 1):
                for band in range(1, 113):
                    if drop == (it, ikx, iky, band):
                        continue
                    occ = 1.0 if band <= 84 else 0.0
                    lines.append(f"{it} 0.0 {ikx} {iky} {band} {occ}\n")
                    if dup and (it, ikx, iky, band) == (its[0], 1, 1, 1):
                        lines.append(f"{it} 0.0 {ikx} {iky} {band} {occ}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_occupation_complete_grid() -> None:
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "good.dat"
        _write_occ(good, nk=2, its=(1, 2))
        rec = parse_band_occupation(good, nk=2, expected_its=[1, 2])
        assert rec["pass_trace"] and rec["pass_edge_abs"] and rec["pass_edge_rel"]
        assert rec["pass_grid"]
        assert abs(rec["trace0"] - 84 * 4) < 1e-12

        missing = Path(td) / "missing.dat"
        _write_occ(missing, nk=2, its=(1, 2), drop=(1, 2, 2, 112))
        try:
            parse_band_occupation(missing, nk=2, expected_its=[1, 2])
        except ValueError as exc:
            assert "incomplete" in str(exc)
        else:
            raise AssertionError("missing k/band must fail")

        dup = Path(td) / "dup.dat"
        _write_occ(dup, nk=2, its=(1, 2), dup=True)
        try:
            parse_band_occupation(dup, nk=2, expected_its=[1, 2])
        except ValueError as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate k/band must fail")


def test_occupation_edge_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.dat"
        lines = ["# it time ikx iky band occ\n"]
        for it in (1, 2):
            for ikx, iky in ((1, 1), (1, 2), (2, 1), (2, 2)):
                for band in range(1, 113):
                    if it == 1:
                        occ = 1.0 if band <= 84 else 0.0
                    else:
                        occ = 0.5 if band >= 105 else (1.0 if band <= 84 else 0.02)
                    lines.append(f"{it} 0.0 {ikx} {iky} {band} {occ}\n")
        bad.write_text("".join(lines), encoding="utf-8")
        rec = parse_band_occupation(bad, nk=2, expected_its=[1, 2])
        assert rec["pass_edge_rel"] is False or rec["pass_edge_abs"] is False


def test_k40_name_pairing() -> None:
    assert (
        peer_k20_name("sv_r2p5_th000_plusN_id22_wide_k40")
        == "sv_r2p5_th000_plusN_id22_wide_k20"
    )
    try:
        peer_k20_name("sv_r2p5_th000_plusN_id22_wide_k20")
    except ValueError:
        pass
    else:
        raise AssertionError("k20 name must not pair as k40")


def test_incomplete_set_is_nonzero() -> None:
    assert gate_exit_code("PASS") == 0
    for status in ("FAIL", "INCOMPLETE", "BLOCKED", "NOT_RUN"):
        assert gate_exit_code(status) == 1
    probes = [
        {"theta_deg": 0, "id": 1, "kind": "corner"},
        {"theta_deg": 0, "id": 49, "kind": "corner"},
        {"theta_deg": 0, "id": 22, "kind": "wide"},
        {"theta_deg": 0, "id": 28, "kind": "wide"},
        {"theta_deg": 180, "id": 1, "kind": "corner"},
        {"theta_deg": 180, "id": 49, "kind": "corner"},
        {"theta_deg": 180, "id": 4, "kind": "wide"},
        {"theta_deg": 180, "id": 46, "kind": "wide"},
    ]
    expected = expected_case_names(probes, 20)
    assert len(expected) == 8
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        freeze = root / "FREEZE.json"
        freeze.write_text(
            json.dumps(
                {
                    "plusN_probes": [
                        {**p, "manifest_sha256": "abc", "I_max": 1.0} for p in probes
                    ],
                    "pinned_binary_sha256": "bin",
                    "tb_plus_sha256": "tb",
                    "git_head": "TO_BE_PINNED",
                    "occ_stride": 336,
                }
            ),
            encoding="utf-8",
        )
        k20 = root / "k20"
        k20.mkdir()
        only = k20 / expected[0]
        only.mkdir()
        (only / "SUCCESS").write_text("", encoding="utf-8")
        report = root / "report.json"
        proc = subprocess.run(
            [sys.executable, str(GATE), "--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--report", str(report)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["status"] != "PASS"
        assert payload["cep_pi_tested"] is False
        assert len(payload["missing_cases"]) == 7


def test_dt2_not_accepted_stage() -> None:
    proc = subprocess.run(
        [sys.executable, str(GATE), "--stage", "dt2", "--freeze", "x"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "dt2" in (proc.stderr + proc.stdout).lower() or proc.returncode == 2


def test_mixed_jones_weak_absolute() -> None:
    weak_a = {n: (1e-16 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    weak_b = {n: (2e-16 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    weak = compare_jones(weak_a, weak_b)
    assert weak["orders"]["2"]["mixed"]["strong"] is False
    assert weak["pass"] is True
    strong_a = {n: (1 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    strong_b = {n: (1.2 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    strong = compare_jones(strong_a, strong_b)
    assert strong["pass"] is False


if __name__ == "__main__":
    test_axes_orthogonal()
    test_corner_and_wide_ids()
    test_occupation_complete_grid()
    test_occupation_edge_fail()
    test_k40_name_pairing()
    test_incomplete_set_is_nonzero()
    test_dt2_not_accepted_stage()
    test_mixed_jones_weak_absolute()
    print("test_gh7_tail_model_gate: PASS")

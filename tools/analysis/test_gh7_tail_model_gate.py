#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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
    peer_k_name,
    select_corner_antinode_pair,
    select_wide_axis_antinode_pair,
)
from gh7_tail_model_gate import (
    JONES_ATOL,
    REQUIRED_OUTPUT_FILES,
    compare_jones,
    gate_exit_code,
    parse_band_occupation,
    same_case_set,
)
from validate_a0_run_strict import validate as validate_run

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
    assert (
        peer_k_name("sv_r2p5_th000_plusN_id22_wide_k2", 2, 2)
        == "sv_r2p5_th000_plusN_id22_wide_k2"
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
                    "freeze_pin_base_head": "TO_BE_PINNED",
                    "occ_stride": 126,
                    "squeeze_r": 2.5,
                    "I_bar": 1.0e11,
                    "dt": 0.35,
                    "T2_cycles": 0.5,
                    "harmonics": [2, 5, 7, 9, 10],
                    "template_sha256": "t" * 64,
                    "worktree_source_sha256": "s" * 64,
                    "sbatch_sha256": "b" * 64,
                    "validator_run_sha256": "v" * 64,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "FREEZE.sha256").write_text(
            hashlib.sha256(freeze.read_bytes()).hexdigest() + "\n", encoding="ascii"
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
    scales = {n: 1.0e-2 for n in (2, 5, 7, 9, 10)}
    weak_a = {n: (1e-16 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    weak_b = {n: (2e-16 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    weak = compare_jones(weak_a, weak_b, scales)
    assert weak["orders"]["2"]["mixed"]["strong"] is False
    assert weak["pass"] is True
    zero = {n: (0j, 0j) for n in (2, 5, 7, 9, 10)}
    drift = {n: (4.9e-5 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    counter = compare_jones(zero, drift, scales)
    assert counter["pass"] is False
    assert counter["orders"]["2"]["mixed"]["e_abs"] > JONES_ATOL
    strong_a = {n: (1 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    strong_b = {n: (1.2 + 0j, 0j) for n in (2, 5, 7, 9, 10)}
    strong = compare_jones(strong_a, strong_b, {n: 1.0 for n in (2, 5, 7, 9, 10)})
    assert strong["pass"] is False


def test_occupation_exc_floor() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "floor.dat"
        lines = ["# it time ikx iky band occ\n"]
        for it in (1, 2):
            for ikx, iky in ((1, 1), (1, 2), (2, 1), (2, 2)):
                for band in range(1, 113):
                    if it == 1 or band <= 84:
                        occ = 1.0 if band <= 84 else 0.0
                    else:
                        occ = 1.0e-15 if band >= 105 else 0.0
                    lines.append(f"{it} 0.0 {ikx} {iky} {band} {occ}\n")
        path.write_text("".join(lines), encoding="utf-8")
        rec = parse_band_occupation(path, nk=2, expected_its=[1, 2])
        assert rec["pass_edge_abs"] is True
        assert rec["pass_edge_rel"] is True


def test_same_case_set_ignores_order() -> None:
    expected = [
        "sv_r2p5_th000_plusN_id01_corner_k20",
        "sv_r2p5_th000_plusN_id49_corner_k20",
        "sv_r2p5_th000_plusN_id22_wide_k20",
        "sv_r2p5_th000_plusN_id28_wide_k20",
    ]
    found = sorted(expected)
    assert found != expected
    assert same_case_set(found, expected)
    assert not same_case_set(found + [found[0]], expected)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PROBES_8 = [
    {"theta_deg": 0, "id": 1, "kind": "corner"},
    {"theta_deg": 0, "id": 49, "kind": "corner"},
    {"theta_deg": 0, "id": 22, "kind": "wide"},
    {"theta_deg": 0, "id": 28, "kind": "wide"},
    {"theta_deg": 180, "id": 1, "kind": "corner"},
    {"theta_deg": 180, "id": 49, "kind": "corner"},
    {"theta_deg": 180, "id": 4, "kind": "wide"},
    {"theta_deg": 180, "id": 46, "kind": "wide"},
]
PIN_SHAS = {
    "template_sha256": "a" * 64,
    "worktree_source_sha256": "b" * 64,
    "sbatch_sha256": "c" * 64,
    "validator_run_sha256": "d" * 64,
}


def _dump_freeze(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("FREEZE.sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _manifest_49() -> str:
    return "".join(f"{nid} 2.04081632653061232E-02 0.0 0.0 1.0 0.0\n" for nid in range(1, 50))


def _nml_text(nk: int) -> str:
    return (
        "&kgrid\n"
        f"  nkx = {nk}\n"
        f"  nky = {nk}\n"
        "/\n"
        "&timestep\n"
        "  dt = 0.35\n"
        "/\n"
        "&laser\n"
        "  wvl_nm = 3200.0\n"
        "  ncyc = 0.001\n"
        "/\n"
        "&dephasing\n"
        "  T2_cycles = 0.5\n"
        "/\n"
        "&output\n"
        "  occ_stride = 1\n"
        "/\n"
    )


def _modes_text(node_id: int, *, jx: float = 1.0) -> str:
    lines = []
    for order in (2, 5, 7, 9, 10):
        lines.append(f"{node_id} 0.1 1.0 0.0 {order} 1.0 {jx} 0.0 0.0 1.0\n")
    return "".join(lines)


def _write_output_sha256(out: Path) -> None:
    (out / "output_sha256.txt").write_text(
        "\n".join(f"{_sha(out / name)}  {name}" for name in REQUIRED_OUTPUT_FILES) + "\n",
        encoding="utf-8",
    )


def _write_prod_case(
    root: Path,
    name: str,
    probe: dict,
    *,
    nk: int,
    actual_head: str,
    base_head: str,
    freeze_sha: str,
    bin_sha: str,
    tb_sha: str,
    man_text: str,
    man_sha: str,
) -> Path:
    out = root / name
    out.mkdir()
    (out / "SUCCESS").write_text("", encoding="utf-8")
    (out / "run_status.txt").write_text("status=PASS\n", encoding="utf-8")
    (out / "input.nml").write_text(_nml_text(nk), encoding="utf-8")
    (out / "nodes_manifest.input.dat").write_text(man_text, encoding="utf-8")
    (out / "HHG_nodes_modes.dat").write_text(_modes_text(int(probe["id"])), encoding="utf-8")
    (out / "chunk_info.txt").write_text(
        f"n_manifest_nodes = 49\nn_propagate_nodes = 1\npropagate_ids = {probe['id']}\n",
        encoding="utf-8",
    )
    (out / "chunk_weighted_spectrum.dat").write_text(
        "# spectrum_scale = 1.0\n"
        "1 2.0 0.1 0.1 0.1 0.0 0.0 0.0\n"
        "2 5.0 0.1 0.1 0.1 0.0 0.0 0.0\n"
        "3 7.0 0.1 0.1 0.0 0.0 0.0 0.1\n"
        "4 9.0 0.1 0.1 0.1 0.0 0.0 0.0\n"
        "5 10.0 0.1 0.1 0.1 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (out / "nodes_moment_check.txt").write_text("pass = T\n", encoding="utf-8")
    (out / "run.log").write_text("ok\n", encoding="utf-8")
    (out / "node_preflight.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    (out / "run_validator.json").write_text('{"status":"PASS","mode":"chunk"}\n', encoding="utf-8")
    (out / "occupation_kt.dat").write_text("# occupation\n", encoding="utf-8")
    _write_occ(out / "occupation_band_kt.dat", nk=nk, its=(1, 2, 3))
    (out / "run_metadata.txt").write_text(
        "\n".join(
            [
                f"binary_sha256={bin_sha}",
                f"tb_sha256={tb_sha}",
                f"nodes_sha256={man_sha}",
                f"propagate_ids={probe['id']}",
                f"freeze_pin_base_head={base_head}",
                f"campaign_actual_head={actual_head}",
                f"git_head={actual_head}",
                f"freeze_sha256={freeze_sha}",
                f"template_sha256={PIN_SHAS['template_sha256']}",
                f"worktree_source_sha256={PIN_SHAS['worktree_source_sha256']}",
                f"source_sha256={PIN_SHAS['worktree_source_sha256']}",
                f"sbatch_sha256={PIN_SHAS['sbatch_sha256']}",
                f"validator_run_sha256={PIN_SHAS['validator_run_sha256']}",
                f"omp_num_threads=36",
                f"mkl_num_threads=36",
                f"nk={nk}",
                "dt=0.35",
                "T2_cycles=0.5",
                "squeeze_r=2.5",
                f"squeeze_theta_deg={int(probe['theta_deg'])}",
                "I_bar=1e+11",
                "harmonics=2,5,7,9,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_output_sha256(out)
    return out


def _freeze_payload(probes: list[dict], man_sha: str, base_head: str = "basehead") -> dict:
    return {
        "plusN_probes": [{**p, "manifest_sha256": man_sha, "I_max": 1.0} for p in probes],
        "pinned_binary_sha256": "bin",
        "tb_plus_sha256": "tb",
        "freeze_pin_base_head": base_head,
        "occ_stride": 1,
        "squeeze_r": 2.5,
        "I_bar": 1.0e11,
        "dt": 0.35,
        "T2_cycles": 0.5,
        "harmonics": [2, 5, 7, 9, 10],
        **PIN_SHAS,
    }


def _run_gate(args: list[str]) -> tuple[int, dict]:
    report = Path(args[args.index("--report") + 1])
    proc = subprocess.run(
        [sys.executable, str(GATE), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    return proc.returncode, payload


def test_eight_cases_pass_lexical_mismatch() -> None:
    probes = list(PROBES_8)
    expected = expected_case_names(probes, 2)
    assert sorted(expected) != expected
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k20.mkdir()
        for name, probe in zip(expected, probes, strict=True):
            _write_prod_case(
                k20,
                name,
                probe,
                nk=2,
                actual_head="actualhead",
                base_head="basehead",
                freeze_sha=digest,
                bin_sha="bin",
                tb_sha="tb",
                man_text=man_text,
                man_sha=man_sha,
            )
        rc, payload = _run_gate(
            ["--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--nk", "2", "--report", str(root / "report.json")]
        )
        assert rc == 0, payload
        assert payload["status"] == "PASS"
        assert payload["campaign_actual_heads"] == ["actualhead"]


def test_k40_cli_pass() -> None:
    probes = list(PROBES_8)
    names20 = expected_case_names(probes, 2)
    names40 = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k40 = root / "k40"
        k20.mkdir()
        k40.mkdir()
        for name, probe in zip(names20, probes, strict=True):
            _write_prod_case(
                k20,
                name,
                probe,
                nk=2,
                actual_head="actualhead",
                base_head="basehead",
                freeze_sha=digest,
                bin_sha="bin",
                tb_sha="tb",
                man_text=man_text,
                man_sha=man_sha,
            )
        for name, probe in zip(names40, probes, strict=True):
            _write_prod_case(
                k40,
                name,
                probe,
                nk=2,
                actual_head="actualhead",
                base_head="basehead",
                freeze_sha=digest,
                bin_sha="bin",
                tb_sha="tb",
                man_text=man_text,
                man_sha=man_sha,
            )
        (k20 / "GH7_TAIL_K20.json").write_text(
            json.dumps({"status": "PASS", "campaign_actual_heads": ["actualhead"]}) + "\n",
            encoding="utf-8",
        )
        rc, payload = _run_gate(
            [
                "--stage",
                "k40",
                "--freeze",
                str(freeze),
                "--k20-root",
                str(k20),
                "--k40-root",
                str(k40),
                "--nk",
                "2",
                "--k40-nk",
                "2",
                "--report",
                str(root / "report.json"),
            ]
        )
        assert rc == 0, payload
        assert payload["status"] == "PASS"
        assert payload["campaign_actual_heads"] == ["actualhead"]
        assert all(row.get("campaign_actual_head") == "actualhead" for row in payload["pairs"])


def test_missing_input_fails() -> None:
    probes = list(PROBES_8)
    expected = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k20.mkdir()
        cases = []
        for name, probe in zip(expected, probes, strict=True):
            cases.append(
                _write_prod_case(
                    k20,
                    name,
                    probe,
                    nk=2,
                    actual_head="actualhead",
                    base_head="basehead",
                    freeze_sha=digest,
                    bin_sha="bin",
                    tb_sha="tb",
                    man_text=man_text,
                    man_sha=man_sha,
                )
            )
        cases[0].joinpath("input.nml").unlink()
        rc, payload = _run_gate(
            ["--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--nk", "2", "--report", str(root / "report.json")]
        )
        assert rc != 0
        assert payload["status"] != "PASS"


def test_missing_meta_fields_fail() -> None:
    probes = list(PROBES_8)
    expected = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k20.mkdir()
        cases = []
        for name, probe in zip(expected, probes, strict=True):
            cases.append(
                _write_prod_case(
                    k20,
                    name,
                    probe,
                    nk=2,
                    actual_head="actualhead",
                    base_head="basehead",
                    freeze_sha=digest,
                    bin_sha="bin",
                    tb_sha="tb",
                    man_text=man_text,
                    man_sha=man_sha,
                )
            )
        meta = cases[0] / "run_metadata.txt"
        kept = [
            line
            for line in meta.read_text(encoding="utf-8").splitlines()
            if not line.startswith("freeze_pin_base_head=") and not line.startswith("freeze_sha256=")
        ]
        meta.write_text("\n".join(kept) + "\n", encoding="utf-8")
        _write_output_sha256(cases[0])
        rc, payload = _run_gate(
            ["--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--nk", "2", "--report", str(root / "report.json")]
        )
        assert rc != 0
        assert payload["status"] != "PASS"


def test_tampered_freeze_sidecar_fails() -> None:
    probes = list(PROBES_8)
    expected = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k20.mkdir()
        for name, probe in zip(expected, probes, strict=True):
            _write_prod_case(
                k20,
                name,
                probe,
                nk=2,
                actual_head="actualhead",
                base_head="basehead",
                freeze_sha=digest,
                bin_sha="bin",
                tb_sha="tb",
                man_text=man_text,
                man_sha=man_sha,
            )
        (root / "FREEZE.sha256").write_text("0" * 64 + "\n", encoding="ascii")
        rc, payload = _run_gate(
            ["--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--nk", "2", "--report", str(root / "report.json")]
        )
        assert rc != 0
        assert payload["status"] == "FAIL"
        assert "FREEZE.sha256" in payload["reason"]


def test_chunk_count_mismatch_fails() -> None:
    probes = list(PROBES_8)
    expected = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k20.mkdir()
        cases = []
        for name, probe in zip(expected, probes, strict=True):
            cases.append(
                _write_prod_case(
                    k20,
                    name,
                    probe,
                    nk=2,
                    actual_head="actualhead",
                    base_head="basehead",
                    freeze_sha=digest,
                    bin_sha="bin",
                    tb_sha="tb",
                    man_text=man_text,
                    man_sha=man_sha,
                )
            )
        cases[0].joinpath("chunk_info.txt").write_text(
            "n_manifest_nodes = 777\nn_propagate_nodes = 2\npropagate_ids = 1\n",
            encoding="utf-8",
        )
        _write_output_sha256(cases[0])
        rc, payload = _run_gate(
            ["--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--nk", "2", "--report", str(root / "report.json")]
        )
        assert rc != 0
        assert payload["status"] != "PASS"


def test_tampered_k20_modes_fails_k40() -> None:
    probes = list(PROBES_8)
    names = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k40 = root / "k40"
        k20.mkdir()
        k40.mkdir()
        cases20 = []
        for name, probe in zip(names, probes, strict=True):
            cases20.append(
                _write_prod_case(
                    k20,
                    name,
                    probe,
                    nk=2,
                    actual_head="actualhead",
                    base_head="basehead",
                    freeze_sha=digest,
                    bin_sha="bin",
                    tb_sha="tb",
                    man_text=man_text,
                    man_sha=man_sha,
                )
            )
            _write_prod_case(
                k40,
                name,
                probe,
                nk=2,
                actual_head="actualhead",
                base_head="basehead",
                freeze_sha=digest,
                bin_sha="bin",
                tb_sha="tb",
                man_text=man_text,
                man_sha=man_sha,
            )
        cases20[0].joinpath("HHG_nodes_modes.dat").write_text(
            _modes_text(int(probes[0]["id"]), jx=10.0),
            encoding="utf-8",
        )
        (k20 / "GH7_TAIL_K20.json").write_text(json.dumps({"status": "PASS"}) + "\n", encoding="utf-8")
        rc, payload = _run_gate(
            [
                "--stage",
                "k40",
                "--freeze",
                str(freeze),
                "--k20-root",
                str(k20),
                "--k40-root",
                str(k40),
                "--nk",
                "2",
                "--k40-nk",
                "2",
                "--report",
                str(root / "report.json"),
            ]
        )
        assert rc != 0
        assert payload["status"] == "FAIL"


def test_output_hash_missing_required_fails() -> None:
    probes = list(PROBES_8)
    expected = expected_case_names(probes, 2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        man_text = _manifest_49()
        man_sha = hashlib.sha256(man_text.encode("utf-8")).hexdigest()
        freeze = root / "FREEZE.json"
        digest = _dump_freeze(freeze, _freeze_payload(probes, man_sha))
        k20 = root / "k20"
        k20.mkdir()
        cases = []
        for name, probe in zip(expected, probes, strict=True):
            cases.append(
                _write_prod_case(
                    k20,
                    name,
                    probe,
                    nk=2,
                    actual_head="actualhead",
                    base_head="basehead",
                    freeze_sha=digest,
                    bin_sha="bin",
                    tb_sha="tb",
                    man_text=man_text,
                    man_sha=man_sha,
                )
            )
        listed = [name for name in REQUIRED_OUTPUT_FILES if name != "occupation_kt.dat"]
        cases[0].joinpath("output_sha256.txt").write_text(
            "\n".join(f"{_sha(cases[0] / name)}  {name}" for name in listed) + "\n",
            encoding="utf-8",
        )
        rc, payload = _run_gate(
            ["--stage", "k20", "--freeze", str(freeze), "--k20-root", str(k20), "--nk", "2", "--report", str(root / "report.json")]
        )
        assert rc != 0
        assert payload["status"] != "PASS"


def test_validate_chunk_ics() -> None:
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        man = run / "man.dat"
        man.write_text("1 1.00000000000000000E-01 0.0 0.0 1.0 0.0\n", encoding="utf-8")
        (run / "HHG_nodes_modes.dat").write_text(
            "1 1.00000000000000000E-01 1.0 0.0 2 1.0 0.0 0.0 0.0 1.0\n",
            encoding="utf-8",
        )
        (run / "chunk_info.txt").write_text(
            "n_manifest_nodes = 1\nn_propagate_nodes = 1\npropagate_ids = 1\n",
            encoding="utf-8",
        )
        (run / "chunk_weighted_spectrum.dat").write_text(
            "# spectrum_scale = 1.0\n"
            "1 2.0 0.1 0.1 0.1 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        (run / "nodes_moment_check.txt").write_text("pass = T\n", encoding="utf-8")
        (run / "run.log").write_text("ok\n", encoding="utf-8")
        result = validate_run(run, man, [2], 2.0e-7, 1, True)
        assert result["status"] == "PASS"
        assert result["mode"] == "chunk"


def test_validate_chunk_count_mismatch() -> None:
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        man = run / "man.dat"
        man.write_text("1 1.00000000000000000E-01 0.0 0.0 1.0 0.0\n", encoding="utf-8")
        (run / "HHG_nodes_modes.dat").write_text(
            "1 1.00000000000000000E-01 1.0 0.0 2 1.0 0.0 0.0 0.0 1.0\n",
            encoding="utf-8",
        )
        (run / "chunk_info.txt").write_text(
            "n_manifest_nodes = 777\nn_propagate_nodes = 2\npropagate_ids = 1\n",
            encoding="utf-8",
        )
        (run / "chunk_weighted_spectrum.dat").write_text(
            "# spectrum_scale = 1.0\n"
            "1 2.0 0.1 0.1 0.1 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        (run / "nodes_moment_check.txt").write_text("pass = T\n", encoding="utf-8")
        (run / "run.log").write_text("ok\n", encoding="utf-8")
        try:
            validate_run(run, man, [2], 2.0e-7, 1, True)
        except ValueError as exc:
            assert "n_manifest_nodes" in str(exc) or "n_propagate_nodes" in str(exc)
        else:
            raise AssertionError("forged chunk counts must fail")


if __name__ == "__main__":
    test_axes_orthogonal()
    test_corner_and_wide_ids()
    test_occupation_complete_grid()
    test_occupation_edge_fail()
    test_k40_name_pairing()
    test_incomplete_set_is_nonzero()
    test_dt2_not_accepted_stage()
    test_mixed_jones_weak_absolute()
    test_occupation_exc_floor()
    test_same_case_set_ignores_order()
    test_eight_cases_pass_lexical_mismatch()
    test_k40_cli_pass()
    test_missing_input_fails()
    test_missing_meta_fields_fail()
    test_tampered_freeze_sidecar_fails()
    test_chunk_count_mismatch_fails()
    test_tampered_k20_modes_fails_k40()
    test_output_hash_missing_required_fails()
    test_validate_chunk_ics()
    test_validate_chunk_count_mismatch()
    print("test_gh7_tail_model_gate: PASS")

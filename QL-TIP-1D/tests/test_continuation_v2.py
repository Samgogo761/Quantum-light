"""R0 contract tests: fixed thresholds, readout plan, preparation binding."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from ql_tip_1d.continuation_v2 import THRESHOLDS_EV, base_energy_range, readout_plan, repair_tasks
from ql_tip_1d.statistics_v2 import bsv_pdf, linear_weights, raw_moment, tail_bound, thermal_pdf
from run_continuation_v2 import validate_prep_binding, validate_inputs


def test_readout_plan_fixes_three_thresholds_independent_of_eta() -> None:
    """Every planned readout must use the same absolute thresholds."""
    specs = readout_plan()
    assert len(specs) == 64
    assert THRESHOLDS_EV == (0.5170816, 0.7170816, 0.9170816)
    for spec in specs:
        assert tuple(spec["thresholds_eV"]) == THRESHOLDS_EV
        assert spec["eta_eV"] in {0.05, 0.025}
    assert len({spec["readout_id"] for spec in specs}) == len(specs)


def test_energy_range_extension_does_not_move_thresholds() -> None:
    """A larger E_max changes only the grid, never the fixed physical windows."""
    tasks = repair_tasks()
    config = tasks[0]["config"]
    lower, upper = base_energy_range(config)
    assert lower < THRESHOLDS_EV[0]
    assert upper > THRESHOLDS_EV[2]
    assert THRESHOLDS_EV[1] == 0.7170816


def test_wrong_preparation_binding_is_rejected() -> None:
    """Changing F without changing the cached preparation must fail."""
    task = repair_tasks()[0]
    bad = dict(task)
    bad["config"] = dict(task["config"])
    bad["config"]["F_dc_V_per_nm"] = 1.0 if task["config"]["F_dc_V_per_nm"] == 0.0 else 0.0
    with pytest.raises(ValueError, match="Preparation F"):
        validate_prep_binding(bad)


def test_current_inputs_and_saved_state_times_match_manifest() -> None:
    """All frozen inputs must hash and all state times must match the manifest."""
    inventory = validate_inputs()
    assert inventory["pass"], inventory["bad_inputs"][:2] or inventory["bad_state_times"][:2]
    assert inventory["input_count"] == 154
    assert inventory["state_time_count"] == 36


def test_weight_moments_and_truncation_normalisation() -> None:
    """The frozen weights reproduce the analytic moments and unnormalised sum."""
    assert [raw_moment("bsv", k) for k in range(1, 5)] == [1.0, 3.0, 15.0, 105.0]
    assert [raw_moment("thermal", k) for k in range(1, 5)] == [1.0, 2.0, 6.0, 24.0]
    nodes = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 32.0, 64.0, 256.0, 1024.0]
    for state, expected_mean in [("bsv", 1.0), ("thermal", 1.0)]:
        weights = linear_weights(nodes, state, 2048.0)
        tail = tail_bound(2048.0, state)
        assert abs(weights.sum() + tail - 1.0) < 5e-8
        mean = float(sum(w * x for w, x in zip(weights, nodes)))
        assert abs(mean - expected_mean) < 1e-5
        assert abs(bsv_pdf(2.0) - __import__("math").exp(-1.0) / __import__("math").sqrt(4.0 * __import__("math").pi)) < 1e-15
        assert abs(thermal_pdf(2.0) - __import__("math").exp(-2.0)) < 1e-15


def test_old_mu_readout_is_visible_to_statistics_index() -> None:
    """The merged index must find frozen v1 states through v2 readouts."""
    from analyze_statistics_v2 import find_readout
    item = find_readout("bias_p1_F0.000_I1_phi000_r02")
    assert item is not None
    assert item["eta_eV"] == 0.05


def test_midpoint_tasks_have_complete_fields() -> None:
    """Midpoint tasks must expose F, intensity and CEP used by the runner."""
    from ql_tip_1d.continuation_v2 import midpoint_tasks
    plan = {"intervals": [{"left": 4.0, "right": 8.0, "midpoint_I_over_mu": 6.0},
                          {"left": 8.0, "right": 12.0, "midpoint_I_over_mu": 10.0}]}
    tasks = midpoint_tasks(plan)
    assert len(tasks) == 16
    assert {task["F_dc_V_per_nm"] for task in tasks} == {0.0, 1.0}
    assert {task["I_over_mu"] for task in tasks} == {6.0, 10.0}
    assert all("CEP_rad" in task for task in tasks)


def test_explicit_energy_range_has_distinct_readout_id() -> None:
    """Manual energy-range changes must not collide with the default identity."""
    from ql_tip_1d.continuation_v2 import readout_id
    base = {"run_id": "r", "expected_readout_fs": 100.0, "eta_eV": 0.05,
            "step_eV": 0.0125, "mask_start_nm": 2.0, "mask_end_nm": 4.0,
            "energy_max_mode": "explicit"}
    a = readout_id({**base, "energy_min_eV": -2.0, "energy_max_eV": 60.0})
    b = readout_id({**base, "energy_min_eV": -2.0, "energy_max_eV": 80.0})
    assert a != b


def test_library_refuses_when_active_evidence_disallows(monkeypatch) -> None:
    """The library entry must not bypass the active-evidence permission."""
    import run_continuation_v2 as runner
    import pytest
    def denied(stage):
        raise RuntimeError("R2 gate is not allowed")
    monkeypatch.setattr(runner, "load_active", denied)
    with pytest.raises(RuntimeError, match="R2 gate"):
        runner.stage_library(True, 1, 0)


def test_library_success_paths_without_tdse(tmp_path, monkeypatch) -> None:
    """Both library dry-run paths must work when mocked gates are PASS."""
    import json
    import run_continuation_v2 as runner
    import ql_tip_1d.continuation_v2 as common
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(runner, "V4_REPORT", tmp_path)
    (tmp_path / "midpoint_plan.json").write_text('{"intervals":[]}', encoding="utf-8")
    monkeypatch.setattr(runner, "load_active", lambda stage: {"allowed_task_ids": ["mock"]})
    task = {"run_id": "mock", "F_dc_V_per_nm": 0.0, "I_over_mu": 1.0,
            "CEP_rad": 0.0, "config": {}, "parent_config_sha256": "x"}
    monkeypatch.setattr(common, "library_tasks", lambda: [{**task, "existing": False}])
    monkeypatch.setattr(common, "midpoint_tasks", lambda plan: [task])
    assert runner.stage_library(True, 1, 0) == 0
    assert runner.stage_library(True, 1, 0, midpoint=True) == 0

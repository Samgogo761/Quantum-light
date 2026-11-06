"""R1 gate and error-budget analysis for continuation-v2.

All inputs are saved readouts; this module never evolves a wavefunction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ql_tip_1d.continuation_v2 import (CONTROL_ETA, CONTROL_STEP, READOUT_ETA, READOUT_STEP,
                                       THRESHOLDS_EV, V2_RESULTS, project_path, readout_id,
                                       readout_plan)
from ql_tip_1d.bias_v1 import sha256

REVISION = "r1_20261105_v1"
REPORT_DIR = V2_RESULTS / "reports" / REVISION
LOW_ENERGY_FLOOR_REL = 3.03396822960638e-4
CONDITIONS = {"F0_I1": (0.0, 1.0), "F1_I1": (1.0, 1.0), "F0_I4": (0.0, 4.0), "F1_I4": (1.0, 4.0)}
PHASES = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]


def rid(run_id: str, kind: str, *, eta: float = READOUT_ETA, step: float = READOUT_STEP,
        mask: tuple[float, float] = (2.0, 4.0), readout_fs: float = 100.0,
        emax: str = "base") -> str:
    """Reconstruct a deterministic readout ID."""
    return readout_id({
        "run_id": run_id, "expected_readout_fs": readout_fs, "eta_eV": eta,
        "step_eV": step, "mask_start_nm": mask[0], "mask_end_nm": mask[1],
        "energy_max_mode": emax, "kind": kind,
    })


def load_readout(run_id: str, kind: str, **kwargs: Any) -> dict[str, Any] | None:
    """Load a readout summary, returning None when a planned point is missing."""
    path = V2_RESULTS / "readouts" / rid(run_id, kind, **kwargs) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def phase_run_ids(F: float, intensity: float) -> list[str]:
    """Return the four production CEP run IDs in fixed order."""
    if intensity == 1.0:
        first = f"bias_p1_F{F:.3f}_I1_phi000_r02"
    else:
        first = f"bias_p2_F{F:.3f}_I{intensity:g}_phi000_r02"
    return [first] + [f"bias_p2_F{F:.3f}_I{intensity:g}_phi{degrees:03d}_r02"
                      for degrees in (90, 180, 270)]


def mean_main(readouts: list[dict[str, Any]]) -> float:
    """Return the equal-weight mean of the absolute main-threshold yield."""
    return float(np.mean([item["Y_at_E_main"] for item in readouts]))


def main_readouts(F: float, intensity: float, *, eta: float = READOUT_ETA,
                  step: float = READOUT_STEP) -> list[dict[str, Any]]:
    """Load all four primary readouts for one condition."""
    items = [load_readout(run_id, "primary" if eta == READOUT_ETA else "primary_eta_control",
                          eta=eta, step=step) for run_id in phase_run_ids(F, intensity)]
    if any(item is None for item in items):
        return [item for item in items if item is not None]
    return [item for item in items if item is not None]


def relative(reference: float, test: float) -> float:
    """Return a signed relative change, refusing a zero reference."""
    if reference == 0.0:
        raise ZeroDivisionError("zero reference yield")
    return (test - reference) / reference


def add_component(rows: list[dict[str, Any]], condition: str, component: str, reference: dict[str, Any] | None,
                  control: dict[str, Any] | None, direct: str, note: str = "") -> float:
    """Record one error component and return its absolute relative value."""
    if reference is None or control is None:
        rows.append({"condition": condition, "component": component, "reference_id": "",
                     "control_id": "", "reference_Y": "", "control_Y": "",
                     "relative_change": "", "abs_relative": "", "coverage": "missing", "note": note})
        return math.nan
    ref_y = float(reference["Y_at_E_main"])
    ctl_y = float(control["Y_at_E_main"])
    change = relative(ref_y, ctl_y)
    rows.append({"condition": condition, "component": component,
                 "reference_id": reference["readout_id"], "control_id": control["readout_id"],
                 "reference_Y": f"{ref_y:.17g}", "control_Y": f"{ctl_y:.17g}",
                 "relative_change": f"{change:.17g}", "abs_relative": f"{abs(change):.17g}",
                 "coverage": direct, "note": note})
    return abs(change)


def run_summary(run_id: str) -> dict[str, Any]:
    """Load the propagation summary from v2 or the frozen v1 tree."""
    v2 = V2_RESULTS / "runs" / run_id / "summary.json"
    v1 = PROJECT / "results/bias_v1" / run_id / "summary.json"
    path = v2 if v2.exists() else v1
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compute_gate() -> dict[str, Any]:
    """Compute the four representative total-error budgets and R1 decision."""
    rows: list[dict[str, Any]] = []
    condition_summary: dict[str, Any] = {}
    references: dict[str, list[dict[str, Any]]] = {}
    for key, (F, intensity) in CONDITIONS.items():
        refs = main_readouts(F, intensity)
        controls_eta = main_readouts(F, intensity, eta=CONTROL_ETA, step=CONTROL_STEP)
        if len(refs) != 4 or len(controls_eta) != 4:
            condition_summary[key] = {"status": "INCOMPLETE", "reason": "primary or eta-control readouts missing"}
            continue
        references[key] = refs
        ref_mean = mean_main(refs)
        eta_mean = mean_main(controls_eta)
        eta_change = abs(relative(ref_mean, eta_mean))
        components: dict[str, float] = {"eta": eta_change}
        cep0_ref = refs[0]
        cep0_runs = phase_run_ids(F, intensity)
        grid = load_readout(cep0_runs[0], "grid_step_eta005", eta=READOUT_ETA, step=0.00625)
        if grid:
            components["energy_grid"] = abs(relative(float(cep0_ref["Y_at_E_main"]), float(grid["Y_at_E_main"])))
        else:
            components["energy_grid"] = math.nan

        # dx and dt: direct only where the saved-state control exists.
        suffix_prefix = "bias_v2_fix" if intensity == 4.0 and F == 0.0 else "bias_p1" if intensity == 1.0 else "bias_p2"
        if F == 0.0 and intensity == 4.0:
            dx_id = "bias_v2_fix_F0_I4_dx048_r01"
            dt_id = "bias_v2_fix_F0_I4_dt005_r01"
            dx_kind = dt_kind = "new_repair_control"
        elif F == 0.0:
            dx_id, dt_id = "bias_p1_F0.000_I1_phi000_dx048_r02", "bias_p1_F0.000_I1_phi000_dt005_r02"
            dx_kind = dt_kind = "numerical_control"
        elif intensity == 1.0:
            dx_id, dt_id = "bias_p1_F1.000_I1_phi000_dx048_r02", "bias_p1_F1.000_I1_phi000_dt005_r02"
            dx_kind = dt_kind = "numerical_control"
        else:
            dx_id, dt_id = "bias_p2_F1.000_I4_phi000_dx048_r02", "bias_p2_F1.000_I4_phi000_dt005_r02"
            dx_kind = dt_kind = "numerical_control"
        dx = load_readout(dx_id, dx_kind, eta=READOUT_ETA, step=READOUT_STEP)
        dt = load_readout(dt_id, dt_kind, eta=READOUT_ETA, step=READOUT_STEP)
        components["dx"] = add_component(rows, key, "dx", cep0_ref, dx, "direct" if dx else "missing")
        components["dt"] = add_component(rows, key, "dt", cep0_ref, dt, "direct" if dt else "missing")

        # Box and CAP: direct for F0/mu, F1/mu, F1/4mu; F0/4mu uses the allowed maximum transfer.
        box_values, cap_values = [], []
        for probe_F, probe_I, paths in [(0.0, 1.0, ("bias_p1_F0.000_I1_phi000_box900_r02", "bias_p1_F0.000_I1_phi000_caphalf_r02")),
                                        (1.0, 1.0, ("bias_p1_F1.000_I1_phi000_box900_r02", "bias_p1_F1.000_I1_phi000_caphalf_r02")),
                                        (1.0, 4.0, ("bias_p2_F1.000_I4_phi000_box900_r02", "bias_p2_F1.000_I4_phi000_caphalf_r02"))]:
            probe_ref = main_readouts(probe_F, probe_I)[0]
            for values, run_id in [(box_values, paths[0]), (cap_values, paths[1])]:
                item = load_readout(run_id, "numerical_control", eta=READOUT_ETA, step=READOUT_STEP)
                if item:
                    values.append(abs(relative(float(probe_ref["Y_at_E_main"]), float(item["Y_at_E_main"]))))
        components["box"] = max(box_values) if box_values else math.nan
        components["cap"] = max(cap_values) if cap_values else math.nan
        rows.append({"condition": key, "component": "box", "reference_id": "mixed", "control_id": "mixed",
                     "reference_Y": "", "control_Y": "", "relative_change": "", "abs_relative": f"{components['box']:.17g}",
                     "coverage": "direct/transferred", "note": "F0/4mu transferred as allowed by plan"})
        rows.append({"condition": key, "component": "cap", "reference_id": "mixed", "control_id": "mixed",
                     "reference_Y": "", "control_Y": "", "relative_change": "", "abs_relative": f"{components['cap']:.17g}",
                     "coverage": "direct/transferred", "note": "F0/4mu transferred as allowed by plan"})

        # Mask and 125 fs: direct at mu, transferred to 4mu.
        mask_cap_values, time_values = [], []
        for probe_F in [0.0, 1.0]:
            base = main_readouts(probe_F, 1.0)[0]
            mask = load_readout(phase_run_ids(probe_F, 1.0)[0], "mask_control", eta=READOUT_ETA,
                                step=READOUT_STEP, mask=(3.0, 5.0))
            late = load_readout(phase_run_ids(probe_F, 1.0)[0], "late_time", eta=READOUT_ETA,
                                step=READOUT_STEP, readout_fs=125.0)
            if mask:
                mask_cap_values.append(abs(relative(float(base["Y_at_E_main"]), float(mask["Y_at_E_main"]))))
            if late:
                time_values.append(abs(relative(float(base["Y_at_E_main"]), float(late["Y_at_E_main"]))))
        components["mask"] = max(mask_cap_values) if mask_cap_values else math.nan
        components["time"] = max(time_values) if time_values else math.nan
        rows.append({"condition": key, "component": "mask", "reference_id": "F0/F1 mu", "control_id": "mixed",
                     "reference_Y": "", "control_Y": "", "relative_change": "", "abs_relative": f"{components['mask']:.17g}",
                     "coverage": "transferred_to_4mu" if intensity == 4.0 else "direct", "note": ""})
        rows.append({"condition": key, "component": "time", "reference_id": "F0/F1 mu", "control_id": "mixed",
                     "reference_Y": "", "control_Y": "", "relative_change": "", "abs_relative": f"{components['time']:.17g}",
                     "coverage": "transferred_to_4mu" if intensity == 4.0 else "direct", "note": ""})

        # Preparation: only F1 has non-default preparation controls.
        if F == 1.0:
            base = references["F1_I1"][0] if "F1_I1" in references else main_readouts(1.0, 1.0)[0]
            prep_values = []
            for run_id in ["bias_v2_fix_F1_I1_ramp080_r01", "bias_v2_fix_F1_I1_wait040_r01"]:
                item = load_readout(run_id, "new_repair_control", eta=READOUT_ETA, step=READOUT_STEP)
                if item:
                    prep_values.append(abs(relative(float(base["Y_at_E_main"]), float(item["Y_at_E_main"]))))
            components["preparation"] = max(prep_values) if prep_values else math.nan
            prep_coverage = "direct_at_mu;transferred_to_4mu" if intensity == 4.0 else "direct_at_mu"
        else:
            components["preparation"] = 0.0
            prep_coverage = "not_applicable_DC_zero"
        rows.append({"condition": key, "component": "preparation", "reference_id": "F1 mu CEP0",
                     "control_id": "ramp080;wait040", "reference_Y": "", "control_Y": "",
                     "relative_change": "", "abs_relative": f"{components['preparation']:.17g}",
                     "coverage": prep_coverage, "note": ""})

        # CEP: 2->4 for every condition; representative 4->8 only for F1/4mu.
        two_mean = mean_main([refs[0], refs[2]])
        four_mean = mean_main(refs)
        cep_own = abs(relative(four_mean, two_mean))
        eight_change = 0.0
        if F == 1.0 and intensity == 4.0:
            extras = [load_readout(f"bias_p2_F1.000_I4_phi{degrees:03d}_r02", "half_phase",
                                   eta=READOUT_ETA, step=READOUT_STEP) for degrees in (45, 135, 225, 315)]
            if all(extras):
                eight_mean = float(np.mean([item["Y_at_E_main"] for item in refs + extras]))
                eight_change = abs(relative(four_mean, eight_mean))
        components["cep"] = max(cep_own, eight_change) if eight_change > 0 else cep_own
        rows.append({"condition": key, "component": "cep", "reference_id": "four-phase mean",
                     "control_id": "two-phase or representative eight-phase", "reference_Y": f"{four_mean:.17g}",
                     "control_Y": "", "relative_change": f"{cep_own:.17g}",
                     "abs_relative": f"{components['cep']:.17g}",
                     "coverage": "direct_2to4; representative_4to8" if eight_change > 0 else "two_to_four_only",
                     "note": "representative 4->8 available only at F1/4mu"})

        energy_range = load_readout(phase_run_ids(F, intensity)[0], "energy_range_control",
                                    eta=READOUT_ETA, step=READOUT_STEP, emax="extended")
        range_change = 0.0
        if intensity == 4.0 and energy_range:
            range_change = abs(relative(float(cep0_ref["Y_at_E_main"]), float(energy_range["Y_at_E_main"])))
        components["energy_range"] = range_change
        rows.append({"condition": key, "component": "energy_range", "reference_id": cep0_ref["readout_id"],
                     "control_id": energy_range["readout_id"] if energy_range else "",
                     "reference_Y": f"{float(cep0_ref['Y_at_E_main']):.17g}",
                     "control_Y": f"{float(energy_range['Y_at_E_main']):.17g}" if energy_range else "",
                     "relative_change": f"{range_change:.17g}", "abs_relative": f"{range_change:.17g}",
                     "coverage": "direct" if intensity == 4.0 and energy_range else "not_applicable_at_mu",
                     "note": ""})

        missing = [name for name in ["energy_grid", "dx", "dt", "box", "cap", "mask", "time", "preparation", "cep"]
                   if not math.isfinite(components[name])]
        total = float(sum(value for value in components.values() if math.isfinite(value)))
        dc = load_readout(f"bias_p1_dc_F{F:.3f}_r02", "dc_only", eta=READOUT_ETA, step=READOUT_STEP)
        dc_fraction = float(dc["Y_at_E_main"]) / ref_mean if dc and ref_mean else None
        masked_norm = max(float(item["masked_norm"]) for item in refs)
        floor = LOW_ENERGY_FLOOR_REL * masked_norm
        signal_ok = ref_mean > 10.0 * floor
        propagation = [run_summary(item["run_id"]) for item in refs]
        budget_ok = all(float(item["max_probability_budget_residual"]) <= 1e-8 for item in propagation)
        prep_ok = all(float(item.get("pre_laser_population_loss",
                                      float(item.get("left_CAP_loss_prep", 0.0))
                                      + float(item.get("right_CAP_loss_prep", 0.0)))) < 0.01
                      for item in propagation)
        status = "PASS"
        reasons = []
        if missing:
            status = "INCOMPLETE"; reasons.append("missing:" + ",".join(missing))
        if not math.isfinite(total) or total > 0.05:
            status = "FAIL"; reasons.append("total_error>5%")
        if components.get("energy_grid", math.inf) > 0.002:
            status = "FAIL"; reasons.append("grid_step>0.2%")
        if not signal_ok:
            status = "FAIL"; reasons.append("signal_below_10x_floor")
        if not budget_ok or not prep_ok:
            status = "FAIL"; reasons.append("budget_or_preparation")
        if dc_fraction is None or dc_fraction >= 0.01:
            status = "FAIL"; reasons.append("DC_background")
        condition_summary[key] = {
            "F_dc_V_per_nm": F, "I_over_mu": intensity, "Y_four_phase": ref_mean,
            "Y_eta_control": eta_mean, "components": components,
            "estimated_total_rel_error": total, "signal_floor": floor, "signal_over_floor": ref_mean / floor,
            "dc_only_fraction": dc_fraction, "probability_budget_ok": budget_ok,
            "preparation_ok": prep_ok, "missing": missing, "status": status, "reasons": reasons,
        }
    all_pass = len(condition_summary) == 4 and all(
        value.get("status") == "PASS" for value in condition_summary.values())
    return {"schema": "continuation-v2-R1-gate", "revision": REVISION,
            "thresholds_eV": list(THRESHOLDS_EV), "conditions": condition_summary,
            "error_budget_rows": rows, "all_conditions_pass": all_pass,
            "status": "PASS" if all_pass else "INCOMPLETE" if any(
                value.get("status") == "INCOMPLETE" for value in condition_summary.values()) else "FAIL",
            "note": "Conservative sum of absolute measured changes, not a confidence interval; transferred components are labelled."}


def write_gate(gate: dict[str, Any]) -> None:
    """Write versioned gate JSON and the human-auditable error-budget CSV."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "R1_gate_r1_20261105_v1.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (REPORT_DIR / "R1_error_budget.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "component", "reference_id", "control_id",
                                                   "reference_Y", "control_Y", "relative_change", "abs_relative",
                                                   "coverage", "note"])
        writer.writeheader()
        writer.writerows(gate["error_budget_rows"])


def main() -> int:
    """CLI for gate evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    gate = compute_gate()
    print(json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=False))
    if args.write:
        write_gate(gate)
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

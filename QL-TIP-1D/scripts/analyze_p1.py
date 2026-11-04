"""Analyse P1 evidence and write the explicit P2 gate decision."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

ROOT = BASE / "results/bias_v1"


def load(run_id: str) -> dict[str, Any]:
    """Read a completed run summary."""
    path = ROOT / run_id / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json_load(path)


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def spectrum(summary: dict[str, Any], label: str) -> dict[str, Any]:
    """Select one saved spectrum descriptor by its exact label."""
    for item in summary["spectra"]:
        if item["label"] == label:
            return item
    raise KeyError(label)


def first_spectrum(summary: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    """Select the first available label without replacing saved spectra."""
    for label in labels:
        try:
            return spectrum(summary, label)
        except KeyError:
            continue
    raise KeyError(labels)


def relative_difference(reference: float, test: float) -> float | None:
    """Return a signed relative difference, or None when the reference vanishes."""
    if reference == 0.0:
        return None
    return (test - reference) / reference


def main() -> int:
    """Evaluate base DC/laser points and all P1 controls against the taskbook."""
    method = json_load(ROOT / "spectrum_method_validation.json")
    prep = {f: load(f"bias_p1_prep_F{f:.3f}_r02") for f in [0.0, 1.0]}
    dc = {f: load(f"bias_p1_dc_F{f:.3f}_r02") for f in [0.0, 1.0]}
    laser = {f: load(f"bias_p1_F{f:.3f}_I1_phi000_r02") for f in [0.0, 1.0]}
    base_label = "t100_eta0p1_dE0p025_mask2to4"
    checks: list[dict[str, Any]] = []
    rows: list[list[Any]] = []

    def add(name: str, passed: bool, details: dict[str, Any]) -> None:
        checks.append({"name": name, "pass": bool(passed), **details})

    add("method_validation", method["status"] == "PASS", {"status": method["status"]})

    for F in [0.0, 1.0]:
        preparation = prep[F]
        add(f"prep_budget_F{F:g}", preparation["max_probability_budget_residual"] <= 1e-8,
            {"residual": preparation["max_probability_budget_residual"]})
        add(f"prep_population_loss_F{F:g}", preparation["pre_laser_population_loss"] < 0.01,
            {"loss": preparation["pre_laser_population_loss"]})
        base = spectrum(laser[F], base_label)
        dc_base = spectrum(dc[F], base_label)
        dc_fraction = dc_base["Y_H_main"] / base["Y_H_main"] if base["Y_H_main"] > 0.0 else None
        add(f"dc_only_background_F{F:g}", dc_fraction is not None and dc_fraction < 0.01,
            {"dc_Y_H": dc_base["Y_H_main"], "laser_Y_H": base["Y_H_main"], "fraction": dc_fraction})
        right_cap_fraction = laser[F]["right_CAP_loss_observation_100fs"] / base["Y_H_main"] if base["Y_H_main"] > 0.0 else None
        add(f"right_CAP_F{F:g}", right_cap_fraction is not None and right_cap_fraction < 1e-4,
            {"right_CAP_loss": laser[F]["right_CAP_loss_observation_100fs"], "fraction": right_cap_fraction})
        integral_difference = abs(base["spectrum_integral"] - base["masked_norm"]) / base["masked_norm"]
        add(f"spectrum_integral_vs_mask_F{F:g}", integral_difference < 0.05,
            {"relative_difference": integral_difference, "integral": base["spectrum_integral"],
             "mask_norm": base["masked_norm"]})
        eta_start = spectrum(laser[F], "t100_eta0p2_dE0p05_mask2to4")
        mask = first_spectrum(laser[F], ["t100_eta0p1_dE0p025_mask3to5",
                                        "t100_eta0p2_dE0p05_mask3to5"])
        late = first_spectrum(laser[F], ["t125_eta0p1_dE0p025_mask2to4",
                                        "t125_eta0p2_dE0p05_mask2to4"])
        eta_start_change = relative_difference(base["Y_H_main"], eta_start["Y_H_main"])
        # Compare mask and time only against a spectrum with the same eta;
        # otherwise a deliberately changed resolution would masquerade as a
        # mask/time failure.
        mask_reference = eta_start if "eta0p2" in mask["label"] else base
        late_reference = eta_start if "eta0p2" in late["label"] else base
        mask_change = relative_difference(mask_reference["Y_H_main"], mask["Y_H_main"])
        late_change = relative_difference(late_reference["Y_H_main"], late["Y_H_main"])
        eta05_path = ROOT / f"bias_p1_F{F:.3f}_I1_phi000_r02/eta_control_t100_eta0p05_dE0p025_mask2to4.json"
        eta05 = json_load(eta05_path) if eta05_path.exists() else None
        eta_residual = relative_difference(base["Y_H_main"], eta05["Y_H_main"]) if eta05 else None
        add(f"eta_convergence_F{F:g}", eta_residual is not None and abs(eta_residual) < 0.05,
            {"eta0p1_Y_H": base["Y_H_main"], "eta0p05_Y_H": eta05["Y_H_main"] if eta05 else None,
             "relative_change": eta_residual, "eta0p2_relative_change_from_eta0p1": eta_start_change})
        add(f"mask_readout_F{F:g}",
            all(value is not None and abs(value) < 0.05 for value in [mask_change, late_change]),
            {"mask_relative_change": mask_change, "t125_relative_change": late_change,
             "mask_Y_H": mask["Y_H_main"], "t125_Y_H": late["Y_H_main"]})
        rows.append([f"base_F{F:g}", "bias_p1", f"bias_p1_F{F:.3f}_I1_phi000_r02", "eta=.1,dE=.025",
                     base["Y_H_main"], 0.0, 0.0, "reference"])
        rows.append([f"eta_start_F{F:g}", "bias_p1", f"bias_p1_F{F:.3f}_I1_phi000_r02", "eta=.2,dE=.05",
                     eta_start["Y_H_main"], eta_start["Y_H_main"] - base["Y_H_main"], eta_start_change,
                     "rejected starting resolution"])
        rows.append([f"eta05_F{F:g}", "bias_p1", f"bias_p1_F{F:.3f}_I1_phi000_r02", "eta=.05,dE=.025",
                     eta05["Y_H_main"] if eta05 else None,
                     eta05["Y_H_main"] - base["Y_H_main"] if eta05 else None, eta_residual,
                     "PASS" if eta_residual is not None and abs(eta_residual) < 0.05 else "FAIL"])
        rows.append([f"mask_F{F:g}", "bias_p1", f"bias_p1_F{F:.3f}_I1_phi000_r02", "mask=3-5nm",
                     mask["Y_H_main"], mask["Y_H_main"] - mask_reference["Y_H_main"], mask_change,
                     "PASS" if mask_change is not None and abs(mask_change) < 0.05 else "FAIL"])
        rows.append([f"time_F{F:g}", "bias_p1", f"bias_p1_F{F:.3f}_I1_phi000_r02", "t=125fs",
                     late["Y_H_main"], late["Y_H_main"] - late_reference["Y_H_main"], late_change,
                     "PASS" if late_change is not None and abs(late_change) < 0.05 else "FAIL"])

    reference_laser_y = spectrum(laser[1.0], base_label)["Y_H_main"]
    for label, ramp, wait in [("ramp080", 80.0, 20.0), ("wait040", 40.0, 40.0)]:
        control = load(f"bias_p1_dc_F1.000_{label}_r02")
        control_spectrum = spectrum(control, base_label)
        reference_spectrum = spectrum(dc[1.0], base_label)
        # DC-only is many orders below the laser signal.  Comparing two
        # near-zero DC yields as a ratio would be numerically meaningless;
        # use the absolute DC change divided by the same-F laser Y_H.
        absolute_change = control_spectrum["Y_H_main"] - reference_spectrum["Y_H_main"]
        change = absolute_change / reference_laser_y
        add(f"preparation_{label}", abs(change) < 0.05,
            {"reference_DC_Y_H": reference_spectrum["Y_H_main"], "control_DC_Y_H": control_spectrum["Y_H_main"],
             "absolute_change": absolute_change, "change_relative_to_laser_Y_H": change,
             "same_F_laser_Y_H": reference_laser_y})
        rows.append([f"prep_{label}", "bias_p1", f"bias_p1_dc_F1.000_{label}_r02",
                     f"ramp={ramp:g}fs,wait={wait:g}fs", control_spectrum["Y_H_main"],
                     absolute_change, change, "PASS" if abs(change) < 0.05 else "FAIL"])

    control_specs = [
        ("dx048", {"well_cells": 48}), ("dt005", {"dt_au": 0.05}),
        ("box900", {"xmax_nm": 900.0}), ("caphalf", {"cap_strength_au": 0.5}),
    ]
    for F in [0.0, 1.0]:
        reference = spectrum(laser[F], base_label)
        for label, _ in control_specs:
            run = load(f"bias_p1_F{F:.3f}_I1_phi000_{label}_r02")
            test = spectrum(run, base_label)
            change = relative_difference(reference["Y_H_main"], test["Y_H_main"])
            passed = change is not None and abs(change) < 0.05
            add(f"{label}_F{F:g}", passed, {"reference_Y_H": reference["Y_H_main"], "test_Y_H": test["Y_H_main"],
                                           "relative_change": change})
            rows.append([f"{label}_F{F:g}", "bias_p1", f"bias_p1_F{F:.3f}_I1_phi000_{label}_r02", label,
                         test["Y_H_main"], test["Y_H_main"] - reference["Y_H_main"], change,
                         "PASS" if passed else "FAIL"])

    status = "PASS" if all(item["pass"] for item in checks) else "FAIL"
    elapsed = sum(load_summary_wall(run_id) for run_id in all_passing_run_ids())
    gate = {"status": status, "checks": checks,
            "completed_tdse_runs": len(all_passing_run_ids()),
            "cumulative_completed_wall_seconds": elapsed,
            "note": "P1 gate permits P2 only; no BSV, thermal, four-bias, server or remote-push work is authorised."}
    # P1_gate.json and convergence.csv are derived audit outputs, not raw
    # scientific runs; they are regenerated as controls are added.
    (ROOT / "P1_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                       encoding="utf-8")
    with (ROOT / "convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["control_id", "reference_run", "test_run", "changed_parameter", "Y_ref", "Y_test",
                         "absolute_difference", "relative_difference", "pass_or_fail"])
        writer.writerows(rows)
    print(gate["status"])
    for item in checks:
        print(item)
    return 0 if status == "PASS" else 2


def load_summary_wall(run_id: str) -> float:
    """Return the wall time of a completed summary without counting missing runs."""
    path = ROOT / run_id / "summary.json"
    if not path.exists():
        return 0.0
    return float(json_load(path).get("wall_seconds", 0.0))


def all_passing_run_ids() -> list[str]:
    """List only summary-bearing runs; failures remain separate evidence."""
    ids = []
    if not ROOT.exists():
        return ids
    for directory in sorted(ROOT.iterdir()):
        summary = directory / "summary.json"
        if summary.is_file():
            ids.append(directory.name)
    return ids


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyse the two-bias/two-intensity pilot and phase-averaging convergence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "results/bias_v1"


def load_y(run_id: str, label: str = "t100_eta0p1_dE0p025_mask2to4") -> float:
    """Read one phase point's main high-energy yield."""
    summary = json.loads((ROOT / run_id / "summary.json").read_text(encoding="utf-8"))
    for item in summary["spectra"]:
        if item["label"] == label:
            return float(item["Y_H_main"])
    raise KeyError((run_id, label))


def run_id(F: float, intensity: float, phase_index: int) -> str:
    """Map a P2 phase point to its saved run identifier."""
    if intensity == 1.0 and phase_index == 0:
        return f"bias_p1_F{F:.3f}_I1_phi000_r02"
    return f"bias_p2_F{F:.3f}_I{intensity:g}_phi{phase_index * 90:03d}_r02"


def main() -> int:
    """Write phase, q and numerical-control tables; never invent missing phases."""
    phases = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    results: dict[str, dict[str, object]] = {}
    rows: list[list[object]] = []
    phase_choice = None
    phase_choice_change = -1.0
    missing = []
    for F in [0.0, 1.0]:
        for intensity in [1.0, 4.0]:
            ys = {}
            for index, phase in enumerate(phases):
                try:
                    ys[index] = load_y(run_id(F, intensity, index))
                except FileNotFoundError:
                    missing.append(run_id(F, intensity, index))
                except KeyError:
                    missing.append(run_id(F, intensity, index))
            if len(ys) < 2:
                continue
            y2 = np.mean([ys[0], ys[2]]) if 0 in ys and 2 in ys else None
            y4 = np.mean([ys[index] for index in [0, 1, 2, 3]]) if len(ys) == 4 else None
            change_2to4 = (y4 - y2) / y4 if y4 not in (None, 0.0) and y2 is not None else None
            key = f"F{F:g}_I{intensity:g}"
            results[key] = {"F_dc_V_per_nm": F, "I_over_mu": intensity,
                            "Y_by_phase": {str(index): value for index, value in ys.items()},
                            "Y_two_phase": y2, "Y_four_phase": y4,
                            "relative_change_two_to_four": change_2to4,
                            "phase_spread_only_two": abs(ys[0] - ys[2]) if 0 in ys and 2 in ys else None}
            if change_2to4 is not None and abs(change_2to4) > phase_choice_change and y4:
                phase_choice_change = abs(change_2to4)
                phase_choice = {"F_dc_V_per_nm": F, "I_over_mu": intensity, "key": key,
                                "relative_change_two_to_four": change_2to4,
                                "reason": "largest absolute 2->4 phase-average change; selected before any eight-phase result"}
            for index, value in ys.items():
                rows.append([key, index * 90, f"{phases[index]:.17g}", value,
                             None if y4 is None else value / y4, f"phase_{index}"])
    q_rows = []
    for intensity in [1.0, 4.0]:
        low = results.get(f"F0_I{intensity:g}")
        high = results.get(f"F1_I{intensity:g}")
        if not low or not high:
            continue
        for count, field in [("two", "Y_two_phase"), ("four", "Y_four_phase")]:
            numerator, denominator = high[field], low[field]
            q = numerator / denominator if numerator is not None and denominator not in (None, 0.0) else None
            q_rows.append([intensity, count, low[field], high[field], q,
                           None if q is None else f"q_{count}_phase"])
    q_two = {row[0]: row[4] for row in q_rows if row[1] == "two"}
    q_four = {row[0]: row[4] for row in q_rows if row[1] == "four"}
    q_difference = None
    if q_four.get(1.0) and q_four.get(4.0):
        q_difference = (q_four[4.0] - q_four[1.0]) / q_four[1.0]
    # Eight-phase check at the most sensitive / highest-intensity condition.
    eight_result = None
    eight_pass = False
    eight_missing = []
    if phase_choice:
        base_key = phase_choice["key"]
        base_condition = results.get(base_key)
        extra_ys = {}
        for degrees in [45, 135, 225, 315]:
            extra_id = f"bias_p2_F{phase_choice['F_dc_V_per_nm']:.3f}_I{phase_choice['I_over_mu']:g}_phi{degrees:03d}_r02"
            try:
                extra_ys[degrees] = load_y(extra_id)
            except FileNotFoundError:
                eight_missing.append(extra_id)
        if base_condition and len(extra_ys) == 4:
            y4 = float(base_condition["Y_four_phase"])
            y8 = (sum(float(value) for value in base_condition["Y_by_phase"].values()) + sum(extra_ys.values())) / 8.0
            change = (y8 - y4) / y8 if y8 else None
            eight_result = {"F_dc_V_per_nm": phase_choice["F_dc_V_per_nm"],
                            "I_over_mu": phase_choice["I_over_mu"], "Y_four_phase": y4,
                            "Y_eight_phase": y8, "relative_change_four_to_eight": change,
                            "extra_Y_by_phase_deg": {str(k): v for k, v in extra_ys.items()},
                            "acceptance_relative": 0.02,
                            "pass": change is not None and abs(change) <= 0.02}
            eight_pass = bool(eight_result["pass"])
            if phase_choice["I_over_mu"] == 4.0:
                low = results.get("F0_I4")
                if low and low["Y_four_phase"]:
                    q_eight_4mu = y8 / float(low["Y_four_phase"])
                    q_rows.append([4.0, "eight_representative", low["Y_four_phase"], y8,
                                   q_eight_4mu, "q_eight_F1_only"])
    phase_pass = (len(results) == 4 and all(item["relative_change_two_to_four"] is not None
                                           and abs(item["relative_change_two_to_four"]) <= 0.02
                                           for item in results.values())) and eight_pass
    # Highest-intensity numerical controls are required before calling all four
    # points usable; missing files remain explicit, not zero-filled.
    highest_controls = {}
    reference = results.get("F1_I4")
    for label in ["dx048", "dt005", "box900", "caphalf"]:
        path = ROOT / f"bias_p2_F1.000_I4_phi000_{label}_r02/summary.json"
        if path.exists() and reference:
            try:
                value = load_y(path.parent.name)
                change = (value - float(reference["Y_by_phase"]["0"])) / float(reference["Y_by_phase"]["0"])
                highest_controls[label] = {"Y_H": value, "relative_change": change, "pass": abs(change) < 0.05}
            except (FileNotFoundError, KeyError):
                highest_controls[label] = {"status": "missing"}
        else:
            highest_controls[label] = {"status": "missing"}
    controls_pass = bool(highest_controls) and all(item.get("pass", False) for item in highest_controls.values())
    status = "PASS" if phase_pass and controls_pass and not missing and q_difference is not None else "INCOMPLETE"
    output = {
        "status": status,
        "conditions": results,
        "q_by_intensity": {"two_phase": q_two, "four_phase": q_four,
                           "relative_q4mu_minus_qmu": q_difference},
        "phase_acceptance_2to4_relative": 0.02,
        "phase_pass": phase_pass,
        "eight_phase_representative": eight_result,
        "eight_phase_missing_runs": eight_missing,
        "highest_intensity_controls": highest_controls,
        "highest_controls_pass": controls_pass,
        "missing_runs": missing,
        "interpretation_limit": "This is a minimal numerical response check, not a BSV or thermal average, not a D(F) result, and not evidence for or against quantum advantage.",
    }
    (ROOT / "P2_pilot.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if phase_choice:
        (ROOT / "P2_phase_choice.json").write_text(json.dumps(phase_choice, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (ROOT / "P2_phase_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "phase_index", "CEP_rad", "Y_H", "fraction_of_four_phase_mean", "source"])
        writer.writerows(rows)
    with (ROOT / "P2_q_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["I_over_mu", "phase_count", "Y_F0", "Y_F1", "q", "label"])
        writer.writerows(q_rows)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

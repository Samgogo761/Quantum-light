"""Export Origin-ready CSVs from saved readouts. READ-ONLY: no TDSE, no spectrum.

Every value comes from an existing summary/config/statistics file. Missing
fields are left empty and flagged, never inferred. Run from QL-TIP-1D/.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
READOUTS = PROJECT / "results/bias_v2/readouts"
RUNS = PROJECT / "results/bias_v2/runs"
REPORT = PROJECT / "results/bias_v2/reports/v3_r2_20261106"
AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ETA, STEP = 0.05, 0.0125
MASK = [2.0, 4.0]
THRESHOLDS = [0.5170816, 0.7170816, 0.9170816]


def jload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


CONTROL_TAGS = ("box900", "dx048", "dt005", "caphalf", "_m3-5")


def is_common(summary: dict, readout_id: str) -> bool:
    if "_emaxplus" in readout_id or "_er" in readout_id:
        return False
    # Numerical controls share eta/dE/mask with production nodes but are NOT
    # production response points; including them would corrupt CEP averages.
    if any(tag in readout_id for tag in CONTROL_TAGS):
        return False
    if abs(float(summary.get("eta_eV", -1)) - ETA) > 1e-12:
        return False
    if abs(float(summary.get("step_eV", -1)) - STEP) > 1e-12:
        return False
    if [float(v) for v in summary.get("mask_bounds_nm", [])] != MASK:
        return False
    return summary.get("execution_status") == "PASS"


def run_config(run_id: str) -> dict:
    return jload(RUNS / run_id / "config.json")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    common = []
    for directory in sorted(READOUTS.iterdir()):
        if not directory.is_dir():
            continue
        summary = jload(directory / "summary.json")
        if summary and is_common(summary, directory.name):
            common.append((directory.name, summary))
    # A. intensity response
    rows = []
    for readout_id, s in common:
        config = run_config(s["run_id"])
        numerics = config.get("numerics", {})
        rows.append({
            "run_id": s["run_id"], "readout_id": readout_id,
            "source_hash": s.get("state_sha256", ""),
            "F_V_per_nm": s.get("F_dc_V_per_nm", ""),
            "I_apex_W_cm2": config.get("I_apex_W_per_cm2", ""),
            "I_over_mu": s.get("I_over_mu", ""), "CEP_rad": s.get("CEP_rad", ""),
            "Y_H_main": s.get("Y_at_E_main", ""),
            "Y_H_threshold_low": s.get("Y_at_E_low", ""),
            "Y_H_threshold_high": s.get("Y_at_E_high", ""),
            "eta_eV": s.get("eta_eV", ""), "dE_eV": s.get("step_eV", ""),
            "mask_nm": "2-4", "save_time_fs": s.get("actual_readout_fs", ""),
            "box_nm": (float(numerics["xmax_nm"]) - float(numerics["xmin_nm"])
                       if "xmax_nm" in numerics and "xmin_nm" in numerics else ""),
            "dx_nm": "", "dt_au": numerics.get("dt_au", ""),
            "readout_status": s.get("execution_status", ""),
        })
    write_csv(OUT / "intensity_response_by_bias.csv",
              ["run_id", "readout_id", "source_hash", "F_V_per_nm", "I_apex_W_cm2",
               "I_over_mu", "CEP_rad", "Y_H_main", "Y_H_threshold_low",
               "Y_H_threshold_high", "eta_eV", "dE_eV", "mask_nm", "save_time_fs",
               "box_nm", "dx_nm", "dt_au", "readout_status"], rows)

    # B. CEP-averaged response
    grouped: dict[tuple[float, float], dict[float, dict]] = defaultdict(dict)
    for readout_id, s in common:
        grouped[(round(float(s["F_dc_V_per_nm"]), 3), float(s["I_over_mu"]))][
            round(float(s["CEP_rad"]), 6)] = s
    phase_direct = {}
    for row in jload(AUDIT / "phase_validation.json").get("rows", []):
        if row.get("status") == "PASS" and row.get("relative_4to8_change") not in (None, ""):
            phase_direct[(float(row["F"]), float(row["I_over_mu"]))] = abs(
                float(row["relative_4to8_change"]))
    pair_audit = PROJECT / "results/bias_v3_audit/r1_20261107_v5/R1_pair_audit_v5.csv"
    if pair_audit.exists():
        with pair_audit.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("pair_label", "")).endswith(":cep:4to8") and row.get("coverage") == "direct_4to8":
                    condition = row["source_condition"]
                    phase_direct[(float(condition.split("_I")[0][1:]),
                                  float(condition.split("_I")[1]))] = abs(
                        float(row["relative_estimate"]))
    cep_source = {}
    node_table = AUDIT / "node_error_table.csv"
    if node_table.exists():
        with node_table.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["component"] == "cep":
                    cep_source[(float(row["F"]), float(row["I_over_mu"]))] = (
                        f'{row["evidence_type"]} | {row["source"]}')
    cep_rows = []
    for (F, I), phases in sorted(grouped.items()):
        if I == 0.0 or len(phases) < 4:
            continue
        values = [float(s["Y_at_E_main"]) for s in phases.values()]
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / mean if mean else ""
        key = (F, I)
        cep_rows.append({
            "F_V_per_nm": F, "I_over_mu": I, "Y_H_mean": f"{mean:.12g}",
            "Y_H_CEP_spread": f"{spread:.12g}" if spread != "" else "",
            "n_CEP": len(values),
            "direct_4to8_check": f"{phase_direct[key]:.12g}" if key in phase_direct else "",
            "CEP_error_source": cep_source.get(key, "envelope_2to4 | not directly measured"),
        })
    write_csv(OUT / "cep_averaged_response.csv",
              ["F_V_per_nm", "I_over_mu", "Y_H_mean", "Y_H_CEP_spread", "n_CEP",
               "direct_4to8_check", "CEP_error_source"], cep_rows)

    # C. bias ratio
    means = {(float(r["F_V_per_nm"]), float(r["I_over_mu"])): float(r["Y_H_mean"])
             for r in cep_rows}
    node_error = {}
    if node_table.exists():
        with node_table.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (float(row["F"]), float(row["I_over_mu"]))
                node_error[key] = node_error.get(key, 0.0) + abs(float(row["value"]))
    ratio_rows = []
    for I in sorted({I for (F, I) in means if F == 0.0}):
        if (1.0, I) not in means:
            continue
        y0, y1 = means[(0.0, I)], means[(1.0, I)]
        e0 = node_error.get((0.0, I), math.nan)
        e1 = node_error.get((1.0, I), math.nan)
        ratio_rows.append({
            "I_over_mu": I, "Y_F0": f"{y0:.12g}", "Y_F1": f"{y1:.12g}",
            "q_bias_ratio": f"{y1 / y0:.12g}" if y0 else "",
            "propagated_error": (f"{e0 + e1:.12g}" if not math.isnan(e0 + e1) else ""),
            "validity_flag": ("complete_4cep" if all(
                (F, I) in means for F in (0.0, 1.0)) else "missing_phase"),
        })
    write_csv(OUT / "bias_ratio_by_intensity.csv",
              ["I_over_mu", "Y_F0", "Y_F1", "q_bias_ratio", "propagated_error",
               "validity_flag"], ratio_rows)

    # D. statistical contributions
    source = REPORT / "intensity_contributions.csv"
    contribution_rows = list(csv.DictReader(source.open(encoding="utf-8", newline=""))) if source.exists() else []
    for state, name in [("bsv", "statistics_contribution_BSV.csv"),
                        ("thermal", "statistics_contribution_thermal.csv")]:
        totals = defaultdict(float)
        for row in contribution_rows:
            if row["state"] == state:
                totals[row["F"]] += float(row["weighted_contribution"])
        out_rows = []
        for row in contribution_rows:
            if row["state"] != state:
                continue
            value = float(row["weighted_contribution"])
            out_rows.append({
                "F_dc_V_per_nm": row["F"],
                "interval_left": row["interval_left"], "interval_right": row["interval_right"],
                "weighted_contribution": row["weighted_contribution"],
                "normalized_share": f"{value / totals[row['F']]:.12g}" if totals[row["F"]] else "",
                "interpolation": "piecewise_linear_nonnegative",
                "node_source": "measured_nodes",
                "contains_midpoint": ("yes" if float(row["interval_left"]) in (4.0, 8.0) and
                                      float(row["interval_right"]) in (6.0, 10.0) else "no"),
            })
        write_csv(OUT / name, ["F_dc_V_per_nm", "interval_left", "interval_right",
                               "weighted_contribution", "normalized_share", "interpolation",
                               "node_source", "contains_midpoint"], out_rows)

    # E. threshold sensitivity
    stats = jload(REPORT / "R3_statistics.json")
    classical = {}
    response_csv = REPORT / "classical_response.csv"
    if response_csv.exists():
        with response_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                classical[(float(row["F_dc_V_per_nm"]), float(row["I_over_mu"]))] = row
    threshold_rows = []
    for key, item in stats.get("yields", {}).items():
        state, F = item["state"], float(item["F"])
        weights = item.get("weights", [])
        nodes = stats.get("nodes", [])
        for label, field in [("low", "Y_low"), ("main", "Y_main"), ("high", "Y_high")]:
            total = 0.0
            ok = True
            for node, weight in zip(nodes, weights):
                row = classical.get((F, float(node)))
                if row is None or row.get(field) in ("", None):
                    ok = False
                    break
                total += float(weight) * float(row[field])
            threshold_rows.append({
                "state": state, "F_dc_V_per_nm": F, "threshold": label,
                "value": f"{total:.12g}" if ok else "",
                "error_status": ("gate_evaluated_main_only" if label == "main" and ok
                                 else "not_separately_evaluated" if ok else "missing_node_value"),
            })
    write_csv(OUT / "threshold_sensitivity.csv",
              ["state", "F_dc_V_per_nm", "threshold", "value", "error_status"], threshold_rows)

    # F. representative spectra
    spectra_dir = OUT / "representative_spectra"
    spectra_dir.mkdir(exist_ok=True)
    index = []
    for F in (0.0, 1.0):
        for I in (1.0, 4.0, 8.0, 32.0):
            candidates = [(rid, s) for rid, s in common
                          if abs(float(s["F_dc_V_per_nm"]) - F) < 1e-9
                          and abs(float(s["I_over_mu"]) - I) < 1e-9
                          and abs(float(s["CEP_rad"])) < 1e-12]
            if not candidates:
                continue
            rid, s = candidates[0]
            src = READOUTS / rid / "spectrum.csv"
            if not src.exists():
                continue
            dst = spectra_dir / f"F{F:g}_I{I:g}_phi000.csv"
            shutil.copyfile(src, dst)
            # Archive objects are capped at 2 MiB; split by data rows, keeping
            # the header on every part, so no spectrum point is dropped.
            parts: list[str] = []
            if dst.stat().st_size > 2 * 1024 * 1024:
                lines = dst.read_text(encoding="utf-8").splitlines(keepends=True)
                header, body = lines[0], lines[1:]
                mid = len(body) // 2
                for number, chunk in enumerate([body[:mid], body[mid:]], start=1):
                    part = spectra_dir / f"{dst.stem}_part{number}.csv"
                    part.write_text(header + "".join(chunk), encoding="utf-8")
                    parts.append(part.name)
                dst.unlink()
            index.append({"file": ";".join(parts) if parts else dst.name,
                          "readout_id": rid, "F_dc_V_per_nm": F,
                          "I_over_mu": I, "eta_eV": s["eta_eV"], "dE_eV": s["step_eV"],
                          "mask_nm": "2-4", "save_time_fs": s["actual_readout_fs"],
                          "split_parts": len(parts) if parts else 1})
    write_csv(spectra_dir / "_index.csv",
              ["file", "readout_id", "F_dc_V_per_nm", "I_over_mu", "eta_eV", "dE_eV",
               "mask_nm", "save_time_fs", "split_parts"], index)
    print(json.dumps({"common_readouts": len(common), "intensity_rows": len(rows),
                      "cep_rows": len(cep_rows), "ratio_rows": len(ratio_rows),
                      "contribution_rows": len(contribution_rows),
                      "threshold_rows": len(threshold_rows),
                      "spectra": len(index)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

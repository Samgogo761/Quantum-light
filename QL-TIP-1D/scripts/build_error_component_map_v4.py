"""Build the per-category numerical error map used by the final statistics.

External review (2026-11-07) required that CEP/phase error be counted exactly
once, that the tiny 4 mu phase change must not be transferred over the whole
high-intensity region, and that unmeasured conditions be labelled as coverage
gaps instead of silently defaulting to zero.

Rules implemented here:

* At 32 mu the seven directly measured R2 controls supply
  dx/dt/box/cap/eta/energy_grid/energy_range.  R1 supplies only the four
  categories not directly measured there (preparation, mask, readout time,
  CEP).  No complete R1 total is added on top of the R2 controls.
* CEP is the ONLY channel for phase error.  The outer statistical delta never
  adds a separate phase term.  Low-intensity nodes use their own measured
  2->4 change; mid/high nodes use the largest saved 2->4 change among the
  8/12/20 mu production candidates, replaced by direct 4->8 evidence once
  phase_validation.json exists.  Unresolved coverage is recorded, not zeroed.
* Once S4 supplies matched dx/dt controls at representative J, those replace
  the provisional 32 mu transfer for the mid region; the 4 mu value is kept
  as a floor.  A dt=.025 residual is never transferred onto coarse-dt nodes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "results/bias_v3_audit/reallocation_20261106"
ACTIVE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"
R1_COMPONENTS = PROJECT / "results/bias_v3_audit/r1_20261107_v5/R1_components_v5.csv"
PHASE_VALIDATION = AUDIT / "phase_validation.json"
READOUTS = PROJECT / "results/bias_v2/readouts"
FIELDS = ["F", "region", "component", "relative_estimate", "coverage",
          "source_gate", "source_key", "transfer_rule", "coverage_gap"]

CATEGORIES = ["dx", "dt", "eta", "energy_grid", "energy_range", "box", "cap",
              "mask", "time", "preparation", "cep"]
R2_DIRECT = {
    "dx": ("controls", "dx048", "matched CEP0 dx48 at 32 mu"),
    "dt": ("controls", "dt005", "matched CEP0 dt=.05 au at 32 mu"),
    "box": ("controls", "box900", "900 nm box at 32 mu"),
    "cap": ("controls", "caphalf", "half-strength CAP at 32 mu"),
    "eta": ("eta32_relative", None, "common eta=.05 vs eta=.025 at 32 mu"),
    "energy_grid": ("grid32_relative", None, "fixed eta fine energy grid at 32 mu"),
    "energy_range": ("range32_relative", None, "extended energy range at 32 mu"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r1_values() -> dict[tuple[str, str], tuple[float, str]]:
    """Return {(F, condition/component): (value, coverage)} from the v5 table."""
    result: dict[tuple[str, str], tuple[float, str]] = {}
    with R1_COMPONENTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition"]
            F = "1" if condition.startswith("F1") else "0"
            suffix = condition.split("_I")[1]
            if row["missing_reason"]:
                raise RuntimeError(f"R1 unresolved: {condition}/{row['component']}")
            result[(F, f"{suffix}/{row['component']}")] = (
                abs(float(row["relative_estimate"])), row["coverage"])
    return result


def base_yield(run_id: str) -> float | None:
    path = READOUTS / f"ro2_{run_id}_t100_eta0p05_dE0p0125_m2-4" / "summary.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["Y_at_E_main"])


def four_phase_ids(F: str, intensity: float) -> list[str]:
    degrees = [0, 90, 180, 270]
    if intensity == 8.0:
        return ["bias_v2_probe_F%s_I8_phi000_r01" % F] + [
            f"bias_v2_lib_F{F}_I8_phi{d:03d}_r01" for d in degrees[1:]]
    return [f"bias_v2_lib_F{F}_I{intensity:g}_phi{d:03d}_r01" for d in degrees]


def phase_envelopes() -> tuple[float | None, dict[str, float], list[str]]:
    """Return saved 2->4 envelope, direct 4->8 evidence and missing labels."""
    values, missing = [], []
    active = json.loads(ACTIVE.read_text(encoding="utf-8")) if ACTIVE.exists() else {}
    extension = [float(value) for value in active.get("tail_extension_nodes", [])]
    for intensity in [8.0, 12.0, 20.0] + extension:
        for F in ["0", "1"]:
            ys = [base_yield(run_id) for run_id in four_phase_ids(F, intensity)]
            if any(value is None for value in ys):
                missing.append(f"F{F}_I{intensity:g}_2to4_incomplete")
                continue
            y2 = (ys[0] + ys[2]) / 2.0
            y4 = sum(ys) / 4.0
            values.append(abs(y4 - y2) / y4 if y4 else 0.0)
    envelope = max(values) if values else None
    direct: dict[str, float] = {}
    if PHASE_VALIDATION.exists():
        validation = json.loads(PHASE_VALIDATION.read_text(encoding="utf-8"))
        for row in validation.get("rows", []):
            if row.get("status") != "PASS" or row.get("relative_4to8_change") in (None, ""):
                continue
            # Keys must match the "0"/"1" F labels used by cep_value().
            key = f"{float(row['F']):g}"
            direct[key] = max(direct.get(key, 0.0), abs(float(row["relative_4to8_change"])))
    return envelope, direct, missing


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    active = json.loads(ACTIVE.read_text(encoding="utf-8"))
    r1_gate_path = PROJECT / active["r1_gate"]
    r2_gate_path = PROJECT / active["r2_gate"]
    if sha256(r1_gate_path) != active["r1_sha256"] or sha256(r2_gate_path) != active["r2_sha256"]:
        raise RuntimeError("Gate hash mismatch: rebuild the evidence pointer first")
    r1 = json.loads(r1_gate_path.read_text(encoding="utf-8"))
    r2 = json.loads(r2_gate_path.read_text(encoding="utf-8"))
    table = r1_values()
    s4_path = AUDIT / "s4_controls.json"
    s4 = json.loads(s4_path.read_text(encoding="utf-8")) if s4_path.exists() else {}
    s4_direct: dict[tuple[str, str], float] = {}
    if s4.get("complete") and s4.get("gate", {}).get("status") == "PROCEED":
        for F in ["0", "1"]:
            values = s4["per_F"][f"F{F}"]
            s4_direct[(F, "dx")] = abs(float(values["dx_relative"]))
            s4_direct[(F, "dt")] = abs(float(values["dt_relative"]))
    phase_envelope, phase_direct, phase_missing = phase_envelopes()
    rows: list[dict[str, Any]] = []

    def add(F: str, region: str, component: str, value: float, coverage: str,
            source_gate: str, source_key: str, rule: str, gap: str = "") -> None:
        rows.append({"F": F, "region": region, "component": component,
                     "relative_estimate": f"{value:.17g}", "coverage": coverage,
                     "source_gate": source_gate, "source_key": source_key,
                     "transfer_rule": rule, "coverage_gap": gap})

    for F in ["0", "1"]:
        label = f"F{F}"
        direct: dict[str, tuple[float, str, str]] = {}
        for component, (group, suffix, rule) in R2_DIRECT.items():
            if suffix is not None:
                item = r2[group][f"{label}_{suffix}"]
                value = abs(float(item["relative_change"]))
                key = f"{group}.{label}_{suffix}.relative_change"
            else:
                value = abs(float(r2[group][label]))
                key = f"{group}.{label}"
            direct[component] = (value, "direct_32mu_control", key)
        r1_mu = r1["components"][f"{label}_I1"]["components"]
        r1_four = r1["components"][f"{label}_I4"]["components"]
        r1_cep_low = float(r1_mu["cep"])
        r1_cep_four = float(r1_four["cep"])

        def cep_value(region: str) -> tuple[float, str, str, str, str]:
            """(value, coverage, gate, key, gap) for the single CEP channel."""
            if region == "low":
                return (r1_cep_low, "direct_R1_mu_2to4", "R1_gate_r1_20261107_v5",
                        f"components.{label}_I1.cep", "")
            candidates = [r1_cep_four]
            if phase_envelope is not None:
                candidates.append(phase_envelope)
            if F in phase_direct:
                candidates.append(phase_direct[F])
            value = max(candidates)
            if F in phase_direct:
                return (value, "direct_4to8_plus_saved_2to4_envelope",
                        "phase_validation.json", f"F{F}.relative_4to8_change", "")
            if phase_envelope is None:
                return (value, "UNRESOLVED_phase_coverage", "R1_gate_r1_20261107_v5",
                        f"components.{label}_I4.cep",
                        "no direct 4->8 evidence and no saved high-intensity 2->4 envelope")
            return (value, "transferred_saved_8_12_20mu_2to4_envelope",
                    "saved_readouts", "8/12/20mu_four_phase",
                    "4->8 not yet measured at these nodes; representative envelope used")

        # HIGH region: seven direct R2 categories + four R1-only categories.
        for component in CATEGORIES:
            if component == "cep":
                value, coverage, gate, key, gap = cep_value("high")
                add(F, "high", component, value, coverage, gate, key,
                    "single CEP channel; no outer phase term", gap)
            elif component in direct:
                value, coverage, key = direct[component]
                add(F, "high", component, value, coverage, "R2_gate_r2_20261106_v2", key,
                    R2_DIRECT[component][2])
            else:
                value = max(float(r1_mu[component]), float(r1_four[component]))
                add(F, "high", component, value, "transferred_max_R1_mu_4mu",
                    "R1_gate_r1_20261107_v5", f"components.{label}_I1/I4.{component}",
                    "R1 supplies only categories not directly controlled at 32 mu")

        # MID region: conservative per-category max of 4 mu and 32 mu, except
        # dx/dt replaced by matched S4 controls and CEP by the phase envelope.
        for component in CATEGORIES:
            if component == "cep":
                value, coverage, gate, key, gap = cep_value("mid")
                add(F, "mid", component, value, coverage, gate, key,
                    "single CEP channel; no outer phase term", gap)
            elif (F, component) in s4_direct:
                value = max(s4_direct[(F, component)], float(r1_four[component]))
                add(F, "mid", component, value, "direct_S4_J_with_R1_4mu_floor",
                    "s4_controls.csv", f"F{F}.{component}_relative",
                    "matched dx/dt control at S4 representative J; 32 mu transfer removed")
            else:
                four = float(r1_four[component])
                high = direct[component][0] if component in direct else 0.0
                value = max(four, high)
                source = ("R1_gate_r1_20261107_v5" if four >= high
                          else "R2_gate_r2_20261106_v2")
                add(F, "mid", component, value, "provisional_max_4mu_32mu_per_category",
                    source, f"components.{label}_I4.{component}" if four >= high else "32mu_direct",
                    "S4 replaces this provisional envelope with direct dx/dt evidence")

        # LOW region: measured mu categories, each counted once.
        for component in CATEGORIES:
            if component == "cep":
                value, coverage, gate, key, gap = cep_value("low")
                add(F, "low", component, value, coverage, gate, key,
                    "single CEP channel; no outer phase term", gap)
            else:
                add(F, "low", component, float(r1_mu[component]),
                    "direct_or_transferred_at_mu", "R1_gate_r1_20261107_v5",
                    f"components.{label}_I1.{component}",
                    "low-intensity nodes transfer the matched mu categories only")

    out = AUDIT / "error_component_map.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 2 * 3 * len(CATEGORIES), "each category once per F/region"
    active["error_component_map"] = str(out.relative_to(PROJECT)).replace("\\", "/")
    active["error_component_map_sha256"] = sha256(out)
    active["phase_coverage"] = {
        "saved_2to4_envelope": phase_envelope,
        "direct_4to8": phase_direct,
        "missing": phase_missing,
        "single_channel": "cep only; final statistics add no separate phase delta",
    }
    ACTIVE.write_text(json.dumps(active, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "path": active["error_component_map"],
                      "sha256": active["error_component_map_sha256"],
                      "phase_envelope": phase_envelope, "direct_4to8": phase_direct,
                      "phase_missing": phase_missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

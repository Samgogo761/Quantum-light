"""Rebuild R1 per-category evidence from saved readouts with correct pairing.

External review (2026-11-07) found that the v3 builder compared control yields
from one (F, I) condition against the reference yield of another condition, and
left control_Y / signed_change / SHA empty while still declaring the numerical
evidence complete.  This v5 builder:

* indexes every real readout by (run_id, eta, step, mask, time, emax);
* pairs every candidate control with ITS OWN matched reference before taking a
  transferable maximum;
* validates existence, parameters, thresholds and non-empty SHA for every
  consumed item, and refuses PASS on any missing or mismatched input;
* writes a new revision only.  It never overwrites an old gate or silently
  replaces the shared active-evidence pointer.

No TDSE and no new spectrum are computed here.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
READOUTS = PROJECT / "results/bias_v2/readouts"
OUT = PROJECT / "results/bias_v3_audit/r1_20261107_v5"
ACTIVE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/active_evidence.json"
V1_GATE = PROJECT / "results/bias_v2/reports/r1_20261105_v1/R1_gate_r1_20261105_v1.json"
V3_GATE = PROJECT / "results/bias_v3_audit/r1_20261106_v3/R1_gate_r1_20261106_v3.json"
THRESHOLDS = [0.5170816, 0.7170816, 0.9170816]
BASE = {"eta": 0.05, "step": 0.0125, "mask": (2.0, 4.0), "t": 100.0, "emax": "base"}
CONDITIONS = {"F0_I1": (0.0, 1.0), "F1_I1": (1.0, 1.0),
              "F0_I4": (0.0, 4.0), "F1_I4": (1.0, 4.0)}
CATEGORIES = ["eta", "energy_grid", "dx", "dt", "box", "cap", "mask", "time",
              "preparation", "cep", "energy_range"]


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


class Index:
    """Deterministic on-disk index of authoritative readout summaries."""

    def __init__(self) -> None:
        self.items: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.problems: list[str] = []
        for path in sorted(READOUTS.glob("*/summary.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            name = path.parent.name
            emax = "extended" if name.endswith("_emaxplus") else "base"
            key = (item["run_id"], round(float(item["eta_eV"]), 8),
                   round(float(item["step_eV"]), 8),
                   tuple(float(v) for v in item["mask_bounds_nm"]),
                   round(float(item["actual_readout_fs"]), 6), emax)
            if not item.get("state_sha256") or not item.get("run_config_sha256"):
                self.problems.append(f"{name}: empty state/config SHA")
            if any(not close(a, b) for a, b in zip(item["thresholds_eV"], THRESHOLDS)):
                self.problems.append(f"{name}: threshold mismatch")
            item["_path"] = path
            item["_readout_id"] = name
            item["_emax"] = emax
            self.items[key] = item

    def get(self, run_id: str, **kwargs: Any) -> dict[str, Any] | None:
        opt = {**BASE, **kwargs}
        key = (run_id, round(float(opt["eta"]), 8), round(float(opt["step"]), 8),
               tuple(float(v) for v in opt["mask"]), round(float(opt["t"]), 6), opt["emax"])
        return self.items.get(key)

    def identity(self, item: dict[str, Any]) -> str:
        return ";".join([item["_readout_id"], item["state_sha256"][:16],
                         item["run_config_sha256"][:16], sha256(item["_path"])[:16]])


def mean_y(items: list[dict[str, Any]]) -> float:
    return float(sum(float(item["Y_at_E_main"]) for item in items) / len(items))


def phase_runs(F: float, I: float) -> list[str]:
    if I == 1.0:
        return [f"bias_p1_F{F:.3f}_I1_phi000_r02"] + [
            f"bias_p2_F{F:.3f}_I1_phi{deg:03d}_r02" for deg in (90, 180, 270)]
    return [f"bias_p2_F{F:.3f}_I{int(I):g}_phi000_r02"] + [
        f"bias_p2_F{F:.3f}_I{int(I):g}_phi{deg:03d}_r02" for deg in (90, 180, 270)]


def dx_dt_runs(F: float, I: float) -> tuple[str, str]:
    if F == 0.0 and I == 4.0:
        return "bias_v2_fix_F0_I4_dx048_r01", "bias_v2_fix_F0_I4_dt005_r01"
    if I == 1.0:
        base = f"bias_p1_F{F:.3f}_I1_phi000"
    else:
        base = f"bias_p2_F{F:.3f}_I4_phi000"
    return f"{base}_dx048_r02", f"{base}_dt005_r02"


def box_cap_candidates() -> list[tuple[float, float, str]]:
    """(F, I, base run) triples whose own box/CAP controls exist."""
    return [(0.0, 1.0, "bias_p1_F0.000_I1_phi000_r02"),
            (1.0, 1.0, "bias_p1_F1.000_I1_phi000_r02"),
            (1.0, 4.0, "bias_p2_F1.000_I4_phi000_r02")]


class PairBuilder:
    def __init__(self, index: Index) -> None:
        self.index = index
        self.missing: list[str] = []
        self.audit_rows: list[list[Any]] = []

    def records(self, run_ids: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        out = []
        for run_id in run_ids:
            item = self.index.get(run_id, **kwargs)
            if item is None:
                self.missing.append(f"missing readout: {run_id} {kwargs}")
                continue
            opt = {**BASE, **kwargs}
            if not (close(item["eta_eV"], opt["eta"]) and close(item["step_eV"], opt["step"])
                    and close(item["actual_readout_fs"], opt["t"])):
                self.missing.append(f"parameter mismatch: {item['_readout_id']}")
            out.append(item)
        return out

    def pair(self, label: str, refs: list[str], ctrls: list[str], *,
             ref_kw: dict[str, Any] | None = None,
             ctrl_kw: dict[str, Any] | None = None,
             coverage: str = "direct", rule: str = "",
             source: str = "") -> dict[str, Any] | None:
        ref_items = self.records(refs, **(ref_kw or {}))
        ctrl_items = self.records(ctrls, **(ctrl_kw or {}))
        if not ref_items or not ctrl_items:
            return None
        ref_y, ctrl_y = mean_y(ref_items), mean_y(ctrl_items)
        if ref_y <= 0.0:
            self.missing.append(f"non-positive reference yield: {label}")
            return None
        signed = ctrl_y - ref_y
        rel = abs(signed) / ref_y
        if not math.isfinite(rel):
            self.missing.append(f"non-finite relative change: {label}")
            return None
        self.audit_rows.append([label, source, ";".join(item["_readout_id"] for item in ref_items),
                                ";".join(item["_readout_id"] for item in ctrl_items),
                                ref_y, ctrl_y, signed, rel, coverage, rule])
        return {"label": label, "reference_readout_ids": ";".join(
                    item["_readout_id"] for item in ref_items),
                "control_readout_ids": ";".join(item["_readout_id"] for item in ctrl_items),
                "reference_Y": ref_y, "control_Y": ctrl_y, "signed_change": signed,
                "abs_change": abs(signed), "relative_estimate": rel,
                "coverage": coverage, "transfer_rule": rule, "source_conditions": source,
                "state_config_summary_source_SHA": "|".join(
                    self.index.identity(item) for item in ref_items + ctrl_items)}

    def best(self, pairs: list[dict[str, Any] | None], label: str) -> dict[str, Any] | None:
        valid = [pair for pair in pairs if pair is not None]
        if not valid:
            self.missing.append(f"no usable candidate for {label}")
            return None
        chosen = max(valid, key=lambda pair: pair["relative_estimate"])
        chosen["transfer_rule"] += f"; selected_max_of_{len(valid)}_own_pairs"
        return chosen


def condition_pairs(builder: PairBuilder, F: float, I: float) -> dict[str, dict[str, Any] | None]:
    cond = f"F{F:g}_I{I:g}"
    runs = phase_runs(F, I)
    cep0 = runs[0]
    dx_run, dt_run = dx_dt_runs(F, I)
    out: dict[str, dict[str, Any] | None] = {}

    out["eta"] = builder.pair(
        f"{cond}:eta", runs, runs, ref_kw=BASE,
        ctrl_kw={"eta": 0.025, "step": 0.00625},
        coverage="direct_four_phase_mean",
        rule="eta=.05 four-phase mean vs eta=.025 four-phase mean", source=cond)
    out["energy_grid"] = builder.pair(
        f"{cond}:energy_grid", [cep0], [cep0], ctrl_kw={"step": 0.00625},
        coverage="direct_CEP0_fixed_eta",
        rule="fixed eta=.05, step 0.0125 vs 0.00625", source=cond)
    out["dx"] = builder.pair(
        f"{cond}:dx", [cep0], [dx_run], coverage="direct_CEP0",
        rule="matched CEP0 dx48 against the same condition CEP0", source=cond)
    out["dt"] = builder.pair(
        f"{cond}:dt", [cep0], [dt_run], coverage="direct_CEP0",
        rule="matched CEP0 dt=.05 au against the same condition CEP0", source=cond)

    # Box/CAP: each candidate is paired with its own condition reference first.
    for component, suffix in [("box", "box900"), ("cap", "caphalf")]:
        pairs = []
        for cand_F, cand_I, base_run in box_cap_candidates():
            ctrl_run = base_run.replace("_r02", f"_{suffix}_r02")
            pairs.append(builder.pair(
                f"{cond}:{component}:cand_F{cand_F:g}_I{cand_I:g}",
                [base_run], [ctrl_run], coverage="own_condition_pair",
                rule=f"{suffix} vs its own CEP0 reference", source=f"F{cand_F:g}_I{cand_I:g}"))
        out[component] = builder.best(pairs, f"{cond}:{component}")
        if out[component]:
            out[component]["coverage"] = "transferred_max_of_own_pairs"
            out[component]["transfer_rule"] = (
                "maximum of F0mu/F1mu/F1_4mu, each computed against its own matched reference")

    mask_kw, time_kw = {"mask": (3.0, 5.0)}, {"t": 125.0}
    if I == 1.0:
        out["mask"] = builder.pair(f"{cond}:mask", [cep0], [cep0], ctrl_kw=mask_kw,
                                   coverage="direct", rule="mask 2-4 vs 3-5 nm", source=cond)
        out["time"] = builder.pair(f"{cond}:time", [cep0], [cep0], ctrl_kw=time_kw,
                                   coverage="direct", rule="100 vs 125 fs", source=cond)
    else:
        mu_runs = ["bias_p1_F0.000_I1_phi000_r02", "bias_p1_F1.000_I1_phi000_r02"]
        for component, kw, rule in [("mask", mask_kw, "mask 2-4 vs 3-5 nm"),
                                    ("time", time_kw, "100 vs 125 fs")]:
            pairs = [builder.pair(f"{cond}:{component}:cand_{run}", [run], [run],
                                  ctrl_kw=kw, coverage="own_condition_pair", rule=rule,
                                  source=run) for run in mu_runs]
            chosen = builder.best(pairs, f"{cond}:{component}")
            if chosen:
                chosen["coverage"] = "transferred_max_mu_own_pairs"
                chosen["transfer_rule"] = "max of F0mu/F1mu, each against its own reference"
            out[component] = chosen

    if F == 1.0:
        prep_runs = ["bias_v2_fix_F1_I1_ramp080_r01", "bias_v2_fix_F1_I1_wait040_r01"]
        ref = "bias_p1_F1.000_I1_phi000_r02"
        pairs = [builder.pair(f"{cond}:preparation:cand_{run}", [ref], [run],
                              coverage="own_condition_pair", rule="preparation variant vs mu CEP0",
                              source="F1_I1") for run in prep_runs]
        chosen = builder.best(pairs, f"{cond}:preparation")
        if chosen:
            chosen["coverage"] = "direct_at_mu" if I == 1.0 else "transferred_to_4mu"
            chosen["transfer_rule"] = "max(ramp80, wait40) vs the same mu CEP0 reference"
        out["preparation"] = chosen
    else:
        out["preparation"] = {
            "label": f"{cond}:preparation", "reference_readout_ids": "", "control_readout_ids": "",
            "reference_Y": "", "control_Y": "", "signed_change": "", "abs_change": 0.0,
            "relative_estimate": 0.0, "coverage": "not_applicable_DC_zero",
            "transfer_rule": "F=0 DC preparation changes only a global phase; no control needed",
            "source_conditions": cond, "state_config_summary_source_SHA": ""}

    # CEP 2->4 from the same condition; 4->8 is available at the F1/4mu representative.
    y2_items = builder.records([runs[0], runs[2]])
    y4_items = builder.records(runs)
    if y2_items and y4_items:
        y2, y4 = mean_y(y2_items), mean_y(y4_items)
        rel24 = abs(y4 - y2) / y4 if y4 else math.inf
        builder.audit_rows.append([f"{cond}:cep:2to4", cond,
                                   ";".join(i["_readout_id"] for i in y2_items),
                                   ";".join(i["_readout_id"] for i in y4_items),
                                   y2, y4, y4 - y2, rel24, "direct_2to4", "Y2=mean(0,180)"])
        extra = [f"bias_p2_F1.000_I4_phi{deg:03d}_r02" for deg in (45, 135, 225, 315)]
        rep4 = builder.records(phase_runs(1.0, 4.0))
        rep8_items = builder.records(extra)
        rel48 = 0.0
        if rep4 and rep8_items:
            y8 = mean_y(rep4 + rep8_items)
            y4r = mean_y(rep4)
            rel48 = abs(y8 - y4r) / y8 if y8 else math.inf
            builder.audit_rows.append(["F1_I4:cep:4to8", "F1_I4",
                                       ";".join(i["_readout_id"] for i in rep4),
                                       ";".join(i["_readout_id"] for i in rep8_items),
                                       y4r, y8, y8 - y4r, rel48, "direct_4to8", "representative"])
        else:
            builder.missing.append("representative F1/4mu 4->8 phases missing for CEP transfer")
        rel = max(rel24, rel48)
        out["cep"] = {"label": f"{cond}:cep",
                      "reference_readout_ids": ";".join(i["_readout_id"] for i in y2_items),
                      "control_readout_ids": ";".join(i["_readout_id"] for i in y4_items),
                      "reference_Y": y2, "control_Y": y4, "signed_change": y4 - y2,
                      "abs_change": abs(y4 - y2), "relative_estimate": rel,
                      "coverage": "direct_2to4_with_representative_4to8",
                      "transfer_rule": f"max(own 2->4={rel24:.6g}, representative 4->8={rel48:.6g})",
                      "source_conditions": f"{cond};F1_I4",
                      "state_config_summary_source_SHA": "|".join(
                          builder.index.identity(i) for i in y2_items + y4_items)}
    else:
        out["cep"] = None
        builder.missing.append(f"{cond}: CEP phase readouts missing")

    # Energy range: 4mu has a direct extended readout; mu transfers the same-F
    # 4mu pair (never pretends a mu extended readout exists).
    if I == 4.0:
        chosen = builder.pair(f"{cond}:energy_range", [cep0], [cep0],
                              ctrl_kw={"emax": "extended"}, coverage="direct",
                              rule="extended energy range vs base at the same 4mu CEP0", source=cond)
    else:
        ref4 = phase_runs(F, 4.0)[0]
        chosen = builder.pair(f"{cond}:energy_range", [ref4], [ref4],
                              ctrl_kw={"emax": "extended"}, coverage="transferred_same_F_4mu",
                              rule="same-F 4mu extended range, paired with that 4mu reference",
                              source=f"F{F:g}_I4")
        if chosen:
            chosen["reference_readout_ids"] += " (transfer source)"
    out["energy_range"] = chosen
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = Index()
    builder = PairBuilder(index)
    builder.missing.extend(index.problems)
    rows, components = [], {}
    fields = ["condition", "component", "reference_readout_ids", "control_readout_ids",
              "reference_Y", "control_Y", "signed_change", "abs_change", "relative_estimate",
              "coverage", "transfer_rule", "source_conditions",
              "state_config_summary_source_SHA", "missing_reason"]
    for condition, (F, I) in CONDITIONS.items():
        pairs = condition_pairs(builder, F, I)
        values = {}
        for component in CATEGORIES:
            pair = pairs.get(component)
            if pair is None:
                rows.append({"condition": condition, "component": component,
                             "reference_readout_ids": "", "control_readout_ids": "",
                             "reference_Y": "", "control_Y": "", "signed_change": "",
                             "abs_change": "", "relative_estimate": "", "coverage": "INCOMPLETE",
                             "transfer_rule": "", "source_conditions": "",
                             "state_config_summary_source_SHA": "",
                             "missing_reason": "matched reference/control pair unavailable"})
                builder.missing.append(f"{condition}:{component}: no valid pair")
                continue
            values[component] = float(pair["relative_estimate"])
            rows.append({"condition": condition, "component": component,
                         **{key: pair[key] for key in fields if key not in ("condition", "component",
                                                                           "missing_reason")},
                         "missing_reason": ""})
        components[condition] = {"total": sum(values.values()), "components": values,
                                "complete_categories": len(values) == len(CATEGORIES)}
    numeric_ok = (not builder.missing
                  and all(value["total"] <= 0.05 for value in components.values())
                  and all(value["complete_categories"] for value in components.values()))
    gate = {
        "schema": "ql-tip-1d-r1-gate-v5", "revision": "r1_20261107_v5",
        "parent_v1": str(V1_GATE.relative_to(PROJECT)).replace("\\", "/"),
        "parent_v1_sha256": sha256(V1_GATE),
        "parent_v3": str(V3_GATE.relative_to(PROJECT)).replace("\\", "/"),
        "parent_v3_sha256": sha256(V3_GATE),
        "components": components, "missing": sorted(set(builder.missing)),
        "execution_ok": True, "numerical_evidence_complete": numeric_ok,
        "status": "PASS" if numeric_ok else "INCOMPLETE",
        "allowed_stage": "SCOUT_BATCHES" if numeric_ok else "NONE",
        "pairing_rule": ("Every control is first compared with its own matched reference; "
                         "only then may the largest relative change be transferred. "
                         "No v1 category value is copied."),
        "note": "Rebuilt from saved readout summaries; no TDSE was run.",
    }
    with (OUT / "R1_components_v5.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "R1_pair_audit_v5.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_label", "source_condition", "reference_readout_ids",
                         "control_readout_ids", "reference_Y", "control_Y", "signed_change",
                         "relative_estimate", "coverage", "rule"])
        writer.writerows(builder.audit_rows)
    (OUT / "R1_gate_r1_20261107_v5.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pointer = {"schema": "ql-tip-1d-r1-pointer-v5",
               "r1_gate": str((OUT / "R1_gate_r1_20261107_v5.json").relative_to(PROJECT)).replace("\\", "/"),
               "r1_sha256": sha256(OUT / "R1_gate_r1_20261107_v5.json"),
               "status": gate["status"], "numerical_evidence_complete": numeric_ok,
               "note": "Apply to active_evidence.json explicitly; this script does not edit it."}
    (OUT / "r1_v5_pointer.json").write_text(
        json.dumps(pointer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "missing_count": len(gate["missing"]),
                      "missing": gate["missing"][:20],
                      "totals": {k: round(v["total"], 8) for k, v in components.items()},
                      "pointer": pointer}, indent=2, ensure_ascii=False))
    return 0 if numeric_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

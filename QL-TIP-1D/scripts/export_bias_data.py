"""Collate saved solo-v1 runs into shared CSV tables without rerunning science."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "results/bias_v1"


def main() -> int:
    """Write manifest, classical response and Origin-ready spectra tables."""
    summaries = []
    for directory in sorted(ROOT.iterdir()):
        path = directory / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        provenance_path = directory / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
        summary["_directory"] = directory.name
        summary["_provenance"] = provenance
        summaries.append(summary)
    manifest_rows = []
    response_rows = []
    spectrum_rows = []
    for summary in summaries:
        source = summary.get("_provenance", {}).get("source_sha256", {})
        combined_source = ";".join(f"{key}={value}" for key, value in sorted(source.items()))
        label = "t100_eta0p1_dE0p025_mask2to4"
        spectra = {item["label"]: item for item in summary.get("spectra", [])}
        main = spectra.get(label)
        if main is None and spectra:
            main = next(iter(spectra.values()))
        manifest_rows.append({
            "run_id": summary["_directory"],
            "F_dc_V_per_nm": summary.get("F_dc_V_per_nm", ""),
            "I_apex_W_per_cm2": summary.get("I_apex_W_per_cm2", ""),
            "I_inc_W_per_cm2": summary.get("I_inc_W_per_cm2", ""),
            "CEP_rad": summary.get("CEP_rad", ""),
            "grid_id": f"well{summary.get('well_cells', '')}_xmax{summary.get('xmax_nm', '')}",
            "dt_au": summary.get("actual_dt_au", summary.get("actual_dt_au_main", "")),
            "readout_fs": ";".join(str(x) for x in summary.get("readout_fs", [])),
            "source_hash": combined_source,
            "status": summary.get("status", ""),
            "wall_seconds": summary.get("wall_seconds", ""),
        })
        if main is not None:
            response_rows.append({
                "run_id": summary["_directory"], "F_dc_V_per_nm": summary.get("F_dc_V_per_nm", ""),
                "I_over_mu": (summary.get("I_over_mu")
                              if summary.get("I_over_mu") is not None
                              else summary.get("I_apex_W_per_cm2", 0.0) / 1.5e12),
                "I_apex_W_per_cm2": summary.get("I_apex_W_per_cm2", ""),
                "CEP_rad": summary.get("CEP_rad", ""), "phase_average": "single_run",
                "Y_high": main["Y_H_main"], "Y_low": main["Y_H_lower"], "Y_upper": main["Y_H_upper"],
                "E_star_eV": 0.7170816000000001, "eta_eV": main["eta_eV"],
                "masked_norm": main["masked_norm"], "spectrum_integral": main["spectrum_integral"],
                "right_CAP_fraction": (
                    summary.get("right_CAP_loss_observation_100fs", 0.0) / main["Y_H_main"]
                    if main["Y_H_main"] else ""
                ),
                "validity": summary.get("status", ""), "run_kind": summary.get("run_kind", ""),
            })
            spectrum_path = ROOT / summary["_directory"] / f"spectrum_{label}.csv"
            if spectrum_path.exists():
                with spectrum_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        spectrum_rows.append([
                            row["E_corr_eV"], summary.get("F_dc_V_per_nm", ""),
                            summary.get("run_kind", ""), summary.get("I_apex_W_per_cm2", ""),
                            row["S_per_eV"], row["eta_eV"], row["K_infinity_eV"], summary["_directory"],
                        ])
    with (ROOT / "run_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]) if manifest_rows else ["run_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (ROOT / "classical_response.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_rows[0]) if response_rows else ["run_id"])
        writer.writeheader()
        writer.writerows(response_rows)
    with (ROOT / "spectra_origin.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["E_corr_eV", "F_dc_V_per_nm", "run_kind", "I_apex_W_per_cm2", "S_per_eV",
                         "eta_eV", "K_infinity_eV", "run_id"])
        writer.writerows(spectrum_rows)
    print(json.dumps({"summaries": len(summaries), "response_rows": len(response_rows),
                      "spectrum_rows": len(spectrum_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Second-order step prediction. The published 2048 file used the opposite sign.

R(h) = R0 + C h^2 gives, when the fine step is half the coarse step,

    R(h/2) = R(h) + (R(h) - R(2h)) / 4
    R0 = R(h) + (R(h) - R(2h)) / 3

The historical JSON stored R(h) - (R(h) - R(2h)) / 4. This module does not
rewrite that file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from q8_m0_audit_artifacts_20261014 import provenance, write_new_json


def predict_half_step(coarse, fine):
    coarse = np.asarray(coarse, complex)
    fine = np.asarray(fine, complex)
    return fine + (fine - coarse) / 4.0


def richardson_floor(coarse, fine):
    coarse = np.asarray(coarse, complex)
    fine = np.asarray(fine, complex)
    return fine + (fine - coarse) / 3.0


def historical_wrong_prediction(coarse, fine):
    """The sign that was written into timestep_2048.json. Kept for the regression."""
    coarse = np.asarray(coarse, complex)
    fine = np.asarray(fine, complex)
    return fine - (fine - coarse) / 4.0


def _unpack(block):
    return np.array(block["real"], float) + 1j * np.array(block["imag"], float)


def publish_correction(base, out):
    base = Path(base)
    ladder = json.loads((base / "timestep_ladder.json").read_text(encoding="utf-8"))
    stored = json.loads((base / "timestep_2048.json").read_text(encoding="utf-8"))
    names = ("against_I", "against_U0_gram", "against_mass_identity", "anomalous")
    corrected = {}
    for name in names:
        coarse = _unpack(ladder["levels"]["512"][name])
        fine = _unpack(ladder["levels"]["1024"][name])
        actual = _unpack(stored["level"][name])
        predicted = predict_half_step(coarse, fine)
        wrong = historical_wrong_prediction(coarse, fine)
        historical = _unpack(stored["comparison_with_512_1024"][name]["predicted_2048_from_512_1024"])
        if np.max(np.abs(wrong - historical)) > 1e-12:
            raise AssertionError(f"Historical file no longer matches the wrong formula: {name}")
        corrected[name] = {
            "predicted_2048": _pack(predicted),
            "actual_2048": _pack(actual),
            "prediction_minus_actual": _pack(predicted - actual),
            "historical_wrong_prediction": _pack(historical),
        }
    report = {
        "formula": "R(h/2) = R(h) + (R(h) - R(2h)) / 4",
        "historical_file_unchanged": "timestep_2048.json",
        "comparison": corrected,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(out, report)
    return report


def _pack(mat):
    mat = np.asarray(mat, complex)
    return {
        "real": np.real(mat).tolist(),
        "imag": np.imag(mat).tolist(),
        "maxabs": float(np.max(np.abs(mat))),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = root / "results/q_line/g1_two_k_timestep_20261014/timestep_2048_prediction_corrected.json"
    publish_correction(root / "results/q_line/g1_two_k_timestep_20261014", out)
    print("wrote", out.name)

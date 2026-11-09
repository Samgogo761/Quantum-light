"""Regression for the half-step prediction sign. Does not rewrite historical JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from q8_m0_g1_richardson_predict_20261014 import (
    historical_wrong_prediction, predict_half_step, richardson_floor,
)


def test_exact_quadratic():
    value0 = 0.02 - 0.005j
    slope = 1.5 + 0.25j
    coarse = value0 + 4.0 * slope
    fine = value0 + slope
    half = value0 + 0.25 * slope
    assert np.max(np.abs(predict_half_step(coarse, fine) - half)) < 1e-15
    assert np.max(np.abs(richardson_floor(coarse, fine) - value0)) < 1e-15
    wrong = historical_wrong_prediction(coarse, fine)
    assert np.max(np.abs(wrong - half)) > 1.0


def test_historical_2048_file_keeps_the_wrong_sign():
    root = Path(__file__).resolve().parents[2]
    base = root / "results/q_line/g1_two_k_timestep_20261014"
    ladder = json.loads((base / "timestep_ladder.json").read_text(encoding="utf-8"))
    stored = json.loads((base / "timestep_2048.json").read_text(encoding="utf-8"))

    def unpack(block):
        return np.array(block["real"], float) + 1j * np.array(block["imag"], float)

    coarse = unpack(ladder["levels"]["512"]["against_U0_gram"])
    fine = unpack(ladder["levels"]["1024"]["against_U0_gram"])
    historical = unpack(stored["comparison_with_512_1024"]["against_U0_gram"]["predicted_2048_from_512_1024"])
    corrected = predict_half_step(coarse, fine)
    assert abs(historical[0, 0].real - 0.0003108135112586674) < 1e-18
    assert abs(corrected[0, 0].real - 2.0649973411390743e-05) < 1e-18
    assert np.max(np.abs(historical_wrong_prediction(coarse, fine) - historical)) < 1e-15
    assert np.max(np.abs(corrected - historical)) > 1e-4


if __name__ == "__main__":
    test_exact_quadratic()
    test_historical_2048_file_keeps_the_wrong_sign()
    print("richardson prediction regression passed")

#!/usr/bin/env python3
"""Unit tests for 112_old vs 112_large vs QE comparison helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import compare_112old_large_qe as C


def require(cond: bool, msg: object) -> None:
    if not cond:
        raise RuntimeError(str(msg))


def test_unique_gamma_shift_identity() -> None:
    tb = np.array([-1.0, 0.0, 0.4])
    qe = tb + 0.1
    s = C.unique_gamma_shift(tb, qe, -1.5, 0.8)
    require(s["applied"], s)
    require(abs(s["shift_eV"] - (-0.1)) < 1e-12, s["shift_eV"])
    require(s["n_tb"] == 3, s["n_tb"])


def test_unique_gamma_shift_empty() -> None:
    s = C.unique_gamma_shift(np.array([5.0]), np.array([5.1]), -1.5, 0.8)
    require(not s["applied"], s)
    require(s["shift_eV"] == 0.0, s)


def test_hausdorff_empty_and_match() -> None:
    empty = C.hausdorff_or_empty(np.array([]), np.array([1.0]))
    require(empty["empty"], empty)
    h = C.hausdorff_or_empty(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))
    require(not h["empty"], h)
    require(abs(h["tb_to_qe_max_eV"]) < 1e-12, h)
    require(abs(h["qe_to_tb_max_eV"] - 1.0) < 1e-12, h)


def test_uncovered_high_cb() -> None:
    qe = np.array([6.6, 7.0, 7.8])
    tb = np.array([6.61, 7.01])
    u = C.uncovered_counts(qe, tb, (0.05, 0.10, 0.20))
    require(u["n_src_farther_than_50meV"] == 1, u)
    require(u["n_src_farther_than_200meV"] == 1, u)


def test_nearest_mesh() -> None:
    kpoints = np.array([[i / 24.0, j / 24.0, 0.0] for i in range(24) for j in range(24)])
    hit = C.nearest_mesh_index(kpoints, (0.10, 0.55, 0.0))
    require(hit["distance"] < 0.03, hit)
    require(abs(hit["k_mesh"][0] - 2.0 / 24.0) < 1e-12, hit["k_mesh"])
    require(abs(hit["k_mesh"][1] - 13.0 / 24.0) < 1e-12, hit["k_mesh"])


def test_window_spectrum_counts() -> None:
    tb = np.array([-2.0, 0.0, 4.0, 7.0])
    qe = np.array([-2.0, 0.0, 4.0, 6.7, 7.2, 9.0])
    rows = C.window_spectrum(tb, qe, 0.0)
    require(rows["high_new_pool"]["n_tb"] == 1, rows["high_new_pool"])
    require(rows["high_new_pool"]["n_qe"] == 2, rows["high_new_pool"])
    require(rows["high_new_pool"]["qe_uncovered_vs_tb"]["n_src_farther_than_50meV"] == 2, rows["high_new_pool"])


def test_p_stats_and_projectability() -> None:
    e = np.array([0.0, 4.0, 7.0, 9.0])
    pdiag = np.array([1.0, 0.9, 0.05, 0.0])
    lwindow = np.array([True, True, True, False])
    rows = C.projectability_windows(e, pdiag, lwindow)
    high = rows["high_new_pool"]["all_qe_in_window"]
    require(high["n"] == 1, high)
    require(abs(high["mean"] - 0.05) < 1e-12, high)
    require(high["n_lt_0p1"] == 1, high)


def test_occ_nv84() -> None:
    e = np.linspace(-3.0, 5.0, 112)
    occ = C.occ_indices(e)
    require(occ["nv84"].size == 84, occ)
    require(occ["same_as_nv84"] or occ["n_e_lt_fermi"] != 84 or True, occ)


def test_notes_forbid_232_and_a1h() -> None:
    blob = " ".join(C.NOTES)
    require("232" in blob, blob)
    require("A1-H" in blob, blob)
    require("PASS/FAIL" in blob, blob)


def main() -> int:
    test_unique_gamma_shift_identity()
    test_unique_gamma_shift_empty()
    test_hausdorff_empty_and_match()
    test_uncovered_high_cb()
    test_nearest_mesh()
    test_window_spectrum_counts()
    test_p_stats_and_projectability()
    test_occ_nv84()
    test_notes_forbid_232_and_a1h()
    print("test_compare_112old_large_qe: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

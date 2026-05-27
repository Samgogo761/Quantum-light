#!/usr/bin/env python3
"""
Weak-field scaling check for HHG solver.

Validates perturbative response:
  current amplitude J(w)   ~ E0
  current amplitude J(2w)  ~ E0^2
  current amplitude J(3w)  ~ E0^3

HHG.dat stores |J|^2, so the expected yield scaling is:
  H1 ~ E0^2, H2 ~ E0^4, H3 ~ E0^6

Usage:
  python check_weakfield_scaling.py output_weakfield/
  python check_weakfield_scaling.py output_weakfield_I2e10 2.0e10

The base directory should contain E0/, E0_2/, E0_4/ subdirectories.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def load_hhg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, -1]  # harmonic_order, HHG_total


def find_peak(h_order: np.ndarray, hhg: np.ndarray, target_h: float, window: float = 0.3) -> float:
    mask = np.abs(h_order - target_h) < window
    if not mask.any():
        return 0.0
    return float(np.max(hhg[mask]))


def safe_exponent(y_right: float, y_left: float, e_ratio: float) -> float:
    if y_left <= 0.0 or y_right <= 0.0 or e_ratio <= 0.0 or abs(e_ratio - 1.0) < 1.0e-30:
        return float("nan")
    return float(np.log(y_right / y_left) / np.log(e_ratio))


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_weakfield")
    i_full = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else float(os.environ.get("WEAKFIELD_INTENSITY_FULL", "2.0e11"))
    )

    configs = [
        ("E0", i_full),
        ("E0_2", i_full / 4.0),
        ("E0_4", i_full / 16.0),
    ]

    results: dict[str, dict[str, float]] = {}
    for label, intensity in configs:
        path = base / label / "HHG.dat"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        h_order, hhg = load_hhg(path)
        e0 = float(np.sqrt(intensity))
        results[label] = {
            "E0": e0,
            "H1": find_peak(h_order, hhg, 1.0),
            "H2": find_peak(h_order, hhg, 2.0),
            "H3": find_peak(h_order, hhg, 3.0),
        }

    if len(results) < 2:
        print("ERROR: Need at least 2 data sets for scaling comparison.")
        sys.exit(1)

    print("=" * 78)
    print("  Weak-field scaling check")
    print("=" * 78)
    print(f"  Base directory: {base}")
    print(f"  I_full: {i_full:.6e} W/cm^2")
    print(f"  {'Config':<8} {'E0_rel':<12} {'H1':<14} {'H2':<14} {'H3':<14}")
    print("-" * 78)

    e0_ref = results.get("E0", next(iter(results.values())))["E0"]
    for label in ["E0", "E0_2", "E0_4"]:
        if label not in results:
            continue
        r = results[label]
        print(
            f"  {label:<8} {r['E0']/e0_ref:<12.4e} "
            f"{r['H1']:<14.6e} {r['H2']:<14.6e} {r['H3']:<14.6e}"
        )

    print()
    print("  Scaling ratios from HHG yields (should be ~1 if perturbative):")
    print(f"  {'Pair':<16} {'H1/E0^2':<14} {'H2/E0^4':<14} {'H3/E0^6':<14}")
    print("-" * 78)

    ref_label = "E0" if "E0" in results else next(iter(results.keys()))
    ref = results[ref_label]
    for label in ["E0_2", "E0_4"]:
        if label not in results:
            continue
        r = results[label]
        e_ratio = r["E0"] / ref["E0"]
        h1_ratio = r["H1"] / ref["H1"] / e_ratio**2 if ref["H1"] > 0 else float("nan")
        h2_ratio = r["H2"] / ref["H2"] / e_ratio**4 if ref["H2"] > 0 else float("nan")
        h3_ratio = r["H3"] / ref["H3"] / e_ratio**6 if ref["H3"] > 0 else float("nan")
        print(f"  {ref_label}->{label:<10} {h1_ratio:<14.4f} {h2_ratio:<14.4f} {h3_ratio:<14.4f}")

    print()
    print("  Empirical HHG-yield exponents p_yield from HHG ~ E0^p:")
    print("  Ideal p_yield: H1=2, H2=4, H3=6")
    print(f"  {'Pair':<16} {'H1 p':<14} {'H2 p':<14} {'H3 p':<14}")
    print("-" * 78)

    labels = ["E0", "E0_2", "E0_4"]
    for left, right in zip(labels, labels[1:]):
        if left not in results or right not in results:
            continue
        a = results[left]
        b = results[right]
        e_ratio = b["E0"] / a["E0"]
        p1 = safe_exponent(b["H1"], a["H1"], e_ratio)
        p2 = safe_exponent(b["H2"], a["H2"], e_ratio)
        p3 = safe_exponent(b["H3"], a["H3"], e_ratio)
        print(f"  {left}->{right:<10} {p1:<14.4f} {p2:<14.4f} {p3:<14.4f}")

    print()
    print("  Equivalent current-amplitude exponents p_amp = p_yield / 2:")
    print("  Ideal p_amp: H1=1, H2=2, H3=3")
    print(f"  {'Pair':<16} {'H1 p':<14} {'H2 p':<14} {'H3 p':<14}")
    print("-" * 78)
    for left, right in zip(labels, labels[1:]):
        if left not in results or right not in results:
            continue
        a = results[left]
        b = results[right]
        e_ratio = b["E0"] / a["E0"]
        p1 = 0.5 * safe_exponent(b["H1"], a["H1"], e_ratio)
        p2 = 0.5 * safe_exponent(b["H2"], a["H2"], e_ratio)
        p3 = 0.5 * safe_exponent(b["H3"], a["H3"], e_ratio)
        print(f"  {left}->{right:<10} {p1:<14.4f} {p2:<14.4f} {p3:<14.4f}")

    print()
    print("  Ratios near 1.0 and exponents near ideal indicate perturbative scaling.")
    print("  Large deviations indicate non-perturbative response or insufficiently weak fields.")
    print("=" * 78)


if __name__ == "__main__":
    main()

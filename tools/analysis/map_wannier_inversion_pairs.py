"""Map Wannier centers under a fractional inversion center.

The script parses final WF centers from a Wannier90 .wout file, converts them
to fractional coordinates using lattice vectors from the corresponding .win,
and finds nearest center partners under r -> 2c - r.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


WF_RE = re.compile(
    r"WF centre and spread\s+(\d+)\s+\(\s*([\-0-9.Ee+]+),\s*([\-0-9.Ee+]+),\s*([\-0-9.Ee+]+)\s*\)\s+([\-0-9.Ee+]+)"
)


def parse_lattice_ang(win: Path) -> np.ndarray:
    lines = win.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "begin unit_cell_cart":
            unit = lines[i + 1].strip().lower()
            if unit != "angstrom":
                raise ValueError(f"Only angstrom unit_cell_cart is supported, got {unit}")
            return np.array([[float(x) for x in lines[i + 2 + j].split()[:3]] for j in range(3)], dtype=float)
    raise ValueError(f"Cannot find unit_cell_cart in {win}")


def parse_final_centers(wout: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = wout.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "Final State":
            start = i
            break
    if start is None:
        raise ValueError(f"Cannot find Final State in {wout}")
    centers: list[tuple[int, list[float], float]] = []
    for line in lines[start + 1 :]:
        m = WF_RE.search(line)
        if m:
            centers.append((int(m.group(1)), [float(m.group(j)) for j in range(2, 5)], float(m.group(5))))
        elif centers and line.strip().startswith("Sum of centres"):
            break
    if not centers:
        raise ValueError(f"No WF centers found in Final State of {wout}")
    centers.sort(key=lambda x: x[0])
    return np.array([c for _, c, _ in centers], dtype=float), np.array([s for _, _, s in centers], dtype=float)


def frac_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 0.5) % 1.0 - 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--win", required=True, type=Path)
    parser.add_argument("--wout", required=True, type=Path)
    parser.add_argument("--center", default="0.499998583,0.333333067,0.372004766")
    parser.add_argument("--max-print", type=int, default=40)
    args = parser.parse_args()

    lattice = parse_lattice_ang(args.win)
    centers_cart, spreads = parse_final_centers(args.wout)
    # Wannier90 centers are Cartesian Angstrom. Rows of lattice are a1,a2,a3.
    centers_frac = centers_cart @ np.linalg.inv(lattice)
    centers_frac = centers_frac % 1.0
    inv_center = np.array([float(x) for x in args.center.split(",")], dtype=float)

    pairs: list[tuple[int, int, float, float, float]] = []
    used: dict[int, int] = {}
    for i, f in enumerate(centers_frac):
        image = (2.0 * inv_center - f) % 1.0
        deltas = np.array([np.linalg.norm(frac_delta(image, g)) for g in centers_frac])
        j = int(np.argmin(deltas))
        pairs.append((i + 1, j + 1, float(deltas[j]), float(spreads[i]), float(spreads[j])))
        used[j + 1] = used.get(j + 1, 0) + 1

    max_dist = max(p[2] for p in pairs)
    mean_dist = sum(p[2] for p in pairs) / len(pairs)
    duplicates = sorted(k for k, v in used.items() if v > 1)
    non_involutive = [(i, j) for i, j, *_ in pairs if pairs[j - 1][1] != i]

    print(f"n_wf = {len(pairs)}")
    print(f"max_pair_distance_fractional = {max_dist:.8e}")
    print(f"mean_pair_distance_fractional = {mean_dist:.8e}")
    print(f"duplicate_targets = {duplicates[:20]} count={len(duplicates)}")
    print(f"non_involutive_pairs_count = {len(non_involutive)}")
    print("# i P(i) frac_distance spread_i spread_Pi")
    for i, j, dist, si, sj in pairs[: args.max_print]:
        print(f"{i:4d} {j:4d} {dist:14.8e} {si:12.6f} {sj:12.6f}")


if __name__ == "__main__":
    main()

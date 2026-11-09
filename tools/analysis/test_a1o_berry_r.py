#!/usr/bin/env python3
"""Unit tests for A1-O star geometry and gauge-aware Berry comparisons."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from a1o_berry_math import (
    align_overlap,
    berry_from_pm_crystal,
    energy_pi_q,
    frobenius_block,
    isolated_mask,
    phase_align_columns,
    polar_right_unitary,
)
from a1o_star_geometry import DELTAS, NNB_CENTER, build_star


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def test_star_unique() -> None:
    star = build_star()
    require(star["n_k"] == 4 + 4 * NNB_CENTER, star["n_k"])
    require(star["nnb"] == 12, star["nnb"])
    require(len(star["nnkpts"]) == star["n_k"] * 12, len(star["nnkpts"]))
    k = np.array(star["kpoints"])
    rounded = {tuple(np.round(row, 12)) for row in k}
    require(len(rounded) == len(k), "duplicate k")
    labels = [c["label"] for c in star["centers"]]
    require(labels == ["k_ctrl", "k_eta", "k_H5", "k_H7"], labels)
    eta = next(c for c in star["centers"] if c["label"] == "k_eta")
    require(eta["k_index_1based"] == 2, eta)
    plus = [nb for nb in eta["neighbors"] if nb["dir"] == "+x" and abs(nb["delta"] - 0.002) < 1e-15][0]
    require(abs(plus["crystal"][0] - 0.102) < 1e-12, plus)
    require(plus["g"] == [0, 0, 0], plus)
    sat_rows = star["nnkpts"][4 * 12 :]
    require(all(a == b for a, b, *_ in sat_rows), "satellite dummy must be self")


def test_polar_block_invariance() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))
    for a in range(3):
        r[a] = 0.5 * (r[a] + r[a].conj().T)
    pi = np.arange(5)
    q = np.arange(5, 8)
    u_pi = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))[0]
    u_q = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
    r2 = np.empty_like(r)
    mix = np.eye(8, dtype=complex)
    mix[np.ix_(pi, pi)] = u_pi
    mix[np.ix_(q, q)] = u_q
    for a in range(3):
        r2[a] = mix.conj().T @ r[a] @ mix
    a = frobenius_block(r, pi, q)
    b = frobenius_block(r2, pi, q)
    for ax in ("x", "y", "z"):
        require(abs(a[f"F_{ax}"] - b[f"F_{ax}"]) < 1e-10, (ax, a[f"F_{ax}"], b[f"F_{ax}"]))
        require(abs(a[f"svd1_{ax}"] - b[f"svd1_{ax}"]) < 1e-10, ax)


def test_isolated_not_invariant() -> None:
    rng = np.random.default_rng(4)
    r = np.zeros((3, 4, 4), dtype=complex)
    r[0, 0, 2] = 1.2
    r[0, 2, 0] = 1.2
    mix = np.eye(4, dtype=complex)
    u = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    mix[2:4, 2:4] = u
    r2 = mix.conj().T @ r[0] @ mix
    require(abs(abs(r[0, 0, 2]) - abs(r2[0, 2])) > 0.05, "degen pair should move")
    blk = np.linalg.norm(r[0][:2, 2:])
    blk2 = np.linalg.norm(r2[:2, 2:])
    require(abs(blk - blk2) < 1e-12, (blk, blk2))


def test_fd_recovers_cartesian_a() -> None:
    recip = np.array(
        [
            [0.8979061, 0.5184063, 0.0],
            [0.0001815, 1.0364982, 0.0],
            [0.0, 0.0, 0.2513274],
        ]
    )
    delta = 0.005
    ax = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=complex)
    ay = np.array([[0.0, -0.25j], [0.25j, 0.0]], dtype=complex)

    def overlap(shift: tuple[float, float, float]) -> np.ndarray:
        b = np.array(shift) @ recip
        gen = -1j * (ax * b[0] + ay * b[1])
        return np.eye(2, dtype=complex) + gen

    r = berry_from_pm_crystal(
        overlap((delta, 0.0, 0.0)),
        overlap((-delta, 0.0, 0.0)),
        overlap((0.0, delta, 0.0)),
        overlap((0.0, -delta, 0.0)),
        delta,
        recip,
    )
    require(np.max(np.abs(r[0] - ax)) < 1e-12, np.max(np.abs(r[0] - ax)))
    require(np.max(np.abs(r[1] - ay)) < 1e-12, np.max(np.abs(r[1] - ay)))
    require(np.max(np.abs(r[2])) < 1e-12, r[2])


def test_column_phase_then_fd() -> None:
    recip = np.array(
        [
            [0.8979061, 0.5184063, 0.0],
            [0.0001815, 1.0364982, 0.0],
            [0.0, 0.0, 0.2513274],
        ]
    )
    delta = 0.005
    ax = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=complex)
    ay = np.array([[0.0, -0.25j], [0.25j, 0.0]], dtype=complex)
    rng = np.random.default_rng(6)

    def overlap(shift: tuple[float, float, float]) -> np.ndarray:
        b = np.array(shift) @ recip
        gen = -1j * (ax * b[0] + ay * b[1])
        m = np.eye(2, dtype=complex) + gen
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=2))
        return m * phases

    r_ok = berry_from_pm_crystal(
        phase_align_columns(overlap((delta, 0.0, 0.0))),
        phase_align_columns(overlap((-delta, 0.0, 0.0))),
        phase_align_columns(overlap((0.0, delta, 0.0))),
        phase_align_columns(overlap((0.0, -delta, 0.0))),
        delta,
        recip,
    )
    require(np.max(np.abs(r_ok[0] - ax)) < 1e-10, np.max(np.abs(r_ok[0] - ax)))
    require(np.max(np.abs(r_ok[1] - ay)) < 1e-10, np.max(np.abs(r_ok[1] - ay)))

    r_wiped = berry_from_pm_crystal(
        align_overlap(overlap((delta, 0.0, 0.0))),
        align_overlap(overlap((-delta, 0.0, 0.0))),
        align_overlap(overlap((0.0, delta, 0.0))),
        align_overlap(overlap((0.0, -delta, 0.0))),
        delta,
        recip,
    )
    require(np.max(np.abs(r_wiped[0])) < 1e-8, np.max(np.abs(r_wiped[0])))


def test_align_then_identity() -> None:
    rng = np.random.default_rng(5)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=6))
    m = np.diag(phases)
    aligned = align_overlap(m)
    require(np.max(np.abs(aligned - np.eye(6))) < 1e-12, aligned)
    u = polar_right_unitary(2 * np.eye(3, dtype=complex))
    require(np.max(np.abs(u - np.eye(3))) < 1e-12, u)


def test_energy_split_and_iso() -> None:
    pi, q = energy_pi_q(144, 32)
    require(pi.size == 112 and q.size == 32 and q[0] == 112, (pi[-1], q[0]))
    e = np.array([-1.0, -0.9, 0.0, 0.8])
    iso = isolated_mask(e, 0.15)
    require(iso.tolist() == [False, False, True, True], iso)


def main() -> int:
    test_star_unique()
    test_polar_block_invariance()
    test_isolated_not_invariant()
    test_fd_recovers_cartesian_a()
    test_column_phase_then_fd()
    test_align_then_identity()
    test_energy_split_and_iso()
    print("test_a1o_berry_r: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

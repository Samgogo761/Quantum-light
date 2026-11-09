#!/usr/bin/env python3
"""Analytic benchmarks for the A1 classical-mixture statistics tool.

Key demonstrations:
1. Single coherent node: g2 = 1, n_cl = m_cl = 0 (Poissonian mixture limit).
2. Gauss-Hermite sampled Gaussian quadrature: for |beta| ~ |x|^p the
   classical g2 equals Gaussian moment ratios <x^{4p}>/<x^{2p}>^2:
     p=1 -> 3 (real-Gaussian amplitude; bright-SV limit only),
     p=2 -> 105/9, p=3 -> 46.2.
   GH5 is exact only through x^8, so p=3 must FAIL on GH5 and PASS on GH7.
   This is the |J|^4 quadrature-order gate in miniature, independent of
   the Hilbert-space / top-buffer gate.
3. +N/-N pairing recovers Je/Jo channels exactly.
4. Two-time factorized ensemble: g2(t,t') equals the single-mode g2.
5. Intensity sensitivity: dropping the high-intensity-tail node and renormalizing
   recovers the inner-node g2 (Hilbert-space sensitivity bracket).
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_mixture_stats import (  # noqa: E402
    cross_g2,
    harmonic_envelope,
    mixture_moments,
    pm_channel_stats,
    intensity_sensitivity,
    subset_by_ids,
    two_time_stats,
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def gh_probability_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Hermite-Gauss nodes/weights as a normalized probability (sigma^2 = 1/2)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w / math.sqrt(math.pi)


def gaussian_moment(p: int) -> float:
    """<x^(2p)> for N(0, 1/2): (2p-1)!! / 2^p."""
    val = 1.0
    for k in range(1, 2 * p, 2):
        val *= k
    return val / (2.0 ** p)


def test_single_coherent_node() -> None:
    m = mixture_moments(np.array([1.0]), np.array([0.3 + 0.4j]))
    require(abs(m["g2"] - 1.0) < 1e-14, f"coherent g2 {m['g2']}")
    require(abs(m["n_cl"]) < 1e-16, "coherent n_cl")
    require(abs(m["m_cl_re"]) < 1e-16 and abs(m["m_cl_im"]) < 1e-16, "coherent m_cl")
    require(abs(m["coherent_fraction"] - 1.0) < 1e-14, "coherent fraction")


def test_two_point_mixture() -> None:
    w = np.array([0.5, 0.5])
    g = np.array([1.0 + 0j, 0.0 + 0j])
    m = mixture_moments(w, g)
    # I2 = 1/2, I4 = 1/2 -> g2 = 2
    require(abs(m["g2"] - 2.0) < 1e-14, f"two-point g2 {m['g2']}")


def test_gh_power_law_g2() -> None:
    for p, n_gh, should_pass in ((1, 5, True), (2, 5, True), (3, 5, False), (3, 7, True)):
        x, w = gh_probability_nodes(n_gh)
        beta = (x.astype(complex)) ** p
        m = mixture_moments(w, beta)
        exact = gaussian_moment(2 * p) / gaussian_moment(p) ** 2
        rel = abs(m["g2"] - exact) / exact
        if should_pass:
            require(rel < 1e-12, f"GH{n_gh} p={p}: rel {rel}")
        else:
            require(rel > 0.05, f"GH{n_gh} p={p} should show quadrature error, rel {rel}")
    # headline numbers
    x, w = gh_probability_nodes(5)
    g2_lin = mixture_moments(w, x.astype(complex))["g2"]
    require(abs(g2_lin - 3.0) < 1e-12,
            f"real-Gaussian amplitude g2 {g2_lin} != 3")


def test_cross_g2_uncorrelated_vs_common_noise() -> None:
    x, w = gh_probability_nodes(7)
    a = x.astype(complex)          # H_n ~ x
    b = (x.astype(complex)) ** 2   # H_m ~ x^2, same underlying noise
    val = cross_g2(w, a, b)
    exact = gaussian_moment(3) / (gaussian_moment(1) * gaussian_moment(2))
    require(abs(val - exact) / exact < 1e-12, f"cross g2 {val} vs {exact}")
    require(val > 1.0, "common-noise cross correlation must bunch")


def _mode_rows(ids, w, order, jx, jy):
    rows = []
    for i, wi, x, y in zip(ids, w, jx, jy):
        rows.append([i, wi, 1.0e11, 0.5, order,
                     x.real, x.imag, y.real, y.imag,
                     abs(x) ** 2 + abs(y) ** 2])
    return rows


def test_pm_channels() -> None:
    ids = [1, 2]
    w = [0.5, 0.5]
    a = np.array([0.6 + 0.1j, -0.2 + 0.3j])   # even channel
    b = np.array([0.05 - 0.2j, 0.15 + 0.1j])  # odd channel
    plus = np.array(_mode_rows(ids, w, 2, a + b, 0 * a), dtype=float)
    minus = np.array(_mode_rows(ids, w, 2, a - b, 0 * a), dtype=float)
    out = pm_channel_stats(plus, minus)
    je = out["2"]["domain_even"]["x"]
    jo = out["2"]["domain_odd"]["x"]
    require(abs(je["m1_re"] - np.sum(np.array(w) * a).real) < 1e-14, "Je mean")
    require(abs(jo["m1_re"] - np.sum(np.array(w) * b).real) < 1e-14, "Jo mean")
    i2_je = float(np.sum(np.array(w) * np.abs(a) ** 2))
    require(abs(je["I2"] - i2_je) < 1e-14, "Je I2")


def test_two_time_factorized() -> None:
    # J_s(t) = c_s * u(t): g2(t,t') = <|c|^4>/<|c|^2>^2 everywhere.
    x, wgh = gh_probability_nodes(5)
    nt = 4096
    dt_au = 0.35
    t_au = np.arange(nt) * dt_au
    omega0 = 0.014246
    u = np.exp(-((t_au - t_au[-1] / 2) / (t_au[-1] / 6)) ** 2) * np.cos(5 * omega0 * t_au)
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        weights = {}
        for k, (xk, wk) in enumerate(zip(x, wgh), start=1):
            j = xk * u
            path = Path(tmp) / f"Jt_node_{k:04d}.dat"
            t_fs = t_au * 2.418884326505e-2
            arr = np.column_stack([np.arange(1, nt + 1), t_fs, j, 0.0 * j])
            header = "# it   time(fs)   Jx(a.u.)   Jy(a.u.)"
            np.savetxt(path, arr, header=header, comments="")
            files[k] = path
            weights[k] = wk
        out = two_time_stats(files, weights, omega0, orders=(5,), decimate=64)
    g2_exact = gaussian_moment(2) / gaussian_moment(1) ** 2  # 3
    diag = [v for v in out["5"]["g2_tt_diag"] if v is not None and v > 0]
    mean_i = np.array(out["5"]["mean_intensity_t"])
    keep = mean_i > 1e-6 * mean_i.max()
    diag_kept = np.array(out["5"]["g2_tt_diag"], dtype=float)[keep]
    require(np.all(np.abs(diag_kept - g2_exact) < 1e-6),
            f"two-time diag g2 max dev {np.max(np.abs(diag_kept - g2_exact))}")
    row = np.array(out["5"]["g2_peak_row"], dtype=float)[keep]
    require(np.all(np.abs(row - g2_exact) < 1e-6), "two-time off-diagonal g2")
    g1 = np.array(out["5"]["abs_g1_peak_row"], dtype=float)[keep]
    require(np.all(np.abs(g1 - 1.0) < 1e-6), "factorized ensemble g1 must be 1")
    require(len(diag) > 0, "no valid diagonal points")


def test_ring_drop_outer() -> None:
    # Two nodes: inner coherent (g2=1) and a bright outer node.
    # After dropping the outer ring and renormalizing, g2 must return to 1.
    plus = np.array(
        _mode_rows(
            [1, 2],
            [0.9, 0.1],
            2,
            [0.4 + 0.0j, 4.0 + 0.0j],
            [0.0j, 0.0j],
        ),
        dtype=float,
    )
    plus[0, 2] = 1.0e10   # COL_INTENSITY inner
    plus[1, 2] = 1.6e12   # outer
    minus = plus.copy()
    minus[:, 5] *= -1.0   # flip ReJx so Jo is nonzero; Je cancels
    br = intensity_sensitivity(plus, minus)
    require(br["inventory"]["low_intensity"]["n_nodes"] == 1, "low count")
    require(br["inventory"]["high_intensity_tail"]["n_nodes"] == 1, "tail count")
    g2_drop = br["drop_high_intensity_tail"]["single_domain"]["2"]["x_g2"]
    require(abs(g2_drop - 1.0) < 1e-12, f"drop_outer g2 {g2_drop}")
    g2_inner = br["low_intensity_only"]["single_domain"]["2"]["x_g2"]
    require(abs(g2_inner - 1.0) < 1e-12, f"inner_only g2 {g2_inner}")
    sub = subset_by_ids(plus, [1])
    require(abs(float(sub[0, 1]) - 1.0) < 1e-12, "renormalized weight")


def test_envelope_isolates_harmonic() -> None:
    nt = 8192
    dt_au = 0.35
    t = np.arange(nt) * dt_au
    omega0 = 0.014246
    sig = np.cos(5 * omega0 * t) + 0.5 * np.cos(7 * omega0 * t)
    env5 = harmonic_envelope(sig, dt_au, omega0, 5)
    env7 = harmonic_envelope(sig, dt_au, omega0, 7)
    mid = slice(nt // 4, 3 * nt // 4)
    r5 = float(np.mean(np.abs(env5[mid]) ** 2))
    r7 = float(np.mean(np.abs(env7[mid]) ** 2))
    require(abs(r5 / r7 - 4.0) < 0.2, f"envelope power ratio {r5 / r7} != 4")


def test_two_time_rejects_mismatched_time_grid() -> None:
    nt = 64
    t_fs = np.arange(nt) * 0.01
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for node in (1, 2):
            t = t_fs.copy()
            if node == 2:
                t[-1] += 1.0e-5
            arr = np.column_stack([np.arange(1, nt + 1), t, np.cos(t), np.sin(t)])
            path = Path(tmp) / f"Jt_node_{node:04d}.dat"
            np.savetxt(path, arr)
            files[node] = path
        try:
            two_time_stats(files, {1: 0.5, 2: 0.5}, 0.014, orders=(2,))
        except RuntimeError as exc:
            require("time grid mismatch" in str(exc), f"wrong time-grid error: {exc}")
            return
    raise RuntimeError("mismatched Jt grids were accepted")


def main() -> int:
    test_single_coherent_node()
    test_two_point_mixture()
    test_gh_power_law_g2()
    test_cross_g2_uncorrelated_vs_common_noise()
    test_pm_channels()
    test_two_time_factorized()
    test_ring_drop_outer()
    test_envelope_isolates_harmonic()
    test_two_time_rejects_mismatched_time_grid()
    print("PASS test_classical_mixture_stats")
    print("A1 classical-mixture gates: 9/9 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

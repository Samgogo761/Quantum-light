#!/usr/bin/env python3
"""Abstract algebraic fixtures for the 2026-09-15 CJJ candidate (Q3–Q11).

These tests check mathematics on small quadratic systems. Passing them does
not freeze the M0 physical definition, operator order, normalization, or
output mode. Do not run M0/P1 CJJ here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fock  # noqa: E402
from wick import equal_time_wick, mean_current, two_time_wick_greater  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/cjj_algebra_fixture_20260915"
RNG = np.random.default_rng(20260915)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def herm(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def rand_h(n: int) -> np.ndarray:
    a = RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n))
    return herm(a)


def sp_unitary(h: np.ndarray, dt: float) -> np.ndarray:
    e, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * e * dt)) @ v.conj().T


def mb_unitary(h_sp: np.ndarray, dt: float) -> np.ndarray:
    return sp_unitary(fock.quadratic_hamiltonian(h_sp), dt)


def slater_psi(n_orb: int, occupied) -> np.ndarray:
    state = 0
    for i in occupied:
        state |= 1 << i
    psi = np.zeros(1 << n_orb, dtype=complex)
    psi[state] = 1.0
    return psi


def slater_p(n_orb: int, occupied) -> np.ndarray:
    p = np.zeros((n_orb, n_orb), dtype=complex)
    for i in occupied:
        p[i, i] = 1.0
    return p


def occup_empty_bases(p: np.ndarray):
    e, v = np.linalg.eigh(p)
    occ = v[:, e > 0.5]
    emp = v[:, e <= 0.5]
    return occ, emp


def q3(p0: np.ndarray, ja_h: np.ndarray, jb_h: np.ndarray) -> complex:
    q0 = np.eye(p0.shape[0], dtype=complex) - p0
    return complex(np.trace(p0 @ ja_h @ q0 @ jb_h))


def q8(p0: np.ndarray, ja_h: np.ndarray, jb_h: np.ndarray) -> complex:
    vv, vc = occup_empty_bases(p0)
    xa = vc.conj().T @ ja_h @ vv
    xb = vc.conj().T @ jb_h @ vv
    return complex(np.trace(xa.conj().T @ xb))


def heisenberg(j: np.ndarray, u_t0: np.ndarray) -> np.ndarray:
    return u_t0.conj().T @ j @ u_t0


def window_sin2(t, tp):
    return np.sin(np.pi * t / tp) ** 2


def test_q3_q4_q8_fock_complex() -> dict:
    n = 4
    occupied = (0, 1)
    p0 = slater_p(n, occupied)
    h1, h2, h3 = rand_h(n), rand_h(n), rand_h(n)
    ja = rand_h(n)
    jb = rand_h(n)
    require(np.max(np.abs(ja - ja.conj().T)) < 1e-14, "ja Hermitian")
    require(np.max(np.abs(jb - jb.conj().T)) < 1e-14, "jb Hermitian")
    dt1, dt2 = 0.17, 0.31
    u01 = sp_unitary(h1, dt1)
    u12 = sp_unitary(h2, dt2)
    u02 = u12 @ u01
    u_t_tp = u12
    rho_tp = u01 @ p0 @ u01.conj().T
    q4 = complex(two_time_wick_greater(ja, jb, rho_tp, u_t_tp))
    ja_h = heisenberg(ja, u02)
    jb_h = heisenberg(jb, u01)
    c3 = q3(p0, ja_h, jb_h)
    c8 = q8(p0, ja_h, jb_h)
    require(abs(q4 - c3) < 1e-10, f"Q4 vs Q3 {q4} {c3}")
    require(abs(c8 - c3) < 1e-10, f"Q8 vs Q3 {c8} {c3}")
    require(abs(c3.imag) > 1e-8, "need a genuine complex value")

    psi = slater_psi(n, occupied)
    rho_mb = np.outer(psi, psi.conj())
    u_mb_01 = mb_unitary(h1, dt1)
    u_mb_12 = mb_unitary(h2, dt2)
    u_mb_02 = u_mb_12 @ u_mb_01
    ja_mb = fock.bilinear_matrix(ja)
    jb_mb = fock.bilinear_matrix(jb)
    rho_tp_mb = u_mb_01 @ rho_mb @ u_mb_01.conj().T
    fock_c = complex(fock.two_time_greater(rho_tp_mb, ja_mb, jb_mb, u_mb_12))
    require(abs(fock_c - c3) < 1e-8, f"Fock vs Q3 {fock_c} {c3}")

    eq = complex(equal_time_wick(ja_h, jb_h, p0))
    require(abs(eq - q3(p0, ja_h, jb_h)) < 1e-10, "equal-time Pauli vs Q3")

    mean_a = complex(mean_current(ja_h, p0))
    mean_b = complex(mean_current(jb_h, p0))
    full = fock_c + mean_a * mean_b
    ja_h_mb = u_mb_02.conj().T @ ja_mb @ u_mb_02
    jb_h_mb = u_mb_01.conj().T @ jb_mb @ u_mb_01
    full_fock = complex(np.trace(rho_mb @ ja_h_mb @ jb_h_mb))
    require(abs(full - full_fock) < 1e-8, "mean product + connected reconstructs full")

    c_less = q3(p0, jb_h, ja_h)
    require(abs(c_less - np.conjugate(c3)) < 1e-10, "Hermitian current C^>_ab(t,t') = [C^>_ba(t',t)]*")

    empty = slater_p(n, ())
    fullp = slater_p(n, tuple(range(n)))
    require(abs(q3(empty, ja_h, jb_h)) < 1e-14, "empty connected kernel is 0")
    require(abs(q3(fullp, ja_h, jb_h)) < 1e-14, "full connected kernel is 0")

    # reversed order stored independently
    c_rev = q3(p0, jb_h, ja_h)
    return {
        "q3": [c3.real, c3.imag],
        "q4": [q4.real, q4.imag],
        "q8": [c8.real, c8.imag],
        "fock": [fock_c.real, fock_c.imag],
        "c_less": [c_less.real, c_less.imag],
        "c_rev": [c_rev.real, c_rev.imag],
    }


def test_non_slater_thermal_not_x_formula() -> dict:
    n = 3
    h = rand_h(n)
    e, v = np.linalg.eigh(h)
    occ = 1.0 / (np.exp(e - e.mean()) + 1.0)
    rho = (v * occ) @ v.conj().T
    require(np.max(np.abs(rho @ rho - rho)) > 1e-3, "thermal rho is not a projector")
    ja, jb = rand_h(n), rand_h(n)
    wick_et = complex(equal_time_wick(ja, jb, rho))
    # Illegal Slater-X on a non-projector must not be treated as the Wick kernel.
    evals, vecs = np.linalg.eigh(rho)
    vv = vecs[:, evals > 0.5]
    vc = vecs[:, evals <= 0.5]
    if vv.size == 0 or vc.size == 0:
        x_illegal = 0.0
    else:
        xa = vc.conj().T @ ja @ vv
        xb = vc.conj().T @ jb @ vv
        x_illegal = complex(np.trace(xa.conj().T @ xb))
    require(abs(wick_et - x_illegal) > 1e-6, "Slater X formula must not match thermal Wick")
    return {"wick": [wick_et.real, wick_et.imag], "illegal_x": [complex(x_illegal).real, complex(x_illegal).imag]}


def test_rotation_invariance_and_gram_psd() -> dict:
    n = 4
    p0 = slater_p(n, (0, 1))
    jx, jy = rand_h(n), rand_h(n)
    times = [0.2, 0.7, 1.1]
    h = rand_h(n)
    xs = []
    mus = []
    cs = []
    for t in times:
        u = sp_unitary(h, t)
        for j in (jx, jy):
            jh = heisenberg(j, u)
            vv, vc = occup_empty_bases(p0)
            xs.append(vc.conj().T @ jh @ vv)
            mus.append(complex(np.trace(p0 @ jh)))
            cs.append(q3(p0, jh, jh))
    kmat = np.zeros((len(xs), len(xs)), dtype=complex)
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            kmat[i, j] = np.trace(xi.conj().T @ xj)
    require(np.max(np.abs(kmat - kmat.conj().T)) < 1e-10, "Gram Hermitian")
    evals = np.linalg.eigvalsh(0.5 * (kmat + kmat.conj().T))
    require(np.min(evals) > -1e-10, f"Gram not PSD {evals}")

    rv = sp_unitary(rand_h(2), 0.4)
    rc = sp_unitary(rand_h(2), 0.9)
    vv, vc = occup_empty_bases(p0)
    jh = heisenberg(jx, sp_unitary(h, 0.55))
    c0 = q8(p0, jh, jh)
    xa = (vc @ rc).conj().T @ jh @ (vv @ rv)
    c_rot = complex(np.trace(xa.conj().T @ xa))
    require(abs(c_rot - c0) < 1e-10, "occupied/empty unitary rotation invariance")
    mu0 = complex(np.trace(p0 @ jh))
    p_rot = (vv @ rv) @ (vv @ rv).conj().T
    require(np.max(np.abs(p_rot - p0)) < 1e-10, "P unchanged by occupied rotation")
    require(abs(mu0 - complex(np.trace(p_rot @ jh))) < 1e-10, "mean rotation invariant")
    return {"gram_min_eig": float(np.min(evals)), "n_blocks": len(xs)}


def test_filter_q10_and_two_level_orientation() -> dict:
    omega = 1.25
    h = np.diag([0.0, omega]).astype(complex)
    j = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    tp = 12.0
    n = 2401
    t = np.linspace(0.0, tp, n)
    wgt = window_sin2(t, tp)
    omega_f = omega
    f = (wgt / tp) * np.exp(1j * omega_f * t)

    def y_for(p0):
        ys = []
        for ti in t:
            u = sp_unitary(h, ti)
            jh = heisenberg(j, u)
            vv, vc = occup_empty_bases(p0)
            ys.append(vc.conj().T @ jh @ vv)
        y = np.trapezoid(f[:, None, None] * np.stack(ys, axis=0), t, axis=0)
        return y

    y_g = y_for(slater_p(2, (0,)))
    y_e = y_for(slater_p(2, (1,)))
    d_g = complex(np.trace(y_g.conj().T @ y_g))
    d_e = complex(np.trace(y_e.conj().T @ y_e))
    require(d_e.real > 20.0 * d_g.real, f"excited resonance should dominate ground tail {d_e} vs {d_g}")
    require(d_g.real > 0.0, "finite-window ground tail is not strictly zero")

    # Q10: double-time integral vs Y†Y
    p0 = slater_p(2, (1,))
    nt_c = 81
    tc = np.linspace(0.0, tp, nt_c)
    fc = (window_sin2(tc, tp) / tp) * np.exp(1j * omega_f * tc)
    cmat = np.zeros((nt_c, nt_c), dtype=complex)
    xs = []
    for ti in tc:
        u = sp_unitary(h, ti)
        jh = heisenberg(j, u)
        vv, vc = occup_empty_bases(p0)
        xs.append(vc.conj().T @ jh @ vv)
    for i in range(nt_c):
        for jix in range(nt_c):
            cmat[i, jix] = np.trace(xs[i].conj().T @ xs[jix])
    d_int = complex(np.trapezoid(np.trapezoid(np.conjugate(fc)[:, None] * cmat * fc[None, :], tc, axis=1), tc))
    y = np.trapezoid(fc[:, None, None] * np.stack(xs, axis=0), tc, axis=0)
    d_y = complex(np.trace(y.conj().T @ y))
    require(abs(d_int - d_y) / max(abs(d_y), 1e-18) < 0.02, f"Q10 double integral vs Y {d_int} {d_y}")

    # pump-minus-reference can go negative and is then not a covariance
    d_diff = d_e - d_g
    k_diff = np.array([[d_e, 0], [0, -d_g]], dtype=complex)
    evals = np.linalg.eigvalsh(0.5 * (k_diff + k_diff.conj().T))
    require(np.min(evals) < 0.0, "reference-subtracted kernel need not be PSD")
    return {
        "d_ground": [d_g.real, d_g.imag],
        "d_excited": [d_e.real, d_e.imag],
        "q10_rel_err": abs(d_int - d_y) / abs(d_y),
        "ref_subtracted_min_eig": float(np.min(evals)),
        "d_diff_real": float(d_diff.real),
    }


def test_weight_conventions_and_ncell() -> dict:
    c_k = 1.7 + 0.3j
    # Q5: split one node into two half-weight copies → per-cell kernel unchanged
    c_q5 = 1.0 * c_k
    c_split = 0.5 * c_k + 0.5 * c_k
    require(abs(c_q5 - c_split) < 1e-15, "Q5 invariant under node duplication")
    # discrete operator J=Σ α_k j_k has α² weights
    var_one = (1.0**2) * c_k
    var_split = (0.5**2 + 0.5**2) * c_k
    require(abs(var_split - 0.5 * var_one) < 1e-15, "discrete J=Σαj variance uses α²")
    require(abs(var_split - c_split) > 0.1, "the two normalizations are not the same object")

    mu = 0.4
    scales = {}
    for ncell in (1, 2, 4):
        scales[str(ncell)] = {
            "mean_I": ncell * mu,
            "C": ncell * c_k.real,  # illustration; keep complex in full formula
            "C_complex": [float((ncell * c_k).real), float((ncell * c_k).imag)],
            "mean_product": (ncell * mu) ** 2,
            "full_re": float(((ncell * mu) ** 2 + ncell * c_k).real),
            "full_im": float(((ncell * mu) ** 2 + ncell * c_k).imag),
        }
    require(abs(scales["2"]["mean_I"] / scales["1"]["mean_I"] - 2) < 1e-12, "mean ∝ N")
    require(abs(scales["4"]["C"] / scales["1"]["C"] - 4) < 1e-12, "connected ∝ N")
    require(abs(scales["4"]["mean_product"] / scales["1"]["mean_product"] - 16) < 1e-12, "mean product ∝ N²")
    return {"q5_split_ok": True, "discrete_alpha2_ratio": abs(var_split / var_one), "ncell": scales}


def test_q13_and_domain_difference_not_psd() -> dict:
    bp = 0.2 - 0.5j
    bm = -bp
    be = 0.5 * (bp + bm)
    bo = 0.5 * (bp - bm)
    lhs = abs(bp) ** 2 - abs(bm) ** 2
    rhs = 4.0 * np.real(np.conjugate(be) * bo)
    require(abs(lhs - rhs) < 1e-12, "Q13")
    require(abs(lhs) < 1e-12, "opposite phase: intensity contrast vanishes")
    # domain-difference kernel
    cplus = np.array([[2.0, 0.1], [0.1, 0.5]], dtype=complex)
    cminus = np.array([[0.4, 0.0], [0.0, 1.8]], dtype=complex)
    c_o = 0.5 * (cplus - cminus)
    evals = np.linalg.eigvalsh(0.5 * (c_o + c_o.conj().T))
    require(np.min(evals) < 0.0, "domain-difference kernel may have negative eigenvalues")
    return {"q13_lhs": float(lhs), "domain_diff_eigs": evals.tolist()}


def main() -> None:
    report = {
        "cjj_definition": "pending_human_review",
        "cjj_algebra_tests": "running",
        "m0_cjj_physics": "not_run",
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "physics_approved": False,
        "tests": {},
    }
    report["tests"]["q3_q4_q8_fock"] = test_q3_q4_q8_fock_complex()
    report["tests"]["thermal_not_slater_x"] = test_non_slater_thermal_not_x_formula()
    report["tests"]["rotation_gram"] = test_rotation_invariance_and_gram_psd()
    report["tests"]["filter_q10_two_level"] = test_filter_q10_and_two_level_orientation()
    report["tests"]["weights_ncell"] = test_weight_conventions_and_ncell()
    report["tests"]["q13_domain_diff"] = test_q13_and_domain_difference_not_psd()
    report["cjj_algebra_tests"] = "PASS"
    OUT.mkdir(parents=True, exist_ok=True)

    def _enc(o):
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    (OUT / "algebra_report.json").write_text(json.dumps(report, indent=2, default=_enc), encoding="utf-8")
    print("test_cjj_algebra_q3q11: PASS")
    print(f"wrote {OUT / 'algebra_report.json'}")


if __name__ == "__main__":
    main()

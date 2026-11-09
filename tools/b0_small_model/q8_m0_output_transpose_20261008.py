#!/usr/bin/env python3
"""Exact output transpose of the existing discrete midpoint column algorithm.

This optional diagnostic does NOT change the frozen forward runner.  It computes
the same finite-time current output functional, not a new radiation model or an
automatic time-discretization correction.  In particular canonical U0 is still
supplied by the existing interface outside this module.

For x=(b,bdag,A), x_next=M x+N f and output += q(x+x_next)/2:
    s = row_next + q/2
    row_prev = q/2 + s M,   forcing_weight = s N.
The low-rank Schur structure below avoids forming M.  This is a bilinear
transpose, NOT a Hermitian adjoint or inverse-time physical propagation.

The first group of rows uses K; the second uses K*.  Conjugating the *entire*
second contraction gives sum h K conj(pol @ I), matching the existing caller
also for complex pol.  b and bdag must remain independent input coefficients.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BackwardRows:
    row0: np.ndarray
    forcing: np.ndarray
    n_modes: int
    n_steps: int
    h: float
    elapsed_s: float


def midpoint_output_pullback(row_next, u, d, h, gamma, detector_rows):
    """One matrix-free step; detector_rows is h*K*pol (one row per output).

    u: (2,n_el); d: (2,2); row_next: (n_rows,2*n_el+2);
    detector_rows: (n_rows,2).  Returns row_prev and F=(n_rows,2).
    h and gamma are the same real numbers used by the forward midpoint step.
    """
    u = np.asarray(u, np.complex128)
    d = np.asarray(d, np.complex128)
    rn = np.asarray(row_next, np.complex128)
    det = np.asarray(detector_rows, np.complex128)
    if u.ndim != 2 or u.shape[0] != 2 or d.shape != (2, 2):
        raise ValueError("Expected u=(2,n_el), d=(2,2)")
    ne = u.shape[1]
    if rn.ndim != 2 or rn.shape[1] != 2*ne+2 or det.shape != (rn.shape[0], 2):
        raise ValueError("Inconsistent row or detector dimensions")
    if not np.isfinite(h) or h <= 0 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Positive h and nonnegative gamma required")
    if np.max(np.abs(d.imag)) > 1e-11 or np.max(np.abs(d-d.T)) > 1e-11:
        raise ValueError("The original real symmetric contact is required")

    uc = u.conj()
    q_b, q_bd, q_a = det @ uc, det @ u, -det @ d
    sb = rn[:, :ne] + 0.5*q_b
    sd = rn[:, ne:2*ne] + 0.5*q_bd
    sa = rn[:, 2*ne:] + 0.5*q_a
    lhs = (np.eye(2) + (h*gamma/2)*d
           - (1j*h*h*gamma/4)*(uc @ u.T - u @ uc.T))
    st = 1j*h*(sb @ u.T - sd @ uc.T) + 2*sa
    # Z = st @ inv(lhs).  Ordinary transpose is essential here.
    zz = np.linalg.solve(lhs.T, st.T).T
    prev = np.empty_like(rn)
    prev[:, :ne] = 0.5*q_b + sb + (h*gamma/2)*(zz @ uc)
    prev[:, ne:2*ne] = 0.5*q_bd + sd + (h*gamma/2)*(zz @ u)
    prev[:, 2*ne:] = 0.5*q_a - sa + zz
    return prev, zz/2


def backward_rows(u, d, k, pol, h, gamma, *, progress: Callable | None = None):
    """Return output rows and source weights for one driven/undriven path.

    Arrays (including read-only memmaps) have u=(Nt,2,n_el), d=(Nt,2,2),
    k=(Nt,), pol=(n_modes,2).  The returned first n_modes rows generate raw U;
    the second group generates conj(raw V).  All array algebra is complex128.
    progress, if provided, is called as progress(completed_steps,total_steps).
    """
    started = time.perf_counter()
    if len(u.shape) != 3 or u.shape[1] != 2:
        raise ValueError("Expected u=(Nt,2,n_el)")
    nt, _, ne = u.shape
    if nt < 1 or d.shape != (nt, 2, 2):
        raise ValueError("Expected a nonempty path and d=(Nt,2,2)")
    k = np.asarray(k, np.complex128)
    pol = np.asarray(pol, np.complex128)
    if k.shape != (nt,) or pol.ndim != 2 or pol.shape[1] != 2 or pol.shape[0] < 1:
        raise ValueError("Expected k=(Nt,), pol=(n_modes,2)")
    if not np.isfinite(k).all() or not np.isfinite(pol).all():
        raise ValueError("Nonfinite detector kernel")
    nm = pol.shape[0]
    row = np.zeros((2*nm, 2*ne+2), np.complex128)
    forcing = np.empty((nt, 2*nm, 2), np.complex128)
    for count, j in enumerate(range(nt-1, -1, -1), start=1):
        det = h*np.concatenate((k[j]*pol, k[j].conjugate()*pol), axis=0)
        row, forcing[j] = midpoint_output_pullback(row, u[j], d[j], h, gamma, det)
        if progress is not None:
            progress(count, nt)
    if not np.isfinite(row).all() or not np.isfinite(forcing).all():
        raise FloatingPointError("Nonfinite output transpose coefficients")
    return BackwardRows(row, forcing, nm, nt, float(h), time.perf_counter()-started)


def contract_columns(rows, b, bdag, A, omega, free_A0, *, t0=0.0,
                     frequency_block=128, time_block=2048):
    """Apply backward rows to selected original initial input columns.

    omega=(ncol,), free_A0=(2,ncol) include all selected columns in their
    existing order.  For dark columns free_A0 must be zero and omega may be
    zero.  Existing sqrt(dnu) weights are already inside input coefficients;
    this function neither adds weights nor normalizes/reorders columns.

    The forcing is evaluated in time/frequency tiles; no Nt*ncol matrix is
    constructed.  Returns raw_u/raw_v before driven-undriven subtraction.
    """
    started = time.perf_counter()
    b, bd, a = (np.asarray(x, np.complex128) for x in (b, bdag, A))
    om = np.asarray(omega, np.float64)
    af = np.asarray(free_A0, np.complex128)
    if b.ndim != 2 or bd.shape != b.shape or a.shape != (2, b.shape[1]):
        raise ValueError("Inconsistent initial state arrays")
    ne, nc = b.shape
    if rows.row0.shape != (2*rows.n_modes, 2*ne+2):
        raise ValueError("Initial state and backward row dimensions differ")
    if om.shape != (nc,) or af.shape != (2, nc):
        raise ValueError("omega/free_A0 must include every selected column")
    if (not np.isfinite(om).all() or np.any(om < 0)
            or not np.isfinite(af).all() or not np.isfinite(t0)):
        raise ValueError("Invalid free-field input")
    if np.any((om == 0) & np.any(af != 0, axis=0)):
        raise ValueError("Zero frequency is only allowed for zero-forcing columns")
    if frequency_block < 1 or time_block < 1:
        raise ValueError("Positive tiling sizes required")
    rr = rows.row0
    yy = rr[:, :ne] @ b + rr[:, ne:2*ne] @ bd + rr[:, 2*ne:] @ a
    active = np.flatnonzero(np.any(af != 0, axis=0))
    nrows = 2*rows.n_modes
    for start in range(0, active.size, int(frequency_block)):
        ids = active[start:start+int(frequency_block)]
        freqs = om[ids]
        integral = np.zeros((nrows, 2, ids.size), np.complex128)
        for tj in range(0, rows.n_steps, int(time_block)):
            stop = min(rows.n_steps, tj+int(time_block))
            times = float(t0) + rows.h*np.arange(tj, stop, dtype=np.float64)
            phase = np.exp(-1j*times[:, None]*freqs[None, :])
            fw = rows.forcing[tj:stop].transpose(1, 2, 0).reshape(nrows*2, stop-tj)
            integral += (fw @ phase).reshape(nrows, 2, ids.size)
        integral *= np.expm1(-1j*freqs*rows.h)[None, None, :]
        yy[:, ids] += np.sum(integral*af[None, :, ids], axis=1)
    if not np.isfinite(yy).all():
        raise FloatingPointError("Nonfinite contracted output")
    nm = rows.n_modes
    return {"raw_u": yy[:nm], "raw_v": yy[nm:].conj(),
            "wall_s": time.perf_counter()-started,
            "frequency_block": int(frequency_block), "time_block": int(time_block),
            "n_columns": int(nc), "n_forced_columns": int(active.size)}


def _forward_fixture(u, d, k, pol, h, gamma, x0, omega, af, t0):
    from q8_m0_continuum_columns_20261004 import ColumnState, midpoint_step, current
    ne = u.shape[2]
    state = ColumnState.unpack(x0, ne)
    out_u = np.zeros((pol.shape[0], x0.shape[1]), complex)
    out_v = np.zeros_like(out_u)
    for j in range(u.shape[0]):
        tt = t0+j*h
        ff = af*np.exp(-1j*omega*tt)[None, :]*np.expm1(-1j*omega*h)[None, :]
        previous = state
        state = midpoint_step(state, u[j], d[j], h, gamma, free_A_increment=ff)
        mid = ColumnState.unpack((previous.packed()+state.packed())/2, ne)
        source = pol @ current(mid, u[j], d[j])
        out_u += h*k[j]*source
        out_v += h*k[j]*source.conj()
    return out_u, out_v


def selftest():
    """Independent forward comparisons; no physical result is claimed."""
    rng = np.random.default_rng(815)
    records = []
    for name, gamma, nm in (("complex_noncommuting", .13, 2),
                            ("zero_feedback_three_modes", 0., 3)):
        ne, nc, nt, h, t0 = 7, 11, 41, .031, .27
        zz = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
        u = .3*zz((nt, 2, ne))
        dd = rng.normal(size=(nt, 2, 2))
        d = .1*(dd @ dd.transpose(0, 2, 1))
        pol, k = zz((nm, 2)), zz((nt,))
        omega, af = np.linspace(.04, 2.9, nc), zz((2, nc))
        af[:, -3:] = 0
        omega[-3:] = 0
        x0 = zz((2*ne+2, nc))
        fu, fv = _forward_fixture(u, d, k, pol, h, gamma, x0, omega, af, t0)
        rows = backward_rows(u, d, k, pol, h, gamma)
        out = contract_columns(rows, x0[:ne], x0[ne:2*ne], x0[2*ne:],
                               omega, af, t0=t0, frequency_block=3, time_block=9)
        alternate = contract_columns(rows, x0[:ne], x0[ne:2*ne], x0[2*ne:],
                                     omega, af, t0=t0, frequency_block=11, time_block=41)
        eu = float(np.max(np.abs(fu-out["raw_u"])))
        ev = float(np.max(np.abs(fv-out["raw_v"])))
        etile = float(max(np.max(np.abs(out[key]-alternate[key]))
                         for key in ("raw_u", "raw_v")))
        # An independent dense formula audits the low-rank single-step pullback.
        jj, testrow = 9, zz((2*nm, 2*ne+2))
        uc = u[jj].conj()
        ll = (np.eye(2)+h*gamma/2*d[jj]
              -1j*h*h*gamma/4*(uc @ u[jj].T-u[jj] @ uc.T))
        rr = np.hstack((h*gamma/2*uc, h*gamma/2*u[jj], np.eye(2)))
        tt = np.vstack((1j*h*u[jj].T, -1j*h*uc.T, 2*np.eye(2)))
        dd0 = np.diag(np.r_[np.ones(2*ne), -np.ones(2)])
        mm, nn = dd0+tt @ np.linalg.solve(ll, rr), .5*tt @ np.linalg.inv(ll)
        det = h*np.vstack((k[jj]*pol, k[jj].conjugate()*pol))
        qq = det @ np.hstack((uc, u[jj], -d[jj]))
        pr, ff = midpoint_output_pullback(testrow, u[jj], d[jj], h, gamma, det)
        dense_error = float(max(np.max(abs(pr-(qq/2+(testrow+qq/2) @ mm))),
                                np.max(abs(ff-(testrow+qq/2) @ nn))))
        bad_u, bad_v = _forward_fixture(u, d, k, pol.conj(), h, gamma,
                                       x0, omega, af, t0)
        wrong_pol_error = float(np.max(abs(bad_v-fv)))
        passed = max(eu, ev, etile, dense_error) < 2e-12 and wrong_pol_error > 1e-3
        records.append({"name": name, "n_el": ne, "n_columns": nc,
                        "n_steps": nt, "n_modes": nm, "gamma": gamma,
                        "U_max_abs": eu, "V_max_abs": ev,
                        "tiling_max_abs": etile, "dense_pullback_max_abs": dense_error,
                        "wrong_pol_conjugation_detected_abs": wrong_pol_error,
                        "backward_wall_s": rows.elapsed_s, "passed": bool(passed)})
    return {"status": "small_discrete_transpose_fixture",
            "all_checks_pass": bool(all(r["passed"] for r in records)),
            "dtype": "complex128", "cases": records,
            "physical_claim": "none; exact same finite discrete functional only",
            "production_code_modified": False, "real_gamma12_tested": False,
            "speedup_claimed": False,
            "python": platform.python_version(), "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["selftest"])
    parser.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[2]
        / "results/q_line/gamma12_output_transpose_20261008/selftest.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refuse to overwrite existing artifact: {args.output}")
    result = selftest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

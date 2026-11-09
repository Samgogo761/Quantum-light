#!/usr/bin/env python3
"""Stage 1AC: manufactured-solution test for the joint time propagators.

A small K_test(t) is built with the same algebraic type as the real joint
generator: K = J H(t), H(t) symmetric, [K(t1), K(t2)] != 0, and a fast phase
comparable to the electronic vertex frequencies seen in stage-1 diagnostics.

A high-precision reference (scipy DOP853 on the matrix ODE) is used to verify
the observed convergence orders of

1. Cayley midpoint          (expected order 2)
2. commutator-free Magnus 4 (expected order 4)

Gate: no large coupled run is allowed before both orders are reproduced here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_phgauss import symplectic_J  # noqa: E402
from q8_m0_stage1w_propagator_crosscheck import expm_nopy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1ac_mms_20260923"

# CF4 (Blanes-Moan commutator-free Magnus, order 4, two exponentials).
C1 = 0.5 - np.sqrt(3.0) / 6.0
C2 = 0.5 + np.sqrt(3.0) / 6.0
ALPHA = 0.25 + np.sqrt(3.0) / 6.0  # weight on the matching-stage K
BETA = 0.25 - np.sqrt(3.0) / 6.0   # weight on the other-stage K


def make_generator(n_osc: int, seed: int = 20260923):
    """Return K(t) = J H(t) with noncommuting samples and fast phases."""
    rng = np.random.default_rng(seed)
    j = symplectic_J(n_osc)
    mats = []
    for _ in range(3):
        a = rng.normal(size=(2 * n_osc, 2 * n_osc))
        mats.append(a + a.T)
    om1, om2 = 17.0, 29.0  # fast relative to the window 2*pi

    def k_of_t(t: float) -> np.ndarray:
        h = (mats[0] + np.cos(om1 * t) * mats[1]
             + np.sin(om2 * t) * mats[2])
        return j @ h

    return k_of_t, j, (om1, om2)


def cayley_step(phi, kmat, dt):
    eye = np.eye(kmat.shape[0])
    return np.linalg.solve(eye - 0.5 * dt * kmat, (eye + 0.5 * dt * kmat) @ phi)


def cf4_step(phi, k1, k2, dt):
    o1 = dt * (ALPHA * k1 + BETA * k2)
    o2 = dt * (BETA * k1 + ALPHA * k2)
    return expm_nopy(o2) @ expm_nopy(o1) @ phi


def propagate(k_of_t, dim, dt, t_final, method):
    phi = np.eye(dim)
    n_steps = int(round(t_final / dt))
    for n in range(n_steps):
        t = n * dt
        if method == "cayley_midpoint":
            kmat = k_of_t(t + 0.5 * dt)
            phi = cayley_step(phi, kmat, dt)
        elif method == "cf4":
            k1 = k_of_t(t + C1 * dt)
            k2 = k_of_t(t + C2 * dt)
            phi = cf4_step(phi, k1, k2, dt)
        else:
            raise ValueError(method)
    return phi


def reference(k_of_t, dim, t_final):
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return (k_of_t(t) @ y.reshape(dim, dim)).reshape(-1)

    sol = solve_ivp(rhs, (0.0, t_final), np.eye(dim).reshape(-1),
                    method="DOP853", rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(dim, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-osc", type=int, default=4)
    ap.add_argument("--min-exp", type=int, default=4)
    ap.add_argument("--max-exp", type=int, default=10)
    args = ap.parse_args()

    n_osc = args.n_osc
    dim = 2 * n_osc
    t_final = 2.0 * np.pi
    k_of_t, j, (om1, om2) = make_generator(n_osc)

    # generator sanity: Hamiltonian and noncommuting
    ham = max(float(np.max(np.abs(j @ k_of_t(t) + k_of_t(t).T @ j)))
              for t in np.linspace(0.0, t_final, 17))
    comm = float(np.max(np.abs(
        k_of_t(0.3) @ k_of_t(1.1) - k_of_t(1.1) @ k_of_t(0.3))))

    print("reference (DOP853 rtol=1e-12) ...", flush=True)
    phi_ref = reference(k_of_t, dim, t_final)
    ref_norm = float(np.linalg.norm(phi_ref))

    rows = []
    for m in range(args.min_exp, args.max_exp + 1):
        dt = t_final / (2 ** m)
        row = {"dt": dt, "n_steps": 2 ** m}
        for method in ("cayley_midpoint", "cf4"):
            phi = propagate(k_of_t, dim, dt, t_final, method)
            err = float(np.linalg.norm(phi - phi_ref) / ref_norm)
            sym = float(np.max(np.abs(phi.T @ j @ phi - j)))
            row[f"{method}_rel_err"] = err
            row[f"{method}_symplectic_abs"] = sym
            print(f"dt=2pi/2^{m} {method:16s} err={err:.3e} sym={sym:.3e}",
                  flush=True)
        rows.append(row)

    def fit_order(key):
        hs = np.array([r["dt"] for r in rows])
        es = np.array([r[key] for r in rows])
        # use the finest four points where asymptotic regime holds
        p = np.polyfit(np.log(hs[-4:]), np.log(es[-4:]), 1)
        return float(p[0])

    payload = {
        "status": "manufactured_solution_gate",
        "n_osc": n_osc,
        "t_final": t_final,
        "fast_frequencies": [om1, om2],
        "generator_hamiltonian_maxres": ham,
        "generator_commutator_maxabs": comm,
        "reference": "scipy DOP853 rtol=1e-12 atol=1e-14",
        "rows": rows,
        "fitted_order_cayley_midpoint": fit_order("cayley_midpoint_rel_err"),
        "fitted_order_cf4": fit_order("cf4_rel_err"),
        "cf4_coefficients": {"c1": C1, "c2": C2, "alpha": ALPHA, "beta": BETA},
        "gate": "both fitted orders must be near 2 and 4 before any large run",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "stage1ac_mms.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "fitted_order_cayley_midpoint", "fitted_order_cf4",
        "generator_hamiltonian_maxres", "generator_commutator_maxabs")},
        indent=2), flush=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()

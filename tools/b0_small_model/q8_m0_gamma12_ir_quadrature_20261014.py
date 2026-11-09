"""Independent quadratures of only [1e-4, 1e-3]. Everything else stays put.

The saved baseline is wing factor 2, core 160, ntpc=4096, omega_lo=1e-4.
That interval contains 3 frequencies and therefore 12 input columns.
Gauss-Legendre uses dν. The logarithmic rule uses ν=e^x and the Jacobian e^x,
so both target the same flat measure ∫ f(ν) dν. sqrt(weight) is applied once
by the existing attach_weights path.
"""
from pathlib import Path
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q

OUT = base.OUT
LO, HI = 1.0e-4, 1.0e-3


def pack(m):
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def gl_panel(a, b, n):
    return q.gauss_segment(a, b, n)


def log_panel(a, b, n):
    xa, xb = np.log(a), np.log(b)
    x, w = np.polynomial.legendre.leggauss(int(n))
    xx = 0.5 * (xb - xa) * x + 0.5 * (xa + xb)
    om = np.exp(xx)
    wt = 0.5 * (xb - xa) * w * om
    return om, wt


def rows_of(cache, rows, gamma, pol, phi_of, eta, oms, wts):
    ph, u0 = base.generated_photon_block(cache, gamma, oms, wts, pol, phi_of, eta)
    got, _tt = base._contract(rows, (ph.initial.b, ph.initial.bdag, ph.initial.A),
                              ph.omega, ph.free_A0, 0.0, 128, 2048)
    du = got["driven_raw_u"] - got["undriven_raw_u"]
    v = got["driven_raw_v"] - got["undriven_raw_v"]
    return u0 + du, v, u0


def contrib(u, v):
    normal = u @ u.conj().T - v @ v.conj().T
    anomalous = u @ v.T - v @ u.T
    return normal, anomalous


def main():
    started = time.perf_counter()
    manifest, cache, rows, _info, _identity = base.load_frozen()
    gamma = float(cache["gamma"])
    pol = np.array(cache["pol"], copy=True)
    prot = base.p1_a2_protocol(base.q.A_STAR)
    op = base.q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, norm = base.q.normalize_phi(op, base.q.phi_unnormalized(op, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    saved = np.load(OUT / "endpoint_low.npz")
    oms, wts = saved["oms"], saved["wts"]
    mask = (oms >= LO) & (oms < HI)
    if int(mask.sum()) != 3:
        raise AssertionError(f"Expected 3 infrared frequencies, found {int(mask.sum())}")
    cols = np.repeat(np.flatnonzero(mask), 4) * 4
    cols = (4 * np.flatnonzero(mask)[:, None] + np.arange(4)).ravel()
    u_old, v_old = saved["U"][:, cols], saved["V"][:, cols]
    n_old, a_old = contrib(u_old, v_old)
    n_all = saved["U"] @ saved["U"].conj().T - saved["V"] @ saved["V"].conj().T
    a_all = saved["U"] @ saved["V"].T - saved["V"] @ saved["U"].T
    # Wiring: rebuild the saved nodes with the same weights. sqrt(w) once.
    u_re, v_re, _u0 = rows_of(cache, rows, gamma, pol, phi_of, eta, oms[mask], wts[mask])
    wiring = {
        "U_maxabs": float(np.max(np.abs(u_re - u_old))),
        "V_maxabs": float(np.max(np.abs(v_re - v_old))),
        "saved_weight_sum": float(wts[mask].sum()),
        "interval_length": HI - LO,
        "n_frequencies": 3,
        "n_columns": 12,
    }
    if wiring["U_maxabs"] > 1e-8 or wiring["V_maxabs"] > 1e-8:
        raise AssertionError(f"Saved infrared columns were not reproduced: {wiring}")
    bare = rows_of(cache, rows, gamma, pol, phi_of, eta, oms[mask], np.ones(3))
    # ones() still goes through sqrt once, so this is sqrt(1) times the bare column.
    scale = np.sqrt(wts[mask])
    u_scaled = np.repeat(scale, 4)[None, :] * bare[0]
    # generated_photon_block with weight 1 applies sqrt(1)=1. Stored columns used sqrt(w).
    wiring["sqrt_weight_reconstruction_U"] = float(np.max(np.abs(u_scaled - u_old)))
    methods = {"saved_sinh_tail": (oms[mask], wts[mask], n_old, a_old)}
    for n in (8, 16, 32):
        methods[f"gauss_legendre_{n}"] = (*gl_panel(LO, HI, n), None, None)
        methods[f"log_frequency_{n}"] = (*log_panel(LO, HI, n), None, None)
    report = {"wiring": wiring, "methods": {}, "interval": [LO, HI],
              "fixed": {"ntpc": 4096, "wing_factor": 2, "core_n_panel": 160,
                        "omega_hi": 12.0, "sqrt_weight_once": True},
              "production_qualification": False}
    for name, (om, wt, nu, au) in methods.items():
        if nu is None:
            u, v, _ = rows_of(cache, rows, gamma, pol, phi_of, eta, om, wt)
            nu, au = contrib(u, v)
        n_full = n_all - n_old + nu
        a_full = a_all - a_old + au
        rn, ra = n_full - np.eye(2), a_full
        report["methods"][name] = {
            "n_frequencies": int(len(om)),
            "weight_sum": float(np.sum(wt)),
            "band_normal": pack(nu),
            "band_anomalous": pack(au),
            "full_normal_residual": pack(rn),
            "full_anomalous_residual": pack(ra),
            "full_normal_maxabs": float(np.max(np.abs(rn))),
            "full_anomalous_maxabs": float(np.max(np.abs(ra))),
        }
        print(name, "nodes", len(om), "w", float(np.sum(wt)),
              "Rmax", report["methods"][name]["full_normal_maxabs"],
              "Amax", report["methods"][name]["full_anomalous_maxabs"], flush=True)
    report["elapsed_s"] = time.perf_counter() - started
    path = OUT / "ir_quadrature_crosscheck.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wiring", wiring)


if __name__ == "__main__":
    main()

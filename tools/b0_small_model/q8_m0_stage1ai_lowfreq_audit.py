#!/usr/bin/env python3
"""Audit the frozen, zero-pump electronic response, without photon closure.

K = -R^A for I = -dH/dkappa. Frequencies denote hbar*omega/t_h,
kappa denotes inverse-Angstrom momentum shift, and K is per cell.
The stable increment is K(z)-K(0), not a replacement for K(z).
Existing stage1ai output is read and preserved.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

from q8_m0_classical import k_grid
from q8_m0_stage1ai_lowfreq_kernel import (
    OMEGA0_OVER_TH, OMEGA_OVER_OMEGA0, pack_matrix, power_fit,
)
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases, parallel_perp_basis,
)


def spectral_data(gap, x, weights):
    """C_ab = sum w X_a* X_b, retaining every c,v transition separately."""
    gap = np.asarray(gap, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(gap <= 0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Positive occupied-empty gaps and normalized weights required")
    c = np.einsum("k,kacv,kbcv->abkcv", weights, x.conj(), x)
    return gap.ravel(), c.reshape(2, 2, -1)


def kernel(d, gap, c, z):
    """Retarded K(z)=D+sum[C/(z-gap)-C*/(z+gap)], Im z >= 0.

Real-axis calls are used only strictly below the smallest gap. No damping
or finite linewidth is introduced into the physical calculation.
"""
    z = complex(z)
    if z.imag < 0 or (z.imag == 0 and abs(z.real) >= np.min(gap)):
        raise ValueError("Use upper half-plane or real subgap frequencies only")
    return (np.asarray(d, complex)
            + np.einsum("abp,p->ab", c, 1 / (z - gap))
            - np.einsum("abp,p->ab", c.conj(), 1 / (z + gap)))


def increment(gap, c, z):
    """Exact algebraic K(z)-K(0), avoiding subtraction of large D/Pi.

Includes the potentially linear, imaginary antisymmetric contribution;
no even-frequency assumption is imposed on the tensor.
"""
    z = complex(z)
    den = gap**2 - z**2
    even = -2 * z**2 * np.einsum("abp,p->ab", c.real, 1 / (gap * den))
    odd = -2j * z * np.einsum("abp,p->ab", c.imag, 1 / den)
    return even + odd, even, odd


def expansion(gap, c):
    """K(z)=K(0)+z L+z^2 Q+..., without altering K(0)."""
    linear = -2j * np.einsum("abp,p->ab", c.imag, 1 / gap**2)
    quadratic = -2 * np.einsum("abp,p->ab", c.real, 1 / gap**3)
    return linear, quadratic


def main():
    root = REPO / "results/q_line/stage1ai_ir_kernel_20260924"
    old_path = root / "lowfreq_kernel.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    geo = build_geometry()
    ep, et = parallel_perp_basis(geo)
    frequencies = OMEGA0_OVER_TH * np.array(OMEGA_OVER_OMEGA0)
    rows = []
    for name, n, shift in (("Gamma_96x96", 96, 0.),
                           ("Gamma_128x128", 128, 0.),
                           ("Gamma_192x192", 192, 0.),
                           ("halfshift_192x192", 192, .5)):
        _, kc, weights = k_grid(geo, n, shift=shift)
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
        x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
        gaps = en[:, 4:, None] - en[:, None, :4]
        gap, c = spectral_data(gaps, x, weights)
        d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
        k0 = kernel(d, gap, c, 0)
        linear, quadratic = expansion(gap, c)
        prior = next(g for g in old["grids"] if g["grid"] == name)
        checks = {"raw_matches_predecessor_maxabs": 0.,
                  "reality_K_minus_w_equals_K_w_conjugate_maxabs": 0.,
                  "subgap_hermiticity_maxabs": 0.,
                  "increment_vs_raw_difference_maxabs": 0.}
        samples = []
        for w, prior_sample in zip(frequencies, prior["kernel"]):
            kw = kernel(d, gap, c, w)
            km = kernel(d, gap, c, -w)
            delta, even, odd = increment(gap, c, w)
            pk = np.array(prior_sample["K"]["real"]) + 1j * np.array(prior_sample["K"]["imag"])
            for key, value in (
                ("raw_matches_predecessor_maxabs", np.max(abs(kw-pk))),
                ("reality_K_minus_w_equals_K_w_conjugate_maxabs", np.max(abs(km-kw.conj()))),
                ("subgap_hermiticity_maxabs", np.max(abs(kw-kw.conj().T))),
                ("increment_vs_raw_difference_maxabs", np.max(abs(delta-(kw-k0)))),
            ):
                checks[key] = max(checks[key], float(value))
            samples.append({"omega": float(w), "omega_over_omega0": float(w / OMEGA0_OVER_TH),
                            "K_raw": pack_matrix(kw),
                            "K_minus_K0_stable": pack_matrix(delta),
                            "even_increment_maxabs": float(np.max(abs(even))),
                            "odd_increment_maxabs": float(np.max(abs(odd))),
                            "parallel_delta_real": float((ep @ delta @ ep).real),
                            "perp_delta_real": float((et @ delta @ et).real),
                            "static_residual_over_increment": float(np.max(abs(k0))/np.max(abs(delta))) if w else None})
        ir = [s for s in samples if 0 < s["omega_over_omega0"] <= .030000001]
        result = {"grid": name, "n_k": int(len(kc)), "shift": shift,
                  "min_direct_gap": float(np.min(gap)),
                  "static_K0": pack_matrix(k0),
                  "linear_coefficient": pack_matrix(linear),
                  "quadratic_coefficient": pack_matrix(quadratic),
                  "quadratic_parallel": float(ep @ quadratic @ ep),
                  "quadratic_perp": float(et @ quadratic @ et),
                  "checks": checks, "samples": samples,
                  "stable_increment_IR_power_fit": power_fit(
                      [s["omega"] for s in ir], [s["K_minus_K0_stable"]["maxabs"] for s in ir])}
        rows.append(result)
        print(name, "K0", result["static_K0"]["maxabs"],
              "Q_parallel/perp", result["quadratic_parallel"], result["quadratic_perp"],
              "L", result["linear_coefficient"]["maxabs"], flush=True)
    def unpack(p):
        return np.array(p["real"]) + 1j*np.array(p["imag"])
    comparisons = []
    for a, b in ((1, 2), (2, 3)):
        ra, rb = rows[a], rows[b]
        qa, qb = unpack(ra["quadratic_coefficient"]), unpack(rb["quadratic_coefficient"])
        comparison = {"grids": [ra["grid"], rb["grid"]],
                      "quadratic_coefficient_relative_maxnorm": float(np.max(abs(qa-qb))/np.max(abs(qb))),
                      "frequency_checks": []}
        for sa, sb in zip(ra["samples"], rb["samples"]):
            ka, kb = unpack(sa["K_raw"]), unpack(sb["K_raw"])
            da, db = unpack(sa["K_minus_K0_stable"]), unpack(sb["K_minus_K0_stable"])
            comparison["frequency_checks"].append({
                "omega_over_omega0": sa["omega_over_omega0"],
                "raw_absolute_maxnorm_difference": float(np.max(abs(ka-kb))),
                "raw_relative_maxnorm_difference": float(np.max(abs(ka-kb))/np.max(abs(kb))),
                "increment_relative_maxnorm_difference": float(np.max(abs(da-db))/np.max(abs(db))) if sa["omega"] else None,
            })
        comparisons.append(comparison)
    sources = [old_path, Path(__file__).resolve(),
               REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_kernel.py",
               REPO / "tools/b0_small_model/q8_m0_model.py",
               REPO / "tools/b0_small_model/q8_m0_response.py"]
    payload = {"status": "zero_pump_electronic_low_frequency_audit_only",
               "definition": "K=-R^kappa; delta I=-K delta kappa; e=hbar=1",
               "units": {"frequency": "hbar omega/t_h", "kappa": "Angstrom^-1",
                         "K": "t_h Angstrom^2 per cell"},
               "static_remainder_retained": True,
               "increment_is_diagnostic_not_counterterm": True,
               "physical_linewidth": 0,
               "polarizations": {"parallel": ep.tolist(), "perp": et.tolist()},
               "grids": rows, "comparisons": comparisons,
               "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "not_established": ["joint stability or poles", "vacuum IR convergence", "initial state choice",
                                    "Pauli redefinition", "driven P1 response", "quantum output"]}
    out = root / "audit"
    out.mkdir(exist_ok=True)
    (out / "lowfreq_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

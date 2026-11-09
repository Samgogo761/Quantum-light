#!/usr/bin/env python3
"""P1=0 full-BZ self-consistent Maxwell pole audit.

Closed loop on the frozen sheet:
    dκ_self/dt = γ I_cell,
Fourier convention of the existing retarded tests: ∫ dt e^{+i z t},
so d/dt → -i z when transients at t0 are dropped for eigenmodes,
    κ_self(z) = (i γ / z) I(z).
Electronics: I = -K κ, K = -R^κ. Zero input ⇒ κ_loc = κ_self, hence
    F(z) I = 0,   F(z) = I_2 + (i γ / z) K(z).

Dual ledger: raw K_N versus ΔK = K_N - K_N(0). The increment uses the
already-audited ΔK/z formula, so z=0 is not an artificial root.
Argument principle on open half-plane rectangles (real-axis p-h poles
stay outside). No noise, initial state, Pauli, or P1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import k_grid
from q8_m0_gsheet import gamma_self
from q8_m0_stage1ag_acceptance_audit import (
    REPO, MU, R0, build_geometry, h8, current_ops, occup_empty_bases,
    x_from_bases, d_ops, mean_d_from_bases,
)
from q8_m0_stage1ai_lowfreq_kernel import OMEGA0_OVER_TH, pack_matrix
from q8_m0_stage1ai_lowfreq_audit import expansion, spectral_data

OUT = REPO / "results/q_line/stage1al_pole_audit_20260924"
GRIDS = (
    ("Gamma_96x96", 96, 0.0),
    ("Gamma_128x128", 128, 0.0),
    ("Gamma_192x192", 192, 0.0),
    ("halfshift_192x192", 192, 0.5),
)
MAX_CONTOUR_POINTS = 2500
MAX_REFINE_PASSES = 8
MAX_DARG = 0.5


def kernel_off_cuts(d, gap, c, zs, chunk=None):
    """K(z)=D+Σ[C/(z-Δ)-C*/(z+Δ)] anywhere off the real-axis poles."""
    del chunk
    zs = np.asarray(zs, dtype=complex).reshape(-1)
    d = np.asarray(d, dtype=complex)
    out = np.empty((zs.size, 2, 2), dtype=complex)
    cc = np.conjugate(c)
    for n, z in enumerate(zs):
        out[n] = (
            d
            + np.einsum("abp,p->ab", c, 1.0 / (z - gap))
            - np.einsum("abp,p->ab", cc, 1.0 / (z + gap))
        )
    return out


def increment_over_z(gap, c, zs, chunk=None):
    """Stable ΔK(z)/z, including the linear (odd) piece. Finite at z=0."""
    del chunk
    zs = np.asarray(zs, dtype=complex).reshape(-1)
    out = np.empty((zs.size, 2, 2), dtype=complex)
    cre, cim = np.real(c), np.imag(c)
    g2 = gap * gap
    for n, z in enumerate(zs):
        den = g2 - z * z
        out[n] = (
            np.einsum("abp,p->ab", cre, (-2.0 * z) / (gap * den))
            + np.einsum("abp,p->ab", cim, (-2.0j) / den)
        )
    return out


def closed_loop_F(k_or_delta_over_z, z, gamma, *, from_increment):
    """F = I + iγ (K/z). from_increment: argument is already ΔK/z."""
    eye = np.eye(2, dtype=complex)
    z = np.asarray(z, dtype=complex).reshape(-1)
    m = np.asarray(k_or_delta_over_z, dtype=complex)
    if m.ndim == 2:
        m = m[None, ...]
    if from_increment:
        return eye + (1j * gamma) * m
    return eye + (1j * gamma) * m / z[:, None, None]


def det2(f):
    f = np.asarray(f, dtype=complex)
    return f[..., 0, 0] * f[..., 1, 1] - f[..., 0, 1] * f[..., 1, 0]


def rectangle_polygon(re_min, re_max, im_min, im_max, n_re, n_im):
    """Closed CCW rectangle, last point repeats the first."""
    bottom = np.linspace(re_min, re_max, n_re) + 1j * im_min
    right = re_max + 1j * np.linspace(im_min, im_max, n_im)[1:]
    top = np.linspace(re_max, re_min, n_re)[1:] + 1j * im_max
    left = re_min + 1j * np.linspace(im_max, im_min, n_im)[1:]
    return np.concatenate([bottom, right, top, left])


def eval_det_on(d, gap, c, k0, zs, gamma, ledger):
    zs = np.asarray(zs, dtype=complex).reshape(-1)
    if ledger == "increment":
        dkz = increment_over_z(gap, c, zs)
        f = closed_loop_F(dkz, zs, gamma, from_increment=True)
    elif ledger == "raw":
        k = kernel_off_cuts(d, gap, c, zs)
        f = closed_loop_F(k, zs, gamma, from_increment=False)
    else:
        raise ValueError(ledger)
    return det2(f), f


def adaptive_winding(d, gap, c, k0, gamma, ledger, z_poly, *,
                     max_darg=MAX_DARG, max_points=MAX_CONTOUR_POINTS,
                     max_passes=MAX_REFINE_PASSES, refine=True):
    z = np.asarray(z_poly, dtype=complex).reshape(-1)
    if z.size < 2 or abs(z[0] - z[-1]) > 1e-14 * max(1.0, abs(z[0])):
        z = np.concatenate([z, z[:1]])
    det, _ = eval_det_on(d, gap, c, k0, z, gamma, ledger)
    saturated = False
    n_pass = max_passes if refine else 0
    for _ in range(n_pass):
        ang = np.unwrap(np.angle(det))
        dang = np.diff(ang)
        worst = np.max(np.abs(dang)) if dang.size else 0.0
        if worst <= max_darg:
            break
        if z.size >= max_points:
            saturated = True
            break
        insert_at = np.nonzero(np.abs(dang) > max_darg)[0]
        if insert_at.size == 0:
            break
        if z.size + insert_at.size > max_points:
            insert_at = insert_at[: max(0, max_points - z.size)]
            saturated = True
        mids = 0.5 * (z[insert_at] + z[insert_at + 1])
        dmid, _ = eval_det_on(d, gap, c, k0, mids, gamma, ledger)
        new_z = np.empty(z.size + mids.size, dtype=complex)
        new_det = np.empty(z.size + mids.size, dtype=complex)
        src = np.arange(z.size)
        dst = src + np.concatenate([[0], np.cumsum(np.isin(src[:-1], insert_at))])
        new_z[dst] = z
        new_det[dst] = det
        new_z[dst[insert_at] + 1] = mids
        new_det[dst[insert_at] + 1] = dmid
        z, det = new_z, new_det
    ang = np.unwrap(np.angle(det))
    wind = float((ang[-1] - ang[0]) / (2.0 * np.pi))
    return {
        "n_points": int(z.size),
        "winding_raw": wind,
        "winding_nearest_int": int(np.rint(wind)),
        "winding_integer_residual": float(abs(wind - np.rint(wind))),
        "max_abs_darg": float(np.max(np.abs(np.diff(ang)))) if z.size > 1 else 0.0,
        "min_abs_det": float(np.min(np.abs(det))),
        "max_abs_det": float(np.max(np.abs(det))),
        "log10_min_abs_det": float(np.log10(max(np.min(np.abs(det)), 1e-300))),
        "closed_det_rel_gap": float(abs(det[-1] - det[0]) / max(abs(det[0]), 1e-30)),
        "refinement_saturated": bool(saturated),
        "refined": bool(refine),
    }


def newton_det_zero(d, gap, c, k0, gamma, ledger, z0, niter=16):
    z = complex(z0)
    history = []
    h = 1e-8 + 1e-8j
    for _ in range(niter):
        det, _ = eval_det_on(d, gap, c, k0, np.array([z, z + h]), gamma, ledger)
        val = det[0]
        der = (det[1] - det[0]) / h
        history.append({"z_real": z.real, "z_imag": z.imag, "abs_det": float(abs(val))})
        if abs(val) < 1e-14 or abs(der) < 1e-30:
            break
        z = z - val / der
    det, _ = eval_det_on(d, gap, c, k0, z, gamma, ledger)
    return {
        "z_real": float(z.real),
        "z_imag": float(z.imag),
        "abs_det": float(np.abs(det[0])),
        "converged": bool(np.abs(det[0]) < 1e-10),
        "steps": history[-4:],
    }


def predicted_raw_soft_poles(k0, gamma):
    """Finite-grid fake poles of I + iγ M_N/z at z = -i γ λ(M_N)."""
    evals = np.linalg.eigvals(np.asarray(k0, dtype=complex))
    poles = []
    for lam in evals:
        z = -1j * gamma * lam
        poles.append({
            "static_eigenvalue": {"real": float(lam.real), "imag": float(lam.imag)},
            "z_real": float(z.real),
            "z_imag": float(z.imag),
            "in_uhp": bool(z.imag > 0.0),
            "in_lhp": bool(z.imag < 0.0),
        })
    return poles


def contour_spec(min_gap, max_gap):
    ir_half = 0.80 * float(min_gap)
    wide_half = max(1.25 * float(max_gap), 8.0)
    return (
        {
            "name": "IR_subgap",
            "re_half": ir_half,
            "im_max": ir_half,
            "n_re": 81,
            "n_im": 41,
            "im_min_increment": 1e-6,
            "im_min_raw": 1e-6,
        },
        {
            "name": "particle_hole_scale_off_cut",
            "re_half": wide_half,
            "im_max": 2.0,
            "n_re": 121,
            "n_im": 41,
            "im_min_increment": 5e-2,
            "im_min_raw": 5e-2,
        },
    )


def raw_soft_box_spec(k0, gamma):
    """Tiny UHP box that should contain the finite-grid fake poles, not z=0."""
    poles = predicted_raw_soft_poles(k0, gamma)
    ims = [abs(p["z_imag"]) for p in poles if p["in_uhp"] or p["in_lhp"]]
    if not ims:
        return None
    im_hi = max(20.0 * max(ims), 1e-8)
    im_lo = min(ims) / 20.0 if min(ims) > 0 else 1e-12
    im_lo = max(im_lo, 1e-12)
    re_half = max(im_hi, 1e-8)
    return {
        "name": "raw_soft_box",
        "re_half": re_half,
        "im_max": im_hi,
        "n_re": 161,
        "n_im": 81,
        "im_min_increment": im_lo,
        "im_min_raw": im_lo,
    }


def near_cut_det_scan(d, gap, c, k0, gamma, ledger, re_max, etas, n_omega):
    rows = []
    omega = np.linspace(-re_max, re_max, n_omega)
    for eta in etas:
        zs = omega + 1j * float(eta)
        det, _ = eval_det_on(d, gap, c, k0, zs, gamma, ledger)
        i = int(np.argmin(np.abs(det)))
        rows.append({
            "eta": float(eta),
            "n_omega": int(n_omega),
            "min_abs_det": float(np.abs(det[i])),
            "omega_at_min": float(omega[i]),
            "log10_min_abs_det": float(np.log10(max(np.abs(det[i]), 1e-300))),
        })
    return rows


def run_halfplane(d, gap, c, k0, gamma, ledger, layer, half):
    im_min = float(layer["im_min_raw" if ledger == "raw" else "im_min_increment"])
    re_half = float(layer["re_half"])
    im_max = float(layer["im_max"])
    if half == "UHP":
        poly = rectangle_polygon(-re_half, re_half, im_min, im_max,
                                 layer["n_re"], layer["n_im"])
    elif half == "LHP":
        poly = rectangle_polygon(-re_half, re_half, -im_max, -im_min,
                                 layer["n_re"], layer["n_im"])
    else:
        raise ValueError(half)
    rec = adaptive_winding(
        d, gap, c, k0, gamma, ledger, poly,
        refine=True,
    )
    rec.update({
        "halfplane": half,
        "layer": layer["name"],
        "ledger": ledger,
        "re_half": re_half,
        "im_min": im_min,
        "im_max": im_max if half == "UHP" else -im_max,
    })
    if rec["winding_nearest_int"] != 0:
        seed_re = np.linspace(-0.6 * re_half, 0.6 * re_half, 9)
        if half == "UHP":
            seed_im = np.linspace(2.0 * im_min, 0.6 * im_max, 6)
        else:
            seed_im = np.linspace(-0.6 * im_max, -2.0 * im_min, 6)
        seeds = (seed_re[:, None] + 1j * seed_im[None, :]).ravel()
        if ledger == "raw":
            extra = np.array(
                [p["z_real"] + 1j * p["z_imag"] for p in predicted_raw_soft_poles(k0, gamma)],
                dtype=complex,
            )
            seeds = np.concatenate([seeds, extra])
        det_s, _ = eval_det_on(d, gap, c, k0, seeds, gamma, ledger)
        z_best = complex(seeds[int(np.argmin(np.abs(det_s)))])
        rec["refined_root"] = newton_det_zero(d, gap, c, k0, gamma, ledger, z_best)
    return rec


def assemble_grid(geo, name, nk, shift, gamma):
    _, kc, weights = k_grid(geo, nk, shift=shift)
    vv, vc, en = occup_empty_bases(h8(geo, R0, kc, 1), MU)
    x = x_from_bases(current_ops(geo, R0, kc, 1), vv, vc)
    gaps = en[:, 4:, None] - en[:, None, :4]
    gap, c = spectral_data(gaps, x, weights)
    d = np.einsum("k,kab->ab", weights, mean_d_from_bases(d_ops(geo, R0, kc, 1), vv))
    k0 = kernel_off_cuts(d, gap, c, 0.0)[0]
    linear, quadratic = expansion(gap, c)
    f0_inc = closed_loop_F(increment_over_z(gap, c, 0.0), 0.0, gamma, from_increment=True)[0]
    layers = list(contour_spec(float(np.min(gap)), float(np.max(gap))))
    soft = raw_soft_box_spec(k0, gamma)
    if soft is not None:
        layers[0]["im_min_raw"] = max(float(layers[0]["im_min_raw"]), 2.0 * float(soft["im_max"]))
    ledgers = {}
    for ledger in ("increment", "raw"):
        rows = []
        use = layers + ([soft] if (soft is not None and ledger == "raw") else [])
        for layer in use:
            for half in ("UHP", "LHP"):
                if layer["name"] == "raw_soft_box" and half == "LHP":
                    continue
                rows.append(run_halfplane(d, gap, c, k0, gamma, ledger, layer, half))
                print(name, ledger, layer["name"], half,
                      "N", rows[-1]["winding_nearest_int"],
                      "resid", f"{rows[-1]['winding_integer_residual']:.2e}",
                      "min|det|", f"{rows[-1]['min_abs_det']:.3e}",
                      flush=True)
        ledgers[ledger] = rows
    uhp_inc = [r for r in ledgers["increment"] if r["halfplane"] == "UHP"]
    uhp_raw = [r for r in ledgers["raw"] if r["halfplane"] == "UHP"]
    wide = layers[1]
    near_cut = {
        "increment": near_cut_det_scan(
            d, gap, c, k0, gamma, "increment", wide["re_half"],
            (5e-2, 1e-2), 201),
        "raw": near_cut_det_scan(
            d, gap, c, k0, gamma, "raw", wide["re_half"],
            (5e-2, 1e-2), 201),
    }
    return {
        "grid": name,
        "n_k": int(len(kc)),
        "shift": shift,
        "n_transitions": int(gap.size),
        "min_direct_gap": float(np.min(gap)),
        "max_direct_gap": float(np.max(gap)),
        "static_K0": pack_matrix(k0),
        "quadratic_maxabs": float(np.max(np.abs(quadratic))),
        "linear_maxabs": float(np.max(np.abs(linear))),
        "F_increment_at_0": pack_matrix(f0_inc),
        "F_increment_at_0_minus_I_maxabs": float(np.max(np.abs(f0_inc - np.eye(2)))),
        "predicted_raw_soft_poles": predicted_raw_soft_poles(k0, gamma),
        "predicted_raw_soft_pole_checks": [
            {
                **p,
                "abs_det_raw": float(np.abs(eval_det_on(
                    d, gap, c, k0, p["z_real"] + 1j * p["z_imag"], gamma, "raw")[0][0])),
                "abs_det_increment": float(np.abs(eval_det_on(
                    d, gap, c, k0, p["z_real"] + 1j * p["z_imag"], gamma, "increment")[0][0])),
            }
            for p in predicted_raw_soft_poles(k0, gamma)
        ],
        "contours": ledgers,
        "near_cut_abs_det": near_cut,
        "uhp_increment_winding_any_nonzero": bool(any(
            r["winding_nearest_int"] != 0 for r in uhp_inc)),
        "uhp_raw_winding_any_nonzero": bool(any(
            r["winding_nearest_int"] != 0 for r in uhp_raw)),
        "raw_soft_box_uhp_winding": next(
            (r["winding_nearest_int"] for r in ledgers["raw"] if r["layer"] == "raw_soft_box"),
            None,
        ),
        "max_winding_integer_residual": max(
            r["winding_integer_residual"]
            for rows in ledgers.values() for r in rows
        ),
    }


def main():
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    rows = []
    for name, nk, shift in GRIDS:
        rec = assemble_grid(geo, name, nk, shift, gamma)
        rows.append(rec)
        print(name, "UHP_inc", rec["uhp_increment_winding_any_nonzero"],
              "UHP_raw", rec["uhp_raw_winding_any_nonzero"], flush=True)
    payload = {
        "status": "p1_zero_self_consistent_maxwell_pole_audit_only",
        "definition": {
            "time_domain": "d kappa_self / dt = gamma I_cell, A_self(t0)=0",
            "fourier": "integral dt e^{+i z t}, Im z>0 retarded",
            "d_dt": "-i z on eigenmodes with vanishing initial value",
            "kappa_self": "(i gamma / z) I",
            "electronics": "I = -K kappa, K = -R^kappa",
            "closed_loop": "F(z) = I_2 + (i gamma / z) K(z)",
            "gate": "det F(z) != 0 for all Im z > 0 on the increment ledger",
            "z0_not_a_mode": "do not multiply by z; Delta K / z is used at the origin",
        },
        "gamma_self": gamma,
        "omega0": OMEGA0_OVER_TH,
        "physical_linewidth": 0,
        "grids": rows,
        "blocked": [
            "dressed noise / I-O with feedback",
            "bare vs dressed vacuum",
            "Pauli redefinition",
            "n,m,V,S_min",
            "P1 drive",
            "joint time propagation",
        ],
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                REPO / "tools/b0_small_model/q8_m0_gsheet.py",
                REPO / "tools/b0_small_model/q8_m0_stage1ai_lowfreq_audit.py",
                REPO / "tools/b0_small_model/q8_m0_response.py",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pole_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


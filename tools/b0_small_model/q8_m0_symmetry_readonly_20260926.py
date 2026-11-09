#!/usr/bin/env python3
"""Read-only M0 symmetry residual audit. Does not change the model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import k_grid
from q8_m0_model import (
    R0, build_geometry, current_ops, h8, mirror_matrix_inplane,
    op_P, op_c2_nhat, op_mirror_a2, op_sewing, op_sz, op_theta_unitary,
    same_spectrum_barred_residual,
)

OUT = Path(__file__).resolve().parents[2] / "results/q_line/m0_symmetry_readonly_20260926"
KS = (
    [0.0, 0.0], [0.5, 0.0], [1.0 / 3.0, 1.0 / 3.0],
    [0.17, -0.11], [0.31, 0.22], [-0.08, 0.41], [0.61, 0.13],
)


def frob(a) -> float:
    return float(np.linalg.norm(np.asarray(a)))


def main() -> int:
    geo = build_geometry()
    p = R0
    P, UT, SZ = op_P(), op_theta_unitary(), op_sz()
    PT = P @ UT
    UM, UC = op_mirror_a2(geo), op_c2_nhat(geo)
    Mm = mirror_matrix_inplane(geo)
    rows = []
    for kf in KS:
        k = geo.k_cart(kf)
        for neel in (+1, -1):
            h = h8(geo, p, k, neel)
            j = current_ops(geo, p, k, neel)
            g = geo.k_cart([1.0, 0.0])
            U = op_sewing(geo, g)
            h_sew = U.conj().T @ h @ U
            j_sew = np.einsum("ij,ajl,lm->aim", U.conj().T, j, U)
            rec = {
                "k_frac": kf,
                "N": neel,
                "herm": frob(h - h.conj().T),
                "P_H": frob(P @ h @ P.conj().T - h8(geo, p, -k, -neel)),
                "Theta_H": frob(UT @ h.conj() @ UT.conj().T - h8(geo, p, -k, -neel)),
                "PT_H": frob(PT @ h.conj() @ PT.conj().T - h),
                "sz": frob(h @ SZ - SZ @ h),
                "sewing_H": frob(h8(geo, p, k + g, neel) - h_sew),
                "sewing_j": frob(current_ops(geo, p, k + g, neel) - j_sew),
                "P_j": frob(-(P @ j @ P.conj().T) - current_ops(geo, p, -k, -neel)),
                "Theta_j_same_sign": frob(UT @ j.conj() @ UT.conj().T - current_ops(geo, p, -k, -neel)),
                "Theta_j_minus": frob(-(UT @ j.conj() @ UT.conj().T) - current_ops(geo, p, -k, -neel)),
                "mirror_H": frob(UM @ h @ UM.conj().T - h8(geo, p, Mm @ k, -neel)),
                "C2_H": frob(UC @ h @ UC.conj().T - h8(geo, p, -(Mm @ k), neel)),
                "mirror_j": frob(
                    np.einsum("ab,bij->aij", Mm, np.einsum("ij,bjl,lm->bim", UM, j, UM.conj().T))
                    - current_ops(geo, p, Mm @ k, -neel)
                ),
                "barred": same_spectrum_barred_residual(geo, p, k, spin=1.0)["residual"],
            }
            rows.append(rec)
    kf, kc, _w = k_grid(geo, 8, shift=0.0)
    partner = ((-np.arange(8)) % 8)
    ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    pidx = (partner[ii] * 8 + partner[jj]).ravel()
    taus = geo.tau_orbitals_cart()[:, :2]
    ph = np.exp(1j * (kc + kc[pidx]) @ taus.T)
    h_plus = h8(geo, p, kc, +1)
    h_m = h8(geo, p, kc[pidx], -1)
    h_mk = np.einsum("ki,kij,kj->kij", ph, h_m, np.conj(ph))
    j_plus = current_ops(geo, p, kc, +1)
    j_m = current_ops(geo, p, kc[pidx], -1)
    j_mk = np.einsum("ki,kaij,kj->kaij", ph, j_m, np.conj(ph))
    grid = {
        "nk": 8,
        "P_H_with_sewing": frob(P @ h_mk @ P.conj().T - h_plus),
        "P_j_with_sewing": frob(-(P @ j_mk @ P.conj().T) - j_plus),
        "Theta_j_same_sign_with_sewing": frob(
            np.einsum("ij,kaij,jl->kalj", UT, j_mk.conj(), UT.conj().T) - j_plus
        ),
        "note": "k_partner = -k + G; sewing phase is exp(i G·τ) on the orbital basis.",
    }
    worst = {key: max(r[key] for r in rows) for key in rows[0] if key not in ("k_frac", "N")}
    payload = {
        "status": "readonly_algebra_audit_no_model_change",
        "not_a_domain_protocol": True,
        "worst": worst,
        "inversion_paired_grid": grid,
        "samples": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "symmetry_residuals.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"worst": worst, "grid": grid}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

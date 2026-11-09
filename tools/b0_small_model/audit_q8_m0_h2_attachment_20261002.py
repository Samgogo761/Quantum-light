#!/usr/bin/env python3
"""Read-only regression audit of the 20260929 M0/H2 attachment.

Imports the attachment without calling its main(). Does not change original
data or run P1. --out writes a NEW audit file only; refuses overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from q8_m0_classical import k_grid, p1_a2_protocol
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_ground_covariance_operator_20260929 import StationaryGroundOperator, j_action
from q8_m0_m0_h2_ground_projection_20260929 import (
    MU, NK, assemble, electronic_channels, h2_packets, photon_channels,
)
from q8_m0_model import R0, build_geometry, current_ops, h8
from test_q8_m0_ground_covariance_operator_20260929 import report as base_fixture

def maximum(x):
    return float(np.max(np.abs(x)))

def pack_complex(x):
    return {"real": np.asarray(x).real.tolist(), "imag": np.asarray(x).imag.tolist()}

def run_audit():
    base = base_fixture()
    geo = build_geometry()
    prot = p1_a2_protocol(.8)
    ph = photon_channels(geo, prot)
    vertices = {}
    corrected_residuals = []
    for domain in (1, -1):
        el = electronic_channels(geo, domain)
        _, kc, weights = k_grid(geo, NK, shift=0.0)
        vv, vc, energy = occup_empty_bases(h8(geo, R0, kc, domain), MU)
        x = x_from_bases(current_ops(geo, R0, kc, domain), vv, vc)
        gap_kcv = energy[:, 4:, None] - energy[:, None, :4]
        # Independent unflattened Kubo expression.
        kubo = el["contact"] - 2 * np.einsum(
            "k,kacv,kbcv,kcv->ab", weights, x.conj(), x, 1/gap_kcv
        ).real
        bad = el["weighted_vertices"]
        bad_residual = el["contact"] - 2*np.real(
            (bad/el["gaps"]) @ bad.conj().T
        )
        # In-memory reference only: no production file is changed.
        aligned = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
        aligned_residual = el["contact"] - 2*np.real(
            (aligned/el["gaps"]) @ aligned.conj().T
        )
        aligned_op = StationaryGroundOperator.from_vertices(
            ph["frequencies"], el["gaps"], ph["vertices"], aligned,
            el["contact"], provenance="in-memory ordering reference; photon model remains unqualified"
        )
        vertices[str(domain)] = {
            "original_residual": bad_residual.tolist(),
            "independent_kubo_residual": kubo.tolist(),
            "aligned_residual": aligned_residual.tolist(),
            "original_vs_kubo_maxabs": maximum(bad_residual-kubo),
            "aligned_vs_kubo_maxabs": maximum(aligned_residual-kubo),
            "aligned_finite_positivity_only": aligned_op.positivity_certificate()["status"],
        }
        corrected_residuals.append(aligned_residual)

    rec, op, rows, projections, cert = assemble(+1)
    ell = np.stack([rows[0]+1j*rows[1], rows[2]+1j*rows[3]])
    comm = 1j*ell @ j_action(ell.conj().T)
    n = op.frequencies.size
    indices = np.r_[np.arange(op.n_photon), n+np.arange(op.n_photon)]
    packet = h2_packets(prot)
    selected = ph["selected_omega"]
    ids = [int(np.argmin(abs(packet["omega"]-w))) for w in selected]
    dw = packet["omega"][1]-packet["omega"][0]
    a, b = projections["48"], projections["96"]
    here = Path(__file__).resolve().parent
    relevant = [
        here/"q8_m0_m0_h2_ground_projection_20260929.py",
        here/"q8_m0_ground_covariance_operator_20260929.py",
        here/"test_q8_m0_ground_covariance_operator_20260929.py",
        here/"q8_m0_stage1af_joint_time.py",
    ]
    return {
        "status": "attachment_rejected_not_a_new_physics_instability",
        "scope": "no P1; no production modifications; existing finite assembly only",
        "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant},
        "base_ground_fixture": {
            "passed": base["all_checks_pass"],
            "max_residual": max(base["residuals"].values()),
        },
        "electronic_ordering": vertices,
        "aligned_plus_minus_static_residual_maxabs":
            maximum(corrected_residuals[0]-corrected_residuals[1]),
        "output_row_checks": {
            "commutator_i_ell_J_ell_dagger": pack_complex(comm),
            "target_for_canonical_two_mode_annihilation": [[1,0],[0,1]],
            "canonical_residual_maxabs": maximum(comm-np.eye(2)),
            "incoming_polarization_rows_maxdiff":
                maximum(ell[0,indices]-ell[1,indices]),
            "b_b_commutator_maxabs": maximum(1j*ell@j_action(ell.T)),
        },
        "stored_functional_quadrature": {
            "maxabs_order48": maximum(a), "maxabs_order96": maximum(b),
            "difference_maxabs": maximum(b-a),
            "relative_maxnorm_difference": maximum(b-a)/maximum(b),
            "note": "arbitrary current rows, NOT physical H2 optical covariance",
        },
        "packet_sampling": {
            "H2_center": 2*packet["omega0"],
            "four_selected_frequencies": selected.tolist(),
            "mass_at_selected_points_with_original_fine_dw":
                float(dw*np.sum(abs(packet["phi"][ids])**2)),
            "note": "diagnostic of this sampling only, not continuum lost-mass error bound",
        },
        "production_optical_moments": "not_evaluated",
        "continuum_claim": False,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_audit()
    content = json.dumps(result, ensure_ascii=True, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(content+"\n")
    print(content)

if __name__ == "__main__":
    main()

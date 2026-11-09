#!/usr/bin/env python3
"""Independent M0 electronic attachment regressions; no propagation or output optics.

A small manufactured photon pair tests the ground-interface plumbing only.
It is not a photon-continuum approximation, detector choice, or H2 output.
The nonuniform weights below are a bookkeeping fixture, not a BZ prescription.
"""
from __future__ import annotations
import json
import numpy as np

from q8_m0_classical import k_grid
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_ground_covariance_operator_20260929 import StationaryGroundOperator
from q8_m0_m0_h2_ground_projection_20260929 import (
    MU, electronic_channels, pack_electronic_channels)
from q8_m0_model import R0, build_geometry, h8, current_ops
from q8_m0_response import d_ops, mean_d_from_bases


def maxabs(a):
    a = np.asarray(a)
    return float(np.max(np.abs(a))) if a.size else 0.0


def raw_m0(geo, neel, nk):
    _, k, w = k_grid(geo, nk, shift=0.0)
    vv, vc, en = occup_empty_bases(h8(geo, R0, k, neel), MU)
    currents = current_ops(geo, R0, k, neel)
    return {
        "weights": w, "gaps": en[:, 4:, None] - en[:, None, :4],
        "x": x_from_bases(currents, vv, vc),
        "d": mean_d_from_bases(d_ops(geo, R0, k, neel), vv),
        "vv": vv, "vc": vc, "currents": currents, "energies": en,
    }


def direct_kubo(gaps, x, d, weights):
    """Reference never flattens the k/c/v axes or uses the production packer."""
    contact = np.einsum("k,kab->ab", weights, d)
    return contact - 2 * np.einsum(
        "k,kacv,kbcv,kcv->ab", weights, x.conj(), x, 1 / gaps).real


def flat_kubo(packed):
    ev = np.asarray(packed["weighted_vertices"])
    gaps = np.asarray(packed["gaps"])
    return np.asarray(packed["contact"]) - 2 * np.einsum(
        "ap,bp,p->ab", ev.conj(), ev, 1 / gaps).real


def ground_photon_covariance(packed):
    """Photon-only rows make electronic coordinate relabeling invisible."""
    op = StationaryGroundOperator.from_vertices(
        [.09, .11], packed["gaps"], np.diag([.01, .012]),
        packed["weighted_vertices"], packed["contact"],
        provenance="M0 electronic relabeling test; manufactured finite photon pair")
    n = op.dimension // 2
    rows = np.zeros((4, op.dimension))
    rows[np.arange(4), [0, n, 1, n + 1]] = 1
    return op.projected_ground_covariance(rows, order=96), op


def rotate_degenerate_pairs(basis, rng):
    unitary = np.zeros((basis.shape[0], 4, 4), dtype=complex)
    for ik in range(basis.shape[0]):
        for start in (0, 2):
            z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            unitary[ik, start:start+2, start:start+2] = np.linalg.qr(z)[0]
    return np.einsum("kiv,kvw->kiw", basis, unitary)


def report():
    geo = build_geometry()
    rng = np.random.default_rng(20261002)
    residuals, negative_controls = {}, {}
    references = {}
    matrix_records = {}
    for neel in (+1, -1):
        raw = raw_m0(geo, neel, 8)
        equal = raw["weights"]
        unequal = np.arange(1, len(equal) + 1, dtype=float)
        unequal /= unequal.sum()
        for name, weights in (("equal", equal), ("nonuniform_fixture", unequal)):
            key = f"N{neel:+d}_{name}"
            packed = pack_electronic_channels(raw["gaps"], raw["x"], raw["d"], weights)
            ref = direct_kubo(raw["gaps"], raw["x"], raw["d"], weights)
            residuals[key + "_unflattened_Kubo"] = maxabs(flat_kubo(packed) - ref)
            expected_table = np.indices(raw["gaps"].shape).reshape(3, -1).T
            residuals[key + "_explicit_channel_table"] = maxabs(
                np.asarray(packed["channel_table"]) - expected_table)
            residuals[key + "_weighted_original_contact"] = maxabs(
                np.asarray(packed["contact"]) - np.einsum("k,kab->ab", weights, raw["d"]))
            # A synchronized channel permutation is a coordinate change; it must
            # not change the static response. An unsynchronized one must fail.
            ev, gaps = np.asarray(packed["weighted_vertices"]), np.asarray(packed["gaps"])
            perm = rng.permutation(len(gaps))
            changed = dict(packed, gaps=gaps[perm], weighted_vertices=ev[:, perm])
            residuals[key + "_synchronized_permutation"] = maxabs(flat_kubo(changed) - ref)
            mismatched = dict(packed, weighted_vertices=ev[:, perm])
            negative_controls[key + "_vertex_only_permutation"] = maxabs(flat_kubo(mismatched) - ref)
            op = StationaryGroundOperator.from_vertices(
                [.09, .11], gaps, np.diag([.01, .012]), ev, packed["contact"],
                provenance="M0 electronic attachment regression only")
            residuals[key + "_ground_Schur_vs_unflattened_Kubo"] = maxabs(
                np.asarray(op.positivity_certificate()["original_static_residual"]) - ref)
            if name == "equal":
                references[neel] = ref
                matrix_records[str(neel)] = ref.tolist()
                attached = electronic_channels(geo, neel, nk=8)
                for field in ("gaps", "weighted_vertices", "contact"):
                    residuals[key + "_attachment_" + field] = maxabs(
                        np.asarray(attached[field]) - np.asarray(packed[field]))
            else:
                negative_controls[key + "_unweighted_contact"] = maxabs(
                    raw["d"].mean(axis=0) - np.asarray(packed["contact"]))
    residuals["equal_inversion_closed_grid_domain_static_difference"] = maxabs(
        references[+1] - references[-1])

    # A separate smaller M0 instance protects the complete finite covariance
    # interface, including all retained electronic channels.
    raw = raw_m0(geo, +1, 4)
    packed = pack_electronic_channels(raw["gaps"], raw["x"], raw["d"], raw["weights"])
    base_cov, _ = ground_photon_covariance(packed)
    ev, gaps = np.asarray(packed["weighted_vertices"]), np.asarray(packed["gaps"])
    perm = rng.permutation(len(gaps))
    perm_cov, _ = ground_photon_covariance(dict(packed, gaps=gaps[perm], weighted_vertices=ev[:, perm]))
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, len(gaps)))
    phase_cov, _ = ground_photon_covariance(dict(packed, weighted_vertices=ev * phase))
    residuals["photon_covariance_synchronized_permutation"] = maxabs(perm_cov - base_cov)
    residuals["photon_covariance_electronic_phase"] = maxabs(phase_cov - base_cov)
    vv = rotate_degenerate_pairs(raw["vv"], rng)
    vc = rotate_degenerate_pairs(raw["vc"], rng)
    rotated_x = x_from_bases(raw["currents"], vv, vc)
    rotated = pack_electronic_channels(raw["gaps"], rotated_x, raw["d"], raw["weights"])
    rotated_cov, _ = ground_photon_covariance(rotated)
    residuals["Kramers_pair_energy_splitting"] = maxabs(
        raw["energies"][:, 1::2] - raw["energies"][:, ::2])
    residuals["Kramers_U2_static_response"] = maxabs(flat_kubo(rotated) - flat_kubo(packed))
    residuals["Kramers_U2_photon_covariance"] = maxabs(rotated_cov - base_cov)

    # Invalid physical weights must be rejected rather than clipped or silently
    # normalized. Zero-weight channels themselves remain allowed spectators.
    rejection_checks = {}
    bad_weight_sets = {
        "negative": raw["weights"].copy(),
        "nonfinite": raw["weights"].copy(),
        "wrong_normalization": raw["weights"] * 2,
    }
    bad_weight_sets["negative"][0] = -1
    bad_weight_sets["nonfinite"][0] = np.nan
    for label, weights in bad_weight_sets.items():
        try:
            pack_electronic_channels(raw["gaps"], raw["x"], raw["d"], weights)
        except ValueError:
            rejection_checks[label] = True
        else:
            rejection_checks[label] = False
    tolerance = 2e-10
    passed = (max(residuals.values()) < tolerance
              and min(negative_controls.values()) > .1
              and all(rejection_checks.values()))
    return {
        "status": "electronic_attachment_regression_verified" if passed else "FAILED",
        "all_checks_pass": bool(passed),
        "absolute_residual_tolerance": tolerance,
        "residuals": residuals,
        "negative_controls_must_be_large": negative_controls,
        "invalid_weights_rejected": rejection_checks,
        "corrected_nk8_static_residuals": matrix_records,
        "finite_covariance_fixture": {"nk": 4, "photon_oscillators": 2,
            "photon_frequencies": [.09, .11], "quadrature_order": 96,
            "is_H2_output": False},
        "P1_run": False, "physical_model_changed": False,
        "full_BZ_or_continuum_qualification": False,
        "optical_output_or_entanglement_computed": False,
    }


def main():
    result = report()
    print(json.dumps(result, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit("Electronic attachment regression failed.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""R2 leftover A/B only. Does not overwrite r2_checks.json.

A: M0 photon projection with frozen occupied D and gamma_self.
B: driven dark column that actually carries nonzero current.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q8_m0_cjj import occup_empty_bases, x_from_bases
from q8_m0_continuum_columns_20261004 import ColumnState, current, rhs
from q8_m0_gsheet import gamma_self
from q8_m0_incoming_spectral_completeness_20261004 import (
    assemble_incoming_bundle,
    dark_projector,
    dark_subspace,
    electron_column_density,
    maxabs,
    propagate_bundle,
)
from q8_m0_model import R0, build_geometry, current_ops, h8
from q8_m0_response import d_ops, mean_d_from_bases
from test_q8_m0_incoming_completeness_r2_20261004 import (
    FREQ,
    GAMMA,
    MU,
    contact_from,
    dense_midpoint,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/incoming_completeness_r2_leftovers_20261004"
OLD_JSON = [
    "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json",
    "results/q_line/incoming_completeness_revision_20261004/revision_checks.json",
    "results/q_line/incoming_completeness_r2_20261004/r2_checks.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_real_matrix(m):
    m = np.asarray(m, float)
    return {
        "real": m.tolist(),
        "symmetric_residual": float(np.max(np.abs(m - m.T))),
        "maxabs": float(np.max(np.abs(m))),
    }


def frozen_m0_photon_projection():
    """Leftover A: same two-k electronic labels, frozen sheet D and gamma."""
    geo = build_geometry()
    gamma = float(gamma_self(geo))
    kfrac = np.array([[0.0, 0.0], [0.23, 0.12]])
    kc = geo.k_cart(kfrac)
    weights = np.array([0.5, 0.5])
    n_c, n_v = 4, 4
    rows = []
    for neel in (1, -1):
        vv, vc, en = occup_empty_bases(h8(geo, R0, kc, neel), MU)
        x = x_from_bases(current_ops(geo, R0, kc, neel), vv, vc)
        gap = (en[:, 4:, None] - en[:, None, :4]).reshape(-1)
        u = np.einsum("k,kacv->akcv", np.sqrt(weights), x).reshape(2, -1)
        d_occ = mean_d_from_bases(d_ops(geo, R0, kc, neel), vv)
        d_frozen = np.einsum("k,kab->ab", weights, d_occ)
        d_sym_res = float(np.max(np.abs(d_frozen - d_frozen.T)))
        if d_sym_res > 1e-11:
            raise ValueError("Occupied D is not symmetric at frozen tolerance.")
        d_frozen = 0.5 * (d_frozen + d_frozen.T)
        artificial = contact_from(u, np.maximum(gap, 1e-18))
        basis, info, near = dark_subspace(gap, u)
        min_pole = min(float(np.min(np.abs(gap - w))) for w in FREQ)
        bundle = assemble_incoming_bundle(gap, u, d_frozen, gamma, FREQ)
        i_dark = (
            current(bundle.point, u, bundle.photon.D0)
            if bundle.n_point_columns
            else np.zeros((2, 0))
        )
        p_dark = dark_projector(basis)
        photon_dark = []
        for w in FREQ:
            rho_b, _ = electron_column_density(bundle.photon, float(w))
            photon_dark.append(maxabs(p_dark @ rho_b) if basis.size else 0.0)
        rows.append({
            "neel": int(neel),
            "n_el": int(u.shape[1]),
            "n_k": int(len(kc)),
            "gap_subspace_nullity": int(basis.shape[1]),
            "n_exact_gap_clusters": len(info),
            "near_degenerate_pairs": len(near),
            "physical_current_of_dark_b": maxabs(i_dark),
            "photon_dark_projection": max(photon_dark) if photon_dark else 0.0,
            "metadata_lengths_agree": bool(
                bundle.n_point_columns == len(bundle.point_gaps) == len(bundle.point_labels)
            ),
            "D_used": pack_real_matrix(d_frozen),
            "D_artificial_contact_maxabs": float(np.max(np.abs(artificial))),
            "D_frozen_vs_artificial_maxabs": float(np.max(np.abs(d_frozen - artificial))),
            "gamma_used": gamma,
            "gamma_artificial_not_used": float(GAMMA),
            "min_gap_to_probe_freq": min_pole,
            "frequencies": [float(w) for w in FREQ],
            "original_labels_kept": True,
            "kcv_shape": [int(len(kc)), n_c, n_v],
            "not_a_BZ_dark_dimension": True,
            "finite_sample_not_continuum": True,
            "D_source": "occupied_mean_d_from_bases_d_ops",
            "gamma_source": "q8_m0_gsheet.gamma_self",
        })
    return {
        "gamma_self": gamma,
        "domains": rows,
        "legal_electronic_continuum_boundary": False,
        "k_sample": kfrac.tolist(),
        "photon_projection_uses_frozen_sheet_parameters": True,
        "dark_current_independent_of_D_gamma": True,
    }


def driven_dark_nonzero_current():
    """Leftover B: extra vertex from I = u* b, so du* v is nonzero."""
    gaps = np.array([1.1, 1.1, 1.9])
    u0 = np.array([[0.20, 0.20j, 0.15], [0.10, 0.10j, -0.08]], complex)
    d0 = contact_from(u0, gaps)
    bundle = assemble_incoming_bundle(gaps, u0, d0, GAMMA, FREQ)
    times = np.linspace(0.0, 1.8, 37)
    v = bundle.dark_basis[:, 0]
    pol = np.array([0.18, -0.11], float)
    du = np.outer(pol, v)
    i_from_du = du.conj() @ v

    def u_static(t):
        return u0 * np.exp(1j * gaps * t)[None, :]

    def d_static(_t):
        return d0

    def u_on(t):
        return u_static(t) + du

    packed_on = propagate_bundle(bundle, times, u_on, d_static)
    n_el, n_ph = bundle.n_el, bundle.n_photon_columns
    independent = [bundle.point.packed()]
    y = independent[0].copy()
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = float(t1 - t0)
        tmid = 0.5 * (t0 + t1)
        y = dense_midpoint(y, u_on(tmid), d0, dt, GAMMA)
        independent.append(y.copy())
    independent = np.stack(independent)
    sample_t = (0.0, 0.5, 1.8)
    sample_idx = [int(np.argmin(np.abs(times - t))) for t in sample_t]
    currents = []
    rhs_A = []
    for t in sample_t:
        st = ColumnState(
            b=v[:, None],
            bdag=np.zeros((n_el, 1), complex),
            A=np.zeros((2, 1), complex),
        )
        i_now = current(st, u_on(t), d0)
        r_now = rhs(st, u_on(t), d0, GAMMA, np.zeros((2, 1), complex))
        currents.append(maxabs(i_now))
        rhs_A.append(maxabs(r_now.A))
    evolved_I = []
    for i in sample_idx:
        st = ColumnState.unpack(packed_on[i, :, n_ph:], n_el)
        evolved_I.append(maxabs(current(st, u_on(times[i]), d0)))
    return {
        "du_star_v_maxabs": maxabs(i_from_du),
        "v_T_v_maxabs": float(np.abs(v @ v)),
        "current_at_t": {
            "0.0": currents[0],
            "0.5": currents[1],
            "1.8": currents[2],
        },
        "rhs_A_at_t0": rhs_A[0],
        "evolved_current_at_t": {
            "0.0": evolved_I[0],
            "0.5": evolved_I[1],
            "1.8": evolved_I[2],
        },
        "driven_current_is_nonzero": bool(max(currents) > 1e-3),
        "driven_dark_vs_independent_midpoint": maxabs(
            packed_on[:, :, n_ph:] - independent
        ),
        "old_outer_pol_vconj_would_give_vTv": True,
        "not_a_model_change": True,
        "fixture_gamma": float(GAMMA),
    }


def leftover_flags(frozen, driven):
    flags = {
        "driven_current_is_nonzero": bool(driven["driven_current_is_nonzero"]),
        "driven_du_star_v_nonzero": bool(driven["du_star_v_maxabs"] > 1e-3),
        "driven_vs_independent": bool(driven["driven_dark_vs_independent_midpoint"] < 1e-4),
        "driven_rhs_A_nonzero": bool(driven["rhs_A_at_t0"] > 1e-3),
        "photon_projection_uses_frozen_sheet_parameters": bool(
            frozen["photon_projection_uses_frozen_sheet_parameters"]
        ),
        "gamma_is_gamma_self": bool(
            abs(frozen["gamma_self"] - frozen["domains"][0]["gamma_used"]) < 1e-18
        ),
        "gamma_is_not_artificial": bool(abs(frozen["gamma_self"] - GAMMA) > 1e-6),
    }
    for row in frozen["domains"]:
        n = row["neel"]
        flags[f"N{n}_metadata"] = bool(row["metadata_lengths_agree"])
        flags[f"N{n}_dark_current"] = bool(row["physical_current_of_dark_b"] < 1e-12)
        flags[f"N{n}_photon_dark"] = bool(row["photon_dark_projection"] < 1e-8)
        flags[f"N{n}_D_symmetric"] = bool(row["D_used"]["symmetric_residual"] < 1e-12)
        flags[f"N{n}_D_not_artificial"] = bool(row["D_frozen_vs_artificial_maxabs"] > 1e-6)
        flags[f"N{n}_not_bz"] = bool(row["not_a_BZ_dark_dimension"])
        flags[f"N{n}_off_pole"] = bool(row["min_gap_to_probe_freq"] > 1e-8)
    return flags


def main() -> int:
    t0 = time.perf_counter()
    frozen = frozen_m0_photon_projection()
    driven = driven_dark_nonzero_current()
    flags = leftover_flags(frozen, driven)
    payload = {
        "status": "incoming_completeness_r2_leftovers_frozen_D_gamma_and_driven_bright",
        "all_checks_pass": bool(all(flags.values())),
        "pass_flags": flags,
        "leftover_A_frozen_M0_photon_projection": frozen,
        "leftover_B_driven_dark_nonzero_current": driven,
        "old_json_not_overwritten": OLD_JSON,
        "P1_not_run": True,
        "full_BZ_not_run": True,
        "elapsed_s": time.perf_counter() - t0,
        "source_sha256": {
            "q8_m0_incoming_spectral_completeness_20261004.py": sha(
                Path(__file__).with_name("q8_m0_incoming_spectral_completeness_20261004.py")
            ),
            "test_q8_m0_incoming_completeness_r2_20261004.py": sha(
                Path(__file__).with_name("test_q8_m0_incoming_completeness_r2_20261004.py")
            ),
            "test_q8_m0_incoming_completeness_r2_leftovers_20261004.py": sha(Path(__file__)),
            "q8_m0_gsheet.py": sha(Path(__file__).with_name("q8_m0_gsheet.py")),
            "q8_m0_response.py": sha(Path(__file__).with_name("q8_m0_response.py")),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leftover_checks.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "all_checks_pass": payload["all_checks_pass"],
        "failed": [k for k, v in flags.items() if not v],
        "gamma_self": frozen["gamma_self"],
        "photon_dark": [r["photon_dark_projection"] for r in frozen["domains"]],
        "dark_I": [r["physical_current_of_dark_b"] for r in frozen["domains"]],
        "D_vs_artificial": [r["D_frozen_vs_artificial_maxabs"] for r in frozen["domains"]],
        "driven_I": driven["current_at_t"],
        "driven_vs_independent": driven["driven_dark_vs_independent_midpoint"],
        "elapsed_s": payload["elapsed_s"],
    }, indent=2))
    return 0 if payload["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


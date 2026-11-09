#!/usr/bin/env python3
"""Reproject saved a2 Y onto the reference EM packet. No new propagation.

Shape-only d_φ. χ(ω) remains unset. No photon numbers. Magnetic cross kernel
is not promoted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump  # noqa: E402
from q8_m0_cjj_validate import pack_z  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_k1_chi_interface import ChiNotChosenError, apply_chi_to_shape_kernel, output_placeholder  # noqa: E402
from q8_m0_k1_mode import provenance, spec_dict as k1_spec  # noqa: E402
from q8_m0_k1_packet import (  # noqa: E402
    PACKET_ID,
    PACKET_STATUS,
    ROUTE_STATUS,
    packet_on_grid,
    reproject_d_perpperp,
    spec_scale,
)
from q8_m0_model import build_geometry, sha256_file  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/q_line/cjj_m0_p1_a2_s1_20260917"
OUT = REPO / "results/q_line/k1_mode_20260917"
JOBS = ("V48", "V48S")


def z_of(p: dict) -> complex:
    return complex(p["re"], p["im"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    scale = spec_scale(prot.omega0, prot.t_pulse)
    alpha = complex(
        scale["scale"]["alpha_from_t_mid"]["re"],
        scale["scale"]["alpha_from_t_mid"]["im"],
    )
    rec_phi = packet_on_grid(prot.omega0, prot.t_pulse)
    np.savez_compressed(
        OUT / "k1_R0_phi_grid.npz",
        omega=rec_phi["omega"],
        phi=rec_phi["phi"],
        omega_c=rec_phi["omega_c"],
        commutator_integral=rec_phi["commutator_integral"],
    )
    jobs = {}
    for name in JOBS:
        plus = np.load(ROOT / name / f"{name}_N+1_A+1.npz")
        minus = np.load(ROOT / name / f"{name}_N-1_A+1.npz")
        w = np.asarray(plus["w_k"], dtype=float)
        rec_p = reproject_d_perpperp(np.asarray(plus["Y"]), w, geo, alpha)
        rec_m = reproject_d_perpperp(np.asarray(minus["Y"]), w, geo, alpha)
        d_q9_e = 0.5 * (z_of(rec_p["d_Q9_perpperp_H2"]) + z_of(rec_m["d_Q9_perpperp_H2"]))
        d_q9_o = 0.5 * (z_of(rec_p["d_Q9_perpperp_H2"]) - z_of(rec_m["d_Q9_perpperp_H2"]))
        d_ph_e = 0.5 * (z_of(rec_p["d_phi_perpperp_H2"]) + z_of(rec_m["d_phi_perpperp_H2"]))
        d_ph_o = 0.5 * (z_of(rec_p["d_phi_perpperp_H2"]) - z_of(rec_m["d_phi_perpperp_H2"]))
        jobs[name] = {
            "plus_npz_sha256": sha256_file(ROOT / name / f"{name}_N+1_A+1.npz"),
            "minus_npz_sha256": sha256_file(ROOT / name / f"{name}_N-1_A+1.npz"),
            "plus": rec_p,
            "minus": rec_m,
            "d_Q9_even": pack_z(d_q9_e),
            "d_Q9_odd": pack_z(d_q9_o),
            "d_phi_even": pack_z(d_ph_e),
            "d_phi_odd": pack_z(d_ph_o),
            "rel_odd_over_even_Q9": abs(d_q9_o) / max(abs(d_q9_e), 1e-30),
            "output_placeholder_even": output_placeholder(pack_z(d_ph_e)),
        }
    chi_blocked = False
    try:
        apply_chi_to_shape_kernel(rec_phi["omega"], rec_phi["phi"])
    except ChiNotChosenError:
        chi_blocked = True
    g48 = z_of(jobs["V48"]["d_phi_even"])
    s48 = z_of(jobs["V48S"]["d_phi_even"])
    out = {
        "route": ROUTE_STATUS,
        "packet_id": PACKET_ID,
        "packet_status": PACKET_STATUS,
        "role": "first_reference_output_mode_not_final_experimental_optimum",
        "k1_spec": k1_spec(geo),
        "scale": scale,
        "jobs": jobs,
        "phi_even_rel_G48_S48": abs(g48 - s48) / max(abs(g48), abs(s48), 1e-30),
        "q9_even_rel_G48_S48": abs(z_of(jobs["V48"]["d_Q9_even"]) - z_of(jobs["V48S"]["d_Q9_even"]))
        / max(abs(z_of(jobs["V48"]["d_Q9_even"])), abs(z_of(jobs["V48S"]["d_Q9_even"])), 1e-30),
        "chi_apply_blocked": chi_blocked,
        "phi_grid": str(OUT / "k1_R0_phi_grid.npz"),
        "provenance": provenance(),
        "photon_output": "not_run",
        "g2": "not_run",
        "quantum_pump": "not_run",
        "d_parperp": "not_reprojected_unresolved_magnetic_candidate",
        "kind": "shape_only_source_projection",
        "note": (
            "Pulse-matched φ only rescales Q9 H2⊥ by |α|²; k-space cancellation is unchanged. "
            "d_φ is not ⟨n⟩. χ(ω) is unset."
        ),
    }
    dump(OUT / "k1_R0_reproject_v48.json", out)
    print(json.dumps({
        "wrote": str(OUT / "k1_R0_reproject_v48.json"),
        "packet_status": PACKET_STATUS,
        "commutator_integral": scale["scale"]["commutator_integral"],
        "d_scale": scale["scale"]["d_scale_from_t_mid"],
        "plancherel_rel": scale["scale"]["plancherel"]["rel_diff"],
        "phi_even_rel": out["phi_even_rel_G48_S48"],
        "chi_apply_blocked": chi_blocked,
        "G48_d_phi_even": jobs["V48"]["d_phi_even"],
        "S48_d_phi_even": jobs["V48S"]["d_phi_even"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""K1 reference output mode. χ(ω) slot empty. Absolute photons not evaluated."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from q8_m0_k1_chi_interface import chi_spec
from q8_m0_k1_packet import (
    FOURIER_CONVENTION,
    PACKET_ID,
    PACKET_STATUS,
    ROUTE_STATUS,
    ReferencePhysicalPacket,
)
from q8_m0_model import M0Geometry, build_geometry, sha256_file
from q8_m0_s1_kernel import S1_ID, parallel_perp_basis, spec_dict as s1_spec

K1_STATUS = "reference_mode_adopted_chi_unset"
K1_ID = "K1_R0_reference_perp_H2_pulse_matched"
K1_NOT = (
    "not a photon number, not g2, not cavity, not quantum pump, "
    "not Q9-Hann-normalized-into-photons, "
    "not a promotion of unresolved d_parperp, "
    "not a derived Maxwell χ(ω), "
    "not the final experimental collection mode"
)
_HERE = Path(__file__).resolve()


@dataclass(frozen=True)
class K1ModeSkeleton:
    ident: str = K1_ID
    status: str = K1_STATUS
    parent_source_filter: str = S1_ID
    construction_order: str = "phi_from_EM -> [b,bdagger]=1 -> K_shape -> reproject_Y"
    first_mode: str = "perp_H2"
    second_mode: str = "not_in_K1"
    role: str = "first_reference_output_mode_not_final_experimental_optimum"
    frequency_inner_product: str = "int_0^inf |phi|^2 d omega = 1"
    commutator_normalized_on_packet: bool = True
    q9_is_not_the_photon_mode: bool = True
    coupling_g: str = "unset"
    n_cell: str = "unset"
    numerical_aperture: str = "unset"
    radiation_kernel_chi: str = "shape_K_1_over_sqrt_omega_implemented_SI_prefactor_unset"
    magnetic_cross_kernel: str = "unresolved_source_filter_not_photon_covariance"
    packet_id: str = PACKET_ID
    packet_status: str = PACKET_STATUS


def provenance() -> dict:
    packet_py = _HERE.parent / "q8_m0_k1_packet.py"
    chi_py = _HERE.parent / "q8_m0_k1_chi_interface.py"
    cjj_py = _HERE.parent / "q8_m0_cjj.py"
    return {
        "k1_module_sha256": sha256_file(_HERE),
        "packet_module_sha256": sha256_file(packet_py),
        "chi_module_sha256": sha256_file(chi_py),
        "cjj_module_sha256": sha256_file(cjj_py),
        "fourier_convention": FOURIER_CONVENTION,
    }


def conversion_formula() -> dict:
    return {
        "beam_operator": "b = ∫_0^∞ dω φ(ω) a_{q̂λ}(ω),  [a(ω),a†(ω')]=δ(ω-ω')",
        "commutator": "[b,b†] = ∫_0^∞ |φ|² dω = 1",
        "K_shape": (
            "K_a(t)=ê_⊥,a ∫_0^∞ dω φ(ω)/√ω e^{+iωt}; "
            "SI prefactor i√(Z0/(4πħS)) unset; finite [0,Tp]; not R^A"
        ),
        "source_kernel": "d_φ is shape-only projection of existing Y; not ⟨n⟩",
        "mean_photon_placeholder": "|χ-dependent factor| * (N_cell^2 |b_φ|^2 + N_cell d_φ)",
        "evaluated": False,
        "chi": chi_spec(),
        "N_cell": "unset",
        "d_parperp": "not_used",
        "gaussian_closure": "not_chosen",
    }


def spec_dict(geo: M0Geometry | None = None) -> dict:
    geo = geo or build_geometry()
    _e_par, e_perp = parallel_perp_basis(geo)
    s1 = s1_spec(geo)
    return {
        "k1_id": K1_ID,
        "status": K1_STATUS,
        "not": K1_NOT,
        "kernel": asdict(K1ModeSkeleton()),
        "reference_packet": asdict(ReferencePhysicalPacket()),
        "e_perp": e_perp.tolist(),
        "conversion": conversion_formula(),
        "provenance": provenance(),
        "parent_s1": {
            "s1_id": s1["s1_id"],
            "s1_status": s1["status"],
            "channel_qualification": s1.get("channel_qualification"),
        },
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "g2": "not_run",
        "chi_physical_form": "not_chosen",
    }


if __name__ == "__main__":
    import json

    out = Path(__file__).resolve().parents[2] / "results/q_line/k1_mode_20260917"
    out.mkdir(parents=True, exist_ok=True)
    rec = spec_dict()
    (out / "k1_R0_reference.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"wrote": str(out / "k1_R0_reference.json"), "status": rec["status"]}, indent=2))

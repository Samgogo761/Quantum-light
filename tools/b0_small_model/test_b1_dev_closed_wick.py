"""Tests for the development-only B1 closed-Wick material interface.

No server data and no production T2 map are used.  Checks use ``require`` so
``python -O`` cannot turn a failure into PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b1_dev_closed_wick import (  # noqa: E402
    ALLOWED_LAYOUT,
    LABEL,
    STATUS,
    ClosedWickPacket,
    closed_wick_reference,
    run_m88_c20_interface,
)
from wick import single_particle_u, two_time_wick_greater  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expect_value_error(fn, contains: str) -> None:
    try:
        fn()
    except ValueError as exc:
        require(contains in str(exc), f"wrong error: {exc}")
        return
    raise RuntimeError(f"expected ValueError containing {contains!r}")


def two_level_packet(*, weights=(1.0,), refs=(0, 1)) -> ClosedWickPacket:
    times = np.array([0.0, 0.25, 0.6])
    nk = len(weights)
    nb = 2
    h = np.diag([-0.4, 0.7]).astype(complex)
    step = np.empty((2, nk, nb, nb), dtype=complex)
    for it, dt in enumerate(np.diff(times)):
        step[it] = np.broadcast_to(single_particle_u(h, float(dt)), (nk, nb, nb))
    rho0 = np.broadcast_to(np.diag([1.0, 0.0]), (nk, nb, nb)).astype(complex).copy()
    current = np.broadcast_to(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        (times.size, nk, nb, nb),
    ).copy()
    return ClosedWickPacket(
        times=times,
        k_weights=np.asarray(weights, dtype=float),
        rho0=rho0,
        step_unitaries=step,
        current_a=current,
        current_b=current,
        reference_indices=refs,
    )


def test_core_matches_direct_wick_and_equal_time() -> None:
    packet = two_level_packet()
    result = closed_wick_reference(packet)
    u_20 = packet.step_unitaries[1, 0] @ packet.step_unitaries[0, 0]
    direct = two_time_wick_greater(
        packet.current_a[2, 0], packet.current_b[0, 0], packet.rho0[0], u_20
    )
    require(abs(result.c_greater[2, 0] - direct) < 1.0e-12, "reference vs direct Wick")
    require(abs(result.c_greater[0, 0] - 1.0) < 1.0e-12, "Pauli equal-time limit")
    require(result.metadata["equal_time_residual"] < 1.0e-14, "equal-time gate")
    require(result.metadata["label"] == LABEL, "development label")
    require(result.metadata["status"] == STATUS, "development status")
    require(result.metadata["not_quantum_output_g2"] is True, "not g2 guard")


def test_k_weight_is_squared_for_connected_intensive_current() -> None:
    packet = two_level_packet(weights=(0.25, 0.75), refs=(0,))
    result = closed_wick_reference(packet)
    expected = 0.25**2 + 0.75**2
    require(abs(result.c_greater[0, 0] - expected) < 1.0e-12, "k-weight square")
    require(abs(result.mean_a[0]) < 1.0e-14, "mean current")


def m88_fixture() -> ClosedWickPacket:
    nt, nk, nb, nv = 2, 1, 78, 58
    times = np.array([0.0, 0.1])
    rho0 = np.zeros((nk, nb, nb), dtype=complex)
    rho0[0, np.arange(nv), np.arange(nv)] = 1.0
    steps = np.broadcast_to(np.eye(nb, dtype=complex), (nt - 1, nk, nb, nb)).copy()
    current = np.zeros((nt, nk, nb, nb), dtype=complex)
    current[:, 0, nv - 1, nv] = 1.0
    current[:, 0, nv, nv - 1] = 1.0
    return ClosedWickPacket(
        times=times,
        k_weights=np.ones(nk),
        rho0=rho0,
        step_unitaries=steps,
        current_a=current,
        current_b=current,
        reference_indices=(0,),
    )


def good_contract() -> dict[str, object]:
    return {
        "target_model": "M88-c20",
        "n_bands": 78,
        "n_valence": 58,
        "basis_layout": ALLOWED_LAYOUT,
        "dynamics": "closed_unitary",
        "dephasing_enabled": False,
        "production_t2_used": False,
        "qrt_used": False,
        "operator_provenance": "analytic-fixture-current-v1",
        "propagator_provenance": "analytic-fixture-identity-v1",
        "k_normalization_provenance": "declared-discrete-independent-block-fixture-v1",
    }


def test_m88_contract_accepts_only_honest_development_packet() -> None:
    result = run_m88_c20_interface(m88_fixture(), contract=good_contract())
    require(result.metadata["target_model"] == "M88-c20", "target model")
    require(result.metadata["n_bands"] == 78, "active bands")
    require(result.metadata["n_valence"] == 58, "occupied bands")
    require(
        result.metadata["material_backend_status"]
        == "interface_only_not_connected_to_production_lg_cov",
        "backend limitation",
    )
    require(result.metadata["absolute_k_normalization_validated"] is False, "normalization guard")
    require(abs(result.c_greater[0, 0] - 1.0) < 1.0e-12, "M88 fixture Pauli pair")


def test_production_t2_and_lg_cov_claims_are_rejected() -> None:
    contract = good_contract()
    contract["production_t2_used"] = True
    expect_value_error(
        lambda: run_m88_c20_interface(m88_fixture(), contract=contract),
        "contract mismatch",
    )

    contract = good_contract()
    contract["dephasing_enabled"] = True
    expect_value_error(
        lambda: run_m88_c20_interface(m88_fixture(), contract=contract),
        "contract mismatch",
    )

    contract = good_contract()
    contract["basis_layout"] = "lg_cov_grid"
    expect_value_error(
        lambda: run_m88_c20_interface(m88_fixture(), contract=contract),
        "contract mismatch",
    )

    contract = good_contract()
    del contract["k_normalization_provenance"]
    expect_value_error(
        lambda: run_m88_c20_interface(m88_fixture(), contract=contract),
        "k_normalization_provenance",
    )


def test_nonunitary_propagator_is_rejected() -> None:
    packet = two_level_packet()
    bad_steps = packet.step_unitaries.copy()
    bad_steps[0, 0, 0, 0] *= 0.9
    bad = ClosedWickPacket(
        times=packet.times,
        k_weights=packet.k_weights,
        rho0=packet.rho0,
        step_unitaries=bad_steps,
        current_a=packet.current_a,
        current_b=packet.current_b,
        reference_indices=packet.reference_indices,
    )
    expect_value_error(lambda: closed_wick_reference(bad), "not unitary")


def test_invalid_fermion_density_is_rejected() -> None:
    packet = two_level_packet()
    bad_rho = packet.rho0.copy()
    bad_rho[0, 0, 0] = 1.01
    bad = ClosedWickPacket(
        times=packet.times,
        k_weights=packet.k_weights,
        rho0=bad_rho,
        step_unitaries=packet.step_unitaries,
        current_a=packet.current_a,
        current_b=packet.current_b,
        reference_indices=packet.reference_indices,
    )
    expect_value_error(lambda: closed_wick_reference(bad), "0 <= rho <= I")


def main() -> int:
    tests = [
        test_core_matches_direct_wick_and_equal_time,
        test_k_weight_is_squared_for_connected_intensive_current,
        test_m88_contract_accepts_only_honest_development_packet,
        test_production_t2_and_lg_cov_claims_are_rejected,
        test_nonunitary_propagator_is_rejected,
        test_invalid_fermion_density_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"B1-dev closed Wick interface: {len(tests)}/{len(tests)} PASS")
    print("B1_DEV_INTERFACE=DEVELOPMENT_ONLY_PASS")
    print("M88_C20_MATERIAL_BACKEND=NOT_CONNECTED")
    print("QUANTUM_OUTPUT_G2=NOT_COMPUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

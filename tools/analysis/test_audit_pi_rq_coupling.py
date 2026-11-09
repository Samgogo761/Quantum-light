#!/usr/bin/env python3
"""Synthetic gates for audit_pi_rq_coupling.py. No production mmn/tb/chk."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from audit_pi_rq_coupling import (
    ETA_DENOM_FLOOR_EV,
    FORBIDDEN_KEYS,
    I_BASE,
    berry_connection_ang,
    complement_basis,
    dumps_strict,
    e0_au,
    eta_r_stats_from_rpol,
    frobenius_pi_r_q,
    frobenius_rect,
    map_k20_to_wannier,
    mv_shell_weights,
    nearest_k,
    pair_table,
    parse_nnkp,
    proxy_index_sets,
    regular_mp_kpoints,
    register_set1_edge_occupation,
    sbe_frac,
    skip_lines_binary,
    summarize_eta,
    to_jsonable,
    walk_edge_max,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def test_sbe_frac_and_exact_mesh_map() -> None:
    frac = sbe_frac(11, 16, 20)
    require(abs(frac[0] - 0.5) < 1e-15, f"kx {frac}")
    require(abs(frac[1] - 0.75) < 1e-15, f"ky {frac}")
    mapped = nearest_k(frac, regular_mp_kpoints())
    require(mapped["exact_on_wannier_mesh"], mapped)
    require(mapped["wannier_k_index_1based"] == 12 * 24 + 18 + 1, mapped)
    require(mapped["wannier_k_frac"][0] == 0.5, mapped)
    require(mapped["wannier_k_frac"][1] == 0.75, mapped)


def test_set1_walker_picks_max_occ() -> None:
    payload_a = {
        "cases": [
            {"dir": "low", "I_Wcm2": 2e11, "global_edge_max": {"ikx": 1, "iky": 1, "occ": 0.002, "band": 106}},
            {"dir": "mid", "I_Wcm2": 1e12, "global_edge_max": {"ikx": 3, "iky": 4, "occ": 0.01, "band": 105}},
        ]
    }
    payload_b = {
        "occupation": {
            "global_edge_max": {"ikx": 11, "iky": 16, "occ": 0.066, "band": 105}
        }
    }
    rows = []
    walk_edge_max(payload_a, "a.json", rows)
    walk_edge_max(payload_b, "b.json", rows)
    winner = max(rows, key=lambda r: r["occ"])
    require(winner["ikx"] == 11 and winner["iky"] == 16, winner)
    require(abs(winner["occ"] - 0.066) < 1e-12, winner)
    require(len(rows) == 3, rows)


def test_mv_weights_square_2d() -> None:
    b = 1.5
    bvecs = np.array(
        [
            [b, 0.0, 0.0],
            [-b, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, -b, 0.0],
        ]
    )
    w = mv_shell_weights(bvecs)
    require(np.allclose(w, 1.0 / (2.0 * b * b)), w)
    tensor = np.zeros((2, 2))
    for vec, weight in zip(bvecs, w):
        tensor += weight * np.outer(vec[:2], vec[:2])
    require(np.allclose(tensor, np.eye(2), atol=1e-12), tensor)


def test_berry_zero_for_identity_overlap() -> None:
    nb = 3
    bvecs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    weights = mv_shell_weights(bvecs)
    mats = [np.eye(nb, dtype=complex) for _ in bvecs]
    r_ang = berry_connection_ang(mats, bvecs, weights)
    require(np.max(np.abs(r_ang)) < 1e-14, np.max(np.abs(r_ang)))


def test_eta_formula_and_forbidden_keys() -> None:
    nb = 4
    r = np.zeros((3, nb, nb), dtype=complex)
    r[0, 0, 3] = 0.2  # Angstrom
    r[0, 3, 0] = 0.2
    eig_k = np.array([0.0, 1.0, 2.0, 4.0])
    _window, pi_idx, q_idx = proxy_index_sets(nb)
    # For nb=4 the production window (57-200) is empty; use explicit indices.
    pi_idx = np.array([0, 1])
    q_idx = np.array([2, 3])
    rows = pair_table(r, eig_k, pi_idx, q_idx)
    hit = [row for row in rows if row["p_qe_1based"] == 1 and row["q_qe_1based"] == 4]
    require(len(hit) == 1, rows)
    den_ha = 4.0 / 27.211386245988
    r_bohr = 0.2 / 0.529177210903
    expected = e0_au(I_BASE) * r_bohr / den_ha
    require(abs(hit[0]["eta_r_baseline_2e11"] - expected) < 1e-10, hit[0])
    require(hit[0]["delta_e_eV"] >= ETA_DENOM_FLOOR_EV, hit[0])
    r_pol = np.tensordot(np.array([1.0, 0.0, 0.0]), r, axes=(0, 0))
    stats = eta_r_stats_from_rpol(r_pol[np.ix_(pi_idx, q_idx)], eig_k[pi_idx], eig_k[q_idx])
    require(abs(stats["baseline_2e11"]["eta_r_max"] - expected) < 1e-10, stats)
    summary = summarize_eta(rows)
    payload = dumps_strict({"eta": summary, "pi_r_q": frobenius_pi_r_q(r, pi_idx, q_idx)})
    parsed = json.loads(payload)
    require("PrQ" not in json.dumps(parsed), parsed)
    for key in FORBIDDEN_KEYS:
        require(key not in parsed, key)


def test_parse_tiny_nnkp() -> None:
    text = """header
begin real_lattice
1 0 0
0 1 0
0 0 2
end real_lattice
begin recip_lattice
6.283185307179586 0 0
0 6.283185307179586 0
0 0 3.141592653589793
end recip_lattice
begin kpoints
2
0.0 0.0 0.0
0.5 0.0 0.0
end kpoints
begin nnkpts
2
1 2 0 0 0
1 1 -1 0 0
2 1 0 0 0
2 2 1 0 0
end nnkpts
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.nnkp"
        path.write_text(text, encoding="utf-8")
        nn = parse_nnkp(path)
        require(nn["num_kpts"] == 2, nn["num_kpts"])
        require(nn["nntot"] == 2, nn["nntot"])
        require(nn["nnkpts"].shape == (2, 2, 5), nn["nnkpts"].shape)


def test_register_set1_from_repo_json() -> None:
    repo = Path(__file__).resolve().parents[2]
    report = register_set1_edge_occupation(repo)
    win = report["winner"]
    require(win["occ"] > 0.06, win)
    require(win["source"].startswith("GH7_tail_k20_29403"), win)
    require(win["ikx"] >= 1 and win["iky"] >= 1, win)
    require("PrQ" not in json.dumps(to_jsonable(report)), report)


def test_true_pi_complement_and_frobenius() -> None:
    nb, nw, nq = 6, 4, 2
    u = np.zeros((nb, nw), dtype=complex)
    u[:nw, :] = np.eye(nw)
    lwindow = np.ones(nb, dtype=bool)
    comp = complement_basis(u, lwindow)
    require(comp["nq"] == nq, comp)
    require(abs(comp["pi_trace"] - nw) < 1e-12, comp["pi_trace"])
    require(abs(comp["q_trace"] - nq) < 1e-12, comp["q_trace"])
    require(comp["u_v_overlap_maxabs"] < 1e-14, comp["u_v_overlap_maxabs"])
    require(comp["v_gram_error"] < 1e-14, comp["v_gram_error"])
    require(comp["q_minus_iw_minus_pi_maxabs"] < 1e-14, comp)
    r = np.zeros((3, nb, nb), dtype=complex)
    r[0, :nw, nw:] = 0.5
    r[0, nw:, :nw] = 0.5
    norms, block = frobenius_rect(r, u, comp["v"])
    expected = 0.5 * float(np.sqrt(nw * nq))
    require(abs(norms["pi_rpol_q_frobenius_ang"] - expected) < 1e-12, norms)
    require(abs(norms["pi_rx_q_frobenius_ang"] - expected) < 1e-12, norms)
    pi_idx = np.arange(nw)
    q_idx = np.arange(nw, nb)
    proxy = frobenius_pi_r_q(r, pi_idx, q_idx)
    require(abs(proxy["pi_rpol_q_frobenius_ang"] - expected) < 1e-12, proxy)
    require(block.shape == (nw, nq), block.shape)


def test_skip_lines_binary_overshoot() -> None:
    payload = b"a\nb\nc\nd\ne\nrest"
    import io

    handle = io.BytesIO(payload)
    skip_lines_binary(handle, 3, bufsize=4, label="test")
    leftover = handle.read()
    require(leftover == b"d\ne\nrest", leftover)


def test_k20_map_set1_and_gamma() -> None:
    rows = map_k20_to_wannier()
    require(len(rows) == 400, len(rows))
    gamma = next(r for r in rows if r["ikx"] == 1 and r["iky"] == 1)
    require(gamma["wannier_k_index_1based"] == 1, gamma)
    require(gamma["exact_on_wannier_mesh"], gamma)
    corner = next(r for r in rows if r["ikx"] == 17 and r["iky"] == 2)
    require(corner["wannier_k_index_1based"] == 458, corner)


def main() -> None:
    test_sbe_frac_and_exact_mesh_map()
    test_set1_walker_picks_max_occ()
    test_mv_weights_square_2d()
    test_berry_zero_for_identity_overlap()
    test_eta_formula_and_forbidden_keys()
    test_parse_tiny_nnkp()
    test_register_set1_from_repo_json()
    test_true_pi_complement_and_frobenius()
    test_skip_lines_binary_overshoot()
    test_k20_map_set1_and_gamma()
    print("test_audit_pi_rq_coupling: PASS")


if __name__ == "__main__":
    main()

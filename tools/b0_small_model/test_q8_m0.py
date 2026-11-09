#!/usr/bin/env python3
"""Gates for the M0 eight-band model and its P0 classical reference (fast, CPU, no quantum light)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import (  # noqa: E402
    P0Protocol,
    cache_identity_matches,
    field_resources,
    initial_rho,
    k_grid,
    kappa_shift,
    p1_protocol,
    project_harmonic,
    propagate,
    run_fingerprint,
)
from q8_m0_model import (  # noqa: E402
    INV_C,
    N_OCC,
    M0Params,
    R0,
    build_geometry,
    current_ops,
    dh8,
    f_and_grad,
    g_and_grad,
    gamma_energies,
    h8,
    h8_and_current,
    mirror_matrix_inplane,
    op_P,
    op_V_specified,
    op_c2_nhat,
    op_mirror_a2,
    op_sewing,
    op_sz,
    op_theta_unitary,
    same_spectrum_barred_residual,
    t0_and_grad,
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def fro(a) -> float:
    return float(np.linalg.norm(a))


def test_geometry_pairing() -> None:
    geo = build_geometry()
    lat = geo.lat()
    c = np.asarray(INV_C)
    for src, dst in (("tA", "bB"), ("tB", "bA")):
        img = (2.0 * c - np.array(geo.sites_paired_frac[src])) % 1.0
        require(np.max(np.abs(img - np.array(geo.sites_paired_frac[dst]))) < 1e-12, f"pairing {src}->{dst}")
    # bA moved only ~3.5e-5 Å, others unchanged
    for name in ("tA", "tB", "bB"):
        dv = (np.array(geo.sites_paired_frac[name]) - np.array(geo.sites_orig_frac[name]))
        dv = (dv + 0.5) % 1.0 - 0.5
        require(np.linalg.norm(dv @ lat) < 1e-6, f"{name} should not move")
    dba = (np.array(geo.sites_paired_frac["bA"]) - np.array(geo.sites_orig_frac["bA"]))
    dba = (dba + 0.5) % 1.0 - 0.5
    require(np.linalg.norm(dba @ lat) < 1e-4, "bA snap must stay tiny")
    # three NN bonds same length, 120° up to production distortion
    ls = [np.linalg.norm(v) for v in geo.nn_deltas_cart]
    require(max(ls) - min(ls) < 5e-3, f"NN lengths {ls}")
    # four interlayer bonds nearly equal length ~7.05 Å
    l4 = [np.linalg.norm(b["delta_cart"]) for b in geo.interlayer_bonds]
    require(max(l4) - min(l4) < 5e-3 and 7.0 < min(l4) < 7.1, f"interlayer lengths {l4}")
    require(geo.interlayer_bonds[2]["R"] == [0, 1, 0], "AB bond must carry R=(0,1,0)")


def test_structure_factors() -> None:
    geo = build_geometry()
    fK, _ = f_and_grad(geo, geo.k_cart([1 / 3, 1 / 3]))
    gK, _ = g_and_grad(geo, geo.k_cart([1 / 3, 1 / 3]))
    gKp, _ = g_and_grad(geo, geo.k_cart([-1 / 3, -1 / 3]))
    fG, _ = f_and_grad(geo, geo.k_cart([0, 0]))
    gG, _ = g_and_grad(geo, geo.k_cart([0, 0]))
    require(abs(fK) < 1e-12, f"f(K)={fK}")
    require(abs(gK + 3 * np.sqrt(3)) < 1e-12, f"g(K)={gK}")
    require(abs(gKp - 3 * np.sqrt(3)) < 1e-12, "g(K')")
    require(abs(fG - 3) < 1e-12 and abs(gG) < 1e-12, "Gamma f,g")


def test_gamma_analytic_and_symmetries() -> None:
    geo = build_geometry()
    p = R0
    e = np.linalg.eigvalsh(h8(geo, p, geo.k_cart([0, 0]), +1))
    ana = np.repeat(gamma_energies(p), 2)
    require(np.max(np.abs(e - ana)) < 1e-12, "Gamma analytic energies")
    P, UT, SZ = op_P(), op_theta_unitary(), op_sz()
    PT = P @ UT
    require(fro(PT @ PT.conj() + np.eye(8)) < 1e-14, "(PΘ)^2 = -1")
    rng = np.random.default_rng(7)
    for _ in range(6):
        kf = rng.uniform(-0.7, 0.7, 2)
        k = geo.k_cart(kf)
        hp, hm = h8(geo, p, k, +1), h8(geo, p, k, -1)
        require(fro(hp - hp.conj().T) < 1e-14, "Hermitian")
        require(fro(P @ hp @ P.conj().T - h8(geo, p, -k, -1)) < 1e-13, "P")
        require(fro(UT @ hp.conj() @ UT.conj().T - h8(geo, p, -k, -1)) < 1e-13, "Theta")
        require(fro(PT @ hp.conj() @ PT.conj().T - hp) < 1e-13, "PTheta")
        require(fro(hp @ SZ - SZ @ hp) < 1e-14, "[H,sz]")
        for gf in ([1, 0], [0, 1], [1, 1]):
            g = geo.k_cart(gf)
            U = op_sewing(geo, g)
            require(fro(h8(geo, p, k + g, +1) - U.conj().T @ hp @ U) < 1e-12, "sewing")
        ev = np.linalg.eigvalsh(hp)
        require(np.max(np.abs(ev[0::2] - ev[1::2])) < 1e-12, "Kramers pairs")
        # spectrum even in Δ (T0 normal) — structural fact used in the report
        require(np.max(np.abs(ev - np.linalg.eigvalsh(hm))) < 1e-12, "E_+N(k) = E_-N(k)")
        t0, _ = t0_and_grad(geo, p, k)
        require(fro(t0 @ t0.conj().T - t0.conj().T @ t0) < 1e-14, "T0 normal")
        # V = tau_x s_z is NOT a symmetry (T0 not Hermitian) and residual equals 2 tau_y Y
        V = op_V_specified()
        y = (t0 - t0.conj().T) / (2j)
        y4 = np.kron(y, np.eye(2))
        pred = np.zeros((8, 8), dtype=complex)
        pred[:4, 4:] = -2j * y4
        pred[4:, :4] = 2j * y4
        got = V @ hp @ V.conj().T - hm
        require(fro(got - pred) < 1e-13, "V residual = 2 tau_y Y")
        # approximate C2h (production metric): small but nonzero
        Mm = mirror_matrix_inplane(geo)
        UM, UC = op_mirror_a2(geo), op_c2_nhat(geo)
        require(fro(UM @ hp @ UM.conj().T - h8(geo, p, Mm @ k, -1)) < 1e-4, "mirror approx")
        require(fro(UC @ hp @ UC.conj().T - h8(geo, p, -(Mm @ k), +1)) < 1e-4, "C2 approx")
        # analytic derivative vs central FD
        D = dh8(geo, p, k, +1)
        for a in range(2):
            dk = np.zeros(2)
            dk[a] = 1e-5
            fd = (h8(geo, p, k + dk, +1) - h8(geo, p, k - dk, +1)) / 2e-5
            require(fro(fd - D[a]) / fro(D[a]) < 1e-7, "FD derivative")
    # breaking v_AB = v_BA destroys T0 normality and the Δ-evenness (negative control of the theorem)
    geo2 = build_geometry()
    geo2.interlayer_bonds[2]["amplitude"] = "u"  # AB now uses u, BA uses v -> non-normal when u≠v
    p2 = M0Params(u=0.17, v=0.05)
    k = geo.k_cart([0.23, -0.31])
    t0, _ = t0_and_grad(geo2, p2, k)
    require(fro(t0 @ t0.conj().T - t0.conj().T @ t0) > 1e-3, "non-normal control")
    e1 = np.linalg.eigvalsh(h8(geo2, p2, k, +1))
    e2 = np.linalg.eigvalsh(h8(geo2, p2, k, -1))
    require(np.max(np.abs(e1 - e2)) > 1e-4, "Δ-evenness should fail for non-normal T0")
    chk = same_spectrum_barred_residual(geo, p, k, spin=1.0)
    require(chk["T_normal_residual"] < 1e-12, "T0 normal in same-spectrum helper")
    require(chk["residual"] < 1e-8, f"barred τx H* τx = H(-Δ) residual {chk}")
    require(chk["hbar_imag_max"] < 1e-8, "barred in-layer block should be real")


def test_no_constant_intertwiner() -> None:
    geo = build_geometry()
    p = R0
    rows = []
    for kf in ([0, 0], [0.5, 0], [1 / 3, 1 / 3], [0.17, -0.11], [0.31, 0.22], [-0.08, 0.41]):
        k = geo.k_cart(kf)
        hp, hm = h8(geo, p, k, +1), h8(geo, p, k, -1)
        rows.append(np.kron(np.eye(8), hm) - np.kron(hp.T, np.eye(8)))
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    require(s[-1] > 1e-2, f"unexpected constant intertwiner nullspace, sv={s[-1]}")


def test_short_propagation_identity() -> None:
    """Field-reversal identity J_N[A] = -J_{-N}[-A] and basic ρ health on a tiny run."""
    geo = build_geometry()
    p = R0
    prot = P0Protocol(n_cycles=1)
    kf, kc, w = k_grid(geo, 12)
    e = np.linalg.eigvalsh(h8(geo, p, kc, +1))
    mu = 0.5 * (e[:, N_OCC - 1].max() + e[:, N_OCC].min())
    require(e[:, N_OCC].min() > e[:, N_OCC - 1].max(), "coarse grid must be gapped for the test")
    rho, _ = initial_rho(geo, p, kc, +1, mu)
    require(np.max(np.abs(np.einsum("kii->k", rho).real - N_OCC)) < 1e-12, "Tr rho0 = 4")
    r1 = propagate(geo, p, prot, nk=12, nt_per_cycle=40, neel=+1, sign=+1, mu=mu, verbose=False)
    r2 = propagate(geo, p, prot, nk=12, nt_per_cycle=40, neel=-1, sign=-1, mu=mu, verbose=False)
    jmax = np.max(np.abs(r1["J_sheet"]))
    require(jmax > 0, "current must respond")
    require(np.max(np.abs(r1["J_sheet"] + r2["J_sheet"])) < 1e-12 * max(jmax, 1.0), "J_N[A] = -J_-N[-A]")
    require(np.max(np.abs(r1["J_sheet"][0])) < 1e-12, "equilibrium current at t=0")
    require(max(r1["diag"]["idem"]) < 1e-10 and max(r1["diag"]["herm"]) < 1e-12, "rho health")
    require(np.max(np.abs(np.array(r1["diag"]["tr_rho"]) - N_OCC)) < 1e-10, "Tr rho conserved")
    # Δ = 0 control: J_o must vanish identically
    p0 = M0Params(delta=0.0)
    e0 = np.linalg.eigvalsh(h8(geo, p0, kc, +1))
    mu0 = 0.5 * (e0[:, N_OCC - 1].max() + e0[:, N_OCC].min())
    if e0[:, N_OCC].min() > e0[:, N_OCC - 1].max():
        c1 = propagate(geo, p0, prot, nk=12, nt_per_cycle=40, neel=+1, sign=+1, mu=mu0, verbose=False)
        c2 = propagate(geo, p0, prot, nk=12, nt_per_cycle=40, neel=-1, sign=+1, mu=mu0, verbose=False)
        require(np.max(np.abs(c1["J_sheet"] - c2["J_sheet"])) < 1e-14, "Δ=0 control J_o ≠ 0")
    # kappa shift amplitude
    a = kappa_shift(np.array([prot.t_pulse / 2]), geo, prot)
    require(abs(abs(a[0, 0]) * geo.a_nn_mean_ang - prot.a0_dimless * abs(np.cos(prot.omega0 * prot.t_pulse / 2))) < 1e-12, "A(t) amplitude")
    r3 = propagate(geo, p, prot, nk=8, nt_per_cycle=20, neel=+1, sign=+1, mu=mu, verbose=False, subspace=True, k_orders=(0, 1, 2))
    rec = r3["J_vv_sheet"] + r3["J_cc_sheet"] + r3["J_vc_sheet"] - r3["J_sheet"]
    require(np.max(np.abs(rec)) < 1e-10, "vv+cc+vc reconstructs J")
    require(r3["subspace_reconstruct_maxabs"] < 1e-10, "stored reconstruct residual")
    dU, wrk = float(r3["delta_U_e"]), float(r3["work_integral"])
    require(abs(dU - wrk) / max(abs(dU), 1e-8) < 0.03, f"ΔU_e vs work relative residual {dU} {wrk}")
    require("k_harm_total" in r3 and r3["k_harm_total"].shape[0] == 3, "k-resolved harmonics")
    _, kc_ops, _ = k_grid(geo, 6, shift=0.0)
    j_ops = current_ops(geo, p, kc_ops, +1)
    _, j_h = h8_and_current(geo, p, kc_ops, +1)
    require(np.max(np.abs(j_ops - j_h)) < 1e-14, "current_ops must match h8_and_current")
    r4 = propagate(geo, p, prot, nk=8, nt_per_cycle=20, neel=+1, sign=+1, mu=mu, verbose=False, subspace=False, k_orders=(2,))
    z_k = np.einsum("k,k->", r4["w_k"], r4["k_harm_total"][0, :, 0]) / geo.a_cell_ang2
    z_j = project_harmonic(r4["times"], r4["J_sheet"], 2.0 * prot.omega0, "sin2")[0]
    require(abs(complex(z_k) - complex(z_j)) < 1e-12, "k_orders without subspace reconstructs J_sheet H2")


def test_p1_resources_and_identity() -> None:
    geo = build_geometry()
    p0 = P0Protocol()
    p1 = p1_protocol(0.8)
    require(abs(p1.hbar_omega0 - 0.05) < 1e-15 and abs(p1.a0_dimless - 0.8) < 1e-15, "P1 defaults")
    require(abs(p1.t_pulse / p0.t_pulse - 4.0) < 1e-12, "8-cycle P1 lasts 4× P0")
    res0 = field_resources(geo, p0, 0.09254653216396126)
    res1 = field_resources(geo, p1, 0.09254653216396126)
    require(abs(res1["F_E"] / res0["F_E"] - 1.0) < 0.05, f"A*=0.8 should match P0 ∫E²dt, got {res1['F_E']/res0['F_E']}")
    require(res1["m_min_center_frequency"] == 2, "P1 center-frequency two-photon threshold")
    require(res0["m_min_center_frequency"] == 1, "P0 one-photon open at center frequency")
    kf0, kc0, _ = k_grid(geo, 8, shift=0.0)
    kf1, kc1, _ = k_grid(geo, 8, shift=0.5)
    require(np.max(np.abs(kc0[0])) < 1e-12, "shift 0 includes Γ")
    require(np.min(np.linalg.norm(kc1, axis=1)) > 1e-8, "shift 0.5 excludes Γ")
    ida = run_fingerprint(geo, R0, p0, nk=8, nt_per_cycle=20, mu=0.0, grid_shift=0.0, subspace=False, k_orders=(), neel=+1, sign=+1)
    idb = run_fingerprint(geo, R0, p1, nk=8, nt_per_cycle=20, mu=0.0, grid_shift=0.0, subspace=False, k_orders=(), neel=+1, sign=+1)
    require(ida["identity_sha256"] != idb["identity_sha256"], "P0/P1 identities must differ")
    require(ida["protocol_id"] != idb["protocol_id"], "protocol_id differs")
    require(ida["physics_identity_sha256"] != idb["physics_identity_sha256"], "physics identity differs")
    require(abs(res1["E_spectrum_weight_above_direct_gap"] - 2.0683e-6) / 2.0683e-6 < 0.02, "P1 continuous above-gap weight")
    require(res0["E_spectrum_weight_above_direct_gap"] > 0.99999, "P0 almost all weight above gap")
    require(res1["E_spectrum_weight_above_direct_gap_legacy_rfft_unpadded"] < 1e-9, "legacy rfft still tiny")
    ida_mod = dict(ida)
    ida_mod["classical_module_sha256"] = "deadbeef"
    require(cache_identity_matches(ida, ida_mod), "module hash must not break physics cache")
    keys = {tuple(np.round(np.mod(row, 1.0), 12)) for row in kf1}
    for row in kf1:
        partner = tuple(np.round(np.mod(-row, 1.0), 12))
        require(partner in keys, "shift 0.5 remains inversion-paired")


def test_h2_classification_boundaries() -> None:
    from q8_m0_h2_qualify import classify_channel

    z_ref = 1e-6 + 0j
    ok = classify_channel(z_ref, z_k=z_ref * 1.01, z_ind=z_ref * 0.99, z_dt=z_ref)
    require(ok["status"] == "qualified_preliminary", "5% stable channel")
    ylike = classify_channel(z_ref, z_k=z_ref * 3.17, z_ind=z_ref * 0.65, z_dt=z_ref)
    require(ylike["status"] == "unresolved", "change larger than |z| is unresolved")
    xlike = classify_channel(z_ref, z_k=z_ref * 0.4, z_ind=-0.9 * z_ref, z_dt=z_ref)
    require(xlike["status"] == "unresolved", "phase flip is unresolved")
    trend = classify_channel(z_ref, z_k=z_ref * 0.7, z_ind=z_ref * 1.2, z_dt=z_ref)
    require(trend["status"] == "trend_not_qualified", "sub-signal change with stable phase")
    tiny = classify_channel(1e-12 + 0j, z_k=1e-12, z_ind=1e-12, z_dt=1e-12)
    require(tiny["status"] == "below_reporting_resolution", "tiny reference and tiny diffs")
    tiny_but_wild = classify_channel(1e-12 + 0j, z_k=1e-6, z_ind=1e-12, z_dt=1e-12)
    require(tiny_but_wild["status"] == "unresolved", "tiny reference with large absolute disagreement")
    missing = classify_channel(z_ref, z_k=z_ref, z_ind=None, z_dt=z_ref)
    require(missing["status"] == "unresolved", "incomplete checks")


def main() -> None:
    test_geometry_pairing()
    test_structure_factors()
    test_gamma_analytic_and_symmetries()
    test_no_constant_intertwiner()
    test_short_propagation_identity()
    test_p1_resources_and_identity()
    test_h2_classification_boundaries()
    print("test_q8_m0: PASS")


if __name__ == "__main__":
    main()

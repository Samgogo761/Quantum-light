#!/usr/bin/env python3
"""Fast gates for M0 CJJ (Q3/Q8/mean/window). Does not run the 8-trajectory campaign."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import (  # noqa: E402
    c_equal_from_x,
    choose_probe_k,
    d_from_y,
    gram_psd_residual,
    occup_empty_bases,
    propagate_cjj,
    undriven_trapz_check,
    undriven_y,
    window_fourier,
    x_from_bases,
)
from q8_m0_classical import p1_protocol  # noqa: E402
from q8_m0_model import N_OCC, R0, build_geometry, current_ops, h8  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def test_window_fourier_zero() -> None:
    tp = 8.0 * (2.0 * np.pi / 0.05)
    i0 = complex(window_fourier(0.0, tp))
    require(abs(i0 - 0.5) < 1e-12, f"I(0)={i0}")


def test_x_mean_and_gram() -> None:
    geo = build_geometry()
    kf = np.array([[0.0, 0.0], [0.2, 0.1], [0.7, 0.4]])
    kc = geo.k_cart(kf)
    mu = -0.009010620372198276
    h0 = h8(geo, R0, kc, +1)
    vv, vc, _e = occup_empty_bases(h0, mu)
    j = current_ops(geo, R0, kc, +1)
    x = x_from_bases(j, vv, vc)
    mean = np.real(np.einsum("kiv,kaij,kjv->ka", vv.conj(), j, vv))
    rho = vv @ np.conj(np.swapaxes(vv, 1, 2))
    mean_rho = np.real(np.einsum("kij,kaji->ka", rho, j))
    require(np.max(np.abs(mean - mean_rho)) < 1e-12, "mean from Vv vs rho")
    c = c_equal_from_x(x)
    require(np.max(np.abs(c - np.conj(np.swapaxes(c, 1, 2)))) < 1e-12, "equal-time c Hermitian")
    evals = np.linalg.eigvalsh(c)
    require(np.min(evals) >= -1e-12, "equal-time c PSD")
    u = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 4, 4)) + 1j * np.random.default_rng(1).normal(size=(3, 4, 4)))[0]
    vv_r = vv @ u
    x_r = x_from_bases(j, vv_r, vc)
    c_r = c_equal_from_x(x_r)
    require(np.max(np.abs(c_r - c)) < 1e-10, "occupied rotation invariance")


def test_short_driven_and_undriven() -> None:
    geo = build_geometry()
    prot = p1_protocol(0.8)
    mu = -0.009010620372198276
    rec = choose_probe_k(geo, R0, mu)
    require(rec["kf"].shape == (6, 2), "six probe k")
    r = propagate_cjj(
        geo, R0, prot, nt_per_cycle=20, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=rec["kf"], save_full_x=True,
    )
    require(r["mean_vs_rho_maxabs"] < 1e-10, f"mean vs rho {r['mean_vs_rho_maxabs']}")
    require(float(np.max(r["diag"]["idem"])) < 1e-10, "idempotent rho")
    b = r["b_n"]
    mu_t = r["mu_per_cell"]
    z1 = complex(b[0, 0])
    from q8_m0_classical import project_harmonic
    z1b = complex(project_harmonic(r["times"], mu_t, prot.omega0, "sin2")[0])
    require(abs(z1 - z1b) < 1e-12, "b reconstructs filtered mean")
    psd = gram_psd_residual(r["d_namb"])
    require(psd["herm_residual"] < 1e-10, "d Hermitian")
    require(psd["psd_ok"], f"d PSD {psd}")
    xa = r["X_full"][5, :, 0]
    xb = r["X_full"][11, :, 1]
    c8 = np.einsum("kcv,kcv->k", np.conj(xa), xb)
    require(np.isfinite(c8).all(), "Q8 finite")
    ud = undriven_y(geo, R0, prot, rec["kf"], mu, +1)
    chk = undriven_trapz_check(geo, R0, prot, rec["kf"], mu, +1, 80)
    require(chk["max_abs_Y_diff"] < 5e-4, f"undriven trapz vs analytic {chk}")
    require(float(np.min(np.linalg.eigvalsh(0.5 * (ud["d_namb"].reshape(18, 18) + ud["d_namb"].reshape(18, 18).conj().T)))) >= -1e-10, "undriven d PSD")


def test_cjj_floor_and_cache_guards() -> None:
    from q8_m0_h2_qualify import classify_channel
    from q8_m0_cjj_validate import (
        CJJ_ABS_REPORT_RESOLUTION,
        CJJ_D_SHAPE,
        cache_reuse_decision,
        signed_kernel_diff,
    )

    z = 1e-11 + 0j
    mean_floor = classify_channel(z, z_k=z, z_ind=z, z_dt=z)
    cjj_floor = classify_channel(z, z_k=z, z_ind=z, z_dt=z, abs_floor=CJJ_ABS_REPORT_RESOLUTION)
    require(mean_floor["status"] == "below_reporting_resolution", "mean-current 1e-9 still tiny")
    require(cjj_floor["status"] == "qualified_preliminary", "CJJ 1e-12 floor treats 1e-11 as finite")
    diff = signed_kernel_diff(0.016, 2e-12)
    require(diff["re"] > 0.0, "signed undriven subtraction keeps sign")
    require(abs(diff["im"]) < 1e-18, "imag residual of real kernels")
    ident = {"cjj_identity_sha256": "abc", "cjj_module_sha256": "mod", "nt_per_cycle": 400, "protocol": {"n_cycles": 8}}
    stored = dict(ident)
    stored["kf_kind"] = "mp"
    n_k = 4
    payload = {
        "n_k": n_k,
        "kf": np.zeros((n_k, 2)),
        "Y": np.zeros((n_k, 9, 2, 4, 4), dtype=complex),
        "d_namb": np.zeros(CJJ_D_SHAPE, dtype=complex),
        "times": np.linspace(0.0, 1.0, 8 * 400 + 1),
        "snapshot_X": np.zeros((3, n_k, 2, 4, 4), dtype=complex),
    }
    ok = cache_reuse_decision(stored, ident, payload, name="V24")
    require(ok["reuse"], f"matching cache should load {ok}")
    bad = dict(payload)
    bad["snapshot_X"] = np.zeros((3, 1, 2, 4, 4), dtype=complex)
    refuse = cache_reuse_decision(stored, ident, bad, name="V24")
    require(not refuse["reuse"], "n_k vs snapshot mismatch must refuse")
    require("snapshot_X_n_k_mismatch" in refuse["reasons"], refuse["reasons"])
    stored_old = dict(stored)
    stored_old["cjj_module_sha256"] = "deadbeef"
    silent = cache_reuse_decision(stored_old, ident, payload, name="V24")
    require(not silent["reuse"], "undocumented module hash must not silently load")
    from q8_m0_cjj_validate import DOCUMENTED_COMPATIBLE_CJJ_MODULE
    old_h = next(iter(DOCUMENTED_COMPATIBLE_CJJ_MODULE))
    rec = DOCUMENTED_COMPATIBLE_CJJ_MODULE[old_h]
    cur_h = rec["allowed_current_module_sha256"][0]
    ident_pin = dict(ident)
    ident_pin["cjj_module_sha256"] = cur_h
    stored_b0 = dict(stored)
    stored_b0["cjj_module_sha256"] = old_h
    b0_ok = cache_reuse_decision(stored_b0, ident_pin, payload, name="B0_sixk")
    require(b0_ok["reuse"], f"pinned B0 hash pair should load {b0_ok}")
    ident_future = dict(ident_pin)
    ident_future["cjj_module_sha256"] = "ffffffffffffffff"
    b0_future = cache_reuse_decision(stored_b0, ident_future, payload, name="B0_sixk")
    require(not b0_future["reuse"], "future cjj.py hash must not inherit B0 compat")
    b0_wrong_name = cache_reuse_decision(stored_b0, ident_pin, payload, name="V24")
    require(not b0_wrong_name["reuse"], "B0 compat must not apply to BZ jobs")
    from q8_m0_cjj_validate import explicit_rel_columns
    cols = explicit_rel_columns(
        {"rel_V24": 0.0218, "rel_shift": 0.0073},
        {"rel_V24": 0.0256, "rel_shift": 0.00085},
    )
    require(abs(cols["plus_rel_V24"] - 0.0218) < 1e-12, "plus column is +N")
    require(abs(cols["max_of_two_domains_rel_V24"] - 0.0256) < 1e-12, "max is worse of two")
    from q8_m0_cjj_qualify_v2 import classify_v2
    ok = classify_v2(1.0, 1.01, 0.99, 0.985)
    require(ok["status"] == "qualified_preliminary", f"stable v2 {ok}")
    flip = classify_v2(1e-4, 1e-4, -1e-4, -1e-4)
    require(flip["status"] == "unresolved", f"G96 vs S96 sign flip {flip}")


def test_cjj_source_kernel_maps() -> None:
    from q8_m0_classical import k_grid
    from q8_m0_cjj_source_diag import gram_cross, gram_diag, partner_index, tile_ids, tile_partner
    from q8_m0_model import build_geometry

    rng = np.random.default_rng(2)
    y = rng.normal(size=(7, 4, 4)) + 1j * rng.normal(size=(7, 4, 4))
    g = gram_diag(y)
    require(np.max(np.abs(g.imag)) < 1e-12, "diag gram imag")
    require(float(np.min(g.real)) >= -1e-12, "diag gram PSD")
    yb = rng.normal(size=(7, 4, 4)) + 1j * rng.normal(size=(7, 4, 4))
    c = gram_cross(y, yb)
    require(c.shape == (7,), "cross gram shape")
    geo = build_geometry()
    for nk, shift in ((8, 0.0), (8, 0.5)):
        kf, _kc, _w = k_grid(geo, nk, shift=shift)
        pidx, unpaired = partner_index(kf)
        require(unpaired == 0, f"unpaired {nk} shift={shift}")
        require(np.allclose(np.mod(kf + kf[pidx], 1.0), 0.0, atol=1e-10), "k + (-k) = G")
        require(np.array_equal(pidx[pidx], np.arange(len(pidx))), "partner involution")
    require(tile_partner(0) == 0, "Gamma tile self-inverse")
    require(tile_partner(tile_partner(7)) == 7, "tile partner involution")
    kf48, _, _ = k_grid(geo, 48, shift=0.0)
    tid = tile_ids(kf48)
    require(tid.min() == 0 and tid.max() == 35, "6x6 tiles")


def test_a2_mirror_eigenvectors() -> None:
    from q8_m0_a2_selection import unit
    from q8_m0_model import build_geometry, mirror_matrix_inplane

    geo = build_geometry()
    m = mirror_matrix_inplane(geo)
    a2 = unit(geo.a2_cart)
    nh = unit(geo.mirror_normal_inplane)
    require(np.linalg.norm(m @ a2 - a2) < 1e-12, "M fixes a2")
    require(np.linalg.norm(m @ nh + nh) < 1e-12, "M flips n-hat")
    require(np.linalg.norm((-m) @ a2 + a2) < 1e-12, "C2 flips a2")
    require(np.linalg.norm((-m) @ nh - nh) < 1e-12, "C2 fixes n-hat")


def test_s1_reduced_kernel() -> None:
    from q8_m0_classical import p1_protocol
    from q8_m0_s1_kernel import (
        parallel_perp_basis,
        parallel_perp_from_cartesian,
        reduced_k_mu_a,
        spec_dict,
    )
    from q8_m0_model import build_geometry

    geo = build_geometry()
    e_par, e_perp = parallel_perp_basis(geo)
    require(abs(np.dot(e_par, e_perp)) < 1e-12, "∥ ⊥ orthonormal")
    require(abs(np.linalg.norm(e_par) - 1.0) < 1e-12, "∥ unit")
    jxy = np.array([e_par[0] + 0.3 * e_perp[0], e_par[1] + 0.3 * e_perp[1]])
    jp, jn = parallel_perp_from_cartesian(jxy, geo)
    require(abs(jp - 1.0) < 1e-12 and abs(jn - 0.3) < 1e-12, "xy → ∥⊥")
    prot = p1_protocol(0.8)
    t = np.linspace(0.0, prot.t_pulse, 21)
    rec = reduced_k_mu_a(t, prot.omega0, prot.t_pulse, geo)
    k = rec["K_tilde"]
    require(k.shape == (2, 2, 21), "K shape μ,a,t")
    # parallel filter has no perp polarization
    require(np.max(np.abs(np.einsum("a,at->t", e_perp, k[0]))) < 1e-12, "H1∥ has no ⊥")
    require(np.max(np.abs(np.einsum("a,at->t", e_par, k[1]))) < 1e-12, "H2⊥ has no ∥")
    require(spec_dict()["status"] == "adopted_as_source_filter_not_photon_mode", "source filter adopted")
    y = np.zeros((3, 9, 2, 4, 4), dtype=complex)
    y[:, 0, 0] = 1.0
    y[:, 1, 1] = 0.5
    from q8_m0_s1_kernel import s1_d_mu_nu

    d = s1_d_mu_nu(y, np.full(3, 1.0 / 3.0), geo)
    require(d.shape == (2, 2), "S1 Gram 2x2")
    require(d[0, 0].real > 0.0 and d[1, 1].real > 0.0, "diagonal PSD-like")


def test_p1_a2_drive_along_a2() -> None:
    from q8_m0_classical import kappa_shift, p1_a2_protocol, p1_protocol, pol_vector
    from q8_m0_s1_kernel import parallel_perp_basis

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    px = p1_protocol(0.8)
    require(prot.protocol_id != px.protocol_id, "a2 protocol id differs")
    require(abs(prot.hbar_omega0 - px.hbar_omega0) < 1e-15, "same ω")
    require(abs(prot.a0_dimless - px.a0_dimless) < 1e-15, "same A*")
    e = pol_vector(prot, geo)
    e_par, e_perp = parallel_perp_basis(geo)
    require(np.linalg.norm(e - e_par) < 1e-12, "pol vector is a2")
    t = 0.3 * prot.t_pulse
    kap = np.asarray(kappa_shift(t, geo, prot, +1), dtype=float)
    require(abs(np.dot(kap, e_perp)) < 1e-12 * max(1.0, np.linalg.norm(kap)), "A has no n-hat")
    require(abs(abs(np.dot(kap, e_par)) - np.linalg.norm(kap)) < 1e-12, "A is along a2")


def test_k1_mode_not_photon() -> None:
    from q8_m0_k1_chi_interface import ChiNotChosenError, apply_chi_to_shape_kernel, chi_spec
    from q8_m0_k1_mode import spec_dict as k1_spec
    from q8_m0_k1_packet import spec_scale

    rec = k1_spec()
    require(rec["status"] == "reference_mode_adopted_chi_unset", "reference adopted")
    require(rec["photon_output"] == "not_run", "no photon output")
    require(rec["kernel"]["q9_is_not_the_photon_mode"] is True, "Q9 is not the photon")
    require(rec["kernel"]["commutator_normalized_on_packet"] is True, "packet commutator")
    require(rec["kernel"]["radiation_kernel_chi"].startswith("shape_K_1_over_sqrt_omega"), "K shape implemented")
    require(rec["reference_packet"]["status"].startswith("adopted_as_reference"), "packet is reference")
    require(rec["reference_packet"]["role"].startswith("first_reference"), "not final experiment")
    require(rec["conversion"]["evaluated"] is False, "do not evaluate ⟨n⟩")
    require(rec["chi_physical_form"] == "not_chosen", "do not invent chi")
    require(chi_spec()["ready_to_evaluate_photons"] is False, "chi not ready")
    blocked = False
    try:
        apply_chi_to_shape_kernel(None, None)
    except ChiNotChosenError:
        blocked = True
    require(blocked, "apply_chi must refuse")
    sc = spec_scale()
    require(abs(sc["scale"]["commutator_integral"] - 1.0) < 1e-6, "∫|φ|²=1")
    require(sc["scale"]["plancherel"]["rel_diff"] < 1e-3, "Plancherel")
    require(sc["scale"]["peaks"]["abs_phi_H2"] > 10.0 * sc["scale"]["peaks"]["abs_phi_H1"], "H2 isolated")
    require(sc["photon_output"] == "not_run", "scale helper is not ⟨n⟩")
    require(sc["chi"]["form"] == "not_chosen", "chi form empty")
    require(sc["chi"]["slot_name"] == "chi_rad", "radiation slot name")
    require(sc["chi"]["not_electronic_response_R_A"] is True, "chi_rad is not R_A")


def test_response_d_and_prefix() -> None:
    from q8_m0_response import (
        ResponseNotChosenError,
        apply_response_to_output,
        d_ops,
        d_ops_finite_difference,
        prefix_state_dmu,
        response_spec,
        two_time_state_dmu,
    )

    geo = build_geometry()
    kf = np.array([[0.0, 0.0], [0.2, 0.1], [0.7, 0.4]])
    kc = geo.k_cart(kf)
    ana = d_ops(geo, R0, kc, +1)
    fd = d_ops_finite_difference(geo, R0, kc, +1, eps=1e-4)
    require(np.max(np.abs(ana - fd)) / max(np.max(np.abs(ana)), 1e-30) < 1e-5, "D analytic vs FD")
    rec = choose_probe_k(geo, R0, -0.009010620372198276)
    prot = p1_protocol(0.8)
    r = propagate_cjj(
        geo, R0, prot, nt_per_cycle=16, neel=+1, sign=+1, mu=-0.009010620372198276,
        verbose=False, kf_frac=rec["kf"], save_full_x=True,
    )
    x = np.asarray(r["X_full"])
    w = np.asarray(r["w_k"], dtype=float)
    dt = float(r["dt"])
    h = np.zeros((x.shape[0], 2), dtype=float)
    h[:, 0] = 1.0
    d1 = prefix_state_dmu(x, w, h, dt)
    d2 = two_time_state_dmu(x, w, h, dt)
    require(np.max(np.abs(d1 - d2)) < 1e-10 * max(1.0, np.max(np.abs(d2))), "prefix vs two-time")
    spec = response_spec()
    require(spec["slot_name"] == "R_A", "response slot")
    require(spec["not_chi_rad"] is True, "R_A is not chi_rad")
    require(spec["ready_to_evaluate_photons"] is False, "no photons from R")
    blocked = False
    try:
        apply_response_to_output()
    except ResponseNotChosenError:
        blocked = True
    require(blocked, "apply_response must refuse")


def test_inventory_classifies_full_x_vs_y() -> None:
    from q8_m0_inventory import classify_npz

    repo = Path(__file__).resolve().parents[2]
    sixk = repo / "results/q_line/cjj_m0_p1_a2_s1_20260917/B0_sixk/B0_sixk_N+1_A+1.npz"
    v48 = repo / "results/q_line/cjj_m0_p1_a2_s1_20260917/V48/V48_N+1_A+1.npz"
    if not sixk.exists() or not v48.exists():
        return
    a = classify_npz(sixk)
    b = classify_npz(v48)
    require(a["has_X_full"] is True, "sixk has X_full")
    require(b["has_X_full"] is False, "V48 has no X_full")
    require(b["has_Y"] is True, "V48 has Y")
    require("not_available" in b["causal_R_from_disk"], "do not invert Y")


def test_stage1_K_and_short_eps() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_k1_kernel import k_shape_on_times, source_beta_nm, k_vector_perp
    from q8_m0_response import predict_delta_mu, test_waveform_sin2_dir
    from q8_m0_s1_kernel import parallel_perp_basis
    from q8_m0_stage1 import stage1_spec

    spec = stage1_spec()
    require("n_munu" in spec["observables"], "n in stage1")
    require("g2" in spec["not_this_round"], "g2 deferred")
    require("farfield" in spec["time_window"] or "Tp" in spec["time_window"], "finite time")
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    rec = choose_probe_k(geo, R0, -0.009010620372198276)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=16, neel=+1, sign=+1, mu=-0.009010620372198276,
        verbose=False, kf_frac=rec["kf"], save_full_x=True, save_mean_D=True,
    )
    ks = k_shape_on_times(r0["times"], prot.omega0, prot.t_pulse)
    require(float(np.max(np.abs(ks["K_shape"] - ks["psi"]))) > 1e-6, "1/sqrt(omega) changes K")
    e_par, _ = parallel_perp_basis(geo)
    h = test_waveform_sin2_dir(r0["times"], prot.t_pulse, e_par)
    require(abs(h[0]).max() < 1e-12, "h(0)=0")
    dt = float(r0["dt"])
    eps = 3e-4
    pred = predict_delta_mu(r0["X_full"], r0["w_k"], eps * h, dt, r0["mean_D_t"])
    require(float(np.max(np.abs(pred["contact"]))) > 0.0, "contact term live")

    def extra(t, e=eps, hu=h, d=dt):
        x = float(t) / d
        i = min(max(int(np.floor(x)), 0), hu.shape[0] - 2)
        f = x - i
        return e * ((1.0 - f) * hu[i] + f * hu[i + 1])

    rp = propagate_cjj(
        geo, R0, prot, nt_per_cycle=16, neel=+1, sign=+1, mu=-0.009010620372198276,
        verbose=False, kf_frac=rec["kf"], a_extra=lambda t: extra(t),
    )
    rm = propagate_cjj(
        geo, R0, prot, nt_per_cycle=16, neel=+1, sign=+1, mu=-0.009010620372198276,
        verbose=False, kf_frac=rec["kf"], a_extra=lambda t: extra(t, e=-eps),
    )
    fd = 0.5 * (np.asarray(rp["mu_per_cell"]) - np.asarray(rm["mu_per_cell"]))
    peak_fd = float(np.max(np.abs(fd)))
    peak_pr = float(np.max(np.abs(pred["total"])))
    require(peak_fd > 0.0 and peak_pr > 0.0, "FD and R^A both fire")
    ratio = peak_pr / peak_fd
    require(0.15 < ratio < 6.0, f"coarse-nt order-of-magnitude ratio={ratio}")
    kv = k_vector_perp(r0["times"], prot.omega0, prot.t_pulse, geo)
    src = source_beta_nm(r0["X_full"], r0["w_k"], r0["mu_per_cell"], kv["K_a"], dt)
    require(src["photon_output"] == "not_run", "source n is not photons")
    require(src["n_source_shape"]["abs"] >= 0.0, "n_source finite")


def test_mode_io_units_shape_and_order() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_mode_io import (
        ALPHA_FS,
        L0_ANG,
        area_scaling_table,
        close_output_moments,
        eta2_ncell,
        f_and_psi,
        f_vs_psi_shape_metrics,
        mode_waveforms,
        term_ledger_spec,
    )
    from q8_m0_stage1 import stage1_spec

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    pref = eta2_ncell(geo)
    expect = ALPHA_FS * (L0_ANG ** 2) / float(geo.a_cell_ang2)
    require(abs(pref - expect) < 1e-18, "eta2 N_cell = alpha_fs / A_uc")
    table = area_scaling_table(geo)
    require(table["connected_n_m"]["S_cancels"] is True, "connected S cancels")
    require(table["coherent_beta"]["S_cancels"] is False, "mean field keeps sqrt(N)")
    require(table["nk_is_not_N_cell"] is True, "nk is not N_cell")
    times = np.linspace(0.0, float(prot.t_pulse), 801)
    rec = f_and_psi(times, prot.omega0, prot.t_pulse)
    met = f_vs_psi_shape_metrics(
        times, rec["F"], rec["psi"], rec["omega_c"], rec["omega"], rec["phi"]
    )
    raw = met["raw_midpoint_|F-psi|/|psi|"]
    require(2.0 < raw < 2.3, f"raw midpoint is the known scale, got {raw}")
    mid = met["scaled_midpoint_|F-psi_s|/|F|"]
    l2 = met["scaled_full_window_L2_|F-psi_s|/||F||"]
    require(mid < 0.002, f"scaled midpoint shape {mid}")
    require(0.01 < l2 < 0.03, f"scaled L2 shape {l2}")
    mw = mode_waveforms(times, prot.omega0, prot.t_pulse, geo)
    require(mw["units"]["K_equals_i_over_hbar_u_star"] is True, "K = i u*/hbar")
    led = term_ledger_spec()
    require("B2" in led["m_out_O_lambda2"], "leading m response is B2")
    require(led["do_not_unit_convert_old_alpha_products_into_V"] is True, "no old alpha")
    rec_k = choose_probe_k(geo, R0, -0.009010620372198276)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=16, neel=+1, sign=+1, mu=-0.009010620372198276,
        verbose=False, kf_frac=rec_k["kf"], save_full_x=True, save_mean_D=True,
    )
    closed = close_output_moments(
        r0["X_full"], r0["w_k"], r0["mu_per_cell"], r0["mean_D_t"],
        r0["times"], float(r0["dt"]), geo, prot,
    )
    require(closed["term_ledger"]["old_|alpha_minus|^2"] == "higher_order_not_used", "n order")
    require(closed["term_ledger"]["old_alpha_plus_alpha_minus"] == "higher_order_not_used", "m order")
    require(closed["term_ledger"]["i_squared_on_anomalous_source"] == -1, "i^2 on m_source")
    require(abs(closed["vacuum_V"] - 0.5) < 1e-15, "vacuum 1/2")
    require(closed["N_cell"] == "unset_not_equal_to_nk", "do not use nk as N_cell")
    require(closed["squeezing_claim"].startswith("not_evaluated"), "no squeezing claim")
    require(closed["beta_still_needs_sqrt_Ncell"] is True, "coherent mean keeps sqrt(N)")
    comm = closed["commutator_check"]["s1_commutator"]
    require(abs(comm["im"]) < 1e-9, "source commutator is real in this X language")
    require("unique_unclosed_item" in closed["commutator_check"], "record the unclosed commutator piece")
    require("kept_order_identity_closed" in closed["commutator_check"], "identity flag present")
    spec = stage1_spec()
    require(spec["born_label"].startswith("O_lambda2_is_Born"), "Born freeze")
    from q8_m0_loop import (
        ResummationNotChosenError,
        apply_resummation_to_output,
        feedback_2x2,
        gram_schmidt_packet,
    )
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True
    require(blocked, "resummation must not write production V")
    from q8_m0_classical import p1_a2_protocol as _p1
    gs = gram_schmidt_packet(_p1(0.8).omega0, _p1(0.8).t_pulse)
    require(abs(gs["overlap_after"]["abs"]) < 1e-10, "H3 packet orthogonal to H2")
    lg = feedback_2x2(0.07 + 0.70j, 0.05 + 0.14j)
    require(lg["complex_linear_1_minus_chi_wrong"] is True, "R is real-linear")
    require(lg["not_a_predicted_commutator"] is True, "partial O4 is not [b,b†]")
    require("approximation_name" in term_ledger_spec(), "Born name on ledger")


def test_gsheet_two_port_and_volterra_adjoint() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_fock_1k import (
        current_ci,
        gs_occupied_orbitals,
        hop_replace,
        onebody_to_ci,
        rdm1_from_ci,
        slater_from_occupied,
        slater_list,
        _index_map,
    )
    from q8_m0_gsheet import (
        gamma_self,
        gamma_vs_eta_table,
        gsheet_spec,
        j_zero_identities,
        ohmic_sheet_transmission,
        prescribed_current_e_self_check,
    )
    from q8_m0_mode_io import eta2_ncell
    from q8_m0_model import current_ops, h8
    from q8_m0_volterra import volterra_adjoint, volterra_forward

    geo = build_geometry()
    spec = gsheet_spec(geo)
    require(spec["not_collection_K"] is True, "G_self is not K")
    require(spec["p1"]["classical_J_does_not_source_G_self"] is True, "P1 local")
    require("L_in,R_in" in spec["ports"], "two vacuum ports")
    tab = gamma_vs_eta_table(geo)
    require(abs(tab["gamma_over_eta2_Ncell"] - 2.0 * np.pi) < 1e-12, "gamma = 2 pi eta2")
    require(abs(gamma_self(geo) - 2.0 * np.pi * eta2_ncell(geo)) < 1e-18, "gamma formula")
    ohm = ohmic_sheet_transmission(0.4)
    require(ohm["pass_T_eq_1_plus_R"], "static sheet T=1+R")
    rng = np.random.default_rng(0)
    aL = rng.normal(size=(11, 2))
    aR = rng.normal(size=(11, 2))
    ident = j_zero_identities(aL, aR, 0.01, gamma_self(geo))
    require(ident["pass"], "J=0 two-port identities")
    i_cell = rng.normal(size=(11, 2))
    pch = prescribed_current_e_self_check(i_cell, 0.01, gamma_self(geo))
    require(pch["pass"], "E_self = -gamma I")

    nt, nk = 7, 2
    x = rng.normal(size=(nt, nk, 2, 4, 4)) + 1j * rng.normal(size=(nt, nk, 2, 4, 4))
    w = np.full(nk, 0.5)
    md = rng.normal(size=(nt, 2, 2))
    md = 0.5 * (md + np.transpose(md, (0, 2, 1)))
    dt = 0.03
    gamma = 0.05
    xi = rng.normal(size=(nt, 2))
    a_in = rng.normal(size=(nt, 2))
    fw = volterra_forward(x, w, md, dt, gamma, a_L_in=a_in, xi=xi)
    kvec = rng.normal(size=(nt, 2))
    g_I = kvec
    L = float(np.sum(g_I * fw["I"]))
    adj = volterra_adjoint(x, w, md, dt, gamma, g_I)
    L_xi = float(np.sum(adj["g_xi"] * xi))
    L_a = float(np.sum(adj["g_Ain"] * a_in))
    require(abs(L - L_xi - L_a) / max(abs(L), 1e-30) < 1e-8, f"adjoint identity {L} vs {L_xi+L_a}")
    fw0 = volterra_forward(x, w, md, dt, 0.0, a_L_in=a_in)
    from q8_m0_response import predict_delta_mu
    born = predict_delta_mu(x, w, a_in, dt, md)
    rel = float(np.max(np.abs(fw0["I"] - born["total"])) / max(float(np.max(np.abs(born["total"]))), 1e-30))
    require(rel < 1e-12, f"gamma=0 recovers R^A {rel}")

    dets = slater_list()
    idx = _index_map(dets)
    require(len(dets) == 70, "C(8,4)=70")
    h0 = h8(geo, R0, geo.k_cart([0.0, 0.0]), +1)
    vv = gs_occupied_orbitals(h0)
    psi = slater_from_occupied(vv, dets)
    rdm = rdm1_from_ci(psi, dets, idx)
    rdm_sp = vv @ vv.conj().T
    err_rdm = float(np.max(np.abs(rdm - rdm_sp)))
    require(err_rdm < 1e-9, f"CI 1-RDM vs Slater {err_rdm}")
    j1 = current_ops(geo, R0, geo.k_cart([0.0, 0.0]), +1)
    i_ci = current_ci(j1, psi, dets, idx)
    i_sp = np.real(np.einsum("iv,aij,jv->a", vv.conj(), j1, vv))
    require(float(np.max(np.abs(i_ci - i_sp))) < 1e-8, "CI current vs Slater")
    h70 = onebody_to_ci(h0, dets, idx)
    e_ci = float(np.real(np.vdot(psi, h70 @ psi)))
    e_sp = float(np.real(np.einsum("iv,ij,jv->", vv.conj(), h0, vv)))
    require(abs(e_ci - e_sp) / max(abs(e_sp), 1e-30) < 1e-8, f"CI energy {e_ci} vs {e_sp}")
    new, sign = hop_replace((0, 1, 2, 3), 1, 5)
    require(new == (0, 2, 3, 5), f"hop dest {new}")
    require(abs(sign) == 1, "hop sign unit")
    from q8_m0_loop import apply_resummation_to_output, ResummationNotChosenError
    blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        blocked = True
    require(blocked, "production V still not resummed")
    prot = p1_a2_protocol(0.8)
    require(prot.polarization == "a2", "P1 a2 frozen")


def test_runaway_energy_and_seed_self() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_gsheet import gamma_self
    from q8_m0_runaway import (
        amplitude_compare,
        band_power,
        band_power_unweighted_legacy,
        e_interval_forward,
        kappa_p1_series,
        left_riemann_work,
        propagate_seed_self,
        split_work,
        split_work_interval,
    )

    geo = build_geometry()
    dt = 0.02
    t = np.arange(40) * dt
    a_in = np.stack([np.sin(0.3 * t), np.zeros_like(t)], axis=1)
    e_in = e_interval_forward(a_in, dt)
    i_abs = e_in.copy()
    w_abs = left_riemann_work(e_in, i_abs, dt)
    require(w_abs["W"] > 0.0, f"passive W should be >0, got {w_abs['W']}")
    require(w_abs["W_sign"] == 1, "numeric sign +")
    i_neg = -e_in
    w_neg = left_riemann_work(e_in, i_neg, dt)
    require(w_neg["W"] < 0.0, f"opposite W should be <0, got {w_neg['W']}")
    gamma = 0.05
    i = np.stack([np.sin(0.4 * t), 0.1 * np.cos(0.2 * t)], axis=1)
    wrk = split_work_interval(np.zeros_like(i), i, dt, gamma)
    require(abs(wrk["W_residual"]) < 1e-12, f"interval residual {wrk['W_residual']}")
    require(abs(wrk["W_self_plus_gamma_I2"]) < 1e-12, "W_self = -gamma left-sum I^2")
    require(abs(wrk["Phi_plus_W_el"]) < 1e-12, f"port identity {wrk['Phi_plus_W_el']}")
    legacy = split_work(np.zeros_like(i), i, dt, gamma)
    require(abs(legacy["W_residual"]) > 1e-8, "legacy mixed-time residual should be visible")
    bp = band_power(i, dt, 0.4)
    require(abs(sum([
        bp["frac_low_below_0.25_omega0"],
        bp["frac_0.25_to_1.5_omega0"],
        bp["frac_H2_1.5_to_2.5_omega0"],
        bp["frac_above_2.5_omega0"],
    ]) - 1.0) < 1e-10, "band fractions sum to 1")
    require(bp["frac_strict_dc_bin"] <= bp["frac_low_below_0.25_omega0"] + 1e-12, "DC bin inside low band")
    old_bp = band_power_unweighted_legacy(i, dt, 0.4)
    require("frac_low_below_0.25_omega0" in old_bp, "legacy bands")
    prot = p1_a2_protocol(0.8)
    kap = kappa_p1_series(np.linspace(0.0, prot.t_pulse, 200), geo, prot)
    amp = amplitude_compare(np.zeros_like(kap), kap, geo, 0.8)
    require(amp["kappa_p1_peak_mod_ang_inv"] > 0.0, "P1 kappa peak")
    require(abs(amp["Astar_p1_protocol"] - 0.8) < 1e-15, "A* label")
    rec = choose_probe_k(geo, R0, -0.009010620372198276)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=12, neel=+1, sign=+1, mu=-0.009010620372198276,
        verbose=False, kf_frac=rec["kf"][:2],
    )
    seed = np.zeros((r0["times"].shape[0], 2), dtype=float)
    run = propagate_seed_self(
        geo, R0, prot, kf=rec["kf"][:2], mu=-0.009010620372198276, neel=+1,
        nt_per_cycle=12, seed=seed, i_ref=r0["mu_per_cell"], gamma=gamma_self(geo),
        verbose=False,
    )
    rel = float(np.max(np.abs(run["dI"])) / max(float(np.max(np.abs(r0["mu_per_cell"]))), 1e-30))
    require(rel < 0.05, f"zero seed should track P1, rel dI={rel}")
    require("U" in run and "n_prop_halt" in run, "energy and prop halt recorded")


def test_adapt_perk_integrand_and_tiles() -> None:
    from q8_m0_adapt_bz import (
        pick_tiles,
        start_leaves,
        tile_features,
        traditional_score,
        perk_born_ab,
        split_leaves,
    )
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_mode_io import eta2_ncell, response_to_complex_scalar
    from q8_m0_response import contact_dmu_perk, prefix_state_dmu, prefix_state_dmu_perk
    from q8_m0_stage1g_bzqual import h2_seed
    from q8_m0_volterra import collect_overlap

    geo = build_geometry()
    leaves = start_leaves(2)
    require(len(leaves) == 16, "4x4 start")
    require(abs(sum(t.area for t in leaves) - 1.0) < 1e-15, "tile areas fill BZ")
    p0 = leaves[0]
    ch = p0.children()
    require(len(ch) == 4, "quad split")
    require(abs(sum(c.area for c in ch) - p0.area) < 1e-15, "child area")
    nxt = split_leaves(leaves, [p0])
    require(len(nxt) == 19, "replace 1 parent by 4 children")
    scores = np.arange(16, dtype=float)
    scores[3] = 100.0
    scores[7] = 90.0
    picked = pick_tiles(leaves, scores, 2, 5)
    require(picked[0] is leaves[3] and picked[1] is leaves[7], "deterministic top-2")

    prot = p1_a2_protocol(0.8)
    mu = -0.009010620372198276
    rec = choose_probe_k(geo, R0, mu)
    kf = rec["kf"][:3]
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=12, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, save_full_x=True, save_perk_D=True,
        w_k=np.array([0.5, 0.3, 0.2]),
    )
    require(abs(float(np.sum(r0["w_k"])) - 1.0) < 1e-15, "custom w sums to 1")
    x = r0["X_full"]
    w = r0["w_k"]
    dt = float(r0["dt"])
    h = np.zeros((x.shape[0], 2), dtype=float)
    h[:, 0] = 1.0
    perk = prefix_state_dmu_perk(x, h, dt)
    tot = prefix_state_dmu(x, w, h, dt)
    require(np.max(np.abs(np.einsum("k,tka->ta", w, perk) - tot)) < 1e-12, "perk prefix weights")
    cperk = contact_dmu_perk(r0["perk_D_t"], h)
    mean_c = -np.einsum("tab,tb->ta", r0["mean_D_t"], h)
    require(np.max(np.abs(np.einsum("k,tka->ta", w, cperk) - mean_c)) < 1e-12, "perk D contact")

    a_in, hk = h2_seed(r0["times"], prot, geo, 3.0e-5)
    pref = eta2_ncell(geo)
    a2, b2 = perk_born_ab(x, r0["perk_D_t"], hk["F"], hk["e_perp"], hk["k_a"], dt, pref)
    resp_fs = response_to_complex_scalar(x, w, np.conj(hk["F"]), hk["e_perp"], dt, r0["mean_D_t"])
    resp_f = response_to_complex_scalar(x, w, hk["F"], hk["e_perp"], dt, r0["mean_D_t"])
    a2_mean = 1j * pref * collect_overlap(hk["k_a"], resp_fs["total"], dt)
    b2_mean = 1j * pref * collect_overlap(hk["k_a"], resp_f["total"], dt)
    require(abs(complex(np.dot(w, a2)) - a2_mean) < 1e-10, "perk a2 recovers mean-D A2")
    require(abs(complex(np.dot(w, b2)) - b2_mean) < 1e-10, "perk b2 recovers mean-D B2")
    feat = tile_features(nxt[0], nxt)
    require(traditional_score(feat) >= 0.0, "score finite")
    from q8_m0_adapt_bz import left_rect_integral, traditional_score_traj, traj_l2
    from q8_m0_gsheet import gamma_self
    from q8_m0_runaway import propagate_seed_self

    sig = np.ones((5, 2, 2), dtype=float)
    integ = left_rect_integral(sig, 0.1, 2.0)
    require(np.max(np.abs(integ[0])) < 1e-15, "left-rect starts at 0")
    require(np.max(np.abs(integ[1] - 0.2)) < 1e-15, "first left-rect step")
    require(np.max(np.abs(integ[4] - 0.8)) < 1e-15, "last left-rect step")
    require(traditional_score_traj(feat) >= 0.0, "traj score finite without kappa traj")
    r_nl = propagate_seed_self(
        geo, R0, prot, kf=kf, mu=mu, neel=+1, nt_per_cycle=12,
        seed=np.zeros((r0["times"].shape[0], 2)), i_ref=r0["mu_per_cell"],
        gamma=gamma_self(geo), verbose=False, w_k=w, save_perk_I=True,
    )
    dI_k = r_nl["I_perk"] - r0["perk_mu_t"]
    kap_k = left_rect_integral(dI_k, dt, float(r_nl["gamma"]))
    recon = np.einsum("k,tka->ta", w, kap_k)
    require(np.max(np.abs(recon - r_nl["A_self"])) < 1e-12, "sum_k w kappa_k recovers A_self")
    require(traj_l2(r_nl["A_self"], dt) >= 0.0, "traj L2 finite")
    del a_in


def test_feedback_op_adjoint_and_scale() -> None:
    from q8_m0_feedback_op import (
        adjoint_residual,
        apply_G,
        apply_Gadj,
        apply_L,
        apply_L_perk,
        apply_R,
        apply_Radj,
        classify_sigma,
        inner_dt,
        leading_singular,
        scale_residual,
    )

    rng = np.random.default_rng(2)
    nt, nk = 9, 2
    dt = 0.04
    gamma = 0.07
    x = rng.normal(size=(nt, nk, 2, 4, 4)) + 1j * rng.normal(size=(nt, nk, 2, 4, 4))
    w = np.array([0.4, 0.6])
    d_perk = rng.normal(size=(nt, nk, 2, 2))
    d_perk = 0.5 * (d_perk + np.transpose(d_perk, (0, 1, 3, 2)))
    mean_d = np.einsum("k,tkab->tab", w, d_perk)
    f = rng.normal(size=(nt, 2))
    g = rng.normal(size=(nt, 2))
    i_t = rng.normal(size=(nt, 2))

    gf = apply_G(i_t, dt, gamma)
    left_g = inner_dt(g, gf, dt)
    right_g = inner_dt(apply_Gadj(g, dt, gamma), i_t, dt)
    require(abs(left_g - right_g) / max(abs(left_g), 1e-30) < 1e-10, "G adjoint")

    rf = apply_R(x, w, mean_d, f, dt)
    left_r = inner_dt(g, rf, dt)
    right_r = inner_dt(apply_Radj(x, w, mean_d, g, dt), f, dt)
    require(abs(left_r - right_r) / max(abs(left_r), 1e-30) < 1e-8, f"R adjoint {left_r} vs {right_r}")

    require(adjoint_residual(x, w, mean_d, f, g, dt, gamma) < 1e-8, "L adjoint")
    require(scale_residual(x, w, mean_d, f, dt, gamma, 3.0) < 1e-12, "L scale")

    lk = apply_L_perk(x, d_perk, f, dt, gamma)
    recov = np.einsum("k,tka->ta", w, lk)
    lt = apply_L(x, w, mean_d, f, dt, gamma)
    require(float(np.max(np.abs(recov - lt))) < 1e-12, "sum_k w L_k recovers L")

    op = leading_singular(x, w, mean_d, dt, gamma, f_probe=f, p=3, n_iter=4, rng=0)
    require(op["sigma_max"] + 1e-8 >= op["gain_h2_dir"], "sigma_max >= H2 directional gain")
    op2 = leading_singular(x, w, mean_d, dt, gamma, f_probe=7.0 * f, p=3, n_iter=4, rng=0)
    require(abs(op2["sigma_max"] - op["sigma_max"]) / max(op["sigma_max"], 1e-30) < 1e-8, "sigma independent of probe scale")
    require(classify_sigma(0.02) == "ll1_born_has_small_control_parameter", "class ll1")
    require(classify_sigma(0.2) == "around_0p2_few_percent_second_pass", "class 0.2")
    require(classify_sigma(1.2) == "gtrsim_1_hidden_strong_direction", "class >=1")


def test_photon_op_maps_and_adjoint() -> None:
    from q8_m0_photon_op import (
        adjoint_residual_ph,
        analyze_adj,
        analyze_phi,
        apply_L_ph,
        inner_phi,
        leading_singular_ph,
        norm_phi,
        omega_weights,
        photon_omega_grid,
        qs_reconstruction,
        synthesize_A,
        synthesize_adj,
    )
    from q8_m0_s1_kernel import parallel_perp_basis

    geo = build_geometry()
    _e_par, e_perp = parallel_perp_basis(geo)
    tp = 8.0 * (2.0 * np.pi / 0.05)
    omega0 = 0.05
    omega = photon_omega_grid(tp, omega0, harm_max=8.0)
    require(float(omega[0]) > 0.0, "omega=0 excluded")
    dw = omega_weights(omega)
    nt = 257
    dt = tp / (nt - 1)
    times = np.arange(nt) * dt
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(omega.size, 2)) + 1j * rng.normal(size=(omega.size, 2))
    a = rng.normal(size=(nt, 2))
    sa = synthesize_A(phi, omega, times, dw)
    left = float(dt) * float(np.sum(a * sa))
    right = inner_phi(synthesize_adj(a, omega, times, dt), phi, dw)
    require(abs(left - right) / max(abs(left), 1e-30) < 1e-9, f"S adjoint {left} vs {right}")
    qa = analyze_phi(a, omega, times, dt)
    left_q = inner_phi(phi, qa, dw)
    right_q = float(dt) * float(np.sum(analyze_adj(phi, omega, times, dw) * a))
    require(abs(left_q - right_q) / max(abs(left_q), 1e-30) < 1e-9, f"Q adjoint {left_q} vs {right_q}")

    from q8_m0_photon_op import h2_phi
    ph2 = h2_phi(omega, omega0, tp, e_perp)
    rec = qs_reconstruction(ph2, omega, times, dt, dw)
    require(rec["overlap"] > 0.5, f"Q S H2 overlap too small {rec}")

    nk = 2
    x = rng.normal(size=(nt, nk, 2, 4, 4)) + 1j * rng.normal(size=(nt, nk, 2, 4, 4))
    w = np.array([0.4, 0.6])
    md = rng.normal(size=(nt, 2, 2))
    md = 0.5 * (md + np.transpose(md, (0, 2, 1)))
    gamma = 0.07
    psi = rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
    require(
        adjoint_residual_ph(x, w, md, phi, psi, omega, times, dt, dw, gamma) < 1e-7,
        "L_ph adjoint",
    )
    lf = apply_L_ph(x, w, md, phi, omega, times, dt, dw, gamma)
    lf2 = apply_L_ph(x, w, md, 3.0 * phi, omega, times, dt, dw, gamma)
    require(float(np.max(np.abs(lf2 - 3.0 * lf))) / max(float(np.max(np.abs(lf))), 1e-30) < 1e-11, "L_ph scale")
    op = leading_singular_ph(
        x, w, md, omega, times, dt, dw, gamma, phi_probe=ph2, p=2, n_iter=3, rng=0,
    )
    require(op["sigma_max"] + 1e-8 >= op["gain_h2_dir"], "sigma_ph >= H2 photon gain")
    require(norm_phi(op["v_right"], dw) > 0.0, "right vector has mass")


def test_canonical_io_audit() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_io_audit import (
        F_from_phi,
        basis_gram,
        c_el_field,
        empty_matter,
        empty_sheet_identity,
        h2_st_from_self,
        inner_c,
        packet_basis,
        st_from_z,
        z_of,
    )
    from q8_m0_photon_op import (
        analyze_phi,
        h2_phi,
        omega_weights,
        photon_omega_grid,
        qs_reconstruction,
        synthesize_A,
    )
    from q8_m0_resummed_io import vacuum_sheet_rms
    from q8_m0_s1_kernel import parallel_perp_basis
    from q8_m0_stage1 import stage1_spec

    spec = stage1_spec()
    require(spec["canonical_io_gate"].startswith("SS_dagger"), "canonical gate")
    require(spec["vacuum_rms_not_a_linear_failure_gate"] is True, "RMS demoted")
    require(spec["fork_nonlinear_vs_dissipation"].startswith("deferred"), "fork deferred")
    require(spec["T2_environment"].startswith("not_added"), "no T2")
    require(spec["nonlinear_quantum_feedback"] == "not_started", "no nonlinear")

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    e_par, e_perp = parallel_perp_basis(geo)
    omega = photon_omega_grid(prot.t_pulse, prot.omega0, harm_max=8.0)
    dw = omega_weights(omega)
    nt = 257
    dt = prot.t_pulse / (nt - 1)
    times = np.arange(nt) * dt
    phis, labs = packet_basis(omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, include_par=False)
    require(labs[0] == "H2_perp", f"H2 first, got {labs[0]}")
    gram = basis_gram(phis[:8], dw)
    require(gram["orthonormal"], f"gram {gram}")

    ph2 = h2_phi(omega, prot.omega0, prot.t_pulse, e_perp)
    qs = qs_reconstruction(ph2, omega, times, dt, dw)
    require(qs["overlap"] > 0.5, f"QS {qs}")
    F = F_from_phi(ph2, omega, times, dw)
    require(float(np.max(np.abs(np.real(F) - synthesize_A(ph2, omega, times, dw)))) < 1e-12, "Re F = Sφ")
    z_re = z_of(ph2, np.real(F), omega, times, dt, dw)
    z_im = z_of(ph2, np.imag(F), omega, times, dt, dw)
    s, t = st_from_z(z_re, z_im)
    require(abs(t) < 0.25, f"empty T {t}")
    require(abs(s) > 0.55, f"empty S {s} qs={qs}")

    empty = empty_sheet_identity(
        times, dt, omega, dw, prot.omega0, prot.t_pulse, e_perp, e_par, (1, 2), gamma=0.05,
    )
    c1 = empty["scan"]["rows"][0]["commutator_real"]
    c2 = empty["scan"]["rows"][1]["commutator_real"]
    require(abs(c1 - 1.0) < 0.45, f"empty n=1 comm {c1}")
    require(abs(c2 - 1.0) < 0.45, f"empty n=2 comm {c2}")
    require(empty["scan"]["C_el"]["C_el_real"] == 0.0, "empty C_el")

    x, w, md = empty_matter(nt)
    rng = np.random.default_rng(4)
    x_rand = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    cel0 = c_el_field(x_rand, w, md, dt, 0.0, ph2, omega, times, dw)
    require(abs(cel0["C_el"]) < 1e-14, f"gamma0 C_el {cel0['C_el']}")

    zeros = np.zeros((nt, 2), dtype=float)
    row = h2_st_from_self(ph2, F, zeros, zeros, omega, times, dt, dw)
    require(abs(row["T_RL"]) < 0.25, f"T_RL {row['T_RL']}")
    require(abs(row["S_RR"]) < 0.15, f"no feedthrough RR {row['S_RR']}")
    require(abs(inner_c(ph2, analyze_phi(np.real(F), omega, times, dt), dw) - z_re) < 1e-12, "z")

    dummy = {
        "X_full": x, "w_k": w, "mean_D_t": md,
        "times": times, "dt": dt,
    }
    ch = {"e_perp": e_perp, "F": F @ e_perp}
    rms = vacuum_sheet_rms(dummy, ch, 0.0, 1e-4, 0.2)
    require(rms["kind"].startswith("provisional_self_consistency"), "RMS kind")
    require(rms["linear_selfcons_fail"] is False, "RMS must not fail-gate")
    require(rms["not_a_pass_fail_gate"] is True, "RMS not a gate")
    require(spec["missing_orthogonal_modes"].startswith("excluded"), "modes excluded")
    require(spec["required_noise_matrix"].startswith("diagnostic"), "ΔC not a patch")
    require(spec["rpa_joint_linear"].startswith("approved_as_P1"), "phgauss approved")


def test_canonical_order_audit() -> None:
    from q8_m0_io_audit import empty_matter, packet_basis
    from q8_m0_order_audit import (
        c_el_born,
        h2_identity_at_gamma,
        leading_power,
        nambu_deficit,
    )
    from q8_m0_photon_op import h2_phi, omega_weights, photon_omega_grid
    from q8_m0_s1_kernel import parallel_perp_basis

    geo = build_geometry()
    e_par, e_perp = parallel_perp_basis(geo)
    tp = 8.0 * (2.0 * np.pi / 0.05)
    omega0 = 0.05
    omega = photon_omega_grid(tp, omega0, harm_max=8.0)
    dw = omega_weights(omega)
    nt = 129
    dt = tp / (nt - 1)
    times = np.arange(nt) * dt
    x, w, md = empty_matter(nt)
    phis, _labs = packet_basis(omega, dw, omega0, tp, e_perp, e_par, include_par=False)
    ph2 = h2_phi(omega, omega0, tp, e_perp)
    z0 = h2_identity_at_gamma(
        x, w, md, times, dt, 0.0, omega, dw, phis[:1], ph2, kind="resummed",
    )
    require(abs(z0["delta_can"]) < 1e-8, f"λ=0 Δ {z0['delta_can']}")
    require(abs(c_el_born(x, w, md, dt, 0.05, ph2, omega, times, dw)) < 1e-14, "born C_el empty")
    z1 = h2_identity_at_gamma(
        x, w, md, times, dt, 0.05, omega, dw, phis[:1], ph2, kind="born",
    )
    require(abs(z1["delta_can"]) < 1e-8, f"empty Born Δ {z1['delta_can']}")

    lam = np.array([0.05, 0.1, 0.2])
    fit = leading_power(lam, 3.0 * lam ** 4, lmax=0.3)
    require(abs(fit["p"] - 4.0) < 0.05, f"fit p {fit}")
    dfn = nambu_deficit(np.eye(2), np.zeros((2, 2)), np.zeros((2, 2)))
    require(dfn["fro_delta"] < 1e-15, "identity deficit")
    require(dfn["note"].startswith("diagnostic"), "not a patch")


def test_phgauss_canonical() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_io_audit import audit_grid
    from q8_m0_mode_io import eta2_ncell
    from q8_m0_phgauss import (
        PHGAUSS_ID,
        born_channel_from_ph,
        c_greater_match,
        h2_two_port_packets,
        io_from_phi,
        propagate_phi,
        response_from_ph_prefix,
        symplectic_J,
    )
    from q8_m0_response import predict_delta_mu
    from q8_m0_s1_kernel import parallel_perp_basis

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    mu = -0.009010620372198276
    kf = np.array([[0.0, 0.0], [0.5, 0.0]])
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=8, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    x, w, md = r0["X_full"], r0["w_k"], r0["mean_D_t"]
    dt = float(r0["dt"])
    times = r0["times"]
    pref = eta2_ncell(geo)
    eta = float(np.sqrt(pref))
    cg = c_greater_match(x, w)
    require(cg["maxabs"] < 1e-14, f"C> {cg}")
    e_par, e_perp = parallel_perp_basis(geo)
    h = np.zeros((x.shape[0], 2))
    h[:, 0] = np.sin(np.pi * np.arange(x.shape[0]) / max(x.shape[0] - 1, 1)) ** 2
    ph = response_from_ph_prefix(x, w, h, dt, md)
    ra = predict_delta_mu(x, w, h, dt, md)
    require(np.max(np.abs(ph["total"] - ra["total"])) < 1e-10, "R^A recovered")
    born = born_channel_from_ph(x, w, md, dt, geo, prot, pref)
    require(born["maxabs_I_ph_minus_RA"] < 1e-10, f"A2 current {born}")
    require(born["kubo_ph_minus_RA"] < 1e-12, f"Kubo ph vs RA {born}")
    require(born["kept_order_closed"], f"Kubo match {born['kubo_residual_abs']}")
    omega, dw = audit_grid(prot, harm_max=8.0)
    flist, _labs = h2_two_port_packets(times, prot.omega0, prot.t_pulse, e_perp, dw, omega)
    rec = propagate_phi(x, w, md, flist, dt, eta)
    require(rec["symplectic_maxabs"] < 1e-9, f"Φ^T J Φ {rec['symplectic_maxabs']}")
    require(rec["hamiltonian_K_maxabs"] < 1e-9, f"K Hamiltonian {rec['hamiltonian_K_maxabs']}")
    require(rec["no_G_self"] is True, "no G_self input")
    io = io_from_phi(rec["Phi"], rec["n_ph"], rec["n_B"])
    require(io["ident_minus_I_maxabs"] < 1e-8, f"I/O identity {io['ident_minus_I_maxabs']}")
    j = symplectic_J(rec["n"])
    require(float(np.max(np.abs(j + j.T))) < 1e-15, "J antisymmetric")
    require(PHGAUSS_ID.startswith("P1_dressed"), "id")


def test_phgauss_pauli() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_io_audit import audit_grid
    from q8_m0_mode_io import eta2_ncell
    from q8_m0_phgauss import (
        PAULI_NORM,
        h2_two_port_packets,
        pauli_area_curve,
        pauli_from_phi,
        propagate_phi,
        vacuum_moment_matrix,
    )
    from q8_m0_s1_kernel import parallel_perp_basis
    from q8_m0_stage1 import stage1_spec

    require(PAULI_NORM["never_N_cell_eq_nk"] is True, "nk is not N_cell")
    require(PAULI_NORM["N_cell"] == "unset_not_equal_to_nk", "N_cell unset")
    require(PAULI_NORM["raw_N_B_is_not_a_pass_fail"] is True, "N_B not a gate")
    n0 = vacuum_moment_matrix(3)
    require(n0.shape == (6, 6), "vacuum shape")
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    mu = -0.009010620372198276
    kf = np.array([[0.0, 0.0], [0.5, 0.0]])
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=8, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    x, w, md = r0["X_full"], r0["w_k"], r0["mean_D_t"]
    dt = float(r0["dt"])
    eta = float(np.sqrt(eta2_ncell(geo)))
    ev0 = propagate_phi(x, w, md, [], dt, eta, save_pauli=True)
    p0 = ev0["pauli_final"]
    require(p0["N_cell"] == "unset_not_equal_to_nk", "N_cell stays unset")
    require(p0["N_B_max"] < 1e-12, f"empty N_B {p0['N_B_max']}")
    require(p0["Ncell_times_eps_Pauli"] < 1e-12, f"empty eps {p0['Ncell_times_eps_Pauli']}")
    require(p0["Ncell_times_C_minus_I"] < 1e-12, f"empty C-I {p0['Ncell_times_C_minus_I']}")
    require(abs(float(np.max(ev0['N_B_max_t']))) < 1e-12, "empty series")
    p_vac = pauli_from_phi(np.eye(2 * ev0["n"]), ev0["n_ph"], ev0["n_B"], w)
    require(p_vac["Ncell_times_eps_Pauli"] < 1e-14, "t0 vacuum")
    e_par, e_perp = parallel_perp_basis(geo)
    omega, dw = audit_grid(prot, harm_max=8.0)
    flist, _ = h2_two_port_packets(r0["times"], prot.omega0, prot.t_pulse, e_perp, dw, omega)
    ev = propagate_phi(x, w, md, flist, dt, eta, save_pauli=True)
    pf = ev["pauli_final"]
    expect = float(np.max(np.asarray(pf["N_B"]) / np.asarray(w)[:, None, None]))
    require(
        abs(pf["Ncell_times_f_micro_max"] - expect) < 1e-12,
        f"Ncell f_micro {pf['Ncell_times_f_micro_max']} vs {expect}",
    )
    require(pf["norm"]["N_cell_cancels"] is False, "S node")
    require(pf["norm"]["S_needed_for_this_gate"] is True, "S needed")
    require(pf["norm"]["S_is_not_a_normalization_box"] is True, "not a box")
    require(stage1_spec()["N_cell"] == "unset_not_equal_to_nk", "spec N_cell")
    require(stage1_spec()["S"].startswith("not_fixed"), "S not fixed")
    curve = pauli_area_curve(1.0, c_p=2.0, c_c=4.0)
    require(abs(curve["N_cell_min"]["eps_1pct"] - 200.0) < 1e-12, "area curve")


def test_gself_emergence() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_gself_emerge import (
        emerge_A_from_I,
        emerge_on_basis,
        eta_split,
        long_window_bins,
    )
    from q8_m0_s1_kernel import parallel_perp_basis

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    e_par, e_perp = parallel_perp_basis(geo)
    sp1 = eta_split(geo, 1.0)
    sp4 = eta_split(geo, 4.0)
    require(abs(sp1["eta_vac"] * 2.0 - sp4["eta_drive"]) < 1e-12, "vac vs drive")
    require(abs(sp1["eta_vac"] * sp1["eta_drive"] - sp4["eta_vac"] * sp4["eta_drive"]) < 1e-14, "product")
    pack1 = long_window_bins(prot, e_perp, t_obs_over_tp=1.0, ntpc=16, harm_max=4.0)
    pack2 = long_window_bins(prot, e_perp, t_obs_over_tp=2.0, ntpc=16, harm_max=4.0)
    z = emerge_A_from_I([], pack2["i_cell"], pack2["dt"], 1.0, 1.0)
    require(float(np.max(np.abs(z))) == 0.0, "empty photons")
    r1 = emerge_on_basis(
        pack1["phis"], pack1["omega"], pack1["times"], pack1["dw"],
        pack1["i_cell"], pack1["dt"], geo, 1.0, n_pulse=pack1["n_pulse"],
    )
    r2 = emerge_on_basis(
        pack2["phis"], pack2["omega"], pack2["times"], pack2["dw"],
        pack2["i_cell"], pack2["dt"], geo, 1.0, n_pulse=pack2["n_pulse"],
    )
    r2b = emerge_on_basis(
        pack2["phis"], pack2["omega"], pack2["times"], pack2["dw"],
        pack2["i_cell"], pack2["dt"], geo, 16.0, n_pulse=pack2["n_pulse"],
    )
    require(r1["compare"]["A0"] < 1e-15, "A(t0)=0")
    require(r2["compare"]["rel_maxabs_A"] < r1["compare"]["rel_maxabs_A"] - 0.05, "IR improves")
    require(abs(r2["compare"]["rel_maxabs_A"] - r2b["compare"]["rel_maxabs_A"]) < 1e-12, "Ncell kernel")
    require(r1["no_hand_G_self"] is True, "no hand G_self")


def test_resummed_gamma0_recovers_born() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_gsheet import gamma_self
    from q8_m0_mode_io import eta2_ncell
    from q8_m0_resummed_io import h2_channel
    from q8_m0_stage1 import stage1_spec
    from q8_m0_volterra import gamma0_recovers_born, h2_kernels

    spec = stage1_spec()
    require(spec["theory_main"].startswith("P1_dressed"), "phgauss main")
    require(spec["expectation_resolvent"].startswith("RA_xi"), "resolvent diagnostic")
    require(spec["born_label"].startswith("O_lambda2_is_Born"), "Born still named")
    require(spec["production_V_blocked"] is True, "production V blocked")

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    mu = -0.009010620372198276
    kf = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=8, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True, save_perk_D=True,
    )
    packed = {
        "X_full": r0["X_full"], "w_k": r0["w_k"], "mean_D_t": r0["mean_D_t"],
        "perk_D_t": r0["perk_D_t"], "times": r0["times"], "dt": float(r0["dt"]),
    }
    pref = eta2_ncell(geo)
    hk = h2_kernels(r0["times"], prot.omega0, prot.t_pulse, geo)
    g0 = gamma0_recovers_born(
        r0["X_full"], r0["w_k"], r0["mean_D_t"], float(r0["dt"]),
        hk["F"], hk["e_perp"], hk["k_a"], pref,
    )
    require(g0["pass"], "gamma=0 Volterra recovers Born current")
    ch = h2_channel(packed, geo, prot, 0.0, pref)
    require(abs(ch["A2_res"]["abs"] - ch["A2_born"]["abs"]) < 1e-8, "gamma=0 A2_res is Born")
    require(ch["recon_res_minus_perk"] < 1e-8, "perk dressed recovers mean A2_res")
    require(gamma_self(geo) > 0.0, "physical gamma positive")


def test_k_cart_k_frac_inverse() -> None:
    geo = build_geometry()
    kf = np.array([[0.0, 0.0], [0.25, 0.5], [0.9, 0.05], [0.07, 0.03]])
    back = geo.k_frac(geo.k_cart(kf))
    require(float(np.max(np.abs(back - kf))) < 1e-12, "k_cart/k_frac inverse")


def test_covariant_permutation_and_hold() -> None:
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_gauge_shift import run_permutation_identity, shift_kf

    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    mu = -0.009010620372198276
    kf = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    kf_p = shift_kf(geo, kf, delta_frac=np.array([0.5, 0.0]))
    require(set(map(tuple, np.round(np.mod(kf_p, 1.0), 8))) == set(map(tuple, np.round(kf, 8))), "perm set")
    perm = run_permutation_identity(geo, R0, prot, 2, mu=mu, neel=+1, ntpc=8)
    require(perm["compare"]["rel_l2_plus"] < 1e-10, f"perm J {perm['compare']}")

    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=8, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=kf[:2],
    )
    r1 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=8, neel=+1, sign=+1, mu=mu,
        verbose=False, kf_frac=kf[:2], n_hold_cycles=1,
    )
    n0 = r0["times"].size
    require(r1["times"].size == n0 + 8, f"hold length {r1['times'].size} vs {n0}")
    require(float(np.max(np.abs(r1["mu_per_cell"][:n0] - r0["mu_per_cell"]))) < 1e-12, "hold matches pulse")
    require(int(r1["n_hold_cycles"]) == 1, "n_hold recorded")


def test_h2_resolvent_identities() -> None:
    from q8_m0_feedback_op import apply_L
    from q8_m0_h2_resolvent import apply_Ln, volterra_matches_resolvent

    rng = np.random.default_rng(5)
    nt, nk = 11, 2
    dt = 0.03
    gamma = 0.05
    x = rng.normal(size=(nt, nk, 2, 4, 4)) + 1j * rng.normal(size=(nt, nk, 2, 4, 4))
    w = np.array([0.35, 0.65])
    md = rng.normal(size=(nt, 2, 2))
    md = 0.5 * (md + np.transpose(md, (0, 2, 1)))
    f = rng.normal(size=(nt, 2))
    l2 = apply_L(x, w, md, apply_L(x, w, md, f, dt, gamma), dt, gamma)
    require(float(np.max(np.abs(l2 - apply_Ln(x, w, md, f, dt, gamma, 2)))) < 1e-12, "L^2 vs two apply_L")
    ident = volterra_matches_resolvent(x, w, md, f, dt, gamma)
    require(ident["rel_l2"] < 1e-10, f"Volterra is (1-L)^{-1} {ident}")


def main() -> None:
    test_window_fourier_zero()
    test_x_mean_and_gram()
    test_short_driven_and_undriven()
    test_cjj_floor_and_cache_guards()
    test_cjj_source_kernel_maps()
    test_a2_mirror_eigenvectors()
    test_s1_reduced_kernel()
    test_p1_a2_drive_along_a2()
    test_k1_mode_not_photon()
    test_response_d_and_prefix()
    test_inventory_classifies_full_x_vs_y()
    test_stage1_K_and_short_eps()
    test_mode_io_units_shape_and_order()
    test_gsheet_two_port_and_volterra_adjoint()
    test_runaway_energy_and_seed_self()
    test_adapt_perk_integrand_and_tiles()
    test_feedback_op_adjoint_and_scale()
    test_photon_op_maps_and_adjoint()
    test_resummed_gamma0_recovers_born()
    test_canonical_io_audit()
    test_canonical_order_audit()
    test_phgauss_canonical()
    test_phgauss_pauli()
    test_gself_emergence()
    test_k_cart_k_frac_inverse()
    test_covariant_permutation_and_hold()
    test_h2_resolvent_identities()
    print("test_q8_m0_cjj: PASS")


if __name__ == "__main__":
    main()

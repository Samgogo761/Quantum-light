"""G1 slices B/C/D/E. Diagnostic only. production_G1_pass=false.

B: assemble U,V from existing 9.5/10 adjoint rows. No new vertices.
C: 11-cycle kernel Q only if k11 JSON already exists; never start k11.
D: 4x4 covariance. Omega_0 is the standard CCR form; Omega_eff is
   the commutator matrix induced by the current U,V (I+R_N, R_A).
E: quasi-boson readonly column magnitudes. approximation_validity=pending.

Q definition unchanged. theta_min=(arg m+pi)/2 (mod pi) is min-noise.
1/15 timestep ratio and four-window swing are conditional estimates,
not strict bounds. Does not overwrite archived JSON. G1 not passed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_g1_q_diagnostics_20261018 as qa
from q8_m0_g1_gram_grouping_20261017 import adjoint_path, assemble, gram

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_q_slices_bcde_20261018"
OUTJSON = OUTDIR / "q_slices_bcde.json"
OUTNPZ = OUTDIR / "uv_9p5_10p0.npz"
PROGRESS = OUTDIR / "progress.txt"
GNPZ = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.npz"
KNPZ = ROOT / "results/q_line/g1_kernel_correction_20261017/kernel_correction_9cycles.npz"
K11JSON = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017/kernel_correction_11cycles.json"
K11NPZ = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017/kernel_correction_11cycles.npz"
K11PROG = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017/progress.txt"
N_DARK = 1976
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def pack_real(m):
    m = np.asarray(m, float)
    return {"data": m.tolist(), "maxabs": float(np.max(np.abs(m)))}


def omega0_xp():
    om = np.zeros((4, 4), float)
    om[0, 1] = 1.0
    om[1, 0] = -1.0
    om[2, 3] = 1.0
    om[3, 2] = -1.0
    return om


def omega_eff_xp(G, A):
    G = np.asarray(G, complex)
    A = np.asarray(A, complex)
    Ad = A.conj().T
    om = np.zeros((4, 4), float)
    for mu in range(2):
        for nu in range(2):
            xmu, pmu = 2 * mu, 2 * mu + 1
            xnu, pnu = 2 * nu, 2 * nu + 1
            xx = 0.5 * (A[mu, nu] + G[mu, nu] - G[nu, mu] + Ad[mu, nu])
            pp = -0.5 * (A[mu, nu] - G[mu, nu] + G[nu, mu] + Ad[mu, nu])
            xp = -0.5j * (A[mu, nu] - G[mu, nu] - G[nu, mu] - Ad[mu, nu])
            px = -0.5j * (A[mu, nu] + G[mu, nu] + G[nu, mu] - Ad[mu, nu])
            om[xmu, xnu] = float(np.real(xx / 1j))
            om[pmu, pnu] = float(np.real(pp / 1j))
            om[xmu, pnu] = float(np.real(xp / 1j))
            om[pmu, xnu] = float(np.real(px / 1j))
    return om


def sigma_xp_legacy_half_xp(n, m, G):
    # SUPERSEDED. Historical half-XP only: writes sig[xmu,pnu], never PX, then
    # 0.5*(sig+sig.T) halves XP. Vacuum m=0 => XP=0, so vacuum cannot catch it.
    # Frozen B–E JSON used this. Call only to reproduce archived 81.408.
    n = np.asarray(n, complex)
    m = np.asarray(m, complex)
    G = np.asarray(G, complex)
    sig = np.zeros((4, 4), float)
    for mu in range(2):
        for nu in range(2):
            mdag = np.conj(m[nu, mu])
            xx = 0.5 * (m[mu, nu] + G[mu, nu] + n[mu, nu] + mdag)
            pp = -0.5 * (m[mu, nu] - G[mu, nu] - n[mu, nu] + mdag)
            xp_z = m[mu, nu] - G[mu, nu] + n[mu, nu] - mdag
            xmu, pmu = 2 * mu, 2 * mu + 1
            xnu, pnu = 2 * nu, 2 * nu + 1
            sig[xmu, xnu] = float(np.real(xx))
            sig[pmu, pnu] = float(np.real(pp))
            sig[xmu, pnu] = 0.5 * float(np.imag(xp_z))
    return 0.5 * (sig + sig.T)


def sigma_xp_repaired(n, m, C):
    # Default 4x4 (production ordering X0,P0,X1,P1). C = I+R_N = UU†-VV†.
    # Fill XP and PX=XP.T fully. Do not symmetrize a half-filled matrix.
    n = np.asarray(n, complex)
    m = np.asarray(m, complex)
    C = np.asarray(C, complex)
    ms = 0.5 * (m + m.T)
    XX = (n + ms + 0.5 * C).real
    PP = (n - ms + 0.5 * C).real
    XP = (n + ms - 0.5 * C).imag
    sig = np.zeros((4, 4), float)
    for mu in range(2):
        for nu in range(2):
            xmu, pmu = 2 * mu, 2 * mu + 1
            xnu, pnu = 2 * nu, 2 * nu + 1
            sig[xmu, xnu] = float(XX[mu, nu])
            sig[pmu, pnu] = float(PP[mu, nu])
            sig[xmu, pnu] = float(XP[mu, nu])
            sig[pmu, xnu] = float(XP[nu, mu])
    return sig


def sigma_xp(*args, **kwargs):
    raise RuntimeError(
        "sigma_xp is SUPERSEDED. Default entry is sigma_xp_repaired(n, m, C=I+R_N) "
        "or q_cov4_reference_check.analyze_uv. Use sigma_xp_legacy_half_xp only "
        "to reproduce archived half-XP (min_eig~81.408). Do not overwrite old JSON."
    )


def min_eigh(h):
    h = 0.5 * (h + h.conj().T)
    w, vec = np.linalg.eigh(h)
    v0 = vec[:, 0]
    return float(np.real(w[0])), {
        "real": np.real(v0).tolist(),
        "imag": np.imag(v0).tolist(),
    }


def williamson(sig, om):
    try:
        inv = np.linalg.inv(om)
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "Omega_singular"}
    w = np.linalg.eigvals(1j * (inv @ sig))
    return {
        "ok": True,
        "eigs_abs_sorted": np.sort(np.abs(np.real(w))).tolist(),
        "eigs_imag_max": float(np.max(np.abs(np.imag(w)))),
    }


def pt_mode1(sig):
    pt = np.array(sig, float)
    pt[3, :] *= -1.0
    pt[:, 3] *= -1.0
    return pt


def vacuum_selfcheck():
    G = np.eye(2, dtype=complex)
    A = np.zeros((2, 2), complex)
    n = np.zeros((2, 2), complex)
    m = np.zeros((2, 2), complex)
    om0 = omega0_xp()
    ome = omega_eff_xp(G, A)
    sig = sigma_xp_repaired(n, m, G)
    d_om = float(np.max(np.abs(ome - om0)))
    d_sig = float(np.max(np.abs(sig - 0.5 * np.eye(4))))
    if d_om > 1e-12 or d_sig > 1e-12:
        raise RuntimeError(f"vacuum 4x4 self-check failed d_om={d_om} d_sig={d_sig}")
    return {"d_Omega_eff_minus_Omega_0": d_om, "d_sigma_minus_half_I": d_sig}


def cov4(u, v, label):
    u = np.asarray(u, complex)
    v = np.asarray(v, complex)
    n = np.conj(v) @ v.T
    m = u @ v.T
    Gcorr = u @ u.conj().T
    vvH = v @ v.conj().T
    rn = Gcorr - vvH - np.eye(2)
    ra = u @ v.T - v @ u.T
    Gccr = np.eye(2) + rn
    sig = sigma_xp_repaired(n, m, Gccr)
    om0 = omega0_xp()
    ome = omega_eff_xp(Gccr, ra)
    h0 = sig + 0.5j * om0
    he = sig + 0.5j * ome
    e0, v0 = min_eigh(h0)
    ee, ve = min_eigh(he)
    es, vs = min_eigh(sig)
    pt = pt_mode1(sig)
    ept0, _ = min_eigh(pt + 0.5j * om0)
    epte, _ = min_eigh(pt + 0.5j * ome)
    q0 = qa.q_mu(n[0, 0], m[0, 0], rn[0, 0])
    q1 = qa.q_mu(n[1, 1], m[1, 1], rn[1, 1])
    return {
        "label": label,
        "ordering": "X0,P0,X1,P1",
        "Omega_0": "standard CCR; [X_mu,P_nu]=i delta; not the live commutator",
        "Omega_eff": "induced by live U,V via [a,a^dag]=I+R_N and R_A=UV^T-VU^T; not via <aa^dag>=UU^H",
        "maxabs_Omega_eff_minus_Omega_0": float(np.max(np.abs(ome - om0))),
        "maxabs_R_N": float(np.max(np.abs(rn))),
        "maxabs_R_A": float(np.max(np.abs(ra))),
        "sigma": pack_real(sig),
        "Omega_0_matrix": pack_real(om0),
        "Omega_eff_matrix": pack_real(ome),
        "min_eig_sigma": es,
        "min_eig_sigma_plus_i_Omega0_over_2": e0,
        "min_eig_sigma_plus_i_Omega_eff_over_2": ee,
        "min_eig_PT_mode1_plus_i_Omega0_over_2": ept0,
        "min_eig_PT_mode1_plus_i_Omega_eff_over_2": epte,
        "PT_is_sensitivity_not_PPT_or_entanglement": True,
        "williamson_Omega_0": williamson(sig, om0),
        "williamson_Omega_eff": williamson(sig, ome),
        "min_vec_sigma": vs,
        "min_vec_sigma_plus_i_Omega0_over_2": v0,
        "min_vec_sigma_plus_i_Omega_eff_over_2": ve,
        "Q0": q0,
        "Q1": q1,
        "n": qa.pack_c(n),
        "m": qa.pack_c(m),
        "production_G1_pass": False,
        "not_a_squeezing_claim": True,
    }


def quasibose_readonly(v, label):
    v = np.asarray(v, complex)
    ncol = v.shape[1]
    n_dark = min(N_DARK, ncol)
    vd, vp = v[:, :n_dark], v[:, n_dark:]
    n = np.conj(v) @ v.T
    return {
        "label": label,
        "n_cols": int(ncol),
        "n_dark_assumed_leading": int(n_dark),
        "n_photon_like": int(ncol - n_dark),
        "assemble_order_note": "gram assemble() puts dark_1976 first, then photon pieces",
        "frobenius_V_dark": float(np.linalg.norm(vd)),
        "frobenius_V_photon": float(np.linalg.norm(vp)),
        "max_col_norm_dark": float(np.max(np.linalg.norm(vd, axis=0))) if vd.size else 0.0,
        "max_col_norm_photon": float(np.max(np.linalg.norm(vp, axis=0))) if vp.size else 0.0,
        "tr_n": float(np.real(np.trace(n))),
        "approximation_validity": "pending",
        "not_Pauli_gate": True,
        "do_not_use_tr_n_as_Pauli": True,
        "N_cell": "not_selected",
        "delta_rho_c_h_v": (
            "not_constructible_from_continuum_V_columns; "
            "2026-09-23 finite-bare formulas are not the live in-vacuum conversion"
        ),
    }


def q_from_uv(u, v):
    n = np.conj(v) @ v.T
    m = u @ v.T
    rn, ra = gram(u, v)
    rn = rn - np.eye(2)
    return {
        "Q0": qa.q_mu(n[0, 0], m[0, 0], rn[0, 0]),
        "Q1": qa.q_mu(n[1, 1], m[1, 1], rn[1, 1]),
        "n": qa.pack_c(n),
        "m": qa.pack_c(m),
        "R_N": qa.pack_rn({"real": np.real(rn).tolist(), "imag": np.imag(rn).tolist()}),
        "maxabs_R_A": float(np.max(np.abs(ra))),
    }


def assemble_one(cycles, cache, gamma, pol, phi_of, eta):
    path = adjoint_path(cycles)
    if not path.exists():
        return None, None, str(path)
    from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
    prog(f"assemble {cycles} from {path.name}")
    _g, u, v = assemble(cache, gamma, pol, phi_of, eta, load_adjoint(path))
    prog(f"assembled {cycles} u={u.shape}")
    return u, v, str(path)


def looks_like_npz(path):
    if not path.exists() or path.stat().st_size < 4:
        return False
    with path.open("rb") as fh:
        return fh.read(2) == b"PK"


def q_from_packs(npack, mpack, rnpack, rapack, label, source):
    n = qa.cmat(npack)
    m = qa.cmat(mpack)
    rn = qa.cmat(rnpack)
    ra = qa.cmat(rapack)
    return {
        "label": label,
        "source": source,
        "Q0": qa.q_mu(n[0, 0], m[0, 0], rn[0, 0]),
        "Q1": qa.q_mu(n[1, 1], m[1, 1], rn[1, 1]),
        "n": qa.pack_c(n),
        "m": qa.pack_c(m),
        "R_N": qa.pack_rn({"real": np.real(rn).tolist(), "imag": np.imag(rn).tolist()}),
        "maxabs_R_A": float(np.max(np.abs(ra))),
    }


def cov4_from_nm(n, m, rn, ra, label):
    n = np.asarray(n, complex)
    m = np.asarray(m, complex)
    rn = np.asarray(rn, complex)
    ra = np.asarray(ra, complex)
    Gcorr = np.eye(2, dtype=complex) + rn + np.conj(n)
    Gccr = np.eye(2, dtype=complex) + rn
    sig = sigma_xp_repaired(n, m, Gccr)
    om0 = omega0_xp()
    ome = omega_eff_xp(Gccr, ra)
    h0 = sig + 0.5j * om0
    he = sig + 0.5j * ome
    e0, v0 = min_eigh(h0)
    ee, ve = min_eigh(he)
    es, vs = min_eigh(sig)
    pt = pt_mode1(sig)
    ept0, _ = min_eigh(pt + 0.5j * om0)
    epte, _ = min_eigh(pt + 0.5j * ome)
    return {
        "label": label,
        "from_2x2_not_UV": True,
        "ordering": "X0,P0,X1,P1",
        "Omega_0": "standard CCR; [X_mu,P_nu]=i delta; not the live commutator",
        "Omega_eff": "induced by live [a,a^dag]=I+R_N and R_A; not via <aa^dag>=UU^H",
        "maxabs_Omega_eff_minus_Omega_0": float(np.max(np.abs(ome - om0))),
        "maxabs_R_N": float(np.max(np.abs(rn))),
        "maxabs_R_A": float(np.max(np.abs(ra))),
        "sigma": pack_real(sig),
        "min_eig_sigma": es,
        "min_eig_sigma_plus_i_Omega0_over_2": e0,
        "min_eig_sigma_plus_i_Omega_eff_over_2": ee,
        "min_eig_PT_mode1_plus_i_Omega0_over_2": ept0,
        "min_eig_PT_mode1_plus_i_Omega_eff_over_2": epte,
        "PT_is_sensitivity_not_PPT_or_entanglement": True,
        "williamson_Omega_0": williamson(sig, om0),
        "williamson_Omega_eff": williamson(sig, ome),
        "min_vec_sigma": vs,
        "min_vec_sigma_plus_i_Omega0_over_2": v0,
        "min_vec_sigma_plus_i_Omega_eff_over_2": ve,
        "Q0": qa.q_mu(n[0, 0], m[0, 0], rn[0, 0]),
        "Q1": qa.q_mu(n[1, 1], m[1, 1], rn[1, 1]),
        "production_G1_pass": False,
        "not_a_squeezing_claim": True,
    }


def k11_status():
    st = {
        "json_exists": K11JSON.exists(),
        "npz_exists": K11NPZ.exists(),
        "npz_looks_like_zip": looks_like_npz(K11NPZ),
        "npz_bytes": int(K11NPZ.stat().st_size) if K11NPZ.exists() else 0,
        "progress_tail": "",
        "slice_C": "pending_k11_not_finished",
        "do_not_start_second_k11": True,
    }
    if K11PROG.exists():
        st["progress_tail"] = K11PROG.read_text(encoding="utf-8")[-400:]
    if K11JSON.exists():
        st["slice_C"] = "json_ready"
    return st


def wall_16_20(k11):
    return {
        "do_not_start_16_or_20": True,
        "basis": "k11 JSON saved; vertices ~514.9s driven / ~1870s undriven; do not start 16/20",
        "linear_window_guess_16_driven_s": 514.9 * 16.0 / 11.0,
        "linear_window_guess_20_driven_s": 514.9 * 20.0 / 11.0,
        "k11_json_exists": k11["json_exists"],
        "note": "linear-in-window is a guess, not a quote. conditional_extrapolation_not_a_job",
    }


def wait_k11(max_s=40 * 60):
    t0 = time.time()
    while not K11JSON.exists() and (time.time() - t0) < max_s:
        prog("waiting for k11 json; not assembling 9.5/10 yet")
        time.sleep(20)
    return K11JSON.exists()


def main():
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("start slices B C D E\n", encoding="utf-8")
    vac = vacuum_selfcheck()
    prog("vacuum 4x4 self-check ok")
    wait_k11()
    prog("k11 json exists=" + str(K11JSON.exists()))

    gz = np.load(GNPZ)
    kz = np.load(KNPZ)
    u9, v9 = gz["u9"], gz["v9"]
    u11, v11 = gz["u11"], gz["v11"]
    up, vp = kz["u_prod"], kz["v_prod"]
    ur, vr = kz["u_ref"], kz["v_ref"]
    prog("loaded existing U,V for 9/11/kernel")

    slice_d = {
        "nine_1024_prod": cov4(u9, v9, "9cycles_macro1024_prod"),
        "eleven_1024_prod": cov4(u11, v11, "11cycles_macro1024_prod"),
        "nine_1024_kernel_prod": cov4(up, vp, "9cycles_kernel_prod"),
        "nine_1024_kernel_ref": cov4(ur, vr, "9cycles_kernel_ref"),
        "nine_2048_prod": "U,V not assembled; 4x4 unavailable",
    }
    slice_e = {
        "nine_1024_prod": quasibose_readonly(v9, "9cycles_macro1024_prod"),
        "eleven_1024_prod": quasibose_readonly(v11, "11cycles_macro1024_prod"),
        "nine_1024_kernel_prod": quasibose_readonly(vp, "9cycles_kernel_prod"),
    }
    k11 = k11_status()
    slice_c = {"status": k11, "k11_npz_not_rebuilt": True}
    if k11["json_exists"]:
        js = json.loads(K11JSON.read_text(encoding="utf-8"))
        src = "kernel_correction_11cycles.json 2x2; k11 npz not loaded"
        slice_c["eleven_kernel_prod"] = q_from_packs(
            js["n_prod_frozen"], js["m_prod"], js["R_N_prod"], js["R_A_prod"],
            "11cycles_kernel_prod", src)
        slice_c["eleven_kernel_ref"] = q_from_packs(
            js["n_ref_frozen"], js["m_ref"], js["R_N_ref"], js["R_A_ref"],
            "11cycles_kernel_ref", src)
        slice_d["eleven_kernel_prod"] = cov4_from_nm(
            qa.cmat(js["n_prod_frozen"]), qa.cmat(js["m_prod"]),
            qa.cmat(js["R_N_prod"]), qa.cmat(js["R_A_prod"]), "11cycles_kernel_prod")
        slice_d["eleven_kernel_ref"] = cov4_from_nm(
            qa.cmat(js["n_ref_frozen"]), qa.cmat(js["m_ref"]),
            qa.cmat(js["R_N_ref"]), qa.cmat(js["R_A_ref"]), "11cycles_kernel_ref")
        slice_e["eleven_kernel_prod"] = (
            "V columns unavailable; k11 npz is not a readable zip; not rebuilt"
        )
        slice_c["status"]["slice_C"] = "included_from_json"
        slice_c["k11_npz_unreadable"] = not k11["npz_looks_like_zip"]
        prog("slice C from k11 JSON; npz_looks_like_zip=" + str(k11["npz_looks_like_zip"]))
        if k11["npz_looks_like_zip"]:
            try:
                k11z = np.load(K11NPZ)
                slice_e["eleven_kernel_prod"] = quasibose_readonly(
                    k11z["v_prod"], "11cycles_kernel_prod")
                slice_d["eleven_kernel_prod_from_UV"] = cov4(
                    k11z["u_prod"], k11z["v_prod"], "11cycles_kernel_prod_UV")
                slice_d["eleven_kernel_ref_from_UV"] = cov4(
                    k11z["u_ref"], k11z["v_ref"], "11cycles_kernel_ref_UV")
                slice_c["k11_npz_loaded_for_E_and_UV_4x4"] = True
                prog("k11 npz loaded for slice E / UV 4x4")
            except Exception as exc:
                slice_c["npz_load_error"] = type(exc).__name__ + ": " + str(exc)
                prog("k11 npz load failed; JSON 2x2 kept")
    else:
        prog("slice C pending; k11 json missing")

    from q8_m0_classical import p1_a2_protocol
    from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
    from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
    from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
    import q8_m0_gamma12_transpose_input_refine_20261008 as base

    _saved, cache, _t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    eta = eta_of(gamma)

    slice_b = {"nine_from_existing_gram": q_from_uv(u9, v9),
               "eleven_from_existing_gram": q_from_uv(u11, v11)}
    u95, v95, p95 = assemble_one(9.5, cache, gamma, pol, phi_of, eta)
    u10, v10, p10 = assemble_one(10.0, cache, gamma, pol, phi_of, eta)
    slice_b["sources"] = {"cycles9p5": p95, "cycles10p0": p10}
    npz_kw = {}
    if u95 is not None:
        slice_b["nine_p5"] = q_from_uv(u95, v95)
        slice_d["nine_p5"] = cov4(u95, v95, "9.5cycles_macro1024")
        npz_kw["u95"] = u95
        npz_kw["v95"] = v95
    else:
        slice_b["nine_p5"] = {"missing": p95, "invented": False}
    if u10 is not None:
        slice_b["ten"] = q_from_uv(u10, v10)
        slice_d["ten"] = cov4(u10, v10, "10cycles_macro1024")
        npz_kw["u10"] = u10
        npz_kw["v10"] = v10
    else:
        slice_b["ten"] = {"missing": p10, "invented": False}
    if npz_kw:
        if OUTNPZ.exists():
            raise FileExistsError(OUTNPZ)
        np.savez_compressed(OUTNPZ, **npz_kw)
        prog("saved " + str(OUTNPZ))

    beats = []
    for key, cyc in (("nine_from_existing_gram", 9.0), ("nine_p5", 9.5),
                     ("ten", 10.0), ("eleven_from_existing_gram", 11.0)):
        rec = slice_b.get(key)
        if isinstance(rec, dict) and "Q0" in rec:
            beats.append({"cycles": cyc, "Q0": rec["Q0"]["Q"], "Q1": rec["Q1"]["Q"],
                          "RN00": rec["R_N"]["diag00"]})
    q0s = [b["Q0"] for b in beats]
    swing = (max(q0s) - min(q0s)) if q0s else None
    slice_b["four_beat"] = beats
    slice_b["four_window_Q0_swing"] = swing
    slice_b["four_window_swing_label"] = "conditional_error_estimate_not_strict_bound"
    slice_b["do_not_start_16_or_20"] = True

    report = {
        "task": "G1 slices B C D E; phase-corrected Q; Omega_0 vs Omega_eff",
        "date": "2026-10-18",
        "production_G1_pass": False,
        "not_a_squeezing_claim": True,
        "not_production_nmV": True,
        "vacuum_4x4_selfcheck": vac,
        "error_estimate_labels": {
            "timestep_ratio_1_over_15": "conditional_error_estimate_not_strict_bound",
            "four_window_swing": "conditional_error_estimate_not_strict_bound",
        },
        "slice_B_window_9p5_10": slice_b,
        "slice_C_k11": slice_c,
        "slice_D_4x4": slice_d,
        "slice_E_quasibose_readonly": slice_e,
        "wall_16_20": wall_16_20(k11),
        "elapsed_s": float(elapsed()),
    }
    OUTJSON.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    prog("saved " + str(OUTJSON))
    print("done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

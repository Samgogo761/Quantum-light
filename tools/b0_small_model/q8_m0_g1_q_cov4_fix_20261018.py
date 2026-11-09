"""G1 Slice D 4x4 consistency repair. Post-process only.

Does not overwrite archived JSON. No new propagation.
production_G1_pass=false. Q / n,m / U,V not promoted.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q_cov4_reference_check as ref
from q8_m0_g1_q_slices_bcde_20261018 import sigma_xp_legacy_half_xp as sigma_xp_legacy

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_q_cov4_fix_20261018"
OUTJSON = OUTDIR / "q_cov4_fix.json"
PROGRESS = OUTDIR / "progress.txt"
GNPZ = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.npz"
UV95 = ROOT / "results/q_line/g1_q_slices_bcde_20261018/uv_9p5_10p0.npz"
K9NPZ = ROOT / "results/q_line/g1_kernel_correction_20261017/kernel_correction_9cycles.npz"
K11NPZ = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017/kernel_correction_11cycles.npz"
K11JSON = ROOT / "results/q_line/g1_kernel_correction_11cycles_20261017/kernel_correction_11cycles.json"
BEJSON = ROOT / "results/q_line/g1_q_slices_bcde_20261018/q_slices_bcde.json"
TSJSON = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables_with_timestep.json"

# prod X0,P0,X1,P1 <-> ref X0,X1,P0,P1 (involution)
PERM = np.array([0, 2, 1, 3], int)
CHI_ROW0 = (0.37, 1.9, -2.4)
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def jfloat(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, complex):
        return {"real": float(x.real), "imag": float(x.imag)}
    return x


def pack_mat(a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return {
            "real": np.real(a).tolist(),
            "imag": np.imag(a).tolist(),
            "maxabs": float(np.max(np.abs(a))),
        }
    return {"data": np.asarray(a, float).tolist(), "maxabs": float(np.max(np.abs(a)))}


def reorder(m, to_ref=True):
    p = PERM
    m = np.asarray(m)
    return m[np.ix_(p, p)]


def sigma_xp_repaired(n, m, C):
    # Old sigma_xp filled only XP then 0.5*(sig+sig.T), which halves XP.
    # Vacuum m=0 => XP=0, so vacuum cannot catch it. Fill XP and PX=XP.T fully.
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


def omega0_xp():
    om = np.zeros((4, 4), float)
    om[0, 1] = 1.0
    om[1, 0] = -1.0
    om[2, 3] = 1.0
    om[3, 2] = -1.0
    return om


def omega_eff_from_ref(om_ref):
    return reorder(om_ref, to_ref=True)


def load_npz(path, keys):
    if not path.exists():
        raise FileNotFoundError(str(path))
    z = np.load(path)
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise KeyError(f"{path.name} missing {missing}")
    return {k: np.asarray(z[k], complex) for k in keys}


def rotate_uv_rows(U, V, chi):
    U = np.array(U, complex, copy=True)
    V = np.array(V, complex, copy=True)
    fac = np.exp(1j * np.asarray(chi, float))
    U *= fac[:, None]
    V *= fac[:, None]
    return U, V


def rotate_u_only(U, V, chi):
    U = np.array(U, complex, copy=True)
    V = np.array(V, complex, copy=True)
    fac = np.exp(1j * np.asarray(chi, float))
    U *= fac[:, None]
    return U, V


def phase_rel(sig_a, sig_b):
    ea = np.linalg.eigvalsh(sig_a)
    eb = np.linalg.eigvalsh(sig_b)
    return float(np.max(np.abs(ea - eb) / np.maximum(np.abs(ea), 1e-300)))


def q_pair(n, m, C):
    return ref.q_values(n, m, C)


def williamson_prod(sig, om):
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


def blocks_prod(sig, n, m, C):
    Q = q_pair(n, m, C)
    out = []
    for mu in (0, 1):
        idx = [2 * mu, 2 * mu + 1]
        blk = sig[np.ix_(idx, idx)]
        th = float(((np.angle(m[mu, mu]) + np.pi) / 2) % np.pi)
        v = np.array([np.cos(th), np.sin(th)])
        out.append({
            "mode": mu,
            "block": pack_mat(blk),
            "block_min_eig": float(np.linalg.eigvalsh(blk)[0]),
            "Q": float(Q[mu]),
            "theta_min": th,
            "var_at_theta_min": float(v @ blk @ v),
        })
    return out, Q


def pt_mode1_prod(sig):
    pt = np.array(sig, float)
    pt[3, :] *= -1.0
    pt[:, 3] *= -1.0
    return pt


def assert_complete(rec, has_uv):
    fails = []
    sig = np.asarray(rec["_sig_prod"], float)
    n, m, C = rec["_n"], rec["_m"], rec["_C"]
    Q = rec["_Q"]
    ev = np.linalg.eigvalsh(sig)
    lmin, lmax = float(ev[0]), float(ev[-1])
    tol = 1e-10 * max(1.0, lmax)
    floor = 1e-9 * max(1.0, lmax)
    if not (lmin <= min(Q) + tol):
        fails.append(f"interlacing {lmin} > minQ {min(Q)}")
    blks, _ = blocks_prod(sig, n, m, C)
    for b in blks:
        if abs(b["block_min_eig"] - b["Q"]) > tol:
            fails.append(
                f"block{b['mode']} {b['block_min_eig']} != Q {b['Q']}"
            )
    ome = rec["_ome_prod"]
    he = float(np.linalg.eigvalsh(sig + 0.5j * ome)[0])
    if he < -floor:
        fails.append(f"sigma+i Omega_eff/2 min {he}")
    if has_uv:
        dnm = rec["maxabs_sigma_nm_minus_S"]
        if dnm >= 1e-11 * max(1.0, lmax):
            fails.append(f"|sigma_nm-S| {dnm}")
        if rec["phase_uv_together_rel"] >= 1e-9:
            fails.append(f"phase UV {rec['phase_uv_together_rel']}")
    rec["assert_fails"] = fails
    rec["asserts_ok"] = len(fails) == 0
    if lmin < 0.5:
        rec["min_eig_below_half_diagnostic_only"] = True
    return fails


def report_uv(U, V, label, reported_min=None):
    U = np.asarray(U, complex)
    V = np.asarray(V, complex)
    if U.shape[0] != 2 and U.ndim == 2 and U.shape[1] == 2:
        U, V = U.T, V.T
    n, m, C, RA = ref.nm_from_uv(U, V)
    sig_S, om_ref = ref.sigma_omega_from_uv(U, V)
    sig_nm_ref = ref.sigma_from_nm(n, m, C)
    sig_prod = sigma_xp_repaired(n, m, C)
    sig_prod_from_S = reorder(sig_S)
    d_order = float(np.max(np.abs(sig_prod - sig_prod_from_S)))
    ome_prod = reorder(om_ref)
    om0 = omega0_xp()
    ev = np.linalg.eigvalsh(sig_prod)
    blks, Q = blocks_prod(sig_prod, n, m, C)
    lmax = float(max(abs(ev[0]), abs(ev[-1]), 1.0))
    chi0 = np.array([CHI_ROW0[0], 0.0])
    Ur, Vr = rotate_uv_rows(U, V, chi0)
    nr, mr, Cr, _ = ref.nm_from_uv(Ur, Vr)
    sig_rot = sigma_xp_repaired(nr, mr, Cr)
    rel_uv = phase_rel(sig_prod, sig_rot)
    Uo, Vo = rotate_u_only(U, V, chi0)
    no, mo, Co, _ = ref.nm_from_uv(Uo, Vo)
    rel_uonly = phase_rel(sig_prod, sigma_xp_repaired(no, mo, Co))
    worst = 0.0
    for chi in CHI_ROW0:
        Uc, Vc = rotate_uv_rows(U, V, np.array([chi, 0.0]))
        nc, mc, Cc, _ = ref.nm_from_uv(Uc, Vc)
        worst = max(worst, phase_rel(sig_prod, sigma_xp_repaired(nc, mc, Cc)))
    for chi in ref.CHI_SETS:
        Uc, Vc = rotate_uv_rows(U, V, chi)
        nc, mc, Cc, _ = ref.nm_from_uv(Uc, Vc)
        worst = max(worst, phase_rel(sig_prod, sigma_xp_repaired(nc, mc, Cc)))
    Gcorr = U @ U.conj().T
    sig_legacy = sigma_xp_legacy(n, m, Gcorr)
    he = float(np.linalg.eigvalsh(sig_prod + 0.5j * ome_prod)[0])
    h0 = float(np.linalg.eigvalsh(sig_prod + 0.5j * om0)[0])
    pt = pt_mode1_prod(sig_prod)
    rec = {
        "label": label,
        "ordering": "X0,P0,X1,P1",
        "ordering_ref_path": "X0,X1,P0,P1",
        "ordering_note": "PERM=[0,2,1,3] involution between production and reference",
        "ncol": int(U.shape[1]),
        "maxabs_R_N": float(np.max(np.abs(C - np.eye(2)))),
        "maxabs_R_A": float(np.max(np.abs(RA))),
        "n": pack_mat(n),
        "m": pack_mat(m),
        "C": pack_mat(C),
        "sigma_prod": pack_mat(sig_prod),
        "sigma_ref_S": pack_mat(sig_S),
        "maxabs_sigma_nm_minus_S": float(np.max(np.abs(sig_nm_ref - sig_S))),
        "maxabs_sigma_prod_minus_S_reordered": d_order,
        "Omega_0_prod": pack_mat(om0),
        "Omega_eff_prod": pack_mat(ome_prod),
        "maxabs_Omega_eff_minus_Omega_0": float(np.max(np.abs(ome_prod - om0))),
        "eig_sigma": ev.tolist(),
        "min_eig_sigma": float(ev[0]),
        "min_eig_sigma_is_ordinary_not_symplectic": True,
        "min_eig_sigma_plus_i_Omega0_over_2": h0,
        "min_eig_sigma_plus_i_Omega_eff_over_2": he,
        "williamson_Omega_0": williamson_prod(sig_prod, om0),
        "williamson_Omega_eff": williamson_prod(sig_prod, ome_prod),
        "symplectic_Omega0_ref_order": ref.symp_eigs(sig_S, ref.OMEGA0).tolist(),
        "pt_symplectic_Omega0_ref_order": ref.spectra(sig_S)["nu_pt"].tolist(),
        "PT_is_sensitivity_not_PPT_or_entanglement": True,
        "min_eig_PT_mode1_plus_i_Omega0_over_2": float(
            np.linalg.eigvalsh(pt + 0.5j * om0)[0]
        ),
        "Q0": float(Q[0]),
        "Q1": float(Q[1]),
        "blocks": blks,
        "interlacing_ok": bool(ev[0] <= min(Q) + 1e-10 * lmax),
        "phase_uv_together_rel": float(worst),
        "phase_uv_row0_chi037_rel": rel_uv,
        "phase_u_only_not_acceptance_rel": rel_uonly,
        "phase_u_only_is_not_output_mode_phase": True,
        "legacy_half_xp_min_eig": float(np.linalg.eigvalsh(sig_legacy)[0]),
        "legacy_XP00": float(sig_legacy[0, 1]),
        "repaired_XP00": float(sig_prod[0, 1]),
        "production_G1_pass": False,
        "not_a_squeezing_claim": True,
        "_sig_prod": sig_prod,
        "_n": n,
        "_m": m,
        "_C": C,
        "_Q": Q,
        "_ome_prod": ome_prod,
    }
    if reported_min is not None:
        rec["legacy_minus_reported"] = rec["legacy_half_xp_min_eig"] - reported_min
        rec["repaired_minus_reported_81"] = rec["min_eig_sigma"] - reported_min
    fails = assert_complete(rec, has_uv=True)
    rec.pop("_sig_prod")
    rec.pop("_n")
    rec.pop("_m")
    rec.pop("_C")
    rec.pop("_Q")
    rec.pop("_ome_prod")
    rec["assert_fails"] = fails
    rec["asserts_ok"] = len(fails) == 0
    return rec, fails


def report_nm(n, m, rn, ra, label):
    n = np.asarray(n, complex)
    m = np.asarray(m, complex)
    rn = np.asarray(rn, complex)
    ra = np.asarray(ra, complex)
    C = np.eye(2, dtype=complex) + rn
    sig_prod = sigma_xp_repaired(n, m, C)
    sig_ref = ref.sigma_from_nm(n, m, C)
    om0 = omega0_xp()
    ome = reorder(ref.OMEGA0)
    # Omega_eff from (I+R_N, R_A) via ref vacuum Omega if RA~0; keep prod CCR form
    # Use production omega_eff_xp analogue: permute standard plus small RA not in nm-only Omega0.
    from q8_m0_g1_q_slices_bcde_20261018 import omega_eff_xp
    ome = omega_eff_xp(C, ra)
    ev = np.linalg.eigvalsh(sig_prod)
    blks, Q = blocks_prod(sig_prod, n, m, C)
    lmax = float(max(abs(ev[0]), abs(ev[-1]), 1.0))
    rec = {
        "label": label,
        "from_2x2_not_UV": True,
        "ordering": "X0,P0,X1,P1",
        "maxabs_R_N": float(np.max(np.abs(rn))),
        "maxabs_R_A": float(np.max(np.abs(ra))),
        "n": pack_mat(n),
        "m": pack_mat(m),
        "sigma_prod": pack_mat(sig_prod),
        "maxabs_sigma_prod_minus_nm_ref_reordered": float(
            np.max(np.abs(sig_prod - reorder(sig_ref)))
        ),
        "eig_sigma": ev.tolist(),
        "min_eig_sigma": float(ev[0]),
        "min_eig_sigma_is_ordinary_not_symplectic": True,
        "Q0": float(Q[0]),
        "Q1": float(Q[1]),
        "blocks": blks,
        "interlacing_ok": bool(ev[0] <= min(Q) + 1e-10 * lmax),
        "maxabs_Omega_eff_minus_Omega_0": float(np.max(np.abs(ome - om0))),
        "min_eig_sigma_plus_i_Omega0_over_2": float(
            np.linalg.eigvalsh(sig_prod + 0.5j * om0)[0]
        ),
        "min_eig_sigma_plus_i_Omega_eff_over_2": float(
            np.linalg.eigvalsh(sig_prod + 0.5j * ome)[0]
        ),
        "williamson_Omega_0": williamson_prod(sig_prod, om0),
        "williamson_Omega_eff": williamson_prod(sig_prod, ome),
        "PT_is_sensitivity_not_PPT_or_entanglement": True,
        "production_G1_pass": False,
        "_sig_prod": sig_prod,
        "_n": n,
        "_m": m,
        "_C": C,
        "_Q": Q,
        "_ome_prod": ome,
        "maxabs_sigma_nm_minus_S": 0.0,
        "phase_uv_together_rel": 0.0,
    }
    fails = assert_complete(rec, has_uv=False)
    for k in ("_sig_prod", "_n", "_m", "_C", "_Q", "_ome_prod"):
        rec.pop(k)
    rec.pop("maxabs_sigma_nm_minus_S")
    rec.pop("phase_uv_together_rel")
    rec["assert_fails"] = fails
    rec["asserts_ok"] = len(fails) == 0
    return rec, fails


def vacuum_selfcheck():
    U = np.hstack([np.eye(2, dtype=complex), np.zeros((2, 3), complex)])
    V = np.zeros((2, 5), complex)
    n, m, C, RA = ref.nm_from_uv(U, V)
    sig = sigma_xp_repaired(n, m, C)
    om0 = omega0_xp()
    sig_S, om_ref = ref.sigma_omega_from_uv(U, V)
    ome = reorder(om_ref)
    d_om = float(np.max(np.abs(ome - om0)))
    d_sig = float(np.max(np.abs(sig - 0.5 * np.eye(4))))
    if d_om > 1e-12 or d_sig > 1e-12:
        raise RuntimeError(f"vacuum 4x4 failed d_om={d_om} d_sig={d_sig}")
    return {"d_Omega_eff_minus_Omega_0": d_om, "d_sigma_minus_half_I": d_sig}


def diagnose_uv(U, V, reported_min, tag):
    ana = ref.analyze_uv(U, V, reported_min=reported_min, verbose=True)
    n, m, C, _RA = ref.nm_from_uv(U, V)
    Gcorr = U @ U.conj().T
    sig_leg = sigma_xp_legacy(n, m, Gcorr)
    ev_leg = np.linalg.eigvalsh(sig_leg)
    sig_fix = sigma_xp_repaired(n, m, C)
    Q = q_pair(n, m, C)
    half = ana["bug_variants"]["half_xp"]
    return {
        "tag": tag,
        "R_N_max": ana["R_N_max"],
        "R_A_max": ana.get("R_A_max"),
        "Q": ana["Q"],
        "ref_min_eig_sigma": ana["eig_sigma"][0],
        "legacy_min_eig_sigma": float(ev_leg[0]),
        "legacy_min_eig_is_ordinary_not_symplectic": True,
        "legacy_XP00": float(sig_leg[0, 1]),
        "repaired_XP00": float(sig_fix[0, 1]),
        "legacy_interlacing_ok": bool(
            ev_leg[0] <= min(Q) + 1e-10 * max(1.0, abs(ev_leg[-1]))
        ),
        "half_xp_min_eig": half["min_eig"],
        "half_xp_diff_to_reported": half.get("diff_to_reported"),
        "reported_min": reported_min,
        "phase_invariance_reference": ana["phase_invariance_reference"],
        "max_abs_sigma_nm_minus_S": ana.get("max_abs_sigma_nm_minus_S"),
    }


def n_series_from_be(be):
    b = be["slice_B_window_9p5_10"]
    keys = [
        ("nine_from_existing_gram", 9.0),
        ("nine_p5", 9.5),
        ("ten", 10.0),
        ("eleven_from_existing_gram", 11.0),
    ]
    rows = []
    for key, cyc in keys:
        rec = b[key]
        n00 = rec["n"]["real"][0][0] + 1j * rec["n"]["imag"][0][0]
        m00 = rec["m"]["real"][0][0] + 1j * rec["m"]["imag"][0][0]
        rows.append({
            "cycles": cyc,
            "n00": float(np.real(n00)),
            "abs_m00": float(abs(m00)),
            "arg_m00": float(np.angle(m00)),
            "Q0": rec["Q0"]["Q"],
            "Q1": rec["Q1"]["Q"],
        })
    return {
        "rows": rows,
        "note": "Q stable does not mean n has converged; window unbounded; do not start 16/20",
        "four_window_swing_label": "conditional_error_estimate_not_strict_bound",
        "do_not_start_16_or_20": True,
    }


def main():
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("start slice D 4x4 fix\n", encoding="utf-8")
    status = "failed_consistency_check"
    diag = {}
    repaired = {}

    prog("reference selftest")
    ok_self = ref.selftest()
    diag["reference_selftest_ok"] = bool(ok_self)
    if not ok_self:
        raise RuntimeError("reference selftest failed")
    vac = vacuum_selfcheck()
    prog("vacuum repaired 4x4 ok")
    diag["vacuum_repaired"] = vac

    prog("load gram 9/11")
    g = load_npz(GNPZ, ["u9", "v9", "u11", "v11"])
    prog("load 9.5/10 uv npz; no assemble")
    u95 = load_npz(UV95, ["u95", "v95", "u10", "v10"])
    prog("load 9-cycle kernel npz")
    k9 = load_npz(K9NPZ, ["u_prod", "v_prod", "u_ref", "v_ref"])
    prog("load k11 npz")
    k11 = load_npz(K11NPZ, ["u_prod", "v_prod", "u_ref", "v_ref"])

    be = json.loads(BEJSON.read_text(encoding="utf-8"))
    reported_9 = float(be["slice_D_4x4"]["nine_1024_prod"]["min_eig_sigma"])
    reported_11 = float(be["slice_D_4x4"]["eleven_1024_prod"]["min_eig_sigma"])
    diag["archived_min_eig_sigma_9"] = reported_9
    diag["archived_min_eig_sigma_11"] = reported_11
    diag["min_eig_sigma_field"] = "ordinary_eigenvalue_not_symplectic"

    prog("diagnose 9@1024 vs 81.408")
    diag["nine_1024"] = diagnose_uv(g["u9"], g["v9"], 81.408, "9cycles_gram")
    prog("diagnose k11 prod vs 81.488")
    diag["k11_prod"] = diagnose_uv(k11["u_prod"], k11["v_prod"], 81.488, "k11_prod")
    prog("diagnose 9 kernel prod")
    diag["nine_kernel_prod"] = diagnose_uv(
        k9["u_prod"], k9["v_prod"], 81.408, "9kernel_prod"
    )

    jobs = [
        ("nine_1024_prod", g["u9"], g["v9"], reported_9),
        ("nine_p5", u95["u95"], u95["v95"], None),
        ("ten", u95["u10"], u95["v10"], None),
        ("eleven_1024_prod", g["u11"], g["v11"], reported_11),
        ("nine_kernel_prod", k9["u_prod"], k9["v_prod"], None),
        ("nine_kernel_ref", k9["u_ref"], k9["v_ref"], None),
        ("k11_prod_S", k11["u_prod"], k11["v_prod"], None),
        ("k11_ref_S", k11["u_ref"], k11["v_ref"], None),
    ]
    for name, U, V, rmin in jobs:
        prog("recompute " + name)
        rec, fails = report_uv(U, V, name, reported_min=rmin)
        repaired[name] = rec
        if fails:
            prog("ASSERT FAIL " + name + " " + str(fails))
            raise RuntimeError(name + ": " + "; ".join(fails))

    prog("k11 JSON nm path")
    js = json.loads(K11JSON.read_text(encoding="utf-8"))
    import q8_m0_g1_q_diagnostics_20261018 as qa
    rec_nm, fails = report_nm(
        qa.cmat(js["n_prod_frozen"]), qa.cmat(js["m_prod"]),
        qa.cmat(js["R_N_prod"]), qa.cmat(js["R_A_prod"]),
        "k11_prod_nm_json",
    )
    repaired["k11_prod_nm_json"] = rec_nm
    dcross = float(np.max(np.abs(
        np.asarray(repaired["k11_prod_S"]["sigma_prod"]["data"])
        - np.asarray(rec_nm["sigma_prod"]["data"])
    )))
    repaired["k11_prod_nm_json"]["maxabs_vs_S_path"] = dcross
    if fails:
        raise RuntimeError("k11 nm: " + "; ".join(fails))
    rec_nmr, fails = report_nm(
        qa.cmat(js["n_ref_frozen"]), qa.cmat(js["m_ref"]),
        qa.cmat(js["R_N_ref"]), qa.cmat(js["R_A_ref"]),
        "k11_ref_nm_json",
    )
    repaired["k11_ref_nm_json"] = rec_nmr
    if fails:
        raise RuntimeError("k11 ref nm: " + "; ".join(fails))

    repaired["nine_2048_prod"] = {
        "status": "incomplete_missing_RN_RA_UV",
        "checks": "not_done",
        "invented_C_eq_I": False,
        "note": "nm_tables_with_timestep.json has n,m 2x2 only; no full U,V / R_N / R_A",
        "timestep_json_exists": TSJSON.exists(),
    }

    nine = repaired["nine_1024_prod"]
    if nine["min_eig_sigma"] > min(nine["Q0"], nine["Q1"]) + 1e-8:
        raise RuntimeError("9@1024 still fails interlacing")
    if abs(nine["min_eig_sigma"] - 81.408) < 1.0:
        raise RuntimeError("9@1024 still reporting ~81")

    status = "assembly_repaired_not_G1_pass"
    nseries = n_series_from_be(be)
    report = {
        "task": "G1 slice D 4x4 consistency repair",
        "date": "2026-10-18",
        "production_G1_pass": False,
        "slice_D_status": status,
        "not_a_squeezing_claim": True,
        "PT_is_sensitivity_not_PPT_or_entanglement": True,
        "do_not_start_second_k11": True,
        "no_new_propagation": True,
        "archived_json_untouched": True,
        "error_estimate_labels": {
            "timestep_ratio_1_over_15": "conditional_error_estimate_not_strict_bound",
            "four_window_swing": "conditional_error_estimate_not_strict_bound",
        },
        "vacuum_4x4_selfcheck": vac,
        "diagnosis": diag,
        "repaired": repaired,
        "n_series_readonly": nseries,
        "P2_H2": {
            "H2_two_row_UV": "not_available",
            "do_not_assemble_or_propagate": True,
        },
        "elapsed_s": float(elapsed()),
    }
    OUTJSON.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    prog("saved " + str(OUTJSON))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        with PROGRESS.open("a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise



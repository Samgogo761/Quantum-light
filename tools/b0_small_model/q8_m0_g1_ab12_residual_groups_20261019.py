"""G1 A + B1/B2, B3 mapping only. No new propagation.

A: regression that default 4x4 is repaired and legacy_half_xp reproduces 81.408.
B1: reuse 9/11 Gram kinds; slice existing U,V for 9/9.5/10/11; no assemble().
B2: boxed weight sensitivity dR_N/de=C_g, dQ/de from grouped U,V (U includes U0).
B3: code/formula map only; no quadrature audit; 1e-8 is not a hard criterion.

production_G1_pass=false. Does not overwrite archives.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q_cov4_reference_check as ref
import q8_m0_g1_q_diagnostics_20261018 as qa
from q8_m0_g1_q_slices_bcde_20261018 import (
    sigma_xp_legacy_half_xp,
    sigma_xp_repaired,
)
from q8_m0_g1_q_cov4_fix_20261018 import sigma_xp_repaired as sigma_xp_fix
from q8_m0_g1_boundary_sensitivity_20261015 import frequencies
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_ab12_residual_groups_20261019"
OUTJSON = OUTDIR / "ab12_residual_groups.json"
PROGRESS = OUTDIR / "progress.txt"
GNPZ = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.npz"
GJSON = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.json"
UV95 = ROOT / "results/q_line/g1_q_slices_bcde_20261018/uv_9p5_10p0.npz"
BEJSON = ROOT / "results/q_line/g1_q_slices_bcde_20261018/q_slices_bcde.json"
N_DARK = 1976
REPORTED_MIN = 81.40810044116564
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def pack(z):
    z = np.asarray(z, complex)
    return {
        "real": np.real(z).tolist(),
        "imag": np.imag(z).tolist(),
        "maxabs": float(np.max(np.abs(z))),
        "diag00": float(np.real(z[0, 0])),
        "diag11": float(np.real(z[1, 1])),
    }


def cmat(p):
    return np.asarray(p["real"], float) + 1j * np.asarray(p["imag"], float)


def gram_uv(u, v):
    return u @ u.conj().T - v @ v.conj().T, u @ v.T - v @ u.T


def nm_uv(u, v):
    n = np.conj(v) @ v.T
    m = u @ v.T
    C, RA = gram_uv(u, v)
    return n, m, C, RA


def group_layout(cache, gamma):
    pieces, edge = frequencies(cache, gamma)
    recs = []
    col = 0
    recs.append({"name": "dark_1976", "kind": "dark", "start": 0, "stop": N_DARK, "n_col": N_DARK})
    col = N_DARK
    for name, om, _wt in pieces:
        n = int(len(om))
        for start in range(0, n, 400):
            stop = min(start + 400, n)
            n_col = 4 * (stop - start)
            recs.append({
                "name": f"{name}[{start}:{stop}]",
                "kind": f"photon:{name}",
                "start": col,
                "stop": col + n_col,
                "n_col": n_col,
            })
            col += n_col
    return recs, col, float(edge)


def coarse_bucket(kind):
    if kind == "dark":
        return "dark"
    if kind == "photon:kept":
        return "photon_kept"
    if kind == "photon:core":
        return "photon_core"
    if kind == "photon:wing230":
        return "photon_wing230"
    if kind == "photon:wing264":
        return "photon_wing264"
    if kind == "photon:infrared_1e-8_log64":
        return "photon_infrared"
    if kind.startswith("photon:tail"):
        return "photon_tails"
    if kind.startswith("photon:"):
        return "photon_other"
    return "other"


def dq_boxed(n, m, n_g, m_g, C_g):
    out = []
    for mu in (0, 1):
        mm = m[mu, mu]
        am = abs(mm)
        phase = np.conj(mm) / am if am > 0.0 else 0.0
        val = n_g[mu, mu] + 0.5 * C_g[mu, mu] - np.real(phase * m_g[mu, mu])
        out.append(float(np.real(val)))
    return out


def window_record(u, v, layout, kinds):
    n, m, C, RA = nm_uv(u, v)
    rn = C - np.eye(2)
    q0 = qa.q_mu(n[0, 0], m[0, 0], rn[0, 0])
    q1 = qa.q_mu(n[1, 1], m[1, 1], rn[1, 1])
    by_kind = {k: {"C": np.zeros((2, 2), complex),
                   "n": np.zeros((2, 2), complex),
                   "m": np.zeros((2, 2), complex),
                   "n_col": 0} for k in kinds}
    for rec in layout:
        sl = slice(rec["start"], rec["stop"])
        ug, vg = u[:, sl], v[:, sl]
        Cg, _ = gram_uv(ug, vg)
        ng = np.conj(vg) @ vg.T
        mg = ug @ vg.T
        bucket = by_kind[rec["kind"]]
        bucket["C"] += Cg
        bucket["n"] += ng
        bucket["m"] += mg
        bucket["n_col"] += rec["n_col"]
    self_sum = sum(v["C"] for v in by_kind.values())
    cross = C - self_sum
    groups = {}
    coarse = {}
    for kind, rec in by_kind.items():
        dQ = dq_boxed(n, m, rec["n"], rec["m"], rec["C"])
        groups[kind] = {
            "n_col": rec["n_col"],
            "C": pack(rec["C"]),
            "n": pack(rec["n"]),
            "m": pack(rec["m"]),
            "dRN_de_diag00": rec["C"][0, 0].real,
            "dRN_de_diag11": rec["C"][1, 1].real,
            "dQ0_de": dQ[0],
            "dQ1_de": dQ[1],
            "dark_scale_is_not_frequency_quadrature": kind == "dark",
        }
        ck = coarse_bucket(kind)
        slot = coarse.setdefault(ck, {"C00": 0.0, "C11": 0.0, "dQ0": 0.0, "dQ1": 0.0, "n_col": 0})
        slot["C00"] += float(rec["C"][0, 0].real)
        slot["C11"] += float(rec["C"][1, 1].real)
        slot["dQ0"] += dQ[0]
        slot["dQ1"] += dQ[1]
        slot["n_col"] += rec["n_col"]
    return {
        "ncol": int(u.shape[1]),
        "R_N": pack(rn),
        "R_A_maxabs": float(np.max(np.abs(RA))),
        "C_minus_I_is_RN": True,
        "Q0": q0["Q"],
        "Q1": q1["Q"],
        "n00": float(n[0, 0].real),
        "abs_m00": float(abs(m[0, 0])),
        "groups": groups,
        "coarse": coarse,
        "cross_between_groups": pack(cross),
        "reconstruction_maxabs_self_plus_cross_minus_C": float(np.max(np.abs(self_sum + cross - C))),
        "U_includes_U0": True,
        "weight_sensitivity_is_artificial_not_a_bound": True,
    }


def slice_A(u9, v9):
    n, m, C, _RA = nm_uv(u9, v9)
    Gcorr = u9 @ u9.conj().T
    sig_leg = sigma_xp_legacy_half_xp(n, m, Gcorr)
    sig_rep = sigma_xp_repaired(n, m, C)
    sig_fix = sigma_xp_fix(n, m, C)
    lam_leg = float(np.linalg.eigvalsh(sig_leg)[0])
    lam_rep = float(np.linalg.eigvalsh(sig_rep)[0])
    Q = [float(n[mu, mu].real + 0.5 * C[mu, mu].real - abs(m[mu, mu])) for mu in (0, 1)]
    raised = False
    try:
        from q8_m0_g1_q_slices_bcde_20261018 import sigma_xp
        sigma_xp(n, m, Gcorr)
    except RuntimeError:
        raised = True
    self_ok = bool(ref.selftest())
    o = ref.analyze_uv(u9, v9, reported_min=REPORTED_MIN, verbose=False)
    return {
        "default_entry": "q_cov4_reference_check.analyze_uv / sigma_xp_repaired",
        "legacy_only_for_reproduction": "sigma_xp_legacy_half_xp",
        "sigma_xp_bare_raises": raised,
        "reference_selftest_ok": self_ok,
        "legacy_min_eig": lam_leg,
        "legacy_minus_reported": lam_leg - REPORTED_MIN,
        "repaired_min_eig": lam_rep,
        "fix_copy_agrees_maxabs": float(np.max(np.abs(sig_rep - sig_fix))),
        "Q0": Q[0],
        "Q1": Q[1],
        "interlacing_ok": bool(lam_rep <= min(Q) + 1e-10 * max(1.0, abs(lam_rep))),
        "two_path_maxabs": o.get("max_abs_sigma_nm_minus_S"),
        "reference_half_xp_min_eig": o["bug_variants"]["half_xp"]["min_eig"],
        "production_G1_pass": False,
        "old_json_not_overwritten": True,
    }


def slice_B3():
    return {
        "status": "code_formula_mapping_only",
        "numerical_integral_audit": "not_started",
        "do_not_set_1e-8_as_hard_criterion": True,
        "W_in_is_not_D": True,
        "D_is_not_radiative_damping": True,
        "gamma_is_radiative_feedback_not_D": True,
        "objects": {
            "W_in": {
                "meaning": "asymptotic incoming vacuum represented in program coordinates; not a dense production matrix",
                "code": [
                    "q8_m0_incoming_spectral_completeness_20261004.assemble_incoming_bundle",
                    "q8_m0_p1_h2_finite_kcv_input_close_20261005.u0_from_bundle",
                    "q8_m0_p1_h2_finite_kcv_input_close_20261005.attach_weights",
                ],
                "math": "columns of incoming a_j; U0 maps free optical modes through detector/port conversion; dark kcv are discrete electronic DOF, not frequency nodes",
            },
            "free_incoming_A": {
                "code": [
                    "q8_m0_continuum_columns_20261004.PhotonReference.free_A0 / free_A / free_Adot",
                    "q8_m0_p1_h2_finite_kcv_caller_fix_20261005.free_A_increment",
                ],
                "math": "A_f(t)=A_f(0) e^{-iωt}; free_A0 = η/√ω × (L,R)×2 pol before √(dν); after attach_weights, √(dν) is in the column",
            },
            "continuum_nodes_weights": {
                "code": [
                    "q8_m0_p1_h2_finite_kcv_input_close_20261005.input_quadrature",
                    "q8_m0_g1_boundary_sensitivity_20261015.frequencies / final_frequencies",
                ],
                "math": "positive-ω measure dν counted once; 4 columns per node (2 sides × 2 pol); IR log64 is an extra panel, not a new physical bath",
            },
            "eta": {
                "code": "q8_m0_p1_h2_finite_kcv_input_close_20261005.eta_of",
                "math": "η=sqrt(γ/2π); detector kernel K(t)=iη∫ dω φ(ω)/√ω e^{+iωt}; do not multiply resolvent by extra 2π",
            },
            "gamma": {
                "code": "q8_m0_gsheet.gamma_self ; rhs Adot in q8_m0_continuum_columns_20261004",
                "math": "γ=2π η² N_cell convention in gsheet; Ȧ = Ȧ_f + γ I ; radiative feedback coupling, not D",
            },
            "D_t": {
                "code": [
                    "q8_m0_continuum_columns_20261004.current / real_frequency_kernel",
                    "frozen cache D0, vertices u(t)",
                ],
                "math": "I = u* b + u b† − D A ; D is the electronic contact (diamagnetic-like), real symmetric; K(ω)=D−Π(ω) is the electronic kernel, not a radiation-damping coefficient",
            },
            "Maxwell_feedback": {
                "code": [
                    "q8_m0_gsheet.a_self_from_current / two_port_from_current",
                    "q8_m0_continuum_columns_20261004.midpoint_step",
                ],
                "math": "thin-sheet E_self=−(Z0/2) j ; Ȧ_self=γ I ; two-port A_out identities; not fluctuation–dissipation of (W_in, D)",
            },
        },
        "what_discrete_commutator_should_be_compared_to": {
            "target": "continuum incoming optical CCR / port-normalized free-field kernel, not D and not W_in as a noise kernel paired with D",
            "must_state_before_any_integral_audit": [
                "ports L/R",
                "two real polarizations",
                "frequency cutoff and panel partition",
                "mode normalization η/√ω and √(dν) once",
                "regularization of IR/UV panels",
                "how discrete [A,E] or [b,b†] on selected columns maps to that continuum kernel",
            ],
            "error_propagation_required": "derive how node/weight error hits R_N before choosing a numerical tolerance; 1e-8 from large-n cancellation is heuristic only",
        },
        "fluctuation_dissipation_warning": (
            "Do not test W_in vs D as a standard noise/dissipation pair. "
            "The frozen pair is optical-input normalization + γ-feedback + electronic contact D, "
            "which must jointly preserve the required CCR; that is a model derivation, not two integrals."
        ),
        "next_audit_proposal": "mapping_only; submit a numerical plan only after the comparison object and error propagation are written down",
    }


def main():
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("start A+B1/B2 B3-map\n", encoding="utf-8")

    gz = np.load(GNPZ)
    uz = np.load(UV95)
    u9, v9 = np.asarray(gz["u9"], complex), np.asarray(gz["v9"], complex)
    u11, v11 = np.asarray(gz["u11"], complex), np.asarray(gz["v11"], complex)
    u95, v95 = np.asarray(uz["u95"], complex), np.asarray(uz["v95"], complex)
    u10, v10 = np.asarray(uz["u10"], complex), np.asarray(uz["v10"], complex)
    prog(f"loaded U,V shapes 9={u9.shape} 9.5={u95.shape} 10={u10.shape} 11={u11.shape}")

    slice_a = slice_A(u9, v9)
    if not (slice_a["sigma_xp_bare_raises"] and slice_a["reference_selftest_ok"]
            and slice_a["interlacing_ok"] and abs(slice_a["legacy_minus_reported"]) < 1e-8):
        raise RuntimeError(f"slice A regression failed: {slice_a}")
    prog("slice A regression ok")

    _saved, cache, _t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    layout, ncol, edge = group_layout(cache, gamma)
    kinds = []
    for rec in layout:
        if rec["kind"] not in kinds:
            kinds.append(rec["kind"])
    gram = json.loads(GJSON.read_text(encoding="utf-8"))
    archived_kinds = list(gram["by_group_self_blocks"].keys())
    if kinds != archived_kinds:
        raise RuntimeError(f"kind order/set changed: {kinds} vs {archived_kinds}")
    for uv in (u9, u95, u10, u11):
        if uv.shape[1] != ncol:
            raise RuntimeError(f"ncol mismatch {uv.shape} vs layout {ncol}")
    prog(f"layout ncol={ncol} n_kind={len(kinds)} ir_edge={edge}")

    windows = {
        "9.0": window_record(u9, v9, layout, kinds),
        "9.5": window_record(u95, v95, layout, kinds),
        "10.0": window_record(u10, v10, layout, kinds),
        "11.0": window_record(u11, v11, layout, kinds),
    }
    # 9 vs 11 self-block change must reuse archived grouping, not recompute vertices.
    d_check = {}
    worst = 0.0
    for kind in kinds:
        dC = cmat(windows["11.0"]["groups"][kind]["C"]) - cmat(windows["9.0"]["groups"][kind]["C"])
        arch = cmat(gram["by_group_self_blocks"][kind]["normal"])
        err = float(np.max(np.abs(dC - arch)))
        worst = max(worst, err)
        d_check[kind] = err
    if worst > 1e-10:
        raise RuntimeError(f"9/11 self-block reuse failed maxerr={worst}")
    prog(f"9/11 self-block reuse maxerr={worst:.3e}")

    table = []
    for cyc, key in (("9.0", "9.0"), ("9.5", "9.5"), ("10.0", "10.0"), ("11.0", "11.0")):
        w = windows[key]
        row = {"cycles": float(cyc), "RN00": w["R_N"]["diag00"], "RN11": w["R_N"]["diag11"],
               "Q0": w["Q0"], "Q1": w["Q1"], "coarse": w["coarse"]}
        table.append(row)

    dW = {}
    for kind in kinds:
        c9 = windows["9.0"]["groups"][kind]["C"]
        c11 = windows["11.0"]["groups"][kind]["C"]
        dW[kind] = {
            "dRN00": c11["diag00"] - c9["diag00"],
            "dRN11": c11["diag11"] - c9["diag11"],
            "dQ0_de_at9": windows["9.0"]["groups"][kind]["dQ0_de"],
            "dQ0_de_at11": windows["11.0"]["groups"][kind]["dQ0_de"],
        }
    coarse_delta = {}
    for ck in windows["9.0"]["coarse"]:
        a, b = windows["9.0"]["coarse"][ck], windows["11.0"]["coarse"][ck]
        coarse_delta[ck] = {
            "dC00_11_minus_9": b["C00"] - a["C00"],
            "dC11_11_minus_9": b["C11"] - a["C11"],
            "dQ0_de_at9": a["dQ0"],
            "dQ0_de_at11": b["dQ0"],
            "n_col": a["n_col"],
        }
    ranked = sorted(coarse_delta.items(), key=lambda kv: abs(kv[1]["dC00_11_minus_9"]), reverse=True)
    dark_ph = {
        "dark_C00_9": windows["9.0"]["coarse"]["dark"]["C00"],
        "kept_C00_9": windows["9.0"]["coarse"]["photon_kept"]["C00"],
        "dark_plus_kept_9": windows["9.0"]["coarse"]["dark"]["C00"] + windows["9.0"]["coarse"]["photon_kept"]["C00"],
        "dark_C00_11": windows["11.0"]["coarse"]["dark"]["C00"],
        "kept_C00_11": windows["11.0"]["coarse"]["photon_kept"]["C00"],
        "dark_plus_kept_11": windows["11.0"]["coarse"]["dark"]["C00"] + windows["11.0"]["coarse"]["photon_kept"]["C00"],
        "note": "large opposite dark/photon self-blocks; cancellation is in C_g not in physical quadrature of dark",
    }

    report = {
        "task": "G1 A+B1/B2 together; B3 mapping only",
        "date": "2026-10-19",
        "production_G1_pass": False,
        "quasi_boson_validity": "pending",
        "no_new_propagation": True,
        "did_not_assemble_9_or_11": True,
        "did_not_start_16_or_20": True,
        "did_not_change_Hamiltonian_P1_phi_eta_gamma_D": True,
        "slice_A": slice_a,
        "slice_B1": {
            "layout_ncol": ncol,
            "n_kind": len(kinds),
            "kinds": kinds,
            "archived_9_11_self_block_maxabs_err": worst,
            "four_window_table": table,
            "window_change_11_minus_9_by_kind_diag": {k: {"dRN00": dW[k]["dRN00"], "dRN11": dW[k]["dRN11"]} for k in kinds},
            "coarse_window_change_11_minus_9": coarse_delta,
            "ranked_coarse_by_abs_dRN00": [k for k, _ in ranked],
            "dark_photon_cancellation": dark_ph,
            "cross_9": windows["9.0"]["cross_between_groups"],
            "cross_11": windows["11.0"]["cross_between_groups"],
            "reconstruction_ok": all(windows[k]["reconstruction_maxabs_self_plus_cross_minus_C"] < 1e-12
                                      for k in windows),
        },
        "slice_B2": {
            "formula": "dR_N/de_g = C_g; dQ_mu/de_g = (n_g)_mumu + (C_g)_mumu/2 - Re[m_mumu*/|m| (m_g)_mumu]",
            "meaning": "artificial weight (1+ε) on the group's already-quadratic contribution / √(1+ε) on columns without inter-group half-cross in the boxed formula",
            "not_a_physical_error_bound": True,
            "dark_scale_is_not_frequency_quadrature": True,
            "U0_kept_inside_each_group": True,
            "windows": {k: {
                "Q0": windows[k]["Q0"],
                "RN00": windows[k]["R_N"]["diag00"],
                "coarse_dQ_de": {ck: windows[k]["coarse"][ck]["dQ0"] for ck in windows[k]["coarse"]},
                "coarse_dRN_de": {ck: windows[k]["coarse"][ck]["C00"] for ck in windows[k]["coarse"]},
                "top_dQ0_kinds": sorted(
                    ((kind, windows[k]["groups"][kind]["dQ0_de"]) for kind in kinds),
                    key=lambda kv: abs(kv[1]), reverse=True)[:8],
                "top_dRN00_kinds": sorted(
                    ((kind, windows[k]["groups"][kind]["dRN_de_diag00"]) for kind in kinds),
                    key=lambda kv: abs(kv[1]), reverse=True)[:8],
            } for k in windows},
        },
        "slice_B3": slice_B3(),
        "not_a_squeezing_or_entanglement_claim": True,
        "elapsed_s": float(elapsed()),
    }
    OUTJSON.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    prog("saved " + str(OUTJSON))
    print("done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

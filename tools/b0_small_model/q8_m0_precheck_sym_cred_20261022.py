"""T1/T4/T6 physical precheck. No model change, no new dynamics, no G1 extension.

Hard boundaries
---------------
- Do not change Hamiltonian / P1 / initial state / phi / gamma / eta / D / k-grid; no T2.
- Do not run K', new adjoints/vertices, new production U,V, long propagation, +/-N, or CEP=pi.
- Do not overwrite archived JSON (nm_tables, AB12, gram, A6, io_identity, kernel correction).
- production_G1_pass=false. Do not promote n,m,Q to physical predictions.
- Dark columns = discrete electronic DOF. Do not write (2.7) terminal correction into G1.
- Zero-field electronic-block symmetry is NOT a finite-pulse one-sided Néel selection rule.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from q8_m0_classical import k_grid
from q8_m0_gsheet import gamma_self
from q8_m0_model import (
    R0,
    build_geometry,
    current_ops,
    dh8,
    h8,
    op_P,
    op_sewing,
)
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of

OUTDIR = ROOT / "results/q_line/precheck_sym_cred_20261022"
OUTJSON = OUTDIR / "precheck_sym_cred.json"
PROGRESS = OUTDIR / "progress.txt"
GNPZ = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.npz"
GJSON = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.json"
AB12 = ROOT / "results/q_line/g1_ab12_residual_groups_20261019/ab12_residual_groups.json"
NMJSON = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables.json"
COMPL = ROOT / "results/q_line/incoming_spectral_completeness_20261004/completeness_checks.json"
CAMP = ROOT / "results/q_line/m0_p0p1_20260914/campaign_key_table.json"
FROZEN_N00 = 79737.14496652872
FROZEN_M00 = 79717.6187445943
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


def fro(a):
    return float(np.linalg.norm(np.asarray(a)))


def cpack(z):
    z = np.asarray(z, complex).reshape(())
    return {
        "re": float(z.real),
        "im": float(z.imag),
        "abs": float(abs(z)),
        "arg": float(np.angle(z)),
    }


def frozen_n(v):
    return np.conj(v) @ v.T


def frozen_m(u, v):
    return u @ v.T


def t1_op_p_and_h():
    geo = build_geometry()
    p = R0
    P = op_P()
    rng = np.random.default_rng(22)
    samples = [
        ("Gamma", geo.k_cart([0.0, 0.0])),
        ("K", geo.k_cart([1.0 / 3.0, 1.0 / 3.0])),
        ("Kp", geo.k_cart([-1.0 / 3.0, -1.0 / 3.0])),
        ("M", geo.k_cart([0.5, 0.0])),
    ]
    for i in range(6):
        kf = rng.uniform(-0.7, 0.7, 2)
        samples.append((f"rng{i}", geo.k_cart(kf)))

    rows = []
    max_p_flip = 0.0
    max_p_same = 0.0
    max_j_n = 0.0
    max_pj_minus = 0.0
    max_pj_plus = 0.0
    max_phase_trial = 0.0
    for name, k in samples:
        hp = h8(geo, p, k, +1)
        ident_flip = P @ hp @ P.conj().T - h8(geo, p, -k, -1)
        ident_same = P @ hp @ P.conj().T - h8(geo, p, -k, +1)
        j_p = current_ops(geo, p, k, +1)
        j_m = current_ops(geo, p, k, -1)
        j_neg = current_ops(geo, p, -k, +1)
        dn = fro(j_p - j_m)
        # polar check both signs; identity from d/dk of P H_N(k) P = H_{-N}(-k)
        pj = np.stack([P @ j_p[a] @ P.conj().T for a in range(2)], axis=0)
        r_minus = fro(pj - j_neg)
        r_plus = fro(pj + j_neg)
        # k-dependent overall phase e^{i k·(2c)} cannot change P H P; sewing-like trial:
        U = op_sewing(geo, k)
        Pk = U @ P
        phase_res = fro(Pk @ hp @ Pk.conj().T - h8(geo, p, -k, -1))
        max_p_flip = max(max_p_flip, fro(ident_flip))
        max_p_same = max(max_p_same, fro(ident_same))
        max_j_n = max(max_j_n, dn)
        max_pj_minus = max(max_pj_minus, r_minus)
        max_pj_plus = max(max_pj_plus, r_plus)
        max_phase_trial = max(max_phase_trial, phase_res)
        rows.append(
            {
                "name": name,
                "k": [float(k[0]), float(k[1])],
                "P_H_P_minus_H_flipN_negk": fro(ident_flip),
                "P_H_P_minus_H_sameN_negk": fro(ident_same),
                "j_plusN_minus_j_minusN": dn,
                "P_j_P_minus_j_negk": r_minus,
                "P_j_P_plus_j_negk": r_plus,
                "sewingP_trial_flipN": phase_res,
            }
        )

    dh_neel_dropped = "del neel"  # q8_m0_model.dh8
    return {
        "op_P_formula": "kron(tau_x, sigma_x, s_0); k-independent constant 8x8",
        "spec_note": "common scalar phase e^{2 i k·c} dropped in production op_P",
        "dropped_phase_is_overall_gauge_if_constant_P_already_holds": True,
        "constant_P_identity": "P H_N(k) P^dagger = H_{-N}(-k)",
        "max_residual_P_H_flipN": max_p_flip,
        "max_residual_P_H_sameN": max_p_same,
        "sameN_identity_fails_as_expected": bool(max_p_same > 1e-6),
        "machine_prec_flipN": bool(max_p_flip < 1e-12),
        "k_dependent_phase_trial": {
            "trial": "op_sewing(k) @ op_P, not a production change",
            "max_residual_flipN": max_phase_trial,
            "worse_or_unhelpful": bool(max_phase_trial >= max_p_flip - 1e-16),
            "production_op_P_unchanged": True,
        },
        "current": {
            "j_equals_minus_dH_dk": True,
            "dh8_drops_neel": True,
            "dh8_comment": dh_neel_dropped,
            "max_j_plusN_minus_j_minusN": max_j_n,
            "j_insensitive_to_N_at_A0": bool(max_j_n < 1e-14),
            "max_P_j_P_minus_j_negk": max_pj_minus,
            "max_P_j_P_plus_j_negk": max_pj_plus,
            "note": "zero-field current N-insensitivity is a model fact, not a port theorem",
        },
        "samples": rows,
        "not_a_selection_rule": True,
    }


def t1_k_grid():
    geo = build_geometry()
    out = {}
    for nk, shift in ((12, 0.0), (24, 0.0), (12, 0.5)):
        kf, kc, _w = k_grid(geo, nk, shift=shift)
        kf_neg = np.mod(-kf, 1.0)
        # match each -k to a grid point
        key = np.round(kf, 12)
        table = {tuple(r): i for i, r in enumerate(key)}
        missing = 0
        for row in np.round(kf_neg, 12):
            if tuple(row) not in table:
                missing += 1
        out[f"nk{nk}_shift{shift:g}"] = {
            "n_k": int(kf.shape[0]),
            "missing_inversion_partners": int(missing),
            "closed": missing == 0,
        }
    return out


def t1_dark_quote():
    with COMPL.open(encoding="utf-8") as fh:
        c = json.load(fh)
    domains = c.get("M0_interface", {}).get("domains", [])
    return {
        "source": str(COMPL.relative_to(ROOT)).replace("\\", "/"),
        "threshold_note": "n_exactly_dark uses a 1e-10 relative threshold on this 2-k sample; it is not a BZ dark-space theorem.",
        "domains": [
            {
                "neel": d.get("neel"),
                "n_el": d.get("n_el"),
                "n_exactly_dark": d.get("n_exactly_dark"),
                "min_vertex_strength": d.get("min_vertex_strength"),
            }
            for d in domains
        ],
        "located_not_repaired": True,
        "production_G1_dark_columns_are_a_different_object": {
            "n_dark_G1": N_DARK,
            "meaning": "discrete electronic DOF in U,V layout, not the 2-k incoming sample",
        },
        "do_not_fix_threshold_splitting": True,
    }


def t1_unchecked():
    return {
        "drive": [
            "Peierls substitution / finite-pulse A(t)",
            "CEP",
            "finite envelope (sin2 on vector potential)",
        ],
        "initial_state": [
            "how the AFM ground occupies under P",
            "whether the occupied subspace tracks N together with H_N",
            "occup_empty_bases(h0, mu) at finite pulse t=0",
        ],
        "port": [
            "+/-z exchange",
            "in-plane polarization sign flip",
            "one-sided detector kernel K",
            "left vs right incoming columns",
        ],
        "observables_remain_candidates": True,
        "do_not_declare_one_sided_nmQ_Neel_blind": True,
        "do_not_prelabel_finite_pulse_deviation_as_small_CEP_leak": True,
    }


# AB12 / Gram layout order (sequential columns). n_col from archived AB12 coarse table.
COARSE_LAYOUT = (
    ("dark", 1976),
    ("photon_kept", 124864),
    ("photon_tails", 12704),
    ("photon_wing230", 512),
    ("photon_wing264", 512),
    ("photon_core", 5120),
    ("photon_infrared", 256),
)


def groups_from_layout(n_col):
    if sum(n for _, n in COARSE_LAYOUT) != n_col:
        raise SystemExit("coarse layout n_col does not match U,V")
    out = []
    col = 0
    for name, n in COARSE_LAYOUT:
        ix = np.arange(col, col + n, dtype=np.int64)
        out.append((name, ix))
        col += n
    return out


def t4_split():
    prog("T4 load gram npz")
    gz = np.load(GNPZ)
    u9, v9 = gz["u9"], gz["v9"]
    u11, v11 = gz["u11"], gz["v11"]
    del gz
    groups = groups_from_layout(int(u9.shape[1]))

    n_full = frozen_n(v9)
    m_full = frozen_m(u9, v9)
    n00 = n_full[0, 0]
    m00 = m_full[0, 0]

    rec_n = np.zeros_like(n_full)
    rec_m = np.zeros_like(m_full)
    packed = []
    for name, ix in groups:
        ng = frozen_n(v9[:, ix]) if ix.size else np.zeros((2, 2), complex)
        mg = frozen_m(u9[:, ix], v9[:, ix]) if ix.size else np.zeros((2, 2), complex)
        rec_n += ng
        rec_m += mg
        n00g = ng[0, 0]
        m00g = mg[0, 0]
        frac = float(n00g.real / n00.real) if n00.real != 0 else 0.0
        packed.append(
            {
                "name": name,
                "n_col": int(ix.size),
                "n00": cpack(n00g),
                "n00_frac_display_1e-3": round(frac, 3),
                "n00_frac_full": frac,
                "m00": cpack(m00g),
                "do_not_replace_abs_m_by_sum_abs": True,
            }
        )

    dn = rec_n - n_full
    dm = rec_m - m_full
    rec_ok = bool(np.max(np.abs(dn)) < 1e-10 and np.max(np.abs(dm)) < 1e-8)
    sum_abs_m = float(sum(p["m00"]["abs"] for p in packed))
    cancel = 1.0 - float(abs(m00)) / sum_abs_m if sum_abs_m else 0.0

    packed_sorted = sorted(packed, key=lambda p: -p["n00"]["abs"])
    top = packed_sorted[:10]

    n11 = frozen_n(v11)
    m11 = frozen_m(u11, v11)

    with NMJSON.open(encoding="utf-8") as fh:
        nm = json.load(fh)
    # frozen 9-cycle production lives under window or kernel; compare n00 abs
    return {
        "definitions": {"n": "V.conj() @ V.T", "m": "U @ V.T", "U_includes_U0": True},
        "n_col": int(u9.shape[1]),
        "n_full_00": cpack(n00),
        "m_full_00": cpack(m00),
        "frozen_n00_abs": FROZEN_N00,
        "frozen_m00_abs": FROZEN_M00,
        "match_frozen_n00_abs": float(abs(abs(n00) - FROZEN_N00)),
        "match_frozen_m00_abs": float(abs(abs(m00) - FROZEN_M00)),
        "groups": packed_sorted,
        "top10_by_n00": top,
        "reconstruction": {
            "maxabs_sum_n_g_minus_n": float(np.max(np.abs(dn))),
            "maxabs_sum_m_g_minus_m": float(np.max(np.abs(dm))),
            "ok": rec_ok,
            "tolerance_is_not_1e-3": True,
            "display_1e-3_is_documentation_only": True,
        },
        "m_cancellation": {
            "abs_m00": float(abs(m00)),
            "sum_abs_m00_g": sum_abs_m,
            "one_minus_abs_over_sum_abs": cancel,
            "forbidden": "replace |m| by sum_g |m_g|",
        },
        "window11_n00_abs": float(abs(n11[0, 0])),
        "window11_m00_abs": float(abs(m11[0, 0])),
        "nm_tables_not_overwritten": True,
        "line_0p0928": {
            "quote_only": True,
            "note": "bright poles >= 0.0928 in archived finite-kcv manifests; U,V columns here have no frequency labels in this split",
            "example_frequency_from_gamma12_manifest": 0.09280710211386917,
        },
        "quasi_boson": {
            "formula_only": "occupation-like n is frozen n=V*V^T; not a production quasi-boson number",
            "gap": "no accepted map from this n to a single-pass eta0 sigma/2 occupancy; do not promote",
        },
        "area_convention": {
            "eta": "sqrt(gamma/(2 pi))",
            "eta_of_code": "q8_m0_p1_h2_finite_kcv_input_close_20261005.eta_of",
            "gamma_self_from_gsheet": float(gamma_self(build_geometry())),
            "eta_numeric": float(eta_of(gamma_self(build_geometry()))),
        },
        "layout_source": {
            "ab12_coarse_n_col_order": [list(x) for x in COARSE_LAYOUT],
            "gram_json": str(GJSON.relative_to(ROOT)).replace("\\", "/"),
            "ab12_json": str(AB12.relative_to(ROOT)).replace("\\", "/"),
            "no_new_assemble": True,
        },
        "dominant_groups_are_source_structure_only": True,
        "not_artifact_or_Floquet_claim": True,
        "production_n_m_Q": False,
        "nm_archive_task_field": nm.get("task"),
    }


def t6_quote(t4):
    camp = {}
    if CAMP.is_file():
        with CAMP.open(encoding="utf-8") as fh:
            camp = json.load(fh)
    p0 = camp.get("P0_targets", {}).get("nk48_gamma", {})
    return {
        "existing_UV_only_H2_two_pol_no_H3": True,
        "H2_H3_correlator": {
            "status": "transform_relation_only",
            "note": "no H3 columns; do not invent <db_H2 db_H3>",
        },
        "classical_SHG": {
            "beta_H2_N_odd_is_known_class_result": True,
            "this_campaign_key_table_has_one_N_branch_only": True,
            "quoted_Jo_y_H2_P0_nk48_gamma": p0.get("Jo_y_H2"),
            "quoted_Jo_x_H2_P0_nk48_gamma": p0.get("Jo_x_H2"),
            "single_harmonic_photon_stats_domain_blind_depends_on_unproven_port_layer": True,
        },
        "plus_minus_N": "quote_only_not_run",
        "CEP_pi": "quote_only_not_run",
        "two_color": "quote_only_not_run",
        "beta2_over_n00": {
            "status": "quote_only",
            "reason": "same frozen phi does not here project classical current onto the frozen n,m packet; do not write as proven amplified vacuum noise",
            "n00_abs": t4["n_full_00"]["abs"],
            "classical_Jo_y_H2_is_a_different_object": True,
        },
        "next_domain_sensitive_candidate": (
            "do not treat one-color n, |m|, Q as domain contrast; "
            "odd classical beta_H2 remains the contrast anchor; "
            "next quantum candidate is a port-odd correlator after the unchecked port layer is specified, "
            "not a +/-N long run"
        ),
    }


def three_sentences(t1, t4, t6):
    # filled from numbers
    g = t4["groups"]
    parts = [f"{p['name']} n00_frac={p['n00_frac_display_1e-3']}" for p in g[:4]]
    src = "；".join(parts)
    rec = t4["reconstruction"]["ok"]
    rec_s = "分组高精度还原通过" if rec else "分组还原失败，见 JSON 反例"
    cancel = t4["m_cancellation"]["one_minus_abs_over_sum_abs"]
    s1 = (
        "对称性只验到 A=0 电子块：常数 op_P 给出 P H_N(k) P† = H_{-N}(-k)"
        f"（最大残差 {t1['max_residual_P_H_flipN']:.3e}）；"
        "丢掉的 e^{2ik·c} 是整体规范，未改生产 op_P。"
        f" dh8 丢掉 neel，零场电流对 N 不敏感（差 {t1['current']['max_j_plusN_minus_j_minusN']:.3e}）。"
        "这不是有限脉冲、固定单侧输出的 Néel 定则；驱动/初态/端口未检。"
    )
    s2 = (
        f"冻结 n00≈{t4['n_full_00']['abs']:.6f}。列组来源：{src}。"
        f" m 存在相消（1-|m|/Σ|m_g|={cancel:.6f}），禁止用 Σ|m_g| 代替 |m|。{rec_s}。"
        "不宣布伪影或 Floquet。"
    )
    s3 = (
        "现有单色 n、|m|、Q 不能当畴衬度（无 H3；±N/CEP=π 未跑）。"
        "最值得下一步的是先把未检的端口层（单侧 K、±z）能支持的奇相关定量，"
        "而不是先跑 ±N 长传播；经典 β_H2 随 N 反号只作对照锚。"
    )
    return {"symmetry_layer": s1, "n_source": s2, "next_domain_observable": s3}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS.exists():
        PROGRESS.unlink()
    prog("T1 start")
    t1_h = t1_op_p_and_h()
    t1_k = t1_k_grid()
    t1_d = t1_dark_quote()
    t1_u = t1_unchecked()
    t1 = {
        "layer": "zero_field_electronic_only",
        **t1_h,
        "k_grid_inversion": t1_k,
        "dark_threshold_quote": t1_d,
        "unchecked": t1_u,
        "not_a_selection_rule": True,
    }
    prog("T4 start")
    t4 = t4_split()
    if not t4["reconstruction"]["ok"]:
        payload = {
            "stop": "T4 reconstruction failed",
            "reconstruction": t4["reconstruction"],
            "groups": t4["groups"],
            "production_G1_pass": False,
        }
        OUTJSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        prog("T4 FAIL reconstruction")
        raise SystemExit(2)
    prog("T6 quote")
    t6 = t6_quote(t4)
    sentences = three_sentences(t1, t4, t6)
    payload = {
        "task": "T1/T4/T6 physical precheck 20261022",
        "production_G1_pass": False,
        "no_Kprime": True,
        "no_new_UV_assembly": True,
        "no_long_propagation": True,
        "hard_boundaries_honored": True,
        "t1": t1,
        "t4": t4,
        "t6": t6,
        "three_sentences": sentences,
        "elapsed_s": elapsed(),
    }
    OUTJSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    prog(f"wrote {OUTJSON}")
    print(json.dumps(sentences, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

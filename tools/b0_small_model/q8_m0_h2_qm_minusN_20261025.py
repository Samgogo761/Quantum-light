"""Exploratory -N H2 two-pol corresponding branch. 9-cycle @ macro1024.

Rebuild electronic incoming columns, vertices, adjoint rows, then U,V.
Do not flip +N columns or adjoint rows. Optical protocol is read-only.
production_G1_pass=false. Stop after this job.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
import q_cov4_reference_check as cov4
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_boundary_sensitivity_20261015 import (
    contract_stacked, frequencies, stack_forces,
)
from q8_m0_g1_gamma12_ir_tail_20261015 import rows_from_vertices, vertices_progress
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_incoming_spectral_completeness_20261004 import (
    dark_subspace, stationary_dark_basis_columns,
)
from q8_m0_k1_packet import normalize_phi, omega_grid, phi_unnormalized
from q8_m0_model import build_geometry
from q8_m0_p1_h2_finite_kcv_bz_access_20261005 import grid_sample
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of

ROOT = Path(__file__).resolve().parents[2]
PLUSN = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007"
PLUS_UV = ROOT / "results/q_line/g1_kernel_correction_20261017/kernel_correction_9cycles.npz"
CLASS_JSON = ROOT / "results/q_line/h2_class_pmN_run_20261025/h2_class_pmN.json"
SEL_JSON = ROOT / "results/q_line/h2_sel_rule_20261025/h2_sel_rule.json"
COST_JSON = ROOT / "results/q_line/h2_tranche2_cost_20261025/h2_tranche2_cost.json"
OUTDIR = ROOT / "results/q_line/h2_qm_minusN_20261025"
OUTJSON = OUTDIR / "h2_qm_minusN.json"
OUTNPZ = OUTDIR / "h2_qm_minusN_uv.npz"
ROWSNPZ = OUTDIR / "adjoint_rows_minusN_macro1024.npz"
PROGRESS = OUTDIR / "progress.txt"

CYCLES = 9
NTPC = 1024
NEEL = -1
BUDGET_S = 5400
TH0_PLUS = 3.097915276657095
TH1_PLUS = 1.3448857412173498
D_REF = np.diag([1.0, -1.0])
T0 = time.time()

DO_NOT_TOUCH = (
    PLUSN,
    PLUS_UV,
    CLASS_JSON,
    SEL_JSON,
    COST_JSON,
    ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables.json",
    ROOT / "results/q_line/g1_q_cov4_fix_20261018/q_cov4_fix.json",
    ROOT / "results/q_line/g1_gamma12_time_order_20261015/adjoint_rows_macro1024.npz",
    ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.npz",
)


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def budget_left():
    if elapsed() > BUDGET_S:
        raise TimeoutError(f"wall-clock budget {BUDGET_S}s exceeded")


def pack(m):
    m = np.asarray(m, complex)
    return {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "abs": np.abs(m).tolist(),
        "maxabs": float(np.max(np.abs(m))) if m.size else 0.0,
    }


def cnum(z):
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)), "arg": float(np.angle(z))}


def abort(status, detail):
    report = {
        "task": "exploratory quantum -N H2 two-pol corresponding branch",
        "date": "2026-10-25",
        "status": status,
        "detail": detail,
        "elapsed_s": float(elapsed()),
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "no_Kprime": True,
        "did_not_overwrite_plusN": True,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    OUTJSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    prog(f"ABORT {status}: {detail}")
    raise SystemExit(2)


def load_plus_optical():
    manifest = json.loads((PLUSN / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["physical_scope"]["neel"]) != 1:
        abort("plusN_neel_not_plus", str(manifest["physical_scope"]["neel"]))
    cache = {}
    for name in ("gaps", "u0", "D0", "gamma", "pol"):
        rec = manifest["files"][name]
        cache[name] = np.load(PLUSN / rec["path"], allow_pickle=False)
    return cache, manifest


def cxx_nm(n, m, th0, th1):
    return float(np.real(np.exp(-1j * (th0 + th1)) * m[0, 1] + np.exp(1j * (th0 - th1)) * n[0, 1]))


def cxx_sigma(sig, th0, th1):
    c0, s0 = np.cos(th0), np.sin(th0)
    c1, s1 = np.cos(th1), np.sin(th1)
    return float(c0 * c1 * sig[0, 1] + c0 * s1 * sig[0, 3] + s0 * c1 * sig[2, 1] + s0 * s1 * sig[2, 3])


def nm_of(u, v):
    n = np.conjugate(v) @ v.T
    m = u @ v.T
    c = u @ u.conj().T - v @ v.conj().T
    ra = u @ v.T - v @ u.T
    return n, m, c, ra


def save_rows(path, row_d, force_d, row_u, force_u):
    if path.exists():
        raise FileExistsError(path)
    payload = {}
    for tag, row, forces in (("d", row_d, force_d), ("u", row_u, force_u)):
        payload[f"row_{tag}"] = row
        payload[f"force_{tag}"] = np.stack([f for f, _t, _h in forces])
        payload[f"time_{tag}"] = np.array([t for _f, t, _h in forces], float)
        payload[f"step_{tag}"] = np.array([h for _f, _t, h in forces], float)
    np.savez(path, **payload)


def load_rows_cache(path):
    with np.load(path, allow_pickle=False) as z:
        def one(tag):
            forces = list(zip(z[f"force_{tag}"], z[f"time_{tag}"], z[f"step_{tag}"]))
            return np.array(z[f"row_{tag}"]), forces
        return one("d"), one("u")


def assemble(rd, fd, ru, fu, dark, minus_cache, gamma, pol, phi_of, eta, pieces):
    sdp, sup = stack_forces(fd), stack_forces(fu)
    b, bd, a = dark.b, dark.bdag, dark.A
    dd, vd = contract_initial(rd, b, bd, a)
    du, vu = contract_initial(ru, b, bd, a)
    u_blocks = [dd - du]
    v_blocks = [vd - vu]
    u0_blocks = [np.zeros((2, b.shape[1]), complex)]
    groups = [f"dark_{b.shape[1]}"]
    n_piece = 0
    for name, om, wt in pieces:
        for start in range(0, len(om), 400):
            budget_left()
            stop = min(start + 400, len(om))
            ph, u0 = base.generated_photon_block(
                minus_cache, gamma, om[start:stop], wt[start:stop], pol, phi_of, eta)
            d = contract_stacked(rd, sdp, ph.initial.b, ph.initial.bdag,
                                 ph.initial.A, ph.free_A0, ph.omega)
            u = contract_stacked(ru, sup, ph.initial.b, ph.initial.bdag,
                                 ph.initial.A, ph.free_A0, ph.omega)
            u_blocks.append(d[0] - u[0])
            v_blocks.append(d[1] - u[1])
            u0_blocks.append(u0)
            groups.append(f"{name}[{start}:{stop}]")
            n_piece += 1
            if n_piece % 8 == 0:
                prog(f"assemble piece {n_piece} {name}[{start}:{stop}]")
            del ph
    u0c = np.concatenate(u0_blocks, axis=1)
    return np.concatenate(u_blocks, axis=1) + u0c, np.concatenate(v_blocks, axis=1), groups


def compare_pair(n_m, m_m, n_p, m_p, sig_m, sig_p):
    n_ref = D_REF.conj() @ n_p @ D_REF.T
    m_ref = D_REF @ m_p @ D_REF.T
    lab = {}
    mir = {}
    for key, a, b in (("n", n_m, n_p), ("m", m_m, m_p)):
        dn = a - b
        lab[key] = {ij: cnum(dn[i, j]) for ij, (i, j) in (
            ("00", (0, 0)), ("11", (1, 1)), ("01", (0, 1)), ("10", (1, 0)))}
        lab[f"{key}_rel_to_plus_abs"] = {
            ij: float(abs(dn[i, j]) / max(abs(b[i, j]), 1e-30))
            for ij, (i, j) in (("00", (0, 0)), ("11", (1, 1)), ("01", (0, 1)))}
    for key, a, ref, plus in (("n", n_m, n_ref, n_p), ("m", m_m, m_ref, m_p)):
        dr = a - ref
        mir[key] = {ij: cnum(dr[i, j]) for ij, (i, j) in (
            ("00", (0, 0)), ("11", (1, 1)), ("01", (0, 1)), ("10", (1, 0)))}
        mir[f"{key}_rel_to_plus_abs"] = {
            ij: float(abs(dr[i, j]) / max(abs(plus[i, j]), 1e-30))
            for ij, (i, j) in (("00", (0, 0)), ("11", (1, 1)), ("01", (0, 1)))}
    angles = {"zero": (0.0, 0.0), "plusN_theta_min": (TH0_PLUS, TH1_PLUS)}
    cxx = {}
    for name, (th0, th1) in angles.items():
        cm, cp = cxx_nm(n_m, m_m, th0, th1), cxx_nm(n_p, m_p, th0, th1)
        sm, sp = cxx_sigma(sig_m, th0, th1), cxx_sigma(sig_p, th0, th1)
        cxx[name] = {
            "theta": [th0, th1],
            "C_XX_nm_minus": cm, "C_XX_nm_plus": cp, "Delta_C_XX_nm": cm - cp,
            "C_XX_sigma_minus": sm, "C_XX_sigma_plus": sp, "Delta_C_XX_sigma": sm - sp,
        }
    return {"D_is_reference_not_theorem": True, "lab_fixed": lab, "mirror_reference": mir, "C_XX": cxx}


def unresolved_flags(comp, n_m, n_p):
    dn00 = abs(n_m[0, 0] - n_p[0, 0])
    timestep = 6.94778067698
    rn00 = 1.0489020496606827e-06
    kernel = 5.6367e-07
    ab12 = 2.2e-05
    a6 = 6.013e-06
    qswing = 0.016837950330227613
    return {
        "do_not_merge_into_one_bound": True,
        "A6_is_object_gap_not_this_error_bar": True,
        "n00_abs_Delta_lab": float(dn00),
        "timestep_abs_dn00": timestep,
        "n00_unresolved_vs_timestep": bool(dn00 <= timestep),
        "RN00_9cycle": rn00,
        "kernel_dRN00_9cycle": kernel,
        "AB12_net_remainder_order": ab12,
        "A6_raw2048_minus_S_omega0p11": a6,
        "Q0_four_window_swing": qswing,
        "Q0_swing_label": "conditional_error_estimate_not_strict_bound",
        "mirror_n00_unresolved_vs_timestep": bool(
            abs(complex(comp["mirror_reference"]["n"]["00"]["re"],
                        comp["mirror_reference"]["n"]["00"]["im"])) <= timestep),
        "note": "flags compare the same quantity to one archived diagnostic; not a joint bound",
    }


def classical_readonly():
    cls = json.loads(CLASS_JSON.read_text(encoding="utf-8"))
    sel = json.loads(SEL_JSON.read_text(encoding="utf-8"))
    return {
        "not_photon_number": True,
        "not_lab_intensity": True,
        "beta_src_plus": cls["beta_src_plus"],
        "beta_src_minus": cls["beta_src_minus"],
        "beta_src_even_odd": cls.get("beta_src_even_odd"),
        "stokes_baseline": sel["stokes_baseline"],
        "I45_contrast_vs_mean": sel["stokes_baseline"]["I45_contrast_vs_mean"],
        "S2_over_S0_plus": sel["stokes_baseline"]["S2_over_S0_plus"],
        "S2_over_S0_minus": sel["stokes_baseline"]["S2_over_S0_minus"],
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OUTJSON.exists() or OUTNPZ.exists():
        raise FileExistsError(OUTJSON)
    for p in DO_NOT_TOUCH:
        if not p.exists():
            abort("missing_readonly_archive", str(p))
    plus_mtime = {str(p): p.stat().st_mtime_ns for p in DO_NOT_TOUCH}

    prog("start exploratory -N; neel=-1; 9-cycle@1024; do not flip +N")
    plus_cache, manifest = load_plus_optical()
    gamma = float(np.asarray(plus_cache["gamma"]))
    pol = np.array(plus_cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    quiet = p1_a2_protocol(0.0)
    eta = eta_of(gamma)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    h = float(prot.period) / NTPC
    n_macro = CYCLES * NTPC
    prog(f"optical frozen gamma={gamma:.12g} eta={eta:.12g} h={h:.12g} n_macro={n_macro}")

    geo = build_geometry()
    sample, inv = grid_sample(geo, 12, 0.0, neel=NEEL)
    if int(sample["neel"]) != NEEL:
        abort("sample_neel_not_minus", str(sample["neel"]))
    minus_cache = {
        "gaps": np.asarray(sample["gap"]),
        "u0": np.asarray(sample["u"]),
        "D0": np.asarray(sample["D"]),
        "gamma": np.asarray(gamma),
        "pol": pol,
    }
    prog(f"sample n_el={sample['n_el']} n_k={sample['n_k']} min_gap={sample['min_direct_gap']}")

    try:
        basis, info, near = dark_subspace(sample["gap"], sample["u"])
        dark, _labels, dark_gaps = stationary_dark_basis_columns(
            sample["gap"], sample["u"], basis, info)
    except ValueError as exc:
        abort("need_new_completeness_fixture", str(exc))
    ndark = int(dark.b.shape[1])
    nel = int(dark.b.shape[0])
    leftover = float(np.max(np.abs(sample["u"].conj() @ basis))) if basis.size else 0.0
    prog(f"dark ndark={ndark} nel={nel} leftover={leftover:.3e} near={len(near)}")
    if leftover > 1e-10:
        abort("need_new_completeness_fixture", f"dark leftover {leftover}")

    pieces, edge = frequencies(plus_cache, gamma)
    n_freq_cols = int(sum(4 * len(om) for _n, om, _w in pieces))
    prog(f"optical pieces={len(pieces)} photon_cols={n_freq_cols} edge={edge}")

    if ROWSNPZ.exists():
        prog("load existing -N adjoint rows cache")
        (row_d, force_d), (row_u, force_u) = load_rows_cache(ROWSNPZ)
    else:
        rows = {}
        for drive, protocol in (("driven", prot), ("undriven", quiet)):
            budget_left()
            prog(f"vertices {drive}")
            coarse = vertices_progress(sample, geo, protocol, n_macro, h, f"coarse {drive}")
            fine = vertices_progress(sample, geo, protocol, 2 * n_macro, 0.5 * h, f"fine {drive}")
            prog(f"vertices done {drive}")
            budget_left()
            prog(f"rows {drive}")
            row, force, tstamp, step = rows_from_vertices(coarse, fine, h, gamma, pol)
            rows[drive] = (row, list(zip(force, tstamp, step)))
            del coarse, fine, force, tstamp, step
            gc.collect()
        row_d, force_d = rows["driven"]
        row_u, force_u = rows["undriven"]
        save_rows(ROWSNPZ, row_d, force_d, row_u, force_u)
        prog("rows cached")
        del rows
        gc.collect()

    if row_d.shape[1] != 2 * nel + 2:
        abort("row_nel_mismatch", f"row {row_d.shape} vs nel {nel}")

    prog("assemble U,V")
    u, v, groups = assemble(
        row_d, force_d, row_u, force_u, dark, minus_cache, gamma, pol, phi_of, eta, pieces)
    prog(f"assembled ncol={u.shape[1]} groups={len(groups)}")
    del row_d, force_d, row_u, force_u, dark, basis
    gc.collect()

    n_m, m_m, c_m, ra_m = nm_of(u, v)
    rn_m = c_m - np.eye(2)
    cov_m = cov4.analyze_uv(u, v, verbose=False)
    sig_m = cov4.sigma_from_nm(n_m, m_m, c_m)

    plus_uv = np.load(PLUS_UV)
    u_p, v_p = np.asarray(plus_uv["u_prod"]), np.asarray(plus_uv["v_prod"])
    n_p, m_p, c_p, ra_p = nm_of(u_p, v_p)
    cov_p = cov4.analyze_uv(u_p, v_p, verbose=False)
    sig_p = cov4.sigma_from_nm(n_p, m_p, c_p)
    del plus_uv

    comp = compare_pair(n_m, m_m, n_p, m_p, sig_m, sig_p)
    flags = unresolved_flags(comp, n_m, n_p)
    cls = classical_readonly()

    sit_a = abs(m_m[0, 1] + m_p[0, 1]) / max(abs(m_p[0, 1]), 1e-30)
    sit_n00_even = abs(n_m[0, 0] - n_p[0, 0]) / max(abs(n_p[0, 0]), 1e-30)
    report = {
        "task": "exploratory quantum -N H2 two-pol corresponding branch",
        "date": "2026-10-25",
        "status": "completed",
        "production_G1_pass": False,
        "quasi_boson": "pending",
        "g1_frozen": True,
        "no_Kprime": True,
        "no_H3": True,
        "no_two_color": True,
        "no_denser_k": True,
        "not_production_nmV": True,
        "not_nonclassical_entanglement_advantage": True,
        "did_not_flip_plusN_columns": True,
        "did_not_overwrite_plusN": True,
        "protocol": {
            "drive": "p1_a2_protocol(0.8)",
            "A_star": 0.8,
            "nk": 12,
            "grid_shift": 0.0,
            "nt_per_cycle": NTPC,
            "n_drive_cycles": 8,
            "n_tail_cycles": 1,
            "window_cycles": CYCLES,
            "neel": NEEL,
            "gamma": gamma,
            "eta": float(eta),
            "map": "U=U0+raw_driven-raw_undriven; V=raw_driven-raw_undriven",
            "plusN_object": str(PLUS_UV),
            "plusN_manifest_neel": int(manifest["physical_scope"]["neel"]),
        },
        "incoming": {
            "n_el": nel,
            "n_dark": ndark,
            "n_photon_cols": n_freq_cols,
            "ncol": int(u.shape[1]),
            "ncol_plusN": int(u_p.shape[1]),
            "dark_leftover": leftover,
            "near_degenerate_pair_count": int(len(near)),
            "inversion": inv,
            "optical_pieces_from_plusN_oms_wts": True,
            "electronic_columns_from_minusN_sample": True,
            "n_groups": len(groups),
        },
        "n_minus": pack(n_m),
        "m_minus": pack(m_m),
        "C_minus": pack(c_m),
        "R_N_minus": pack(rn_m),
        "R_A_minus": pack(ra_m),
        "n_plus": pack(n_p),
        "m_plus": pack(m_p),
        "cov4_minus": {
            "Q": cov_m["Q"],
            "eig_sigma": cov_m["eig_sigma"],
            "interlacing_ok": cov_m["interlacing_ok"],
            "R_N_max": cov_m["R_N_max"],
            "R_A_max": cov_m.get("R_A_max"),
            "max_abs_sigma_nm_minus_S": cov_m.get("max_abs_sigma_nm_minus_S"),
            "max_abs_Omega_eff_minus_Omega0": cov_m["max_abs_Omega_eff_minus_Omega0"],
            "symplectic_Omega0": cov_m["symplectic_Omega0"],
            "lambda_min_sigma": cov_m["eig_sigma"][0],
            "lambda_min_sigma_minus_I_over_2": float(cov_m["eig_sigma"][0] - 0.5),
        },
        "cov4_plus": {
            "Q": cov_p["Q"],
            "eig_sigma": cov_p["eig_sigma"],
            "lambda_min_sigma": cov_p["eig_sigma"][0],
            "lambda_min_sigma_minus_I_over_2": float(cov_p["eig_sigma"][0] - 0.5),
        },
        "comparisons": comp,
        "uncertainty": flags,
        "classical_baseline_readonly": cls,
        "situation_metrics": {
            "m01_minus_plus_plus_rel": float(sit_a),
            "n00_minus_minus_plus_rel": float(sit_n00_even),
            "situation_A_is_approx_odd_m01": "m01(-N)+m01(+N) small relative to |m01(+N)|",
            "do_not_call_sign_flip_a_discovery": True,
        },
        "elapsed_s": float(elapsed()),
        "plusN_mtime_ns_unchanged": {
            str(p): plus_mtime[str(p)] == p.stat().st_mtime_ns for p in DO_NOT_TOUCH
        },
        "stop": {
            "now": "completed; do not start K-prime / G1 / H3 / two-color / denser k",
            "not_certified_physics": True,
        },
    }
    still_ok = all(plus_mtime[str(p)] == p.stat().st_mtime_ns for p in DO_NOT_TOUCH)
    report["plusN_mtime_ns_unchanged"] = {
        str(p): plus_mtime[str(p)] == p.stat().st_mtime_ns for p in DO_NOT_TOUCH
    }
    if not still_ok:
        report["status"] = "completed_but_plusN_mtime_changed"
    OUTJSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUTNPZ, u=u, v=v, n=n_m, m=m_m, C=c_m, R_N=rn_m, R_A=ra_m)
    prog(f"saved {OUTJSON.name} ncol={u.shape[1]} n00={abs(n_m[0,0]):.6g} m01={abs(m_m[0,1]):.6g}")
    prog("STOP")


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        abort("budget_exceeded", str(exc))
    except SystemExit:
        raise
    except FileExistsError:
        raise
    except Exception as exc:
        abort("failed", f"{type(exc).__name__}: {exc}")

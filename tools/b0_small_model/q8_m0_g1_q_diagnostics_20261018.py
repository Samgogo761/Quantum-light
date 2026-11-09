"""G1 signed-Q diagnostics from existing n,m JSON. No new propagation.

Definitions (frozen):
    n = V* V^T
    m = U V^T          (U already includes U0)
    Q_mu = n_mumu + (1 + R_N,mumu)/2 - |m_mumu|
    Q_half = n_mumu + 1/2 - |m_mumu|     # R_N/2 ~ 1e-6, kept as side note
    X(theta) = (a e^{-i theta} + a^dagger e^{i theta}) / sqrt(2)
    Var[X(theta)] = n + (1+R_N)/2 + Re[m e^{-2i theta}]
    theta_min = (arg(m) + pi) / 2   (mod pi)   # minimum-noise direction
    theta_max = arg(m) / 2          (mod pi)   # maximum-noise direction
    dm_parallel = Re(dm * conj(m) / |m|)
    dm_perp     = Im(dm * conj(m) / |m|)

Diagnostic only. production_G1_pass = false.
Does not overwrite archived JSON. Does not promote Q to a G1 pass.
Does not announce squeezing / entanglement.
Q definition is unchanged; only the min/max phase labels are corrected.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_q_diagnostics_20261018"
OUTJSON = OUTDIR / "q_signed_from_existing_nm.json"
PHASEJSON = OUTDIR / "q_signed_from_existing_nm_phase_corrected.json"
PROGRESS = OUTDIR / "progress.txt"

NM = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables.json"
NM_TS = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables_with_timestep.json"
GRAM = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.json"
BUDGET = ROOT / "results/q_line/g1_nm_tables_20261017/error_budget_existing.json"
KJSON = ROOT / "results/q_line/g1_kernel_correction_20261017/kernel_correction_9cycles.json"

T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmat(pack):
    return np.asarray(pack["real"], float) + 1j * np.asarray(pack["imag"], float)


def pack_c(z):
    z = np.asarray(z, complex)
    return {
        "real": np.real(z).tolist(),
        "imag": np.imag(z).tolist(),
        "abs": np.abs(z).tolist(),
        "phase": np.angle(z).tolist(),
        "maxabs": float(np.max(np.abs(z))),
    }


def pack_rn(pack):
    z = cmat(pack)
    return {
        "real": np.real(z).tolist(),
        "imag": np.imag(z).tolist(),
        "maxabs": float(np.max(np.abs(z))),
        "diag00": complex(z[0, 0]).real,
        "diag11": complex(z[1, 1]).real,
    }


def wrap_mod_pi(theta):
    t = float(np.mod(theta, np.pi))
    return t if t >= 0.0 else t + np.pi


def var_X_theta(n_ii, m_ii, rn_ii, theta):
    n_ii = float(np.real(n_ii))
    rn_ii = float(np.real(rn_ii))
    return n_ii + 0.5 * (1.0 + rn_ii) + float(np.real(m_ii * np.exp(-2.0j * theta)))


def q_mu(n_ii, m_ii, rn_ii):
    n_ii = float(np.real(n_ii))
    abs_m = float(abs(m_ii))
    rn_ii = float(np.real(rn_ii))
    q = n_ii + 0.5 * (1.0 + rn_ii) - abs_m
    q_half = n_ii + 0.5 - abs_m
    arg_m = float(np.angle(m_ii))
    theta_max = 0.5 * arg_m
    theta_min = 0.5 * (arg_m + np.pi)
    v_min = var_X_theta(n_ii, m_ii, rn_ii, theta_min)
    v_max = var_X_theta(n_ii, m_ii, rn_ii, theta_max)
    return {
        "n": n_ii,
        "abs_m": abs_m,
        "n_minus_abs_m": n_ii - abs_m,
        "R_N": rn_ii,
        "Q": q,
        "Q_n_plus_half_minus_abs_m": q_half,
        "Q_minus_half_form": q - q_half,
        "arg_m": arg_m,
        "theta_max": theta_max,
        "theta_min": theta_min,
        "theta_max_mod_pi": wrap_mod_pi(theta_max),
        "theta_min_mod_pi": wrap_mod_pi(theta_min),
        "var_X_theta_min": v_min,
        "var_X_theta_max": v_max,
        "var_min_minus_Q": v_min - q,
        "var_max_minus_n_half_plus_abs_m": v_max - (n_ii + 0.5 * (1.0 + rn_ii) + abs_m),
        "theta_half_arg_m_is_MAX_not_min": True,
    }


def dm_split(m_ref, m_new):
    dm = m_new - m_ref
    abs_m = abs(m_ref)
    if abs_m == 0.0:
        return {"dm": dm, "parallel": float("nan"), "perp": float("nan"), "abs_m_ref": 0.0}
    proj = dm * np.conj(m_ref) / abs_m
    return {
        "dm_real": float(np.real(dm)),
        "dm_imag": float(np.imag(dm)),
        "dm_abs": float(abs(dm)),
        "parallel": float(np.real(proj)),
        "perp": float(np.imag(proj)),
        "abs_m_ref": float(abs_m),
        "d_abs_m": float(abs(m_new) - abs_m),
    }


def scalar_nm(n, m, rn, ra_max, label, source, uv_status):
    q0 = q_mu(n[0, 0], m[0, 0], rn[0, 0])
    q1 = q_mu(n[1, 1], m[1, 1], rn[1, 1])
    return {
        "label": label,
        "source": source,
        "production_G1_pass": False,
        "uv_for_4x4": uv_status,
        "n": pack_c(n),
        "m": pack_c(m),
        "R_N": pack_rn({"real": np.real(rn).tolist(), "imag": np.imag(rn).tolist()}),
        "maxabs_R_A": ra_max,
        "Q0": q0,
        "Q1": q1,
    }


def vs_baseline(cfg, base):
    n, m = cmat(cfg["n"]), cmat(cfg["m"])
    n0, m0 = cmat(base["n"]), cmat(base["m"])
    dn = n - n0
    out = {
        "dQ0": cfg["Q0"]["Q"] - base["Q0"]["Q"],
        "dQ1": cfg["Q1"]["Q"] - base["Q1"]["Q"],
        "dn00_signed": float(np.real(dn[0, 0])),
        "dn11_signed": float(np.real(dn[1, 1])),
        "dn01": pack_c(dn[0, 1]),
        "m00": dm_split(m0[0, 0], m[0, 0]),
        "m11": dm_split(m0[1, 1], m[1, 1]),
        "m01": dm_split(m0[0, 1], m[0, 1]),
    }
    return out


def kernel_ref_rn_ra():
    """Read only the header of the 37MB kernel JSON. Do not parse the full file."""
    keys = {}
    buf = []
    with KJSON.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            buf.append(line)
            if i > 140:
                break
    text = "".join(buf)
    # Close the truncated object enough to extract R_N_ref / R_A_ref via line scan.
    rn = None
    ra = None
    # Use the already-audited header numbers by a tiny JSON fragment rebuild.
    # Fall back: scan with numpy from gram-equivalent + published delta.
    # Direct: eval the first complete keys by finding matching braces after key.
    def extract(tag):
        i = text.find(f'"{tag}"')
        if i < 0:
            return None
        j = text.find("{", i)
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[j : k + 1])
        return None

    rn = extract("R_N_ref")
    ra = extract("R_A_ref")
    return rn, ra


def main():
    if PHASEJSON.exists():
        raise FileExistsError(PHASEJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    prog("load nm / gram / budget")
    nm = load(NM)
    nm_ts = load(NM_TS)
    gram = load(GRAM)
    budget = load(BUDGET)
    rn_ref, ra_ref = kernel_ref_rn_ra()
    prog("kernel header RN_ref extracted=" + str(rn_ref is not None))

    k = nm["kernel_9cycles_prod_vs_ref"]
    w = nm["window_9_vs_11_production"]
    ts = nm_ts["timestep_1024_vs_2048_production"]

    rn9 = cmat(gram["full_RN_minus_I"]["9_cycles"])
    rn11 = cmat(gram["full_RN_minus_I"]["11_cycles"])
    ra9 = float(gram["full_RA"]["9_cycles"]["maxabs"])
    ra11 = float(gram["full_RA"]["11_cycles"]["maxabs"])
    rn_kref = cmat(rn_ref) if rn_ref is not None else rn9
    ra_kref = float(ra_ref["maxabs"]) if ra_ref is not None else None

    rn2048 = np.zeros((2, 2), complex)
    rn2048[0, 0] = budget["nine_cycle_IR_corrected_RN_00"]["macro2048"]
    # published JSON has RN_00 only for 2048; RN_11 left 0 and flagged
    rn2048_note = "R_N(1,1) not in published 2048 JSON; Q1 uses 0 for RN_11 (~1e-5 vs Q1~2e2)"

    nine = scalar_nm(
        cmat(k["n_a"]), cmat(k["m_a"]), rn9, ra9,
        "9cycles_macro1024_prod",
        str(NM.relative_to(ROOT)) + " :: kernel_9cycles_prod_vs_ref.a",
        "available gram_grouping_9_vs_11.npz u9,v9",
    )
    kref = scalar_nm(
        cmat(k["n_b"]), cmat(k["m_b"]), rn_kref, ra_kref,
        "9cycles_macro1024_kernel_ref",
        str(NM.relative_to(ROOT)) + " :: kernel_9cycles_prod_vs_ref.b",
        "available kernel_correction_9cycles.npz u_ref,v_ref",
    )
    eleven = scalar_nm(
        cmat(w["n_b"]), cmat(w["m_b"]), rn11, ra11,
        "11cycles_macro1024_prod",
        str(NM.relative_to(ROOT)) + " :: window_9_vs_11_production.b",
        "available gram_grouping_9_vs_11.npz u11,v11",
    )
    ts2048 = scalar_nm(
        cmat(ts["n_b"]), cmat(ts["m_b"]), rn2048, None,
        "9cycles_macro2048_prod",
        str(NM_TS.relative_to(ROOT)) + " :: timestep_1024_vs_2048_production.b",
        "not assembled; 4x4 marked unavailable",
    )
    ts2048["R_N_note"] = rn2048_note
    ts2048["maxabs_R_A"] = None

    diffs = {
        "kernel_ref_minus_prod": vs_baseline(kref, nine),
        "window_11_minus_9": vs_baseline(eleven, nine),
        "timestep_2048_minus_1024": vs_baseline(ts2048, nine),
    }
    dQ_win = diffs["window_11_minus_9"]["dQ0"]
    checkpoint = {
        "window_dQ0": dQ_win,
        "window_dQ0_abs": abs(dQ_win),
        "scale_is_1e-2_not_O10": abs(dQ_win) < 1.0,
        "window_dn00_signed": diffs["window_11_minus_9"]["dn00_signed"],
        "window_dm00_parallel": diffs["window_11_minus_9"]["m00"]["parallel"],
        "window_dm00_perp": diffs["window_11_minus_9"]["m00"]["perp"],
        "window_d_abs_m00": diffs["window_11_minus_9"]["m00"]["d_abs_m"],
        "note": (
            "dn00 and d|m00| nearly cancel in Q0. "
            "Do not read O(10) |dn00| as a Q shift. "
            "Not a squeezing statement."
        ),
    }

    report = {
        "task": "signed Q / phase / dm_parallel / dm_perp from existing n,m JSON",
        "date": "2026-10-18",
        "production_G1_pass": False,
        "not_a_squeezing_claim": True,
        "not_production_nmV": True,
        "definitions": {
            "n": "V.conj() @ V.T  (= V* V^T)",
            "m": "U @ V.T  (U includes U0)",
            "Q_mu": "n_mumu + (1 + R_N_mumu)/2 - |m_mumu|",
            "Q_half_side": "n_mumu + 1/2 - |m_mumu|",
            "Var_X_theta": "n + (1+R_N)/2 + Re[m exp(-2i theta)]",
            "X_theta": "(a*exp(-i theta) + adag*exp(i theta))/sqrt(2)",
            "theta_min": "(arg(m)+pi)/2  (mod pi); minimum-noise direction",
            "theta_max": "arg(m)/2  (mod pi); maximum-noise direction",
            "theta_half_arg_m_is": "theta_max, not theta_min",
            "dm_parallel": "Re(dm * conj(m) / |m|)",
            "dm_perp": "Im(dm * conj(m) / |m|)",
        },
        "phase_convention_correction": {
            "old_file": str(OUTJSON.relative_to(ROOT)),
            "old_label_error": "q_signed_from_existing_nm.json labeled theta=arg(m)/2 as min-noise; that direction is max-noise",
            "Q_definition_unchanged": True,
            "old_file_not_overwritten": True,
        },
        "error_estimate_labels": {
            "timestep_ratio_1_over_15": "conditional_error_estimate_not_strict_bound",
            "four_window_swing": "conditional_error_estimate_not_strict_bound",
        },
        "baseline": "9cycles_macro1024_prod",
        "configs": {
            "nine_1024_prod": nine,
            "nine_1024_kernel_ref": kref,
            "eleven_1024_prod": eleven,
            "nine_2048_prod": ts2048,
        },
        "signed_diff_vs_baseline": diffs,
        "checkpoint_window_dQ": checkpoint,
        "k11_status": "running_or_unread; slice C not in this file",
        "slice_D_4x4": "not in this JSON; Omega_0 vs Omega_eff is slice D",
        "elapsed_s": float(elapsed()),
    }
    PHASEJSON.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    prog("saved " + str(PHASEJSON))
    print("done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

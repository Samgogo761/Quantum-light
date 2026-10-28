#!/usr/bin/env python3
"""Independent R0/R1 recompute: moments, Lambda_eff, Peierls contact FD.

Read-only on frozen QED/GH5 JSON. No electron-photon propagation.
Writes:
  docs/04_BSV量子光/qmath_r1_independent.json
  docs/04_BSV量子光/qmath_link_audit.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from q8_m0_model import R0, build_geometry, current_ops, dh8, h8, sha256_file


def a2_polarization(geo) -> np.ndarray:
    """Same unit vector as q8_m0_qinput_minimal_20261020.a2_polarization.

    Inlined so this audit does not import scipy (that module pulls Laguerre).
    """
    e = np.asarray(geo.a2_cart, dtype=float)[:2]
    return e / np.linalg.norm(e)

ROOT = HERE.parents[1]
DOCS = ROOT / "docs" / "04_BSV量子光"
PACK = DOCS / "qmath_pack_16_11_20261023"
OUT_IND = DOCS / "qmath_r1_independent.json"
OUT_AUDIT = DOCS / "qmath_link_audit.json"

K_FRAC = [0.173, 0.271]
ALPHA = 0.5
R_SQ = 0.3
G_COUPLING = 0.005
OMEGA0 = 0.05
NMAX = 8
N_FULL = 11154
DT_APPROX = 0.067597
PROTOCOL = "Qin_a2_1_4_1"

FROZEN_JSON = {
    "thermal": ROOT / "results/q_line/qinput_qin_a2_141_gh5_thermal_20261120/gh5_thermal.json",
    "qed_theta_pi": ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_20261120/qed_theta_pi.json",
    "mech": ROOT / "results/q_line/qinput_qin_a2_141_qed_theta_pi_mech_20261120/mech.json",
    "q1a": ROOT / "results/q_line/qinput_qin_a2_141_q1a_20261120/q1a.json",
    "gh5": ROOT / "results/q_line/qinput_qin_a2_141_gh5_20261120/gh5.json",
    "gh5_theta_pi": ROOT / "results/q_line/qinput_qin_a2_141_gh5_theta_pi_20261120/gh5_theta_pi.json",
}
SOURCE_PY = {
    "q8_m0_model.py": HERE / "q8_m0_model.py",
    "q8_m0_qinput_dynamics_20261026.py": HERE / "q8_m0_qinput_dynamics_20261026.py",
    "q8_m0_qinput_minimal_20261020.py": HERE / "q8_m0_qinput_minimal_20261020.py",
}
EXPECTED_HASHES = {
    "q8_m0_model.py":
        "5d53090cabb60238c6c16a74b92ddc061d5adf2d8081e48b46b9bdf0089261d3",
    "q8_m0_qinput_dynamics_20261026.py":
        "bf6d4e407bf55f1e6ebc3767f1b2bd6075ac7ef564f195abc1217c7d51671f94",
}
DOWNLOADS_ZIP = Path(r"C:\Users\26507\Downloads\Q-MATH_完整研究接续包_截至16_11.zip")
REPO_ZIP = ROOT / "Q-MATH_完整研究接续包_截至16_11.zip"


def sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cplx(block) -> complex:
    return complex(float(block["re"]), float(block["im"]))


def c_to_dict(z: complex) -> dict:
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z)),
            "arg": float(np.angle(z))}


def herm_err(m: np.ndarray) -> float:
    return float(np.max(np.abs(m - m.conj().T)))


def analytic_moments(alpha: float, r: float) -> dict:
    ns = math.sinh(r) ** 2
    nbar = abs(alpha) ** 2 + ns
    m0 = -0.5 * math.sinh(2.0 * r)
    mpi = +0.5 * math.sinh(2.0 * r)
    varx0 = math.exp(-2.0 * r)
    varxpi = math.exp(+2.0 * r)
    var_th = math.cosh(2.0 * r)
    x4_th = 3.0 * var_th ** 2
    x4_mix = 1.5 * (math.exp(-4.0 * r) + math.exp(+4.0 * r))
    m_cl = nbar - abs(alpha) ** 2
    varx_cl_lo = 1.0
    varx_cl_hi = 1.0 + 4.0 * m_cl
    return {
        "alpha": alpha,
        "r": r,
        "mean_X": 2.0 * alpha,
        "mean_P": 0.0,
        "ns_sinh2r": ns,
        "nbar": nbar,
        "M0": m0,
        "Mpi": mpi,
        "dM": mpi - m0,
        "dM_is_sinh_2r": True,
        "sinh_2r": math.sinh(2.0 * r),
        "VarX_theta0": varx0,
        "VarX_theta_pi": varxpi,
        "VarX_thermal": var_th,
        "VarP_theta0": varxpi,
        "VarP_theta_pi": varx0,
        "VarP_thermal": var_th,
        "X4_thermal": x4_th,
        "X4_mix": x4_mix,
        "X4_rel_to_th": (x4_mix - x4_th) / x4_th,
        "classical_P_abs_M_bound": m_cl,
        "squeezed_abs_M": abs(mpi),
        "squeezed_M_outside_classical_bound": abs(mpi) > m_cl + 1e-15,
        "classical_P_VarX_interval": [varx_cl_lo, varx_cl_hi],
        "theta0_VarX_outside_classical": varx0 < varx_cl_lo - 1e-15,
        "thetapi_VarX_outside_classical": varxpi > varx_cl_hi + 1e-15,
        "note_M_theta": "M_theta = -0.5 * exp(i theta) * sinh(2r) for D(alpha)S(r e^{i theta})|0>",
        "X_normalization": "X=a+a^dagger, vacuum Var(X)=1",
    }


def finite_difference_contact(eps_list=(1e-4, 3e-5, 1e-5)) -> dict:
    geo = build_geometry()
    e = a2_polarization(geo)
    k0 = geo.k_cart(K_FRAC)
    out = {
        "K_FRAC": K_FRAC,
        "k_cart": k0.tolist(),
        "e_a2": e.tolist(),
        "R0": R0.as_dict(),
        "peierls_rule": "P exp(i eta X) P, not exp(i eta P X P)",
        "A_operator": "A_hat = g s(t) X, X=a+a^dagger, Schrodinger picture",
        "j_definition": "j_a = - dH/dk_a  (e=hbar=1)",
        "dh8_drops_neel": True,
        "eps_list": list(eps_list),
        "per_neel": {},
        "neel_j_independence": {},
    }
    j0_store = {}
    for neel in (1, -1):
        h0 = h8(geo, R0, k0, neel)
        j0 = current_ops(geo, R0, k0, neel)
        dh = dh8(geo, R0, k0, neel)
        v_from_dh = np.tensordot(dh, e, axes=(0, 0))
        v_from_j = -np.tensordot(j0, e, axes=(0, 0))
        rec = {
            "H0_herm_err": herm_err(h0),
            "j0_herm_err": [herm_err(j0[a]) for a in range(2)],
            "V_from_dh_minus_minus_j0_dot_e_max": float(np.max(np.abs(v_from_dh - v_from_j))),
            "eps": {},
        }
        j0_store[neel] = j0
        for eps in eps_list:
            kp = k0 + eps * e
            km = k0 - eps * e
            hp = h8(geo, R0, kp, neel)
            hm = h8(geo, R0, km, neel)
            jp = current_ops(geo, R0, kp, neel)
            jm = current_ops(geo, R0, km, neel)
            v_fd = (hp - hm) / (2.0 * eps)
            w_fd = (hp - 2.0 * h0 + hm) / (eps ** 2)
            j1_fd = (jp - jm) / (2.0 * eps)
            j2_fd = (jp - 2.0 * j0 + jm) / (eps ** 2)
            w_from_j1 = -np.tensordot(j1_fd, e, axes=(0, 0))
            rec["eps"][str(eps)] = {
                "V_fd_minus_V_dh_max": float(np.max(np.abs(v_fd - v_from_dh))),
                "W_fd_minus_minus_j1_dot_e_max": float(np.max(np.abs(w_fd - w_from_j1))),
                "j1_herm_err": [herm_err(j1_fd[a]) for a in range(2)],
                "j2_herm_err": [herm_err(j2_fd[a]) for a in range(2)],
                "||V_fd||_F": float(np.linalg.norm(v_fd)),
                "||W_fd||_F": float(np.linalg.norm(w_fd)),
                "||j0||_F": float(np.linalg.norm(j0)),
                "||j1||_F": float(np.linalg.norm(j1_fd)),
                "||j2||_F": float(np.linalg.norm(j2_fd)),
                "contact_j2_nonzero": float(np.linalg.norm(j2_fd)) > 1e-12,
            }
        out["per_neel"][str(neel)] = rec
    dj = j0_store[1] - j0_store[-1]
    out["neel_j_independence"] = {
        "max_abs_j_plus_minus_j_minus": float(np.max(np.abs(dj))),
        "conclusion": "current operators are Neel-independent at this k; magnetic-odd current comes from rho, not from j_a(N)",
    }
    return out


def lambda_eff_from_frozen() -> dict:
    thermal = load_json(FROZEN_JSON["thermal"])
    qed = load_json(FROZEN_JSON["qed_theta_pi"])
    j0_g = cplx(thermal["h2_main"]["jx_0"])
    jpi_g = cplx(thermal["h2_main"]["jx_pi"])
    jmix = cplx(thermal["h2_main"]["jx_mix"])
    jth = cplx(thermal["h2_main"]["jx_th"])
    r2 = cplx(thermal["h2_main"]["jx_R2"])
    dtheta_g = jpi_g - j0_g
    j0_q = cplx(qed["h2_main"]["jx_0"])
    jpi_q = cplx(qed["h2_main"]["jx_pi"])
    dtheta_q = jpi_q - j0_q
    mom = analytic_moments(ALPHA, R_SQ)
    dm = mom["dM"]
    lam_g = dtheta_g / dm
    lam_q = dtheta_q / dm
    eps2 = abs(r2) / abs(dtheta_g)
    eps2_vs_mix = abs(r2) / abs(jmix)
    coverage = qed.get("coverage", {})
    back_missing = (not coverage.get("covers_5T0", False)) or abs(qed["h2_main"].get("abs_back", 0.0)) == 0.0
    qed_vs = qed.get("qed_vs_gh5", {})
    return {
        "window": "[2 T0, 4 T0] rectangular Fourier of J_{2,x}^odd",
        "fourier_convention": "J_q = 1/(t1-t0) int exp(+i q omega0 t) J(t) dt",
        "not_certified_HHG": True,
        "GH5": {
            "J_theta0": c_to_dict(j0_g),
            "J_thetapi": c_to_dict(jpi_g),
            "J_mix": c_to_dict(jmix),
            "J_thermal": c_to_dict(jth),
            "R2": c_to_dict(r2),
            "dtheta": c_to_dict(dtheta_g),
            "eps2_R2_over_dtheta": float(eps2),
            "eps2_R2_over_mix": float(eps2_vs_mix),
            "eps2_is_not_quadrature_error_bound": True,
            "eps2_is_not_strict_higher_moment_bound": True,
            "Lambda_eff_secant": c_to_dict(lam_g),
            "Lambda_eff_is_not_microscopic_chi2": True,
        },
        "QED": {
            "J_theta0": c_to_dict(j0_q),
            "J_thetapi": c_to_dict(jpi_q),
            "dtheta": c_to_dict(dtheta_q),
            "amp_ratio_2d": qed["h2_main"].get("amp_ratio_pi_over_0"),
            "dphi_rad": qed["h2_main"].get("dphi_pi_minus_0_rad"),
            "Lambda_eff_secant": c_to_dict(lam_q),
            "coverage_t_over_T0": coverage.get("t_min_over_T0"),
            "covers_4T0": coverage.get("covers_4T0"),
            "covers_5T0": coverage.get("covers_5T0"),
            "rear_window_missing": back_missing,
            "abs_back_is_data_gap_not_physical_zero": back_missing,
            "qed_vs_gh5_rel_vec": qed_vs.get("rel_vec"),
            "job_status": coverage.get("job_status"),
        },
        "Lambda_eff_GH5_minus_QED": c_to_dict(lam_g - lam_q),
        "two_real_M_cannot_split_M_and_Mstar": True,
        "thermal_json_eps2": thermal["h2_main"].get("eps2"),
    }


def pack_audit() -> dict:
    names_path = PACK / "original_names.txt"
    items = []
    if names_path.exists():
        for line in names_path.read_text(encoding="utf-8").splitlines():
            items.append(line.strip())
    inner9 = sorted(p.name for p in (PACK / "inner_16_9").glob("i9_*")) if (PACK / "inner_16_9").exists() else []
    inner10 = sorted(p.name for p in (PACK / "inner_16_10").iterdir()) if (PACK / "inner_16_10").exists() else []
    has_162 = False
    for p in PACK.rglob("*"):
        n = p.name.lower()
        if "16.2" in n or "16_2" in n or "qmath-16.2" in n or "qmath_16_2" in n:
            has_162 = True
            break
    dl_hash = sha256_path(DOWNLOADS_ZIP)
    repo_hash = sha256_path(REPO_ZIP)
    return {
        "pack_dir": str(PACK),
        "original_names": items,
        "inner_16_10_names": inner10,
        "inner_16_9_extracted_names": inner9,
        "Q-MATH-16.2_source_in_pack": has_162,
        "inner_08_is_16_9_archive_not_16_2": True,
        "downloads_zip_exists": DOWNLOADS_ZIP.exists(),
        "downloads_zip_sha256": dl_hash,
        "repo_zip_exists": REPO_ZIP.exists(),
        "repo_zip_sha256": repo_hash,
        "do_not_claim_16_2_reproduced": True,
        "theorem_A": "zero-mean nonnegative Glauber-Sudarshan P can realize M iff ||M||_* <= E",
        "theorem_B": "at leading two-photon coherent conversion, the normalized output pattern can be copied by M_cl = t M with t=E/||M||_*. Forbidden to claim quantum advantage from a normalized H2 pattern.",
        "theorem_B_does_not_directly_apply": [
            "current protocol is displaced squeezed, not zero-mean BSV",
            "observable is finite-gate QED current Fourier, not elastic two-photon S-matrix",
            "g=0.005 nonperturbative joint evolution, not lowest-order scattering",
        ],
        "positive_P_means_Glauber_Sudarshan_not_SDE": True,
        "input_is_displaced_squeezed_not_pure_BSV": True,
    }


def source_hashes() -> dict:
    out = {}
    for name, path in SOURCE_PY.items():
        h = sha256_file(path)
        out[name] = {
            "sha256": h,
            "matches_expected": h == EXPECTED_HASHES.get(name) if name in EXPECTED_HASHES else None,
            "path": str(path),
        }
    for name, path in FROZEN_JSON.items():
        out[name] = {"sha256": sha256_path(path), "exists": path.exists(), "path": str(path)}
    return out


def main():
    moments = analytic_moments(ALPHA, R_SQ)
    fd = finite_difference_contact()
    lam = lambda_eff_from_frozen()
    pack = pack_audit()
    hashes = source_hashes()
    independent = {
        "task": "R0_R1_readonly_independent_recompute",
        "no_new_propagation": True,
        "protocol": PROTOCOL,
        "g": G_COUPLING,
        "omega0": OMEGA0,
        "nmax": NMAX,
        "n_full_steps": N_FULL,
        "dt_approx": DT_APPROX,
        "moments": moments,
        "finite_difference_contact": fd,
        "lambda_eff": lam,
        "source_hashes": hashes,
        "answers": {
            "affine_M_response": "compatible_not_proven",
            "eightband_derives_magnetic_odd_H2": "parity_yes_quantitative_K2_not_derived",
            "strict_classical_negative_control": "input_nonclassical_yes_output_not_proven",
        },
    }
    audit = {
        "pack": pack,
        "expected_model_hashes": EXPECTED_HASHES,
        "observed_model_hashes": {k: hashes[k] for k in SOURCE_PY},
        "frozen_json_hashes": {k: hashes[k] for k in FROZEN_JSON},
        "lambda_eff_summary": {
            "GH5": lam["GH5"]["Lambda_eff_secant"],
            "QED": lam["QED"]["Lambda_eff_secant"],
            "eps2": lam["GH5"]["eps2_R2_over_dtheta"],
            "eps2_vs_mix": lam["GH5"]["eps2_R2_over_mix"],
            "qed_vs_gh5_rel_vec": lam["QED"]["qed_vs_gh5_rel_vec"],
            "rear_window_missing": lam["QED"]["rear_window_missing"],
        },
        "contact_consistency": {
            "neel_j_independence": fd["neel_j_independence"],
            "V_identity_plus": fd["per_neel"]["1"]["V_from_dh_minus_minus_j0_dot_e_max"],
            "W_identity_plus_1e-4": fd["per_neel"]["1"]["eps"]["0.0001"]["W_fd_minus_minus_j1_dot_e_max"],
            "j2_contact_nonzero": fd["per_neel"]["1"]["eps"]["0.0001"]["contact_j2_nonzero"],
        },
        "do_not_run_R2": True,
        "do_not_claim_certified_HHG": True,
        "do_not_claim_output_entanglement": True,
        "do_not_claim_quantum_advantage": True,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    with OUT_IND.open("w", encoding="utf-8") as f:
        json.dump(independent, f, indent=2, ensure_ascii=False)
    with OUT_AUDIT.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print("wrote", OUT_IND)
    print("wrote", OUT_AUDIT)
    print("eps2", lam["GH5"]["eps2_R2_over_dtheta"])
    print("Lambda_GH5", lam["GH5"]["Lambda_eff_secant"])
    print("Lambda_QED", lam["QED"]["Lambda_eff_secant"])
    print("j_neel_diff", fd["neel_j_independence"]["max_abs_j_plus_minus_j_minus"])
    print("16.2_in_pack", pack["Q-MATH-16.2_source_in_pack"])
    print("downloads_zip", pack["downloads_zip_exists"], pack["downloads_zip_sha256"])


if __name__ == "__main__":
    main()

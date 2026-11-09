#!/usr/bin/env python3
"""Read-only 1600 nm protocol-A exact-QED four-angle postprocess.

Load NPZ/JSON, integrate, fit, compare. Never import or call a propagator.
Never read QED pi/4 currents. Do not call GH5 analyze_fourangle().
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qinput_spectrum_reanalysis_20261028 import fit_bac, packed, spectrum, unpack

ROOT = Path(__file__).resolve().parents[2]
QED_OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_qed_fourangle_20261102"
GH5_OUT = ROOT / "results/q_line/qinput_1600nm_protocolA_gh5_20261102"
ANALYSIS = QED_OUT / "analysis_endpoint_v1"
FIT_PATH = QED_OUT / "fourangle_fit.json"
FIT_SHA_PATH = QED_OUT / "fourangle_fit.sha256"
SPECTRUM_PY = HERE / "qinput_spectrum_reanalysis_20261028.py"
WRAPPER_PY = HERE / "q8_m0_qinput_1600nm_protocolA_qed_fourangle_20261102.py"

OMEGA0 = 0.775
PERIOD = 2.0 * np.pi / OMEGA0
N_FULL = 11154
N_MAX = 8
ALPHA = 0.5
R_SQ = 0.3
G_ABS = 0.005
K_FRAC = (0.173, 0.271)
SCHEMA_VERSION = "1600nm_protocolA_qed_fourangle_fit_v1"
ANALYZER_VERSION = "qed_postprocess_20261102_v1"
FOURIER_SIGN = "Jq=(tb-ta)^{-1} int_ta^tb exp(+i q omega t) J(t) dt"
WINDOW_MAIN = (2.0, 4.0)
WINDOWS = {"main": (2.0, 4.0), "front": (1.0, 3.0), "back": (3.0, 5.0)}
TRAINING_TAGS = ("p_0", "m_0", "p_h", "m_h", "p_pi", "m_pi", "p_t", "m_t")
ANGLE_TAGS = ("0", "h", "pi", "t")
ANGLE_PAIRS = (("0", "p_0", "m_0"), ("h", "p_h", "m_h"), ("pi", "p_pi", "m_pi"), ("t", "p_t", "m_t"))
THETA_ORDER = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
HOLD_THETA = 0.25 * np.pi
FORBIDDEN_QED_TAGS = frozenset({"p_q", "m_q"})
ESYM_MARK = 0.02
IDEAL_R = float(np.sinh(2.0 * R_SQ) / (2.0 * ALPHA * ALPHA))
FLOAT_ATOL = 1e-12
PHYSICS_FILES = (
    "q8_m0_qinput_dynamics_20261026.py",
    "q8_m0_qinput_minimal_20261020.py",
    "q8_m0_qinput_shared_kk_static_20261029.py",
    "q8_m0_qinput_shared_kk_dyn_preflight_20261030.py",
    "q8_m0_qinput_shared_kk_ds_odd_20261102.py",
    "q8_m0_model.py",
    "q8_m0_qinput_1600nm_protocolA_qed_fockaudit_20261102.py",
)
HISTORICAL_GH5 = {
    "note": "legacy rect_jq, missed window endpoints; x-component only; not a production reference",
    "amp_ratio_pi_over_0_x": 8.777927804975437,
    "rho_absA_over_absB": 1.273948419062064,
    "arg_A_over_B_deg": 179.04119106765012,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_training_not_holdout(theta_order=THETA_ORDER, training_tags=TRAINING_TAGS) -> None:
    if any(abs(float(th) - HOLD_THETA) < 1e-12 for th in theta_order):
        raise RuntimeError("training labels include pi/4")
    if any(str(tag) in FORBIDDEN_QED_TAGS for tag in training_tags):
        raise RuntimeError("training tags include holdout")


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def closed_form_bac(z):
    z0, zh, zpi, zt = [complex(v) for v in z]
    b = (z0 + zh + zpi + zt) / 4.0
    a = (z0 - 1j * zh - zpi + 1j * zt) / 4.0
    c = (z0 + 1j * zh - zpi - 1j * zt) / 4.0
    return np.array([b, a, c], complex)


def cz(z) -> complex:
    if isinstance(z, dict):
        return unpack(z)
    return complex(z)


def ratio_metric(num, den, sensitivity):
    num, den = complex(num), complex(den)
    if abs(den) == 0.0:
        return None, "zero_denominator"
    val = float(abs(num) / abs(den))
    if abs(den) <= float(sensitivity):
        return val, "不宜解释"
    return val, "ok"


def principal_deg(rad) -> float:
    return float(np.degrees(np.angle(np.exp(1j * float(rad)))))


def load_job(out: Path, tag: str) -> dict:
    if out.resolve() == QED_OUT.resolve() and tag in FORBIDDEN_QED_TAGS:
        raise RuntimeError(f"refused to load QED holdout job {tag} during four-angle postprocess")
    path = out / f"job_{tag}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_series(out: Path, tag: str):
    if out.resolve() == QED_OUT.resolve() and tag in FORBIDDEN_QED_TAGS:
        raise RuntimeError(f"refused to load QED holdout series {tag} during four-angle postprocess")
    path = out / f"series_{tag}.npz"
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z["t"], float)
        jx = np.asarray(z["Jx"])
        jy = np.asarray(z["Jy"])
    j = np.column_stack([jx, jy]).astype(complex, copy=False)
    return t, j


def official_n_full(rec: dict):
    if not rec:
        return None
    defn = rec.get("definition") or {}
    if defn.get("n_full") is None:
        return None
    n_full = int(defn["n_full"])
    top = rec.get("n_steps")
    if top is not None and int(top) != n_full:
        return None
    return n_full


def _near(a, b, atol=FLOAT_ATOL) -> bool:
    return abs(float(a) - float(b)) <= atol


def preflight_qed() -> dict:
    sources = {}
    jobs = {}
    series = {}
    problems = []
    for tag in TRAINING_TAGS:
        job_path = QED_OUT / f"job_{tag}.json"
        ser_path = QED_OUT / f"series_{tag}.npz"
        if not job_path.is_file() or not ser_path.is_file():
            problems.append(f"missing {tag}")
            continue
        sources[str(job_path.relative_to(ROOT))] = sha256_file(job_path)
        sources[str(ser_path.relative_to(ROOT))] = sha256_file(ser_path)
        rec = load_job(QED_OUT, tag)
        jobs[tag] = rec
        n_full = official_n_full(rec)
        ledger = rec.get("ledger") or {}
        defn = rec.get("definition") or {}
        env = rec.get("envelope") or {}
        want_neel, want_theta = {
            "p_0": (1, 0.0), "m_0": (-1, 0.0),
            "p_h": (1, 0.5 * np.pi), "m_h": (-1, 0.5 * np.pi),
            "p_pi": (1, np.pi), "m_pi": (-1, np.pi),
            "p_t": (1, 1.5 * np.pi), "m_t": (-1, 1.5 * np.pi),
        }[tag]
        checks = [
            rec.get("status") == "ok",
            rec.get("fock_alarm_0p05") is False,
            rec.get("finite_state") is True,
            rec.get("covers_requested") is True,
            rec.get("error") is None,
            n_full == N_FULL,
            int(ledger.get("steps_completed") or -1) == N_FULL,
            rec.get("n_max") == N_MAX,
            rec.get("used_pf_OMEGA0") is False,
            rec.get("did_not_scale_g") is True,
            _near(rec.get("omega0"), OMEGA0),
            _near(rec.get("g"), G_ABS),
            int(rec.get("neel")) == want_neel,
            _near(rec.get("theta"), want_theta),
            _near(defn.get("omega0"), OMEGA0),
            _near(defn.get("g"), G_ABS),
            _near(defn.get("alpha"), ALPHA),
            _near(defn.get("r"), R_SQ),
            int(defn.get("n_max")) == N_MAX,
            int(defn.get("n_full")) == N_FULL,
            list(defn.get("k_fractional") or []) == list(K_FRAC),
            _near(env.get("omega0"), OMEGA0),
        ]
        if not all(checks):
            problems.append(f"protocol mismatch {tag}")
        t, j = load_series(QED_OUT, tag)
        if t.shape != (350,) or j.shape != (350, 2):
            problems.append(f"series shape {tag} {t.shape} {j.shape}")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(j)):
            problems.append(f"nonfinite series {tag}")
        if t.ndim != 1 or np.any(np.diff(t) <= 0):
            problems.append(f"time not strictly increasing {tag}")
        if t[-1] < 6.0 * PERIOD - 1e-9:
            problems.append(f"series does not cover 6T {tag}")
        series[tag] = (t, j)
    for lab, ptag, mtag in ANGLE_PAIRS:
        if ptag in series and mtag in series:
            if not np.array_equal(series[ptag][0], series[mtag][0]):
                problems.append(f"time arrays differ for {lab}")
    physics = {}
    for name in PHYSICS_FILES:
        path = HERE / name
        physics[name] = sha256_file(path) if path.is_file() else None
    job_physics = (jobs.get("p_0") or {}).get("source_hashes") or {}
    for name, got in physics.items():
        want = job_physics.get(name)
        if want and got != want:
            problems.append(f"physics hash drift {name}")
    if problems:
        raise RuntimeError("QED preflight failed: " + "; ".join(problems))
    return {"sources": sources, "jobs": jobs, "series": series, "physics": physics}


def harmonics_from_pair(t, jp, jm, methods=("linear_exact", "quadratic_exact", "legacy")):
    if not np.array_equal(t, t):
        raise RuntimeError("internal time mismatch")
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    qs = np.arange(1, 7, dtype=float)
    out = {}
    for method in methods:
        out[method] = {}
        for wname, (a, b) in WINDOWS.items():
            lo, hi = a * PERIOD, b * PERIOD
            even = spectrum(t, je, lo, hi, qs, method=method, omega=OMEGA0)
            odd = spectrum(t, jo, lo, hi, qs, method=method, omega=OMEGA0)
            rec = []
            for iq, q in enumerate(qs):
                rec.append({
                    "q": int(q),
                    "even_x": packed(even[iq, 0]), "even_y": packed(even[iq, 1]),
                    "odd_x": packed(odd[iq, 0]), "odd_y": packed(odd[iq, 1]),
                    "odd_norm2d": float(np.linalg.norm(odd[iq])),
                    "even_norm2d": float(np.linalg.norm(even[iq])),
                })
            out[method][wname] = rec
    return out


def z_from_harmonics(harms, method="linear_exact", window="main", parity="odd", axis="x"):
    key = f"{parity}_{axis}"
    zs = []
    for lab, _, _ in ANGLE_PAIRS:
        row = harms[lab][method][window][1]
        zs.append(cz(row[key]))
    return zs


def input_R_from_job(rec: dict):
    xp = rec.get("photon_xp_input") or {}
    mean_a = cz(xp.get("mean_a", {"re": 0.0, "im": 0.0}))
    m = xp.get("M")
    if m is None:
        return None, None, None
    m = cz(m)
    den = abs(mean_a) ** 2
    if den == 0.0:
        return None, float(abs(m)), float(abs(mean_a))
    return float(abs(m) / den), float(abs(m)), float(abs(mean_a))


def fit_from_z(zs, sensitivity):
    zs = [complex(z) for z in zs]
    coeff, stats = fit_bac(zs)
    closed = closed_form_bac(zs)
    if np.max(np.abs(coeff - closed)) > max(1e-18, 1e-12 * max(np.linalg.norm(zs), 1e-30)):
        raise RuntimeError("lstsq BAC disagrees with closed form")
    B, A, C = [complex(v) for v in coeff]
    zfit = [B + A * np.exp(1j * th) + C * np.exp(-1j * th) for th in THETA_ORDER]
    resid = [zs[i] - zfit[i] for i in range(4)]
    zn = float(np.linalg.norm(zs))
    rho, rho_flag = ratio_metric(A, B, sensitivity)
    c_over_a, ca_flag = ratio_metric(C, A, sensitivity)
    if abs(B) == 0.0:
        arg_ab = None
        arg_flag = "zero_denominator"
        dist_pi = None
    else:
        arg_ab = principal_deg(np.angle(A * np.conj(B)))
        arg_flag = "ok"
        dist_pi = abs(principal_deg(np.radians(arg_ab) - np.pi))
    pred = B + A * np.exp(1j * HOLD_THETA) + C * np.exp(-1j * HOLD_THETA)
    z0, zpi = zs[0], zs[2]
    x_ratio, x_flag = ratio_metric(zpi, z0, sensitivity)
    return {
        "coeff": (B, A, C),
        "z": zs,
        "zfit": zfit,
        "resid": resid,
        "stats": stats,
        "relative_L2_residual": float(np.linalg.norm(resid) / max(zn, 1e-300)),
        "rho": rho, "rho_flag": rho_flag,
        "arg_AB_deg": arg_ab, "arg_flag": arg_flag,
        "arg_distance_to_pi_deg": dist_pi,
        "abs_C_over_A": c_over_a, "abs_C_over_A_flag": ca_flag,
        "x_amp_ratio_pi_over_0": x_ratio, "x_amp_ratio_flag": x_flag,
        "holdout_prediction": pred,
        "znorm": zn,
    }


def two_d_ratio(harms, method="linear_exact", window="main"):
    r0 = harms["0"][method][window][1]
    rpi = harms["pi"][method][window][1]
    n0 = np.array([cz(r0["odd_x"]), cz(r0["odd_y"])], complex)
    npi = np.array([cz(rpi["odd_x"]), cz(rpi["odd_y"])], complex)
    den = float(np.linalg.norm(n0))
    num = float(np.linalg.norm(npi))
    if den == 0.0:
        return None, "zero_denominator"
    return num / den, "ok"


def fit_record(zs, harms, sensitivity, extra=None):
    fit = fit_from_z(zs, sensitivity)
    amp2d, amp2d_flag = two_d_ratio(harms)
    B, A, C = fit["coeff"]
    rec = {
        "B": packed(B), "A": packed(A), "C": packed(C),
        "closed_form_agrees_with_lstsq": True,
        "relative_L2_residual": fit["relative_L2_residual"],
        "residuals": [packed(z) for z in fit["resid"]],
        "observations": {
            "theta_0": packed(fit["z"][0]),
            "theta_h": packed(fit["z"][1]),
            "theta_pi": packed(fit["z"][2]),
            "theta_t": packed(fit["z"][3]),
        },
        "derived_metrics": {
            "rho_absA_over_absB": fit["rho"],
            "rho_flag": fit["rho_flag"],
            "arg_A_conjB_deg": fit["arg_AB_deg"],
            "arg_flag": fit["arg_flag"],
            "arg_distance_to_pi_deg": fit["arg_distance_to_pi_deg"],
            "abs_C_over_A": fit["abs_C_over_A"],
            "abs_C_over_A_flag": fit["abs_C_over_A_flag"],
            "x_amp_ratio_pi_over_0": fit["x_amp_ratio_pi_over_0"],
            "x_amp_ratio_flag": fit["x_amp_ratio_flag"],
            "amp2d_ratio_pi_over_0": amp2d,
            "amp2d_ratio_flag": amp2d_flag,
            "did_not_compare_x_ratio_to_q6_4p98": True,
        },
        "holdout_prediction": packed(fit["holdout_prediction"]),
    }
    if extra:
        rec.update(extra)
    return rec, fit


def snapshot_status() -> None:
    src = QED_OUT / "fourangle_status.json"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    dst = ANALYSIS / "fourangle_status_pre_analyze_snapshot.json"
    if src.is_file() and not dst.is_file():
        shutil.copy2(src, dst)


def build_code_hashes() -> dict:
    return {
        "postprocess": sha256_file(Path(__file__)),
        "wrapper": sha256_file(WRAPPER_PY),
        "spectrum": sha256_file(SPECTRUM_PY),
        "analyzer_version": ANALYZER_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def frozen_schema(payload: dict) -> list[str]:
    need = [
        "schema_version", "method", "frequency", "window", "fourier_sign",
        "normalization", "component", "domain_parity", "theta_order",
        "training_tags", "sources", "protocol", "code_hashes", "observations",
        "coefficients", "residuals", "derived_metrics", "holdout_theta",
        "holdout_prediction", "holdout_used_in_fit", "frozen_at",
    ]
    missing = [k for k in need if k not in payload]
    if payload.get("schema_version") != SCHEMA_VERSION:
        missing.append("schema_version_mismatch")
    if payload.get("method") != "exact_QED":
        missing.append("method")
    if payload.get("holdout_used_in_fit") is not False:
        missing.append("holdout_used_in_fit")
    if list(payload.get("training_tags") or []) != list(TRAINING_TAGS):
        missing.append("training_tags")
    return missing


def validate_frozen_fit(path: Path | None = None) -> dict:
    path = path or FIT_PATH
    if not path.is_file() or not FIT_SHA_PATH.is_file():
        return {"ok": False, "why": "missing freeze files"}
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    want = FIT_SHA_PATH.read_text(encoding="utf-8").strip().split()[0]
    if got != want:
        return {"ok": False, "why": "sha256 mismatch", "got": got, "want": want}
    payload = json.loads(raw.decode("utf-8"))
    missing = frozen_schema(payload)
    if missing:
        return {"ok": False, "why": "schema", "missing": missing}
    for rel, h in (payload.get("sources") or {}).items():
        p = ROOT / rel
        if not p.is_file() or sha256_file(p) != h:
            return {"ok": False, "why": f"source changed {rel}"}
    if payload.get("code_hashes", {}).get("spectrum") != sha256_file(SPECTRUM_PY):
        return {"ok": False, "why": "spectrum integrator hash changed"}
    if payload.get("code_hashes", {}).get("analyzer_version") != ANALYZER_VERSION:
        return {"ok": False, "why": "analyzer_version changed"}
    proto = payload.get("protocol") or {}
    if abs(float(payload.get("frequency")) - OMEGA0) > FLOAT_ATOL:
        return {"ok": False, "why": "frequency"}
    if abs(float(proto.get("omega0", payload.get("frequency"))) - OMEGA0) > FLOAT_ATOL:
        return {"ok": False, "why": "protocol.omega0"}
    if abs(float(proto.get("g", G_ABS)) - G_ABS) > FLOAT_ATOL:
        return {"ok": False, "why": "protocol.g"}
    if int(proto.get("n_max", N_MAX)) != N_MAX or int(proto.get("n_full", N_FULL)) != N_FULL:
        return {"ok": False, "why": "protocol n_max/n_full"}
    return {"ok": True, "sha256": got, "payload": payload}


def constant_false_h2_on_grid(t) -> dict:
    qs = np.arange(1, 7, dtype=float)
    lo, hi = 2.0 * PERIOD, 4.0 * PERIOD
    old = spectrum(t, np.ones_like(t), lo, hi, qs, method="legacy", omega=OMEGA0)
    new = spectrum(t, np.ones_like(t), lo, hi, qs, method="linear_exact", omega=OMEGA0)
    return {
        "legacy_abs_h2": float(abs(old[1, 0])),
        "linear_exact_abs_max": float(np.max(np.abs(new))),
        "legacy_leaks": bool(abs(old[1, 0]) > 1e-3),
        "linear_exact_ok": bool(np.max(np.abs(new)) < 1e-12),
    }


def gh5_reference(sensitivity) -> dict:
    harms = {}
    sources = {}
    for lab, ptag, mtag in ANGLE_PAIRS:
        for tag in (ptag, mtag):
            sources[str((GH5_OUT / f"job_{tag}.json").relative_to(ROOT))] = sha256_file(GH5_OUT / f"job_{tag}.json")
            sources[str((GH5_OUT / f"series_{tag}.npz").relative_to(ROOT))] = sha256_file(GH5_OUT / f"series_{tag}.npz")
        tp, jp = load_series(GH5_OUT, ptag)
        tm, jm = load_series(GH5_OUT, mtag)
        if not np.array_equal(tp, tm):
            raise RuntimeError(f"GH5 time mismatch {lab}")
        harms[lab] = harmonics_from_pair(tp, jp, jm)
    zs = z_from_harmonics(harms)
    rec, fit = fit_record(zs, harms, sensitivity)
    rec.update({
        "status": "gh5_fourangle_reference_corrected_linear_exact",
        "method": "local_GH5_not_Cohen_MTEF",
        "frequency": OMEGA0,
        "window": {"lo_over_T": 2.0, "hi_over_T": 4.0, "T": PERIOD},
        "component": "Jx",
        "domain_parity": "Neel-odd",
        "did_not_call_analyze_fourangle": True,
        "did_not_overwrite_protocolA_gh5_json": True,
        "historical_legacy_rect_jq": HISTORICAL_GH5,
        "sources": sources,
        "holdout_not_evaluated_here": True,
    })
    return rec


def freeze_fourangle(write: bool = True) -> dict:
    t0 = time.perf_counter()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    snapshot_status()
    assert_training_not_holdout()
    if FIT_PATH.is_file():
        chk = validate_frozen_fit()
        if chk.get("ok"):
            return {"status": "already_frozen", "idempotent": True, "sha256": chk["sha256"]}
        raise RuntimeError(f"existing freeze conflicts: {chk}")
    pre = preflight_qed()
    harms = {}
    for lab, ptag, mtag in ANGLE_PAIRS:
        t = pre["series"][ptag][0]
        jp = pre["series"][ptag][1]
        jm = pre["series"][mtag][1]
        harms[lab] = harmonics_from_pair(t, jp, jm)
    zs_lin = z_from_harmonics(harms, "linear_exact", "main")
    zs_quad = z_from_harmonics(harms, "quadratic_exact", "main")
    zs_legacy = z_from_harmonics(harms, "legacy", "main")
    zs_front = z_from_harmonics(harms, "linear_exact", "front")
    zs_back = z_from_harmonics(harms, "linear_exact", "back")
    sens = max(
        1e-18,
        float(np.max(np.abs(np.asarray(zs_lin) - np.asarray(zs_quad)))),
        1e-12 * float(np.linalg.norm(zs_lin)),
    )
    qed_fit_rec, qed_fit = fit_record(zs_lin, harms, sens)
    input_R = {}
    for tag, rec in pre["jobs"].items():
        rval, abs_m, abs_a = input_R_from_job(rec)
        input_R[tag] = {"R_absM_over_abs_mean_a2": rval, "abs_M": abs_m, "abs_mean_a": abs_a,
                        "mean_n": (rec.get("photon_xp_input") or {}).get("mean_n")}
    code = build_code_hashes()
    protocol = {
        "model": "M0_R0",
        "shared_single_optical_mode": True,
        "k_frac": list(K_FRAC),
        "input": "D(0.5)S(0.3 exp(i theta))|0>",
        "omega0": OMEGA0,
        "g": G_ABS,
        "did_not_scale_g_omega_minus_half": True,
        "n_max": N_MAX,
        "n_full": N_FULL,
        "envelope": "1+4+1 coupling switch, not 25 fs free-space pulse",
        "ideal_R": IDEAL_R,
        "R_formula": "sinh(2r)/(2 |alpha|^2)",
    }
    harmonics_dump = {
        "omega0": OMEGA0, "period": PERIOD,
        "fourier": FOURIER_SIGN,
        "primary": "linear_exact",
        "windows": {k: {"lo_over_T": a, "hi_over_T": b} for k, (a, b) in WINDOWS.items()},
        "angles": {lab: harms[lab]["linear_exact"] for lab, _, _ in ANGLE_PAIRS},
        "quadratic_main_odd_x": [packed(z) for z in zs_quad],
        "legacy_main_odd_x": [packed(z) for z in zs_legacy],
        "front_main_odd_x": [packed(z) for z in zs_front],
        "back_main_odd_x": [packed(z) for z in zs_back],
        "linear_minus_quadratic_abs_max": float(np.max(np.abs(np.asarray(zs_lin) - np.asarray(zs_quad)))),
        "linear_minus_legacy_abs_max": float(np.max(np.abs(np.asarray(zs_lin) - np.asarray(zs_legacy)))),
        "neighbor_window_note": "differences are not a rigorous error bound",
        "not_certified_HHG": True,
    }
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "method": "exact_QED",
        "frequency": OMEGA0,
        "window": {"lo_over_T": 2.0, "hi_over_T": 4.0, "T": PERIOD, "lo": 2.0 * PERIOD, "hi": 4.0 * PERIOD},
        "fourier_sign": FOURIER_SIGN,
        "normalization": "1/(tb-ta)",
        "component": "Jx",
        "domain_parity": "Neel-odd",
        "theta_order": list(THETA_ORDER),
        "theta_tags": list(ANGLE_TAGS),
        "training_tags": list(TRAINING_TAGS),
        "sources": pre["sources"],
        "protocol": protocol,
        "code_hashes": code,
        "physics_hashes": pre["physics"],
        "input_moments": input_R,
        "observations": qed_fit_rec["observations"],
        "coefficients": {"B": qed_fit_rec["B"], "A": qed_fit_rec["A"], "C": qed_fit_rec["C"]},
        "residuals": qed_fit_rec["residuals"],
        "derived_metrics": qed_fit_rec["derived_metrics"],
        "relative_L2_residual": qed_fit_rec["relative_L2_residual"],
        "holdout_theta": HOLD_THETA,
        "holdout_prediction": qed_fit_rec["holdout_prediction"],
        "holdout_used_in_fit": False,
        "holdout_criteria": {
            "Esym_max": ESYM_MARK,
            "Esym": "|z_q-z_pred|/max(|z_q|,|z_pred|)",
            "EA": "|z_q-z_pred|/|A|",
            "EA_is_not_the_pass_fail_denominator": True,
            "note": "2% is a working criterion for the three-term angular model on these samples, not Fock/dt/output certification",
        },
        "stage_status": "四角已拟合、独立留出未验证",
        "mechanism_claimed": False,
        "did_not_read_qed_pi4_current": True,
        "n8_p_q_is_not_holdout": True,
        "sensitivity": {
            "linear_minus_quadratic_abs_max": harmonics_dump["linear_minus_quadratic_abs_max"],
            "front_minus_main_abs_max": float(np.max(np.abs(np.asarray(zs_front) - np.asarray(zs_lin)))),
            "back_minus_main_abs_max": float(np.max(np.abs(np.asarray(zs_back) - np.asarray(zs_lin)))),
            "used_as_threshold": sens,
            "not_a_rigorous_bound": True,
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    missing = frozen_schema(freeze)
    if missing:
        raise RuntimeError(f"freeze schema incomplete: {missing}")
    t_grid = pre["series"]["p_0"][0]
    fourier_self = constant_false_h2_on_grid(t_grid)
    if not fourier_self["linear_exact_ok"]:
        raise RuntimeError("linear_exact failed constant-H2 leak test on the production grid")
    gh5 = gh5_reference(sens)
    validation = {
        "constant_false_h2": fourier_self,
        "closed_form_agrees": True,
        "training_source_count": len(pre["sources"]),
        "did_not_read_qed_pi4": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if write:
        dump_json(ANALYSIS / "source_manifest.json", {
            "qed_training": pre["sources"],
            "physics": pre["physics"],
            "code": code,
            "protocol": protocol,
        })
        dump_json(ANALYSIS / "fourangle_harmonics.json", harmonics_dump)
        dump_json(FIT_PATH, freeze)
        sha = sha256_file(FIT_PATH)
        FIT_SHA_PATH.write_text(sha + "  fourangle_fit.json\n", encoding="utf-8")
        dump_json(ANALYSIS / "gh5_reference_corrected.json", gh5)
        after = {rel: sha256_file(ROOT / rel) for rel in pre["sources"]}
        validation["old_inputs_unchanged"] = after == pre["sources"]
        validation["freeze_sha256"] = sha
        if not validation["old_inputs_unchanged"]:
            raise RuntimeError("training inputs changed during postprocess")
        dump_json(ANALYSIS / "validation.json", validation)
        freeze_out = dict(freeze)
        freeze_out["sha256_file"] = str(FIT_SHA_PATH)
        freeze_out["sha256"] = sha
    else:
        freeze_out = freeze
        sha = None
    freeze_out["wall_seconds"] = time.perf_counter() - t0
    freeze_out["status"] = "fourangle_fit_frozen"
    freeze_out["gh5_rho_corrected"] = gh5["derived_metrics"]["rho_absA_over_absB"]
    freeze_out["gh5_arg_corrected"] = gh5["derived_metrics"]["arg_A_conjB_deg"]
    freeze_out["validation"] = validation
    return freeze_out


if __name__ == "__main__":
    rec = freeze_fourangle(write=True)
    print(json.dumps({
        "status": rec.get("status"),
        "stage_status": rec.get("stage_status"),
        "rho": (rec.get("derived_metrics") or {}).get("rho_absA_over_absB"),
        "arg": (rec.get("derived_metrics") or {}).get("arg_A_conjB_deg"),
        "sha256": rec.get("sha256"),
        "wall_seconds": rec.get("wall_seconds"),
        "mechanism_claimed": False,
    }, ensure_ascii=False, indent=2))

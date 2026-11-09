#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent audit of the 2026-10-20 A-baseline. No propagation.

Does not overwrite qinput_minimal_preflight_20261020/report.json,
q8_m0_qinput_minimal_20261020.py, or frozen h8. Not production G1.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATE = "2026-10-25"
OUT_DIR = ROOT / "results" / "q_line" / "qinput_a_baseline_20261025"
LOG_PATH = OUT_DIR / "audit.log"
JSON_PATH = OUT_DIR / "audit.json"
ARCHIVED_REPORT = ROOT / "results" / "q_line" / "qinput_minimal_preflight_20261020" / "report.json"
FROZEN_MODEL = HERE / "q8_m0_model.py"
ASM = HERE / "q8_m0_qinput_minimal_20261020.py"
TST = HERE / "test_q8_m0_qinput_minimal_20261020.py"
TOL = 2.0e-12
PRODUCTION_G1_PASS = False
NOT_PHYSICS_CLAIM = True


def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def exclusive_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(path), flags)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(jsonable(payload), fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def check(name: str, cond, detail="") -> dict:
    return dict(name=name, ok=bool(cond), detail=str(detail))


# --- resource-matched classical contrast (algebra only) -------------------
def sv_number_moments(nbar: float) -> dict:
    nbar = float(nbar)
    return dict(
        nbar=nbar,
        var_n=2.0 * nbar * (nbar + 1.0),
        abs_aa=float(np.sqrt(nbar * (nbar + 1.0))),
    )


def thermal_or_rpe_number_moments(nbar: float) -> dict:
    nbar = float(nbar)
    var_I = nbar * nbar
    return dict(
        nbar=nbar,
        var_I=var_I,
        var_n=nbar + var_I,
        note="exponential intensity + Poisson = thermal Var(n)=n(n+1); cannot match SV",
    )


def gamma_intensity_match_sv_number(nbar: float) -> dict:
    nbar = float(nbar)
    shape = nbar / (2.0 * nbar + 1.0)
    scale = 2.0 * nbar + 1.0
    e_I = shape * scale
    var_I = shape * scale * scale
    return dict(
        nbar=nbar,
        shape=shape,
        scale=scale,
        e_I=e_I,
        var_I=var_I,
        var_n=e_I + var_I,
        note="matches <n> and Var(n) of SV; random phase drops squeeze phase; |<a^2>|<=<n>",
    )


def resource_match_selftest(nbar: float = 1.0) -> dict:
    sv = sv_number_moments(nbar)
    rpe = thermal_or_rpe_number_moments(nbar)
    gam = gamma_intensity_match_sv_number(nbar)
    tests = [
        check("SV Var(n)=2 n (n+1)", abs(sv["var_n"] - 2.0 * nbar * (nbar + 1.0)) < TOL, sv["var_n"]),
        check("RPE/thermal Var(n)=n(n+1) != SV", abs(rpe["var_n"] - sv["var_n"]) > 0.5 * nbar, f"rpe={rpe['var_n']} sv={sv['var_n']}"),
        check("Gamma E[I]=nbar", abs(gam["e_I"] - nbar) < TOL, gam["e_I"]),
        check("Gamma Var(n)=SV Var(n)", abs(gam["var_n"] - sv["var_n"]) < 1e-12, gam["var_n"]),
        check("positive-P |<a^2>|<=nbar < SV |<a^2>|", sv["abs_aa"] > nbar + 1e-12, f"|aa|_SV={sv['abs_aa']} nbar={nbar}"),
        check("nbar=0 edge Gamma shape=0", abs(gamma_intensity_match_sv_number(0.0)["shape"]) < TOL),
    ]
    return dict(
        nbar=nbar,
        squeezed_vacuum=sv,
        rpe_or_thermal=rpe,
        gamma_intensity_positive_P=gam,
        positive_P_bound="|<a^2>| <= <n> for any P>=0 mixture",
        squeezed_vacuum_aa="|<a^2>|=sqrt(n(n+1)) > n",
        tests=tests,
        all_ok=all(t["ok"] for t in tests),
        not_a_complete_classical_exclusion=True,
    )


# --- photon parity of truncated input kets --------------------------------
def photon_parity_selftest() -> dict:
    sys.path.insert(0, str(HERE))
    import q8_m0_qinput_minimal_20261020 as qm

    n_max = 24
    r, phi = 0.40, 0.70
    sv = qm.squeezed_vacuum_coefficients(r, phi, n_max)
    coh = qm.coherent_coefficients(0.55 + 0.20j, n_max)
    n = np.arange(n_max + 1)
    pi_sv = ((-1.0) ** n) * sv
    pi_coh = ((-1.0) ** n) * coh
    coh_minus = qm.coherent_coefficients(-(0.55 + 0.20j), n_max)
    odd = float(np.linalg.norm(sv[1::2]))
    tests = [
        check("SV odd-n amplitude ~ 0", odd < 1e-14, odd),
        check("Pi SV = SV (any squeeze angle)", float(np.linalg.norm(pi_sv - sv)) < 1e-14),
        check("Pi |alpha> = |-alpha>", float(np.linalg.norm(pi_coh - coh_minus)) < 1e-12),
        check("Pi |alpha> != |alpha| unless alpha=0", float(np.linalg.norm(pi_coh - coh)) > 0.1),
    ]
    return dict(
        n_max=n_max,
        squeeze_r=r,
        squeeze_phi=phi,
        sv_norm=float(np.linalg.norm(sv)),
        sv_odd_norm=odd,
        tests=tests,
        all_ok=all(t["ok"] for t in tests),
        domain_note="pure SV is photon-parity even; intensity even under inversion-closed shared mode",
    )


def hash_ledger(archived: dict) -> dict:
    live = {
        "q8_m0_model.py": sha256_file(FROZEN_MODEL),
        "q8_m0_qinput_minimal_20261020.py": sha256_file(ASM),
        "test_q8_m0_qinput_minimal_20261020.py": sha256_file(TST),
    }
    src = archived.get("files") or {}
    orig = (archived.get("frozen_model_and_geometry_sources") or {}).get("model_module") or {}
    want = {
        "q8_m0_qinput_minimal_20261020.py": src.get("q8_m0_qinput_minimal_20261020.py"),
        "test_q8_m0_qinput_minimal_20261020.py": src.get("test_q8_m0_qinput_minimal_20261020.py"),
        "q8_m0_model.py": orig.get("sha256"),
    }
    matches = {k: (live[k] == want[k]) for k in live}
    return dict(live=live, archived_want=want, matches=matches, all_match=all(matches.values()))


def rerun_static_seven() -> dict:
    sys.path.insert(0, str(HERE))
    import test_q8_m0_qinput_minimal_20261020 as tmod

    tmod.EVIDENCE.clear()
    buf = io.StringIO()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(tmod.QuantumInputStaticTests)
    t0 = time.perf_counter()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    wall = time.perf_counter() - t0
    names = [str(x) for x in result.failures] + [str(x) for x in result.errors]
    return dict(
        tests_run=int(result.testsRun),
        failures=int(len(result.failures)),
        errors=int(len(result.errors)),
        all_checks_pass=bool(result.wasSuccessful()),
        wall_seconds=wall,
        failed_names=names,
        evidence=tmod.EVIDENCE,
        k_fractional=[0.173, 0.271],
        test_g=0.02,
        test_omega=0.05,
        test_n_max=8,
        electron_dimension=70,
        joint_dimension_nmax8=70 * 9,
        spin_sector_36_available_not_used=True,
    )


def electron_symmetry_ledger() -> dict:
    """Zero-field electronic P and the dh8 `del neel` fact. Static only."""
    sys.path.insert(0, str(HERE))
    import q8_m0_model as m0

    geo = m0.build_geometry()
    k = geo.k_cart([0.173, 0.271])
    u = np.asarray(m0.op_P(), complex)
    unitary_err = float(np.max(np.abs(u @ u.conj().T - np.eye(u.shape[0]))))
    residuals = {}
    for neel in (-1, 1):
        for tag, kk in (("k", k), ("-k", -k)):
            lhs = u @ m0.h8(geo, m0.R0, kk, neel) @ u.conj().T
            rhs = m0.h8(geo, m0.R0, -kk, -neel)
            residuals[f"N{neel:+d}_{tag}"] = float(np.max(np.abs(lhs - rhs)))
    j_plus = np.asarray(m0.current_ops(geo, m0.R0, k, 1), complex)
    j_minus = np.asarray(m0.current_ops(geo, m0.R0, k, -1), complex)
    neel_current_gap = float(np.max(np.abs(j_plus - j_minus)))
    return dict(
        op_P_unitarity_max_abs=unitary_err,
        P_residuals=residuals,
        P_residual_max=max(residuals.values()),
        P_is_machine_precision=bool(max(residuals.values()) < 1e-14),
        current_ops_neel_gap_max_abs=neel_current_gap,
        current_neel_blind=bool(neel_current_gap == 0.0),
        note="zero-field electronic P only; not a finite-pulse or single-side selection rule",
    )


def frozen_spec_notes() -> dict:
    return dict(
        electronic_space="A: frozen h8, single k, half-filled 4-fermion, dim 70",
        coupling="Peierls projected displacement P e^{iqX} P, X=a+a†, Schrödinger picture",
        g_meaning="abstract single-mode vector-potential parameter in 1/Å, not experimental V_eff",
        not_macroscopic_BSV=True,
        layer1="single-k method benchmark only",
        layer2="k,-k must share one photon mode; 70^2*(n_max+1); two independent 70-runs are not equivalent",
        observables={
            "mean_J": "<J_q>",
            "source_corr": "<J_q^dagger J_q>",
            "pol_cross": "<J_parallel^dagger J_perp>",
            "connected": "<J^dagger J> - <J>^* <J>",
            "forbidden_compare": "exact <J^dagger J> vs semiclassical |<J>|^2 as quantum advantage",
        },
        conditional_domain={
            "statement": "inversion-closed + shared mode + photon-parity-even input => <J> flips, <J^dagger J> even in N",
            "holds_for": ["pure squeezed vacuum any angle", "thermal", "uniform random-phase mix"],
            "does_not_hold_for": ["fixed-alpha coherent (maps (N,alpha)->(-N,-alpha))"],
            "not_a_unidirectional_experiment_theorem": True,
            "do_not_expect": "S_q(+N) != S_q(-N) as a new effect in this closed model",
        },
        literature_limit={
            "de_la_Pena_JCTC_2025": "squeezed-coherent alpha0~10.46, not pure BSV; mean-dipole Fourier, not source two-time corr",
            "do_not_call": "Cohen 2025 Husimi-Q exact comparison",
            "rasputnyi_NatPhys_2024": "macroscopic single-spatiotemporal-mode BSV driving solid HHG exists; does not turn a few-photon large-g test into that experiment",
        },
        semiclassical_ledger_required=[
            "distribution: Husimi-Q / Wigner-symmetric / positive-P / Gamma-intensity",
            "light-field feedback on electrons: kept or dropped",
            "source observable: mean J vs J^dagger J, one window, one operator order",
            "averaging order: average-then-modulus vs modulus-then-average",
        ],
        production_G1_pass=PRODUCTION_G1_PASS,
        not_physics_claim=NOT_PHYSICS_CLAIM,
        no_propagation=True,
    )


def main() -> int:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log("audit start")
    if JSON_PATH.exists():
        log("refuse overwrite " + str(JSON_PATH))
        raise SystemExit("refuse overwrite " + str(JSON_PATH))
    if not ARCHIVED_REPORT.exists():
        raise SystemExit("missing archived 20261020 report")
    archived = json.loads(ARCHIVED_REPORT.read_text(encoding="utf-8"))
    hashes = hash_ledger(archived)
    log("hashes all_match=" + str(hashes["all_match"]))
    resource = resource_match_selftest(1.0)
    log("resource all_ok=" + str(resource["all_ok"]))
    parity = photon_parity_selftest()
    log("parity all_ok=" + str(parity["all_ok"]))
    seven = rerun_static_seven()
    log("seven pass=" + str(seven["all_checks_pass"]) + " wall=" + f"{seven['wall_seconds']:.3f}")
    esym = electron_symmetry_ledger()
    log("electron_P_residual=" + f"{esym['P_residual_max']:.3e}" + " current_neel_blind=" + str(esym["current_neel_blind"]))
    payload = dict(
        record_date=DATE,
        model_id="M0_single_k_four_fermion_quantum_input_A_baseline_audit_v1",
        production_G1_pass=PRODUCTION_G1_PASS,
        not_physics_claim=NOT_PHYSICS_CLAIM,
        no_propagation=True,
        did_not_overwrite_20261020=True,
        archived_report=str(ARCHIVED_REPORT.relative_to(ROOT)).replace("\\", "/"),
        archived_all_checks_pass=bool(archived.get("all_checks_pass")),
        hash_ledger=hashes,
        resource_match=resource,
        photon_parity=parity,
        independent_seven=seven,
        electron_symmetry=esym,
        frozen_spec=frozen_spec_notes(),
        interpreter=sys.executable,
        numpy=np.__version__,
        wall_seconds=time.perf_counter() - t0,
        all_ok=bool(
            hashes["all_match"]
            and resource["all_ok"]
            and parity["all_ok"]
            and seven["all_checks_pass"]
            and esym["P_is_machine_precision"]
        ),
    )
    exclusive_write_json(JSON_PATH, payload)
    log("wrote " + str(JSON_PATH))
    log("all_ok=" + str(payload["all_ok"]))
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

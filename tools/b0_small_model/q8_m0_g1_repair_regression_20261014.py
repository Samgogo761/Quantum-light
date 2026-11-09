"""Independent complex Gaussian and no-overwrite regressions; no propagation."""
import json
from pathlib import Path
import tempfile
import numpy as np
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_io_crosscheck_review_20261014/repair_regression.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    omega = np.kron(np.eye(2), [[0., 1.], [-1., 0.]])
    rng = np.random.default_rng(20261014)
    mixes = [np.array([[1, 1j], [1j, 1]]) / np.sqrt(2), np.eye(2)]
    for _ in range(20):
        z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        mixes.append(np.linalg.qr(z)[0])
    checks = []
    for k, w in enumerate(mixes):
        r = np.array([0.7, 0.2]) if k == 0 else rng.uniform(0., 1., 2)
        u, v = w @ np.diag(np.cosh(r)), w @ np.diag(np.sinh(r))
        n, m = v.conj() @ v.T, u @ v.T
        # Direct Hermitian quadrature coefficients of the input annihilators.
        f = np.empty((4, 2), complex)
        f[::2] = (u + v.conj()) / np.sqrt(2)
        f[1::2] = (u - v.conj()) / (1j * np.sqrt(2))
        direct = (f @ f.conj().T).real
        got = q.two_mode_quad_check(n, m)
        delta = float(np.max(np.abs(direct - got["covariance_x1_p1_x2_p2"])))
        eig = float(np.linalg.eigvalsh(direct + 0.5j * omega)[0])
        checks.append({"case": k, "covariance_maxabs_difference": delta,
                       "direct_uncertainty_min_eig": eig,
                       "repaired_uncertainty_min_eig": got["min_eig_cov_plus_i_omega_over_2"]})
        assert delta < 2e-14
        assert abs(eig - got["min_eig_cov_plus_i_omega_over_2"]) < 2e-14
    # Symmetry violations must remain visible, not be silently certified away.
    bad = q.two_mode_quad_check([[1., 0.1j], [0., 1.]], [[0., 0.2], [0., 0.]])
    assert bad["input_n_hermiticity_maxabs"] == 0.1
    assert bad["input_m_symmetry_maxabs"] == 0.2
    invalid_rejected = 0
    for n, m in ((np.eye(3), np.eye(3)), (np.full((2, 2), np.nan), np.eye(2))):
        try:
            q.two_mode_quad_check(n, m)
        except ValueError:
            invalid_rejected += 1
    assert invalid_rejected == 2
    with tempfile.TemporaryDirectory(dir=ROOT / "results/q_line") as tmp:
        path = Path(tmp) / "audit.json"
        write_new_json(path, {"kept": 1})
        try:
            write_new_json(path, {"kept": 2})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Old audit was overwritten")
        assert json.loads(path.read_text(encoding="utf-8")) == {"kept": 1}
        assert not list(Path(tmp).glob(".audit-*"))
    report = {"all_checks_pass": True, "canonical_complex_gaussian_cases": checks,
              "nonhermitian_n_and_nonsymmetric_m_defects_reported": True,
              "invalid_input_cases_rejected": invalid_rejected,
              "exclusive_atomic_writer_preserves_first_result": True,
              "provenance_sha256": provenance([__file__, q.__file__,
                  Path(__file__).with_name("q8_m0_audit_artifacts_20261014.py")]),
              "scope": "Covariance diagnostic and audit artifact protection only; no changes to U,V,n,m or production equations.",
              "cache_policy": "Changed input_close source hash intentionally invalidates old execution identity; do not edit old manifests to bypass the check."}
    write_new_json(OUT, report)
    print(json.dumps({"all_checks_pass": True, "cases": len(checks),
                      "max_covariance_error": max(c["covariance_maxabs_difference"] for c in checks)}))


if __name__ == "__main__":
    main()

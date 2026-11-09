"""Frozen n,m 2x2 tables from existing U,V caches. No new propagation.

Definitions (handoff / protocol):
    n = V* V^T
    m = U V^T     (U already includes U0)

Sources
-------
- 9-cycle kernel prod vs ref: kernel_correction_9cycles.npz
- 9 vs 11 cycle production maps: gram_grouping_9_vs_11.npz
- optional argv 'timestep': assemble 9-cycle macro1024 vs macro2048 from
  existing adjoint rows (no vertices).

Does not overwrite archived JSON.
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

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_nm_tables_20261017"
OUTJSON = OUTDIR / "nm_tables.json"
PROGRESS = OUTDIR / "progress.txt"
KNPZ = ROOT / "results/q_line/g1_kernel_correction_20261017/kernel_correction_9cycles.npz"
GNPZ = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.npz"
T0 = time.time()


def elapsed():
    return time.time() - T0


def prog(msg):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"{elapsed():.1f}s {msg}\n"
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def pack(m):
    m = np.asarray(m, complex)
    return {
        "real": np.real(m).tolist(),
        "imag": np.imag(m).tolist(),
        "abs": np.abs(m).tolist(),
        "maxabs": float(np.max(np.abs(m))),
    }


def frozen_n(v):
    return np.conj(v) @ v.T


def frozen_m(u, v):
    return u @ v.T


def pair(u_a, v_a, u_b, v_b):
    na, nb = frozen_n(v_a), frozen_n(v_b)
    ma, mb = frozen_m(u_a, v_a), frozen_m(u_b, v_b)
    dn, dm = nb - na, mb - ma
    return {
        "n_a": pack(na),
        "n_b": pack(nb),
        "dn": pack(dn),
        "m_a": pack(ma),
        "m_b": pack(mb),
        "dm": pack(dm),
        "cross_pol": {
            "n_a_01_abs": float(abs(na[0, 1])),
            "n_b_01_abs": float(abs(nb[0, 1])),
            "dn_01_abs": float(abs(dn[0, 1])),
            "m_a_01_abs": float(abs(ma[0, 1])),
            "m_b_01_abs": float(abs(mb[0, 1])),
            "dm_01_abs": float(abs(dm[0, 1])),
            "n_a_00_abs": float(abs(na[0, 0])),
            "m_a_00_abs": float(abs(ma[0, 0])),
            "dn_00_abs": float(abs(dn[0, 0])),
            "dm_00_abs": float(abs(dm[0, 0])),
        },
        "note_vvT_is_not_frozen_n": pack((v_b @ v_b.T) - (v_a @ v_a.T)),
    }


def extract_existing():
    prog("load kernel npz")
    kz = np.load(KNPZ)
    prog("kernel keys=" + ",".join(kz.files))
    kernel = pair(kz["u_prod"], kz["v_prod"], kz["u_ref"], kz["v_ref"])
    kernel["label"] = "9cycles_macro1024 kernel prod vs ref"
    kernel["a"] = "prod"
    kernel["b"] = "ref"
    del kz
    prog("load gram npz")
    gz = np.load(GNPZ)
    prog("gram keys=" + ",".join(gz.files))
    window = pair(gz["u9"], gz["v9"], gz["u11"], gz["v11"])
    window["label"] = "production map 9 vs 11 cycles, macro1024"
    window["a"] = "9cycles"
    window["b"] = "11cycles"
    del gz
    return kernel, window


def assemble_timestep():
    from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
    from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
    from q8_m0_g1_gram_grouping_20261017 import assemble
    from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
    from q8_m0_classical import p1_a2_protocol
    from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi
    from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
    import q8_m0_gamma12_transpose_input_refine_20261008 as base

    prog("timestep assemble start")
    path1024, _d1 = require_cache(1024)
    path2048, _d2 = require_cache(2048)
    _saved, cache, _t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    eta = eta_of(gamma)
    prog("assemble macro1024")
    _g, u1024, v1024 = assemble(cache, gamma, pol, phi_of, eta, load_adjoint(path1024))
    prog("assemble macro2048")
    _g, u2048, v2048 = assemble(cache, gamma, pol, phi_of, eta, load_adjoint(path2048))
    out = pair(u1024, v1024, u2048, v2048)
    out["label"] = "production map 9cycles macro1024 vs macro2048"
    out["a"] = "macro1024"
    out["b"] = "macro2048"
    out["sources"] = {"rows1024": str(path1024), "rows2048": str(path2048)}
    return out


def main():
    if OUTJSON.exists() and "timestep" not in sys.argv:
        raise FileExistsError(OUTJSON)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("start\n", encoding="utf-8")
    kernel, window = extract_existing()
    report = {
        "task": "frozen n=V*V^T and m=UV^T 2x2 tables",
        "date": "2026-10-17",
        "production_G1_pass": False,
        "definitions": {
            "n": "V.conj() @ V.T  (= V* V^T)",
            "m": "U @ V.T  (U includes U0)",
            "WorkBuddy_n_change_was": "V @ V.T  (not frozen n; imag diagonal nonzero)",
        },
        "kernel_9cycles_prod_vs_ref": kernel,
        "window_9_vs_11_production": window,
        "elapsed_s": float(elapsed()),
    }
    if "timestep" in sys.argv:
        report["timestep_1024_vs_2048_production"] = assemble_timestep()
        report["elapsed_s"] = float(elapsed())
        out = OUTDIR / "nm_tables_with_timestep.json"
        if out.exists():
            raise FileExistsError(out)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        prog("saved " + str(out))
    else:
        OUTJSON.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        prog("saved " + str(OUTJSON))
    print("done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

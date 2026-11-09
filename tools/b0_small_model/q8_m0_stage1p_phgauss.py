#!/usr/bin/env python3
"""Stage-1P: P1-dressed p-h + photon Gaussian prototype.

No large BZ, no production V, no G_self input, no ΔC patch.
Weak coupling must recover C^>, R^A, A2/B2, Kubo. Φ itself must be symplectic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import p1_a2_protocol  # noqa: E402
from q8_m0_io_audit import audit_grid  # noqa: E402
from q8_m0_loop import ResummationNotChosenError, apply_resummation_to_output  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_phgauss import (  # noqa: E402
    NO_GSELF_NOTE,
    PHGAUSS_ID,
    born_channel_from_ph,
    c_greater_match,
    h2_two_port_packets,
    io_from_phi,
    propagate_phi,
)
from q8_m0_s1_kernel import pack_complex, parallel_perp_basis  # noqa: E402
from q8_m0_stage1 import stage1_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1p_phgauss_20260923"
MU = -0.009010620372198276
NTPC = 32
G1P_ID = "phgauss_canonical_prototype_20260923"


def log(msg: str) -> None:
    print(f"[stage1p] {msg}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    pref = eta2_ncell(geo)
    eta = float(np.sqrt(pref))
    e_par, e_perp = parallel_perp_basis(geo)
    t0 = time.perf_counter()
    prod_blocked = False
    try:
        apply_resummation_to_output()
    except ResummationNotChosenError:
        prod_blocked = True

    kf = np.array([[0.0, 0.0], [0.17, 0.41]])
    log(f"CJJ nk=2 ntpc={NTPC}")
    rec = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    x, w, md = rec["X_full"], rec["w_k"], rec["mean_D_t"]
    dt = float(rec["dt"])
    times = rec["times"]
    log(f"nt={x.shape[0]} eta={eta:.6g}")

    cg = c_greater_match(x, w)
    born = born_channel_from_ph(x, w, md, dt, geo, prot, pref)
    log(f"C> match {cg['maxabs']:.3e}")
    log(
        f"R^A match {born['maxabs_I_ph_minus_RA']:.3e} "
        f"Kubo {born['kubo_residual_abs']:.3e} "
        f"|A2_ph|={born['A2_ph']['abs']:.6g}"
    )

    omega, dw = audit_grid(prot, harm_max=8.0)
    flist, labs = h2_two_port_packets(times, prot.omega0, prot.t_pulse, e_perp, dw, omega)
    ev = propagate_phi(x, w, md, flist, dt, eta)
    io = io_from_phi(ev["Phi"], ev["n_ph"], ev["n_B"])
    log(
        f"symplectic {ev['symplectic_maxabs']:.3e} "
        f"K-ham {ev['hamiltonian_K_maxabs']:.3e} "
        f"|SSd-TTd+Cel-I|={io['ident_minus_I_maxabs']:.3e} "
        f"diag={io['diag_ident']}"
    )

    weak_ok = bool(
        cg["maxabs"] < 1e-12
        and born["maxabs_I_ph_minus_RA"] < 1e-9
        and born["kept_order_closed"]
    )
    can_ok = bool(ev["symplectic_maxabs"] < 1e-8 and io["ident_minus_I_maxabs"] < 1e-7)
    verdict = {
        "weak_coupling_recovers_CJJ_Kubo": weak_ok,
        "evolution_is_symplectic": bool(ev["symplectic_maxabs"] < 1e-8),
        "io_identity_from_structure": bool(io["ident_minus_I_maxabs"] < 1e-7),
        "prototype_pass": bool(weak_ok and can_ok),
        "G_self": NO_GSELF_NOTE,
        "production_n_m_V": "not_written",
        "quasi_boson_occupation": "not_this_round",
        "bz": "not_this_round",
    }
    log(f"verdict {verdict}")

    payload = {
        "g1p_id": G1P_ID,
        "phgauss_id": PHGAUSS_ID,
        "spec": stage1_spec(),
        "frozen": {
            "theory_main": "P1_dressed_particle_hole_photon_canonical_Gaussian",
            "expectation_resolvent": "diagnostic",
            "no_G_self_input": True,
            "no_delta_C_patch": True,
            "no_T2": True,
            "no_nonlinear": True,
            "no_large_BZ": True,
            "no_production_V": True,
            "nk_is_not_N_cell": True,
        },
        "production_V_blocked": prod_blocked,
        "grid": {"n_k": int(kf.shape[0]), "n_t": int(x.shape[0]), "ntpc": NTPC, "kf": kf.tolist()},
        "c_greater": cg,
        "born": born,
        "evolution": {
            "n_ph": ev["n_ph"],
            "n_B": ev["n_B"],
            "photon_labels": labs,
            "symplectic_maxabs": ev["symplectic_maxabs"],
            "hamiltonian_K_maxabs": ev["hamiltonian_K_maxabs"],
            "eta": ev["eta"],
        },
        "io": {
            "ident_minus_I_maxabs": io["ident_minus_I_maxabs"],
            "diag_ident": io["diag_ident"].tolist(),
            "S": {"re": np.real(io["S"]).tolist(), "im": np.imag(io["S"]).tolist()},
            "T": {"re": np.real(io["T"]).tolist(), "im": np.imag(io["T"]).tolist()},
            "C_el": pack_complex(complex(np.trace(io["C_el"]))),
        },
        "verdict": verdict,
        "cjj_wall_s": rec["wall_seconds"],
        "human_gate": False,
        "nk96": "not_run",
        "wall_s": time.perf_counter() - t0,
    }
    dump(OUT / "stage1p_phgauss.json", payload)
    print(json.dumps({
        "wrote": str(OUT / "stage1p_phgauss.json"),
        "verdict": verdict,
        "born_match": born["maxabs_I_ph_minus_RA"],
        "kubo": born["kubo_residual_abs"],
        "symplectic": ev["symplectic_maxabs"],
        "io_ident": io["ident_minus_I_maxabs"],
        "wall_s": payload["wall_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

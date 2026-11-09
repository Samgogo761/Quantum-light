"""4x4 from existing 2x2 n,m,R_N,R_A. No U,V load. No new propagation.

G is reconstructed as I+R_N+conj(n), i.e. I+R_N+VV^H under n=V*V^T.
Omega_0 = standard CCR. Omega_eff from live G and R_A.
production_G1_pass=false. Does not overwrite archives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import q8_m0_g1_q_diagnostics_20261018 as qa
import q8_m0_g1_q_slices_bcde_20261018 as sl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_q_diagnostics_20261018/q_4x4_from_existing_nm.json"
NM = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables.json"
GRAM = ROOT / "results/q_line/g1_gram_grouping_20261017/gram_grouping_9_vs_11.json"
NMTS = ROOT / "results/q_line/g1_nm_tables_20261017/nm_tables_with_timestep.json"


def report_one(n, m, rn, ra, label, g_note):
    Gcorr = np.eye(2, dtype=complex) + rn + np.conj(n)
    Gccr = np.eye(2, dtype=complex) + rn
    # Default constructor is repaired; Gcorr kept only as a note. Do not re-write archived JSON.
    sig = sl.sigma_xp_repaired(n, m, Gccr)
    om0 = sl.omega0_xp()
    ome = sl.omega_eff_xp(Gccr, ra)
    e0, v0 = sl.min_eigh(sig + 0.5j * om0)
    ee, ve = sl.min_eigh(sig + 0.5j * ome)
    es, vs = sl.min_eigh(sig)
    pt = sl.pt_mode1(sig)
    ept0, _ = sl.min_eigh(pt + 0.5j * om0)
    epte, _ = sl.min_eigh(pt + 0.5j * ome)
    return {
        "label": label,
        "G_reconstruction": g_note,
        "ordering": "X0,P0,X1,P1",
        "Omega_0": "standard CCR; not the live commutator",
        "Omega_eff": "induced by [a,a^dag]=I+R_N and R_A; not by <aa^dag>",
        "maxabs_Omega_eff_minus_Omega_0": float(np.max(np.abs(ome - om0))),
        "maxabs_R_N": float(np.max(np.abs(rn))),
        "maxabs_R_A": float(np.max(np.abs(ra))),
        "min_eig_sigma": es,
        "min_eig_sigma_plus_i_Omega0_over_2": e0,
        "min_eig_sigma_plus_i_Omega_eff_over_2": ee,
        "min_eig_PT_mode1_plus_i_Omega0_over_2": ept0,
        "min_eig_PT_mode1_plus_i_Omega_eff_over_2": epte,
        "PT_is_sensitivity_not_PPT_or_entanglement": True,
        "williamson_Omega_0": sl.williamson(sig, om0),
        "williamson_Omega_eff": sl.williamson(sig, ome),
        "Q0": qa.q_mu(n[0, 0], m[0, 0], rn[0, 0]),
        "Q1": qa.q_mu(n[1, 1], m[1, 1], rn[1, 1]),
        "production_G1_pass": False,
        "not_a_squeezing_claim": True,
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    nm = json.loads(NM.read_text(encoding="utf-8"))
    gram = json.loads(GRAM.read_text(encoding="utf-8"))
    ts = json.loads(NMTS.read_text(encoding="utf-8"))
    note = "sigma uses Gcorr=I+R_N+conj(n)=<aa^dag>; Omega_eff uses Gccr=I+R_N=[a,a^dag]"
    k = nm["kernel_9cycles_prod_vs_ref"]
    w = nm["window_9_vs_11_production"]
    t = ts["timestep_1024_vs_2048_production"]
    rn9 = qa.cmat(gram["full_RN_minus_I"]["9_cycles"])
    ra9 = qa.cmat(gram["full_RA"]["9_cycles"])
    rn11 = qa.cmat(gram["full_RN_minus_I"]["11_cycles"])
    ra11 = qa.cmat(gram["full_RA"]["11_cycles"])
    rn2048 = np.zeros((2, 2), complex)
    rn2048[0, 0] = 1.6062753275036812e-05
    configs = {
        "nine_1024_prod": report_one(qa.cmat(k["n_a"]), qa.cmat(k["m_a"]), rn9, ra9, "9cycles_macro1024_prod", note),
        "eleven_1024_prod": report_one(qa.cmat(w["n_b"]), qa.cmat(w["m_b"]), rn11, ra11, "11cycles_macro1024_prod", note),
        "nine_2048_prod": {
            "label": "9cycles_macro2048_prod",
            "4x4": "unavailable: R_A missing and U,V not assembled",
            "production_G1_pass": False,
        },
        "nine_kernel_ref": "deferred to npz slice D; do not reuse prod R_A",
    }
    report = {
        "task": "4x4 from existing 2x2; Omega_0 vs Omega_eff",
        "date": "2026-10-18",
        "production_G1_pass": False,
        "vacuum_selfcheck": sl.vacuum_selfcheck(),
        "error_estimate_labels": {
            "timestep_ratio_1_over_15": "conditional_error_estimate_not_strict_bound",
            "four_window_swing": "conditional_error_estimate_not_strict_bound",
        },
        "configs": configs,
        "note_2048": "U,V not on disk; 4x4 unavailable",
    }
    del t
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    raise SystemExit(main())

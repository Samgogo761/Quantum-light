"""P3: Full Gram regrouping from the stored 9- and 11-cycle adjoint rows.

No new propagation. Contracts the existing adjoint rows against every input
column (photon pieces by frequency group, plus all 1976 dark columns),
assembles the FULL map U = U0 + ΔU, V = ΔV for both windows, and decomposes
the window difference of R_N = UU† − VV† − I and R_A = UVᵀ − VUᵀ by column
group. Cross terms between groups/columns are reported separately so the
group reconstruction matches the full matrices exactly.

Outputs JSON + npz into results/q_line/g1_gram_grouping_20261017/.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_boundary_sensitivity_20261015 import contract_stacked, frequencies, stack_forces
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache
from q8_m0_g1_gamma12_column_order_20261015 import CACHE
from q8_m0_g1_gamma12_ir_tail_20261015 import load_adjoint
from q8_m0_g1_gamma12_richardson_adjoint_20261015 import contract_initial
from q8_m0_g1_gamma12_tail_columns_20261014 import load_rows
from q8_m0_p1_h2_finite_kcv_input_close_20261005 import eta_of
from q8_m0_k1_packet import phi_unnormalized, omega_grid, normalize_phi

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/q_line/g1_gram_grouping_20261017"
OUTJSON = OUTDIR / "gram_grouping_9_vs_11.json"
OUTNPZ = OUTDIR / "gram_grouping_9_vs_11.npz"

DARK_TOP = [133, 131, 127, 140]  # strongest dark columns by Gram change (audit 2026-10-17)


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def adjoint_path(cycles):
    if cycles == 9.0:
        path, _digest = require_cache(1024)
        return path
    label = f"{cycles:.1f}".replace(".", "p")
    return ROOT / "results/q_line/g1_gamma12_time_order_20261015" / f"adjoint_rows_macro1024_cycles{label}.npz"


def assemble(cache, gamma, pol, phi_of, eta, rows):
    """Return (groups, u_full, v_full) with groups = list of dicts:
    {name, cols: (U_block, V_block, U0_block), kind}."""
    row_d, force_d, row_u, force_u = rows
    sd, su = stack_forces(force_d), stack_forces(force_u)
    groups = []

    # Dark columns (1976).
    b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
    a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)
    dd, vd = contract_initial(row_d, b, bd, a)
    du, vu = contract_initial(row_u, b, bd, a)
    groups.append({"name": "dark_1976", "kind": "dark",
                   "U": dd - du, "V": vd - vu,
                   "U0": np.zeros((2, 1976), complex)})

    # Photon pieces by frequency group.
    pieces, edge = frequencies(cache, gamma)
    for name, om, wt in pieces:
        for start in range(0, len(om), 400):
            stop = min(start + 400, len(om))
            ph, u0 = base.generated_photon_block(
                cache, gamma, om[start:stop], wt[start:stop], pol, phi_of, eta)
            d = contract_stacked(row_d, sd, ph.initial.b, ph.initial.bdag,
                                 ph.initial.A, ph.free_A0, ph.omega)
            u = contract_stacked(row_u, su, ph.initial.b, ph.initial.bdag,
                                 ph.initial.A, ph.free_A0, ph.omega)
            groups.append({"name": f"{name}[{start}:{stop}]", "kind": f"photon:{name}",
                           "U": d[0] - u[0], "V": d[1] - u[1], "U0": u0})
        print("assembled", name, flush=True)
    u_full = np.concatenate([g["U"] + g["U0"] for g in groups], axis=1)
    v_full = np.concatenate([g["V"] for g in groups], axis=1)
    return groups, u_full, v_full


def gram(u, v):
    return u @ u.conj().T - v @ v.conj().T, u @ v.T - v @ u.T


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OUTJSON.exists():
        raise FileExistsError(OUTJSON)
    path9 = adjoint_path(9.0)
    path11 = adjoint_path(11.0)
    rows9 = load_adjoint(path9)
    rows11 = load_adjoint(path11)
    _saved, cache, _t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, nrm = normalize_phi(grid, phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, nrm)
    eta = eta_of(gamma)

    groups9, u9, v9 = assemble(cache, gamma, pol, phi_of, eta, rows9)
    print("assembled 9 cycles", flush=True)
    groups11, u11, v11 = assemble(cache, gamma, pol, phi_of, eta, rows11)
    print("assembled 11 cycles", flush=True)

    eye = np.eye(2)
    rn9, ra9 = gram(u9, v9)
    rn11, ra11 = gram(u11, v11)
    rn9, ra9 = rn9 - eye, ra9
    rn11 = rn11 - eye
    d_rn, d_ra = rn11 - rn9, ra11 - ra9

    # Per-group self contribution to the Gram change (diagonal blocks).
    by_kind = {}
    for g9, g11 in zip(groups9, groups11):
        assert g9["name"] == g11["name"]
        kind = g9["kind"]
        uf9, uf11 = g9["U"] + g9["U0"], g11["U"] + g11["U0"]
        self_d = (uf11 @ uf11.conj().T - g11["V"] @ g11["V"].conj().T
                  - uf9 @ uf9.conj().T + g9["V"] @ g9["V"].conj().T)
        an_d = (uf11 @ g11["V"].T - g11["V"] @ uf11.T
                - uf9 @ g9["V"].T + g9["V"] @ uf9.T)
        rec = by_kind.setdefault(kind, {"normal": np.zeros((2, 2), complex),
                                        "anomalous": np.zeros((2, 2), complex)})
        rec["normal"] += self_d
        rec["anomalous"] += an_d

    # Cross-term remainder: full change minus sum of group self blocks.
    self_sum_n = sum(v["normal"] for v in by_kind.values())
    self_sum_a = sum(v["anomalous"] for v in by_kind.values())
    cross_n = d_rn - self_sum_n
    cross_a = d_ra - self_sum_a

    # Single-column Gram change for the top dark columns (own u_j u_j† - v_j v_j†
    # block only, including its U0 (=0) ; column index within the 1976 block).
    dark9, dark11 = groups9[0], groups11[0]
    per_dark = {}
    for j in DARK_TOP:
        uj9, vj9 = dark9["U"][:, j:j+1], dark9["V"][:, j:j+1]
        uj11, vj11 = dark11["U"][:, j:j+1], dark11["V"][:, j:j+1]
        own = (uj11 @ uj11.conj().T - vj11 @ vj11.conj().T
               - uj9 @ uj9.conj().T + vj9 @ vj9.conj().T)
        per_dark[str(j)] = {"max_abs_own_change": float(np.max(np.abs(own))),
                            "own_change": pack(own)}

    report = {
        "task": "P3 full Gram regrouping, 9 vs 11 cycles, macro1024 adjoint rows",
        "date": "2026-10-17",
        "map_assembly": "U = U0 + raw_driven - raw_undriven; V = raw_driven - raw_undriven; Hermitian VV†",
        "full_RN_minus_I": {"9_cycles": pack(rn9), "11_cycles": pack(rn11)},
        "full_RA": {"9_cycles": pack(ra9), "11_cycles": pack(ra11)},
        "window_change_full": {"normal": pack(d_rn), "anomalous": pack(d_ra),
                               "max_abs_normal": float(np.max(np.abs(d_rn))),
                               "max_abs_anomalous": float(np.max(np.abs(d_ra)))},
        "by_group_self_blocks": {
            k: {"normal": pack(v["normal"]), "anomalous": pack(v["anomalous"]),
                "max_abs_normal": float(np.max(np.abs(v["normal"])))}
            for k, v in by_kind.items()},
        "cross_terms_between_groups": {
            "normal": pack(cross_n), "anomalous": pack(cross_a),
            "max_abs_normal": float(np.max(np.abs(cross_n))),
            "max_abs_anomalous": float(np.max(np.abs(cross_a)))},
        "top_dark_single_column_own_blocks": per_dark,
        "consistency": "cross = full - sum(group self blocks); reported explicitly",
        "sources": {"rows9": str(path9), "rows11": str(path11)},
    }
    import json
    with open(OUTJSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    np.savez_compressed(OUTNPZ, u9=u9, v9=v9, u11=u11, v11=v11,
                        rn9=rn9, rn11=rn11, ra9=ra9, ra11=ra11)
    print("saved", OUTJSON, flush=True)
    print("full RN(0,0) 9/11:", rn9[0, 0], rn11[0, 0], flush=True)
    print("window change max|dRN|:", float(np.max(np.abs(d_rn))), flush=True)


if __name__ == "__main__":
    main()

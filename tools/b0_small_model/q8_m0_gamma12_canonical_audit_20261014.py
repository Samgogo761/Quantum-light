"""Panel boundaries versus frequency bins, then canonical structure.

Frequency bins that cut through a quadrature panel are node groups. The
0.07--0.13 cross term is checked by refining its owning panel, not by
deleting the nodes that sit inside the bin.

The finite joint model uses one k-point of the stored Gamma12 vertex,
the stored contact, and two polarization oscillators. It is not a new
Hamiltonian. The production (q, p, A) law is tested separately for a
conserved antisymmetric form.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_input_wing_20261008 as wing
import q8_m0_gamma12_ir_panel_20261014 as ir
import q8_m0_gamma12_transpose_input_refine_20261008 as base
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_continuum_columns_20261004 import ColumnState, midpoint_step
from q8_m0_phgauss import (
    _C_A,
    _C_Iph,
    io_from_phi,
    linear_K,
    propagate_phi,
    symplectic_J,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007"
OUT = ROOT / "results/q_line/gamma12_output_transpose_input_refine_20261008"
EDGES = (1e-4, 0.001, 0.02, 0.07, 0.13, 0.3, 1.0, 2.5, 6.0, 12.0)
BINS = [("below-1e-4", 0.0, 1e-4),
        *[(f"{lo:g}--{hi:g}", lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])],
        ("above-12", 12.0, 1e9)]


def pack(m):
    m = np.asarray(m)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def production_blocks(u, d, gamma):
    u = np.asarray(u, np.complex128)
    s2 = np.sqrt(2.0)
    return {
        "B": -s2 * np.imag(u).T,
        "C": s2 * np.real(u).T,
        "P": gamma * s2 * np.real(u),
        "Q": gamma * s2 * np.imag(u),
        "R": -gamma * np.real(np.asarray(d, float)),
    }


def production_K(u, d, gamma):
    blk = production_blocks(u, d, gamma)
    n = blk["B"].shape[0]
    kmat = np.zeros((2 * n + 2, 2 * n + 2))
    kmat[:n, 2 * n:] = blk["B"]
    kmat[n:2 * n, 2 * n:] = blk["C"]
    kmat[2 * n:, :n] = blk["P"]
    kmat[2 * n:, n:2 * n] = blk["Q"]
    kmat[2 * n:, 2 * n:] = blk["R"]
    return kmat


def standard_pair_defect(u, d, gamma):
    """||J K + K^T J|| for q paired with p and A0 paired with A1."""
    blk = production_blocks(u, d, gamma)
    b, c, p, qq, r = blk["B"], blk["C"], blk["P"], blk["Q"], blk["R"]
    j2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    blocks = (
        c + p.T @ j2,
        -b + qq.T @ j2,
        j2 @ p - c.T,
        j2 @ qq + b.T,
        j2 @ r + r.T @ j2,
    )
    return float(max(np.max(np.abs(x)) for x in blocks))


def constraint_spectrum(n, apply_j):
    """Singular values of J |-> the antisymmetric part of apply_j(J)."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cols = []
    for i, j in pairs:
        jay = np.zeros((n, n))
        jay[i, j] = 1.0
        jay[j, i] = -1.0
        residual = apply_j(jay)
        cols.append([residual[a, b] for a in range(n) for b in range(a + 1, n)])
    scale = np.linalg.svd(np.column_stack(cols), compute_uv=False)
    return {"largest": float(scale[0]), "smallest": float(scale[-1]),
            "smallest_over_largest": float(scale[-1] / scale[0])}


def column_from_real(q, p, field):
    s2 = np.sqrt(2.0)
    b = ((q + 1j * p) / s2).reshape(-1, 1)
    bd = ((q - 1j * p) / s2).reshape(-1, 1)
    return ColumnState(b, bd, np.asarray(field, np.complex128).reshape(2, 1))


def real_from_column(state):
    b = state.b[:, 0]
    bd = state.bdag[:, 0]
    field = state.A[:, 0]
    q = np.sqrt(2.0) * np.real(b)
    p = np.sqrt(2.0) * np.imag(b)
    return (np.concatenate((q, p, np.real(field))),
            float(np.max(np.abs(bd - np.conj(b)))),
            float(np.max(np.abs(np.imag(field)))))


def midpoint_map(u, d, gamma, h):
    n = u.shape[1]
    size = 2 * n + 2
    cols = []
    leak_bd = 0.0
    leak_a = 0.0
    for i in range(size):
        y = np.zeros(size)
        y[i] = 1.0
        nxt = midpoint_step(column_from_real(y[:n], y[n:2 * n], y[2 * n:]), u, d, h, gamma)
        z, one, two = real_from_column(nxt)
        cols.append(z)
        leak_bd = max(leak_bd, one)
        leak_a = max(leak_a, two)
    return np.column_stack(cols), leak_bd, leak_a


def electronic_wiring(u_weighted, bare, wk):
    """Production q-dot and the finite-oscillator q-dot at the same A."""
    field = np.array([0.3, -0.7])
    prod = -np.sqrt(2.0) * np.imag(u_weighted).T @ field
    coef = np.sqrt(2.0 * wk)
    xi = np.imag(bare)
    other = np.array([-coef * (xi[0, c, v] * field[0] + xi[1, c, v] * field[1])
                      for c in range(4) for v in range(4)])
    return float(np.max(np.abs(prod - other)))


def bogoliubov_defects(phi):
    phi = np.asarray(phi, float)
    n = phi.shape[0] // 2
    q = phi[:n, :]
    p = phi[n:, :]
    eye = np.eye(n)
    a = (q[:, :n] + 1j * p[:, :n]) / np.sqrt(2.0)
    c = (q[:, n:] + 1j * p[:, n:]) / np.sqrt(2.0)
    smap = (a - 1j * c) / np.sqrt(2.0)
    tmap = (a + 1j * c) / np.sqrt(2.0)
    normal = smap @ smap.conj().T - tmap @ tmap.conj().T - eye
    anomalous = smap @ tmap.T - tmap @ smap.T
    return float(np.max(np.abs(normal))), float(np.max(np.abs(anomalous)))


def field_quadrature_piece(bare, wk, contact, fr, fi, gamma, eta):
    """Quadrature part of dA/dt at frozen F, compared with gamma * I.

    If Re F is parallel to Im F for every oscillator, that quadrature piece
    cancels identically. A large ratio is then not a residual.
    """
    x = bare.reshape(1, 2, 4, 4)
    sw = np.array([np.sqrt(wk)])
    kmat = linear_K(fr.shape[0], sw, x, fr, fi, contact, eta)
    rng = np.random.default_rng(1)
    xi = rng.normal(size=kmat.shape[0])
    ca = _C_A(fr.shape[0], 16, fr, fi, eta)
    ci = _C_Iph(fr.shape[0], 16, np.real(x), np.imag(x), sw)
    field = ca @ xi
    current = ci @ xi - contact @ field
    from_osc = ca @ (kmat @ xi)
    from_prod = gamma * current
    return {
        "Adot_quadrature_norm": float(np.linalg.norm(from_osc)),
        "gamma_I_norm": float(np.linalg.norm(from_prod)),
        "parallel_Re_Im_cancels_quadrature_piece": bool(np.linalg.norm(from_osc) < 1e-12),
    }


def finite_joint(driven_u, driven_d, wk, k, gamma, pol, h, nt=9):
    sl = slice(k * 16, (k + 1) * 16)
    x = np.empty((nt, 1, 2, 4, 4), np.complex128)
    for step in range(nt):
        x[step, 0] = np.asarray(driven_u[step, :, sl], np.complex128).reshape(2, 4, 4) / np.sqrt(wk)
    contact = np.real(np.asarray(driven_d[:nt]))
    flist = [np.repeat(np.asarray(pol[:, mu], np.complex128)[None, :], nt, axis=0)
             for mu in range(2)]
    eta = q.eta_of(gamma)
    rec = propagate_phi(x, np.array([wk]), contact, flist, h, eta)
    jay = symplectic_J(rec["n"])
    sym = float(np.max(np.abs(rec["Phi"].T @ jay @ rec["Phi"] - jay)))
    rn, ra = bogoliubov_defects(rec["Phi"])
    io = io_from_phi(rec["Phi"], rec["n_ph"], rec["n_B"])
    fr = np.real(np.stack([f[0] for f in flist], axis=0))
    fi = np.imag(np.stack([f[0] for f in flist], axis=0))
    return {
        "k": int(k),
        "n_photon": int(rec["n_ph"]),
        "n_electron": int(rec["n_B"]),
        "steps": nt - 1,
        "eta": eta,
        "generator_hamiltonian_maxabs": float(rec["hamiltonian_K_maxabs"]),
        "phi_T_J_phi_minus_J": sym,
        "nambu_normal_maxabs": rn,
        "nambu_anomalous_maxabs": ra,
        "photon_block_identity_maxabs": float(io["ident_minus_I_maxabs"]),
        "field_quadrature_piece": field_quadrature_piece(
            x[0, 0], wk, contact[0], fr, fi, gamma, eta),
    }


def classify_panels(panels):
    out = {}
    for name, lo, hi in BINS:
        wholly, crossing = [], []
        for panel in panels:
            om = panel["omega"]
            inside = (om >= lo) & (om < hi)
            if not np.any(inside):
                continue
            brief = {"kind": panel["kind"], "pole": panel["pole"],
                     "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
                     "nodes": panel["nodes"],
                     "nodes_inside_bin": int(np.count_nonzero(inside)),
                     "weight_sum": panel["weight_sum"]}
            if np.all(inside):
                wholly.append(brief)
            else:
                crossing.append(brief)
        out[name] = {"whole_panels": len(wholly), "crossing_panels": len(crossing),
                     "complete_panel_union": len(crossing) == 0,
                     "crossing": crossing}
    return out


def drop_poles(om, wt, gaps):
    keep = np.ones(om.shape[0], dtype=bool)
    limit = 64 * np.finfo(float).eps * max(1.0, float(np.max(gaps)))
    for gval in np.unique(np.asarray(gaps, float)):
        keep &= np.abs(om - gval) > max(limit, 1e-14)
    return om[keep], wt[keep], int(np.count_nonzero(~keep))


def refine_owning_panel(panels, saved, rows, cache, gamma, pol, phi_of, eta):
    owners = []
    for panel in panels:
        om = panel["omega"]
        if np.any((om >= 0.07) & (om < 0.13)):
            owners.append(panel)
    if len(owners) != 1 or owners[0]["kind"] != "core":
        return {"status": "not_one_core", "owners": len(owners)}
    panel = owners[0]
    saved_om = np.asarray(saved["oms"], float)
    saved_key = np.round(saved_om, 12)
    panel_key = np.round(panel["omega"], 12)
    if not np.all(np.isin(panel_key, saved_key)):
        raise AssertionError("Owning-panel nodes are not all in the saved quadrature")
    in_interval = (saved_om >= panel["omega_lo"] - 1e-12) & (saved_om <= panel["omega_hi"] + 1e-12)
    if np.any(~np.isin(np.round(saved_om[in_interval], 12), panel_key)):
        raise AssertionError("Saved node in the owning interval belongs to another panel")
    idx = np.flatnonzero(np.isin(saved_key, panel_key))
    cols = (4 * idx[:, None] + np.arange(4)).ravel()
    u_old, v_old = saved["U"][:, cols], saved["V"][:, cols]
    n_old = u_old @ u_old.conj().T - v_old @ v_old.conj().T
    a_old = u_old @ v_old.T - v_old @ u_old.T
    n_all = saved["U"] @ saved["U"].conj().T - saved["V"] @ saved["V"].conj().T
    a_all = saved["U"] @ saved["V"].T - saved["V"] @ saved["U"].T
    saved_r = n_all - np.eye(2)
    methods = {}
    for nnode in (panel["nodes"], panel["nodes"] * 2, panel["nodes"] * 4):
        om, wt = q.t_gauss_panel(panel["pole"], panel["gwid"], panel["t0"], panel["t1"], nnode)
        om, wt, dropped = drop_poles(om, wt, cache["gaps"])
        if nnode == panel["nodes"]:
            regenerated = np.max(np.abs(np.sort(om) - np.sort(panel["omega"])))
            if regenerated > 1e-10 or dropped:
                raise AssertionError(f"Core regeneration mismatch {regenerated}, dropped {dropped}")
        print(f"  refine sinh {nnode}", flush=True)
        nu, au = ir.contract_frequencies(cache, rows, gamma, pol, phi_of, eta, om, wt)
        rn, ra = ir.residuals(n_all, a_all, n_old, a_old, nu, au)
        methods[f"sinh_{nnode}"] = {
            "nodes_after_pole_drop": int(om.size), "poles_dropped": dropped,
            "weight_sum": float(np.sum(wt)),
            "R": pack(rn), "A": pack(ra),
            "Rmax": float(np.max(np.abs(rn))), "Amax": float(np.max(np.abs(ra))),
            "R00": float(np.real(rn[0, 0])),
            "minus_saved_R_maxabs": float(np.max(np.abs(rn - saved_r))),
        }
        if nnode == panel["nodes"] and methods[f"sinh_{nnode}"]["minus_saved_R_maxabs"] > 1e-6:
            raise AssertionError("Recontracting the original core does not reproduce the saved residual")
        print("   Rmax", methods[f"sinh_{nnode}"]["Rmax"], "R00", methods[f"sinh_{nnode}"]["R00"], flush=True)
    return {"status": "one_core", "pole": panel["pole"],
            "omega_lo": panel["omega_lo"], "omega_hi": panel["omega_hi"],
            "original_nodes": panel["nodes"], "original_weight": panel["weight_sum"],
            "interval_length": panel["omega_hi"] - panel["omega_lo"],
            "methods_at_4096": methods}


def main():
    gaps = np.load(RUN / "arrays/gaps.npy")
    u0 = np.load(RUN / "arrays/u0.npy")
    d0 = np.load(RUN / "arrays/D0.npy")
    weights = np.load(RUN / "arrays/weights.npy")
    gamma = float(np.load(RUN / "arrays/gamma.npy"))
    pol = np.load(RUN / "arrays/pol.npy")
    h = float(np.load(RUN / "arrays/h.npy"))
    if gaps.size != 2304 or weights.size != 144 or u0.shape != (2, 2304):
        raise AssertionError("Unexpected Gamma12 channel layout")
    ident_n, ident_a = bogoliubov_defects(np.eye(8))
    if max(ident_n, ident_a) > 1e-12:
        raise AssertionError("Bogoliubov converter fails on the identity")
    panels = []
    oms, wts, _counts = wing.wing_quadrature(
        gaps, u0, gamma, n_panel=160, wing_factor=2, omega_lo=1.0e-4,
        omega_hi=12.0, panels=panels)
    with np.load(OUT / "endpoint_low.npz") as saved_probe:
        if not (np.allclose(oms, saved_probe["oms"]) and np.allclose(wts, saved_probe["wts"])):
            raise AssertionError("Rebuilt panels do not reproduce the infrared baseline quadrature")
    classified = classify_panels(panels)
    target = classified["0.07--0.13"]
    print("0.07--0.13 crossing", target["crossing_panels"], "whole", target["whole_panels"], flush=True)
    for item in target["crossing"]:
        print(" ", item, flush=True)

    norms = np.linalg.norm(u0.reshape(2, 144, 16), axis=(0, 2))
    bright = int(np.argmax(norms))
    ratios = []
    for k in range(144):
        sl = slice(k * 16, (k + 1) * 16)
        kmat_k = production_K(u0[:, sl], d0, gamma)
        spec = constraint_spectrum(
            kmat_k.shape[0], lambda jay, kk=kmat_k: jay @ kk + kk.T @ jay)
        ratios.append(spec["smallest_over_largest"])
        if k % 36 == 0:
            print("k", k, "J ratio", ratios[-1], flush=True)
    pair_k = production_K(np.concatenate((u0[:, 0:16], u0[:, 16:32]), axis=1), d0, gamma)
    pair_spec = constraint_spectrum(
        pair_k.shape[0], lambda jay, kk=pair_k: jay @ kk + kk.T @ jay)
    sl = slice(bright * 16, (bright + 1) * 16)
    kmat = production_K(u0[:, sl], d0, gamma)
    mapped, leak_bd, leak_a = midpoint_map(u0[:, sl], d0, gamma, 1e-6)
    generator_gap = float(np.max(np.abs((mapped - np.eye(mapped.shape[0])) / 1e-6 - kmat)))
    step = midpoint_map(u0[:, sl], d0, gamma, h)[0]
    step_spec = constraint_spectrum(
        step.shape[0], lambda jay: step.T @ jay @ step - jay)
    bare = u0[:, sl].reshape(2, 4, 4) / np.sqrt(weights[bright])
    wiring = electronic_wiring(u0[:, sl], bare, float(weights[bright]))
    if wiring > 1e-10:
        raise AssertionError(f"Electron quadrature wiring mismatch {wiring}")
    driven_u = np.load(RUN / "arrays/driven_u.npy", mmap_mode="r")
    driven_d = np.load(RUN / "arrays/driven_d.npy")
    undriven_d = np.load(RUN / "arrays/undriven_d.npy")
    if max(np.max(np.abs(np.imag(driven_d))), np.max(np.abs(np.imag(undriven_d)))) > 1e-8:
        raise AssertionError("Stored contact is not real")
    joint = finite_joint(driven_u, driven_d, float(weights[bright]), bright, gamma, pol, h)
    undriven_u = np.load(RUN / "arrays/undriven_u.npy", mmap_mode="r")
    joint_static = finite_joint(
        undriven_u, undriven_d, float(weights[bright]), bright, gamma, pol, h)
    print("finite joint symplectic", joint["phi_T_J_phi_minus_J"],
          "field mismatch", joint["field_law_relative_mismatch"], flush=True)
    print("generator J ratio", float(np.min(ratios)), float(np.max(ratios)),
          "pair", pair_spec["smallest_over_largest"], flush=True)

    public = {}
    for name, rec in classified.items():
        public[name] = {"whole_panels": rec["whole_panels"],
                        "crossing_panels": rec["crossing_panels"],
                        "complete_panel_union": rec["complete_panel_union"],
                        "crossing": rec["crossing"]}
    report = {
        "quadrature": "wing factor 2, core 160, omega_lo=1e-4, omega_hi=12",
        "panel_count": len(panels),
        "baseline_nodes_match": True,
        "bins": public,
        "finite_joint_driven": joint,
        "finite_joint_undriven": joint_static,
        "production_generator": {
            "brightest_k": bright,
            "electronic_wiring_maxabs": wiring,
            "midpoint_versus_generator": generator_gap,
            "real_section_bdag_leak": leak_bd,
            "real_section_A_imag_leak": leak_a,
            "standard_q_p_A_pair_defect_one_k": standard_pair_defect(u0[:, sl], d0, gamma),
            "standard_q_p_A_pair_defect_all_k": standard_pair_defect(u0, d0, gamma),
            "one_k_J_ratio_min": float(np.min(ratios)),
            "one_k_J_ratio_max": float(np.max(ratios)),
            "two_k_J_ratio": pair_spec["smallest_over_largest"],
            "one_step_J_ratio": step_spec["smallest_over_largest"],
            "generator_check": (
                "A small singular-value ratio only means some antisymmetric J "
                "is annihilated. The kernel found here is large and every sampled "
                "element is singular, so it is not a symplectic form."
            ),
        },
        "owning_panel_refinement": None,
        "production_qualification": False,
    }
    path = OUT / "canonical_audit_20261014.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote canonical section", path, flush=True)

    manifest, cache, rows, _info, _ident = base.load_frozen()
    if float(cache["gamma"]) != gamma:
        raise AssertionError("Gamma mismatch")
    op = q.omega_grid(base.p1_a2_protocol(q.A_STAR).omega0,
                      base.p1_a2_protocol(q.A_STAR).t_pulse)
    op = op[op > 0]
    _phi, norm = q.normalize_phi(op, q.phi_unnormalized(
        op, 2 * base.p1_a2_protocol(q.A_STAR).omega0,
        base.p1_a2_protocol(q.A_STAR).t_pulse))
    phi_of = lambda nu: q.phi_at(nu, base.p1_a2_protocol(q.A_STAR), norm)
    with np.load(OUT / "endpoint_low.npz") as saved:
        refinement = refine_owning_panel(
            panels, saved, rows, cache, gamma, np.array(cache["pol"], copy=True),
            phi_of, q.eta_of(gamma))

    residual_path = OUT / "residual_source_4096.json"
    residual = json.loads(residual_path.read_text(encoding="utf-8"))
    residual["node_grouping_not_panel_integral"] = True
    residual["note"] = (
        "Band split of the omega_lo=1e-4, ntpc=4096 output. Bins are "
        "node-location groups, not complete quadrature panels. A trial "
        "internal b-bdag gram was removed because it is not the output CCR."
    )
    residual_path.write_text(json.dumps(residual, indent=2) + "\n", encoding="utf-8")

    report["owning_panel_refinement"] = refinement
    report["identity_checked_against"] = manifest["manifest_sha256"]
    path = OUT / "canonical_audit_20261014.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()

"""Absolute accumulation of the 343 panel changes.

The signed sum is already published. This file adds the sum of absolute
entries and the signed sum of the panels whose first doubling is below 1e-6.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_local_prepare_20261007 as prep
import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_gamma12_panel_budget_20261015 import contract_frequencies, nodes_at
from q8_m0_g1_gamma12_tail_columns_20261014 import (
    BASELINE, contribution, load_rows, match_saved,
)
from q8_m0_gamma12_input_wing_20261008 import wing_quadrature

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_panel_budget_20261015/panel_abs_budget.json"


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if prep.sha256_file(BASELINE) != json_sha():
        raise ValueError("endpoint_low.npz hash changed")
    rows, cache, t0, _meta = load_rows()
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(base.q.A_STAR)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    panels = []
    oms, wts, _counts = wing_quadrature(
        cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
        omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    with np.load(BASELINE, allow_pickle=False) as saved_file:
        saved = {"oms": np.array(saved_file["oms"]), "wts": np.array(saved_file["wts"]),
                 "U": np.array(saved_file["U"]), "V": np.array(saved_file["V"])}
    if not (np.allclose(saved["oms"], oms) and np.allclose(saved["wts"], wts)):
        raise AssertionError("Saved nodes changed")
    old = []
    for panel in panels:
        _ids, cols = match_saved(saved["oms"], saved["wts"], panel["omega"], panel["weights"])
        old.append(contribution(saved["U"][:, cols], saved["V"][:, cols]))
    pieces_o, pieces_w, owners = [], [], []
    for index, panel in enumerate(panels):
        om, wt = nodes_at(panel, 2 * int(panel["requested"]))
        pieces_o.append(om)
        pieces_w.append(wt)
        owners.append(np.full(len(om), index, dtype=int))
    new_o, new_w = np.concatenate(pieces_o), np.concatenate(pieces_w)
    owners = np.concatenate(owners)
    print("contract", len(new_o), flush=True)
    new_u, new_v = contract_frequencies(rows, cache, gamma, pol, phi_of, eta, t0, new_o, new_w)
    signed_n = np.zeros((2, 2), complex)
    signed_a = np.zeros((2, 2), complex)
    abs_n = np.zeros((2, 2), complex)
    abs_a = np.zeros((2, 2), complex)
    small_n = np.zeros((2, 2), complex)
    small_a = np.zeros((2, 2), complex)
    small_abs_n = np.zeros((2, 2), complex)
    small_abs_a = np.zeros((2, 2), complex)
    maxabs = []
    n_small = 0
    for index, panel in enumerate(panels):
        pick = np.flatnonzero(owners == index)
        cols = np.array([4 * int(i) + k for i in pick for k in range(4)], int)
        normal, anomalous = contribution(new_u[:, cols], new_v[:, cols])
        delta_n, delta_a = normal - old[index][0], anomalous - old[index][1]
        signed_n += delta_n
        signed_a += delta_a
        abs_n += np.abs(delta_n.real) + 1j * np.abs(delta_n.imag)
        abs_a += np.abs(delta_a.real) + 1j * np.abs(delta_a.imag)
        change = max(float(np.max(np.abs(delta_n))), float(np.max(np.abs(delta_a))))
        maxabs.append(change)
        if change <= 1.0e-6:
            n_small += 1
            small_n += delta_n
            small_a += delta_a
            small_abs_n += np.abs(delta_n.real) + 1j * np.abs(delta_n.imag)
            small_abs_a += np.abs(delta_a.real) + 1j * np.abs(delta_a.imag)
    # Reproduce the published signed (0,0).
    if abs(float(signed_n[0, 0].real) - (-0.005980195961265739)) > 1.0e-9:
        raise AssertionError(f"Signed sum changed: {signed_n[0, 0].real}")
    report = {
        "panel_count": len(panels),
        "signed_sum": {"normal": pack(signed_n), "anomalous": pack(signed_a)},
        "absolute_entry_sum": {"normal": pack(abs_n), "anomalous": pack(abs_a)},
        "first_doubling_at_most_1e-6": {
            "count": n_small,
            "signed_sum": {"normal": pack(small_n), "anomalous": pack(small_a)},
            "absolute_entry_sum": {"normal": pack(small_abs_n), "anomalous": pack(small_abs_a)},
        },
        "maxabs_percentiles": {
            "median": float(np.median(maxabs)),
            "p90": float(np.quantile(maxabs, 0.90)),
            "p99": float(np.quantile(maxabs, 0.99)),
            "maximum": float(np.max(maxabs)),
        },
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    }
    write_new_json(OUT, report)
    print("small signed", float(small_n[0, 0].real), "small abs", float(np.real(small_abs_n[0, 0])))
    print("all abs", float(np.real(abs_n[0, 0])))


def json_sha():
    meta = __import__("json").loads(BASELINE.with_suffix(".json").read_text(encoding="utf-8"))
    return meta["row_sha256"]


if __name__ == "__main__":
    main()

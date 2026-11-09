"""Even/odd port diagnostic. Does not delete a port or rebuild the rows.

c+ = (c_L+c_R)/sqrt(2), c- = (c_L-c_R)/sqrt(2). The same real rotation
acts on annihilation and creation columns. R_N and R_A are invariant.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import q8_m0_gamma12_input_wing_20261008 as wing
import q8_m0_gamma12_transpose_input_refine_20261008 as base
from q8_m0_audit_artifacts_20261014 import provenance, write_new_json
from q8_m0_classical import p1_a2_protocol
from q8_m0_g1_dynamic_map_audit_20261015 import require_cache, split_raw
from q8_m0_g1_gamma12_ir_tail_20261015 import POLE, load_adjoint, log_gauss
from q8_m0_g1_gamma12_richardson_2048_20261015 import final_frequencies
from q8_m0_g1_gamma12_richardson_corrected_20261015 import select_panels
from q8_m0_g1_gamma12_richardson_full_20261015 import accumulate_arrays
from q8_m0_g1_gamma12_tail_columns_20261014 import BASELINE, load_rows
from q8_m0_g1_gamma12_column_order_20261015 import CACHE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q_line/g1_gamma12_time_order_20261015/port_parity_macro2048.json"
S2 = np.sqrt(2.0)


def pack(m):
    m = np.asarray(m, complex)
    return {"real": np.real(m).tolist(), "imag": np.imag(m).tolist(),
            "maxabs": float(np.max(np.abs(m)))}


def sides(arr):
    """(2, 4n) in order L0,L1,R0,R1 -> L,R each (2, 2n)."""
    n = arr.shape[1] // 4
    left = np.empty((2, 2 * n), complex)
    right = np.empty_like(left)
    left[:, 0::2] = arr[:, 0::4]
    left[:, 1::2] = arr[:, 1::4]
    right[:, 0::2] = arr[:, 2::4]
    right[:, 1::2] = arr[:, 3::4]
    return left, right


def parity(arr):
    left, right = sides(arr)
    return (left + right) / S2, (left - right) / S2


class Gram:
    def __init__(self):
        self.uu = np.zeros((2, 2), complex)
        self.vv = np.zeros((2, 2), complex)
        self.uv = np.zeros((2, 2), complex)
        self.vu = np.zeros((2, 2), complex)
        self.max_abs = 0.0

    def add(self, u, v):
        self.uu += u @ u.conj().T
        self.vv += v @ v.conj().T
        self.uv += u @ v.T
        self.vu += v @ u.T
        if u.size:
            self.max_abs = max(self.max_abs, float(np.max(np.abs(u))), float(np.max(np.abs(v))))

    def normal(self):
        return self.uu - self.vv

    def anomalous(self):
        return self.uv - self.vu


def pack_gram(gram, subtract_identity=False):
    normal = gram.normal()
    out = {"normal_without_I": pack(normal), "anomalous": pack(gram.anomalous()),
           "max_abs_column": gram.max_abs}
    if subtract_identity:
        out["normal_residual"] = pack(normal - np.eye(2))
    return out


def columns_of(rows, cache, gamma, pol, phi_of, eta, oms, wts):
    ph, u0 = base.generated_photon_block(
        cache, gamma, oms, wts, pol, phi_of, eta)
    di_u, di_v, df_u, df_v = split_raw(rows[0], rows[1], ph)
    ui_u, ui_v, uf_u, uf_v = split_raw(rows[2], rows[3], ph)
    du = (di_u + df_u) - (ui_u + uf_u)
    v = (di_v + df_v) - (ui_v + uf_v)
    return u0, du, v


def absorb(store, u0, du, v):
    u0p, u0m = parity(u0)
    dup, dum = parity(du)
    vp, vm = parity(v)
    store["delta_minus"].add(dum, vm)
    store["delta_plus"].add(dup, vp)
    store["u0_plus"].add(u0p, np.zeros_like(u0p))
    store["u0_minus"].add(u0m, np.zeros_like(u0m))
    store["full_plus"].add(u0p + dup, vp)
    store["full_minus"].add(u0m + dum, vm)
    # Cross between + and - uses the full columns.
    up, um = u0p + dup, u0m + dum
    store["cross_uu"] += up @ um.conj().T + um @ up.conj().T
    store["cross_vv"] += vp @ vm.conj().T + vm @ vp.conj().T
    store["cross_uv"] += up @ vm.T + um @ vp.T - vp @ um.T - vm @ up.T
    diff_u = sides(du)[0] - sides(du)[1]
    diff_v = sides(v)[0] - sides(v)[1]
    if diff_u.size:
        store["max_du_L_minus_R"] = max(store["max_du_L_minus_R"], float(np.max(np.abs(diff_u))))
        store["max_v_L_minus_R"] = max(store["max_v_L_minus_R"], float(np.max(np.abs(diff_v))))
    diff_0 = sides(u0)[0] - sides(u0)[1]
    if diff_0.size:
        store["max_u0_L_minus_R"] = max(store["max_u0_L_minus_R"], float(np.max(np.abs(diff_0))))


def fresh():
    return {
        "delta_minus": Gram(), "delta_plus": Gram(),
        "u0_plus": Gram(), "u0_minus": Gram(),
        "full_plus": Gram(), "full_minus": Gram(),
        "cross_uu": np.zeros((2, 2), complex),
        "cross_vv": np.zeros((2, 2), complex),
        "cross_uv": np.zeros((2, 2), complex),
        "max_du_L_minus_R": 0.0, "max_v_L_minus_R": 0.0, "max_u0_L_minus_R": 0.0,
    }


def walk(rows, cache, gamma, pol, phi_of, eta, oms, wts, store, label):
    for start in range(0, len(oms), 100):
        stop = min(start + 100, len(oms))
        u0, du, v = columns_of(rows, cache, gamma, pol, phi_of, eta, oms[start:stop], wts[start:stop])
        absorb(store, u0, du, v)
        if stop == len(oms) or stop % 4000 == 0:
            print(label, stop, store["max_du_L_minus_R"], flush=True)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    path, digest = require_cache(2048)
    rows = load_adjoint(path)
    _saved, cache, t0, _meta = load_rows()
    if abs(t0) > 0:
        raise AssertionError(t0)
    gamma = float(np.asarray(cache["gamma"]))
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(0.8)
    grid = base.q.omega_grid(prot.omega0, prot.t_pulse)
    grid = grid[grid > 0.0]
    _phi, norm = base.q.normalize_phi(grid, base.q.phi_unnormalized(grid, 2.0 * prot.omega0, prot.t_pulse))
    phi_of = lambda nu: base.q.phi_at(nu, prot, norm)
    eta = base.q.eta_of(gamma)
    with np.load(BASELINE, allow_pickle=False) as saved:
        saved_oms, saved_wts = np.array(saved["oms"]), np.array(saved["wts"])
    tails, wing_panel, adjacent, core, big, identities = select_panels(
        cache, gamma, saved_oms, saved_wts)
    pieces = final_frequencies(saved_oms, saved_wts, tails, wing_panel, adjacent, core, big, identities)
    store = fresh()
    for name, om, wt in pieces:
        print("piece", name, len(om), flush=True)
        walk(rows, cache, gamma, pol, phi_of, eta, om, wt, store, name)
    panels = []
    wing.wing_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=160, wing_factor=2,
                         omega_lo=1.0e-4, omega_hi=12.0, panels=panels)
    panel = next(item for item in panels if item["kind"] == "sinh_tail"
                 and abs(item["pole"] - POLE) < 1e-10 and item["omega_hi"] < 0.05)
    ir_om, ir_wt = log_gauss(1.0e-8, panel["omega_lo"], 64)
    walk(rows, cache, gamma, pol, phi_of, eta, ir_om, ir_wt, store, "infrared")
    dark_b = np.array(np.load(CACHE / "b_cols.npy", mmap_mode="r")[-1976:].T)
    dark_bd = np.array(np.load(CACHE / "bdag_cols.npy", mmap_mode="r")[-1976:].T)
    dark_a = np.array(np.load(CACHE / "A_cols.npy", mmap_mode="r")[-1976:].T)
    dark = Gram()
    for start in range(0, 1976, 200):
        stop = min(start + 200, 1976)
        sl = slice(start, stop)
        zeros = np.zeros((2, stop - start), complex)
        omega = np.zeros(stop - start)
        du, dv = accumulate_arrays(rows[0], rows[1], dark_b[:, sl], dark_bd[:, sl], dark_a[:, sl], zeros, omega)
        uu, uv = accumulate_arrays(rows[2], rows[3], dark_b[:, sl], dark_bd[:, sl], dark_a[:, sl], zeros, omega)
        dark.add(du - uu, dv - uv)
    # UU† sums outer products of columns. U_+ U_-† is a coherence between
    # different inputs and is not a term in R_N or R_A.
    photon = store["full_plus"].normal() + store["full_minus"].normal()
    total = photon + dark.normal() - np.eye(2)
    anomalous = (store["full_plus"].anomalous() + store["full_minus"].anomalous()
                 + dark.anomalous())
    write_new_json(OUT, {
        "macro_ntpc": 2048,
        "cache_sha256": digest,
        "transform": "c_+-(c_L+-c_R)/sqrt(2), same real rotation on creation columns",
        "max_du_L_minus_R": store["max_du_L_minus_R"],
        "max_v_L_minus_R": store["max_v_L_minus_R"],
        "max_u0_L_minus_R": store["max_u0_L_minus_R"],
        "dynamic_minus": pack_gram(store["delta_minus"]),
        "dynamic_plus": pack_gram(store["delta_plus"]),
        "u0_plus": pack_gram(store["u0_plus"]),
        "u0_minus": pack_gram(store["u0_minus"]),
        "full_plus": pack_gram(store["full_plus"]),
        "full_minus": pack_gram(store["full_minus"]),
        "plus_minus_cross_normal": pack(store["cross_uu"] - store["cross_vv"]),
        "plus_minus_cross_anomalous": pack(store["cross_uv"]),
        "dark_normal_without_I": pack(dark.normal()),
        "dark_anomalous": pack(dark.anomalous()),
        "total_normal_residual": pack(total),
        "total_anomalous": pack(anomalous),
        "antisymmetric_input_not_deleted": True,
        "production_G1_pass": False,
        "provenance_sha256": provenance([__file__]),
    })
    print("dU minus", store["max_du_L_minus_R"], "V minus", store["max_v_L_minus_R"], flush=True)
    print("RN", float(total[0, 0].real), float(np.max(np.abs(total))), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Line-Q static audit: one bond table, named T_perp candidates, V = 2 tau_y B.

Does not freeze C3, does not downfold, does not attach a propagator.
Historical cartoons in q8_tperp_material_audit.py stay untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCF = REPO / "deploy/a0_layerA_m88full/a1_fewk_nscf/CrI3_scf_soc_a1h_clean.in"
TB = Path(r"C:\Users\26507\Documents\量子光研究\01_输入与模型\wannier\CrI3_tb.dat")
OLD_DIR = REPO / "results/q_line/tperp_audit_20260910"
OUT = REPO / "results/q_line/tperp_static_audit_20260910"

INV_C = np.array([0.499998583, 0.333333067, 0.372004766])
N_ORB8 = 8
T_SCALE = 0.15
DELTA = 0.5
NN_WINDOW = (3.90, 4.20)
NNN_WINDOW = (6.90, 7.10)
INTER_WINDOW = (6.90, 7.20)
RES_ATOL = 1.0e-9


def wrap_frac(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) + 0.5) % 1.0 - 0.5


def pauli():
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return s0, sx, sy, sz


def parse_scf(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    lat = None
    atoms: list[tuple[str, np.ndarray]] = []
    mode = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("CELL_PARAMETERS"):
            lat = np.array([[float(x) for x in lines[i + j + 1].split()[:3]] for j in range(3)])
        elif s.startswith("ATOMIC_POSITIONS"):
            mode = "pos"
        elif mode == "pos":
            if not s or s.startswith("HUBBARD"):
                mode = None
                continue
            p = s.split()
            atoms.append((p[0], np.array([float(x) for x in p[1:4]], dtype=float)))
    if lat is None:
        raise RuntimeError("no lattice")
    return lat, atoms


def integer_R(tau_from: np.ndarray, tau_to: np.ndarray) -> np.ndarray:
    """R such that wrap(to-from) = R + to - from (nearest image)."""
    raw = np.asarray(tau_to, dtype=float) - np.asarray(tau_from, dtype=float)
    wrapped = wrap_frac(raw)
    r = np.rint(wrapped - raw)
    return r.astype(int)


def bond_vector(lat: np.ndarray, tau_from: np.ndarray, tau_to: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    delta_frac = rvec.astype(float) + tau_to - tau_from
    return delta_frac @ lat


def verify_delta_identity(lat, tau_from, tau_to, rvec, atol=1.0e-12) -> float:
    wrapped = wrap_frac(tau_to - tau_from)
    recon = rvec.astype(float) + tau_to - tau_from
    return float(np.max(np.abs(recon - wrapped)))


def cr_sites(atoms):
    cr = [f for n, f in atoms if n.startswith("Cr")]
    names = ("tA", "tB", "bA", "bB")
    return {names[i]: cr[i] for i in range(4)}


def collect_bonds(lat, tau_from, tau_to, window, rmax=2):
    out = []
    for n1 in range(-rmax, rmax + 1):
        for n2 in range(-rmax, rmax + 1):
            rvec = np.array([n1, n2, 0], dtype=int)
            vec = bond_vector(lat, tau_from, tau_to, rvec)
            d = float(np.linalg.norm(vec[:2]))
            if window[0] <= d <= window[1]:
                out.append({"R": [n1, n2, 0], "delta_cart": vec.tolist(), "dxy": d, "d3": float(np.linalg.norm(vec))})
    out.sort(key=lambda b: b["dxy"])
    return out


def build_bond_table(lat, sites) -> dict:
    nn_top = collect_bonds(lat, sites["tA"], sites["tB"], NN_WINDOW)
    nn_bot = collect_bonds(lat, sites["bA"], sites["bB"], NN_WINDOW)
    nnn_aa = collect_bonds(lat, sites["tA"], sites["tA"], NNN_WINDOW)
    nnn_bb = collect_bonds(lat, sites["tB"], sites["tB"], NNN_WINDOW)
    interlayer = []
    for top in ("tA", "tB"):
        for bot in ("bA", "bB"):
            rvec = integer_R(sites[top], sites[bot])
            residual = verify_delta_identity(lat, sites[top], sites[bot], rvec)
            vec = bond_vector(lat, sites[top], sites[bot], rvec)
            interlayer.append(
                {
                    "from": top,
                    "to": bot,
                    "channel": top[1] + bot[1],
                    "R": [int(x) for x in rvec],
                    "tau_from": sites[top].tolist(),
                    "tau_to": sites[bot].tolist(),
                    "delta_frac": (rvec.astype(float) + sites[bot] - sites[top]).tolist(),
                    "delta_cart": vec.tolist(),
                    "dxy": float(np.linalg.norm(vec[:2])),
                    "d3": float(np.linalg.norm(vec)),
                    "identity_residual": residual,
                    "reverse_R": [int(x) for x in integer_R(sites[bot], sites[top])],
                }
            )
    interlayer.sort(key=lambda b: b["d3"])
    nearest = [b for b in interlayer if INTER_WINDOW[0] <= b["d3"] <= INTER_WINDOW[1]]
    # KM winding from top-layer NN (same construction on bottom is reported, not mixed)
    def _nnn_from_nn(nn_bonds):
        vecs = [np.array(b["delta_cart"][:2], dtype=float) for b in nn_bonds]
        signed = []
        if len(vecs) >= 3:
            for i in range(3):
                j = (i + 1) % 3
                v = vecs[i] - vecs[j]
                nu = float(np.sign(vecs[i][0] * vecs[j][1] - vecs[i][1] * vecs[j][0]))
                if nu == 0.0:
                    nu = 1.0
                signed.append({"vector": v.tolist(), "nu": nu, "length": float(np.linalg.norm(v))})
        return vecs, signed

    nn_vecs, nnn_signed = _nnn_from_nn(nn_top)
    _nn_bot_vecs, nnn_signed_bot = _nnn_from_nn(nn_bot)
    g_note = (
        "g(k)=sum_cyc 2 nu sin(k·(δ_i-δ_j)) from the same three NN; "
        "default h0 copies the top-layer table onto both layers"
    )
    angles = []
    if len(nn_vecs) >= 3:
        for i in range(3):
            u = nn_vecs[i]
            w = nn_vecs[(i + 1) % 3]
            c = float(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1.0, 1.0))
            angles.append(math.degrees(math.acos(c)))
    inv = {}
    for src, tau in sites.items():
        image = (2.0 * INV_C - tau) % 1.0
        best = min(sites, key=lambda dst: float(np.linalg.norm(wrap_frac(image - sites[dst]))))
        rimg = integer_R(np.zeros(3), 2.0 * INV_C - tau - sites[best])
        inv[src] = {
            "maps_to": best,
            "frac_residual": float(np.linalg.norm(wrap_frac(image - sites[best]))),
            "R_image": [int(x) for x in integer_R(tau, 2.0 * INV_C - sites[best])],
        }
        del rimg
    recip = 2.0 * np.pi * np.linalg.inv(lat).T
    return {
        "lattice_ang": lat.tolist(),
        "reciprocal_2pi_inv_aT": recip.tolist(),
        "sites_frac": {k: v.tolist() for k, v in sites.items()},
        "inversion_center_frac": INV_C.tolist(),
        "inversion_pairs": inv,
        "P_geom_QE_labels": "tau_x sigma_x",
        "fourier_convention": (
            "T_αβ(k)=Σ t_αβ(R) exp(i k_cart · (R+τ_bβ-τ_tα)); "
            "k_cart in Å^{-1}; crystal k via k_cart = k_frac @ recip"
        ),
        "monolayer_NN_top_A_to_B": nn_top,
        "monolayer_NN_bot_A_to_B": nn_bot,
        "monolayer_NNN_top_AA": nnn_aa,
        "monolayer_NNN_top_BB": nnn_bb,
        "nn_angles_deg_top": angles,
        "ideal_120_minus_actual_deg": [a - 120.0 for a in angles],
        "km_nnn_from_same_NN": nnn_signed,
        "km_nnn_from_bot_NN": nnn_signed_bot,
        "g_construction": g_note,
        "interlayer_nearest": nearest,
        "max_delta_identity_residual": max((b["identity_residual"] for b in nearest), default=1.0),
        "idealization": (
            "No extra rotation of the monolayer. NN/NNN/T use production "
            "coordinates and the same Bravais lattice. Angle deviations from 120° "
            "are reported, not removed."
        ),
    }


def k_cart_from_frac(recip, kfrac):
    return np.asarray(kfrac, dtype=float) @ recip


def f_from_nn(k_cart, nn_bonds) -> complex:
    acc = 0.0 + 0.0j
    for b in nn_bonds:
        d = np.array(b["delta_cart"][:2])
        acc += np.exp(1j * float(np.dot(k_cart[:2], d)))
    return complex(acc)


def g_from_nn(k_cart, nnn_signed) -> float:
    g = 0.0
    k2 = np.asarray(k_cart[:2], dtype=float)
    for item in nnn_signed:
        v = np.array(item["vector"], dtype=float)
        g += 2.0 * item["nu"] * math.sin(float(np.dot(k2, v)))
    return g


def tab_from_interlayer(k_cart, bonds, amplitude: float) -> np.ndarray:
    idx = {"A": 0, "B": 1}
    tab = np.zeros((2, 2), dtype=complex)
    k2 = np.asarray(k_cart[:2], dtype=float)
    for b in bonds:
        d = np.array(b["delta_cart"][:2], dtype=float)
        phase = np.exp(1j * float(np.dot(k2, d)))
        tab[idx[b["from"][1]], idx[b["to"][1]]] += amplitude * phase
    return tab


def historical_cartoon_tab(kx_cart: float, ky_cart: float) -> np.ndarray:
    """Frozen handwritten C3; not generated from the bond table."""
    a = 6.998304941
    d = a / math.sqrt(3.0)
    vecs_aa = [np.array([0.0, d]), np.array([-0.5 * math.sqrt(3.0) * d, -0.5 * d]), np.array([0.5 * math.sqrt(3.0) * d, -0.5 * d])]
    vecs_ab = [np.array([d, 0.0]), np.array([-0.5 * d, 0.5 * math.sqrt(3.0) * d]), np.array([-0.5 * d, -0.5 * math.sqrt(3.0) * d])]
    faa = sum(np.exp(1j * (kx_cart * v[0] + ky_cart * v[1])) for v in vecs_aa) / 3.0
    fab = sum(np.exp(1j * (kx_cart * v[0] + ky_cart * v[1])) for v in vecs_ab) / 3.0
    return T_SCALE * (0.55 * faa * np.eye(2) + 0.45 * np.array([[0.0, fab], [np.conj(fab), 0.0]], dtype=complex))


def d_mu(tab: np.ndarray) -> dict:
    s0, sx, sy, sz = pauli()
    out = {}
    for name, s in (("d0", s0), ("dx", sx), ("dy", sy), ("dz", sz)):
        val = 0.5 * np.trace(s @ tab)
        out[name] = [float(val.real), float(val.imag)]
    return out


def split_ab(tab: np.ndarray):
    a = 0.5 * (tab + tab.conj().T)
    b = (tab - tab.conj().T) / (2.0j)
    return a, b


def honeycomb_h(k_cart, bonds, t=1.0, lso=0.1, delta=0.0, layer: str = "top") -> np.ndarray:
    s0, sx, sy, sz = pauli()
    if layer == "bot":
        nn = bonds["monolayer_NN_bot_A_to_B"]
        nnn = bonds["km_nnn_from_bot_NN"]
    else:
        nn = bonds["monolayer_NN_top_A_to_B"]
        nnn = bonds["km_nnn_from_same_NN"]
    fk = f_from_nn(k_cart, nn)
    hop = t * np.kron(np.array([[0.0, fk], [np.conj(fk), 0.0]], dtype=complex), s0)
    gk = g_from_nn(k_cart, nnn)
    km = lso * gk * np.kron(sz, sz)
    mag = delta * np.kron(s0, sz)
    return hop + km + mag


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def assemble_h8(h0_plus: np.ndarray, h0_minus: np.ndarray, tab: np.ndarray) -> np.ndarray:
    s0 = np.eye(2, dtype=complex)
    tp = np.kron(tab, s0)
    h = np.zeros((N_ORB8, N_ORB8), dtype=complex)
    h[:4, :4] = h0_plus
    h[4:, 4:] = h0_minus
    h[:4, 4:] = tp
    h[4:, :4] = tp.conj().T
    return h


def tau_y_B_residual_matrix(b_ab: np.ndarray) -> np.ndarray:
    s0, sx, sy, sz = pauli()
    b4 = np.kron(b_ab, s0)
    ty = kron3(sy, s0, s0)
    # 2 τ_y ⊗ B ⊗ s0 acting on 8: ty @ blkdiag-like embedding of B is not this.
    # Build [[0, -2i B4],[2i B4, 0]] which equals 2 τ_y ⊗ B4 if τ_y is layer.
    out = np.zeros((8, 8), dtype=complex)
    out[:4, 4:] = -2.0j * b4
    out[4:, :4] = 2.0j * b4
    del ty, sz, sx
    return out


def named_tab(name: str, k_cart, bonds) -> np.ndarray:
    s0, sx, sy, sz = pauli()
    del sy, sz
    if name == "C1_SIGMA0":
        return T_SCALE * s0
    if name == "C2_SIGMA_X":
        return T_SCALE * sx
    if name == "GEO_EQUAL_AMP":
        n = max(len(bonds["interlayer_nearest"]), 1)
        return tab_from_interlayer(k_cart, bonds["interlayer_nearest"], T_SCALE / n)
    if name == "C3_HERM_KDEP":
        raw = named_tab("GEO_EQUAL_AMP", k_cart, bonds)
        return 0.5 * (raw + raw.conj().T)
    if name == "C3_CARTOON_055_045":
        return historical_cartoon_tab(float(k_cart[0]), float(k_cart[1]))
    raise KeyError(name)


CANDIDATE_META = {
    "C1_SIGMA0": {
        "role": "control",
        "source": "k-independent t_perp I_2, scale 0.15; not from materials",
        "is_effective_C3": False,
    },
    "C2_SIGMA_X": {
        "role": "control",
        "source": "k-independent t_perp sigma_x, scale 0.15; not from materials",
        "is_effective_C3": False,
    },
    "GEO_EQUAL_AMP": {
        "role": "geometry_illustration",
        "source": "equal amp on the four nearest interlayer Cr-Cr vectors with corrected R; not a Wannier fit",
        "is_effective_C3": False,
    },
    "C3_HERM_KDEP": {
        "role": "algebra_control",
        "source": "Hermitian part of GEO_EQUAL_AMP; proves k-dependence alone does not break V",
        "is_effective_C3": False,
    },
    "C3_CARTOON_055_045": {
        "role": "historical_cartoon",
        "source": "q8_tperp_material_audit.py handwritten 0.55/0.45; old residual ~0.094 belongs only here",
        "is_effective_C3": False,
        "legacy_json": str(OLD_DIR / "tperp_candidate_symmetry.json"),
    },
    "C3_EFF_NOT_BUILT": {
        "role": "placeholder",
        "source": "material-constrained effective C3 is not constructed this round (no downfold, no eV freeze)",
        "is_effective_C3": False,
    },
}


def sample_kpts(recip):
    fracs = [
        ("Gamma", [0.0, 0.0, 0.0]),
        ("M", [0.5, 0.0, 0.0]),
        ("Kfrac", [1.0 / 3.0, 1.0 / 3.0, 0.0]),
        ("generic1", [0.17, -0.11, 0.0]),
        ("generic2", [0.31, 0.22, 0.0]),
        ("generic3", [-0.08, 0.41, 0.0]),
    ]
    out = []
    for name, kf in fracs:
        kc = k_cart_from_frac(recip, kf)
        out.append({"name": name, "k_frac": kf, "k_cart": kc.tolist()})
    return out


def frobenius(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def static_for_candidate(name, bonds, kpts, t=1.0, lso=0.1):
    s0, sx, sy, sz = pauli()
    v = kron3(sx, s0, sz)
    rows = []
    max_v = max_id = max_pt = max_tr = max_pr = 0.0
    max_mismatch = 0.0
    for kp in kpts:
        kc = np.array(kp["k_cart"])
        mk = -kc
        tab = named_tab(name, kc, bonds)
        a_ab, b_ab = split_ab(tab)
        h0p = honeycomb_h(kc, bonds, t=t, lso=lso, delta=DELTA)
        h0m = honeycomb_h(kc, bonds, t=t, lso=lso, delta=-DELTA)
        hp = assemble_h8(h0p, h0m, tab)
        hm = assemble_h8(h0m, h0p, tab)
        ident = frobenius(v @ hp @ v.conj().T - hm)
        pred = tau_y_B_residual_matrix(b_ab)
        got = v @ hp @ v.conj().T - hm
        mismatch = frobenius(got - pred)
        # TR
        ut = kron3(s0, s0, 1.0j * sy)
        h0p_mk = honeycomb_h(mk, bonds, t=t, lso=lso, delta=DELTA)
        h0m_mk = honeycomb_h(mk, bonds, t=t, lso=lso, delta=-DELTA)
        tab_mk = named_tab(name, mk, bonds)
        hm_mk = assemble_h8(h0m_mk, h0p_mk, tab_mk)
        tr = frobenius(ut @ hp.conj() @ ut.conj().T - hm_mk)
        # geometric P = tau_x sigma_x, k -> -k, N -> -N
        up = kron3(sx, sx, s0)
        hp_mk_minus = assemble_h8(h0m_mk, h0p_mk, tab_mk)
        pr = frobenius(up @ hp @ up.conj().T - hp_mk_minus)
        # PT
        pt = frobenius(up @ (ut @ hp.conj() @ ut.conj().T) @ up.conj().T - hp)
        rows.append(
            {
                "k": kp["name"],
                "d_mu": d_mu(tab),
                "A": [[float(z.real), float(z.imag)] for z in a_ab.ravel()],
                "B": [[float(z.real), float(z.imag)] for z in b_ab.ravel()],
                "T_minus_Tdag_F": frobenius(tab - tab.conj().T),
                "B_F": frobenius(b_ab),
                "V_residual_F": ident,
                "two_tauy_B_F": frobenius(pred),
                "V_minus_two_tauy_B_F": mismatch,
                "T_isyK_residual_F": tr,
                "P_taux_sigmax_residual_F": pr,
                "PT_residual_F": pt,
                "herm_T_F": frobenius(tab - tab.conj().T),
            }
        )
        max_v = max(max_v, ident)
        max_id = max(max_id, frobenius(tab - tab.conj().T))
        max_pt = max(max_pt, pt)
        max_tr = max(max_tr, tr)
        max_pr = max(max_pr, pr)
        max_mismatch = max(max_mismatch, mismatch)
    return {
        "meta": CANDIDATE_META[name],
        "max_V_residual_F": max_v,
        "max_T_antiHerm_F": max_id,
        "max_V_minus_2tauyB_F": max_mismatch,
        "max_T_residual_F": max_tr,
        "max_P_residual_F": max_pr,
        "max_PT_residual_F": max_pt,
        "k_points": rows,
    }


def layer_phase_unitaries():
    s0, sx, sy, sz = pauli()
    out = {}
    for label, u2 in (
        ("tau_x", sx),
        ("tau_y", sy),
        ("tau_x_sz", None),
    ):
        if label == "tau_x_sz":
            out[label] = kron3(sx, s0, sz)
        else:
            out[label] = kron3(u2, s0, s0)
    for n, phi in enumerate(np.linspace(0.0, math.pi, 9)):
        phase = np.exp(1j * phi)
        u = np.zeros((8, 8), dtype=complex)
        u[:4, 4:] = phase * np.eye(4)
        u[4:, :4] = np.conj(phase) * np.eye(4)
        out[f"layer_phase_{n}*pi/8"] = u
    out["tau_x_sigma_x"] = kron3(sx, sx, s0)
    out["tau_x_sigma_x_sz"] = kron3(sx, sx, sz)
    return out


def intertwiner_scan(name, bonds, kpts):
    """Constant U: stack H_-(k)U - U H_+(k) = 0 over sample k. Do not fit U(k)."""
    mats = []
    for kp in kpts:
        kc = np.array(kp["k_cart"])
        tab = named_tab(name, kc, bonds)
        h0p = honeycomb_h(kc, bonds, delta=DELTA)
        h0m = honeycomb_h(kc, bonds, delta=-DELTA)
        hp = assemble_h8(h0p, h0m, tab)
        hm = assemble_h8(h0m, h0p, tab)
        # (I ⊗ H_-) vec U - (H_+^T ⊗ I) vec U
        mats.append(np.kron(np.eye(8), hm) - np.kron(hp.T, np.eye(8)))
    big = np.vstack(mats)
    _u, s, vh = np.linalg.svd(big, full_matrices=False)
    tiny = [float(x) for x in s if x < 1.0e-8]
    known = {}
    for uname, u in layer_phase_unitaries().items():
        worst = 0.0
        for kp in kpts:
            kc = np.array(kp["k_cart"])
            tab = named_tab(name, kc, bonds)
            hp = assemble_h8(honeycomb_h(kc, bonds, delta=DELTA), honeycomb_h(kc, bonds, delta=-DELTA), tab)
            hm = assemble_h8(honeycomb_h(kc, bonds, delta=-DELTA), honeycomb_h(kc, bonds, delta=DELTA), tab)
            worst = max(worst, frobenius(hm @ u - u @ hp))
        known[uname] = worst
    return {
        "svd_smallest": [float(x) for x in s[-6:]],
        "n_sv_below_1e-8": len(tiny),
        "constant_U_nullspace_dim_estimate": len(tiny),
        "known_constant_maps_max_F": known,
        "coverage": [kp["name"] for kp in kpts],
        "note": (
            "The stacked intertwiner is 8×8 complex (64 entries). The SVD already "
            "covers all constant linear combinations of matrices; the Pauli/phase "
            "list only names known maps. Not a U(k). Antiunitary maps not included. "
            "Do not enlarge the Pauli enumeration."
        ),
    }


def site_cart(bonds) -> dict:
    lat = np.array(bonds["lattice_ang"], dtype=float)
    return {name: np.array(frac, dtype=float) @ lat for name, frac in bonds["sites_frac"].items()}


def sewing_U8(g_cart, bonds, origin: str = "site") -> np.ndarray:
    """U_ii = exp(i G·τ_i). Then H(k+G) = U† H(k) U in the user Fourier convention.

    origin='site': each orbital uses its stored τ (tA,tB,bA,bB).
    origin='shared_top': both layers use top τ. This matches the written model
    that copies one h0 onto both layers; it is not a claim that the bottom
    atoms sit at the top positions.
    """
    taus = site_cart(bonds)
    if origin == "shared_top":
        names = ("tA", "tA", "tB", "tB", "tA", "tA", "tB", "tB")
    else:
        names = ("tA", "tA", "tB", "tB", "bA", "bA", "bB", "bB")
    u = np.zeros((8, 8), dtype=complex)
    for i, name in enumerate(names):
        u[i, i] = np.exp(1j * float(np.dot(g_cart, taus[name])))
    return u


def sewing_U_tab(g_cart, bonds):
    taus = site_cart(bonds)
    st = np.diag(
        [
            np.exp(1j * float(np.dot(g_cart, taus["tA"]))),
            np.exp(1j * float(np.dot(g_cart, taus["tB"]))),
        ]
    )
    sb = np.diag(
        [
            np.exp(1j * float(np.dot(g_cart, taus["bA"]))),
            np.exp(1j * float(np.dot(g_cart, taus["bB"]))),
        ]
    )
    return st, sb


def inversion_pairing_proposal(lat, sites_orig: dict) -> dict:
    """Freeze top sites and INV_C; snap bottom to exact images. Original coords kept."""
    paired = {
        "tA": np.array(sites_orig["tA"], dtype=float),
        "tB": np.array(sites_orig["tB"], dtype=float),
        "bB": (2.0 * INV_C - sites_orig["tA"]) % 1.0,
        "bA": (2.0 * INV_C - sites_orig["tB"]) % 1.0,
    }
    rows = []
    for name in ("tA", "tB", "bA", "bB"):
        dfrac = wrap_frac(paired[name] - sites_orig[name])
        dcart = dfrac @ lat
        image = (2.0 * INV_C - sites_orig[name]) % 1.0
        best = min(sites_orig, key=lambda dst: float(np.linalg.norm(wrap_frac(image - sites_orig[dst]))))
        rows.append(
            {
                "site": name,
                "original_frac": np.asarray(sites_orig[name], dtype=float).tolist(),
                "paired_frac": paired[name].tolist(),
                "delta_frac": dfrac.tolist(),
                "delta_cart_ang": dcart.tolist(),
                "delta_cart_norm_ang": float(np.linalg.norm(dcart)),
                "original_inversion_partner": best,
                "original_inversion_frac_residual": float(np.linalg.norm(wrap_frac(image - sites_orig[best]))),
            }
        )
    return {
        "rule": "keep production tA,tB and INV_C; set bA=2c-tB, bB=2c-tA (mod 1)",
        "kept_original_coordinates": True,
        "not_a_new_crystal": True,
        "sites": rows,
        "max_delta_cart_ang": max(r["delta_cart_norm_ang"] for r in rows),
        "paired_sites_frac": {k: v.tolist() for k, v in paired.items()},
    }, paired


def sewing_check(bonds, kpts):
    recip = np.array(bonds["reciprocal_2pi_inv_aT"])
    g_list = [
        ("G100", [1.0, 0.0, 0.0]),
        ("G010", [0.0, 1.0, 0.0]),
        ("G110", [1.0, 1.0, 0.0]),
    ]
    names = [n for n in CANDIDATE_META if n != "C3_EFF_NOT_BUILT"]
    matrix_rows = []
    max_by_name = {
        n: {"T": 0.0, "H8_shared_top": 0.0, "H8_site_tau": 0.0, "h0": 0.0, "H8_layer_specific_h0_site_tau": 0.0}
        for n in names
    }
    for gname, gfrac in g_list:
        gcart = k_cart_from_frac(recip, gfrac)
        u_site = sewing_U8(gcart, bonds, origin="site")
        u_shared = sewing_U8(gcart, bonds, origin="shared_top")
        st, sb = sewing_U_tab(gcart, bonds)
        for kp in kpts:
            kc = np.array(kp["k_cart"], dtype=float)
            h0 = honeycomb_h(kc, bonds, delta=DELTA, layer="top")
            h0g = honeycomb_h(kc + gcart, bonds, delta=DELTA, layer="top")
            u4 = np.kron(st, np.eye(2, dtype=complex))
            h0_sew = frobenius(h0g - u4.conj().T @ h0 @ u4)
            h0b = honeycomb_h(kc, bonds, delta=-DELTA, layer="bot")
            h0bg = honeycomb_h(kc + gcart, bonds, delta=-DELTA, layer="bot")
            for name in names:
                tab = named_tab(name, kc, bonds)
                tabg = named_tab(name, kc + gcart, bonds)
                t_sew = frobenius(tabg - st.conj().T @ tab @ sb)
                hp = assemble_h8(h0, honeycomb_h(kc, bonds, delta=-DELTA, layer="top"), tab)
                hpg = assemble_h8(
                    h0g,
                    honeycomb_h(kc + gcart, bonds, delta=-DELTA, layer="top"),
                    tabg,
                )
                h8_shared = frobenius(hpg - u_shared.conj().T @ hp @ u_shared)
                h8_site = frobenius(hpg - u_site.conj().T @ hp @ u_site)
                hp2 = assemble_h8(h0, h0b, tab)
                hpg2 = assemble_h8(h0g, h0bg, tabg)
                h8_two = frobenius(hpg2 - u_site.conj().T @ hp2 @ u_site)
                matrix_rows.append(
                    {
                        "G": gname,
                        "k": kp["name"],
                        "candidate": name,
                        "T_sewing_F": t_sew,
                        "H8_shared_top_F": h8_shared,
                        "H8_site_tau_F": h8_site,
                        "H8_layer_specific_h0_site_tau_F": h8_two,
                        "h0_sewing_F": h0_sew,
                    }
                )
                max_by_name[name]["T"] = max(max_by_name[name]["T"], t_sew)
                max_by_name[name]["H8_shared_top"] = max(max_by_name[name]["H8_shared_top"], h8_shared)
                max_by_name[name]["H8_site_tau"] = max(max_by_name[name]["H8_site_tau"], h8_site)
                max_by_name[name]["h0"] = max(max_by_name[name]["h0"], h0_sew)
                max_by_name[name]["H8_layer_specific_h0_site_tau"] = max(
                    max_by_name[name]["H8_layer_specific_h0_site_tau"], h8_two
                )
    tau_diff = (np.array(bonds["sites_frac"]["tB"]) - np.array(bonds["sites_frac"]["tA"])) @ np.array(
        bonds["lattice_ang"]
    )
    f_rows = []
    gcart100 = k_cart_from_frac(recip, [1.0, 0.0, 0.0])
    for kp in kpts:
        kc = np.array(kp["k_cart"])
        f = f_from_nn(kc, bonds["monolayer_NN_top_A_to_B"])
        fg = f_from_nn(kc + gcart100, bonds["monolayer_NN_top_A_to_B"])
        pred = np.exp(1j * float(np.dot(gcart100, tau_diff))) * f
        f_rows.append({"k": kp["name"], "f_sewing_residual": abs(fg - pred)})
    h0 = honeycomb_h(np.array(kpts[0]["k_cart"]), bonds, delta=0.0)
    h0m = honeycomb_h(-np.array(kpts[0]["k_cart"]), bonds, delta=0.0)
    _, sx, _, _ = pauli()
    p_ab = np.kron(sx, np.eye(2, dtype=complex))
    return {
        "G_frac_list": [g[1] for g in g_list],
        "convention": "H(k+G)=U(G)† H(k) U(G), U_ii=exp(i G_cart·τ_i); T(k+G)=S_t† T(k) S_b",
        "monolayer_f_sewing": f_rows,
        "matrix_sewing": matrix_rows,
        "max_sewing_F_by_candidate": max_by_name,
        "sigma_x_h0_k_vs_h0_minus_k_F": frobenius(p_ab @ h0 @ p_ab.conj().T - h0m),
        "note": (
            "H(k+G) need not be elementwise equal to H(k). "
            "GEO T sews with site τ. Shared h0 copied onto both layers sews with "
            "shared_top U, not with site τ on the bottom block. "
            "C1/C2 are k-independent and are not bond-table Fourier transforms. "
            "Layer-specific h0 is a comparison only, not a model change. "
            "Static sewing is not same-drive current equivalence."
        ),
    }


def pairing_symmetry_compare(lat, sites_orig, kpts_builder):
    proposal, paired = inversion_pairing_proposal(lat, sites_orig)
    bonds_orig = build_bond_table(lat, sites_orig)
    bonds_paired = build_bond_table(lat, paired)
    recip_o = np.array(bonds_orig["reciprocal_2pi_inv_aT"])
    recip_p = np.array(bonds_paired["reciprocal_2pi_inv_aT"])
    kpts_o = kpts_builder(recip_o)
    kpts_p = kpts_builder(recip_p)
    names = ("GEO_EQUAL_AMP", "C3_HERM_KDEP", "C1_SIGMA0", "C2_SIGMA_X")
    cmp = {}
    for name in names:
        rec_o = static_for_candidate(name, bonds_orig, kpts_o)
        rec_p = static_for_candidate(name, bonds_paired, kpts_p)
        cmp[name] = {
            "original_max_P_F": max(r["P_taux_sigmax_residual_F"] for r in rec_o["k_points"]),
            "original_max_PT_F": rec_o["max_PT_residual_F"],
            "original_max_T_F": rec_o["max_T_residual_F"],
            "paired_max_P_F": max(r["P_taux_sigmax_residual_F"] for r in rec_p["k_points"]),
            "paired_max_PT_F": rec_p["max_PT_residual_F"],
            "paired_max_T_F": rec_p["max_T_residual_F"],
            "paired_max_V_F": rec_p["max_V_residual_F"],
        }
    sew_p = sewing_check(bonds_paired, kpts_p)
    return {
        "proposal": proposal,
        "pt_p_compare": cmp,
        "paired_GEO_H8_shared_top_sewing_max_F": sew_p["max_sewing_F_by_candidate"]["GEO_EQUAL_AMP"]["H8_shared_top"],
        "paired_GEO_T_sewing_max_F": sew_p["max_sewing_F_by_candidate"]["GEO_EQUAL_AMP"]["T"],
        "paired_GEO_H8_site_tau_sewing_max_F": sew_p["max_sewing_F_by_candidate"]["GEO_EQUAL_AMP"]["H8_site_tau"],
        "origin_of_original_PT": (
            "Equal-amp GEO already uses one amplitude on all four nearest bonds. "
            "The 1e-6-class PT residual tracks the incomplete inversion pairing "
            "(tB image vs bA, frac residual ~6e-6), not unequal hopping weights."
        ),
    }


def recount_tb(path: Path) -> dict:
    cr_layer = {0: "t", 1: "t", 2: "b", 3: "b"}
    cr_ab = {0: "A", 1: "B", 2: "A", 3: "B"}
    interface_top = {0, 1, 2}
    interface_bot = {9, 10, 11}

    def wf_atom(i):
        if i < 40:
            return "Cr", i // 10
        return "I", (i - 40) // 6

    def wf_spin(i):
        if i < 40:
            return (i % 10) % 2
        return ((i - 40) % 6) % 2

    def wf_i_orb(i):
        return ("pz", "px", "py")[((i - 40) % 6) // 2]

    acc = {
        "weight_definitions": {
            "sum_abs2_raw": "Σ |H_mn(R)|², Wannier90 raw coefficient",
            "sum_abs2_over_ndegen": "Σ |H_mn(R)|² / n_R  (old w2; not (old amp)²)",
            "sum_abs_over_ndegen": "Σ |H_mn(R)| / n_R  (old amp)",
            "fourier_coeff": "H_mn(R)/n_R is the Wannier90 Fourier weight for exp(ik·R), no τ in the phase",
        },
        "direction": "top->bottom only for Cr-Cr, I-I, and Cr-I",
        "label_source": "nnkp spinor_projections: Cr 0-3 then I 0-11; d m=1..5 × (↑↓); I (pz,px,py)×(↑↓)",
        "R_objects": "Wannier90 translation buckets, not distance shells and not an approved effective model",
        "CrCr": defaultdict(float),
        "II": defaultdict(float),
        "CrI": defaultdict(float),
        "II_R_abs2_over_ndegen": defaultdict(float),
        "II_orb_abs2_over_ndegen": defaultdict(float),
        "CrCr_AB_spincons": defaultdict(float),
        "CrCr_AB_total": defaultdict(float),
    }
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for _ in range(3):
            fh.readline()
        nw = int(fh.readline().split()[0])
        nr = int(fh.readline().split()[0])
        ndegen = []
        while len(ndegen) < nr:
            ndegen.extend(int(x) for x in fh.readline().split())
        for ir in range(nr):
            line = fh.readline()
            while line and not line.strip():
                line = fh.readline()
            rvec = tuple(int(x) for x in line.split()[:3])
            deg = float(ndegen[ir])
            for _ in range(nw * nw):
                p = fh.readline().split()
                i, j = int(p[0]) - 1, int(p[1]) - 1
                re, im = float(p[2]), float(p[3])
                abs2 = re * re + im * im
                si, ai = wf_atom(i)
                sj, aj = wf_atom(j)
                if si == "Cr" and sj == "Cr":
                    if cr_layer[ai] != "t" or cr_layer[aj] != "b":
                        continue
                    key = cr_ab[ai] + cr_ab[aj]
                    acc["CrCr"]["sum_abs2_raw"] += abs2
                    acc["CrCr"]["sum_abs2_over_ndegen"] += abs2 / deg
                    acc["CrCr"]["sum_abs_over_ndegen"] += math.sqrt(abs2) / deg
                    acc["CrCr_AB_total"][key] += abs2 / deg
                    if wf_spin(i) == wf_spin(j):
                        acc["CrCr_AB_spincons"][key] += abs2 / deg
                    else:
                        acc["CrCr"]["spinflip_abs2_over_ndegen"] += abs2 / deg
                elif si == "I" and sj == "I":
                    if not (ai in interface_top and aj in interface_bot):
                        continue
                    acc["II"]["sum_abs2_raw"] += abs2
                    acc["II"]["sum_abs2_over_ndegen"] += abs2 / deg
                    acc["II"]["sum_abs_over_ndegen"] += math.sqrt(abs2) / deg
                    acc["II_R_abs2_over_ndegen"][str(list(rvec))] += abs2 / deg
                    acc["II_orb_abs2_over_ndegen"][wf_i_orb(i) + "-" + wf_i_orb(j)] += abs2 / deg
                    if wf_spin(i) != wf_spin(j):
                        acc["II"]["spinflip_abs2_over_ndegen"] += abs2 / deg
                elif si == "Cr" and sj == "I":
                    if cr_layer[ai] != "t" or aj not in interface_bot:
                        continue
                    acc["CrI"]["sum_abs2_over_ndegen"] += abs2 / deg
                elif si == "I" and sj == "Cr":
                    continue
    def _plain(d):
        if isinstance(d, defaultdict):
            return {k: (float(v) if not isinstance(v, dict) else v) for k, v in d.items()}
        return d
    ii = acc["II"]["sum_abs2_over_ndegen"]
    dd = acc["CrCr"]["sum_abs2_over_ndegen"]
    buckets = sorted(acc["II_R_abs2_over_ndegen"].items(), key=lambda kv: -kv[1])
    cum = 0.0
    bucket_rows = []
    for key, val in buckets[:8]:
        cum += val
        bucket_rows.append({"R_bucket": key, "sum_abs2_over_ndegen": val, "cumulative_of_II": cum / ii if ii else 0.0})
    return {
        "nw": nw,
        "nrpts": nr,
        "weights": acc["weight_definitions"],
        "direction": acc["direction"],
        "label_source": acc["label_source"],
        "R_objects": acc["R_objects"],
        "CrCr": _plain(acc["CrCr"]),
        "II": _plain(acc["II"]),
        "CrI_top_to_bot_interface": _plain(acc["CrI"]),
        "CrCr_AB_spincons": dict(acc["CrCr_AB_spincons"]),
        "CrCr_AB_total": dict(acc["CrCr_AB_total"]),
        "II_orb": dict(acc["II_orb_abs2_over_ndegen"]),
        "II_R_buckets": bucket_rows,
        "ratio_II_over_CrCr_abs2_over_ndegen": (ii / dd) if dd else None,
        "II_spinflip_fraction_of_II": acc["II"].get("spinflip_abs2_over_ndegen", 0.0) / ii if ii else None,
        "caveat": (
            "1.6%-class spinflip fraction is a microscopic I-I tag statistic, "
            "not an effective interlayer-SOC or Jo error bar. "
            "Five R buckets are not five approved effective shells."
        ),
    }


def source_table():
    rows = []
    for name, meta in CANDIDATE_META.items():
        rows.append({"name": name, **meta})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tb", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    lat, atoms = parse_scf(SCF)
    sites = cr_sites(atoms)
    bonds = build_bond_table(lat, sites)
    recip = np.array(bonds["reciprocal_2pi_inv_aT"])
    kpts = sample_kpts(recip)
    (OUT / "bond_table.json").write_text(json.dumps(bonds, indent=2), encoding="utf-8")
    (OUT / "candidate_source_table.json").write_text(json.dumps(source_table(), indent=2), encoding="utf-8")
    names = [n for n in CANDIDATE_META if n != "C3_EFF_NOT_BUILT"]
    static = {n: static_for_candidate(n, bonds, kpts) for n in names}
    (OUT / "static_residuals.json").write_text(json.dumps(static, indent=2), encoding="utf-8")
    inter = {n: intertwiner_scan(n, bonds, kpts) for n in names}
    (OUT / "intertwiner_constant_U.json").write_text(json.dumps(inter, indent=2), encoding="utf-8")
    sew = sewing_check(bonds, kpts)
    (OUT / "bloch_sewing.json").write_text(json.dumps(sew, indent=2), encoding="utf-8")
    pairing = pairing_symmetry_compare(lat, sites, sample_kpts)
    (OUT / "inversion_pairing.json").write_text(json.dumps(pairing, indent=2), encoding="utf-8")
    hop = None
    if not args.skip_tb and TB.is_file():
        hop = recount_tb(TB)
        (OUT / "hopping_recount.json").write_text(json.dumps(hop, indent=2), encoding="utf-8")
    index = {
        "schema": "q8_tperp_static_audit_v1",
        "not_a_physics_result": True,
        "C3_frozen": False,
        "downfold_performed": False,
        "old_results_kept_at": str(OLD_DIR),
        "bond_identity_max": bonds["max_delta_identity_residual"],
        "files": [
            "bond_table.json",
            "candidate_source_table.json",
            "static_residuals.json",
            "intertwiner_constant_U.json",
            "bloch_sewing.json",
            "inversion_pairing.json",
            "hopping_recount.json" if hop else None,
        ],
    }
    (OUT / "audit_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("OUT", OUT)
    print("delta identity max", bonds["max_delta_identity_residual"])
    print("NN top", len(bonds["monolayer_NN_top_A_to_B"]), "angles", bonds["nn_angles_deg_top"])
    print("nearest R", [(b["from"], b["to"], b["R"], round(b["d3"], 3)) for b in bonds["interlayer_nearest"]])
    for n, rec in static.items():
        print(
            n,
            "V",
            rec["max_V_residual_F"],
            "P",
            rec["max_P_residual_F"],
            "PT",
            rec["max_PT_residual_F"],
            "TantiH",
            rec["max_T_antiHerm_F"],
        )
    print("pairing max dr Ang", pairing["proposal"]["max_delta_cart_ang"])
    print("paired GEO PT", pairing["pt_p_compare"]["GEO_EQUAL_AMP"]["paired_max_PT_F"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""M0: eight-band PT-AFM mechanism model (line Q), version m0_v1.

Definition: docs/04_BSV量子光/线Q_M0八带模型与经典基线_Cursor任务单_20260911.md §2–3.
Phenomenological, geometry- and evidence-constrained. Not a material fit, not a downfold.

Basis (fixed):  (tA↑, tA↓, tB↑, tB↓, bA↑, bA↓, bB↑, bB↓).
Embedded Bloch convention:  H_ab(k) = Σ_R t_ab(R) exp(i k·(R + τ_b − τ_a)),
k Cartesian in Å^-1, kz = 0.  With this convention the Peierls substitution is
k → κ = k + eA/ħ on every term and the current is j_a = -(e/ħ) ∂H/∂k_a.

Units inside the code: ħ = 1, e = 1, energies in t_h, lengths in Å.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# single geometric source shared with the static audit
from q8_tperp_static_audit import INV_C, SCF, cr_sites, parse_scf

MODEL_ID = "Q_M0_eightband_v1"
BASIS = ("tA_up", "tA_dn", "tB_up", "tB_dn", "bA_up", "bA_dn", "bB_up", "bB_dn")
SITE_OF_ORBITAL = ("tA", "tA", "tB", "tB", "bA", "bA", "bB", "bB")
N_ORB = 8
N_OCC = 4  # half filling: four electrons per cell (spinor bands)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pauli():
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return s0, sx, sy, sz


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


# --------------------------------------------------------------------------- geometry
@dataclass
class M0Geometry:
    lattice_ang: list  # 3x3 rows a1,a2,a3
    sites_orig_frac: dict
    sites_paired_frac: dict
    inversion_center_frac: list
    d_cart: list  # τ_tB - τ_tA
    q_cart: list  # τ_bA - τ_tA
    a1_cart: list
    a2_cart: list
    nn_deltas_cart: list  # A->B in-layer: d, d-a1, d+a2
    nnn_terms: list  # [nu, b_cart] with g = 2 Σ nu sin(k·b)
    interlayer_bonds: list  # dicts: channel, from, to, R, delta_cart, amplitude_symbol
    a_nn_mean_ang: float
    a_cell_ang2: float
    reciprocal_2d: list  # rows b1,b2 (Å^-1, includes 2π)
    mirror_normal_inplane: list  # n̂ ⊥ a2
    source_hashes: dict
    pairing_rule: str = "keep production tA,tB and INV_C; bA=(2c-tB) mod 1; bB=(2c-tA) mod 1"
    idealization: str = (
        "production lattice metric kept; only bA moved ~3.5e-5 Å to the exact inversion image; "
        "no 120° forcing, no relabeling, no stacking change"
    )

    def lat(self) -> np.ndarray:
        return np.array(self.lattice_ang, dtype=float)

    def recip(self) -> np.ndarray:
        return np.array(self.reciprocal_2d, dtype=float)

    def k_cart(self, kfrac) -> np.ndarray:
        """fractional (n1,n2) -> Cartesian 2D."""
        kf = np.asarray(kfrac, dtype=float)
        return kf[..., :2] @ self.recip()

    def site_cart(self, name: str) -> np.ndarray:
        return np.array(self.sites_paired_frac[name], dtype=float) @ self.lat()

    def tau_orbitals_cart(self) -> np.ndarray:
        return np.array([self.site_cart(s) for s in SITE_OF_ORBITAL])


def build_geometry(scf_path: Path = SCF) -> M0Geometry:
    lat, atoms = parse_scf(scf_path)
    sites = cr_sites(atoms)
    c = np.asarray(INV_C, dtype=float)
    paired = {
        "tA": np.array(sites["tA"], dtype=float),
        "tB": np.array(sites["tB"], dtype=float),
        "bA": (2.0 * c - sites["tB"]) % 1.0,
        "bB": (2.0 * c - sites["tA"]) % 1.0,
    }
    a1, a2 = lat[0], lat[1]
    tau = {k: v @ lat for k, v in paired.items()}
    d = tau["tB"] - tau["tA"]
    q = tau["bA"] - tau["tA"]
    # exact pairing gives τ_bB - τ_bA = d
    assert np.max(np.abs((tau["bB"] - tau["bA"]) - d)) < 1.0e-9
    nn = [d, d - a1, d + a2]
    nnn = [[+1.0, (a1 + a2).tolist()], [-1.0, a1.tolist()], [-1.0, a2.tolist()]]
    bonds = [
        {"channel": "AA", "from": "tA", "to": "bA", "R": [0, 0, 0], "delta_cart": q.tolist(), "amplitude": "u"},
        {"channel": "BB", "from": "tB", "to": "bB", "R": [0, 0, 0], "delta_cart": (tau["bB"] - tau["tB"]).tolist(), "amplitude": "u"},
        {"channel": "AB", "from": "tA", "to": "bB", "R": [0, 1, 0], "delta_cart": (a2 + tau["bB"] - tau["tA"]).tolist(), "amplitude": "v"},
        {"channel": "BA", "from": "tB", "to": "bA", "R": [0, 0, 0], "delta_cart": (tau["bA"] - tau["tB"]).tolist(), "amplitude": "v"},
    ]
    # identities used in the closed form of T0
    assert np.allclose(bonds[1]["delta_cart"], q, atol=1e-9)
    assert np.allclose(bonds[2]["delta_cart"], q + d + a2, atol=1e-9)
    assert np.allclose(bonds[3]["delta_cart"], q - d, atol=1e-9)
    a_nn = float(np.mean([np.linalg.norm(v) for v in nn]))
    a_cell = float(abs(a1[0] * a2[1] - a1[1] * a2[0]))
    lat2 = np.array([a1[:2], a2[:2]])
    recip2 = 2.0 * np.pi * np.linalg.inv(lat2).T
    a2u = a2[:2] / np.linalg.norm(a2[:2])
    n_hat = np.array([a2u[1], -a2u[0]])
    here = Path(__file__).resolve()
    hashes = {
        "scf_input": {"path": str(scf_path), "sha256": sha256_file(scf_path)},
        "model_module": {"path": str(here), "sha256": sha256_file(here)},
        "geometry_source_module": {
            "path": str(here.parent / "q8_tperp_static_audit.py"),
            "sha256": sha256_file(here.parent / "q8_tperp_static_audit.py"),
        },
    }
    return M0Geometry(
        lattice_ang=lat.tolist(),
        sites_orig_frac={k: np.asarray(v, dtype=float).tolist() for k, v in sites.items()},
        sites_paired_frac={k: v.tolist() for k, v in paired.items()},
        inversion_center_frac=c.tolist(),
        d_cart=d.tolist(),
        q_cart=q.tolist(),
        a1_cart=a1.tolist(),
        a2_cart=a2.tolist(),
        nn_deltas_cart=[v.tolist() for v in nn],
        nnn_terms=nnn,
        interlayer_bonds=bonds,
        a_nn_mean_ang=a_nn,
        a_cell_ang2=a_cell,
        reciprocal_2d=recip2.tolist(),
        mirror_normal_inplane=n_hat.tolist(),
        source_hashes=hashes,
    )


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class M0Params:
    t_h: float = 1.0  # eV, also the energy unit
    lam_so: float = 0.1
    delta: float = 0.8
    u: float = 0.1
    v: float = 0.1
    label: str = "R0"

    def as_dict(self) -> dict:
        return asdict(self)


R0 = M0Params()


# --------------------------------------------------------------------------- structure factors
def _phase(k2: np.ndarray, vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    return np.exp(1j * (k2[..., 0] * vec[0] + k2[..., 1] * vec[1]))


def f_and_grad(geo: M0Geometry, k2: np.ndarray):
    f = np.zeros(k2.shape[:-1], dtype=complex)
    df = np.zeros(k2.shape[:-1] + (2,), dtype=complex)
    for delta in geo.nn_deltas_cart:
        ph = _phase(k2, delta)
        f = f + ph
        df[..., 0] += 1j * delta[0] * ph
        df[..., 1] += 1j * delta[1] * ph
    return f, df


def g_and_grad(geo: M0Geometry, k2: np.ndarray):
    g = np.zeros(k2.shape[:-1], dtype=float)
    dg = np.zeros(k2.shape[:-1] + (2,), dtype=float)
    for nu, b in geo.nnn_terms:
        arg = k2[..., 0] * b[0] + k2[..., 1] * b[1]
        g = g + 2.0 * nu * np.sin(arg)
        dg[..., 0] += 2.0 * nu * b[0] * np.cos(arg)
        dg[..., 1] += 2.0 * nu * b[1] * np.cos(arg)
    return g, dg


def t0_and_grad(geo: M0Geometry, p: M0Params, k2: np.ndarray):
    """T0(k) (…,2,2) and ∂T0/∂k_a (…,2,2,2) from the four bonds."""
    t0 = np.zeros(k2.shape[:-1] + (2, 2), dtype=complex)
    dt0 = np.zeros(k2.shape[:-1] + (2, 2, 2), dtype=complex)
    idx = {"A": 0, "B": 1}
    amp = {"u": p.u, "v": p.v}
    for b in geo.interlayer_bonds:
        i, j = idx[b["from"][1]], idx[b["to"][1]]
        ph = amp[b["amplitude"]] * _phase(k2, b["delta_cart"])
        t0[..., i, j] += ph
        dt0[..., 0, i, j] += 1j * b["delta_cart"][0] * ph
        dt0[..., 1, i, j] += 1j * b["delta_cart"][1] * ph
    return t0, dt0


# --------------------------------------------------------------------------- Hamiltonian
def _layer_block(f, g, exch, t_h, lam):
    """4x4 (sub ⊗ spin) block: t_h[Re f σx − Im f σy] s0 + lam g σz sz + exch σ0 sz."""
    shp = np.shape(f)
    h = np.zeros(shp + (4, 4), dtype=complex)
    h[..., 0, 2] = t_h * f
    h[..., 1, 3] = t_h * f
    h[..., 2, 0] = t_h * np.conj(f)
    h[..., 3, 1] = t_h * np.conj(f)
    soc = lam * g
    h[..., 0, 0] += soc + exch
    h[..., 1, 1] += -soc - exch
    h[..., 2, 2] += -soc + exch
    h[..., 3, 3] += soc - exch
    return h


def _interlayer_block(t0):
    shp = t0.shape[:-2]
    tp = np.zeros(shp + (4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            tp[..., 2 * a, 2 * b] = t0[..., a, b]
            tp[..., 2 * a + 1, 2 * b + 1] = t0[..., a, b]
    return tp


def assemble(ht, hb, tp):
    shp = ht.shape[:-2]
    h = np.zeros(shp + (8, 8), dtype=complex)
    h[..., :4, :4] = ht
    h[..., 4:, 4:] = hb
    h[..., :4, 4:] = tp
    h[..., 4:, :4] = np.conj(np.swapaxes(tp, -1, -2))
    return h


def h8(geo: M0Geometry, p: M0Params, k2, neel: int) -> np.ndarray:
    """H_{8,N}(k) for Cartesian k2 (…,2). neel = ±1."""
    k2 = np.asarray(k2, dtype=float)
    f, _ = f_and_grad(geo, k2)
    g, _ = g_and_grad(geo, k2)
    t0, _ = t0_and_grad(geo, p, k2)
    ht = _layer_block(f, g, +neel * p.delta, p.t_h, p.lam_so)
    hb = _layer_block(f, g, -neel * p.delta, p.t_h, p.lam_so)
    return assemble(ht, hb, _interlayer_block(t0))


def dh8(geo: M0Geometry, p: M0Params, k2, neel: int) -> np.ndarray:
    """∂H/∂k_a, shape (…,2,8,8). Exchange term is k-independent."""
    k2 = np.asarray(k2, dtype=float)
    _, df = f_and_grad(geo, k2)
    _, dg = g_and_grad(geo, k2)
    _, dt0 = t0_and_grad(geo, p, k2)
    out = np.zeros(k2.shape[:-1] + (2, 8, 8), dtype=complex)
    for a in range(2):
        ht = _layer_block(df[..., a], dg[..., a], 0.0, p.t_h, p.lam_so)
        out[..., a, :, :] = assemble(ht, ht, _interlayer_block(dt0[..., a, :, :]))
    del neel
    return out


def current_ops(geo: M0Geometry, p: M0Params, k2, neel: int) -> np.ndarray:
    """j_a = -(e/ħ) ∂H/∂k_a with e = ħ = 1 → (…,2,8,8)."""
    return -dh8(geo, p, k2, neel)


def h8_and_current(geo, p, k2, neel):
    k2 = np.asarray(k2, dtype=float)
    f, df = f_and_grad(geo, k2)
    g, dg = g_and_grad(geo, k2)
    t0, dt0 = t0_and_grad(geo, p, k2)
    h = assemble(
        _layer_block(f, g, +neel * p.delta, p.t_h, p.lam_so),
        _layer_block(f, g, -neel * p.delta, p.t_h, p.lam_so),
        _interlayer_block(t0),
    )
    j = np.zeros(k2.shape[:-1] + (2, 8, 8), dtype=complex)
    for a in range(2):
        hd = _layer_block(df[..., a], dg[..., a], 0.0, p.t_h, p.lam_so)
        j[..., a, :, :] = -assemble(hd, hd, _interlayer_block(dt0[..., a, :, :]))
    return h, j


# --------------------------------------------------------------------------- symmetry operators
def op_P() -> np.ndarray:
    s0, sx, _, _ = pauli()
    return kron3(sx, sx, s0)


def op_theta_unitary() -> np.ndarray:
    """Θ = U_Θ K with U_Θ = τ0 σ0 (i s_y)."""
    s0, _, sy, _ = pauli()
    return kron3(s0, s0, 1.0j * sy)


def op_sz() -> np.ndarray:
    s0, _, _, sz = pauli()
    return kron3(s0, s0, sz)


def op_V_specified() -> np.ndarray:
    """Old 4-band-type same-k map V = τx σ0 sz (not a symmetry of M0 unless T0 = T0†)."""
    s0, sx, _, sz = pauli()
    return kron3(sx, s0, sz)


def op_sewing(geo: M0Geometry, g_cart2) -> np.ndarray:
    """U_G = diag(exp(i G·τ_i)); H(k+G) = U_G† H(k) U_G."""
    taus = geo.tau_orbitals_cart()
    g = np.asarray(g_cart2, dtype=float)
    ph = np.exp(1j * (taus[:, 0] * g[0] + taus[:, 1] * g[1]))
    return np.diag(ph)


def mirror_matrix_inplane(geo: M0Geometry) -> np.ndarray:
    n = np.asarray(geo.mirror_normal_inplane, dtype=float)
    return np.eye(2) - 2.0 * np.outer(n, n)


def op_mirror_a2(geo: M0Geometry) -> np.ndarray:
    """Vertical mirror containing a2 and z: A↔B in both layers, spin C2 about n̂.
    Expected: U_M H_N(k) U_M† = H_{-N}(M k)."""
    s0, sx, sy, _ = pauli()
    n = np.asarray(geo.mirror_normal_inplane, dtype=float)
    spin = -1.0j * (n[0] * sx + n[1] * sy)
    return kron3(s0, sx, spin)


def op_c2_nhat(geo: M0Geometry) -> np.ndarray:
    """C2 about in-plane n̂ (= P∘M): layers swapped, sublattice kept, spin C2(n̂).
    Expected: U H_N(k) U† = H_N(-M k)."""
    return op_P() @ op_mirror_a2(geo)


# --------------------------------------------------------------------------- analytic references
def gamma_energies(p: M0Params) -> list:
    out = []
    for r in (+1, -1):
        root = np.sqrt(p.delta**2 + (p.u + r * p.v) ** 2)
        out.append(3.0 * p.t_h * r + root)
        out.append(3.0 * p.t_h * r - root)
    return sorted(out)


def inlayer_spin_block(geo: M0Geometry, p: M0Params, k2, spin: float):
    """2×2 sublattice block for one spin: NN hopping + Kane–Mele, no exchange."""
    f, _ = f_and_grad(geo, k2)
    g, _ = g_and_grad(geo, k2)
    h = np.zeros(np.asarray(k2).shape[:-1] + (2, 2), dtype=complex)
    h[..., 0, 1] = p.t_h * f
    h[..., 1, 0] = p.t_h * np.conj(f)
    soc = p.lam_so * g * spin
    h[..., 0, 0] = soc
    h[..., 1, 1] = -soc
    return h


def same_spectrum_barred_residual(geo: M0Geometry, p: M0Params, k2, *, spin: float = 1.0, delta: float | None = None):
    """Corrected same-spectrum map on one spin 4×4 block.

    T0 is normal for M0 (u_AA=u_BB and |v_AB|=|v_BA|). After a unitary that
    diagonalises T0 and a diagonal phase that makes the in-layer block real,
    τx H_δ* τx = H_{-δ} in that barred basis. This is a per-k antiunitary
    spectral map, not a same-pulse dynamical symmetry, so it does not imply Jo=0.
    """
    if delta is None:
        delta = p.delta
    t0, _ = t0_and_grad(geo, p, np.asarray(k2, dtype=float))
    h = inlayer_spin_block(geo, p, np.asarray(k2, dtype=float), spin)
    _evals, w_raw = np.linalg.eig(t0)
    w_u, _r = np.linalg.qr(w_raw)
    d = w_u.conj().T @ t0 @ w_u
    hp = w_u.conj().T @ h @ w_u
    phase = 0.0 if abs(hp[0, 1]) < 1e-15 else -np.angle(hp[0, 1])
    s = np.diag([1.0 + 0.0j, np.exp(1j * phase)])
    hbar = s.conj().T @ hp @ s
    dbar = s.conj().T @ d @ s
    eye = np.eye(2, dtype=complex)
    h_plus = np.block([[hbar + delta * eye, dbar], [dbar.conj().T, hbar - delta * eye]])
    h_minus = np.block([[hbar - delta * eye, dbar], [dbar.conj().T, hbar + delta * eye]])
    tau_x = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), eye)
    residual = tau_x @ h_plus.conj() @ tau_x - h_minus
    return {
        "hbar_imag_max": float(np.max(np.abs(hbar.imag))),
        "dbar_offdiag": float(np.max(np.abs(dbar - np.diag(np.diag(dbar))))),
        "residual": float(np.linalg.norm(residual)),
        "T_normal_residual": float(np.linalg.norm(t0 @ t0.conj().T - t0.conj().T @ t0)),
    }


def spec_dict(geo: M0Geometry, p: M0Params) -> dict:
    return {
        "model_id": MODEL_ID,
        "basis": list(BASIS),
        "bloch_convention": "embedded: H_ab(k)=Σ_R t_ab(R) exp(i k·(R+τ_b−τ_a)); k Cartesian Å^-1; kz=0",
        "units": {"hbar": 1, "e": 1, "energy": "t_h (=1 eV at R0)", "length": "Å", "time": "ħ/t_h (0.6582 fs at t_h=1 eV)"},
        "hamiltonian": {
            "h_layer": "t_h[Re f σx − Im f σy] s0 + λ_SO g σz sz + N η_ℓ Δ σ0 sz, η_t=+1, η_b=−1",
            "f": "exp(i k·d)[1 + exp(−i k·a1) + exp(i k·a2)], d = τ_tB − τ_tA",
            "g": "2{sin[k·(a1+a2)] − sin(k·a1) − sin(k·a2)}  (not normalised; |g(K)| = 3√3)",
            "T0": "exp(i k·q) [[u, v exp(i k·(d+a2))], [v exp(−i k·d), u]], q = τ_bA − τ_tA",
            "H8": "[[h_t, T0⊗s0], [T0†⊗s0, h_b]]",
            "H_R": 0,
            "interlayer_SOC": 0,
            "extra_interlayer_hopping": 0,
            "energy_zero": "Tr-free onsite 0; no k-dependent scalar dispersion (choice, not omission)",
        },
        "parameters": p.as_dict(),
        "symmetry_operators": {
            "P": "τx σx s0 (k→−k, N→−N); common scalar phase e^{2ik·c} dropped",
            "Theta": "τ0 σ0 (i sy) K (k→−k, N→−N)",
            "PTheta_squared": -1,
            "spin_conservation": "[H, sz] = 0 (model assumption)",
            "mirror_a2": "τ0 σx (−i n̂·s): H_N(k) → H_{−N}(Mk)",
            "C2_nhat": "τx σ0 (−i n̂·s): H_N(k) → H_N(−Mk); requires u_AA=u_BB and v_AB=v_BA",
        },
        "geometry": asdict(geo),
        "not_claimed": [
            "material fit", "downfold", "Rashba", "interlayer SOC", "HHG mechanism", "quantum input/output",
        ],
    }


def write_spec(path: Path, geo: M0Geometry, p: M0Params) -> None:
    path.write_text(json.dumps(spec_dict(geo, p), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    geo = build_geometry()
    print(json.dumps({k: v for k, v in asdict(geo).items() if k in ("d_cart", "q_cart", "a_nn_mean_ang", "a_cell_ang2")}, indent=2))
    print("Gamma energies", gamma_energies(R0))

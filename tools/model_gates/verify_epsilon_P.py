#!/usr/bin/env python3
"""verify_epsilon_P.py — explicit U_P(k) / S and ε_P gate (classical AFM L0).

Follows tools/model_gates/SPEC_epsilon_P.md and Obsidian「量子光理论核心」.

Two embedding gauges (do not mix):

  stored (default for P-conjugate TB on disk):
      H_{-N}(-k) = S H_{+N}(k) S†,  S_{a'b} = η_b δ_{a',P(b)}
      Matches construct_p_conjugate_tb real-space re-celling (no k-phases).

  crystallographic:
      H_{-N}(-k) = U_P(k) H_{+N}(k) U_P†(k),
      U_P(k) = diag(e^{2π i k·L_a}) S
      Original-cell Wannier gauge; not the stored TB convention.

Usage:
  python verify_epsilon_P.py --config config_epsilon_P.yml
  python verify_epsilon_P.py --config config_epsilon_P.yml --mode file
  python verify_epsilon_P.py --config config_epsilon_P.yml --gauge stored
  python verify_epsilon_P.py --config config_epsilon_P.yml --gauge both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse pairing / TB IO from the P-conjugate constructor (same physics).
_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import construct_p_conjugate_tb as pct  # noqa: E402


def load_config(path: Path) -> dict:
    cfg: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def parse_vec3(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.replace(" ", "").split(",")], dtype=float)


def gate_level(value: float, pass_t: float, warn_t: float) -> str:
    if value < pass_t:
        return "PASS"
    if value < warn_t:
        return "WARN"
    return "FAIL"


def hk_from_tb(irvec, ndegen, h_r, kfrac: np.ndarray) -> np.ndarray:
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    hk = np.tensordot(phase, h_r, axes=(0, 0))
    return 0.5 * (hk + hk.conj().T)


def build_L_and_U_parts(
    centers_cart: np.ndarray,
    lattice: np.ndarray,
    center_frac: np.ndarray,
    perm: np.ndarray,
    parity: np.ndarray,
):
    """Return L (nw,3) integer and residual (nw,).

    Convention (notes): 2c - tau_a = tau_{a'} + L_a, L_a in Z^3.
    """
    del parity  # parity enters U_P, not L
    centers_frac = (centers_cart @ np.linalg.inv(lattice)) % 1.0
    nw = centers_frac.shape[0]
    L = np.zeros((nw, 3), dtype=int)
    resid = np.zeros(nw)
    for a in range(nw):
        ap = int(perm[a])
        raw = (2.0 * center_frac - centers_frac[a]) - centers_frac[ap]
        L[a] = np.rint(raw).astype(int)
        resid[a] = float(np.linalg.norm(raw - L[a]))
    return L, resid, centers_frac


def S_matrix(perm: np.ndarray, parity: np.ndarray) -> np.ndarray:
    """Stored-gauge map: S_{a'b} = η_b δ_{a',P(b)} (k-independent)."""
    nw = perm.shape[0]
    S = np.zeros((nw, nw), dtype=np.complex128)
    for b in range(nw):
        S[int(perm[b]), b] = float(parity[b])
    return S


def U_P_at_k(perm: np.ndarray, parity: np.ndarray, L: np.ndarray, kfrac: np.ndarray) -> np.ndarray:
    """Crystallographic gauge: U_P(k) = D_L(k) S."""
    nw = perm.shape[0]
    U = np.zeros((nw, nw), dtype=np.complex128)
    phases = np.exp(2j * np.pi * (L @ kfrac))
    for b in range(nw):
        U[int(perm[b]), b] = phases[b] * float(parity[b])
    return U


def rk_from_tb(irvec, ndegen, r_r, kfrac: np.ndarray) -> np.ndarray:
    phase = np.exp(2j * np.pi * (irvec @ kfrac)) / ndegen
    return np.tensordot(phase, r_r, axes=(0, 0))


def rk_stored_map(
    S: np.ndarray, rp: np.ndarray, c_cart: np.ndarray, ndegen0: float
) -> np.ndarray:
    """r_{-N}(-k) = -S r_{+N}(k) S† + diag(2c)/ndegen(0)  (polar vector, stored gauge)."""
    rm = np.empty_like(rp)
    for a in range(3):
        rm[:, :, a] = -S @ rp[:, :, a] @ S.conj().T
    shift = 2.0 * c_cart / ndegen0
    for i in range(rp.shape[0]):
        rm[i, i, :] += shift
    return rm


def velocity_from_h(
    irvec, ndegen, h_r, kfrac: np.ndarray, dk_frac: float = 1.0e-5
) -> np.ndarray:
    """v_a = ∂H/∂k_a with k in fractional reciprocal coords (same as Wannier90 TB phase)."""
    kfrac = np.asarray(kfrac, float)
    v = np.zeros((h_r.shape[1], h_r.shape[1], 3), dtype=np.complex128)
    for a in range(3):
        step = np.zeros(3)
        step[a] = dk_frac
        hp = hk_from_tb(irvec, ndegen, h_r, kfrac + step)
        hm = hk_from_tb(irvec, ndegen, h_r, kfrac - step)
        v[:, :, a] = (hp - hm) / (2.0 * dk_frac)
    return v


def vk_stored_map(S: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """v is a polar vector: v_{-N}(-k) = -S v_{+N}(k) S† (no 2c shift)."""
    vm = np.empty_like(vp)
    for a in range(3):
        vm[:, :, a] = -S @ vp[:, :, a] @ S.conj().T
    return vm


def frobenius_eps_tensor(ref: np.ndarray, mapped: np.ndarray, delta: float) -> float:
    num = float(np.linalg.norm((ref - mapped).ravel(), ord=2))
    den = float(np.linalg.norm(ref.ravel(), ord=2)) + delta
    return num / den


def interband_dipole_strength(hk: np.ndarray, rk: np.ndarray) -> np.ndarray:
    """Per-band sum_{m!=n} |d_{nm}|^2 over Cartesian comps (gauge-invariant |d|)."""
    evals, evecs = np.linalg.eigh(hk)
    del evals
    return pct.interband_strength(evecs, rk)


def frobenius_eps(H_ref: np.ndarray, H_map: np.ndarray, delta: float) -> float:
    num = np.linalg.norm(H_ref - H_map, ord="fro")
    den = np.linalg.norm(H_ref, ord="fro") + delta
    return float(num / den)


def make_kmesh(nk1: int, nk2: int, nk3: int) -> np.ndarray:
    pts = []
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                pts.append((i / nk1, j / nk2, k / nk3 if nk3 > 1 else 0.0))
    return np.array(pts, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Explicit U_P(k) and ε_P gate")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=["pconj", "file"], default=None,
                    help="Override mode.minus in config")
    ap.add_argument("--quick", action="store_true",
                    help="Only named k-points (no full mesh)")
    ap.add_argument(
        "--gauge",
        choices=["stored", "crystallographic", "both"],
        default=None,
        help="Embedding gauge for ε_P (default: gate.gauge in config, else stored)",
    )
    ap.add_argument(
        "--operators",
        default=None,
        help="Comma-separated audit list: H,r,v (default: audit.operators in config)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    plus_tb = Path(cfg["path.plus_tb"])
    plus_win = Path(cfg["path.plus_win"])
    plus_wout = Path(cfg["path.plus_wout"])
    outdir = Path(cfg["path.outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    center = parse_vec3(cfg["inversion.center_frac"])
    mode = args.mode or cfg.get("mode.minus", "pconj")
    gauge = args.gauge or cfg.get("gate.gauge", "stored")
    ops_raw = args.operators or cfg.get("audit.operators", "H,r,v")
    audit_ops = {x.strip().upper() for x in ops_raw.split(",") if x.strip()}
    delta = float(cfg.get("eps.delta", "1e-30"))
    dk_frac = float(cfg.get("velocity.dk_frac", "1e-5"))
    run_stored = gauge in ("stored", "both")
    run_cryst = gauge in ("crystallographic", "both")
    need_pos = "R" in audit_ops

    lattice = pct.parse_lattice_ang(plus_win)
    c_cart = center @ lattice
    centers_cart = pct.parse_final_centers(plus_wout)
    centers_frac = (centers_cart @ np.linalg.inv(lattice)) % 1.0
    irvec, ndegen, h_plus, r_plus = pct.read_tb_blocks(plus_tb, read_position=need_pos)
    nw = h_plus.shape[1]
    ir0 = next(i for i in range(irvec.shape[0]) if tuple(irvec[i]) == (0, 0, 0))
    ndegen0 = float(ndegen[ir0])
    meta = pct.build_wf_metadata(nw)
    perm, diag = pct.build_permutation(centers_frac, meta, center)
    parity = meta["parity"]
    L, resid, _ = build_L_and_U_parts(
        centers_cart, lattice, center, perm, parity
    )

    # --- structure / pairing report ---
    struct_lines = [
        "# structure_summary\n",
        f"- inversion.center_frac = {center.tolist()}\n",
        f"- nwann = {nw}\n",
        f"- atom_pair max/mean frac dist = {diag['max_atom_pair_dist']:.3e} / {diag['mean_atom_pair_dist']:.3e}\n",
        f"- duplicate_targets = {diag['duplicate_targets']}\n",
        f"- non_involutive = {diag['non_involutive']}\n",
        f"- L residual: max={resid.max():.3e}, mean={resid.mean():.3e}\n",
        f"- mode.minus = {mode}\n",
        f"- gate.gauge = {gauge}\n",
        f"- audit.operators = {','.join(sorted(audit_ops))}\n",
    ]
    (outdir / "structure_summary.md").write_text("".join(struct_lines), encoding="utf-8")

    with (outdir / "pairing_table.csv").open("w", encoding="utf-8") as fh:
        fh.write("a,a_prime,eta,Lx,Ly,Lz,L_residual\n")
        for a in range(nw):
            fh.write(
                f"{a},{int(perm[a])},{int(parity[a])},"
                f"{L[a,0]},{L[a,1]},{L[a,2]},{resid[a]:.6e}\n"
            )

    if diag["duplicate_targets"] or diag["non_involutive"]:
        print("FAIL: permutation not a clean involution. See structure_summary.md")
        return 2

    # --- build / load H_{-N}(R) ---
    neg_idx = pct.neg_R_index(irvec)
    r_minus = None
    if mode == "pconj":
        h_minus = pct.build_minus_hr(h_plus, perm, parity, neg_idx)
        if need_pos:
            if r_plus is None:
                raise SystemExit("audit R requires position block in +N TB")
            r_minus = pct.build_minus_rr(
                r_plus, perm, parity, neg_idx, irvec, c_cart
            )
        irvec_m, ndegen_m = irvec, ndegen
    else:
        minus_tb = Path(cfg["path.minus_tb"])
        irvec_m, ndegen_m, h_minus, r_minus = pct.read_tb_blocks(
            minus_tb, read_position=need_pos
        )
        if h_minus.shape[1] != nw:
            raise SystemExit(f"nw mismatch: +N {nw} vs -N {h_minus.shape[1]}")

    # --- unitarity spot-check (S and U_P(Γ) = S) ---
    S = S_matrix(perm, parity)
    s_unit_err = float(np.linalg.norm(S.conj().T @ S - np.eye(nw), ord="fro"))
    U0 = U_P_at_k(perm, parity, L, np.zeros(3))
    u0_vs_s = float(np.linalg.norm(U0 - S, ord="fro"))
    unit_err = float(np.linalg.norm(U0.conj().T @ U0 - np.eye(nw), ord="fro"))
    (outdir / "U_P_check_unitarity.md").write_text(
        "# gauge unitarity\n\n"
        f"- gate.gauge run: `{gauge}`\n"
        f"- ||S†S - I||_F = {s_unit_err:.3e}\n"
        f"- ||U_P(Γ) - S||_F = {u0_vs_s:.3e}\n"
        f"- ||U_P(Γ)†U_P(Γ) - I||_F = {unit_err:.3e}\n",
        encoding="utf-8",
    )

    # --- k points ---
    named = {}
    for key, val in cfg.items():
        if key.startswith("kpoint."):
            named[key.split(".", 1)[1]] = parse_vec3(val)

    if args.quick:
        klist = list(named.values())
        knames = list(named.keys())
    else:
        nk1 = int(cfg.get("kmesh.nk1", "12"))
        nk2 = int(cfg.get("kmesh.nk2", "12"))
        nk3 = int(cfg.get("kmesh.nk3", "1"))
        mesh = make_kmesh(nk1, nk2, nk3)
        klist = [mesh[i] for i in range(len(mesh))]
        knames = [f"m{i}" for i in range(len(mesh))]
        # append named
        for name, kv in named.items():
            klist.append(kv)
            knames.append(name)

    def summarize_eps(
        eps_arr: np.ndarray,
        prefix: str = "gate",
    ) -> tuple[float, float, float, str, str, str, str]:
        med = float(np.median(eps_arr))
        p99 = float(np.quantile(eps_arr, 0.99))
        mx = float(np.max(eps_arr))
        g_med = gate_level(
            med,
            float(cfg.get(f"{prefix}.median.pass", cfg["gate.median.pass"])),
            float(cfg.get(f"{prefix}.median.warn", cfg["gate.median.warn"])),
        )
        g_p99 = gate_level(
            p99,
            float(cfg.get(f"{prefix}.p99.pass", cfg["gate.p99.pass"])),
            float(cfg.get(f"{prefix}.p99.warn", cfg["gate.p99.warn"])),
        )
        g_max = gate_level(
            mx,
            float(cfg.get(f"{prefix}.max.pass", cfg["gate.max.pass"])),
            float(cfg.get(f"{prefix}.max.warn", cfg["gate.max.warn"])),
        )
        overall = "FAIL" if "FAIL" in (g_med, g_p99, g_max) else (
            "WARN" if "WARN" in (g_med, g_p99, g_max) else "PASS"
        )
        return med, p99, mx, g_med, g_p99, g_max, overall

    rows = []
    eig_worst = 0.0
    dip_worst = 0.0
    for name, kfrac in zip(knames, klist):
        kfrac = np.asarray(kfrac, float)
        km = (-kfrac) % 1.0
        Hp = hk_from_tb(irvec, ndegen, h_plus, kfrac)
        Hm = hk_from_tb(irvec_m, ndegen_m, h_minus, km)
        eps_s = np.nan
        eps_u = np.nan
        eps_r = np.nan
        eps_v = np.nan
        dip_rel = np.nan
        if run_stored and "H" in audit_ops:
            Hmap_s = S @ Hp @ S.conj().T
            eps_s = frobenius_eps(Hm, Hmap_s, delta)
        if run_cryst and "H" in audit_ops:
            U = U_P_at_k(perm, parity, L, kfrac)
            Hmap_u = U @ Hp @ U.conj().T
            eps_u = frobenius_eps(Hm, Hmap_u, delta)
        if run_stored and need_pos and r_minus is not None:
            Rp = rk_from_tb(irvec, ndegen, r_plus, kfrac)
            Rm = rk_from_tb(irvec_m, ndegen_m, r_minus, km)
            Rmap = rk_stored_map(S, Rp, c_cart, ndegen0)
            eps_r = frobenius_eps_tensor(Rm, Rmap, delta)
            s_m = interband_dipole_strength(Hm, Rm)
            s_p = interband_dipole_strength(Hp, Rp)
            denom = np.maximum(s_p, 1e-6)
            dip_rel = float(np.max(np.abs(s_m - s_p) / denom))
            dip_worst = max(dip_worst, dip_rel)
        if run_stored and "V" in audit_ops:
            vp = velocity_from_h(irvec, ndegen, h_plus, kfrac, dk_frac)
            vm = velocity_from_h(irvec_m, ndegen_m, h_minus, km, dk_frac)
            vmap = vk_stored_map(S, vp)
            eps_v = frobenius_eps_tensor(vm, vmap, delta)
        ep = np.linalg.eigvalsh(Hp)
        em = np.linalg.eigvalsh(Hm)
        eig_err = float(np.max(np.abs(np.sort(ep) - np.sort(em))))
        eig_worst = max(eig_worst, eig_err)
        rows.append(
            (
                name,
                kfrac[0],
                kfrac[1],
                kfrac[2],
                eps_s,
                eps_u,
                eps_r,
                eps_v,
                dip_rel,
                eig_err,
                float(np.linalg.norm(Hm, ord="fro")),
            )
        )

    header = "name,kx,ky,kz"
    if run_stored and "H" in audit_ops:
        header += ",eps_P_S"
    if run_cryst and "H" in audit_ops:
        header += ",eps_P_U"
    if run_stored and need_pos:
        header += ",eps_r_S,dip_rel"
    if run_stored and "V" in audit_ops:
        header += ",eps_v_S"
    header += ",eig_abs_max_eV,Hminus_F\n"
    with (outdir / "stored_gauge_kgrid.csv").open("w", encoding="utf-8") as fh:
        fh.write(header)
        for r in rows:
            line = f"{r[0]},{r[1]:.8f},{r[2]:.8f},{r[3]:.8f}"
            idx = 4
            if run_stored and "H" in audit_ops:
                line += f",{r[idx]:.6e}"
                idx += 1
            if run_cryst and "H" in audit_ops:
                line += f",{r[idx]:.6e}"
                idx += 1
            if run_stored and need_pos:
                line += f",{r[6]:.6e},{r[8]:.6e}"
                idx = 9
            if run_stored and "V" in audit_ops:
                line += f",{r[7]:.6e}"
            line += f",{r[9]:.6e},{r[10]:.6e}\n"
            fh.write(line)

    # Legacy filename when only H is audited
    if audit_ops == {"H"}:
        with (outdir / "epsilon_P_kgrid.csv").open("w", encoding="utf-8") as fh:
            fh.write(header)
            for r in rows:
                line = f"{r[0]},{r[1]:.8f},{r[2]:.8f},{r[3]:.8f}"
                idx = 4
                if run_stored:
                    line += f",{r[idx]:.6e}"
                    idx += 1
                if run_cryst:
                    line += f",{r[idx]:.6e}"
                    idx += 1
                line += f",{r[idx]:.6e},{r[idx + 1]:.6e}\n"
                fh.write(line)

    eig_gate = float(cfg.get("eig.max_abs_eV", "1e-9"))
    eig_status = "PASS" if eig_worst < eig_gate else "FAIL"

    dip_gate = float(cfg.get("dip.max_rel", "1e-4"))
    dip_status = "PASS" if dip_worst < dip_gate else "CHECK"

    stats = [
        "# stored_gauge_operator_stats\n\n",
        f"- mode.minus: `{mode}`\n",
        f"- gate.gauge: `{gauge}`\n",
        f"- audit.operators: `{','.join(sorted(audit_ops))}`\n",
        f"- n_k: {len(rows)}\n",
        f"- spectral corollary max|E_+N(k)-E_-N(-k)| (sorted): {eig_worst:.6e} eV → **{eig_status}**\n",
        f"- ||S†S-I||_F: {s_unit_err:.6e}; ||U_P(Γ)-S||_F: {u0_vs_s:.6e}\n",
        f"- ndegen(R=0) = {ndegen0:g}\n",
    ]

    exit_code = 0
    overalls: dict[str, str] = {}

    if run_stored and "H" in audit_ops:
        eps_s_arr = np.array([r[4] for r in rows])
        med, p99, mx, g_med, g_p99, g_max, overall = summarize_eps(eps_s_arr)
        overalls["H"] = overall
        stats.extend([
            "\n## H — stored gauge: ε_P^S\n\n",
            f"- median: {med:.6e} → **{g_med}**\n",
            f"- P99: {p99:.6e} → **{g_p99}**\n",
            f"- max: {mx:.6e} → **{g_max}**\n",
            f"- overall: **{overall}**\n",
        ])
        print(f"H       overall={overall}  median={med:.3e}  max={mx:.3e}")

    if run_cryst and "H" in audit_ops:
        eps_u_arr = np.array([r[5] for r in rows])
        med, p99, mx, g_med, g_p99, g_max, overall = summarize_eps(eps_u_arr)
        stats.extend([
            "\n## H — crystallographic gauge: ε_P^U (reference only)\n\n",
            f"- median: {med:.6e} → **{g_med}**\n",
            f"- max: {mx:.6e} → **{g_max}**\n",
        ])
        print(f"H(cryst) median={med:.3e}  max={mx:.3e}")

    if run_stored and need_pos:
        eps_r_arr = np.array([r[6] for r in rows])
        med, p99, mx, g_med, g_p99, g_max, overall = summarize_eps(eps_r_arr, "gate.r")
        overalls["r"] = overall
        stats.extend([
            "\n## r — stored gauge: ε_r^S = ||r_-N(-k) - (-S r_+N(k) S† + 2c/ndegen0)||_F / ...\n\n",
            f"- median: {med:.6e} → **{g_med}**\n",
            f"- P99: {p99:.6e} → **{g_p99}**\n",
            f"- max: {mx:.6e} → **{g_max}**\n",
            f"- overall: **{overall}** (k-independent floor ~1.2e-10 from +N r Hermiticity)\n",
            f"- D4 corollary max rel |d^band|: {dip_worst:.6e} → **{dip_status}**\n",
        ])
        print(f"r       overall={overall}  median={med:.3e}  max={mx:.3e}  dip={dip_worst:.3e}")

    if run_stored and "V" in audit_ops:
        eps_v_arr = np.array([r[7] for r in rows])
        med, p99, mx, g_med, g_p99, g_max, overall = summarize_eps(eps_v_arr, "gate.v")
        overalls["v"] = overall
        stats.extend([
            "\n## v — stored gauge: ε_v^S = ||v_-N(-k) + S v_+N(k) S†||_F / ...\n\n",
            f"- v from central diff ∂H/∂k, dk_frac={dk_frac:g}\n",
            f"- median: {med:.6e} → **{g_med}**\n",
            f"- P99: {p99:.6e} → **{g_p99}**\n",
            f"- max: {mx:.6e} → **{g_max}**\n",
            f"- overall: **{overall}**\n",
        ])
        print(f"v       overall={overall}  median={med:.3e}  max={mx:.3e}")

    stats.extend([
        "\n## Named k-points\n\n",
        "| name | ε_P^S | ε_r^S | ε_v^S | dip_rel | eig (eV) |\n",
        "|------|------:|------:|------:|--------:|---------:|\n",
    ])
    for r in rows:
        if r[0] not in named:
            continue
        stats.append(
            f"| {r[0]} | {r[4]:.3e} | {r[6]:.3e} | {r[7]:.3e} | {r[8]:.3e} | {r[9]:.3e} |\n"
        )

    stats.append(
        "\n## Interpretation\n\n"
        "- **Production −N TB**: judge stored-gauge **ε_P^S, ε_r^S, ε_v^S**.\n"
        "- r map includes polar-vector minus sign and `+2c/ndegen(0)` on Wannier diagonal.\n"
        "- v map: `v_{-N}(-k) = -S v_{+N}(k) S†` with v = ∂H/∂k (lg_cov intraband channel).\n"
        "- D4 |d^band| corollary is gauge-invariant; matrix ε_r^S is the operator-level test.\n"
        "- ε_P^U FAIL with ε_P^S PASS ⇒ crystallographic/stored gauge mismatch only.\n"
        "- L0 magnetic-domain operator chain closes when H+r+v stored gates all PASS.\n"
        "- Do not treat this gate as quantum-light / QRT readiness.\n"
    )
    (outdir / "stored_gauge_operator_stats.md").write_text("".join(stats), encoding="utf-8")
    if audit_ops == {"H"}:
        (outdir / "epsilon_P_stats.md").write_text("".join(stats), encoding="utf-8")

    primary_overall = "PASS"
    for op, status in overalls.items():
        if status == "FAIL":
            primary_overall = "FAIL"
            break
        if status == "WARN" and primary_overall == "PASS":
            primary_overall = "WARN"

    print(f"outdir: {outdir}")
    print(f"mode={mode}  gauge={gauge}  ops={','.join(sorted(audit_ops))}  overall={primary_overall}")
    print(f"eig_worst={eig_worst:.3e} eV ({eig_status})")
    if primary_overall == "FAIL":
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

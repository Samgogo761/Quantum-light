#!/usr/bin/env python3
"""M0 classical drive: P0 reference and P1 protocols for the eight-band PT-AFM model.

Closed system, no dephasing. Embedded-orbital Peierls coupling κ(t) = k0 + eA(t)/ħ;
current j_a = -(e/ħ) ∂H/∂k_a from the same H. Midpoint unitary exponential
propagation. ħ = e = 1, energies in t_h, lengths Å.

Not quantum light. quantum_input = not_run, quantum_output = not_run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from q8_m0_model import MODEL_ID, N_OCC, M0Params, R0, build_geometry, current_ops, h8, h8_and_current, sha256_file

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "results/q_line/m0_v1_20260912"
ALGORITHM = "midpoint_unitary_exponential_v1"
CLASSICAL_MODULE = Path(__file__).resolve()


@dataclass(frozen=True)
class DriveProtocol:
    hbar_omega0: float = 0.2  # units of t_h
    n_cycles: int = 8
    a0_dimless: float = 0.4  # e a_NN A0 / ħ
    polarization: str = "x"
    envelope: str = "sin2_on_vector_potential"
    carrier: str = "cos"
    cep: float = 0.0
    label: str = "P0"

    @property
    def omega0(self) -> float:
        return self.hbar_omega0

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega0

    @property
    def t_pulse(self) -> float:
        return self.n_cycles * self.period

    @property
    def protocol_id(self) -> str:
        return (
            f"{self.label}_w{self.hbar_omega0:g}_A{self.a0_dimless:g}"
            f"_ncyc{self.n_cycles}_{self.polarization}_{self.envelope}_{self.carrier}_cep{self.cep:g}"
        )


P0Protocol = DriveProtocol


def p1_protocol(a_star: float) -> DriveProtocol:
    return DriveProtocol(hbar_omega0=0.05, a0_dimless=float(a_star), label=f"P1_A{a_star:g}")


def pol_vector(prot: DriveProtocol) -> np.ndarray:
    if prot.polarization == "x":
        return np.array([1.0, 0.0])
    if prot.polarization == "y":
        return np.array([0.0, 1.0])
    raise ValueError(prot.polarization)


def kappa_shift(t, geo, prot: DriveProtocol, sign: int = +1) -> np.ndarray:
    """a(t) = eA(t)/ħ  [Å^-1], shape (...,2)."""
    t = np.asarray(t, dtype=float)
    a0 = prot.a0_dimless / geo.a_nn_mean_ang
    inside = (t >= 0.0) & (t <= prot.t_pulse)
    env = np.where(inside, np.sin(np.pi * t / prot.t_pulse) ** 2, 0.0)
    scal = sign * a0 * env * np.cos(prot.omega0 * t + prot.cep)
    return scal[..., None] * pol_vector(prot)


def e_field(t, geo, prot: DriveProtocol, sign: int = +1) -> np.ndarray:
    """eE/ħ = -d a/dt  [Å^-1 t_h/ħ]. With t_h = 1 eV this is numerically E in V/Å."""
    t = np.asarray(t, dtype=float)
    a0 = prot.a0_dimless / geo.a_nn_mean_ang
    inside = (t >= 0.0) & (t <= prot.t_pulse)
    ph = np.pi * t / prot.t_pulse
    env = np.sin(ph) ** 2
    denv = (np.pi / prot.t_pulse) * np.sin(2.0 * ph)
    dadt = a0 * (denv * np.cos(prot.omega0 * t + prot.cep) - prot.omega0 * env * np.sin(prot.omega0 * t + prot.cep))
    dadt = np.where(inside, dadt, 0.0)
    return (-sign * dadt)[..., None] * pol_vector(prot)


def k_grid(geo, nk: int, shift: float = 0.0):
    """nk×nk Monkhorst-Pack grid (inversion symmetric), Cartesian, equal weights.

    shift=0: Γ-centred (original P0). shift=0.5: half-step shifted, still inversion-paired.
    """
    g1 = (np.arange(nk) + float(shift)) / nk
    kf = np.stack(np.meshgrid(g1, g1, indexing="ij"), axis=-1).reshape(-1, 2)
    kc = geo.k_cart(kf)
    w = np.full(kc.shape[0], 1.0 / kc.shape[0])
    return kf, kc, w


def run_fingerprint(geo, p: M0Params, prot: DriveProtocol, *, nk: int, nt_per_cycle: int, mu: float,
                    grid_shift: float, subspace: bool, k_orders: tuple[int, ...], neel: int, sign: int) -> dict:
    body = {
        "model_id": MODEL_ID,
        "algorithm": ALGORITHM,
        "dtype": "complex128",
        "params": p.as_dict(),
        "protocol": asdict(prot),
        "protocol_id": prot.protocol_id,
        "nk": int(nk),
        "nt_per_cycle": int(nt_per_cycle),
        "mu": float(mu),
        "grid_shift": float(grid_shift),
        "subspace": bool(subspace),
        "k_orders": [int(x) for x in k_orders],
        "neel": int(neel),
        "field_sign": int(sign),
        "a_nn_ang": float(geo.a_nn_mean_ang),
        "a_cell_ang2": float(geo.a_cell_ang2),
        "source_hashes": geo.source_hashes,
        "classical_module_sha256": sha256_file(CLASSICAL_MODULE),
    }
    blob = json.dumps(body, sort_keys=True, ensure_ascii=True)
    body["identity_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return body


def initial_rho(geo, p: M0Params, kc, neel: int, mu: float):
    h0 = h8(geo, p, kc, neel)
    e, w = np.linalg.eigh(h0)
    occ = (e < mu).astype(float)
    n_occ = occ.sum(axis=1)
    if not np.all(n_occ == N_OCC):
        raise RuntimeError(f"common μ does not give {N_OCC} occupied bands at every k: {np.unique(n_occ)}")
    rho = np.einsum("kin,kn,kjn->kij", w, occ, w.conj())
    return rho, e


def trace_current(rho, j):
    """j (Nk,2,8,8), rho (Nk,8,8) -> (Nk,2) real."""
    return np.real(np.einsum("kij,kaji->ka", rho, j))


def valence_projector(h):
    e, w = np.linalg.eigh(h)
    occ = np.zeros_like(e)
    occ[:, :N_OCC] = 1.0
    return np.einsum("kin,kn,kjn->kij", w, occ, w.conj())


def project_current_blocks(j, pv):
    """Instantaneous valence/conduction split: j = jvv + jcc + jvc."""
    pc = np.eye(8, dtype=complex)[None, :, :] - pv
    jvv = np.empty_like(j)
    jcc = np.empty_like(j)
    for a in range(2):
        jvv[:, a] = pv @ j[:, a] @ pv
        jcc[:, a] = pc @ j[:, a] @ pc
    jvc = j - jvv - jcc
    return jvv, jcc, jvc


def _sin2_window(t, tp):
    return np.sin(np.pi * t / tp) ** 2


def propagate(geo, p: M0Params, prot: DriveProtocol, *, nk: int, nt_per_cycle: int, neel: int, sign: int, mu: float,
              n_diag: int = 64, verbose: bool = True, grid_shift: float = 0.0, subspace: bool = False,
              k_orders=()) -> dict:
    k_orders = tuple(int(x) for x in k_orders)
    kf, kc, w = k_grid(geo, nk, shift=grid_shift)
    rho, e0 = initial_rho(geo, p, kc, neel, mu)
    rho0 = rho.copy()
    h0 = h8(geo, p, kc, neel)
    u_e0 = float(np.real(np.einsum("k,kij,kji->", w, rho0, h0)))
    nt = prot.n_cycles * nt_per_cycle
    dt = prot.period / nt_per_cycle
    times = np.arange(nt + 1) * dt
    j_t = np.zeros((nt + 1, 2))
    j_vv = np.zeros((nt + 1, 2)) if subspace else None
    j_cc = np.zeros((nt + 1, 2)) if subspace else None
    j_vc = np.zeros((nt + 1, 2)) if subspace else None
    diag_every = max(1, nt // n_diag)
    diag = {"t": [], "n_exc": [], "tr_rho": [], "herm": [], "idem": []}
    n_exc_peak = -1.0
    n_exc_k_peak = None
    n_exc_k_final = None
    k_acc = None
    k_prev = None
    if k_orders:
        n_blk = 4 if subspace else 1
        k_acc = np.zeros((n_blk, len(k_orders), kc.shape[0], 2), dtype=complex)
    work = 0.0
    e_prev = None
    t0 = time.perf_counter()
    omegas = np.array(k_orders, dtype=float) * prot.omega0
    for n in range(nt + 1):
        t = times[n]
        kap = kc + kappa_shift(t, geo, prot, sign)
        need_h = subspace or bool(k_orders) or n % diag_every == 0 or n == nt
        if need_h:
            h, j = h8_and_current(geo, p, kap, neel)
        else:
            h = None
            j = current_ops(geo, p, kap, neel)
        jk = trace_current(rho, j)
        j_t[n] = w @ jk
        pv = None
        n_exc = None
        n_exc_k = None
        if need_h:
            pv = valence_projector(h)
            n_exc_k = np.real(np.einsum("kii->k", rho - rho @ pv))
            n_exc = float(w @ n_exc_k)
            if n_exc > n_exc_peak:
                n_exc_peak = n_exc
                n_exc_k_peak = n_exc_k.copy()
            if n == nt:
                n_exc_k_final = n_exc_k
        if subspace:
            jvv, jcc, jvc = project_current_blocks(j, pv)
            j_vv[n] = w @ trace_current(rho, jvv)
            j_cc[n] = w @ trace_current(rho, jcc)
            j_vc[n] = w @ trace_current(rho, jvc)
        e_now = np.asarray(e_field(np.array([t]), geo, prot, sign), dtype=float)[0]
        if n > 0:
            work += 0.5 * dt * (float(np.dot(j_t[n - 1], e_prev)) + float(np.dot(j_t[n], e_now)))
        e_prev = e_now
        if k_orders:
            wgt = _sin2_window(t, prot.t_pulse)
            phases = np.exp(1j * omegas * t)
            blocks = [jk]
            if subspace:
                blocks.extend([trace_current(rho, jvv), trace_current(rho, jcc), trace_current(rho, jvc)])
            term = np.stack([(wgt * phases)[:, None, None] * blk[None, :, :] for blk in blocks], axis=0)
            if n > 0:
                k_acc += 0.5 * dt * (k_prev + term)
            k_prev = term
        if n % diag_every == 0 or n == nt:
            if n_exc is None:
                h = h8(geo, p, kap, neel)
                pv = valence_projector(h)
                n_exc = float(np.real(np.einsum("k,kii->", w, rho - rho @ pv)))
            diag["t"].append(float(t))
            diag["n_exc"].append(n_exc)
            diag["tr_rho"].append(float(np.real(np.einsum("k,kii->", w, rho))))
            diag["herm"].append(float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, 1, 2))))))
            diag["idem"].append(float(np.max(np.abs(rho @ rho - rho))))
        if n == nt:
            break
        kmid = kc + kappa_shift(t + 0.5 * dt, geo, prot, sign)
        hm = h8(geo, p, kmid, neel)
        em, wm = np.linalg.eigh(hm)
        ph = np.exp(-1j * em * dt)
        u = (wm * ph[:, None, :]) @ np.conj(np.swapaxes(wm, 1, 2))
        rho = u @ rho @ np.conj(np.swapaxes(u, 1, 2))
        if verbose and n % max(1, nt // 8) == 0:
            print(f"  step {n}/{nt}  t={t:.1f}  elapsed {time.perf_counter() - t0:.1f}s", flush=True)
    u_eT = float(np.real(np.einsum("k,kij,kji->", w, rho, h0)))
    j_sheet = j_t / geo.a_cell_ang2
    j_dimless = j_t / geo.a_nn_mean_ang
    a = kappa_shift(times, geo, prot, sign)
    ef = e_field(times, geo, prot, sign)
    out = {
        "neel": neel,
        "field_sign": sign,
        "nk": nk,
        "nt_per_cycle": nt_per_cycle,
        "grid_shift": float(grid_shift),
        "dt": dt,
        "times": times,
        "J_sheet": j_sheet,
        "J_dimless": j_dimless,
        "A_over": a,
        "E_over": ef,
        "diag": {k: np.array(v) for k, v in diag.items()},
        "J_equilibrium_t0": j_sheet[0].tolist(),
        "wall_seconds": time.perf_counter() - t0,
        "e0_band_ranges": [[float(e0[:, n].min()), float(e0[:, n].max())] for n in range(8)],
        "delta_U_e": u_eT - u_e0,
        "U_e0": u_e0,
        "U_eT": u_eT,
        "work_integral": work,
        "n_exc_peak": float(n_exc_peak if n_exc_peak >= 0.0 else np.max(diag["n_exc"])),
        "kf": kf,
        "w_k": w,
    }
    if subspace:
        out["J_vv_sheet"] = j_vv / geo.a_cell_ang2
        out["J_cc_sheet"] = j_cc / geo.a_cell_ang2
        out["J_vc_sheet"] = j_vc / geo.a_cell_ang2
        rec = np.max(np.abs(j_vv + j_cc + j_vc - j_t))
        out["subspace_reconstruct_maxabs"] = float(rec)
    if n_exc_k_peak is not None:
        out["n_exc_k_peak"] = n_exc_k_peak
    if n_exc_k_final is not None:
        out["n_exc_k_final"] = n_exc_k_final
    if k_orders:
        out["k_orders"] = np.array(k_orders)
        out["k_harm_total"] = k_acc[0] / prot.t_pulse
        if subspace:
            out["k_harm_vv"] = k_acc[1] / prot.t_pulse
            out["k_harm_cc"] = k_acc[2] / prot.t_pulse
            out["k_harm_vc"] = k_acc[3] / prot.t_pulse
    return out


def project_harmonic(times, signal, omega, window: str):
    tp = times[-1]
    if window == "sin2":
        wgt = np.sin(np.pi * times / tp) ** 2
    elif window == "boxcar":
        wgt = np.ones_like(times)
    else:
        raise ValueError(window)
    integrand = wgt[:, None] * signal * np.exp(1j * omega * times)[:, None]
    return np.trapezoid(integrand, times, axis=0) / tp  # (2,) complex


def harmonic_table(times, signal, omega0, orders, window):
    return {int(n): project_harmonic(times, signal, n * omega0, window) for n in orders}


def spectrum(times, signal, omega0, wmax=12.0, dw=0.02, window="sin2"):
    ws = np.arange(0.0, wmax + 1e-9, dw)
    vals = np.array([project_harmonic(times, signal, w * omega0, window) for w in ws])
    return ws, vals


def field_resources(geo, prot: DriveProtocol, eg_direct: float, n_sample: int = 20001) -> dict:
    """Pulse resources from A(t), E(t). No propagation."""
    times = np.linspace(0.0, prot.t_pulse, n_sample)
    a = kappa_shift(times, geo, prot, +1)
    ef = e_field(times, geo, prot, +1)
    fe = float(np.trapezoid(np.sum(ef**2, axis=-1), times))
    e_peak = float(np.max(np.abs(ef)))
    a_peak = float(np.max(np.abs(a)))
    dt = times[1] - times[0]
    spec = np.fft.rfft(ef[:, 0])
    freq = 2.0 * np.pi * np.fft.rfftfreq(len(times), d=dt)
    power = np.abs(spec) ** 2
    pos = freq > 1.0e-12
    tot = float(np.trapezoid(power[pos], freq[pos])) if np.any(pos) else 0.0
    above = pos & (freq >= eg_direct)
    w_above = float(np.trapezoid(power[above], freq[above]) / tot) if tot > 0.0 else 0.0
    return {
        "protocol_id": prot.protocol_id,
        "label": prot.label,
        "hbar_omega": prot.hbar_omega0,
        "n_cycles": prot.n_cycles,
        "A_star": prot.a0_dimless,
        "period_hbar_over_th": prot.period,
        "t_pulse_hbar_over_th": prot.t_pulse,
        "t_pulse_fs_if_th_1eV": prot.t_pulse * 0.6582119569,
        "Delta_k_inv_ang": prot.a0_dimless / geo.a_nn_mean_ang,
        "a_peak_inv_ang": a_peak,
        "eE_peak_over_hbar_inv_ang_th": e_peak,
        "E_peak_V_per_ang_if_th_1eV": e_peak,
        "E_star": e_peak * geo.a_nn_mean_ang / eg_direct if eg_direct else None,
        "F_E": fe,
        "E_g_direct_over_hbar_omega": eg_direct / prot.hbar_omega0,
        "m_min_center_frequency": int(np.ceil(eg_direct / prot.hbar_omega0 - 1e-12)),
        "E_spectrum_weight_above_direct_gap": w_above,
        "note_spectrum_weight": "relative |Ẽ(ω)|² weight for ω ≥ E_g^dir/ħ; not an absorption probability",
    }


def _save_run(fn: Path, r: dict, ident: dict) -> None:
    payload = {
        "times": r["times"],
        "J_sheet": r["J_sheet"],
        "J_dimless": r["J_dimless"],
        "A_over": r["A_over"],
        "E_over": r["E_over"],
        "diag_t": r["diag"]["t"],
        "diag_n_exc": r["diag"]["n_exc"],
        "diag_tr_rho": r["diag"]["tr_rho"],
        "diag_herm": r["diag"]["herm"],
        "diag_idem": r["diag"]["idem"],
        "wall_seconds": r["wall_seconds"],
        "nk": r["nk"],
        "nt_per_cycle": r["nt_per_cycle"],
        "dt": r["dt"],
        "grid_shift": r["grid_shift"],
        "delta_U_e": r["delta_U_e"],
        "U_e0": r["U_e0"],
        "U_eT": r["U_eT"],
        "work_integral": r["work_integral"],
        "n_exc_peak": r["n_exc_peak"],
        "kf": r["kf"],
        "w_k": r["w_k"],
    }
    for key in (
        "J_vv_sheet",
        "J_cc_sheet",
        "J_vc_sheet",
        "subspace_reconstruct_maxabs",
        "n_exc_k_peak",
        "n_exc_k_final",
        "k_orders",
        "k_harm_total",
        "k_harm_vv",
        "k_harm_cc",
        "k_harm_vc",
    ):
        if key in r:
            payload[key] = r[key]
    np.savez_compressed(fn, **payload)
    meta = {k: ident[k] for k in ident}
    fn.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_run(fn: Path) -> dict:
    d = np.load(fn, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    return out


def run_set(geo, p, prot, mu, nk, nt_per_cycle, *, signs=(+1, -1), neels=(+1, -1), tag="base", outdir: Path,
            verbose=True, grid_shift: float = 0.0, subspace: bool = False, k_orders=()):
    outdir.mkdir(parents=True, exist_ok=True)
    k_orders = tuple(int(x) for x in k_orders)
    runs = {}
    for neel in neels:
        for sign in signs:
            key = f"N{neel:+d}_A{sign:+d}"
            fn = outdir / f"run_{tag}_{key}.npz"
            ident = run_fingerprint(
                geo, p, prot, nk=nk, nt_per_cycle=nt_per_cycle, mu=mu, grid_shift=grid_shift,
                subspace=subspace, k_orders=k_orders, neel=neel, sign=sign,
            )
            meta_path = fn.with_suffix(".meta.json")
            if fn.exists() and meta_path.exists():
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
                if stored.get("identity_sha256") == ident["identity_sha256"]:
                    runs[key] = _load_run(fn)
                    runs[key]["neel"], runs[key]["field_sign"] = neel, sign
                    print(f"[{tag}] loaded {key} identity={ident['identity_sha256'][:12]}", flush=True)
                    continue
                print(f"[{tag}] identity mismatch for {key}; recomputing", flush=True)
            elif fn.exists() and not meta_path.exists():
                print(f"[{tag}] refusing cache without identity file: {fn.name}", flush=True)
            print(f"[{tag}] run {key} nk={nk} nt/cycle={nt_per_cycle} shift={grid_shift} subspace={subspace}", flush=True)
            r = propagate(
                geo, p, prot, nk=nk, nt_per_cycle=nt_per_cycle, neel=neel, sign=sign, mu=mu,
                verbose=verbose, grid_shift=grid_shift, subspace=subspace, k_orders=k_orders,
            )
            _save_run(fn, r, ident)
            runs[key] = _load_run(fn)
            runs[key]["neel"], runs[key]["field_sign"] = neel, sign
            print(f"[{tag}] done {key} in {float(r['wall_seconds']):.1f}s", flush=True)
    return runs


def _cplx_entry(v):
    return {
        "x": [float(v[0].real), float(v[0].imag)],
        "y": [float(v[1].real), float(v[1].imag)],
        "abs_x": float(abs(v[0])),
        "abs_y": float(abs(v[1])),
        "hbar_Omega_x_over_th": None,
    }


def analyse_set(runs: dict, prot: DriveProtocol, orders=range(1, 10)) -> dict:
    times = np.asarray(runs["N+1_A+1"]["times"], dtype=float)
    jp = np.asarray(runs["N+1_A+1"]["J_sheet"], dtype=float)
    jm = np.asarray(runs["N-1_A+1"]["J_sheet"], dtype=float)
    je = 0.5 * (jp + jm)
    jo = 0.5 * (jp - jm)
    out = {"orders": list(orders), "windows": {}, "protocol_id": prot.protocol_id}
    jmax = float(np.max(np.abs(jp)))
    out["J_max_sheet"] = jmax
    ident = {}
    for neel in (+1, -1):
        kp = f"N{neel:+d}_A+1"
        km = f"N{-neel:+d}_A-1"
        if kp in runs and km in runs:
            res = np.asarray(runs[kp]["J_sheet"]) + np.asarray(runs[km]["J_sheet"])
            ident[f"N{neel:+d}"] = {"max_abs": float(np.max(np.abs(res))), "relative_to_Jmax": float(np.max(np.abs(res)) / max(jmax, 1e-300))}
    out["field_reversal_identity"] = ident
    if "N+1_A-1" in runs and "N-1_A-1" in runs:
        je_m = 0.5 * (np.asarray(runs["N+1_A-1"]["J_sheet"]) + np.asarray(runs["N-1_A-1"]["J_sheet"]))
        jo_m = 0.5 * (np.asarray(runs["N+1_A-1"]["J_sheet"]) - np.asarray(runs["N-1_A-1"]["J_sheet"]))
        out["Je_odd_in_A_residual"] = float(np.max(np.abs(je + je_m)) / max(jmax, 1e-300))
        out["Jo_even_in_A_residual"] = float(np.max(np.abs(jo - jo_m)) / max(jmax, 1e-300))
    signals = {"J_plus": jp, "J_minus": jm, "J_e": je, "J_o": jo}
    if "J_vv_sheet" in runs["N+1_A+1"] and "J_vv_sheet" in runs["N-1_A+1"]:
        for blk in ("vv", "cc", "vc"):
            bp = np.asarray(runs["N+1_A+1"][f"J_{blk}_sheet"])
            bm = np.asarray(runs["N-1_A+1"][f"J_{blk}_sheet"])
            signals[f"J_e_{blk}"] = 0.5 * (bp + bm)
            signals[f"J_o_{blk}"] = 0.5 * (bp - bm)
        rec_p = np.max(np.abs(
            np.asarray(runs["N+1_A+1"]["J_vv_sheet"])
            + np.asarray(runs["N+1_A+1"]["J_cc_sheet"])
            + np.asarray(runs["N+1_A+1"]["J_vc_sheet"])
            - jp
        ))
        out["subspace_reconstruct_Jplus_maxabs"] = float(rec_p)
    for window in ("sin2", "boxcar"):
        tab = {}
        for name, sig in signals.items():
            h = harmonic_table(times, sig, prot.omega0, orders, window)
            tab[name] = {}
            for n, v in h.items():
                ent = _cplx_entry(v)
                ent["hbar_Omega_over_th"] = float(n * prot.hbar_omega0)
                tab[name][str(n)] = ent
            dc = project_harmonic(times, sig, 0.0, window)
            ent0 = _cplx_entry(dc)
            ent0["hbar_Omega_over_th"] = 0.0
            tab[name]["0"] = ent0
        if "J_e_vv" in signals:
            rec_h = 0.0
            for n in list(orders) + [0]:
                for who in ("J_e", "J_o"):
                    tot = complex(*tab[who][str(n)]["x"]) + 1j * 0
                    # compare complex 2-vector
                    vx = complex(*tab[who][str(n)]["x"])
                    vy = complex(*tab[who][str(n)]["y"])
                    sx = sum(complex(*tab[f"{who}_{b}"][str(n)]["x"]) for b in ("vv", "cc", "vc"))
                    sy = sum(complex(*tab[f"{who}_{b}"][str(n)]["y"]) for b in ("vv", "cc", "vc"))
                    rec_h = max(rec_h, abs(vx - sx), abs(vy - sy))
            tab["subspace_harmonic_reconstruct_maxabs"] = rec_h
        hp = harmonic_table(times, jp, prot.omega0, orders, window)
        hm = harmonic_table(times, jm, prot.omega0, orders, window)
        he = harmonic_table(times, je, prot.omega0, orders, window)
        lin = max(float(np.max(np.abs(0.5 * (hp[n] + hm[n]) - he[n]))) for n in orders)
        tab["linearity_residual"] = lin
        out["windows"][window] = tab
    for key, r in runs.items():
        nexc = np.asarray(r["diag_n_exc"], dtype=float)
        out.setdefault("diagnostics", {})[key] = {
            "n_exc_max": float(np.max(nexc)),
            "n_exc_final": float(nexc[-1]),
            "n_exc_peak_stored": float(r["n_exc_peak"]) if "n_exc_peak" in r else None,
            "tr_rho_dev_max": float(np.max(np.abs(np.asarray(r["diag_tr_rho"]) - N_OCC))),
            "herm_max": float(np.max(r["diag_herm"])),
            "idem_max": float(np.max(r["diag_idem"])),
            "J_equilibrium_t0_sheet": [float(x) for x in np.asarray(r["J_sheet"])[0]],
            "wall_seconds": float(r["wall_seconds"]),
            "delta_U_e": float(r["delta_U_e"]) if "delta_U_e" in r else None,
            "work_integral": float(r["work_integral"]) if "work_integral" in r else None,
            "energy_balance_abs": (
                abs(float(r["delta_U_e"]) - float(r["work_integral"]))
                if "delta_U_e" in r and "work_integral" in r
                else None
            ),
        }
    a = np.asarray(runs["N+1_A+1"]["A_over"])
    ef = np.asarray(runs["N+1_A+1"]["E_over"])
    out["pulse"] = {
        "a0_shift_inv_ang": float(np.max(np.abs(a))),
        "eE_peak_over_hbar_inv_ang_th": float(np.max(np.abs(ef))),
        "E_peak_V_per_ang_if_th_1eV": float(np.max(np.abs(ef))),
    }
    tail = times >= 0.9 * prot.t_pulse
    out["end_of_pulse"] = {}
    for name, sig in (("J_e", je), ("J_o", jo)):
        out["end_of_pulse"][name] = {
            "final_x": float(sig[-1, 0]),
            "final_y": float(sig[-1, 1]),
            "tail_mean_0p9_to_Tp_x": float(np.mean(sig[tail, 0])),
            "tail_mean_0p9_to_Tp_y": float(np.mean(sig[tail, 1])),
            "max_abs": float(np.max(np.abs(sig))),
            "interval": "[0.9 Tp, Tp]; not a post-pulse (t>Tp) record",
        }
    return out, (times, jp, jm, je, jo)


def low_frequency_summary(sets: dict, prot: DriveProtocol) -> dict:
    out = {}
    for key, s in sets.items():
        if key == "ref":
            continue
        times, jp, jm, je, jo = s["traces"]
        tail = times >= 0.9 * prot.t_pulse
        rec = {"nk": s["nk"], "ntpc": s["ntpc"]}
        for name, sig in (("J_e", je), ("J_o", jo)):
            dc_sin2 = project_harmonic(times, sig, 0.0, "sin2")
            dc_box = project_harmonic(times, sig, 0.0, "boxcar")
            rec[name] = {
                "dc_sin2_x": float(dc_sin2[0].real),
                "dc_sin2_y": float(dc_sin2[1].real),
                "dc_boxcar_x": float(dc_box[0].real),
                "dc_boxcar_y": float(dc_box[1].real),
                "final_x": float(sig[-1, 0]),
                "final_y": float(sig[-1, 1]),
                "tail_mean_x": float(np.mean(sig[tail, 0])),
                "tail_mean_y": float(np.mean(sig[tail, 1])),
                "max_abs": float(np.max(np.abs(sig))),
                "tail_interval": "[0.9 Tp, Tp] end-of-pulse, not t>Tp",
            }
        out[key] = rec
    return out


def compare_harmonics(tab_a: dict, tab_b: dict, orders, names=("J_e", "J_o")) -> dict:
    out = {}
    for name in names:
        out[name] = {}
        for n in orders:
            for comp in ("x", "y"):
                va = complex(*tab_a[name][str(n)][comp])
                vb = complex(*tab_b[name][str(n)][comp])
                out[name][f"{n}{comp}"] = {
                    "abs_a": abs(va),
                    "abs_b": abs(vb),
                    "abs_diff": abs(va - vb),
                    "rel_diff": abs(va - vb) / max(abs(vb), 1e-300),
                }
    return out


def target_quantities(tab_sin2: dict) -> dict:
    """The four P0 quantities requested for extra evidence."""
    def grab(name, n, comp):
        z = complex(*tab_sin2[name][str(n)][comp])
        return {"re": z.real, "im": z.imag, "abs": abs(z)}

    return {
        "Jo_y_0": grab("J_o", 0, "y"),
        "Jo_y_H2": grab("J_o", 2, "y"),
        "Jo_x_H2": grab("J_o", 2, "x"),
        "Je_x_H1": grab("J_e", 1, "x"),
    }


def make_figures(sets: dict, prot: DriveProtocol, outdir: Path, geo) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    times, jp, jm, je, jo = sets["ref"]["traces"]
    tt = times / prot.period
    fig, axs = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    a = np.asarray(sets["ref"]["runs"]["N+1_A+1"]["A_over"])[:, 0] * geo.a_nn_mean_ang
    axs[0].plot(tt, a, "k", lw=0.8)
    axs[0].set_ylabel(r"$e a_{NN} A_x/\hbar$")
    axs[1].plot(tt, je[:, 0], label=r"$J_{e,x}$")
    axs[1].plot(tt, je[:, 1], label=r"$J_{e,y}$", alpha=0.8)
    axs[1].set_ylabel(r"$J_e$ [$e t_h/(\hbar$Å)]")
    axs[1].legend(loc="upper right", fontsize=8)
    axs[2].plot(tt, jo[:, 0], label=r"$J_{o,x}$", color="C3")
    axs[2].plot(tt, jo[:, 1], label=r"$J_{o,y}$", color="C2", alpha=0.8)
    axs[2].set_ylabel(r"$J_o$ [$e t_h/(\hbar$Å)]")
    axs[2].set_xlabel(r"$t/T_0$")
    axs[2].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"M0 {R0.label} / {prot.label}: reference nk={sets['ref']['nk']}, nt/cycle={sets['ref']['ntpc']}")
    f1 = outdir / "m0_classical_traces.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    files.append(str(f1))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name, sig in ((axs[0], r"$J_e$", je), (axs[1], r"$J_o$", jo)):
        for window, ls in (("sin2", "-"), ("boxcar", ":")):
            ws, vals = spectrum(times, sig, prot.omega0, window=window)
            ax.semilogy(ws, np.abs(vals[:, 0]) ** 2 + 1e-40, ls, color="C0", label=f"x, {window}")
            ax.semilogy(ws, np.abs(vals[:, 1]) ** 2 + 1e-40, ls, color="C1", label=f"y, {window}")
        for n in range(1, 12):
            ax.axvline(n, color="gray", lw=0.4, alpha=0.5)
        ax.set_title(name + r"  $|\mathcal{J}(\omega)|^2$")
        ax.set_xlabel(r"$\omega/\omega_0$")
        ax.legend(fontsize=7)
    axs[0].set_ylabel(r"[$e t_h/(\hbar$Å)]$^2$")
    f2 = outdir / "m0_classical_spectra.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    files.append(str(f2))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    r = sets["ref"]["runs"]["N+1_A+1"]
    ax.plot(np.array(r["diag_t"]) / prot.period, np.array(r["diag_n_exc"]) / N_OCC, "o-", ms=3)
    ax.set_xlabel(r"$t/T_0$")
    ax.set_ylabel(r"$N_{exc}/N_{occ}$ (instantaneous valence of $H[\kappa(t)]$)")
    f3 = outdir / "m0_classical_excitation.png"
    fig.tight_layout()
    fig.savefig(f3, dpi=140)
    plt.close(fig)
    files.append(str(f3))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nk", type=int, default=48)
    ap.add_argument("--ntpc", type=int, default=400)
    ap.add_argument("--nk-fine", type=int, default=96)
    ap.add_argument("--ntpc-fine", type=int, default=800)
    ap.add_argument("--extra-nk", type=int, nargs="*", default=[], help="optional further k levels (ladder)")
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args(argv)
    outdir = OUT_ROOT / "classical"
    outdir.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    p = R0
    prot = P0Protocol()
    gap = json.loads((OUT_ROOT / "static/gap_report.json").read_text(encoding="utf-8"))
    if not gap["insulating_initial_state_supported"]:
        raise SystemExit("static gap check did not support the insulating initial state; stop (human review)")
    mu = gap["mu_half_filling"]
    orders = list(range(1, 10))
    sets = {}
    runs_base = run_set(geo, p, prot, mu, args.nk, args.ntpc, tag=f"nk{args.nk}_ntpc{args.ntpc}", outdir=outdir)
    ana_base, traces_base = analyse_set(runs_base, prot, orders)
    sets["base"] = {"nk": args.nk, "ntpc": args.ntpc, "runs": runs_base, "analysis": ana_base, "traces": traces_base}
    runs_dt = run_set(geo, p, prot, mu, args.nk, args.ntpc_fine, signs=(+1,), tag=f"nk{args.nk}_ntpc{args.ntpc_fine}", outdir=outdir)
    ana_dt, traces_dt = analyse_set(runs_dt, prot, orders)
    sets["dt_fine"] = {"nk": args.nk, "ntpc": args.ntpc_fine, "runs": runs_dt, "analysis": ana_dt, "traces": traces_dt}
    runs_k = run_set(geo, p, prot, mu, args.nk_fine, args.ntpc, signs=(+1,), tag=f"nk{args.nk_fine}_ntpc{args.ntpc}", outdir=outdir)
    ana_k, traces_k = analyse_set(runs_k, prot, orders)
    sets["k_fine"] = {"nk": args.nk_fine, "ntpc": args.ntpc, "runs": runs_k, "analysis": ana_k, "traces": traces_k}
    ladder = ["base", "k_fine"]
    for i, nk_extra in enumerate(args.extra_nk):
        key = f"k_fine{i + 2}"
        runs_x = run_set(geo, p, prot, mu, nk_extra, args.ntpc, signs=(+1,), tag=f"nk{nk_extra}_ntpc{args.ntpc}", outdir=outdir)
        ana_x, traces_x = analyse_set(runs_x, prot, orders)
        sets[key] = {"nk": nk_extra, "ntpc": args.ntpc, "runs": runs_x, "analysis": ana_x, "traces": traces_x}
        ladder.append(key)
    ref_key = ladder[-1]
    sets["ref"] = sets[ref_key]
    conv = {
        "dt": {w: compare_harmonics(sets["base"]["analysis"]["windows"][w], sets["dt_fine"]["analysis"]["windows"][w], orders) for w in ("sin2", "boxcar")},
    }
    for prev, nxt in zip(ladder[:-1], ladder[1:]):
        conv[f"k_{sets[prev]['nk']}_to_{sets[nxt]['nk']}"] = {
            w: compare_harmonics(sets[prev]["analysis"]["windows"][w], sets[nxt]["analysis"]["windows"][w], orders) for w in ("sin2", "boxcar")
        }
    last_k_key = f"k_{sets[ladder[-2]]['nk']}_to_{sets[ladder[-1]]['nk']}"
    floor = {}
    for name in ("J_e", "J_o"):
        floor[name] = {}
        for n in orders:
            for comp in ("x", "y"):
                key = f"{n}{comp}"
                d_dt = conv["dt"]["sin2"][name][key]["abs_diff"]
                d_k = conv[last_k_key]["sin2"][name][key]["abs_diff"]
                ref_val = complex(*sets["ref"]["analysis"]["windows"]["sin2"][name][str(n)][comp])
                fl = max(d_dt, d_k)
                floor[name][key] = {
                    "ref_abs": abs(ref_val),
                    "floor": fl,
                    "ratio_ref_over_floor": abs(ref_val) / max(fl, 1e-300),
                    "resolved_10x": bool(abs(ref_val) > 10.0 * fl),
                    "converged_1pct": bool(fl < 0.01 * abs(ref_val)),
                    "note": "empirical last-level change, not a strict error bound",
                }
    control = None
    if not args.skip_control:
        p0 = M0Params(t_h=p.t_h, lam_so=p.lam_so, delta=0.0, u=p.u, v=p.v, label="control_delta0")
        kf, kc, wgt = k_grid(geo, 24)
        e = np.linalg.eigvalsh(h8(geo, p0, kc, +1))
        mu0 = 0.5 * (e[:, N_OCC - 1].max() + e[:, N_OCC].min())
        gap0 = e[:, N_OCC].min() - e[:, N_OCC - 1].max()
        cdir = outdir / "control_delta0"
        try:
            rc = run_set(geo, p0, prot, mu0, 24, 100, signs=(+1,), tag="ctrl_nk24_ntpc100", outdir=cdir, verbose=False)
            jo_c = 0.5 * (rc["N+1_A+1"]["J_sheet"] - rc["N-1_A+1"]["J_sheet"])
            control = {"delta": 0.0, "nk": 24, "ntpc": 100, "indirect_gap_at_delta0": float(gap0),
                       "max_abs_Jo": float(np.max(np.abs(jo_c))), "max_abs_Jplus": float(np.max(np.abs(rc["N+1_A+1"]["J_sheet"])))}
        except RuntimeError as exc:
            control = {"delta": 0.0, "skipped": str(exc), "indirect_gap_at_delta0": float(gap0)}
    figs = make_figures(sets, prot, outdir, geo)
    a_peak = float(np.max(np.abs(sets["ref"]["runs"]["N+1_A+1"]["E_over"])))
    report = {
        "model_id": MODEL_ID,
        "stage": "B_classical_reference",
        "params": p.as_dict(),
        "protocol": asdict(prot),
        "protocol_derived": {
            "period_hbar_over_th": prot.period,
            "t_pulse_hbar_over_th": prot.t_pulse,
            "t_pulse_fs_if_th_1eV": prot.t_pulse * 0.6582119569,
            "a0_shift_inv_ang": prot.a0_dimless / geo.a_nn_mean_ang,
            "eE_peak_over_hbar_inv_ang_th": a_peak,
            "E_peak_V_per_ang_if_th_1eV": a_peak,
            "e_aNN_Epeak_over_hbar_omega0": a_peak * geo.a_nn_mean_ang / prot.omega0,
            "I_peak_W_cm2_if_th_1eV_for_orientation_only": 1.327e13 * a_peak**2,
            "E_g_indirect_over_hbar_omega0": gap["E_g_indirect"] / prot.hbar_omega0,
            "E_g_direct_over_hbar_omega0": gap["E_g_direct"] / prot.hbar_omega0,
            "mu": mu,
            "A_cell_ang2": geo.a_cell_ang2,
            "a_NN_ang": geo.a_nn_mean_ang,
            "current_unit": "e t_h/(ħ Å) sheet current; dimensionless = J / (e t_h a_NN/(ħ A_cell))",
        },
        "sets": {k: {"nk": v["nk"], "ntpc": v["ntpc"], "analysis": v["analysis"]} for k, v in sets.items() if k != "ref"},
        "reference_set": ref_key,
        "k_ladder": [(k, sets[k]["nk"]) for k in ladder],
        "convergence": conv,
        "low_frequency_Jo": low_frequency_summary(sets, prot),
        "error_floor_sin2": floor,
        "control_delta0": control,
        "figures": figs,
        "quantum_input": "not_run",
        "quantum_output": "not_run",
        "not_a_pass_criterion": "non-zero J_o or H2 was not required; finite-pulse parity is not a strict harmonic selection rule",
    }
    (outdir / "classical_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("protocol_derived", "reference_set", "control_delta0")}, indent=2))
    print("field reversal identity:", json.dumps(sets["base"]["analysis"]["field_reversal_identity"], indent=1))
    print("diag:", json.dumps(sets["ref"]["analysis"]["diagnostics"], indent=1))
    for name in ("J_e", "J_o"):
        print(name)
        for n in orders:
            for comp in ("x", "y"):
                fl = floor[name][f"{n}{comp}"]
                print(f"  H{n}{comp}: |J|={fl['ref_abs']:.3e} floor={fl['floor']:.3e} ratio={fl['ratio_ref_over_floor']:.1f} conv1%={fl['converged_1pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

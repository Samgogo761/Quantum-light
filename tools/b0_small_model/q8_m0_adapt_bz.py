#!/usr/bin/env python3
"""Bounded adaptive BZ quadrature for line-Q four-quantity qualification.

Production integrand uses per-k X and per-k D (not the Stage-1G one-hot + mean-D
proxy). Adaptive ranking never replaces the physical solver. The AI branch only
orders which tile to split next.

No 48/96/192 grids, no T2, no DC cut, no FNO-for-A2, no production V write.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q8_m0_cjj import propagate_cjj
from q8_m0_k1_kernel import trapz_weight
from q8_m0_response import contact_dmu_perk, prefix_state_dmu_perk
from q8_m0_runaway import a_priori_windows, amplitude_compare, propagate_seed_self
from q8_m0_s1_kernel import pack_complex
from q8_m0_volterra import h2_kernels

ADAPT_ID = "bounded_adapt_bz_quad_20260922"
BUDGET = {
    "max_k": 64,
    "max_rounds": 6,
    "start_level": 2,
    "splits_per_round": 2,
    "max_level": 5,
    "min_train_for_ai": 4,
    "no_48_96_192": True,
    "no_fno_for_A2": True,
}
THRESH = {
    "dA2_abs": 0.02,
    "dB2_abs": 0.02,
    "dA2_cross": 0.03,
    "output_weak_abs": 0.15,
    "kappa_ratio_ll1": 0.1,
    "odd_abs_report_only": True,
    "no_percent_pass_on_near_zero": True,
    "no_A2_lt_0.1_alone": True,
}


def _kf_key(kf) -> tuple[float, float]:
    return (round(float(kf[0]) % 1.0, 12), round(float(kf[1]) % 1.0, 12))


@dataclass
class Tile:
    level: int
    ix: int
    iy: int
    a2: complex | None = None
    b2: complex | None = None
    odd_abs: float | None = None
    inherited_disc: float = 0.0
    kappa_traj: np.ndarray | None = None
    di_traj: np.ndarray | None = None
    traj_dt: float = 1.0
    inherited_kappa_disc: float = 0.0

    @property
    def n_div(self) -> int:
        return 1 << self.level

    @property
    def width(self) -> float:
        return 1.0 / float(self.n_div)

    @property
    def area(self) -> float:
        w = self.width
        return w * w

    @property
    def center(self) -> np.ndarray:
        w = self.width
        return np.array([(self.ix + 0.5) * w, (self.iy + 0.5) * w], dtype=float)

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.level, self.ix, self.iy)

    def children(self) -> list["Tile"]:
        L = self.level + 1
        return [
            Tile(L, 2 * self.ix, 2 * self.iy),
            Tile(L, 2 * self.ix + 1, 2 * self.iy),
            Tile(L, 2 * self.ix, 2 * self.iy + 1),
            Tile(L, 2 * self.ix + 1, 2 * self.iy + 1),
        ]

    def contains(self, fx: float, fy: float) -> bool:
        fx = float(fx) % 1.0
        fy = float(fy) % 1.0
        x0 = self.ix * self.width
        y0 = self.iy * self.width
        return (x0 <= fx < x0 + self.width) and (y0 <= fy < y0 + self.width)


def start_leaves(level: int) -> list[Tile]:
    n = 1 << level
    return [Tile(level, ix, iy) for ix in range(n) for iy in range(n)]


def covering_leaf(leaves: list[Tile], fx: float, fy: float) -> Tile:
    fx = float(fx) % 1.0
    fy = float(fy) % 1.0
    for t in leaves:
        if t.contains(fx, fy):
            return t
    raise RuntimeError(f"no leaf covers ({fx}, {fy})")


def perk_born_ab(x, d_perk, F, ehat, k_a, dt, pref) -> tuple[np.ndarray, np.ndarray]:
    """True per-k Born a2(k), b2(k) from local X and local D.

    A2 = sum_k w_k a2(k), B2 = sum_k w_k b2(k).
    Contact uses D(k,t), not the BZ-mean D.
    """
    x = np.asarray(x)
    d_perk = np.asarray(d_perk, dtype=float)
    F = np.asarray(F)
    ehat = np.asarray(ehat, dtype=float)
    k_a = np.asarray(k_a)
    hr = np.real(F)[:, None] * ehat[None, :]
    hi = np.imag(F)[:, None] * ehat[None, :]
    ire = prefix_state_dmu_perk(x, hr, dt) + contact_dmu_perk(d_perk, hr)
    iim = prefix_state_dmu_perk(x, hi, dt) + contact_dmu_perk(d_perk, hi)
    wt = trapz_weight(k_a.shape[0], dt)
    i_f = ire + 1j * iim
    i_fs = ire - 1j * iim
    beta_f = np.einsum("t,ta,tka->k", wt, k_a, i_f)
    beta_fs = np.einsum("t,ta,tka->k", wt, k_a, i_fs)
    a2 = 1j * pref * beta_fs
    b2 = 1j * pref * beta_f
    return a2, b2


def net_gross_complex(values: np.ndarray, w: np.ndarray) -> dict:
    v = np.asarray(values)
    w = np.asarray(w, dtype=float)
    q_net = complex(np.dot(w, v))
    q_gross = float(np.dot(w, np.abs(v)))
    return {
        "net": pack_complex(q_net),
        "gross": q_gross,
        "cancel_ratio": float(abs(q_net) / max(q_gross, 1e-30)),
    }


def left_rect_integral(signal_tnk: np.ndarray, dt: float, gamma: float) -> np.ndarray:
    """Per-k left-rect integrator matching propagate_seed_self A_self.

    out[0]=0, out[n+1] = out[n] + γ dt signal[n]. Shape (nt, nk, 2).
    """
    sig = np.asarray(signal_tnk, dtype=float)
    out = np.zeros_like(sig)
    if sig.shape[0] < 2:
        return out
    out[1:] = np.cumsum(float(gamma) * float(dt) * sig[:-1], axis=0)
    return out


def traj_l2(arr, dt: float) -> float:
    a = np.asarray(arr, dtype=float)
    return float(np.sqrt(float(dt) * np.sum(a * a)))


def traj_rel_l2(a, b, ref, dt: float) -> float:
    return traj_l2(np.asarray(a) - np.asarray(b), dt) / max(traj_l2(ref, dt), 1e-30)


def multi_objective_target(*, dA2, dB2, d_odd, kappa_rel_l2, d_ratio) -> float:
    """Dimensionless mix so AI ranks A2, B2, cycle-1 odd, and κ(t) together."""
    return float(
        abs(dA2) / 0.02
        + abs(dB2) / 0.02
        + abs(d_odd) / 5.0e-6
        + abs(kappa_rel_l2) / 0.05
        + abs(d_ratio) / 0.05
    )


class PerkCache:
    """Independent-k CJJ cache. Weights are applied only at assemble time."""

    def __init__(self):
        self.store: dict[tuple[float, float], dict] = {}
        self.times = None
        self.dt = None
        self.wall_cjj = 0.0
        self.n_cjj_k = 0

    def fill(self, geo, p, prot, kfs, *, mu, neel, ntpc, sign=1, n_hold_cycles=0):
        missing = []
        seen = set()
        for kf in np.asarray(kfs, dtype=float):
            key = _kf_key(kf)
            if key in self.store or key in seen:
                continue
            seen.add(key)
            missing.append([key[0], key[1]])
        if not missing:
            return
        rec = propagate_cjj(
            geo, p, prot, nt_per_cycle=ntpc, neel=neel, sign=sign, mu=mu,
            verbose=False, kf_frac=np.asarray(missing, dtype=float),
            save_full_x=True, save_perk_D=True, n_hold_cycles=n_hold_cycles,
        )
        self.wall_cjj += float(rec["wall_seconds"])
        self.n_cjj_k += len(missing)
        self.times = rec["times"]
        self.dt = float(rec["dt"])
        for i, kf in enumerate(missing):
            key = _kf_key(kf)
            self.store[key] = {
                "X": rec["X_full"][:, i],
                "D": rec["perk_D_t"][:, i],
                "mu": rec["perk_mu_t"][:, i],
            }

    def assemble(self, kfs, weights) -> dict:
        kfs = np.mod(np.asarray(kfs, dtype=float), 1.0)
        w = np.asarray(weights, dtype=float)
        w = w / float(np.sum(w))
        nk = int(kfs.shape[0])
        sample = self.store[_kf_key(kfs[0])]
        nt = int(sample["X"].shape[0])
        x = np.zeros((nt, nk, 2, 4, 4), dtype=complex)
        d = np.zeros((nt, nk, 2, 2), dtype=float)
        mu_k = np.zeros((nt, nk, 2), dtype=float)
        for i, kf in enumerate(kfs):
            rec = self.store[_kf_key(kf)]
            x[:, i] = rec["X"]
            d[:, i] = rec["D"]
            mu_k[:, i] = rec["mu"]
        mean_d = np.einsum("k,tkab->tab", w, d)
        mu = np.einsum("k,tka->ta", w, mu_k)
        return {
            "X_full": x,
            "perk_D_t": d,
            "perk_mu_t": mu_k,
            "mean_D_t": mean_d,
            "mu_per_cell": mu,
            "w_k": w,
            "kf": kfs,
            "times": self.times,
            "dt": self.dt,
        }


def _neighbor_vals(tile: Tile, leaves: list[Tile], attr: str) -> list:
    n = tile.n_div
    out = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        cx = (tile.ix + dx + 0.5) / n
        cy = (tile.iy + dy + 0.5) / n
        nb = covering_leaf(leaves, cx, cy)
        val = getattr(nb, attr)
        if val is not None:
            out.append(val)
    return out


def tile_features(tile: Tile, leaves: list[Tile]) -> dict:
    a2 = 0j if tile.a2 is None else complex(tile.a2)
    b2 = 0j if tile.b2 is None else complex(tile.b2)
    na = _neighbor_vals(tile, leaves, "a2")
    nb = _neighbor_vals(tile, leaves, "b2")
    grad_a = max([abs(a2 - z) for z in na], default=0.0) / max(tile.width, 1e-12)
    grad_b = max([abs(b2 - z) for z in nb], default=0.0) / max(tile.width, 1e-12)
    flip = 0.0
    for z in na:
        if np.sign(a2.real) != np.sign(complex(z).real) and (abs(a2.real) + abs(complex(z).real) > 1e-8):
            flip = 1.0
            break
    var_a = float(np.var([complex(z).real for z in na] + [a2.real])) if na else 0.0
    var_b = float(np.var([complex(z).real for z in nb] + [b2.real])) if nb else 0.0
    odd = 0.0 if tile.odd_abs is None else float(tile.odd_abs)
    dt = float(tile.traj_dt) if tile.traj_dt else 1.0
    kap = tile.kappa_traj
    di = tile.di_traj
    kappa_l2 = traj_l2(kap, dt) if kap is not None else 0.0
    di_l2 = traj_l2(di, dt) if di is not None else 0.0
    neigh_k = 0.0
    neigh_d = 0.0
    n = tile.n_div
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        nb = covering_leaf(leaves, (tile.ix + dx + 0.5) / n, (tile.iy + dy + 0.5) / n)
        if kap is not None and nb.kappa_traj is not None:
            neigh_k = max(neigh_k, traj_l2(kap - nb.kappa_traj, dt))
        if di is not None and nb.di_traj is not None:
            neigh_d = max(neigh_d, traj_l2(di - nb.di_traj, dt))
    tmax_mod = 0.0
    if kap is not None:
        tmax_mod = float(np.max(np.linalg.norm(kap, axis=1)))
    return {
        "area": float(tile.area),
        "abs_a2": float(abs(a2)),
        "abs_b2": float(abs(b2)),
        "re_a2": float(a2.real),
        "im_a2": float(a2.imag),
        "grad_a2": float(grad_a),
        "grad_b2": float(grad_b),
        "sign_flip": float(flip),
        "var_a2": float(var_a),
        "var_b2": float(var_b),
        "inherited_disc": float(tile.inherited_disc),
        "odd_abs": odd,
        "level": float(tile.level),
        "gross_a2": float(abs(a2) * tile.area),
        "gross_b2": float(abs(b2) * tile.area),
        "kappa_l2": float(kappa_l2),
        "di_l2": float(di_l2),
        "neigh_kappa_l2": float(neigh_k),
        "neigh_di_l2": float(neigh_d),
        "inherited_kappa": float(tile.inherited_kappa_disc),
        "kappa_tmax_mod": tmax_mod,
    }


FEATURE_KEYS = (
    "area", "abs_a2", "abs_b2", "grad_a2", "grad_b2", "sign_flip",
    "var_a2", "inherited_disc", "odd_abs", "gross_a2", "level",
)
FEATURE_KEYS_TRAJ = (
    "area", "kappa_l2", "di_l2", "neigh_kappa_l2", "neigh_di_l2",
    "inherited_kappa", "abs_a2", "abs_b2", "odd_abs",
)


def feature_vector(feat: dict, keys=FEATURE_KEYS) -> np.ndarray:
    return np.array([1.0] + [float(feat[k]) for k in keys], dtype=float)


def traditional_score(feat: dict) -> float:
    return float(
        feat["area"] * (feat["grad_a2"] + feat["grad_b2"])
        + feat["area"] * (2.0 * feat["sign_flip"] * (feat["abs_a2"] + feat["abs_b2"]))
        + feat["area"] * (feat["var_a2"] + feat["var_b2"])
        + 0.25 * (feat["gross_a2"] + feat["gross_b2"])
        + 0.5 * feat["inherited_disc"]
        + feat["area"] * feat["odd_abs"]
    )


def traditional_score_traj(feat: dict) -> float:
    """Main baseline: refine tiles that move the κ_self(t) / δI(t) curves."""
    return float(
        feat["area"] * (feat["neigh_kappa_l2"] + feat["neigh_di_l2"])
        + 0.5 * feat["inherited_kappa"]
        + 0.2 * feat["area"] * feat["kappa_l2"]
        + 0.1 * traditional_score(feat)
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray | None:
    if x.shape[0] < 4:
        return None
    a = x.T @ x + lam * np.eye(x.shape[1])
    try:
        return np.linalg.solve(a, x.T @ y)
    except np.linalg.LinAlgError:
        return None


def pick_tiles(leaves: list[Tile], scores: np.ndarray, n_split: int, max_level: int) -> list[Tile]:
    ranked = []
    for t, s in zip(leaves, scores):
        if t.level >= max_level:
            continue
        ranked.append((-float(s), t.level, t.ix, t.iy, t))
    ranked.sort()
    picked = []
    seen = set()
    for rec in ranked:
        t = rec[4]
        if t.key in seen:
            continue
        seen.add(t.key)
        picked.append(t)
        if len(picked) >= n_split:
            break
    return picked


def split_leaves(leaves: list[Tile], parents: list[Tile]) -> list[Tile]:
    drop = {p.key for p in parents}
    kept = [t for t in leaves if t.key not in drop]
    for p in parents:
        kept.extend(p.children())
    return kept


class ActiveRanker:
    def __init__(self, *, score_fn=None, keys=None):
        self.score_fn = score_fn or traditional_score
        self.keys = keys or FEATURE_KEYS
        self.x = []
        self.y = []
        self.coef = None
        self.n_ai_picks = 0
        self.n_fallback = 0

    def observe(self, feat: dict, delta_abs: float) -> None:
        self.x.append(feature_vector(feat, self.keys))
        self.y.append(float(delta_abs))
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        self.coef = ridge_fit(x, y)

    def scores(self, feats: list[dict]) -> tuple[np.ndarray, str]:
        trad = np.array([self.score_fn(f) for f in feats], dtype=float)
        if self.coef is None or len(self.y) < BUDGET["min_train_for_ai"]:
            self.n_fallback += 1
            return trad, "traditional_cold_start"
        x = np.vstack([feature_vector(f, self.keys) for f in feats])
        pred = x @ self.coef
        pred = np.maximum(pred, 0.0)
        self.n_ai_picks += 1
        return pred, "ridge_active"


def assign_born(leaves: list[Tile], packed: dict, F, ehat, k_a, pref) -> tuple[np.ndarray, np.ndarray]:
    a2, b2 = perk_born_ab(
        packed["X_full"], packed["perk_D_t"], F, ehat, k_a, packed["dt"], pref,
    )
    for t, za, zb in zip(leaves, a2, b2):
        t.a2 = complex(za)
        t.b2 = complex(zb)
    return a2, b2


def leaf_arrays(leaves: list[Tile]) -> tuple[np.ndarray, np.ndarray]:
    kf = np.array([t.center for t in leaves], dtype=float)
    w = np.array([t.area for t in leaves], dtype=float)
    return kf, w


def four_quantity_record(
    *,
    a2,
    b2,
    w,
    odd,
    even,
    a_self,
    kappa_p1,
    i_perk_odd,
    times,
    period,
    geo,
    a_star,
    prev=None,
    split_delta_a2=0.0,
    split_delta_b2=0.0,
    kappa_k=None,
    dt=None,
    prev_kappa_t=None,
    split_delta_kappa_l2=0.0,
) -> dict:
    w = np.asarray(w, dtype=float)
    a2 = np.asarray(a2)
    b2 = np.asarray(b2)
    n1 = a_priori_windows(times, period)["cycle1"]
    sl = slice(0, n1 + 1)
    odd_w = np.asarray(odd)[sl]
    even_w = np.asarray(even)[sl]
    as_w = np.asarray(a_self)
    amp = amplitude_compare(as_w, kappa_p1, geo, a_star)
    amp_c1 = amplitude_compare(as_w[sl], kappa_p1, geo, a_star)
    qa = net_gross_complex(a2, w)
    qb = net_gross_complex(b2, w)
    q_net_odd = float(np.max(np.abs(odd_w)))
    if i_perk_odd is None:
        q_gross_odd = None
        cancel_odd = None
    else:
        odd_k = np.asarray(i_perk_odd)[sl]
        q_gross_odd = float(np.max(np.einsum("k,tka->ta", w, np.abs(odd_k))))
        cancel_odd = float(q_net_odd / max(q_gross_odd, 1e-30))
    rec = {
        "n_k": int(w.size),
        "A2": qa,
        "B2": qb,
        "I_odd_cycle1": {
            "net": q_net_odd,
            "gross": q_gross_odd,
            "cancel_ratio": cancel_odd,
            "maxabs_even": float(np.max(np.abs(even_w))),
            "even_over_odd": float(np.max(np.abs(even_w)) / max(q_net_odd, 1e-30)),
        },
        "kappa_self": {
            "net_maxmod": amp["kappa_self_peak_mod_ang_inv"],
            "ratio_full": amp["ratio_kappa_self_mod_over_p1"],
            "ratio_cycle1": amp_c1["ratio_kappa_self_mod_over_p1"],
            "gross": amp["kappa_self_peak_mod_ang_inv"],
            "traj_l2": traj_l2(as_w, float(dt)) if dt is not None else None,
            "recon_err": None,
            "net_traj_l2": None,
            "gross_traj_l2": None,
            "cancel_traj": None,
        },
        "delta": {
            "A2_split_sum_abs": float(abs(split_delta_a2)),
            "B2_split_sum_abs": float(abs(split_delta_b2)),
            "kappa_split_l2": float(abs(split_delta_kappa_l2)),
        },
    }
    if kappa_k is not None and dt is not None:
        kk = np.asarray(kappa_k, dtype=float)
        recon = np.einsum("k,tka->ta", w, kk)
        rec["kappa_self"]["recon_err"] = float(np.max(np.abs(recon - as_w)))
        rec["kappa_self"]["net_traj_l2"] = traj_l2(recon, float(dt))
        rec["kappa_self"]["gross_traj_l2"] = float(np.dot(
            w, np.array([traj_l2(kk[:, i], float(dt)) for i in range(kk.shape[1])])
        ))
        rec["kappa_self"]["cancel_traj"] = float(
            rec["kappa_self"]["net_traj_l2"]
            / max(rec["kappa_self"]["gross_traj_l2"], 1e-30)
        )
    if prev is not None:
        rec["delta"]["A2_successive_abs"] = float(abs(
            complex(qa["net"]["re"], qa["net"]["im"])
            - complex(prev["A2"]["net"]["re"], prev["A2"]["net"]["im"])
        ))
        rec["delta"]["B2_successive_abs"] = float(abs(
            complex(qb["net"]["re"], qb["net"]["im"])
            - complex(prev["B2"]["net"]["re"], prev["B2"]["net"]["im"])
        ))
        rec["delta"]["I_odd_successive_abs"] = float(abs(
            q_net_odd - prev["I_odd_cycle1"]["net"]
        ))
        rec["delta"]["kappa_maxmod_successive_abs"] = float(abs(
            amp["kappa_self_peak_mod_ang_inv"] - prev["kappa_self"]["net_maxmod"]
        ))
        rec["delta"]["kappa_ratio_successive_abs"] = float(abs(
            amp["ratio_kappa_self_mod_over_p1"] - prev["kappa_self"]["ratio_full"]
        ))
        if prev_kappa_t is not None and dt is not None:
            rec["delta"]["kappa_traj_rel_l2"] = traj_rel_l2(
                as_w, prev_kappa_t, kappa_p1, float(dt),
            )
        else:
            rec["delta"]["kappa_traj_rel_l2"] = None
    else:
        rec["delta"]["A2_successive_abs"] = None
        rec["delta"]["B2_successive_abs"] = None
        rec["delta"]["I_odd_successive_abs"] = None
        rec["delta"]["kappa_maxmod_successive_abs"] = None
        rec["delta"]["kappa_ratio_successive_abs"] = None
        rec["delta"]["kappa_traj_rel_l2"] = None
    return rec


def run_nl_block(geo, p, prot, packed, a_in, gamma, mu, neel, ntpc, cache=None):
    kf = packed["kf"]
    w = packed["w_k"]
    if cache is not None:
        key = tuple(sorted(
            (_kf_key(k)[0], _kf_key(k)[1], round(float(wi), 12))
            for k, wi in zip(kf, w)
        ))
        hit = cache.get(key)
        if hit is not None:
            return hit
    i_vv = packed["mu_per_cell"]
    zeros = np.zeros_like(a_in)
    kwargs = dict(
        geo=geo, p=p, prot=prot, kf=kf, mu=mu, neel=neel, nt_per_cycle=ntpc,
        verbose=False, w_k=w, save_perk_I=True,
    )
    i0 = propagate_seed_self(seed=zeros, i_ref=i_vv, gamma=0.0, **kwargs)
    i_ref = i0["I"]
    zrun = propagate_seed_self(seed=zeros, i_ref=i_ref, gamma=gamma, **kwargs)
    prun = propagate_seed_self(seed=a_in, i_ref=i_ref, gamma=gamma, **kwargs)
    mrun = propagate_seed_self(seed=-a_in, i_ref=i_ref, gamma=gamma, **kwargs)
    odd = 0.5 * (prun["I"] - mrun["I"])
    even = 0.5 * (prun["I"] + mrun["I"]) - zrun["I"]
    odd_k = 0.5 * (prun["I_perk"] - mrun["I_perk"])
    wall = (
        float(i0["wall_seconds"]) + float(zrun["wall_seconds"])
        + float(prun["wall_seconds"]) + float(mrun["wall_seconds"])
    )
    n1 = a_priori_windows(packed["times"], prot.period)["cycle1"]
    odd_abs_k = np.max(np.abs(odd_k[: n1 + 1]), axis=(0, 2))
    dI_k = prun["I_perk"] - i0["I_perk"]
    dt = float(i0["dt"])
    kappa_k = left_rect_integral(dI_k, dt, float(prun["gamma"]))
    recon = np.einsum("k,tka->ta", w, kappa_k)
    out = {
        "odd": odd,
        "even": even,
        "A_self": prun["A_self"],
        "kappa_p1": i0["kappa_p1"],
        "odd_k": odd_k,
        "odd_abs_k": odd_abs_k,
        "dI_k": dI_k,
        "kappa_k": kappa_k,
        "dt": dt,
        "recon_err": float(np.max(np.abs(recon - prun["A_self"]))),
        "zero_rel": float(
            np.max(np.abs(zrun["I"] - i_ref)) / max(float(np.max(np.abs(i_ref))), 1e-30)
        ),
        "plus_halted": bool(prun["halted"]),
        "wall_s": wall,
    }
    if cache is not None:
        cache[key] = out
    return out


def born_restore_verdict(hist_trad: list, hist_ai: list) -> dict:
    t = hist_trad[-1]
    a = hist_ai[-1]
    a2_t = complex(t["A2"]["net"]["re"], t["A2"]["net"]["im"])
    a2_a = complex(a["A2"]["net"]["re"], a["A2"]["net"]["im"])
    b2_t = complex(t["B2"]["net"]["re"], t["B2"]["net"]["im"])
    b2_a = complex(a["B2"]["net"]["re"], a["B2"]["net"]["im"])
    dA_t = t["delta"]["A2_successive_abs"]
    dA_a = a["delta"]["A2_successive_abs"]
    dB_t = t["delta"]["B2_successive_abs"]
    dB_a = a["delta"]["B2_successive_abs"]
    cross = float(abs(a2_t - a2_a))
    stable = bool(
        dA_t is not None and dA_a is not None
        and dA_t < THRESH["dA2_abs"] and dA_a < THRESH["dA2_abs"]
        and dB_t < THRESH["dB2_abs"] and dB_a < THRESH["dB2_abs"]
        and cross < THRESH["dA2_cross"]
    )
    weak = bool(
        abs(a2_t) < THRESH["output_weak_abs"]
        and abs(a2_a) < THRESH["output_weak_abs"]
        and abs(b2_t) < THRESH["output_weak_abs"]
        and abs(b2_a) < THRESH["output_weak_abs"]
    )
    kap_t = t["kappa_self"]["ratio_full"]
    kap_a = a["kappa_self"]["ratio_full"]
    time_ll1 = bool(kap_t < THRESH["kappa_ratio_ll1"] and kap_a < THRESH["kappa_ratio_ll1"])
    born = bool(stable and weak and time_ll1)
    if stable and weak and not time_ll1:
        node = "continuum_feedback_not_H2_mode_Born"
        gate = True
    elif stable and not weak:
        node = "output_mode_feedback_stable_non_negligible"
        gate = True
    else:
        node = None
        gate = False
    return {
        "born_restore_qualified": born,
        "output_mode_stable": stable,
        "output_mode_weak": weak,
        "time_domain_ll1": time_ll1,
        "A2_trad": pack_complex(a2_t),
        "A2_ai": pack_complex(a2_a),
        "B2_trad": pack_complex(b2_t),
        "B2_ai": pack_complex(b2_a),
        "A2_cross_abs": cross,
        "kappa_ratio_trad": kap_t,
        "kappa_ratio_ai": kap_a,
        "human_node": node,
        "human_gate": gate,
        "production_V": "Born_reference_diagnostic_not_qualified",
        "note": (
            "Mechanical AND of written layers. production_V stays reference "
            "even if born_restore_qualified; human reads the four quantities. "
            "Do not pass on |A2|<0.1 alone or on a percentage of a near-zero net."
        ),
    }


def assign_traj(leaves: list[Tile], kappa_k, dI_k, dt: float) -> None:
    kap = np.asarray(kappa_k, dtype=float)
    di = np.asarray(dI_k, dtype=float)
    for i, t in enumerate(leaves):
        t.kappa_traj = kap[:, i]
        t.di_traj = di[:, i]
        t.traj_dt = float(dt)


def traj_verdict(hist_trad: list, hist_ai: list) -> dict:
    """Qualification for time-trajectory-aware quadrature. Traditional is baseline."""
    t = hist_trad[-1]
    a = hist_ai[-1]
    a2_t = complex(t["A2"]["net"]["re"], t["A2"]["net"]["im"])
    a2_a = complex(a["A2"]["net"]["re"], a["A2"]["net"]["im"])
    b2_t = complex(t["B2"]["net"]["re"], t["B2"]["net"]["im"])
    b2_a = complex(a["B2"]["net"]["re"], a["B2"]["net"]["im"])
    dA_t = t["delta"]["A2_successive_abs"]
    dA_a = a["delta"]["A2_successive_abs"]
    dB_t = t["delta"]["B2_successive_abs"]
    dB_a = a["delta"]["B2_successive_abs"]
    cross = float(abs(a2_t - a2_a))
    a2_in_band = bool(
        0.13 <= abs(a2_t) <= 0.20 and 0.13 <= abs(a2_a) <= 0.20
        and dA_t is not None and dA_a is not None
        and dA_t < 0.02 and dA_a < 0.02
        and cross < 0.03
    )
    b2_stable = bool(
        abs(b2_t) < 0.05 and abs(b2_a) < 0.05
        and dB_t is not None and dB_a is not None
        and dB_t < 0.02 and dB_a < 0.02
    )
    r_t = t["kappa_self"]["ratio_full"]
    r_a = a["kappa_self"]["ratio_full"]
    dr_t = t["delta"]["kappa_ratio_successive_abs"]
    dr_a = a["delta"]["kappa_ratio_successive_abs"]
    l2_t = t["delta"].get("kappa_traj_rel_l2")
    l2_a = a["delta"].get("kappa_traj_rel_l2")
    kappa_agree = bool(
        abs(r_t - r_a) < 0.05
        and dr_t is not None and dr_a is not None
        and dr_t < 0.05 and dr_a < 0.05
        and l2_t is not None and l2_a is not None
        and l2_t < 0.05 and l2_a < 0.05
    )
    mean_r = 0.5 * (r_t + r_a)
    time_ll1 = bool(kappa_agree and mean_r < 0.10)
    node = None
    gate = False
    if kappa_agree:
        if 0.08 <= mean_r <= 0.20:
            node = "continuum_feedback_10_15pct_approx_layer"
            gate = True
        elif mean_r >= 0.25:
            node = "full_pulse_feedback_0p3_resummation"
            gate = True
    born = bool(a2_in_band and b2_stable and time_ll1)
    return {
        "born_restore_qualified": born,
        "probe6_A2_retired": bool(a2_in_band),
        "A2_in_0p16_pm_0p03": a2_in_band,
        "B2_stable_few_percent": b2_stable,
        "kappa_strategies_agree": kappa_agree,
        "kappa_ratio_trad": r_t,
        "kappa_ratio_ai": r_a,
        "kappa_ratio_mean": mean_r,
        "kappa_traj_rel_l2_trad": l2_t,
        "kappa_traj_rel_l2_ai": l2_a,
        "A2_trad": pack_complex(a2_t),
        "A2_ai": pack_complex(a2_a),
        "B2_trad": pack_complex(b2_t),
        "B2_ai": pack_complex(b2_a),
        "A2_cross_abs": cross,
        "time_domain_ll1": time_ll1,
        "human_node": node,
        "human_gate": gate,
        "production_V": "Born_reference_diagnostic_not_qualified",
        "traditional_is_baseline": True,
        "note": (
            "Human gate only if both independent trajectory-aware strategies "
            "agree on a non-negligible full-pulse kappa ratio. Jumping 0.1 vs 0.4 "
            "stays numerical. production_V stays reference."
        ),
    }


def k_points_to_abs_error(hist: list, key: str, target: float) -> int | None:
    for rec in hist:
        d = rec["delta"].get(key)
        if d is None:
            continue
        if d < target and rec is not hist[0]:
            return int(rec["n_k"])
    return None

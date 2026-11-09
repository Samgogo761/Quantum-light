#!/usr/bin/env python3
"""A1 classical-mixture photon-statistics benchmark (理论笔记 §13.2, 式 13.1).

Builds rho_clmix = sum_s w_s |gamma_s><gamma_s| with gamma_{n,s} = kappa_n J_{n,s}
from per-node classical trajectory amplitudes (HHG_nodes_modes.dat) and, if
per-node Jt_node_*.dat exist, two-time classical-mixture g1/g2(t,t').

Outputs are labeled CLASSICAL_MIXTURE_BENCHMARK. They are NOT quantum b_n
statistics: no squeezing, no entanglement, no sub-Poissonian claims are
possible from this construction (g2_classical >= 1 up to quadrature error).

The analytic g2=3 for beta ∝ x is the exact moment ratio of a real Gaussian
amplitude (Hermite-Gauss quadrature of N(0,1/2)). It matches the *bright*
squeezed-vacuum limit g2(0)->3 (n_bar >> 1), not the general single-mode
squeezed vacuum g2(0)=3+1/n_bar.

kappa-independent quantities: g2_nn, g2_nm, coherent_fraction, g1/g2(t,t').
kappa-dependent (photon number, Mandel Q): reported only as kappa^2-scaled
placeholders; absolute values need the input-output kernel K_{n,a} (B2).

Two independent gates before physics claims:
  1. quadrature: |J|^4 moments vs a higher GH order (GH5 vs GH7)
  2. Hilbert space: each node's J itself must be in a closed band window
GH5≈GH7 cannot substitute for a closed top-buffer / expanded-parent check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from combine_pmN_nodes_strict import (  # noqa: E402
    COL_ID,
    COL_IM_JX,
    COL_IM_JY,
    COL_INTENSITY,
    COL_ORDER,
    COL_PHASE,
    COL_RE_JX,
    COL_RE_JY,
    COL_WEIGHT,
    file_sha256,
    load_modes,
)
from finite_bandwidth_modes import build_modes  # noqa: E402

LABEL = "CLASSICAL_MIXTURE_BENCHMARK"
SCHEMA = "classical_mixture_stats_v3"
AU_TO_FS = 2.418884326505e-2
I2_ABS_FLOOR = 1.0e-300
DEFAULT_SIGNAL_REL_FLOOR = 1.0e-12
# GH5 descriptive intensity cuts (W/cm^2), not geometric GH shells.
DEFAULT_LOW_INTENSITY_MAX = 5.0e10
DEFAULT_HIGH_INTENSITY_TAIL_MIN = 1.0e12
DEFAULT_STATUS = "exploratory / quadrature not closed"

NOT_QUANTUM = [
    "quantum_squeezing",
    "quantum_entanglement",
    "sub_poissonian",
    "exact_photon_statistics",
    "matter_generated_quantum_correlation",
]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


# ----------------------------------------------------------------------
# single-channel mixture moments
# ----------------------------------------------------------------------

def mixture_moments(
    weights: np.ndarray,
    gamma: np.ndarray,
    *,
    i2_floor: float = I2_ABS_FLOOR,
) -> dict:
    """Normally-ordered classical-mixture moments of one complex channel.

    m1 = <gamma>, i2 = <|gamma|^2>, i4 = <|gamma|^4>,
    g2 = i4/i2^2 (kappa-independent),
    n_cl = i2 - |m1|^2   (trajectory covariance, classical analog of N_nn),
    m_cl = <gamma^2> - m1^2 (phase-sensitive classical analog of M_nn).
    """
    w = np.asarray(weights, dtype=float)
    g = np.asarray(gamma, dtype=complex)
    require(w.shape == g.shape, "weights/gamma shape mismatch")
    require(np.all(w >= 0.0), "negative weight")
    sw = float(w.sum())
    require(abs(sw - 1.0) < 5.0e-8, f"weights not normalized: {sw}")
    inten = np.abs(g) ** 2
    m1 = complex(np.sum(w * g))
    i2 = float(np.sum(w * inten))
    i4 = float(np.sum(w * inten ** 2))
    m2c = complex(np.sum(w * g * g))
    out = {
        "m1_re": m1.real,
        "m1_im": m1.imag,
        "I2": i2,
        "I4": i4,
        "n_cl": i2 - abs(m1) ** 2,
        "m_cl_re": (m2c - m1 * m1).real,
        "m_cl_im": (m2c - m1 * m1).imag,
    }
    if i2 > i2_floor:
        out["g2"] = i4 / (i2 * i2)
        out["coherent_fraction"] = abs(m1) ** 2 / i2
        # Mandel Q = kappa^2 * I2 * (g2 - 1); kappa unknown at A1.
        out["mandel_Q_over_kappa2"] = i2 * (out["g2"] - 1.0)
    else:
        out["g2"] = None
        out["coherent_fraction"] = None
        out["mandel_Q_over_kappa2"] = None
        out["unavailable_reason"] = "mean intensity below floor"
    return out


def cross_g2(
    weights: np.ndarray,
    gamma_n: np.ndarray,
    gamma_m: np.ndarray,
    *,
    i2_floor: float = I2_ABS_FLOOR,
) -> float | None:
    w = np.asarray(weights, dtype=float)
    a = np.abs(np.asarray(gamma_n, dtype=complex)) ** 2
    b = np.abs(np.asarray(gamma_m, dtype=complex)) ** 2
    i2a = float(np.sum(w * a))
    i2b = float(np.sum(w * b))
    if i2a <= i2_floor or i2b <= i2_floor:
        return None
    return float(np.sum(w * a * b)) / (i2a * i2b)


# ----------------------------------------------------------------------
# per-harmonic channels from HHG_nodes_modes.dat
# ----------------------------------------------------------------------

def channels_from_rows(rows: np.ndarray) -> dict[str, np.ndarray]:
    jx = rows[:, COL_RE_JX] + 1j * rows[:, COL_IM_JX]
    jy = rows[:, COL_RE_JY] + 1j * rows[:, COL_IM_JY]
    # circ_a matches finite_bandwidth_modes.project_current (jx + i jy)/sqrt2.
    # Mapping of circ_a/circ_b onto sigma+- depends on the e^{-i omega t}
    # convention and propagation direction; do not relabel without a test.
    return {
        "x": jx,
        "y": jy,
        "circ_a": (jx + 1j * jy) / math.sqrt(2.0),
        "circ_b": (jx - 1j * jy) / math.sqrt(2.0),
    }


def unique_nodes(data: np.ndarray) -> np.ndarray:
    """One row per node id, taken from the lowest stored harmonic."""
    orders = sorted({int(round(v)) for v in data[:, COL_ORDER]})
    rows = data[np.round(data[:, COL_ORDER]).astype(int) == orders[0]]
    return rows[np.argsort(rows[:, COL_ID])]


def classify_intensity(
    intensity: float,
    *,
    low_intensity_max: float = DEFAULT_LOW_INTENSITY_MAX,
    high_intensity_tail_min: float = DEFAULT_HIGH_INTENSITY_TAIL_MIN,
) -> str:
    if intensity < low_intensity_max:
        return "low_intensity"
    if intensity >= high_intensity_tail_min:
        return "high_intensity_tail"
    return "mid_intensity"


def intensity_inventory(
    data: np.ndarray,
    *,
    low_intensity_max: float = DEFAULT_LOW_INTENSITY_MAX,
    high_intensity_tail_min: float = DEFAULT_HIGH_INTENSITY_TAIL_MIN,
) -> dict:
    rows = unique_nodes(data)
    inv: dict[str, dict] = {
        tag: {"n_nodes": 0, "weight": 0.0, "ids": []}
        for tag in ("low_intensity", "mid_intensity", "high_intensity_tail")
    }
    for r in rows:
        tag = classify_intensity(
            float(r[COL_INTENSITY]),
            low_intensity_max=low_intensity_max,
            high_intensity_tail_min=high_intensity_tail_min,
        )
        inv[tag]["n_nodes"] += 1
        inv[tag]["weight"] += float(r[COL_WEIGHT])
        inv[tag]["ids"].append(int(round(r[COL_ID])))
    return inv


def subset_by_ids(data: np.ndarray, keep_ids: list[int]) -> np.ndarray:
    keep = set(int(i) for i in keep_ids)
    ids = np.round(data[:, COL_ID]).astype(int)
    sub = data[np.array([i in keep for i in ids])].copy()
    require(sub.size > 0, "empty node subset")
    rows0 = unique_nodes(sub)
    sw = float(rows0[:, COL_WEIGHT].sum())
    require(sw > 0.0, "subset weight sum is 0")
    sub[:, COL_WEIGHT] *= 1.0 / sw
    return sub


def compact_harmonic_g2(stats: dict) -> dict:
    out: dict[str, dict] = {}
    for h, entry in stats["harmonics"].items():
        rec = {
            "x_g2": entry["x"]["g2"],
            "y_g2": entry["y"]["g2"],
            "circ_a_g2": entry["circ_a"]["g2"],
            "circ_b_g2": entry["circ_b"]["g2"],
            "total_g2": entry["total_intensity"]["g2"],
            "x_coherent_fraction": entry["x"]["coherent_fraction"],
            "y_coherent_fraction": entry["y"]["coherent_fraction"],
        }
        out[h] = rec
    out["cross_g2_total_intensity"] = dict(stats["cross_g2_total_intensity"])
    return out


def compact_pm_g2(pm: dict) -> dict:
    out: dict[str, dict] = {}
    for h, rec in pm.items():
        out[h] = {
            "domain_even_total_g2": rec["domain_even"]["total_intensity_g2"],
            "domain_odd_total_g2": rec["domain_odd"]["total_intensity_g2"],
            "domain_even_I2_total": rec["domain_even"]["I2_total"],
            "domain_odd_I2_total": rec["domain_odd"]["I2_total"],
            "domain_even_x_g2": rec["domain_even"]["x"]["g2"],
            "domain_odd_x_g2": rec["domain_odd"]["x"]["g2"],
            "domain_even_y_g2": rec["domain_even"]["y"]["g2"],
            "domain_odd_y_g2": rec["domain_odd"]["y"]["g2"],
        }
    return out


def _global_signal_i2_scale(data: np.ndarray) -> float:
    jx = data[:, COL_RE_JX] + 1j * data[:, COL_IM_JX]
    jy = data[:, COL_RE_JY] + 1j * data[:, COL_IM_JY]
    return float(np.max(np.abs(jx) ** 2 + np.abs(jy) ** 2))


def stats_from_modes(
    data: np.ndarray,
    *,
    signal_rel_floor: float = DEFAULT_SIGNAL_REL_FLOOR,
) -> dict:
    orders = sorted({int(round(v)) for v in data[:, COL_ORDER]})
    per_harm: dict[str, dict] = {}
    gam_tot: dict[int, np.ndarray] = {}
    w_ref: np.ndarray | None = None
    signal_i2_scale = _global_signal_i2_scale(data)
    i2_floor = max(I2_ABS_FLOOR, signal_rel_floor * signal_i2_scale)
    for order in orders:
        rows = data[np.round(data[:, COL_ORDER]).astype(int) == order]
        rows = rows[np.argsort(rows[:, COL_ID])]
        w = rows[:, COL_WEIGHT].astype(float)
        if w_ref is None:
            w_ref = w
        ch = channels_from_rows(rows)
        entry = {name: mixture_moments(w, g, i2_floor=i2_floor) for name, g in ch.items()}
        # total-intensity statistics: I_s = |Jx|^2 + |Jy|^2 (production probe)
        i_tot = np.abs(ch["x"]) ** 2 + np.abs(ch["y"]) ** 2
        i2 = float(np.sum(w * i_tot))
        i4 = float(np.sum(w * i_tot ** 2))
        entry["total_intensity"] = {
            "I2": i2,
            "I4": i4,
            "g2": (i4 / (i2 * i2)) if i2 > i2_floor else None,
        }
        per_harm[str(order)] = entry
        gam_tot[order] = np.sqrt(i_tot).astype(complex)  # amplitude for cross-g2
    cross = {}
    for i, n in enumerate(orders):
        for m in orders[i + 1:]:
            val = cross_g2(w_ref, gam_tot[n], gam_tot[m], i2_floor=i2_floor)
            if val is not None:
                cross[f"{n}_{m}"] = val
    return {
        "harmonics": per_harm,
        "cross_g2_total_intensity": cross,
        "signal_floor": {
            "relative_to_global_max_node_intensity": signal_rel_floor,
            "global_max_node_intensity": signal_i2_scale,
            "I2_floor": i2_floor,
        },
    }


# ----------------------------------------------------------------------
# +N/-N pairing (Je/Jo channels)
# ----------------------------------------------------------------------

def pm_channel_stats(
    plus: np.ndarray,
    minus: np.ndarray,
    *,
    signal_rel_floor: float = DEFAULT_SIGNAL_REL_FLOOR,
) -> dict:
    """Algebraic domain-result even/odd amplitudes, not operator Je/Jo.

    An operator-level magnetic even/odd decomposition needs a common Hilbert
    basis and the C_ee/C_eo/C_oe/C_oo correlators; this function does not build
    that mapping.
    """
    def keyed(data: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        return {
            (int(round(r[COL_ID])), int(round(r[COL_ORDER]))): r
            for r in data
        }

    kp, km = keyed(plus), keyed(minus)
    require(set(kp) == set(km), "plus/minus (id, order) key sets differ")
    orders = sorted({o for _, o in kp})
    out: dict[str, dict] = {}
    signal_i2_scale = max(_global_signal_i2_scale(plus), _global_signal_i2_scale(minus))
    i2_floor = max(I2_ABS_FLOOR, signal_rel_floor * signal_i2_scale)
    for order in orders:
        ids = sorted(i for i, o in kp if o == order)
        w = np.array([kp[(i, order)][COL_WEIGHT] for i in ids])
        for i in ids:
            rp, rm = kp[(i, order)], km[(i, order)]
            for col, name in ((COL_WEIGHT, "weight"), (COL_INTENSITY, "I"), (COL_PHASE, "phi")):
                require(
                    math.isclose(rp[col], rm[col], rel_tol=1e-12, abs_tol=1e-12),
                    f"node {i} H{order}: plus/minus {name} mismatch",
                )
        jp = np.array([kp[(i, order)][COL_RE_JX] + 1j * kp[(i, order)][COL_IM_JX] for i in ids]), \
             np.array([kp[(i, order)][COL_RE_JY] + 1j * kp[(i, order)][COL_IM_JY] for i in ids])
        jm = np.array([km[(i, order)][COL_RE_JX] + 1j * km[(i, order)][COL_IM_JX] for i in ids]), \
             np.array([km[(i, order)][COL_RE_JY] + 1j * km[(i, order)][COL_IM_JY] for i in ids])
        je = ((jp[0] + jm[0]) / 2.0, (jp[1] + jm[1]) / 2.0)
        jo = ((jp[0] - jm[0]) / 2.0, (jp[1] - jm[1]) / 2.0)
        rec: dict[str, dict] = {}
        for tag, (x, y) in (("domain_even", je), ("domain_odd", jo)):
            i_tot = np.abs(x) ** 2 + np.abs(y) ** 2
            i2 = float(np.sum(w * i_tot))
            i4 = float(np.sum(w * i_tot ** 2))
            rec[tag] = {
                "x": mixture_moments(w, x, i2_floor=i2_floor),
                "y": mixture_moments(w, y, i2_floor=i2_floor),
                "total_intensity_g2": (i4 / (i2 * i2)) if i2 > i2_floor else None,
                "I2_total": i2,
            }
        rec["definition_warning"] = (
            "domain_even/domain_odd are half-sum/half-difference of final amplitudes; "
            "they are not common-basis current operators."
        )
        out[str(order)] = rec
    return out


# ----------------------------------------------------------------------
# two-time classical-mixture g1/g2 from per-node Jt files
# ----------------------------------------------------------------------

def load_jt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    require(data.ndim == 2 and data.shape[1] >= 4, f"invalid Jt columns: {path}")
    t_fs = data[:, 1]
    require(np.all(np.isfinite(data[:, :4])), f"non-finite Jt data: {path}")
    require(t_fs.size >= 2 and np.all(np.diff(t_fs) > 0.0), f"invalid Jt time axis: {path}")
    return t_fs, data[:, 2], data[:, 3]


def harmonic_envelope(
    j_t: np.ndarray, dt_au: float, omega0: float, order: int,
    half_width_over_omega0: float = 0.45,
) -> np.ndarray:
    """Positive-frequency filtered complex envelope of one harmonic.

    Full FFT (e^{-i omega t}), zero negative frequencies, apply the same
    raised-cosine F_n used by finite_bandwidth_modes, inverse FFT.
    """
    nt = j_t.size
    jw = np.fft.fft(j_t)
    omega = 2.0 * math.pi * np.fft.fftfreq(nt, d=dt_au)
    mask = np.zeros(nt)
    pos = omega >= 0.0
    modes = build_modes(omega[pos], omega0, harmonics=(order,),
                        half_width_over_omega0=half_width_over_omega0)
    mask[pos] = np.abs(modes[order]) / np.max(np.abs(modes[order]))
    return np.fft.ifft(jw * mask)


def two_time_stats(
    jt_files: dict[int, Path],
    weights_by_id: dict[int, float],
    omega0: float,
    orders: tuple[int, ...],
    decimate: int = 25,
    signal_rel_floor: float = DEFAULT_SIGNAL_REL_FLOOR,
) -> dict:
    ids = sorted(jt_files)
    require(len(ids) > 0, "no Jt files")
    w = np.array([weights_by_id[i] for i in ids], dtype=float)
    w = w / w.sum()
    out: dict[str, dict] = {}
    t_dec = None
    t_reference = None
    env: dict[int, list[np.ndarray]] = {n: [] for n in orders}
    loaded: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in ids:
        t_fs, jx, jy = load_jt(jt_files[i])
        if t_reference is None:
            t_reference = t_fs.copy()
        else:
            require(
                t_fs.shape == t_reference.shape
                and np.allclose(t_fs, t_reference, rtol=0.0, atol=1.0e-12),
                f"Jt time grid mismatch for node {i}",
            )
        loaded.append((i, t_fs, jx, jy))

    # Validate every node's time axis before any FFT work.  Besides producing a
    # clearer diagnostic, this prevents a malformed first trace from masking a
    # cross-node provenance error with an unrelated mode-construction failure.
    for _i, t_fs, jx, jy in loaded:
        dt_au = float(np.mean(np.diff(t_fs))) / AU_TO_FS
        for n in orders:
            ex = harmonic_envelope(jx, dt_au, omega0, n)
            ey = harmonic_envelope(jy, dt_au, omega0, n)
            env[n].append(np.stack([ex[::decimate], ey[::decimate]]))
        if t_dec is None:
            t_dec = t_fs[::decimate]
    for n in orders:
        stack = np.stack(env[n])              # (n_nodes, 2, n_dec)
        inten = np.sum(np.abs(stack) ** 2, axis=1)   # (n_nodes, n_dec)
        mean_i = np.einsum("s,st->t", w, inten)
        g1_num = np.einsum("s,sat,sau->tu", w, np.conjugate(stack), stack)
        g2_num = np.einsum("s,st,su->tu", w, inten, inten)
        denom = np.outer(mean_i, mean_i)
        i2_floor = max(I2_ABS_FLOOR, signal_rel_floor * float(np.max(mean_i)))
        signal_ok = mean_i > i2_floor
        ok = np.outer(signal_ok, signal_ok)
        g1 = np.zeros_like(g2_num)
        g2 = np.zeros_like(g2_num)
        g1[ok] = np.abs(g1_num[ok]) / np.sqrt(denom[ok])
        g2[ok] = g2_num[ok] / denom[ok]
        peak = int(np.argmax(mean_i))
        out[str(n)] = {
            "t_fs_decimated": [float(x) for x in t_dec],
            "mean_intensity_t": [float(x) for x in mean_i],
            "g2_tt_diag": [float(g2[k, k]) if ok[k, k] else None for k in range(len(t_dec))],
            "g2_peak_row": [float(g2[peak, k]) if ok[peak, k] else None for k in range(len(t_dec))],
            "abs_g1_peak_row": [float(g1[peak, k]) if ok[peak, k] else None for k in range(len(t_dec))],
            "peak_index": peak,
            "signal_floor": {
                "relative_to_peak_mean_intensity": signal_rel_floor,
                "I2_floor": i2_floor,
            },
        }
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _subset_report(
    plus: np.ndarray,
    minus: np.ndarray | None,
    keep_ids: list[int],
    *,
    signal_rel_floor: float = DEFAULT_SIGNAL_REL_FLOOR,
) -> dict:
    sub_p = subset_by_ids(plus, keep_ids)
    rec: dict = {
        "n_nodes": len(keep_ids),
        "ids": [int(i) for i in keep_ids],
        "single_domain": compact_harmonic_g2(
            stats_from_modes(sub_p, signal_rel_floor=signal_rel_floor)
        ),
    }
    if minus is not None:
        sub_m = subset_by_ids(minus, keep_ids)
        rec["domain_result_channels"] = compact_pm_g2(
            pm_channel_stats(sub_p, sub_m, signal_rel_floor=signal_rel_floor)
        )
    return rec


def intensity_sensitivity(
    plus: np.ndarray,
    minus: np.ndarray | None,
    *,
    low_intensity_max: float = DEFAULT_LOW_INTENSITY_MAX,
    high_intensity_tail_min: float = DEFAULT_HIGH_INTENSITY_TAIL_MIN,
    signal_rel_floor: float = DEFAULT_SIGNAL_REL_FLOOR,
) -> dict:
    inv = intensity_inventory(
        plus,
        low_intensity_max=low_intensity_max,
        high_intensity_tail_min=high_intensity_tail_min,
    )
    retained = inv["low_intensity"]["ids"] + inv["mid_intensity"]["ids"]
    brackets: dict = {
        "low_intensity_max_Wcm2": low_intensity_max,
        "high_intensity_tail_min_Wcm2": high_intensity_tail_min,
        "inventory": {
            tag: {
                "n_nodes": inv[tag]["n_nodes"],
                "weight": inv[tag]["weight"],
                "ids": inv[tag]["ids"],
            }
            for tag in ("low_intensity", "mid_intensity", "high_intensity_tail")
        },
        "note": (
            "drop_high_intensity_tail removes I above the declared threshold and "
            "renormalizes the remaining weights. It is a conditional sensitivity "
            "test, not a mathematical bracket, not a new quadrature, and not an "
            "original squeezed-vacuum distribution."
        ),
    }
    if retained:
        brackets["drop_high_intensity_tail"] = _subset_report(
            plus, minus, retained, signal_rel_floor=signal_rel_floor
        )
    for tag in ("high_intensity_tail", "low_intensity", "mid_intensity"):
        if inv[tag]["ids"]:
            brackets[f"{tag}_only"] = _subset_report(
                plus, minus, inv[tag]["ids"], signal_rel_floor=signal_rel_floor
            )
    return brackets


# Compatibility name for callers; schema-v3 output uses intensity_sensitivity.
ring_brackets = intensity_sensitivity


def _git_head(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(args: argparse.Namespace) -> dict:
    plus = load_modes(Path(args.modes))
    minus = load_modes(Path(args.modes_minus)) if args.modes_minus else None
    report: dict = {
        "schema": SCHEMA,
        "label": LABEL,
        "status": args.status,
        "not_a_physics_result": True,
        "not_quantum": NOT_QUANTUM,
        "gaussian_anchor_note": (
            "Analytic g2=3 for beta∝x is the real-Gaussian amplitude moment "
            "<x^4>/<x^2>^2. It agrees with the bright squeezed-vacuum limit "
            "g2(0)->3 (n_bar>>1), not the general squeezed-vacuum result "
            "g2(0)=3+1/n_bar."
        ),
        "kappa_note": (
            "g2, cross-g2, coherent_fraction, g1/g2(t,t') are kappa-independent; "
            "photon number and Mandel Q need the B2 input-output kernel K_{n,a}"
        ),
        "quadrature_gate_note": (
            "|J|^4 moments probe higher Q-moments than the mean spectrum. "
            "A GH-order gate (GH5 vs GH7 on w|J|^4) is independent of the "
            "Hilbert-space / top-buffer gate. Passing one does not pass the other."
        ),
        "b1_minus_a1_note": (
            "B1-A1 is a candidate excess over classical-mixture statistical "
            "transfer, not a quantitative isolation of matter-generated quantum "
            "correlations, until the same input state, the same K_{n,a}, and "
            "input-matter cross terms are closed."
        ),
        "domain_channel_note": (
            "domain_even/domain_odd are algebraic half-sum/half-difference of final "
            "amplitudes. They are not common-basis Je/Jo current operators and do not "
            "define C_ee/C_eo/C_oe/C_oo."
        ),
        "inputs": {"modes": str(args.modes), "modes_sha256": file_sha256(Path(args.modes))},
        "single_domain": stats_from_modes(plus, signal_rel_floor=args.signal_rel_floor),
        "intensity_sensitivity": intensity_sensitivity(
            plus,
            minus,
            low_intensity_max=args.low_intensity_max,
            high_intensity_tail_min=args.high_intensity_tail_min,
            signal_rel_floor=args.signal_rel_floor,
        ),
    }
    if minus is not None:
        report["inputs"]["modes_minus"] = str(args.modes_minus)
        report["inputs"]["modes_minus_sha256"] = file_sha256(Path(args.modes_minus))
        report["domain_result_channels"] = pm_channel_stats(
            plus, minus, signal_rel_floor=args.signal_rel_floor
        )
    if args.jt_dir:
        jt_dir = Path(args.jt_dir)
        files = {}
        for p in sorted(jt_dir.glob("Jt_node_*.dat")):
            files[int(p.stem.split("_")[-1])] = p
        require(len(files) > 0, f"no Jt_node_*.dat under {jt_dir}")
        require(args.omega0_au is not None, "--omega0-au required with --jt-dir")
        rows0 = plus[np.round(plus[:, COL_ORDER]).astype(int)
                     == int(round(plus[0, COL_ORDER]))]
        weights = {int(round(r[COL_ID])): float(r[COL_WEIGHT]) for r in rows0}
        missing = [i for i in files if i not in weights]
        require(not missing, f"Jt node ids not in manifest: {missing}")
        subset_w = sum(weights[i] for i in files)
        orders = tuple(int(o) for o in args.two_time_harmonics)
        report["two_time"] = {
            "note": (
                "classical-mixture g1/g2(t,t') over the available Jt subset; "
                f"subset quadrature weight = {subset_w:.6f} of 1. Renormalized within subset."
            ),
            "n_jt_nodes": len(files),
            "inputs": {
                str(i): {"path": str(files[i]), "sha256": file_sha256(files[i])}
                for i in sorted(files)
            },
            "stats": two_time_stats(files, weights, args.omega0_au, orders,
                                    decimate=args.decimate,
                                    signal_rel_floor=args.signal_rel_floor),
        }
    repo = Path(__file__).resolve().parents[2]
    report["provenance"] = {
        "generator": "tools/analysis/classical_mixture_stats.py",
        "generator_sha256": file_sha256(Path(__file__)),
        "git_head": _git_head(repo),
        "upstream_provenance": None,
    }
    if args.upstream_provenance:
        upstream = Path(args.upstream_provenance)
        report["provenance"]["upstream_provenance"] = {
            "path": str(upstream),
            "sha256": file_sha256(upstream),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", required=True, help="HHG_nodes_modes.dat (+N or single domain)")
    ap.add_argument("--modes-minus", default=None, help="optional -N HHG_nodes_modes.dat")
    ap.add_argument("--jt-dir", default=None, help="directory with Jt_node_*.dat")
    ap.add_argument("--omega0-au", type=float, default=None)
    ap.add_argument("--two-time-harmonics", type=int, nargs="+", default=[2, 5, 7, 9])
    ap.add_argument("--decimate", type=int, default=25)
    ap.add_argument("--signal-rel-floor", type=float, default=DEFAULT_SIGNAL_REL_FLOOR)
    ap.add_argument("--low-intensity-max", type=float, default=DEFAULT_LOW_INTENSITY_MAX)
    ap.add_argument(
        "--high-intensity-tail-min",
        type=float,
        default=DEFAULT_HIGH_INTENSITY_TAIL_MIN,
    )
    ap.add_argument(
        "--upstream-provenance",
        default=None,
        help="optional durable upstream provenance JSON recorded by path and SHA256",
    )
    ap.add_argument(
        "--status",
        default=DEFAULT_STATUS,
        help="provenance status string written into the JSON",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    require(args.signal_rel_floor >= 0.0, "--signal-rel-floor must be nonnegative")
    require(args.low_intensity_max < args.high_intensity_tail_min,
            "intensity cut ordering is invalid")
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    report["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

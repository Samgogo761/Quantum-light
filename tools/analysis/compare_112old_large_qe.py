#!/usr/bin/env python3
"""Compare 112_old vs 112_large vs QE: high-CB spectrum, chk P, TB r_mn.

This does not announce an HHG mechanism, does not promote 232 or I-d, and does
not treat H5/H7 peak heights as a PASS/FAIL gate.

QE high-CB truth is the nbnd=400 8-k NSCF (job 32911) plus the 576-k .eig from
112_large.  A1-H 6-k (nbnd=200) is not a 6.6–8 eV reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1o_berry_math import frobenius_block, interpolate_tb_hr, isolated_pair_abs
from audit_wannier_chk_subspace import index_checkpoint
from construct_p_conjugate_tb import read_tb_blocks
from count_outer_window_bands import parse_qe_nscf_kblocks, select_kblocks

SCHEMA = "compare_112old_large_qe_v1"
N_OCC = 84
E_FERMI_EV = 0.0843
ISO_GAP_EV = 0.050
N_K_COUNT = 8

WINDOWS = (
    ("frozen", -1.5, 0.8),
    ("mid", 0.8, 3.5),
    ("high_old_pool", 3.5, 6.5),
    ("high_new_pool", 6.5, 8.0),
    ("high_35_8", 3.5, 8.0),
    ("advertised", -6.0, 8.0),
)

COUNT_K = (
    {"label": "gamma", "crystal": (0.0, 0.0, 0.0), "optics": True},
    {"label": "eta", "crystal": (0.10, 0.55, 0.0), "optics": True},
    {"label": "H5", "crystal": (0.45, 0.55, 0.0), "optics": True},
    {"label": "H7", "crystal": (0.45, 0.05, 0.0), "optics": True},
    {"label": "mesh_X", "crystal": (0.50, 0.00, 0.0), "optics": False},
    {"label": "mesh_Y", "crystal": (0.00, 0.50, 0.0), "optics": False},
    {"label": "mesh_XY", "crystal": (0.50, 0.50, 0.0), "optics": False},
    {"label": "mesh_K", "crystal": (1.0 / 3.0, 1.0 / 3.0, 0.0), "optics": False},
)

NOTES = (
    "112_large constructed; this file only asks whether the enlarged Bloch "
    "pool made high-CB eigenvalues and optical coupling closer to QE.",
    "A1-H 6-k NSCF is nbnd=200 and is not used as 3.5–8 eV QE truth.",
    "Off-mesh QE Berry r above ~6.6 eV does not exist (no 400-band wfc at eta/H5/H7).",
    "chk projectability is on the 24x24 mesh only; hot k use the nearest mesh point.",
    "TB r_mn is Hamiltonian-gauge interpolation; band indices are not identified across models.",
    "H5/H7 HHG peaks are not a PASS/FAIL gate and are not in this JSON.",
    "Do not start 232_I-d from this file alone.",
)


def dumps_strict(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _spec_cell(row: dict, model: str, window: str) -> dict:
    spec = row[model]["spectrum"][window]
    h = spec["hausdorff"]
    qe_u = spec["qe_uncovered_vs_tb"]
    out = {
        "n_tb": spec["n_tb"],
        "n_qe": spec["n_qe"],
        "qe_to_tb_max_eV": None if h.get("empty") else h.get("qe_to_tb_max_eV"),
        "tb_to_qe_max_eV": None if h.get("empty") else h.get("tb_to_qe_max_eV"),
        "n_qe_farther_than_200meV": qe_u.get("n_src_farther_than_200meV"),
    }
    rmn = row[model].get("r_mn", {}).get("blocks", {}).get(window)
    if rmn and not rmn.get("empty"):
        out["r_occ_to_window"] = {
            "n_right": rmn["n_right"],
            "F_x": rmn["frobenius"]["F_x"],
            "F_y": rmn["frobenius"]["F_y"],
            "maxabs_x": rmn["frobenius"]["maxabs_x"],
            "maxabs_y": rmn["frobenius"]["maxabs_y"],
        }
    elif rmn:
        out["r_occ_to_window"] = {"n_right": rmn.get("n_right", 0), "empty": True}
    return out


def headline(per_k: list[dict], proj_old: dict, proj_large: dict) -> dict:
    optics = [row for row in per_k if row["label"] in ("gamma", "eta", "H5", "H7")]
    by_k = {}
    for row in optics:
        by_k[row["label"]] = {
            "high_old_pool": {
                "old": _spec_cell(row, "old", "high_old_pool"),
                "large": _spec_cell(row, "large", "high_old_pool"),
            },
            "high_new_pool": {
                "old": _spec_cell(row, "old", "high_new_pool"),
                "large": _spec_cell(row, "large", "high_new_pool"),
            },
            "frozen": {
                "old": _spec_cell(row, "old", "frozen"),
                "large": _spec_cell(row, "large", "frozen"),
            },
        }
    p_old = proj_old["mesh_windows"]
    p_large = proj_large["mesh_windows"]
    return {
        "question": "Did enlarging the Bloch pool fix the main high-CB / optics deficit?",
        "not_answered_here": True,
        "hot_k": by_k,
        "mesh_mean_P_all_qe": {
            "old_high_old_pool": p_old["high_old_pool"]["all_qe_in_window_over_mesh"],
            "large_high_old_pool": p_large["high_old_pool"]["all_qe_in_window_over_mesh"],
            "old_high_new_pool": p_old["high_new_pool"]["all_qe_in_window_over_mesh"],
            "large_high_new_pool": p_large["high_new_pool"]["all_qe_in_window_over_mesh"],
        },
        "old_eig_has_no_states_above_E200": True,
    }


def in_window(e: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (e >= lo) & (e <= hi)


def load_eig(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    nb = int(data[:, 0].max())
    nk = int(data[:, 1].max())
    eig = np.zeros((nk, nb))
    eig[data[:, 1].astype(int) - 1, data[:, 0].astype(int) - 1] = data[:, 2]
    return eig


def hausdorff(a: np.ndarray, b: np.ndarray) -> dict:
    da = np.min(np.abs(a[:, None] - b[None, :]), axis=1)
    db = np.min(np.abs(b[:, None] - a[None, :]), axis=1)
    return {
        "tb_to_qe_max_eV": float(da.max()),
        "qe_to_tb_max_eV": float(db.max()),
        "symmetric_eV": float(max(da.max(), db.max())),
        "tb_to_qe_p95_eV": float(np.quantile(da, 0.95)),
        "qe_to_tb_p95_eV": float(np.quantile(db, 0.95)),
        "n_tb_farther_than_20meV": int(np.sum(da > 0.02)),
        "n_qe_farther_than_20meV": int(np.sum(db > 0.02)),
        "argmax_tb_to_qe": {
            "tb_index0": int(np.argmax(da)),
            "tb_eV": float(a[np.argmax(da)]),
            "nearest_qe_eV": float(b[np.argmin(np.abs(b - a[np.argmax(da)]))]),
            "gap_eV": float(da.max()),
        },
    }


def unique_gamma_shift(tb: np.ndarray, qe: np.ndarray, lo: float, hi: float) -> dict:
    """Mean (TB − nearest QE) over TB levels in [lo, hi] at Gamma. Not a fit."""
    tb_w = np.sort(tb[in_window(tb, lo, hi)])
    qe_w = np.sort(qe[in_window(qe, lo, hi)])
    if tb_w.size == 0 or qe_w.size == 0:
        return {
            "shift_eV": 0.0,
            "n_tb": int(tb_w.size),
            "n_qe": int(qe_w.size),
            "applied": False,
            "reason": "empty_frozen_window",
        }
    nearest = np.array([qe_w[int(np.argmin(np.abs(qe_w - x)))] for x in tb_w])
    shift = float(np.mean(tb_w - nearest))
    return {
        "shift_eV": shift,
        "n_tb": int(tb_w.size),
        "n_qe": int(qe_w.size),
        "rms_to_nearest_eV": float(np.sqrt(np.mean((tb_w - nearest) ** 2))),
        "maxabs_to_nearest_eV": float(np.max(np.abs(tb_w - nearest))),
        "applied": True,
        "window": [lo, hi],
    }


def uncovered_counts(src: np.ndarray, ref: np.ndarray, thresholds_eV: tuple[float, ...]) -> dict:
    if src.size == 0 or ref.size == 0:
        return {
            "n_src": int(src.size),
            "n_ref": int(ref.size),
            **{f"n_src_farther_than_{int(1000 * t)}meV": int(src.size) for t in thresholds_eV},
        }
    dist = np.min(np.abs(src[:, None] - ref[None, :]), axis=1)
    out = {"n_src": int(src.size), "n_ref": int(ref.size)}
    for t in thresholds_eV:
        out[f"n_src_farther_than_{int(1000 * t)}meV"] = int(np.sum(dist > t))
    return out


def hausdorff_or_empty(a: np.ndarray, b: np.ndarray) -> dict:
    if a.size == 0 or b.size == 0:
        return {
            "empty": True,
            "n_tb": int(a.size),
            "n_qe": int(b.size),
        }
    h = hausdorff(a, b)
    h["empty"] = False
    h["n_tb"] = int(a.size)
    h["n_qe"] = int(b.size)
    return h


def window_spectrum(tb: np.ndarray, qe: np.ndarray, shift: float) -> dict:
    tb_s = tb + shift
    rows = {}
    for name, lo, hi in WINDOWS:
        tb_w = np.sort(tb_s[in_window(tb_s, lo, hi)])
        qe_w = np.sort(qe[in_window(qe, lo, hi)])
        rows[name] = {
            "lo_eV": lo,
            "hi_eV": hi,
            "n_tb": int(tb_w.size),
            "n_qe": int(qe_w.size),
            "hausdorff": hausdorff_or_empty(tb_w, qe_w),
            "qe_uncovered_vs_tb": uncovered_counts(qe_w, tb_w, (0.05, 0.10, 0.20)),
            "tb_uncovered_vs_qe": uncovered_counts(tb_w, qe_w, (0.05, 0.10, 0.20)),
        }
    return rows


def nearest_mesh_index(kpoints: np.ndarray, kfrac: tuple[float, float, float]) -> dict:
    k = np.array(kfrac, dtype=float)
    d = kpoints - k
    d -= np.round(d)
    dist = np.sqrt(np.sum(d * d, axis=1))
    ik = int(np.argmin(dist))
    return {
        "ik0": ik,
        "k_mesh": [float(x) for x in kpoints[ik]],
        "distance": float(dist[ik]),
    }


def occ_indices(evals: np.ndarray) -> dict:
    by_count = np.arange(min(N_OCC, evals.size), dtype=int)
    by_energy = np.flatnonzero(evals < E_FERMI_EV)
    return {
        "n_e_lt_fermi": int(by_energy.size),
        "same_as_nv84": bool(by_energy.size == N_OCC and np.array_equal(by_energy, by_count)),
        "nv84": by_count,
        "e_lt_fermi": by_energy,
    }


def r_report(evals: np.ndarray, r_cart: np.ndarray, energy_shift: float = 0.0) -> dict:
    occ = occ_indices(evals)
    left = occ["nv84"]
    e_win = evals + energy_shift
    out: dict = {
        "n_e_lt_fermi": occ["n_e_lt_fermi"],
        "nv84_matches_e_lt_fermi": occ["same_as_nv84"],
        "energy_shift_applied_to_windows_eV": float(energy_shift),
        "blocks": {},
    }
    for name, lo, hi in WINDOWS:
        if name not in ("high_old_pool", "high_new_pool", "high_35_8", "frozen"):
            continue
        right = np.flatnonzero(in_window(e_win, lo, hi))
        if right.size == 0 or left.size == 0:
            out["blocks"][name] = {"n_right": int(right.size), "empty": True}
            continue
        fb = frobenius_block(r_cart, left, right, zero_diag=False)
        pairs_x = isolated_pair_abs(r_cart[0], e_win, left, right, ISO_GAP_EV, max_pairs=8)
        pairs_y = isolated_pair_abs(r_cart[1], e_win, left, right, ISO_GAP_EV, max_pairs=8)
        out["blocks"][name] = {
            "empty": False,
            "n_right": int(right.size),
            "lo_eV": lo,
            "hi_eV": hi,
            "frobenius": {k: float(v) for k, v in fb.items()},
            "top_isolated_|rx|": pairs_x,
            "top_isolated_|ry|": pairs_y,
        }
    return out


def pdiag_at_k(index, udis_raw, ik: int) -> np.ndarray:
    u_stored = np.asarray(udis_raw[ik]).T
    nwin = int(index.ndimwin[ik])
    window_rows = np.flatnonzero(index.lwindow[ik])
    if window_rows.size != nwin:
        raise RuntimeError(f"window row count mismatch at k={ik + 1}")
    u = np.zeros_like(u_stored)
    u[window_rows, :] = u_stored[:nwin, :]
    return np.sum(np.abs(u) ** 2, axis=1).real


def open_udis(index):
    dtype = np.dtype(index.endian + "c16")
    return np.memmap(
        index.path,
        mode="r",
        dtype=dtype,
        offset=index.udis_offset,
        shape=(index.num_kpts, index.num_wann, index.num_bands),
        order="C",
    )


def p_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "n_lt_0p1": int(np.sum(values < 0.1)),
        "n_lt_0p5": int(np.sum(values < 0.5)),
        "n_ge_0p9": int(np.sum(values >= 0.9)),
    }


def projectability_windows(e: np.ndarray, pdiag: np.ndarray, lwindow: np.ndarray) -> dict:
    rows = {}
    for name, lo, hi in WINDOWS:
        mask = in_window(e, lo, hi)
        rows[name] = {
            "all_qe_in_window": p_stats(pdiag[mask]),
            "in_disentangle_window": p_stats(pdiag[mask & lwindow]),
            "n_qe_in_window_outside_lwindow": int(np.sum(mask & ~lwindow)),
        }
    return rows


def mesh_projectability(index, eig: np.ndarray, hot_map: dict[str, dict]) -> dict:
    if eig.shape[0] != index.num_kpts:
        raise RuntimeError(f"eig nk={eig.shape[0]} != chk nk={index.num_kpts}")
    if eig.shape[1] < index.num_bands:
        raise RuntimeError(f"eig nb={eig.shape[1]} < chk nb={index.num_bands}")
    eig = eig[:, : index.num_bands]
    udis = open_udis(index)
    acc: dict[str, list[np.ndarray]] = {name: [] for name, _lo, _hi in WINDOWS}
    acc_in: dict[str, list[np.ndarray]] = {name: [] for name, _lo, _hi in WINDOWS}
    per_hot = {}
    want = {item["ik0"] for item in hot_map.values()}
    for ik in range(index.num_kpts):
        pdiag = pdiag_at_k(index, udis, ik)
        e = eig[ik]
        lw = index.lwindow[ik]
        if ik in want or ik == 0:
            label_hits = [lab for lab, m in hot_map.items() if m["ik0"] == ik]
            per_hot[f"ik0_{ik}"] = {
                "ik0": ik,
                "labels": label_hits,
                "k_frac": [float(x) for x in index.kpoints[ik]],
                "windows": projectability_windows(e, pdiag, lw),
            }
        for name, lo, hi in WINDOWS:
            mask = in_window(e, lo, hi)
            if np.any(mask):
                acc[name].append(pdiag[mask])
            mask_in = mask & lw
            if np.any(mask_in):
                acc_in[name].append(pdiag[mask_in])
    del udis
    mesh = {}
    for name, _lo, _hi in WINDOWS:
        stacked = np.concatenate(acc[name]) if acc[name] else np.array([])
        stacked_in = np.concatenate(acc_in[name]) if acc_in[name] else np.array([])
        mesh[name] = {
            "all_qe_in_window_over_mesh": p_stats(stacked),
            "in_disentangle_window_over_mesh": p_stats(stacked_in),
        }
    return {
        "checkpoint": {
            "path_name": index.path.name,
            "num_bands": index.num_bands,
            "num_wann": index.num_wann,
            "num_kpts": index.num_kpts,
            "omega_invariant": index.omega_invariant,
            "checkpoint_stage": index.checkpoint,
        },
        "mesh_windows": mesh,
        "hot_or_gamma": per_hot,
        "nearest_mesh": hot_map,
    }


def tb_at_k(irvec, ndegen, h_r, r_r, crystal):
    k = np.array(crystal, dtype=float)
    if r_r is None:
        from compare_tb_eigenvalues import eigvals_at

        return eigvals_at(irvec, ndegen, h_r, tuple(crystal)), None
    evals, r_cart, _u = interpolate_tb_hr(irvec, ndegen, h_r, r_r, k)
    return evals, r_cart


def compare_one_k(
    spec: dict,
    qe: np.ndarray,
    tb_old,
    r_old,
    tb_large,
    r_large,
    shift_old: float,
    shift_large: float,
) -> dict:
    row = {
        "label": spec["label"],
        "crystal": list(spec["crystal"]),
        "qe_nbnd": int(qe.size),
        "qe_e_min": float(qe.min()),
        "qe_e_max": float(qe.max()),
        "old": {"spectrum": window_spectrum(tb_old, qe, shift_old)},
        "large": {"spectrum": window_spectrum(tb_large, qe, shift_large)},
    }
    if spec["optics"] and r_old is not None and r_large is not None:
        row["old"]["r_mn"] = r_report(tb_old, r_old, shift_old)
        row["large"]["r_mn"] = r_report(tb_large, r_large, shift_large)
    return row


def build_report(args: argparse.Namespace) -> dict:
    t0 = time.time()
    nscf_text = Path(args.nscf).read_text(encoding="utf-8", errors="replace")
    blocks = select_kblocks(parse_qe_nscf_kblocks(nscf_text), N_K_COUNT)
    if any(int(b["e_eV"].size) != 400 for b in blocks):
        raise RuntimeError("QE 8-k reference must be nbnd=400 (job 32911), not A1-H nbnd=200")

    print(f"Reading old TB {args.tb_old}", flush=True)
    ir_o, nd_o, h_o, r_o = read_tb_blocks(Path(args.tb_old), read_position=True)
    print(f"  old nwann={h_o.shape[1]} nrpts={h_o.shape[0]}", flush=True)
    print(f"Reading large TB {args.tb_large}", flush=True)
    ir_l, nd_l, h_l, r_l = read_tb_blocks(Path(args.tb_large), read_position=True)
    print(f"  large nwann={h_l.shape[1]} nrpts={h_l.shape[0]}", flush=True)
    if h_o.shape[1] != 112 or h_l.shape[1] != 112:
        raise RuntimeError(f"expected num_wann=112, got old={h_o.shape[1]} large={h_l.shape[1]}")

    evals_old = []
    evals_large = []
    rcart_old = []
    rcart_large = []
    for spec in COUNT_K:
        e_o, r_o_k = tb_at_k(ir_o, nd_o, h_o, r_o, spec["crystal"])
        e_l, r_l_k = tb_at_k(ir_l, nd_l, h_l, r_l, spec["crystal"])
        evals_old.append(e_o)
        evals_large.append(e_l)
        rcart_old.append(r_o_k)
        rcart_large.append(r_l_k)
        print(f"  interpolated {spec['label']}", flush=True)

    qe_gamma = blocks[0]["e_eV"]
    shift_old = unique_gamma_shift(evals_old[0], qe_gamma, -1.5, 0.8)
    shift_large = unique_gamma_shift(evals_large[0], qe_gamma, -1.5, 0.8)

    per_k = []
    for i, spec in enumerate(COUNT_K):
        per_k.append(
            compare_one_k(
                spec,
                blocks[i]["e_eV"],
                evals_old[i],
                rcart_old[i],
                evals_large[i],
                rcart_large[i],
                shift_old["shift_eV"] if shift_old["applied"] else 0.0,
                shift_large["shift_eV"] if shift_large["applied"] else 0.0,
            )
        )

    payload = {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_COMPARISON_NO_PASS_FAIL",
        "notes": list(NOTES),
        "timing_s": None,
        "gamma_shift": {"old": shift_old, "large": shift_large},
        "tb": {
            "old": {"nwann": int(h_o.shape[1]), "nrpts": int(h_o.shape[0]), "path": str(args.tb_old)},
            "large": {"nwann": int(h_l.shape[1]), "nrpts": int(h_l.shape[0]), "path": str(args.tb_large)},
        },
        "qe_8k": {
            "nscf": str(args.nscf),
            "nbnd": 400,
            "n_k": N_K_COUNT,
            "source": "job_32911_nbnd400_not_A1H",
        },
        "per_k": per_k,
    }

    del h_o, r_o, h_l, r_l, ir_o, ir_l, nd_o, nd_l

    print("Indexing checkpoints", flush=True)
    idx_old = index_checkpoint(Path(args.chk_old))
    idx_large = index_checkpoint(Path(args.chk_large))
    if idx_old.num_bands != 200:
        raise RuntimeError(f"old chk num_bands={idx_old.num_bands}, expected 200")
    if idx_large.num_bands != 400:
        raise RuntimeError(f"large chk num_bands={idx_large.num_bands}, expected 400")
    if idx_old.num_wann != 112 or idx_large.num_wann != 112:
        raise RuntimeError("both checkpoints must have num_wann=112")
    eig_old = load_eig(Path(args.eig_old))
    eig_large = load_eig(Path(args.eig_large))
    hot_old = {s["label"]: nearest_mesh_index(idx_old.kpoints, s["crystal"]) for s in COUNT_K[:4]}
    hot_large = {s["label"]: nearest_mesh_index(idx_large.kpoints, s["crystal"]) for s in COUNT_K[:4]}
    print("Mesh projectability old", flush=True)
    payload["projectability_old"] = mesh_projectability(idx_old, eig_old, hot_old)
    print("Mesh projectability large", flush=True)
    payload["projectability_large"] = mesh_projectability(idx_large, eig_large, hot_large)

    g_mesh = {
        "old_eig_gamma_E200": float(eig_old[0, 199]) if eig_old.shape[1] >= 200 else None,
        "large_eig_gamma_E200": float(eig_large[0, 199]) if eig_large.shape[1] >= 200 else None,
        "nscf8_gamma_E200": float(qe_gamma[199]),
        "nscf8_vs_large_eig_gamma_maxabs_eV": float(np.max(np.abs(qe_gamma[: min(400, eig_large.shape[1])] - eig_large[0, : min(400, eig_large.shape[1])]))),
    }
    payload["gamma_qe_consistency"] = g_mesh
    payload["headline"] = headline(per_k, payload["projectability_old"], payload["projectability_large"])
    payload["timing_s"] = float(time.time() - t0)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nscf", type=Path, required=True)
    ap.add_argument("--tb-old", type=Path, required=True)
    ap.add_argument("--tb-large", type=Path, required=True)
    ap.add_argument("--chk-old", type=Path, required=True)
    ap.add_argument("--chk-large", type=Path, required=True)
    ap.add_argument("--eig-old", type=Path, required=True)
    ap.add_argument("--eig-large", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_strict(payload), encoding="utf-8")
    print(f"wrote {args.out} in {payload['timing_s']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

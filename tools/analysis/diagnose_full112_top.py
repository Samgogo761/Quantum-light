#!/usr/bin/env python3
"""Authoritative full112 top-occupation diagnosis generator.

Does NOT generate a >112 parent or submit SBE. Rebuilds
docs/08_QE与Wannier/full112_top_occupation_diagnosis_20260821.json
with schema full112_top_diagnosis_v2 (strict JSON, no NaN).

Labels: docs/08_QE与Wannier/full112_顶部占据原因诊断_20260821.md
Trial-A leftover is a PROXY, not U_dis. P_tot is not a probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "full112_top_diagnosis_v2"
QL = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(r"C:\Users\26507\Documents\量子光研究\01_输入与模型")
OUT_DEFAULT = QL / "docs" / "08_QE与Wannier" / "full112_top_occupation_diagnosis_20260821.json"

DIS_WIN = (-6.0, 8.0)
QE_FERMI_OVERLAY = 0.0810
N_WAN = 112
N_CRD = 40
N_OCC = 84
NK_SBE = 400
EDGE_LO, EDGE_HI = 105, 112

VERDICT = {
    "SELECTED_CR_D_I_P_PROJECTOR_INVENTORY": "COMPLETE",
    "TOP8_BAND_ENERGY_INTERPOLATION": "SUPPORTED",
    "FULL112_TOP_BUFFER": "FAIL",
    "WANNIER_OPERATOR_QUALITY": "UNTESTED",
    "MISSING_HIGH_ENERGY_SPACE": "LEADING_HYPOTHESIS",
    "IMPACT_ON_HHG": "UNRESOLVED",
    "SCIENTIFIC_NARRATIVE": "ACCEPTED_WITH_CAVEATS",
    "MACHINE_EVIDENCE": "SCHEMA_V2",
    "NEW_QE_SBE_RUN": "NOT_REQUIRED",
}


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not math.isfinite(x):
            raise ValueError(f"non-finite float {x}")
        return x
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"not jsonable: {type(obj)}")


def dumps_strict(obj: dict) -> str:
    return json.dumps(to_jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_gnu(path: Path):
    bands, k, e = [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            if k:
                bands.append((np.asarray(k, float), np.asarray(e, float)))
                k, e = [], []
            continue
        a, b = map(float, s.split()[:2])
        k.append(a)
        e.append(b)
    if k:
        bands.append((np.asarray(k, float), np.asarray(e, float)))
    return bands[0][0], np.vstack([ee for _, ee in bands])


def load_eig(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    nb = int(data[:, 0].max())
    nk = int(data[:, 1].max())
    eig = np.zeros((nk, nb))
    eig[data[:, 1].astype(int) - 1, data[:, 0].astype(int) - 1] = data[:, 2]
    return eig


def stream_amn(path: Path, nk: int, nb: int, nw: int = N_WAN):
    p_d = np.zeros((nk, nb))
    p_p = np.zeros((nk, nb))
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        hdr = f.readline().split()
        require(int(hdr[0]) == nb and int(hdr[1]) == nk and int(hdr[2]) == nw, "amn header")
        for line in f:
            parts = line.split()
            n = int(parts[0]) - 1
            m = int(parts[1]) - 1
            k = int(parts[2]) - 1
            mag2 = float(parts[3]) ** 2 + float(parts[4]) ** 2
            if m < N_CRD:
                p_d[k, n] += mag2
            else:
                p_p[k, n] += mag2
    return p_d, p_p, p_d + p_p


def window_stats(eig, p_d, p_p, p_tot, lo, hi):
    mask = (eig >= lo) & (eig < hi)
    if not mask.any():
        return {"n_states": 0}
    pt, pd, pp = p_tot[mask], p_d[mask], p_p[mask]
    return {
        "n_states": int(mask.sum()),
        "n_per_k_mean": float(mask.sum(axis=1).mean()),
        "n_per_k_min": int(mask.sum(axis=1).min()),
        "n_per_k_max": int(mask.sum(axis=1).max()),
        "P_tot_mean": float(pt.mean()),
        "P_tot_median": float(np.median(pt)),
        "P_d_mean": float(pd.mean()),
        "P_p_mean": float(pp.mean()),
        "frac_P_tot_lt_0.3": float((pt < 0.3).mean()),
        "frac_P_tot_lt_0.5": float((pt < 0.5).mean()),
        "frac_P_tot_gt_0.8": float((pt > 0.8).mean()),
        "note": "P_tot is trial-overlap sum, not a normalized orbital percentage",
    }


def nearest_match_errors(e_w_sel: np.ndarray, e_qe: np.ndarray) -> np.ndarray:
    """e_w_sel (nband, nk), e_qe (nqe, nk) -> abs error (nband, nk)."""
    return np.min(np.abs(e_qe[:, None, :] - e_w_sel[None, :, :]), axis=0)


def ordered_unique_errors(e_w_sel: np.ndarray, e_qe: np.ndarray) -> np.ndarray:
    """一一有序能量匹配: at each k, take the 8 QE eigenvalues nearest the
    Wannier 105-112 cluster center, sort both sets, and pair in order.

    Unrestricted nearest-neighbor can map two Wannier bands onto one QE band.
    Greedy unique assignment across *all* QE bands is too harsh: the top-8
    Wannier shell is only ~0.33 eV thick and would be forced onto the next
    QE manifold ~0.6 eV above. Cluster pairing stays inside the interpolated
    shell (independent audit: MAE ~15 meV, MAX ~40 meV).
    """
    nband, nk = e_w_sel.shape
    out = np.empty((nband, nk))
    for ik in range(nk):
        w = e_w_sel[:, ik]
        q = e_qe[:, ik]
        mid = 0.5 * (w.min() + w.max())
        idx = np.argsort(np.abs(q - mid))[:nband]
        out[:, ik] = np.abs(np.sort(w) - np.sort(q[idx]))
    return out


def summarize_err(err: np.ndarray) -> dict:
    return {
        "MAE_eV": float(err.mean()),
        "RMS_eV": float(np.sqrt((err ** 2).mean())),
        "MAX_eV": float(err.max()),
    }


def sbe_bands_at(path: Path, nkx: int, nky: int, f1: float, f2: float) -> dict:
    """Read TB eigenvalues at crystal momentum (f1, f2) on an nkx x nky mesh."""
    ikx = int(round(f1 * nkx)) + 1
    iky = int(round(f2 * nky)) + 1
    e = np.full(N_WAN, np.nan)
    g84 = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            ix, iy, n = int(p[0]), int(p[1]), int(p[2])
            val = float(p[3])
            if ix == 1 and iy == 1 and n == N_OCC:
                g84 = val
            if ix == ikx and iy == iky:
                e[n - 1] = val
    if not np.isfinite(e).all() or g84 is None:
        return {
            "values": None,
            "unavailable_reason": f"incomplete bands.dat read at ikx={ikx} iky={iky}",
        }
    return {
        "mesh": [nkx, nky],
        "ikx": ikx,
        "iky": iky,
        "frac_k": [f1, f2],
        "Gamma_band84_eV": float(g84),
        "hot_105_112_raw_eV": [float(x) for x in e[104:112]],
        "hot_105_112_vs_Gamma_b84_eV": [float(x - g84) for x in e[104:112]],
    }


def occ_from_audit(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    def walk(obj):
        if isinstance(obj, dict):
            if "edge_peak" in obj and "I_Wcm2" in obj:
                yield obj
            else:
                for v in obj.values():
                    yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)

    seen = set()
    for c in walk(payload):
        d = c.get("dir", "")
        if d in seen:
            continue
        seen.add(d)
        gem = c.get("global_edge_max")
        if isinstance(gem, dict):
            edge_max = gem.get("occ")
            gem_band = gem.get("band")
            gem_t = gem.get("t_fs")
        else:
            edge_max, gem_band, gem_t = gem, None, None
        ep = c.get("edge_peak") or {}
        p_raw = ep.get("edge_pos", c.get("edge_pop_final"))
        if p_raw is None:
            continue
        rows.append(
            {
                "dir": d,
                "I_Wcm2": c.get("I_Wcm2"),
                "phi_cep_deg": c.get("phi_cep_deg"),
                "edge_max": edge_max,
                "edge_max_band": gem_band,
                "edge_max_t_fs": gem_t,
                "P_edge_raw": p_raw,
                "P_edge_per_cell": p_raw / NK_SBE,
                "mean_f_top8": p_raw / (NK_SBE * 8),
                "edge_over_exc_at_peak": ep.get("edge_over_exc"),
                "max_edge_over_exc": c.get("max_edge_over_exc"),
                "ratio_112_over_105": c.get("ratio_112_over_105_pos_sum_at_peak"),
                "note": "cross-config magnitude example; not a controlled I-scan",
            }
        )
    rows.sort(key=lambda r: (r["I_Wcm2"] or 0, r["dir"]))
    return rows


def build_report(data_root: Path, repo: Path) -> dict:
    wan = data_root / "wannier"
    qe_gnu = data_root / "qe_bands" / "CrI3_bands_nbnd200.dat.gnu"
    eig_path = wan / "CrI3.eig"
    amn_path = wan / "CrI3.amn"
    wband = wan / "CrI3_band.dat"
    k40_bands = (
        data_root
        / "CrI3_TB-N"
        / "output_afm_pconj_fullvalence_T2_0p5cycle"
        / "lgcov_k40_nb112"
        / "bands.dat"
    )
    occ_files = {
        "29458": repo / "docs" / "04_BSV量子光" / "GH5_node_occ_29458_bracket.json",
        "29452": repo / "docs" / "04_BSV量子光" / "Isafe_occ_29452_bracket.json",
        "29403": repo / "docs" / "04_BSV量子光" / "GH7_tail_k20_29403_edge_band_audit.json",
    }
    inputs = {
        "CrI3.eig": str(eig_path),
        "CrI3.amn": str(amn_path),
        "CrI3_band.dat": str(wband),
        "CrI3_bands_nbnd200.dat.gnu": str(qe_gnu),
    }
    input_sha = {k: sha256_file(Path(v)) for k, v in inputs.items()}
    occ_sha = {k: sha256_file(p) for k, p in occ_files.items() if p.exists()}
    if k40_bands.exists():
        input_sha["k40_nb112_bands.dat"] = sha256_file(k40_bands)

    eig = load_eig(eig_path)
    nk, nb = eig.shape
    n_win = ((eig >= DIS_WIN[0]) & (eig <= DIS_WIN[1])).sum(axis=1)
    p_d, p_p, p_tot = stream_amn(amn_path, nk, nb)

    bins = [
        ("deep_win", -6.0, -2.0),
        ("valence", -2.0, 0.0),
        ("CB1", 0.8, 2.0),
        ("CB2_in_wannier", 2.0, 3.6),
        ("above_wannier_top", 3.6, 5.0),
        ("highCB_5_6.5", 5.0, 6.5),
        ("window_edge_6.5_8", 6.5, 8.01),
        ("above_window", 8.0, 20.0),
    ]
    proj = {
        name: {"lo": lo, "hi": hi, **window_stats(eig, p_d, p_p, p_tot, lo, hi)}
        for name, lo, hi in bins
    }

    proxy_lo, proxy_hi = [], []
    leftover_P, kept_P, left_E, keep_max = [], [], [], []
    for ik in range(nk):
        in_win = np.where((eig[ik] >= DIS_WIN[0]) & (eig[ik] <= DIS_WIN[1]))[0]
        P = p_tot[ik, in_win]
        E = eig[ik, in_win]
        n_left = len(in_win) - N_WAN
        require(n_left == 32, f"complement dim {n_left} at k={ik}")
        order = np.argsort(P)
        left_idx = in_win[order[:n_left]]
        proxy_lo.append(int(left_idx.min() + 1))
        proxy_hi.append(int(left_idx.max() + 1))
        leftover_P.append(float(P[order[:n_left]].mean()))
        kept_P.append(float(P[order[n_left:]].mean()))
        left_E.append(E[order[:n_left]])
        keep_max.append(float(E[order[n_left:]].max()))
    left_cat = np.concatenate(left_E)
    proxy_fixed = all(a == 169 and b == 200 for a, b in zip(proxy_lo, proxy_hi))

    kq, eq_raw = load_gnu(qe_gnu)
    eq = eq_raw - QE_FERMI_OVERLAY
    kw, ew = load_gnu(wband)
    xq = (kq - kq[0]) / (kq[-1] - kq[0])
    xw = (kw - kw[0]) / (kw[-1] - kw[0])
    eq_i = np.vstack([np.interp(xw, xq, eq[ib]) for ib in range(eq.shape[0])])
    vbm_q = float(eq[eq <= 0].max())
    vbm_w = float(ew[83].max())
    ew_v = ew - vbm_w
    eq_v = eq_i - vbm_q
    top = ew_v[104:112]
    nearest = nearest_match_errors(top, eq_v)
    ordered = ordered_unique_errors(top, eq_v)

    e_top = ew_v[111]
    next_qe = np.full(e_top.shape, np.nan)
    n_above = np.zeros_like(e_top, dtype=int)
    n_1eV = np.zeros_like(e_top, dtype=int)
    for ik in range(e_top.size):
        higher = eq_v[:, ik][eq_v[:, ik] > e_top[ik] + 0.05]
        n_above[ik] = int((eq_v[:, ik] > e_top[ik] + 0.05).sum())
        n_1eV[ik] = int(((eq_v[:, ik] > e_top[ik] + 0.05) & (eq_v[:, ik] <= e_top[ik] + 1.0)).sum())
        if higher.size:
            next_qe[ik] = higher.min()
    require(np.isfinite(next_qe).all(), "truncation gap has NaN")
    gap = next_qe - e_top

    occ_rows = []
    for tag, p in occ_files.items():
        if p.exists():
            for row in occ_from_audit(p):
                row["job"] = tag
                occ_rows.append(row)

    hot = (
        sbe_bands_at(k40_bands, 40, 40, 0.40, 0.50)
        if k40_bands.exists()
        else {"values": None, "unavailable_reason": "k40 bands.dat not in data-root"}
    )

    ix, iy = 10, 12
    ik = ix * 24 + iy
    Emp = eig[ik]
    mask_high = Emp > 3.0

    report = {
        "schema": SCHEMA,
        "verdict": VERDICT,
        "index_warning": (
            "SBE/Wannier band n is the n-th 112-Hamiltonian eigenvalue, not QE Bloch n. "
            "QE160 bands 105-112 are deep valence. On this 24x24 NSCF the lowest "
            "trial-overlap proxy is QE169-200 at every k; that is still not U_dis Q-space."
        ),
        "jobs": {
            "occupation_campaigns": ["29458", "29452", "29403"],
            "nscf_wannier": "historical CrI3_TB nscf nbnd=200 / Wannier90 112",
            "gates": {
                "POSITIVITY": "rho_nn in [-1e-8, 1+1e-8]",
                "TOP_BUFFER_ABS": "global_edge_max <= 1e-3",
                "TOP_BUFFER_REL": "edge/excitation <= 0.01",
            },
        },
        "provenance": {
            "generator": "tools/analysis/diagnose_full112_top.py",
            "input_sha256": input_sha,
            "occupation_audit_sha256": occ_sha,
            "QE_FERMI_OVERLAY_eV": QE_FERMI_OVERLAY,
        },
        "qe_mp_grid": {
            "nk": nk,
            "nbnd": nb,
            "n_in_outer_window": int(n_win.min()),
            "n_in_outer_window_constant": bool(n_win.min() == n_win.max() == 144),
            "complement_dimension": 32,
            "highest_band_E_minmax": [float(eig[:, -1].min()), float(eig[:, -1].max())],
        },
        "projectability_by_energy": proj,
        "leftover_by_trial_P": {
            "note": (
                "PROXY ONLY: 32 lowest trial-A overlap states in the outer window. "
                "Not U_dis complement. 32 is subspace complement dimension, not deleted bands."
            ),
            "leftover_P_mean": float(np.mean(leftover_P)),
            "kept_P_mean": float(np.mean(kept_P)),
            "leftover_E_min": float(left_cat.min()),
            "leftover_E_max": float(left_cat.max()),
            "leftover_E_mean": float(left_cat.mean()),
            "proxy_band_index_min": int(min(proxy_lo)),
            "proxy_band_index_max": int(max(proxy_hi)),
            "proxy_band_index_fixed_QE169_200": proxy_fixed,
            "kept_Emax_mean": float(np.mean(keep_max)),
        },
        "path_match": {
            "method_nearest_neighbor": {
                "note": "each Wannier eigenvalue to closest QE eigenvalue; collisions allowed",
                "bands_105_112": summarize_err(nearest),
            },
            "method_ordered_unique": {
                "note": (
                    "一一有序: 8 QE eigenvalues nearest the Wannier 105-112 cluster "
                    "center, both sorted, paired in order"
                ),
                "bands_105_112": summarize_err(ordered),
            },
            "vbm_qe": vbm_q,
            "vbm_wannier_band84_max": vbm_w,
            "E_105_112_vs_VBM": {
                "min": float(top.min()),
                "max": float(top.max()),
                "mean": float(top.mean()),
            },
        },
        "truncation_gap_path": {
            "W112_vs_VBM_mean": float(e_top.mean()),
            "W112_vs_VBM_minmax": [float(e_top.min()), float(e_top.max())],
            "next_QE_vs_VBM_mean": float(next_qe.mean()),
            "gap_mean_eV": float(gap.mean()),
            "gap_min_eV": float(gap.min()),
            "gap_max_eV": float(gap.max()),
            "n_QE_above_W112_mean": float(n_above.mean()),
            "n_QE_within_1eV_mean": float(n_1eV.mean()),
            "hw_3200nm_eV": 1239.84193 / 3200.0,
        },
        "sbe_tb_at_occupation_hot_crystal_momentum": hot,
        "qe_mp_near_hotk": {
            "frac_k": [ix / 24.0, iy / 24.0],
            "ik_1based": ik + 1,
            "E_gt_3": [float(x) for x in Emp[mask_high]],
            "P_tot": [float(x) for x in p_tot[ik, mask_high]],
            "P_d": [float(x) for x in p_d[ik, mask_high]],
            "P_p": [float(x) for x in p_p[ik, mask_high]],
            "QE_band_index_1based": [int(x) for x in (np.where(mask_high)[0] + 1)],
        },
        "P_edge": {
            "definition": "sum_{k, n=105..112} max(f,0); N_k=400; physical e/cell = raw/400",
            "table_is": "cross-config magnitude example, not a controlled I-monotonic curve",
            "rows": occ_rows,
        },
        "pdos_existing": {
            "unavailable_reason": (
                "Archived projwfc PDOS (Emin=-3, Emax=+3) is not required to rebuild "
                "schema v2; atomic projection channels in that archive do not include I-d. "
                "Raising Emax does not add projection channels."
            )
        },
    }
    script_sha = sha256_file(Path(__file__))
    report["provenance"]["generator_sha256"] = script_sha
    body = dumps_strict({k: v for k, v in report.items() if k != "content_sha256"})
    report["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return report


def write_report(report: dict, dest: Path) -> None:
    dest.write_text(dumps_strict(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--repo", type=Path, default=QL)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    report = build_report(args.data_root, args.repo)
    write_report(report, args.out)
    print("wrote", args.out)
    print("ordered MAE_eV", report["path_match"]["method_ordered_unique"]["bands_105_112"]["MAE_eV"])
    print("nearest MAE_eV", report["path_match"]["method_nearest_neighbor"]["bands_105_112"]["MAE_eV"])
    print("proxy_fixed", report["leftover_by_trial_P"]["proxy_band_index_fixed_QE169_200"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

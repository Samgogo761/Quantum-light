#!/usr/bin/env python3
"""score_reduced_model.py — unified 4-term scoring for reduced Wannier models.

Curated copy also lives in tools/model_gates/score/ (2026-07-28).
Prefer that folder as the entry point together with the ε_P gate.

Implements plan section 5 (first four terms):
  L_E   : nearest-band RMS error vs QE reference, averaged over energy windows
  L_sub : window state-count mismatch vs QE (proxy for ||P_red - P_ref||_F;
          full projector version pending .chk / U-matrix parsing)
  L_loc : Omega_total/N_WF + lambda_max * max_alpha Omega_alpha
  L_hop : 1 - sum|t|^2(kept pairs) / sum|t|^2(all pairs), from full112 tb.dat
          (off-diagonal + inter-site couplings; onsite i==j @ R=0 excluded)

Total = sum_i weight_i * (L_i / scale_i).  Each term also gets a PASS/WARN/FAIL
gate ("四级检验" per term).  Reserved (weight 0 this round): L_opt, L_HHG.

Usage:
  python score_reduced_model.py --config score_config.yml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- config ----
def load_config(path: Path) -> dict:
    cfg = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        cfg[key.strip()] = val.strip()
    return cfg


def cfg_models(cfg: dict) -> dict:
    models = {}
    for key, val in cfg.items():
        m = re.match(r"model\.([^.]+)\.(band|wout|kept|hop_keep_frac)$", key)
        if m:
            models.setdefault(m.group(1), {})[m.group(2)] = val
    return models


# ------------------------------------------------------------- band data ----
def load_gnu_blocks(path: Path):
    bands, k, e = [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            if k:
                bands.append((k, e))
                k, e = [], []
            continue
        a, b = map(float, s.split()[:2])
        k.append(a)
        e.append(b)
    if k:
        bands.append((k, e))
    k0 = np.asarray(bands[0][0], float)
    earr = np.vstack([np.asarray(ee, float) for _, ee in bands])
    return k0, earr


def norm_path(k):
    return (k - k[0]) / (k[-1] - k[0])


def vbm_of(e):
    v = e[:, 0]
    return float(v[v <= 0].max())


def nearest_band_rms(qe_x, qe_e, w_x, w_e, lo, hi):
    qe_on_w = np.vstack([np.interp(w_x, qe_x, qe_e[ib]) for ib in range(qe_e.shape[0])])
    errs = []
    for ib in range(w_e.shape[0]):
        wb = w_e[ib]
        mask = (wb >= lo) & (wb <= hi)
        if not np.any(mask):
            continue
        errs.append(np.abs(qe_on_w[:, mask] - wb[mask][None, :]).min(axis=0))
    if not errs:
        return float("nan")
    a = np.concatenate(errs)
    return float(np.sqrt((a**2).mean()))


def statecount_mismatch(qe_x, qe_e, w_x, w_e, lo, hi):
    """RMS over k of |N_red(k;W) - N_ref(k;W)| / max(N_ref, 1)."""
    qe_on_w = np.vstack([np.interp(w_x, qe_x, qe_e[ib]) for ib in range(qe_e.shape[0])])
    n_ref = ((qe_on_w >= lo) & (qe_on_w <= hi)).sum(axis=0).astype(float)
    n_red = ((w_e >= lo) & (w_e <= hi)).sum(axis=0).astype(float)
    frac = np.abs(n_red - n_ref) / np.maximum(n_ref, 1.0)
    return float(np.sqrt((frac**2).mean()))


# ------------------------------------------------------------------ wout ----
def parse_wout(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for tag, pat in [("Omega_I", r"Omega I\s*=\s*([0-9.Ee+-]+)"),
                     ("Omega_total", r"Omega Total\s*=\s*([0-9.Ee+-]+)")]:
        ms = re.findall(pat, txt)
        out[tag] = float(ms[-1]) if ms else float("nan")
    fs = txt.rfind("Final State")
    centres, spreads = [], []
    if fs >= 0:
        for m in re.finditer(
            r"WF centre and spread\s+\d+\s+\(\s*([0-9.Ee+-]+),\s*([0-9.Ee+-]+),\s*([0-9.Ee+-]+)\s*\)\s+([0-9.Ee+-]+)",
            txt[fs:],
        ):
            centres.append([float(m.group(i)) for i in (1, 2, 3)])
            spreads.append(float(m.group(4)))
    return out, np.asarray(centres), np.asarray(spreads)


# -------------------------------------------------- 112 orbital classes -----
def parse_win_structure(win_path: Path):
    txt = win_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"begin unit_cell_cart(.*?)end unit_cell_cart", txt, re.DOTALL)
    lines = [l for l in m.group(1).strip().splitlines() if l.strip() and "angstrom" not in l.lower()]
    cell = np.array([[float(x) for x in l.split()[:3]] for l in lines[:3]])
    m = re.search(r"begin atoms_frac(.*?)end atoms_frac", txt, re.DOTALL)
    species, frac = [], []
    for l in m.group(1).strip().splitlines():
        p = l.split()
        if len(p) >= 4:
            species.append(p[0])
            frac.append([float(x) for x in p[1:4]])
    cart = np.asarray(frac) @ cell
    return cell, species, cart


def classify_wfs(centres: np.ndarray, cell: np.ndarray, species: list, atoms_cart: np.ndarray):
    """Nearest atom (with in-plane periodic images) decides d-like (Cr) vs p-like (I)."""
    images = [n1 * cell[0] + n2 * cell[1] for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)]
    labels = []
    for c in centres:
        best_d, best_s = 1e9, "?"
        for a, sp in zip(atoms_cart, species):
            for img in images:
                d = np.linalg.norm(c - (a + img))
                if d < best_d:
                    best_d, best_s = d, sp
        labels.append("d" if best_s.startswith("Cr") else "p")
    return np.asarray(labels)


def hopping_weights(tb_path: Path, labels: np.ndarray):
    """Stream the H(R) block of tb.dat; accumulate sum|t|^2 per channel."""
    nw_expect = labels.size
    sums = {"dd": 0.0, "dp": 0.0, "pp": 0.0}
    with open(tb_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()  # comment
        for _ in range(3):
            f.readline()  # lattice
        nw = int(f.readline().split()[0])
        nrpts = int(f.readline().split()[0])
        if nw != nw_expect:
            raise SystemExit(f"tb.dat num_wann={nw} != labels {nw_expect}")
        ndeg = 0
        while ndeg < nrpts:
            ndeg += len(f.readline().split())
        per_block = nw * nw
        for ir in range(nrpts):
            # skip blank lines then R vector
            line = f.readline()
            while line and not line.strip():
                line = f.readline()
            rvec = [int(x) for x in line.split()[:3]]
            onsite_R0 = rvec == [0, 0, 0]
            for _ in range(per_block):
                p = f.readline().split()
                i, j = int(p[0]) - 1, int(p[1]) - 1
                if onsite_R0 and i == j:
                    continue
                w2 = float(p[2]) ** 2 + float(p[3]) ** 2
                ch = labels[i] + labels[j]
                if ch == "pd":
                    ch = "dp"
                sums[ch] += w2
    return sums


# ------------------------------------------------------------------ main ----
def grade(value: float, thr_pass: float, thr_warn: float) -> str:
    if np.isnan(value):
        return "N/A"
    if value <= thr_pass:
        return "PASS"
    if value <= thr_warn:
        return "WARN"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--skip-hopping", action="store_true", help="skip tb.dat streaming (reuse cached)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = Path(cfg["path.outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    ef = float(cfg["qe_fermi_eV"])
    windows = {
        "near": (float(cfg["window.near.lo"]), float(cfg["window.near.hi"])),
        "mid": (float(cfg["window.mid.lo"]), float(cfg["window.mid.hi"])),
    }
    weights = {t: float(cfg[f"weight.{t}"]) for t in ("energy", "subspace", "localization", "hopping")}
    scales = {
        "energy": float(cfg["scale.energy_eV"]),
        "subspace": float(cfg["scale.subspace_frac"]),
        "localization": float(cfg["scale.localization_A2"]),
        "hopping": float(cfg["scale.hopping_frac"]),
    }
    lam = float(cfg["localization.lambda_max"])

    # QE reference
    qk, qe = load_gnu_blocks(Path(cfg["path.qe_gnu"]))
    qe = qe - ef
    qe = qe - vbm_of(qe)
    qx = norm_path(qk)

    # hopping channel sums from full112 (once)
    hop_cache = outdir / "hopping_channel_sums.csv"
    if args.skip_hopping and hop_cache.exists():
        vals = dict(l.split(",") for l in hop_cache.read_text().splitlines()[1:])
        sums = {k: float(v) for k, v in vals.items()}
    else:
        w112_meta, c112, s112 = parse_wout(Path(cfg["path.wout112"]))
        win112 = Path(cfg["path.wout112"]).parent / "CrI3.win"
        cell, species, atoms = parse_win_structure(win112)
        labels = classify_wfs(c112, cell, species, atoms)
        nd, npp = int((labels == "d").sum()), int((labels == "p").sum())
        print(f"112 WF classification: d-like={nd}  p-like={npp}")
        sums = hopping_weights(Path(cfg["path.tb112"]), labels)
        hop_cache.write_text(
            "channel,sum_t2\n" + "\n".join(f"{k},{v:.8e}" for k, v in sums.items()) + "\n", encoding="utf-8"
        )
    tot_t2 = sum(sums.values())
    kept_frac = {"d": sums["dd"] / tot_t2, "d+p": 1.0}
    print("hopping |t|^2 fractions: dd=%.4f dp=%.4f pp=%.4f" % (
        sums["dd"] / tot_t2, sums["dp"] / tot_t2, sums["pp"] / tot_t2))

    rows = []
    for name, files in cfg_models(cfg).items():
        wk, we = load_gnu_blocks(Path(files["band"]))
        we = we - vbm_of(we)
        wx = norm_path(wk)
        L_E = float(np.nanmean([nearest_band_rms(qx, qe, wx, we, lo, hi) for lo, hi in windows.values()]))
        L_sub = float(np.nanmean([statecount_mismatch(qx, qe, wx, we, lo, hi) for lo, hi in windows.values()]))
        meta, centres, spreads = parse_wout(Path(files["wout"]))
        n_wf = spreads.size
        L_loc = meta["Omega_total"] / n_wf + lam * float(spreads.max())
        if "hop_keep_frac" in files:
            L_hop = 1.0 - float(files["hop_keep_frac"])
        else:
            L_hop = 1.0 - kept_frac[files["kept"]]

        terms = {"energy": L_E, "subspace": L_sub, "localization": L_loc, "hopping": L_hop}
        normed = {t: terms[t] / scales[t] for t in terms}
        total = sum(weights[t] * normed[t] for t in terms)
        grades = {
            t: grade(terms[t], float(cfg[f"gate.{t}.pass"]), float(cfg[f"gate.{t}.warn"]))
            for t in terms
        }
        rows.append({
            "model": name, "N_WF": n_wf,
            "L_E_eV": L_E, "L_sub": L_sub, "L_loc_A2": L_loc, "L_hop": L_hop,
            **{f"n_{t}": normed[t] for t in terms},
            **{f"g_{t}": grades[t] for t in terms},
            "total": total,
        })

    rows.sort(key=lambda r: r["total"])
    hdr = list(rows[0].keys())
    csv = [",".join(hdr)]
    for r in rows:
        csv.append(",".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in hdr))
    (outdir / "model_ranking.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print("\n".join(csv))

    # report
    md = ["# Reduced-model scoring report", "",
          f"- QE ref: `{cfg['path.qe_gnu']}`", f"- windows: near={windows['near']}, mid={windows['mid']} (E-E_VBM, eV)",
          f"- weights: {weights}", f"- scales: {scales}",
          f"- hopping fractions (full112 |t|^2): dd={sums['dd']/tot_t2:.4f}, dp={sums['dp']/tot_t2:.4f}, pp={sums['pp']/tot_t2:.4f}",
          "", "| model | N_WF | L_E (eV) | L_sub | L_loc (Å²) | L_hop | total | gates (E/sub/loc/hop) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['N_WF']} | {r['L_E_eV']:.4f} | {r['L_sub']:.4f} | {r['L_loc_A2']:.2f} "
            f"| {r['L_hop']:.4f} | **{r['total']:.3f}** | {r['g_energy']}/{r['g_subspace']}/{r['g_localization']}/{r['g_hopping']} |")
    md += ["", "Notes:",
           "- L_sub is a state-count proxy for ||P_red - P_ref||_F (projector version needs .chk U-matrices).",
           "- L_hop uses full112 tb.dat |t|^2 with onsite (i==j, R=0) excluded; pure-d models keep only the dd channel.",
           "- L_opt / L_HHG reserved (weight 0 this round)."]
    (outdir / "model_score_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # radar
    axes_names = ["energy", "subspace", "localization", "hopping"]
    ang = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150, subplot_kw=dict(polar=True))
    for r in rows:
        vals = [min(r[f"n_{t}"], 5.0) for t in axes_names]
        ax.plot(ang + ang[:1], vals + vals[:1], lw=1.4, label=r["model"])
        ax.fill(ang + ang[:1], vals + vals[:1], alpha=0.12)
    ax.set_xticks(ang)
    ax.set_xticklabels([f"L_{t}" for t in axes_names])
    ax.set_title("Reduced-model normalized penalties (lower = better, cap 5)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "radar_plot.png", bbox_inches="tight")
    print("wrote", outdir / "model_ranking.csv")
    print("wrote", outdir / "model_score_report.md")
    print("wrote", outdir / "radar_plot.png")


if __name__ == "__main__":
    main()

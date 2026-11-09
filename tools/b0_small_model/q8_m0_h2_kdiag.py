#!/usr/bin/env python3
"""H2 source diagnosis from k-resolved complex harmonics.

Uses existing nk=48 k_harm_total. Optionally replays the original nk=192,
Γ-centred, 400 steps/cycle ±N pair with k_orders=(2,) only.

Does not change M0/R0/P1, does not densify the mesh, and does not claim
convergence. CJJ physics is not run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_classical import (  # noqa: E402
    analyse_set,
    p1_protocol,
    project_harmonic,
    run_set,
)
from q8_m0_h2_qualify import ABS_REPORT_RESOLUTION, dump, z_of  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NK48_DIR = REPO / "results/q_line/m0_p0p1_20260914/p1_A0.8_nk48_nt400"
NK48_TAG = "p1_A0.8_nk48_nt400"
QUAL192_DIR = REPO / "results/q_line/p1_h2_qualification_20260915/p1_A0.8_nk192_nt400"
OLD_STATIC = REPO / "results/q_line/m0_v1_20260912/static/gap_report.json"
OUT_ROOT = REPO / "results/q_line/p1_h2_kresolved_diag_20260915"
REPLAY_NAME = "p1_A0.8_nk192_nt400_k2"
N_PART = 6
H2_ORDER = 2
RECON_ABS = 1e-14
RECON_REL = 1e-10
PHASE_ABS_FRAC = 0.01


def _c(z) -> complex:
    return complex(z)


def load_pair(outdir: Path, tag: str) -> tuple[dict, dict, dict]:
    plus = dict(np.load(outdir / f"run_{tag}_N+1_A+1.npz", allow_pickle=True))
    minus = dict(np.load(outdir / f"run_{tag}_N-1_A+1.npz", allow_pickle=True))
    meta = json.loads((outdir / f"run_{tag}_N+1_A+1.meta.json").read_text(encoding="utf-8"))
    return plus, minus, meta


def order_index(arr, n: int) -> int:
    stored = np.asarray(arr, dtype=int)
    hit = np.where(stored == int(n))[0]
    if hit.size != 1:
        raise RuntimeError(f"order {n} not unique in {stored}")
    return int(hit[0])


def sheet_from_k(w, z_k, a_cell: float) -> complex:
    return _c(np.dot(w, z_k) / a_cell)


def kappa_cancel(w, z_k) -> dict:
    w = np.asarray(w, dtype=float)
    z_k = np.asarray(z_k)
    num = float(np.dot(w, np.abs(z_k)))
    den = abs(_c(np.dot(w, z_k)))
    out = {"numerator_sum_w_abs_z": num, "denominator_abs_sum_w_z": den}
    if den <= max(1e-30, 1e-12 * num):
        out["kappa"] = None
        out["kappa_status"] = "undefined_near_zero_net"
    else:
        out["kappa"] = num / den
        out["kappa_status"] = "defined"
    return out


def reconstruct_channel(plus, minus, meta, order: int, xy: int, a_cell: float, prot) -> dict:
    w = np.asarray(plus["w_k"], dtype=float)
    i_ord = order_index(plus["k_orders"], order)
    hp = np.asarray(plus["k_harm_total"])[i_ord, :, xy]
    hm = np.asarray(minus["k_harm_total"])[i_ord, :, xy]
    jo = 0.5 * (hp - hm)
    je = 0.5 * (hp + hm)
    z_o = sheet_from_k(w, jo, a_cell)
    z_e = sheet_from_k(w, je, a_cell)
    z_p = sheet_from_k(w, hp, a_cell)
    z_m = sheet_from_k(w, hm, a_cell)
    times = np.asarray(plus["times"], dtype=float)
    jp = np.asarray(plus["J_sheet"], dtype=float)
    jm = np.asarray(minus["J_sheet"], dtype=float)
    omega = order * prot.omega0
    z_o_j = _c(project_harmonic(times, 0.5 * (jp - jm), omega, "sin2")[xy])
    z_e_j = _c(project_harmonic(times, 0.5 * (jp + jm), omega, "sin2")[xy])
    err_o = abs(z_o - z_o_j)
    ok = err_o <= max(RECON_ABS, RECON_REL * max(abs(z_o_j), 1e-30))
    return {
        "order": order,
        "xy": "xy"[xy],
        "z_o_from_k": {"re": z_o.real, "im": z_o.imag, "abs": abs(z_o)},
        "z_e_from_k": {"re": z_e.real, "im": z_e.imag, "abs": abs(z_e)},
        "z_plus_from_k": {"re": z_p.real, "im": z_p.imag, "abs": abs(z_p)},
        "z_minus_from_k": {"re": z_m.real, "im": z_m.imag, "abs": abs(z_m)},
        "z_o_from_J_sheet": {"re": z_o_j.real, "im": z_o_j.imag, "abs": abs(z_o_j)},
        "z_e_from_J_sheet": {"re": z_e_j.real, "im": z_e_j.imag, "abs": abs(z_e_j)},
        "reconstruct_abs_err_Jo": err_o,
        "reconstruct_ok": bool(ok),
        "kappa_Jo": kappa_cancel(w, jo),
        "kappa_Je": kappa_cancel(w, je),
        "jo_k": jo,
        "je_k": je,
        "hp_k": hp,
        "hm_k": hm,
    }


def tile_id(kf_row, npart: int) -> tuple[int, int]:
    x, y = np.mod(np.asarray(kf_row, dtype=float), 1.0)
    ix = min(int(np.floor(x * npart)), npart - 1)
    iy = min(int(np.floor(y * npart)), npart - 1)
    return ix, iy


def partition_report(kf, w, z_k, a_cell: float, npart: int = N_PART) -> list[dict]:
    tiles = {}
    for i, row in enumerate(kf):
        key = tile_id(row, npart)
        rec = tiles.setdefault(key, {"idx": []})
        rec["idx"].append(i)
    out = []
    for ix in range(npart):
        for iy in range(npart):
            idx = np.asarray(tiles[(ix, iy)]["idx"], dtype=int)
            ww = w[idx]
            zz = z_k[idx]
            net = sheet_from_k(ww, zz, a_cell)
            kap = kappa_cancel(ww, zz)
            out.append({
                "ix": ix,
                "iy": iy,
                "label": f"i{ix}_j{iy}",
                "frac_box": [ix / npart, (ix + 1) / npart, iy / npart, (iy + 1) / npart],
                "n_k": int(idx.size),
                "weight_sum": float(ww.sum()),
                "net_sheet": {"re": net.real, "im": net.imag, "abs": abs(net)},
                "kappa": kap,
                "gross_sheet": float(np.dot(ww, np.abs(zz)) / a_cell),
            })
    return out


def roughness(kf, z_k, nk: int) -> dict:
    z = np.asarray(z_k).reshape(nk, nk)
    dx = np.abs(np.diff(z, axis=0))
    dy = np.abs(np.diff(z, axis=1))
    return {
        "mean_abs_nn_diff": float(0.5 * (np.mean(dx) + np.mean(dy))),
        "max_abs_nn_diff": float(max(np.max(dx), np.max(dy))),
        "std_abs_z": float(np.std(np.abs(z))),
        "max_abs_z": float(np.max(np.abs(z))),
    }


def spectrum_neighborhood(plus, minus, prot, xy: int) -> dict:
    times = np.asarray(plus["times"], dtype=float)
    jo = 0.5 * (np.asarray(plus["J_sheet"]) - np.asarray(minus["J_sheet"]))
    ws = np.linspace(0.0, 4.0, 401)
    vals = np.array([project_harmonic(times, jo, w * prot.omega0, "sin2")[xy] for w in ws])
    i2 = int(np.argmin(np.abs(ws - 2.0)))
    return {
        "w_over_omega0": ws.tolist(),
        "re": vals.real.tolist(),
        "im": vals.imag.tolist(),
        "abs": np.abs(vals).tolist(),
        "abs_at_H2": float(np.abs(vals[i2])),
        "abs_at_H0": float(np.abs(vals[0])),
        "note": "finer Fourier sampling for display only; pulse frequency resolution is unchanged",
    }


def pack_z(z: complex) -> dict:
    return {"re": float(np.real(z)), "im": float(np.imag(z)), "abs": float(abs(z))}


def analyze_pair(plus, minus, meta, prot, label: str) -> dict:
    a_cell = float(meta["a_cell_ang2"])
    kf = np.asarray(plus["kf"], dtype=float)
    w = np.asarray(plus["w_k"], dtype=float)
    nk = int(round(np.sqrt(kf.shape[0])))
    if nk * nk != kf.shape[0]:
        raise RuntimeError("k grid is not square")
    rec = {
        "label": label,
        "nk": nk,
        "a_cell_ang2": a_cell,
        "k_orders": [int(x) for x in np.asarray(plus["k_orders"])],
        "grid_shift": float(plus["grid_shift"]) if "grid_shift" in plus else float(meta.get("grid_shift", 0.0)),
        "nt_per_cycle": int(plus["nt_per_cycle"]),
        "weight_sum": float(w.sum()),
        "units": {
            "k_harm_total": "per-k current harmonic before /A_cell; trapezoid of sin2 window / t_pulse",
            "sheet_z": "e t_h/(ħ Å); sum_k w_k z_k / A_cell",
        },
        "channels": {},
        "n_exc_note": (
            "n_exc maps are occupancy sidecars at a diagnostic snapshot, not H2 source maps"
        ),
    }
    for xy, name in ((0, "Jo_x_H2"), (1, "Jo_y_H2")):
        ch = reconstruct_channel(plus, minus, meta, H2_ORDER, xy, a_cell, prot)
        jo = ch.pop("jo_k")
        je = ch.pop("je_k")
        ch.pop("hp_k")
        ch.pop("hm_k")
        tiles = partition_report(kf, w, jo, a_cell)
        net_from_tiles = sum(_c(t["net_sheet"]["re"] + 1j * t["net_sheet"]["im"]) for t in tiles)
        ch["tiles_cover_full_bz"] = len(tiles) == N_PART * N_PART
        ch["tile_net_sum_minus_total"] = pack_z(net_from_tiles - _c(ch["z_o_from_k"]["re"] + 1j * ch["z_o_from_k"]["im"]))
        ch["tiles"] = tiles
        ch["roughness_unweighted_z_k"] = roughness(kf, jo, nk)
        ch["spectrum_0_to_4"] = spectrum_neighborhood(plus, minus, prot, xy)
        rec["channels"][name] = ch
        rec["_arrays"] = rec.get("_arrays", {})
        rec["_arrays"][name] = {
            "jo_k": jo,
            "je_k": je,
            "contrib_sheet": (w * jo) / a_cell,
        }
    rec["kf"] = kf
    rec["w"] = w
    rec["n_exc_k_peak"] = np.asarray(plus["n_exc_k_peak"]) if "n_exc_k_peak" in plus else None
    return rec


def _masked_phase(z, floor: float):
    ang = np.angle(z, deg=True)
    return np.ma.masked_where(np.abs(z) < floor, ang)


def save_maps(rec: dict, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    nk = rec["nk"]
    label = rec["label"]
    for cname, arrs in rec["_arrays"].items():
        contrib = arrs["contrib_sheet"].reshape(nk, nk)
        jo = arrs["jo_k"].reshape(nk, nk)
        floor = PHASE_ABS_FRAC * float(np.max(np.abs(jo)))
        packs = {
            "Re_contrib": contrib.real,
            "Im_contrib": contrib.imag,
            "abs_contrib": np.abs(contrib),
            "phase_deg_masked": _masked_phase(jo, floor),
        }
        for q, img in packs.items():
            fig, ax = plt.subplots(figsize=(4.4, 3.7))
            if q.startswith("phase"):
                im = ax.imshow(np.ma.filled(img, np.nan).T, origin="lower", cmap="twilight", vmin=-180, vmax=180)
            elif q.startswith("Re") or q.startswith("Im"):
                vmax = float(np.max(np.abs(contrib)))
                im = ax.imshow(img.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(img.T, origin="lower", cmap="viridis")
            ax.set_title(f"{label} {cname} {q}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(figdir / f"{label}_{cname}_{q}.png", dpi=130)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.4, 3.7))
        order = np.argsort(-np.abs(arrs["contrib_sheet"]))
        walk = np.cumsum(arrs["contrib_sheet"][order])
        ax.plot(walk.real, walk.imag, lw=0.8)
        ax.scatter([walk.real[-1]], [walk.imag[-1]], c="k", s=12, zorder=3)
        ax.set_title(f"{label} {cname} cumulative sum by |contrib|")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.axhline(0.0, color="k", lw=0.3)
        ax.axvline(0.0, color="k", lw=0.3)
        fig.tight_layout()
        fig.savefig(figdir / f"{label}_{cname}_cumwalk.png", dpi=130)
        plt.close(fig)
        spec = rec["channels"][cname]["spectrum_0_to_4"]
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.semilogy(spec["w_over_omega0"], spec["abs"], lw=0.9)
        ax.axvline(2.0, color="k", lw=0.6, ls="--")
        ax.set_xlabel(r"$\Omega/\omega_0$")
        ax.set_ylabel(r"$|J_o(\Omega)|$")
        ax.set_title(f"{label} {cname} sin2 spectrum (display sampling)")
        fig.tight_layout()
        fig.savefig(figdir / f"{label}_{cname}_spectrum.png", dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        for t in rec["channels"][cname]["tiles"]:
            ax.scatter([t["net_sheet"]["re"]], [t["net_sheet"]["im"]], s=18)
            ax.annotate(t["label"], (t["net_sheet"]["re"], t["net_sheet"]["im"]), fontsize=6)
        ax.axhline(0.0, color="k", lw=0.3)
        ax.axvline(0.0, color="k", lw=0.3)
        ax.set_title(f"{label} {cname} 6x6 tile nets")
        ax.set_xlabel("Re net")
        ax.set_ylabel("Im net")
        fig.tight_layout()
        fig.savefig(figdir / f"{label}_{cname}_tiles.png", dpi=130)
        plt.close(fig)
    if rec["n_exc_k_peak"] is not None:
        fig, ax = plt.subplots(figsize=(4.4, 3.7))
        im = ax.imshow(np.asarray(rec["n_exc_k_peak"]).reshape(nk, nk).T / 4.0, origin="lower", cmap="magma")
        ax.set_title(f"{label} n_exc/4 sidecar (not H2 source)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figdir / f"{label}_nexc_over_4_sidecar.png", dpi=130)
        plt.close(fig)


def jsonable_analysis(rec: dict) -> dict:
    skip = {"_arrays", "kf", "w", "n_exc_k_peak"}
    out = {k: v for k, v in rec.items() if k not in skip}
    return out


def analyze_nk48(prot) -> dict:
    plus, minus, meta = load_pair(NK48_DIR, NK48_TAG)
    rec = analyze_pair(plus, minus, meta, prot, "nk48")
    figdir = OUT_ROOT / "figures"
    save_maps(rec, figdir)
    payload = jsonable_analysis(rec)
    dump(OUT_ROOT / "nk48_kresolved.json", payload)
    return rec


def replay_nk192(geo, mu, prot) -> dict:
    outdir = OUT_ROOT / REPLAY_NAME
    t0 = time.perf_counter()
    runs = run_set(
        geo, R0, prot, mu, 192, 400,
        signs=(+1,), neels=(+1, -1), tag=REPLAY_NAME, outdir=outdir, verbose=True,
        grid_shift=0.0, subspace=False, k_orders=(2,),
    )
    rec = {
        "name": REPLAY_NAME,
        "nk": 192,
        "ntpc": 400,
        "grid_shift": 0.0,
        "subspace": False,
        "k_orders": [2],
        "protocol_id": prot.protocol_id,
        "wall_seconds_pair": time.perf_counter() - t0,
        "note": "diagnostic k-export of the original nk192 main grid; not a new convergence check",
    }
    ana, _ = analyse_set(runs, prot, [1, 2])
    rec["analysis"] = ana
    dump(outdir / "analysis.json", rec)
    dump(OUT_ROOT / "export_schema.json", {
        "k_harm_total": "(n_order=1, Nk, 2) complex; per-k current harmonic before /A_cell",
        "k_orders": [2],
        "kf": "(Nk, 2) fractional",
        "w_k": "(Nk,) positive weights summing to 1",
        "reconstruction": "sheet z = sum_k w_k k_harm / A_cell, same sin2 trapezoid as project_harmonic",
        "recon_ok_if": f"abs err <= max({RECON_ABS}, {RECON_REL}*|z_J|)",
        "old_cache": "qualification nk192 totals-only files were not rewritten",
    })
    return rec


def analyze_nk192(prot, replay_meta_rec: dict | None = None) -> dict:
    plus, minus, meta = load_pair(OUT_ROOT / REPLAY_NAME, REPLAY_NAME)
    rec = analyze_pair(plus, minus, meta, prot, "nk192_k2")
    save_maps(rec, OUT_ROOT / "figures")
    payload = jsonable_analysis(rec)
    dump(OUT_ROOT / "nk192_kresolved.json", payload)
    if QUAL192_DIR.exists():
        old = json.loads((QUAL192_DIR / "analysis.json").read_text(encoding="utf-8"))
        tab = old["analysis"]["windows"]["sin2"]
        vs_old = {}
        for cname, n, xy in (("Jo_x_H2", 2, "x"), ("Jo_y_H2", 2, "y")):
            z_old = z_of(tab, "J_o", n, xy)
            z_new = _c(rec["channels"][cname]["z_o_from_k"]["re"] + 1j * rec["channels"][cname]["z_o_from_k"]["im"])
            vs_old[cname] = {
                "abs_diff_vs_qualification_totals": abs(z_new - z_old),
                "z_old": pack_z(z_old),
                "z_kexport": pack_z(z_new),
            }
        dump(OUT_ROOT / "nk192_vs_qualification_totals.json", vs_old)
        rec["vs_qualification_totals"] = vs_old
    if replay_meta_rec is not None:
        rec["replay_wall_seconds_pair"] = replay_meta_rec.get("wall_seconds_pair")
    return rec


def compare_grids(nk48: dict, nk192: dict) -> dict:
    out = {"npart": N_PART, "channels": {}}
    for cname in ("Jo_x_H2", "Jo_y_H2"):
        t48 = { (t["ix"], t["iy"]): t for t in nk48["channels"][cname]["tiles"] }
        t192 = { (t["ix"], t["iy"]): t for t in nk192["channels"][cname]["tiles"] }
        rows = []
        for ix in range(N_PART):
            for iy in range(N_PART):
                a = t48[(ix, iy)]
                b = t192[(ix, iy)]
                za = _c(a["net_sheet"]["re"] + 1j * a["net_sheet"]["im"])
                zb = _c(b["net_sheet"]["re"] + 1j * b["net_sheet"]["im"])
                rows.append({
                    "label": a["label"],
                    "ix": ix,
                    "iy": iy,
                    "z48": a["net_sheet"],
                    "z192": b["net_sheet"],
                    "abs_diff": abs(za - zb),
                    "gross48": a["gross_sheet"],
                    "gross192": b["gross_sheet"],
                    "kappa48": a["kappa"],
                    "kappa192": b["kappa"],
                })
        rows.sort(key=lambda r: -r["abs_diff"])
        z48 = _c(nk48["channels"][cname]["z_o_from_k"]["re"] + 1j * nk48["channels"][cname]["z_o_from_k"]["im"])
        z192 = _c(nk192["channels"][cname]["z_o_from_k"]["re"] + 1j * nk192["channels"][cname]["z_o_from_k"]["im"])
        out["channels"][cname] = {
            "total_z48": pack_z(z48),
            "total_z192": pack_z(z192),
            "total_abs_diff": abs(z48 - z192),
            "kappa48": nk48["channels"][cname]["kappa_Jo"],
            "kappa192": nk192["channels"][cname]["kappa_Jo"],
            "roughness48": nk48["channels"][cname]["roughness_unweighted_z_k"],
            "roughness192": nk192["channels"][cname]["roughness_unweighted_z_k"],
            "tiles_ranked_by_abs_diff": rows,
            "top3_diff_tiles": [r["label"] for r in rows[:3]],
            "sum_abs_tile_diff": float(sum(r["abs_diff"] for r in rows)),
        }
    dump(OUT_ROOT / "nk48_vs_nk192_tiles.json", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("nk48", "replay-nk192", "compare", "auto"), default="nk48")
    args = ap.parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    prot = p1_protocol(0.8)
    geo = build_geometry()
    mu = float(json.loads(OLD_STATIC.read_text(encoding="utf-8"))["mu_half_filling"])
    rec48 = None
    rec192 = None
    replay_rec = None
    if args.stage in ("nk48", "auto", "compare"):
        rec48 = analyze_nk48(prot)
        print(json.dumps({
            "stage": "nk48",
            "Jo_x_kappa": rec48["channels"]["Jo_x_H2"]["kappa_Jo"],
            "Jo_y_kappa": rec48["channels"]["Jo_y_H2"]["kappa_Jo"],
            "Jo_x_recon_ok": rec48["channels"]["Jo_x_H2"]["reconstruct_ok"],
            "Jo_y_recon_ok": rec48["channels"]["Jo_y_H2"]["reconstruct_ok"],
        }, indent=2), flush=True)
    if args.stage in ("replay-nk192", "auto"):
        replay_rec = replay_nk192(geo, mu, prot)
        rec192 = analyze_nk192(prot, replay_rec)
        print(json.dumps({
            "stage": "nk192_k2",
            "wall_seconds_pair": replay_rec["wall_seconds_pair"],
            "Jo_x_kappa": rec192["channels"]["Jo_x_H2"]["kappa_Jo"],
            "Jo_y_kappa": rec192["channels"]["Jo_y_H2"]["kappa_Jo"],
            "Jo_x_recon_ok": rec192["channels"]["Jo_x_H2"]["reconstruct_ok"],
            "Jo_y_recon_ok": rec192["channels"]["Jo_y_H2"]["reconstruct_ok"],
        }, indent=2), flush=True)
    if args.stage == "compare" and (OUT_ROOT / REPLAY_NAME / f"run_{REPLAY_NAME}_N+1_A+1.npz").exists():
        rec192 = analyze_nk192(prot)
    if rec48 is not None and rec192 is not None:
        cmp_ = compare_grids(rec48, rec192)
        print(json.dumps({
            "stage": "compare",
            "Jo_x_top3": cmp_["channels"]["Jo_x_H2"]["top3_diff_tiles"],
            "Jo_y_top3": cmp_["channels"]["Jo_y_H2"]["top3_diff_tiles"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

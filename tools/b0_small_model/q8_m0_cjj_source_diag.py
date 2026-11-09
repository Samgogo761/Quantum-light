#!/usr/bin/env python3
"""CJJ source-kernel k-maps from saved Y. No new propagation.

Diagnoses g_± and g_o for H2x/H6y and Tr(Y2x† Y3x) domain difference.
Does not copy mean-H2 filament maps into CJJ refinement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump  # noqa: E402
from q8_m0_cjj_validate import CJJ_ABS_UNIT  # noqa: E402
from q8_m0_h2_cancel_symmetry import residual_report, static_c2_check, static_p_check  # noqa: E402
from q8_m0_model import build_geometry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROUND1 = REPO / "results/q_line/cjj_m0_p1_validation_20260916"
ROUND2 = REPO / "results/q_line/cjj_m0_p1_qualification_v2_20260917"
OUT = REPO / "results/q_line/cjj_source_diag_20260917"
N_TILE = 6
JOBS = (
    ("G48", ROUND1, "V48", 48, 0.0),
    ("S48", ROUND1, "V48S", 48, 0.5),
    ("G96", ROUND2, "V96", 96, 0.0),
    ("S96", ROUND2, "V96S", 96, 0.5),
)
# order index in Y: harmonic n is Y[:, n-1, a]
CHANNELS = {
    "H2x": {"kind": "diag", "n": 2, "a": 0},
    "H6y": {"kind": "diag", "n": 6, "a": 1},
    "H2xH3x": {"kind": "cross", "n": 2, "a": 0, "m": 3, "b": 0},
}


def gram_diag(y_na: np.ndarray) -> np.ndarray:
    return np.einsum("kcv,kcv->k", np.conj(y_na), y_na)


def gram_cross(y_na: np.ndarray, y_mb: np.ndarray) -> np.ndarray:
    return np.einsum("kcv,kcv->k", np.conj(y_na), y_mb)


def partner_index(kf: np.ndarray) -> tuple[np.ndarray, int]:
    keys = {tuple(np.round(np.mod(row, 1.0), 12)): i for i, row in enumerate(np.asarray(kf, dtype=float))}
    pidx = np.empty(len(kf), dtype=int)
    unpaired = 0
    for i, row in enumerate(kf):
        partner = tuple(np.round(np.mod(-np.asarray(row, dtype=float), 1.0), 12))
        if partner not in keys:
            unpaired += 1
            pidx[i] = i
        else:
            pidx[i] = keys[partner]
    return pidx, unpaired


def tile_ids(kf: np.ndarray, n_tile: int = N_TILE) -> np.ndarray:
    ij = np.floor(np.mod(kf, 1.0) * n_tile).astype(int) % n_tile
    return ij[:, 0] * n_tile + ij[:, 1]


def tile_partner(tid: int, n_tile: int = N_TILE) -> int:
    i, j = divmod(int(tid), n_tile)
    return (n_tile - i) % n_tile * n_tile + (n_tile - j) % n_tile


def signed_mass(z: np.ndarray, w: np.ndarray) -> dict:
    z = np.asarray(z)
    w = np.asarray(w, dtype=float)
    re = np.real(z)
    im = np.imag(z) if np.iscomplexobj(z) else np.zeros_like(re)
    is_c = bool(np.max(np.abs(im)) > 1e-12)
    net_re = float(np.dot(w, re))
    net_im = float(np.dot(w, im))
    net = complex(net_re, net_im)
    gross = float(np.dot(w, np.abs(z) if is_c else np.abs(re)))
    return {
        "net": net_re if not is_c else None,
        "net_re": net_re,
        "net_im": net_im,
        "net_abs": float(abs(net)),
        "gross": gross,
        "pos": float(np.dot(w, np.clip(re, 0.0, None))),
        "neg": float(np.dot(w, np.clip(re, None, 0.0))),
        "pos_re": float(np.dot(w, np.clip(re, 0.0, None))),
        "neg_re": float(np.dot(w, np.clip(re, None, 0.0))),
        "pos_im": float(np.dot(w, np.clip(im, 0.0, None))),
        "neg_im": float(np.dot(w, np.clip(im, None, 0.0))),
        "kappa_cancel": None if abs(net) < 1e-30 else gross / abs(net),
        "complex": is_c,
    }


def load_pair(root: Path, name: str) -> dict:
    plus = np.load(root / name / f"{name}_N+1_A+1.npz", mmap_mode="r")
    minus = np.load(root / name / f"{name}_N-1_A+1.npz", mmap_mode="r")
    return {
        "Y+": np.asarray(plus["Y"]),
        "Y-": np.asarray(minus["Y"]),
        "w": np.asarray(plus["w_k"], dtype=float),
        "kf": np.asarray(plus["kf"], dtype=float),
        "d+": np.asarray(plus["d_namb"]),
        "d-": np.asarray(minus["d_namb"]),
        "nk_lin": int(np.asarray(plus["nk"])),
    }


def channel_fields(yp, ym, spec: dict) -> dict:
    if spec["kind"] == "diag":
        n, a = spec["n"] - 1, spec["a"]
        gp = gram_diag(yp[:, n, a])
        gm = gram_diag(ym[:, n, a])
        # numerical imag should be ~0
        gp = np.real(gp)
        gm = np.real(gm)
    else:
        n, a, m, b = spec["n"] - 1, spec["a"], spec["m"] - 1, spec["b"]
        gp = gram_cross(yp[:, n, a], yp[:, m, b])
        gm = gram_cross(ym[:, n, a], ym[:, m, b])
    go = 0.5 * (gp - gm)
    ge = 0.5 * (gp + gm)
    return {"g+": gp, "g-": gm, "go": go, "ge": ge}


def reconstruct_check(fields: dict, dplus, dminus, spec: dict, w) -> dict:
    if spec["kind"] == "diag":
        n, a = spec["n"] - 1, spec["a"]
        saved_p = complex(dplus[n, a, n, a])
        saved_m = complex(dminus[n, a, n, a])
        rec_p = complex(np.dot(w, fields["g+"]))
        rec_m = complex(np.dot(w, fields["g-"]))
    else:
        n, a, m, b = spec["n"] - 1, spec["a"], spec["m"] - 1, spec["b"]
        saved_p = complex(dplus[n, a, m, b])
        saved_m = complex(dminus[n, a, m, b])
        rec_p = complex(np.dot(w, fields["g+"]))
        rec_m = complex(np.dot(w, fields["g-"]))
    return {
        "plus_abs_diff": abs(rec_p - saved_p),
        "minus_abs_diff": abs(rec_m - saved_m),
        "odd_abs_diff": abs(0.5 * (rec_p - rec_m) - 0.5 * (saved_p - saved_m)),
    }


def inversion_parts(go: np.ndarray, pidx: np.ndarray, w: np.ndarray) -> dict:
    go_p = go[pidx]
    even = 0.5 * (go + go_p)
    odd = 0.5 * (go - go_p)
    return {
        "even": signed_mass(even, w),
        "odd": signed_mass(odd, w),
        "go_minus_go_inv": residual_report(go, go_p, w) if not np.iscomplexobj(go) else residual_report(np.real(go), np.real(go_p), w),
        "go_plus_go_inv": residual_report(go, -go_p, w) if not np.iscomplexobj(go) else residual_report(np.real(go), np.real(-go_p), w),
        "note": "even part of g_o under k→-k; nonzero even part means the integrand is not a strict odd function",
    }


def tile_integrals(go, w, kf) -> dict:
    tid = tile_ids(kf)
    rows = []
    for t in range(N_TILE * N_TILE):
        mask = tid == t
        z = go[mask]
        wt = w[mask]
        i, j = divmod(t, N_TILE)
        rows.append({
            "label": f"i{i}_j{j}",
            "tid": t,
            "partner": f"i{(N_TILE - i) % N_TILE}_j{(N_TILE - j) % N_TILE}",
            "n_k": int(np.count_nonzero(mask)),
            **signed_mass(z, wt),
        })
    return rows


def load_all_channels() -> dict:
    out = {}
    for label, root, name, nk, shift in JOBS:
        pair = load_pair(root, name)
        pidx, unpaired = partner_index(pair["kf"])
        job = {
            "nk": nk,
            "shift": shift,
            "n_k": int(pair["kf"].shape[0]),
            "unpaired": unpaired,
            "channels": {},
        }
        for cname, spec in CHANNELS.items():
            fields = channel_fields(pair["Y+"], pair["Y-"], spec)
            rec = reconstruct_check(fields, pair["d+"], pair["d-"], spec, pair["w"])
            job["channels"][cname] = {
                "recon": rec,
                "plus": signed_mass(fields["g+"], pair["w"]),
                "minus": signed_mass(fields["g-"], pair["w"]),
                "even": signed_mass(fields["ge"], pair["w"]),
                "odd": signed_mass(fields["go"], pair["w"]),
                "inversion": inversion_parts(fields["go"], pidx, pair["w"]),
                "tiles": tile_integrals(fields["go"], pair["w"], pair["kf"]),
            }
            job["channels"][cname]["_go"] = np.asarray(fields["go"])
            job["channels"][cname]["_ge"] = np.asarray(fields["ge"])
            job["channels"][cname]["_gp"] = np.asarray(fields["g+"])
            job["channels"][cname]["_gm"] = np.asarray(fields["g-"])
            job["channels"][cname]["_kf"] = pair["kf"]
            job["channels"][cname]["_w"] = pair["w"]
        out[label] = job
        print(json.dumps({
            "loaded": label,
            "H2x_odd": job["channels"]["H2x"]["odd"],
            "H6y_odd": job["channels"]["H6y"]["odd"],
            "recon_H2x": rec if cname == "H2x" else job["channels"]["H2x"]["recon"],
        }, indent=2, default=str), flush=True)
    return out


def disagreement_tiles(a_tiles, b_tiles) -> dict:
    """Compare same-label tile nets between two quadratures."""
    a = {t["label"]: t for t in a_tiles}
    rows = []
    for t in b_tiles:
        lab = t["label"]
        ta, tb = a[lab], t
        da = abs(complex(ta["net_re"], ta["net_im"]) - complex(tb["net_re"], tb["net_im"]))
        ga = 0.5 * (ta["gross"] + tb["gross"])
        rows.append({"label": lab, "partner": t["partner"], "abs_dnet": da, "mean_gross": ga})
    rows.sort(key=lambda r: -r["abs_dnet"])
    tot = sum(r["abs_dnet"] for r in rows) or 1.0
    top6 = rows[:6]
    return {
        "tiles_by_abs_dnet": rows,
        "frac_top6_of_sum_abs_dnet": sum(r["abs_dnet"] for r in top6) / tot,
        "frac_top12_of_sum_abs_dnet": sum(r["abs_dnet"] for r in rows[:12]) / tot,
        "n_tiles": len(rows),
    }


def _heat(ax, z, title: str, diverging: bool) -> None:
    z = np.asarray(z, dtype=float)
    vmax = float(np.percentile(np.abs(z), 99.5) or 1.0)
    if diverging:
        im = ax.imshow(z.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    else:
        im = ax.imshow(z.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_maps(all_jobs: dict, disagreements: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for cname in CHANNELS:
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
        for ax, lab in zip(axes.ravel(), ("G48", "S48", "G96", "S96")):
            nk = all_jobs[lab]["nk"]
            z = np.real(all_jobs[lab]["channels"][cname]["_go"]).reshape(nk, nk)
            im = _heat(ax, z, f"{cname} g_o  {lab}", True)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"CJJ domain-odd integrand (not mean H2). {CJJ_ABS_UNIT}")
        fig.savefig(figdir / f"go_{cname}_four_grids.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
        nk = 96
        ch = all_jobs["G96"]["channels"][cname]
        panels = (
            (np.real(ch["_gp"]), f"{cname} g_+  G96", not np.iscomplexobj(ch["_gp"])),
            (np.real(ch["_gm"]), f"{cname} g_-  G96", not np.iscomplexobj(ch["_gm"])),
            (np.real(ch["_go"]), f"{cname} g_o  G96 Re", True),
            (np.real(ch["_ge"]), f"{cname} g_e  G96 Re", not np.iscomplexobj(ch["_ge"])),
        )
        for ax, (z, title, div) in zip(axes.ravel(), panels):
            im = _heat(ax, np.asarray(z).reshape(nk, nk), title, div)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(figdir / f"gpm_{cname}_G96.png", dpi=140)
        plt.close(fig)

        if CHANNELS[cname]["kind"] == "cross":
            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
            go = all_jobs["G96"]["channels"][cname]["_go"]
            for ax, z, title in (
                (axes[0], np.real(go).reshape(96, 96), f"{cname} g_o Re G96"),
                (axes[1], np.imag(go).reshape(96, 96), f"{cname} g_o Im G96"),
            ):
                im = _heat(ax, z, title, True)
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.savefig(figdir / f"go_{cname}_G96_reim.png", dpi=140)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
        rows = disagreements[cname]["G96_vs_S96"]["tiles_by_abs_dnet"]
        z = np.zeros((N_TILE, N_TILE))
        for r in rows:
            i, j = (int(p[1:]) for p in r["label"].split("_"))
            z[i, j] = r["abs_dnet"]
        im = ax.imshow(z.T, origin="lower", cmap="magma")
        ax.set_title(f"{cname} tile |Δnet| G96 vs S96")
        ax.set_xlabel("tile i")
        ax.set_ylabel("tile j")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(figdir / f"tile_disagree_{cname}_G96_S96.png", dpi=140)
        plt.close(fig)


def quadrature_proposal(disagreements: dict, odd_summary: dict) -> dict:
    """One scheme, design only. Prefer tiles that disagree between G96 and S96."""
    h2 = disagreements["H2x"]["G96_vs_S96"]
    h6 = disagreements["H6y"]["G96_vs_S96"]
    loc_h2 = h2["frac_top6_of_sum_abs_dnet"]
    loc_h6 = h6["frac_top6_of_sum_abs_dnet"]
    localized = loc_h2 >= 0.45 or loc_h6 >= 0.45
    # inversion-close the hottest disagreement tiles of H2x (primary odd leftover)
    hot = []
    for row in h2["tiles_by_abs_dnet"][:8]:
        hot.append(row["label"])
        hot.append(row["partner"])
    hot = sorted(set(hot))
    if localized:
        idea = (
            "Keep full-BZ coverage and Σw=1. In the inversion-closed tiles that carry most of "
            "the G96–S96 disagreement of CJJ g_o(H2x), use local spacing 1/192; elsewhere 1/96. "
            "Do not copy mean-H2 hot belts. Independent control: the same composite rule with "
            "the complementary half-step shift. Stop after one ±N pair if |z_G - z_S| for the "
            "odd H2x/H6y kernels does not fall below the current G96–S96 absolute split."
        )
        kind = "disagreement_tile_refined_inversion_paired"
    else:
        idea = (
            "G96–S96 disagreement of g_o is not concentrated enough to justify a local densify. "
            "Do not invent a hot-belt from mean H2. Do not run uniform nk192. The only stock-supported "
            "next numerical statement is to keep reporting the two 96-grid odd values and their "
            "absolute split as an empirical discrete spread, not a guaranteed error bar."
        )
        kind = "no_local_refine_stock_insufficient_to_site_a_belt"
        hot = []
    return {
        "do_not_run_this_round": True,
        "kind": kind,
        "localized_enough_to_site_tiles": localized,
        "H2x_G96S96_frac_top6": loc_h2,
        "H6y_G96S96_frac_top6": loc_h6,
        "candidate_tiles_inversion_closed": hot,
        "idea": idea,
        "cost_if_run_later": (
            "If localized: 8–16 tiles of 36 at 4× density vs 96, rest at 96: roughly 0.3–0.6 of a "
            "uniform nk192 pair. If not localized: do not run."
        ),
        "stop_rule": (
            "At most one ±N pair. Stop if odd H2x/H6y |z_G-z_S| does not beat the current "
            f"H2x split {odd_summary['H2x']['G96_S96_abs']:.3e}. Never auto-start nk192."
        ),
        "not_chosen": "uniform nk192; mean-H2 filament tiles; treating two-grid spread as a proven bound",
        "quantum_input": "not_run",
    }


def symmetry_archive(geo) -> dict:
    p = static_p_check(geo, 24)
    c2 = static_c2_check(geo, 24)
    return {
        "operator_map": {
            "P": "(N,k,A) → (−N,−k,−A)",
            "current": "polar vector: P j P† = −j after sewing to the folded −k",
            "two_current_CJJ": (
                "each current contributes a minus, so c^>_N,ab[A](t,t') = c^>_{-N,ab}[-A](t,t')"
            ),
            "filtered_d": "d_N[A] = d_{-N}[-A] for a fixed source window",
            "even_odd_in_A": "d_e even under A→−A; d_o odd under A→−A",
            "unlike_mean_current": (
                "mean Je is odd in A and even in N-average; mean Jo is even in A. "
                "Do not copy 'Néel-odd = even harmonic' from the mean current onto CJJ."
            ),
        },
        "same_pulse_lab": {
            "two_domain_experiment_uses_same_A": True,
            "P_does_not_force_d_o_zero": True,
            "finite_pulse_is_not_half_period_A_flip": (
                "8-cycle sin² envelope and the present time window do not license replacing "
                "A→−A by a strict half-cycle time translation"
            ),
            "negative_field_CJJ": "not_run",
        },
        "static_P": p,
        "static_C2_M": c2,
        "k_to_minus_k_same_A": (
            "If g_o were strictly odd under k→−k on an inversion-closed equal-weight grid, "
            "the BZ sum would vanish. A clearly nonzero even part means that map is not an identity."
        ),
        "cannot_declare": (
            "Absence of a same-pulse forcing map is not a proof that leftover cancellation is purely dynamical."
        ),
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    all_jobs = load_all_channels()
    slim_jobs = {}
    for lab, job in all_jobs.items():
        slim_jobs[lab] = {
            "nk": job["nk"],
            "shift": job["shift"],
            "n_k": job["n_k"],
            "unpaired": job["unpaired"],
            "channels": {
                c: {k: v for k, v in ch.items() if not k.startswith("_")}
                for c, ch in job["channels"].items()
            },
        }
    disagreements = {}
    for cname in CHANNELS:
        disagreements[cname] = {
            "G48_vs_S48": disagreement_tiles(
                all_jobs["G48"]["channels"][cname]["tiles"],
                all_jobs["S48"]["channels"][cname]["tiles"],
            ),
            "G96_vs_S96": disagreement_tiles(
                all_jobs["G96"]["channels"][cname]["tiles"],
                all_jobs["S96"]["channels"][cname]["tiles"],
            ),
            "G48_vs_G96": disagreement_tiles(
                all_jobs["G48"]["channels"][cname]["tiles"],
                all_jobs["G96"]["channels"][cname]["tiles"],
            ),
            "S48_vs_S96": disagreement_tiles(
                all_jobs["S48"]["channels"][cname]["tiles"],
                all_jobs["S96"]["channels"][cname]["tiles"],
            ),
        }
    odd_summary = {}
    for cname in CHANNELS:
        odd_summary[cname] = {
            lab: all_jobs[lab]["channels"][cname]["odd"] for lab in ("G48", "S48", "G96", "S96")
        }
        g96 = all_jobs["G96"]["channels"][cname]["odd"]
        s96 = all_jobs["S96"]["channels"][cname]["odd"]
        g48 = all_jobs["G48"]["channels"][cname]["odd"]
        s48 = all_jobs["S48"]["channels"][cname]["odd"]
        d96 = abs(complex(g96["net_re"], g96["net_im"]) - complex(s96["net_re"], s96["net_im"]))
        d48 = abs(complex(g48["net_re"], g48["net_im"]) - complex(s48["net_re"], s48["net_im"]))
        odd_summary[cname]["G48_S48_abs"] = d48
        odd_summary[cname]["G96_S96_abs"] = d96
    even_rel_shift = {}
    for cname in ("H2x", "H6y"):
        e48 = all_jobs["G48"]["channels"][cname]["even"]["net_re"]
        s48 = all_jobs["S48"]["channels"][cname]["even"]["net_re"]
        e96 = all_jobs["G96"]["channels"][cname]["even"]["net_re"]
        s96 = all_jobs["S96"]["channels"][cname]["even"]["net_re"]
        even_rel_shift[cname] = {
            "G48": e48,
            "S48": s48,
            "G96": e96,
            "S96": s96,
            "rel_GS_48": abs(e48 - s48) / abs(e48) if e48 else None,
            "rel_GS_96": abs(e96 - s96) / abs(e96) if e96 else None,
        }
    save_maps(all_jobs, disagreements)
    # drop heavy arrays before dump
    rec = {
        "postprocess": "saved_Y_only_no_new_propagation",
        "abs_unit": CJJ_ABS_UNIT,
        "jobs": slim_jobs,
        "odd_summary": odd_summary,
        "even_common_shift_rel": even_rel_shift,
        "disagreement_tiles": {
            c: {k: {kk: vv for kk, vv in pack.items() if kk != "tiles_by_abs_dnet"}
                | {"top8": pack["tiles_by_abs_dnet"][:8]}
                for k, pack in disagreements[c].items()}
            for c in CHANNELS
        },
        "quadrature_design": quadrature_proposal(disagreements, odd_summary),
        "symmetry": symmetry_archive(geo),
        "corrections": {
            "H6y_sign": "G96 and S96 odd H6y are both positive; the sign change is S48→S96, not G96↔S96",
            "H2x_even_not_monotonic": "Γ/shift relative split of common H2x grew from ~0.32% at 48 to ~3.19% at 96",
            "odd_abs_split_improved_but_unresolved": True,
            "not_mean_H2_map": True,
        },
        "quantum_light_line": {
            "nonzero_odd_is_not_a_gate_for_the_project": True,
            "common_fluctuation_is_a_usable_preliminary_baseline": True,
            "output_mode_and_quantum_pump": "human_review_not_frozen",
        },
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "new_drive_protocol": "not_run",
    }
    dump(OUT / "cjj_source_diag.json", rec)
    dump(OUT / "cjj_source_diag_slim.json", {
        "odd_summary": odd_summary,
        "even_common_shift_rel": even_rel_shift,
        "disagreement_top": rec["disagreement_tiles"],
        "quadrature_design": rec["quadrature_design"],
        "corrections": rec["corrections"],
        "symmetry_same_pulse": rec["symmetry"]["same_pulse_lab"],
        "H2x_G96_inversion_even": slim_jobs["G96"]["channels"]["H2x"]["inversion"]["even"],
        "H2x_G96_inversion_odd": slim_jobs["G96"]["channels"]["H2x"]["inversion"]["odd"],
        "quantum_input": "not_run",
    })
    print(json.dumps({
        "wrote": str(OUT / "cjj_source_diag_slim.json"),
        "H2x_odd_G96_S96": (odd_summary["H2x"]["G96"]["net"], odd_summary["H2x"]["S96"]["net"]),
        "H6y_odd_signs": {lab: odd_summary["H6y"][lab]["net"] for lab in ("G48", "S48", "G96", "S96")},
        "H2x_gross_G96": odd_summary["H2x"]["G96"]["gross"],
        "quad": rec["quadrature_design"]["kind"],
        "H2x_top6": rec["disagreement_tiles"]["H2x"]["G96_vs_S96"]["frac_top6_of_sum_abs_dnet"],
        "H6y_top6": rec["disagreement_tiles"]["H6y"]["G96_vs_S96"]["frac_top6_of_sum_abs_dnet"],
        "even_rel_H2x": even_rel_shift["H2x"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

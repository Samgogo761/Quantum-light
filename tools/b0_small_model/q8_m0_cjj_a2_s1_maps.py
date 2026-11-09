#!/usr/bin/env python3
"""S1-v0 k-maps from saved a2 Y. No new propagation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump  # noqa: E402
from q8_m0_cjj_source_diag import (  # noqa: E402
    disagreement_tiles,
    gram_cross,
    gram_diag,
    inversion_parts,
    partner_index,
    signed_mass,
    tile_integrals,
)
from q8_m0_cjj_validate import CJJ_ABS_UNIT  # noqa: E402
from q8_m0_model import build_geometry  # noqa: E402
from q8_m0_s1_kernel import y_cartesian_to_parallel_perp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/q_line/cjj_m0_p1_a2_s1_20260917"
OUT = ROOT / "source_maps"
JOBS = (("G48", "V48", 48), ("S48", "V48S", 48))


def load_pair(name: str) -> dict:
    plus = np.load(ROOT / name / f"{name}_N+1_A+1.npz", mmap_mode="r")
    minus = np.load(ROOT / name / f"{name}_N-1_A+1.npz", mmap_mode="r")
    return {
        "Y+": np.asarray(plus["Y"]),
        "Y-": np.asarray(minus["Y"]),
        "w": np.asarray(plus["w_k"], dtype=float),
        "kf": np.asarray(plus["kf"], dtype=float),
    }


def s1_fields(yp, ym, geo) -> dict:
    yp_par, yp_perp = y_cartesian_to_parallel_perp(yp, geo)
    ym_par, ym_perp = y_cartesian_to_parallel_perp(ym, geo)
    out = {}
    for cname, ga_p, ga_m, gb_p, gb_m, cross in (
        ("H1par", yp_par[:, 0], ym_par[:, 0], None, None, False),
        ("H2perp", yp_perp[:, 1], ym_perp[:, 1], None, None, False),
        ("H1parH2perp", yp_par[:, 0], ym_par[:, 0], yp_perp[:, 1], ym_perp[:, 1], True),
    ):
        if not cross:
            gp = np.real(gram_diag(ga_p))
            gm = np.real(gram_diag(ga_m))
        else:
            gp = gram_cross(ga_p, gb_p)
            gm = gram_cross(ga_m, gb_m)
        out[cname] = {"g+": gp, "g-": gm, "go": 0.5 * (gp - gm), "ge": 0.5 * (gp + gm)}
    return out


def _heat(ax, z, title: str, diverging: bool):
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    all_jobs = {}
    for label, name, nk in JOBS:
        pair = load_pair(name)
        pidx, unpaired = partner_index(pair["kf"])
        fields = s1_fields(pair["Y+"], pair["Y-"], geo)
        job = {"nk": nk, "n_k": int(pair["kf"].shape[0]), "unpaired": unpaired, "channels": {}}
        for cname, ch in fields.items():
            job["channels"][cname] = {
                "plus": signed_mass(ch["g+"], pair["w"]),
                "minus": signed_mass(ch["g-"], pair["w"]),
                "even": signed_mass(ch["ge"], pair["w"]),
                "odd": signed_mass(ch["go"], pair["w"]),
                "inversion": inversion_parts(ch["go"], pidx, pair["w"]),
                "tiles": tile_integrals(ch["go"], pair["w"], pair["kf"]),
            }
            job["channels"][cname]["_go"] = np.asarray(ch["go"])
            job["channels"][cname]["_ge"] = np.asarray(ch["ge"])
            job["channels"][cname]["_kf"] = pair["kf"]
            job["channels"][cname]["_w"] = pair["w"]
        all_jobs[label] = job
        print(json.dumps({
            "loaded": label,
            "H1par_even": job["channels"]["H1par"]["even"]["net_re"],
            "H2perp_even": job["channels"]["H2perp"]["even"]["net_re"],
            "cross_odd_re": job["channels"]["H1parH2perp"]["odd"]["net_re"],
            "cross_odd_abs": job["channels"]["H1parH2perp"]["odd"]["net_abs"],
            "cross_even_abs": job["channels"]["H1parH2perp"]["even"]["net_abs"],
            "cross_odd_kappa": job["channels"]["H1parH2perp"]["odd"]["kappa_cancel"],
        }, indent=2), flush=True)

    disagreements = {
        cname: disagreement_tiles(all_jobs["G48"]["channels"][cname]["tiles"], all_jobs["S48"]["channels"][cname]["tiles"])
        for cname in ("H1par", "H2perp", "H1parH2perp")
    }
    slim = {}
    for lab, job in all_jobs.items():
        slim[lab] = {
            "nk": job["nk"],
            "unpaired": job["unpaired"],
            "channels": {c: {k: v for k, v in ch.items() if not k.startswith("_")} for c, ch in job["channels"].items()},
        }
    dump(OUT / "maps_summary.json", {
        "unit": CJJ_ABS_UNIT,
        "jobs": slim,
        "disagreements_G48_vs_S48": disagreements,
        "quantum_input": "not_run",
        "photon_output": "not_run",
        "note": "S1 projected Y; no new trajectories",
    })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for cname in ("H1par", "H2perp", "H1parH2perp"):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
        panels = (
            ("G48", "_ge", "g_e"),
            ("G48", "_go", "g_o"),
            ("S48", "_ge", "g_e"),
            ("S48", "_go", "g_o"),
        )
        for ax, (lab, key, tag) in zip(axes.ravel(), panels):
            z = all_jobs[lab]["channels"][cname][key]
            z = np.real(z).reshape(48, 48)
            im = _heat(ax, z, f"{cname} {tag}  {lab}", True)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"a2 S1 {cname} (not photon number). {CJJ_ABS_UNIT}")
        fig.savefig(figdir / f"{cname}_ge_go_G48_S48.png", dpi=140)
        plt.close(fig)

        if cname == "H1parH2perp":
            fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
            for ax, lab, part in zip(axes.ravel(), ("G48", "G48", "S48", "S48"), ("re", "im", "re", "im")):
                z = all_jobs[lab]["channels"][cname]["_go"]
                z = (np.real(z) if part == "re" else np.imag(z)).reshape(48, 48)
                im = _heat(ax, z, f"cross g_o {part}  {lab}", True)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(figdir / "H1parH2perp_go_reim.png", dpi=140)
            plt.close(fig)

    print(json.dumps({
        "wrote": str(OUT),
        "H1par_tile_top6_frac": disagreements["H1par"]["frac_top6_of_sum_abs_dnet"],
        "cross_tile_top6_frac": disagreements["H1parH2perp"]["frac_top6_of_sum_abs_dnet"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

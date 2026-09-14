#!/usr/bin/env python3
"""P0/P1 classical campaign for M0 (2026-09-14 execution note).

Keeps M0/R0/P0. Writes a new result tree; does not overwrite
results/q_line/m0_v1_20260912. Not quantum light.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from q8_m0_classical import (
    DriveProtocol,
    P0Protocol,
    analyse_set,
    compare_harmonics,
    field_resources,
    k_grid,
    p1_protocol,
    run_set,
    target_quantities,
)
from q8_m0_model import MODEL_ID, N_OCC, R0, build_geometry, h8

REPO = Path(__file__).resolve().parents[2]
OLD_STATIC = REPO / "results/q_line/m0_v1_20260912/static/gap_report.json"
OUT_ROOT = REPO / "results/q_line/m0_p0p1_20260914"
ORDERS = list(range(1, 10))


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class Job:
    name: str
    protocol: DriveProtocol
    nk: int
    ntpc: int
    grid_shift: float = 0.0
    subspace: bool = False
    k_orders: tuple = ()
    signs: tuple = (+1,)
    neels: tuple = (+1, -1)


def coarse_gap_check(geo, p, nk: int = 120) -> dict:
    _kf, kc, _w = k_grid(geo, nk, shift=0.0)
    e = np.linalg.eigvalsh(h8(geo, p, kc, +1))
    e4max = float(e[:, N_OCC - 1].max())
    e5min = float(e[:, N_OCC].min())
    dmin = float(np.min(e[:, N_OCC] - e[:, N_OCC - 1]))
    return {
        "nk": nk,
        "grid": "gamma_centred_MP",
        "E4_max": e4max,
        "E5_min": e5min,
        "E_g_indirect_lower_bound_on_grid": e5min - e4max,
        "E_g_direct_min_on_grid": dmin,
        "insulating_on_this_grid": bool(e5min > e4max),
        "note": "coarse full-BZ coverage check, not a 1e-10 global bound on the continuous BZ",
    }


def one_job(geo, p, mu, job: Job, verbose: bool) -> dict:
    outdir = OUT_ROOT / job.name
    runs = run_set(
        geo, p, job.protocol, mu, job.nk, job.ntpc,
        signs=job.signs, neels=job.neels, tag=job.name, outdir=outdir,
        verbose=verbose, grid_shift=job.grid_shift, subspace=job.subspace, k_orders=job.k_orders,
    )
    rec = {
        "name": job.name,
        "nk": job.nk,
        "ntpc": job.ntpc,
        "grid_shift": job.grid_shift,
        "subspace": job.subspace,
        "k_orders": list(job.k_orders),
        "protocol": asdict(job.protocol),
        "protocol_id": job.protocol.protocol_id,
        "keys": list(runs),
    }
    if "N+1_A+1" in runs and "N-1_A+1" in runs:
        ana, traces = analyse_set(runs, job.protocol, ORDERS)
        rec["analysis"] = ana
        rec["targets_sin2"] = target_quantities(ana["windows"]["sin2"])
        rec["traces_meta"] = {
            "n_times": int(len(traces[0])),
            "Je_max": float(np.max(np.abs(traces[3]))),
            "Jo_max": float(np.max(np.abs(traces[4]))),
        }
        dump(outdir / "analysis.json", rec)
        return rec, runs, traces
    dump(outdir / "analysis.json", rec)
    return rec, runs, None


def kmap_payload(runs: dict, orders=(0, 1, 2)) -> dict | None:
    if "k_harm_total" not in runs["N+1_A+1"] or "k_harm_total" not in runs["N-1_A+1"]:
        return None
    kf = np.asarray(runs["N+1_A+1"]["kf"])
    nk = int(np.sqrt(kf.shape[0]))
    stored = np.asarray(runs["N+1_A+1"]["k_orders"], dtype=int)
    out = {"nk": nk, "orders": [int(x) for x in stored], "n_exc": {}}
    for key in ("n_exc_k_peak", "n_exc_k_final"):
        if key in runs["N+1_A+1"]:
            out["n_exc"][key + "_N+"] = np.asarray(runs["N+1_A+1"][key]).reshape(nk, nk)
            out["n_exc"][key + "_N-"] = np.asarray(runs["N-1_A+1"][key]).reshape(nk, nk)
    for blk in ("total", "vv", "cc", "vc"):
        kname = f"k_harm_{blk}"
        if kname not in runs["N+1_A+1"]:
            continue
        hp = np.asarray(runs["N+1_A+1"][kname])
        hm = np.asarray(runs["N-1_A+1"][kname])
        je = 0.5 * (hp + hm)
        jo = 0.5 * (hp - hm)
        out[blk] = {}
        for i, n in enumerate(stored):
            out[blk][str(int(n))] = {
                "Je_abs": np.sqrt(np.abs(je[i, :, 0]) ** 2 + np.abs(je[i, :, 1]) ** 2).reshape(nk, nk),
                "Jo_abs": np.sqrt(np.abs(jo[i, :, 0]) ** 2 + np.abs(jo[i, :, 1]) ** 2).reshape(nk, nk),
                "Jo_y_abs": np.abs(jo[i, :, 1]).reshape(nk, nk),
                "Jo_x_abs": np.abs(jo[i, :, 0]).reshape(nk, nk),
                "Je_x_abs": np.abs(je[i, :, 0]).reshape(nk, nk),
            }
    return out


def save_kmaps(name: str, payload: dict | None) -> None:
    if payload is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = OUT_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for key, arr in payload.get("n_exc", {}).items():
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(arr.T, origin="lower", cmap="magma")
        ax.set_title(f"{name} {key}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figdir / f"{name}_{key}.png", dpi=130)
        plt.close(fig)
    for blk in ("total", "vv", "cc", "vc"):
        if blk not in payload:
            continue
        for n, pack in payload[blk].items():
            for q in ("Jo_y_abs", "Jo_x_abs", "Je_x_abs"):
                fig, ax = plt.subplots(figsize=(4.2, 3.6))
                im = ax.imshow(pack[q].T, origin="lower", cmap="viridis")
                ax.set_title(f"{name} {blk} n={n} {q} (k0)")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(figdir / f"{name}_{blk}_n{n}_{q}.png", dpi=130)
                plt.close(fig)
    slim = {
        "nk": payload["nk"],
        "orders": payload["orders"],
        "n_exc_peak_mean_N+": float(np.mean(payload["n_exc"]["n_exc_k_peak_N+"])) if "n_exc_k_peak_N+" in payload.get("n_exc", {}) else None,
        "maxima": {},
    }
    for blk in ("total", "vv", "cc", "vc"):
        if blk not in payload:
            continue
        slim["maxima"][blk] = {
            n: {q: float(np.max(pack[q])) for q in ("Jo_y_abs", "Jo_x_abs", "Je_x_abs", "Jo_abs", "Je_abs")}
            for n, pack in payload[blk].items()
        }
    dump(OUT_ROOT / name / "kmap_summary.json", slim)


def traces_figure(name: str, prot: DriveProtocol, traces) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, _jp, _jm, je, jo = traces
    figdir = OUT_ROOT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tt = times / prot.period
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    axs[0].plot(tt, je[:, 0], label=r"$J_{e,x}$")
    axs[0].plot(tt, je[:, 1], label=r"$J_{e,y}$", alpha=0.8)
    axs[0].set_ylabel(r"$J_e$")
    axs[0].legend(fontsize=8)
    axs[1].plot(tt, jo[:, 0], label=r"$J_{o,x}$", color="C3")
    axs[1].plot(tt, jo[:, 1], label=r"$J_{o,y}$", color="C2")
    axs[1].set_ylabel(r"$J_o$")
    axs[1].set_xlabel(r"$t/T$")
    axs[1].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(figdir / f"{name}_traces.png", dpi=140)
    plt.close(fig)


def jobs_for(args) -> list[Job]:
    p0 = P0Protocol()
    smoke = [
        Job("smoke_p1_A0.8", p1_protocol(0.8), 8, 40, 0.0, True, (0, 1, 2), (+1,), (+1, -1)),
    ]
    if args.smoke:
        return smoke
    phase1 = [
        Job("p1_A0.8_nk48_nt400", p1_protocol(0.8), 48, 400, 0.0, True, (0, 1, 2), (+1,), (+1, -1)),
        Job("p1_A0.4_nk48_nt400", p1_protocol(0.4), 48, 400, 0.0, True, (), (+1,), (+1, -1)),
        Job("p1_A1.6_nk48_nt400", p1_protocol(1.6), 48, 400, 0.0, True, (), (+1,), (+1, -1)),
        Job("p0_nk48_nt400", p0, 48, 400, 0.0, True, (0, 1, 2), (+1,), (+1, -1)),
        Job("p0_nk48_nt400_shift", p0, 48, 400, 0.5, False, (), (+1,), (+1, -1)),
        Job("p0_nk48_nt800", p0, 48, 800, 0.0, False, (), (+1,), (+1, -1)),
        Job("p1_A0.8_nk48_nt800", p1_protocol(0.8), 48, 800, 0.0, False, (), (+1,), (+1, -1)),
        Job("p1_A0.8_rev_nk24", p1_protocol(0.8), 24, 200, 0.0, False, (), (+1, -1), (+1, -1)),
    ]
    phase2 = [
        Job("p0_nk96_nt400", p0, 96, 400, 0.0, False, (), (+1,), (+1, -1)),
        Job("p1_A0.8_nk96_nt400", p1_protocol(0.8), 96, 400, 0.0, False, (), (+1,), (+1, -1)),
    ]
    if args.phase == "pilot":
        return phase1
    if args.phase == "refine":
        return phase2
    return phase1 + phase2


def summarize_campaign(geo, gap, resources, records: list[dict]) -> dict:
    by = {r["name"]: r for r in records}
    p0_archival = {
        "source": "results/q_line/m0_v1_20260912/classical/classical_report.json",
        "note": "original NPZ not retained in tree; JSON is the archival P0 record. Partial k-convergence remains.",
        "targets_nk192_sin2": {
            "Jo_y_0": 2.3396423462554736e-05,
            "Jo_y_H2": 2.760e-6,
            "Jo_x_H2": 1.766e-6,
            "Je_x_H1": 0.00466186343107176,
        },
    }

    def tgt(name):
        rec = by.get(name)
        if not rec or "targets_sin2" not in rec:
            return None
        return rec["targets_sin2"]

    envelope = {}
    pairs = [
        ("p0_gamma_vs_shift_nk48", "p0_nk48_nt400", "p0_nk48_nt400_shift"),
        ("p0_dt_nk48_400_vs_800", "p0_nk48_nt400", "p0_nk48_nt800"),
        ("p0_k_48_vs_96", "p0_nk48_nt400", "p0_nk96_nt400"),
        ("p1_A0.8_dt_nk48_400_vs_800", "p1_A0.8_nk48_nt400", "p1_A0.8_nk48_nt800"),
        ("p1_A0.8_k_48_vs_96", "p1_A0.8_nk48_nt400", "p1_A0.8_nk96_nt400"),
    ]
    for label, a, b in pairs:
        if a in by and b in by and "analysis" in by[a] and "analysis" in by[b]:
            envelope[label] = {
                "targets_a": tgt(a),
                "targets_b": tgt(b),
                "harmonics_sin2": compare_harmonics(by[a]["analysis"]["windows"]["sin2"], by[b]["analysis"]["windows"]["sin2"], ORDERS),
            }
    return {
        "model_id": MODEL_ID,
        "stage": "P0_P1_classical_campaign",
        "quantum_input": "not_run",
        "quantum_output": "not_run",
        "gap": {
            "E_g_direct": gap["E_g_direct"],
            "E_g_indirect": gap["E_g_indirect"],
            "mu": gap["mu_half_filling"],
            "uncertainty_is": "last-stage local extremum change, not a continuous-BZ strict bound",
        },
        "resources": resources,
        "p0_archival_nk192": p0_archival,
        "jobs": {r["name"]: {k: r[k] for k in r if k not in ("analysis",)} | {"has_analysis": "analysis" in r} for r in records},
        "empirical_error_envelope": envelope,
        "not_a_pass_criterion": "non-zero Jo, plateau, or maximum harmonic intensity were not required",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phase", choices=("all", "pilot", "refine"), default="all")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    p = R0
    gap = json.loads(OLD_STATIC.read_text(encoding="utf-8"))
    if not gap["insulating_initial_state_supported"]:
        raise SystemExit("static gap check did not support the insulating initial state")
    mu = gap["mu_half_filling"]
    eg = float(gap["E_g_direct"])
    gap_chk = coarse_gap_check(geo, p, nk=80 if args.smoke else 160)
    dump(OUT_ROOT / "gap_coarse_check.json", {"stored": {"E_g_direct": gap["E_g_direct"], "E_g_indirect": gap["E_g_indirect"]}, "coarse": gap_chk})
    resources = {
        "P0": field_resources(geo, P0Protocol(), eg),
        "P1_A0.4": field_resources(geo, p1_protocol(0.4), eg),
        "P1_A0.8": field_resources(geo, p1_protocol(0.8), eg),
        "P1_A1.6": field_resources(geo, p1_protocol(1.6), eg),
    }
    fe0 = resources["P0"]["F_E"]
    for key, rec in resources.items():
        rec["F_E_over_P0"] = rec["F_E"] / fe0
        rec["E_peak_over_P0"] = rec["eE_peak_over_hbar_inv_ang_th"] / resources["P0"]["eE_peak_over_hbar_inv_ang_th"]
        rec["Delta_k_over_P0"] = rec["Delta_k_inv_ang"] / resources["P0"]["Delta_k_inv_ang"]
    dump(OUT_ROOT / "field_resources.json", resources)
    records = []
    verbose = not args.quiet
    for job in jobs_for(args):
        print(f"==== {job.name} ====", flush=True)
        rec, runs, traces = one_job(geo, p, mu, job, verbose)
        records.append(rec)
        if traces is not None:
            traces_figure(job.name, job.protocol, traces)
        if rec.get("subspace") or rec.get("k_orders"):
            save_kmaps(job.name, kmap_payload(runs))
        dump(OUT_ROOT / "campaign_partial.json", {"completed": [r["name"] for r in records]})
    summary = summarize_campaign(geo, gap, resources, records)
    dump(OUT_ROOT / "campaign_summary.json", summary)
    print(json.dumps({"completed": [r["name"] for r in records], "out": str(OUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""k-resolved Born A2 proxy on the Stage-1G coarse grids. No new physics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import dump, propagate_cjj  # noqa: E402
from q8_m0_classical import k_grid, p1_a2_protocol  # noqa: E402
from q8_m0_mode_io import eta2_ncell  # noqa: E402
from q8_m0_model import R0, build_geometry  # noqa: E402
from q8_m0_runaway import k_sets  # noqa: E402
from q8_m0_stage1g_bzqual import h2_seed, perk_a2_state  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/stage1g_bzqual_20260922"
MU = -0.009010620372198276
NTPC = 400
EPS = 3.0e-5


def one(geo, prot, name, kf):
    print(f"[kmap] {name} nk={len(kf)}", flush=True)
    nt = prot.n_cycles * NTPC
    dt = prot.period / NTPC
    times = np.arange(nt + 1) * dt
    a_in, hk = h2_seed(times, prot, geo, EPS)
    r0 = propagate_cjj(
        geo, R0, prot, nt_per_cycle=NTPC, neel=+1, sign=+1, mu=MU,
        verbose=False, kf_frac=kf, save_full_x=True, save_mean_D=True,
    )
    pref = eta2_ncell(geo)
    rows = perk_a2_state(
        r0["X_full"], r0["mean_D_t"], hk["F"], hk["e_perp"], hk["k_a"], dt, pref,
    )
    kf = np.asarray(kf, dtype=float)
    re = np.array([p["A2_state_proxy"]["re"] for p in rows])
    im = np.array([p["A2_state_proxy"]["im"] for p in rows])
    ab = np.array([p["A2_state_proxy"]["abs"] for p in rows])
    rec = {
        "name": name,
        "n_k": int(len(kf)),
        "kf": kf.tolist(),
        "A2_re": re.tolist(),
        "A2_im": im.tolist(),
        "A2_abs": ab.tolist(),
        "mean_re": float(np.mean(re)),
        "mean_abs": float(np.mean(ab)),
        "pos_frac": float(np.mean(re > 0.0)),
        "note": "one-hot R[Re F*]; contact uses mean D. Proxy, not production A2.",
    }
    np.savez(OUT / f"{name}_kmap.npz", kf=kf, A2_re=re, A2_im=im, A2_abs=ab)
    return rec


def figure(maps: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0), sharex=True, sharey=True)
    for ax, rec in zip(axes.ravel(), maps):
        kf = np.asarray(rec["kf"])
        re = np.asarray(rec["A2_re"])
        sc = ax.scatter(kf[:, 0], kf[:, 1], c=re, cmap="coolwarm", s=55, vmin=-0.4, vmax=0.4)
        ax.set_title(f"{rec['name']}  mean Re={rec['mean_re']:.3f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="Re A2 proxy")
    fig.suptitle("Stage-1G k-resolved Born A2 proxy (not a 48/96/192 run)", fontsize=11)
    fig.savefig(OUT / "stage1g_kmap_reA2.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_a2_protocol(0.8)
    probe = k_sets(geo, R0, MU)["probe6"]
    jobs = [
        ("probe6", probe["kf"]),
        ("mp4", k_grid(geo, 4, shift=0.0)[0]),
        ("mp4_shift", k_grid(geo, 4, shift=0.5)[0]),
        ("mp6", k_grid(geo, 6, shift=0.0)[0]),
    ]
    maps = [one(geo, prot, n, kf) for n, kf in jobs]
    figure(maps)
    dump(OUT / "stage1g_kmap.json", {
        "maps": [{k: v for k, v in m.items() if k != "kf"} | {"n_k": m["n_k"], "mean_re": m["mean_re"], "mean_abs": m["mean_abs"], "pos_frac": m["pos_frac"]} for m in maps],
        "production_V": "Born_reference_diagnostic_not_qualified",
        "nk96": "not_run",
    })
    print({m["name"]: {"mean_re": m["mean_re"], "pos_frac": m["pos_frac"], "mean_abs": m["mean_abs"]} for m in maps})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-cost residual ranking for chunked-vs-full compare (no gate changes).

Reports absolute and strong-signal-normalized errors:
  eps_abs_norm = |A-B| / max_|B|_ref
where the reference max is taken over the same harmonic (modes) or the full
continuum (ICS/CS). Does NOT modify compare thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_modes(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10:
            raise ValueError(f"{path}: expected 10 columns")
        key = (int(f[0]), int(f[4]))
        out[key] = {
            "weight": float(f[1]),
            "intensity": float(f[2]),
            "phase": float(f[3]),
            "jx": complex(float(f[5]), float(f[6])),
            "jy": complex(float(f[7]), float(f[8])),
            "power": float(f[9]),
        }
    return out


def load_continuum(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        h, omega, ics, cs = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
        iw = len(rows) + 1
        rows.append({"iw": iw, "h_order": h, "omega": omega, "ICS": ics, "CS": cs})
    return rows


def phase_abs(a: float, b: float) -> float:
    e = abs(a - b)
    return min(e, 2.0 * math.pi - e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--full-run-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    modes_m = load_modes(args.merged / "HHG_nodes_modes.dat")
    modes_f = load_modes(args.full_run_dir / "HHG_nodes_modes.dat")
    keys = sorted(set(modes_m) & set(modes_f))

    # Per-harmonic strong-signal refs from FULL run
    harm_max: dict[int, dict[str, float]] = {}
    for (_nid, h), mf in modes_f.items():
        bucket = harm_max.setdefault(
            h,
            {"amp_j": 0.0, "power": 0.0, "intensity": 0.0, "weight": 0.0},
        )
        amp = math.hypot(abs(mf["jx"]), abs(mf["jy"]))
        bucket["amp_j"] = max(bucket["amp_j"], amp)
        bucket["power"] = max(bucket["power"], abs(mf["power"]))
        bucket["intensity"] = max(bucket["intensity"], abs(mf["intensity"]))
        bucket["weight"] = max(bucket["weight"], abs(mf["weight"]))

    mode_rows: list[dict] = []
    for key in keys:
        mm, mf = modes_m[key], modes_f[key]
        nid, h = key
        href = harm_max[h]
        # scalar fields
        for field in ("weight", "intensity", "power"):
            a, b = float(mm[field]), float(mf[field])
            abs_err = abs(a - b)
            rel = abs_err / max(abs(a), abs(b), 1.0e-300)
            denom = max(href[field if field != "weight" else "weight"], 1.0e-300)
            if field == "power":
                denom = max(href["power"], 1.0e-300)
            mode_rows.append(
                {
                    "kind": "mode_scalar",
                    "node": nid,
                    "harmonic": h,
                    "field": field,
                    "merged": a,
                    "full": b,
                    "abs_err": abs_err,
                    "rel_err_pairwise": rel,
                    "eps_abs_norm": abs_err / denom,
                    "ref_max_same_h": denom,
                }
            )
        # phase: absolute on circle; norm by pi (not amp)
        dphi = phase_abs(float(mm["phase"]), float(mf["phase"]))
        mode_rows.append(
            {
                "kind": "mode_phase",
                "node": nid,
                "harmonic": h,
                "field": "phase",
                "merged": float(mm["phase"]),
                "full": float(mf["phase"]),
                "abs_err": dphi,
                "rel_err_pairwise": dphi / max(abs(mf["phase"]), 1.0e-300),
                "eps_abs_norm": dphi / math.pi,
                "ref_max_same_h": math.pi,
            }
        )
        for field in ("jx", "jy"):
            a, b = mm[field], mf[field]
            abs_err = abs(a - b)
            rel = abs_err / max(abs(a), abs(b), 1.0e-300)
            denom = max(href["amp_j"], 1.0e-300)
            mode_rows.append(
                {
                    "kind": "mode_complex",
                    "node": nid,
                    "harmonic": h,
                    "field": field,
                    "merged_re": a.real,
                    "merged_im": a.imag,
                    "full_re": b.real,
                    "full_im": b.imag,
                    "merged_abs": abs(a),
                    "full_abs": abs(b),
                    "abs_err": abs_err,
                    "rel_err_pairwise": rel,
                    "eps_abs_norm": abs_err / denom,
                    "ref_max_amp_j_same_h": denom,
                }
            )

    mode_by_pair = sorted(mode_rows, key=lambda r: r["rel_err_pairwise"], reverse=True)
    mode_by_norm = sorted(mode_rows, key=lambda r: r["eps_abs_norm"], reverse=True)

    cont_m = load_continuum(args.merged / "HHG_ics_cs.dat")
    cont_f = load_continuum(args.full_run_dir / "HHG_ics_cs.dat")
    n = min(len(cont_m), len(cont_f))
    max_ics = max(abs(r["ICS"]) for r in cont_f) if cont_f else 0.0
    max_cs = max(abs(r["CS"]) for r in cont_f) if cont_f else 0.0

    cont_rows: list[dict] = []
    for i in range(n):
        a, b = cont_m[i], cont_f[i]
        for field, refmax in (("ICS", max_ics), ("CS", max_cs)):
            va, vb = float(a[field]), float(b[field])
            abs_err = abs(va - vb)
            rel = abs_err / max(abs(va), abs(vb), 1.0e-300)
            cont_rows.append(
                {
                    "iw": a["iw"],
                    "h_order": b["h_order"],
                    "omega": b["omega"],
                    "field": field,
                    "merged": va,
                    "full": vb,
                    "abs_err": abs_err,
                    "rel_err_pairwise": rel,
                    "eps_abs_norm": abs_err / max(refmax, 1.0e-300),
                    "ref_max_full_spectrum": refmax,
                    "weak_vs_spectrum": abs(vb) < 1.0e-6 * max(refmax, 1.0e-300),
                }
            )

    cont_by_pair = sorted(cont_rows, key=lambda r: r["rel_err_pairwise"], reverse=True)
    cont_by_norm = sorted(cont_rows, key=lambda r: r["eps_abs_norm"], reverse=True)

    # Highlight the compare-reported maxima
    max_mode_pair = mode_by_pair[0] if mode_by_pair else None
    max_cs_pair = next(r for r in cont_by_pair if r["field"] == "CS")

    # Classification helpers
    weak_mode_pair = (
        max_mode_pair is not None
        and max_mode_pair.get("full_abs", max_mode_pair.get("full", 0.0))
        < 1.0e-6 * max_mode_pair.get("ref_max_amp_j_same_h", max_mode_pair.get("ref_max_same_h", 1.0))
    )
    # For scalar power on node5, use power abs vs harm max
    if max_mode_pair and max_mode_pair["field"] in ("power", "jx", "jy"):
        if max_mode_pair["field"] == "power":
            weak_mode_pair = abs(float(max_mode_pair["full"])) < 1.0e-6 * float(
                max_mode_pair["ref_max_same_h"]
            )
        else:
            weak_mode_pair = float(max_mode_pair["full_abs"]) < 1.0e-6 * float(
                max_mode_pair["ref_max_amp_j_same_h"]
            )

    report = {
        "merged": str(args.merged.resolve()),
        "full_run_dir": str(args.full_run_dir.resolve()),
        "note": (
            "Diagnostic only. Pairwise rel_err is what compare_chunked_vs_full uses; "
            "eps_abs_norm normalizes by strong-signal max on same harmonic / full spectrum. "
            "FAIL mixes chunking effects with source/binary differences vs 27992."
        ),
        "headline": {
            "max_mode_rel_err_entry": max_mode_pair,
            "max_mode_classified_weak_vs_same_h": weak_mode_pair,
            "max_CS_rel_err_entry": max_cs_pair,
            "max_CS_classified_weak_vs_spectrum": bool(max_cs_pair["weak_vs_spectrum"]),
        },
        "mode_top_by_pairwise_rel": mode_by_pair[: args.top],
        "mode_top_by_eps_abs_norm": mode_by_norm[: args.top],
        "continuum_top_by_pairwise_rel": cont_by_pair[: args.top],
        "continuum_top_by_eps_abs_norm": cont_by_norm[: args.top],
        "summary_stats": {
            "n_mode_entries": len(mode_rows),
            "n_continuum_entries": len(cont_rows),
            "max_mode_rel_err": mode_by_pair[0]["rel_err_pairwise"] if mode_by_pair else None,
            "max_mode_eps_abs_norm": mode_by_norm[0]["eps_abs_norm"] if mode_by_norm else None,
            "max_CS_rel_err": max(
                (r["rel_err_pairwise"] for r in cont_rows if r["field"] == "CS"), default=None
            ),
            "max_CS_eps_abs_norm": max(
                (r["eps_abs_norm"] for r in cont_rows if r["field"] == "CS"), default=None
            ),
            "max_ICS_eps_abs_norm": max(
                (r["eps_abs_norm"] for r in cont_rows if r["field"] == "ICS"), default=None
            ),
            "spectrum_max_ICS": max_ics,
            "spectrum_max_CS": max_cs,
        },
        "verdict_hint": (
            "If max pairwise rel errors sit on weak node5 / weak CS bins while "
            "eps_abs_norm is tiny, record as historically highly consistent but NOT "
            "strict chunk equivalence. Strict isolation needs same-binary unchunked control."
        ),
    }

    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")

    # Human-readable summary
    print("=== HEADLINE: max pairwise mode rel_err ===")
    e = max_mode_pair
    if e:
        print(
            f"  node={e['node']} H={e['harmonic']} field={e['field']}\n"
            f"  pairwise_rel={e['rel_err_pairwise']:.6e}  abs_err={e['abs_err']:.6e}  "
            f"eps_abs_norm={e['eps_abs_norm']:.6e}\n"
            f"  weak_vs_same_h={weak_mode_pair}"
        )
        if e["field"] in ("jx", "jy"):
            print(
                f"  merged={e['merged_re']:.16e}{e['merged_im']:+.16e}j\n"
                f"  full  ={e['full_re']:.16e}{e['full_im']:+.16e}j\n"
                f"  |merged|={e['merged_abs']:.6e} |full|={e['full_abs']:.6e} "
                f"ref_max_|J|_H={e['ref_max_amp_j_same_h']:.6e}"
            )
        else:
            print(f"  merged={e['merged']!r} full={e['full']!r} ref_max={e.get('ref_max_same_h')}")

    print("\n=== HEADLINE: max pairwise CS rel_err ===")
    c = max_cs_pair
    print(
        f"  iw={c['iw']} h_order={c['h_order']:.6g} omega={c['omega']:.6e}\n"
        f"  merged_CS={c['merged']:.6e} full_CS={c['full']:.6e}\n"
        f"  abs_err={c['abs_err']:.6e} pairwise_rel={c['rel_err_pairwise']:.6e}\n"
        f"  eps_abs_norm={c['eps_abs_norm']:.6e}  weak_vs_spectrum={c['weak_vs_spectrum']}\n"
        f"  spectrum_max_CS={c['ref_max_full_spectrum']:.6e}"
    )

    print("\n=== TOP 10 modes by pairwise rel_err ===")
    for r in mode_by_pair[:10]:
        print(
            f"  n{r['node']}/H{r['harmonic']}/{r['field']}: "
            f"rel={r['rel_err_pairwise']:.3e} abs={r['abs_err']:.3e} "
            f"norm={r['eps_abs_norm']:.3e}"
        )
    print("\n=== TOP 10 modes by eps_abs_norm ===")
    for r in mode_by_norm[:10]:
        print(
            f"  n{r['node']}/H{r['harmonic']}/{r['field']}: "
            f"norm={r['eps_abs_norm']:.3e} abs={r['abs_err']:.3e} "
            f"rel={r['rel_err_pairwise']:.3e}"
        )
    print("\n=== TOP 10 continuum by pairwise rel_err ===")
    for r in cont_by_pair[:10]:
        print(
            f"  iw={r['iw']} h={r['h_order']:.4g} {r['field']}: "
            f"rel={r['rel_err_pairwise']:.3e} abs={r['abs_err']:.3e} "
            f"norm={r['eps_abs_norm']:.3e} weak={r['weak_vs_spectrum']}"
        )
    print("\n=== TOP 10 continuum by eps_abs_norm ===")
    for r in cont_by_norm[:10]:
        print(
            f"  iw={r['iw']} h={r['h_order']:.4g} {r['field']}: "
            f"norm={r['eps_abs_norm']:.3e} abs={r['abs_err']:.3e} "
            f"rel={r['rel_err_pairwise']:.3e} weak={r['weak_vs_spectrum']}"
        )

    ss = report["summary_stats"]
    print("\n=== SUMMARY ===")
    print(f"  max_mode_rel_err     = {ss['max_mode_rel_err']:.6e}")
    print(f"  max_mode_eps_abs_norm= {ss['max_mode_eps_abs_norm']:.6e}")
    print(f"  max_CS_rel_err       = {ss['max_CS_rel_err']:.6e}")
    print(f"  max_CS_eps_abs_norm  = {ss['max_CS_eps_abs_norm']:.6e}")
    if args.report:
        print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

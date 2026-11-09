#!/usr/bin/env python3
"""Zero-cost GH5 quadrature diagnostics. Does NOT retune or overwrite v1=FAIL.

Reports:
  - full complex Jones (amp / global phase / polarization)
  - A22^o, F22^o odd-channel moments (A1, not A0 mean current)
  - GH5 outer-ring weight and ICS share
  - Jo cancellation factor
  - Je as Neel-even mean: absolute zero-channel vs ||Jo|| or sqrt(A22)
  - GH5 25-node CEP-π V2 archive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_full112_gh3_candidates import HARD_ORDERS, load_manifest, pair_nodes_by_neg_alpha
from audit_full112_gh3_candidates import load_node_modes as load_modes_audit
from cep_pi_gate_v2 import CEP_V2_ABS_ATOL, CEP_V2_ETA, CEP_V2_REL_ATOL, evaluate_pair_harmonic, scale_A_n
from gh3_gh5_quadrature_gate import GATE_VERSION, STATES, ensemble_current
from hhg_fft_utils import parse_input_nml, pick_ics_cs_at_order
from rank_full112_gh3_candidates import amp, load_node_modes

ALL_ORDERS = (2, 5, 7, 9, 10)
DIAG_VERSION = "gh3_gh5_quadrature_diag_v1_20260816"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v1_report(path: Path, gh3_root: Path, gh5_root: Path) -> dict:
    """Read the frozen v1 gate report. Does not retune it."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    gate = raw.get("gate")
    status = raw.get("status")
    if gate != GATE_VERSION:
        raise ValueError(f"v1 gate mismatch: {gate!r} != {GATE_VERSION!r}")
    if status != "FAIL":
        raise ValueError(
            f"v1 status must remain FAIL (got {status!r}); do not retune or overwrite"
        )
    v1_gh3 = Path(str(raw.get("gh3_root", ""))).resolve()
    v1_gh5 = Path(str(raw.get("gh5_root", ""))).resolve()
    got_gh3 = gh3_root.resolve()
    got_gh5 = gh5_root.resolve()
    if v1_gh3 != got_gh3:
        raise ValueError(f"v1 gh3_root {v1_gh3} != {got_gh3}")
    if v1_gh5 != got_gh5:
        raise ValueError(f"v1 gh5_root {v1_gh5} != {got_gh5}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "gate": gate,
        "status": status,
        "gh3_root": str(v1_gh3),
        "gh5_root": str(v1_gh5),
        "roots_match": True,
    }


def jones_decomp(jx_a: complex, jy_a: complex, jx_b: complex, jy_b: complex) -> dict:
    na = amp(jx_a, jy_a)
    nb = amp(jx_b, jy_b)
    d_vec = amp(jx_b - jx_a, jy_b - jy_a)
    inner = (jx_a.conjugate() * jx_b) + (jy_a.conjugate() * jy_b)
    if abs(inner) == 0.0:
        dphi = 0.0
    else:
        dphi = math.atan2(inner.imag, inner.real)
    phase = abs(dphi)
    phase = min(phase, 2.0 * math.pi - phase)
    align = complex(math.cos(-dphi), math.sin(-dphi))
    jx_ba, jy_ba = jx_b * align, jy_b * align
    if na > 0.0 and nb > 0.0:
        ua_x, ua_y = jx_a / na, jy_a / na
        ub_x, ub_y = jx_ba / nb, jy_ba / nb
        pol = amp(ub_x - ua_x, ub_y - ua_y)
    else:
        pol = 0.0
    return {
        "amp_a": na,
        "amp_b": nb,
        "amp_rel": abs(nb - na) / max(na, 1.0e-300),
        "phase_rad": phase,
        "vec_rel": d_vec / max(na, 1.0e-300),
        "pol_chord": pol,
    }


def odd_even_node(
    plus: dict, minus: dict, nid: int, order: int
) -> tuple[complex, complex, complex, complex, float]:
    jxp, jyp, w = plus[(nid, order)]
    jxm, jym, _ = minus[(nid, order)]
    jox = 0.5 * (jxp - jxm)
    joy = 0.5 * (jyp - jym)
    jex = 0.5 * (jxp + jxm)
    jey = 0.5 * (jyp + jym)
    return jox, joy, jex, jey, w


def outer_ring_ids(n_side: int) -> list[int]:
    ids = []
    for ix in range(1, n_side + 1):
        for iy in range(1, n_side + 1):
            if ix in (1, n_side) or iy in (1, n_side):
                ids.append((ix - 1) * n_side + iy)
    return ids


def diagnose_state(stem: str, gh3_root: Path, gh5_root: Path, man5: Path) -> dict:
    g3p, g3m = gh3_root / f"{stem}_plusN", gh3_root / f"{stem}_minusN"
    g5p, g5m = gh5_root / f"{stem}_plusN", gh5_root / f"{stem}_minusN"
    grid = parse_input_nml(g3p / "input.nml")
    dt = float(grid["dt_au"])
    nt = int(grid["nt"])
    omega0 = float(grid["omega0"])

    m3p = load_node_modes(g3p / "HHG_nodes_modes.dat")
    m3m = load_node_modes(g3m / "HHG_nodes_modes.dat")
    m5p = load_node_modes(g5p / "HHG_nodes_modes.dat")
    m5m = load_node_modes(g5m / "HHG_nodes_modes.dat")
    man = load_manifest(man5)
    by_id = {int(r["id"]): r for r in man}
    n_side = int(round(math.sqrt(len(man))))
    outer = set(outer_ring_ids(n_side))
    w_outer = sum(float(by_id[i]["weight"]) for i in outer)
    w_all = sum(float(r["weight"]) for r in man)

    ids5 = sorted({nid for nid, _o in m5p})
    ids3 = sorted({nid for nid, _o in m3p})

    jones = {}
    moments = {}
    tail = {}
    cancel = {}
    je_zero = {}
    for order in ALL_ORDERS:
        jox3, joy3, _ = ensemble_current(m3p, m3m, order, odd=True)
        jox5, joy5, _ = ensemble_current(m5p, m5m, order, odd=True)
        decomp = jones_decomp(jox3, joy3, jox5, joy5)
        ics3 = pick_ics_cs_at_order(g3p / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)["ICS"]
        ics5 = pick_ics_cs_at_order(g5p / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)["ICS"]
        den_sym = max(abs(ics3), abs(ics5), 1.0e-300)
        jones[str(order)] = {
            **decomp,
            "ICS_rel_vs_gh3": abs(ics5 - ics3) / max(abs(ics3), 1.0e-300),
            "ICS_rel_symmetric": abs(ics5 - ics3) / den_sym,
            "ICS_gh3": ics3,
            "ICS_gh5": ics5,
        }

        a22_3 = a22_5 = f22_3 = f22_5 = 0.0
        l1_jo = 0.0
        ics_outer = ics_tot = 0.0
        a_je = 0.0
        node_ics: list[dict] = []
        for nid in ids5:
            jox, joy, jex, jey, w = odd_even_node(m5p, m5m, nid, order)
            n_o = amp(jox, joy)
            a22_5 += w * n_o ** 2
            f22_5 += w * n_o ** 4
            l1_jo += w * n_o
            a_je += w * amp(jex, jey)
            jxp, jyp, wp = m5p[(nid, order)]
            pwr = abs(jxp) ** 2 + abs(jyp) ** 2
            contrib = wp * pwr
            ics_tot += contrib
            if nid in outer:
                ics_outer += contrib
            node_ics.append(
                {
                    "id": nid,
                    "weight": w,
                    "intensity": float(by_id[nid]["intensity"]),
                    "outer": nid in outer,
                    "ics_contrib": contrib,
                }
            )
        jo_bar = amp(jox5, joy5)
        cancel_f = jo_bar / max(l1_jo, 1.0e-300)
        node_ics.sort(key=lambda r: r["intensity"], reverse=True)
        cum = 0.0
        for row in node_ics:
            share = row["ics_contrib"] / max(ics_tot, 1.0e-300)
            cum += share
            row["ics_share"] = share
            row["ics_cum_by_I"] = cum

        for nid in ids3:
            jox, joy, _jex, _jey, w = odd_even_node(m3p, m3m, nid, order)
            n_o = amp(jox, joy)
            a22_3 += w * n_o ** 2
            f22_3 += w * n_o ** 4

        moments[str(order)] = {
            "A22o_gh3": a22_3,
            "A22o_gh5": a22_5,
            "A22o_rel": (a22_5 - a22_3) / max(abs(a22_3), 1.0e-300),
            "F22o_gh3": f22_3,
            "F22o_gh5": f22_5,
            "F22o_rel": (f22_5 - f22_3) / max(abs(f22_3), 1.0e-300),
        }
        tail[str(order)] = {
            "outer_ics_share": ics_outer / max(ics_tot, 1.0e-300),
            "ics_nodes_plus": ics_tot,
            "cum_by_intensity": node_ics,
        }
        cancel[str(order)] = cancel_f
        je_bar = amp(*ensemble_current(m5p, m5m, order, odd=False)[:2])
        scale = max(jo_bar, math.sqrt(max(a22_5, 0.0)), 1.0e-300)
        je_zero[str(order)] = {
            "Je_bar": je_bar,
            "mean_abs_Je_s": a_je,
            "scale_Jo_or_sqrtA": scale,
            "Je_over_scale": je_bar / scale,
            "note": "Neel-even ensemble mean; not a CEP-even label; not a 5% rel gate",
        }

    # CEP-π on GH5 25 nodes (Jones only; no metadata compare)
    modes_p = load_modes_audit(g5p / "HHG_nodes_modes.dat")
    modes_m = load_modes_audit(g5m / "HHG_nodes_modes.dat")
    pairs = pair_nodes_by_neg_alpha(man)
    cep_by = {}
    max_e_abs = max_e_rel = 0.0
    for order in ALL_ORDERS:
        A_n = scale_A_n(modes_p, modes_m, pairs, order)
        eabs = erel = 0.0
        for pr in pairs:
            dirs = [(int(pr["id_alpha"]), int(pr["id_neg"]))]
            if pr["kind"] != "self_alpha0":
                dirs.append((int(pr["id_neg"]), int(pr["id_alpha"])))
            for id_p, id_m in dirs:
                res = evaluate_pair_harmonic(
                    jx_p=modes_p[(id_p, order)]["jx"],
                    jy_p=modes_p[(id_p, order)]["jy"],
                    jx_m=modes_m[(id_m, order)]["jx"],
                    jy_m=modes_m[(id_m, order)]["jy"],
                    A_n=A_n,
                    abs_atol=CEP_V2_ABS_ATOL,
                    rel_atol=CEP_V2_REL_ATOL,
                    eta=CEP_V2_ETA,
                )
                eabs = max(eabs, float(res["e_abs"]))
                if res["strong"]:
                    erel = max(erel, float(res["e_rel"]))
        cep_by[str(order)] = {"max_e_abs": eabs, "max_e_rel_strong": erel, "A_n": A_n}
        if order in HARD_ORDERS:
            max_e_abs = max(max_e_abs, eabs)
            max_e_rel = max(max_e_rel, erel)

    return {
        "state": stem,
        "n_gh5": len(ids5),
        "n_gh3": len(ids3),
        "outer_ring_ids": sorted(outer),
        "outer_weight": w_outer,
        "outer_weight_frac": w_outer / max(w_all, 1.0e-300),
        "jones": jones,
        "a1_moments": moments,
        "tail_ics": tail,
        "Jo_cancellation": cancel,
        "Je_neel_even": je_zero,
        "cep_pi_v2_gh5": {
            "max_e_abs_hard": max_e_abs,
            "max_e_rel_strong_hard": max_e_rel,
            "pass": (max_e_abs <= CEP_V2_ABS_ATOL) and (max_e_rel <= CEP_V2_REL_ATOL),
            "scope": "Jones_only",
            "metadata_provenance_rechecked": False,
            "not_a_full_production_audit": True,
            "by_order": cep_by,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gh3-root", type=Path, required=True)
    ap.add_argument("--gh5-root", type=Path, required=True)
    ap.add_argument("--nodes-dir", type=Path, required=True)
    ap.add_argument("--v1-report", type=Path, required=True, help="frozen v1 JSON; status must be FAIL")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    v1 = load_v1_report(args.v1_report, args.gh3_root, args.gh5_root)
    man_map = {
        "sv_r2p5_th000": args.nodes_dir / "nodes_sv_r2p5_th0_gh5.dat",
        "sv_r2p5_th180": args.nodes_dir / "nodes_sv_r2p5_th180_gh5.dat",
    }
    states = [diagnose_state(s, args.gh3_root, args.gh5_root, man_map[s]) for s in STATES]
    cep_abs = max(s["cep_pi_v2_gh5"]["max_e_abs_hard"] for s in states)
    cep_rel = max(s["cep_pi_v2_gh5"]["max_e_rel_strong_hard"] for s in states)
    headline = []
    for s in states:
        h2 = s["jones"]["2"]
        m2 = s["a1_moments"]["2"]
        headline.append(
            {
                "state": s["state"],
                "outer_weight_frac": s["outer_weight_frac"],
                "H2_amp_rel": h2["amp_rel"],
                "H2_phase_rad": h2["phase_rad"],
                "H2_vec_rel": h2["vec_rel"],
                "H2_pol_chord": h2["pol_chord"],
                "H2_A22o_rel": m2["A22o_rel"],
                "H2_F22o_rel": m2["F22o_rel"],
                "outer_ics_share": {o: s["tail_ics"][o]["outer_ics_share"] for o in s["tail_ics"]},
                "Jo_cancellation": s["Jo_cancellation"],
            }
        )
    report = {
        "diag": DIAG_VERSION,
        "v1_verified": v1,
        "v1_gate": v1["gate"],
        "v1_status": v1["status"],
        "v1_status_permanent": True,
        "label": "GH5 estimate / quadrature not closed",
        "a1_quantitative": "NOT_FROZEN",
        "fft_note": "nearest integer harmonic bin only; not a finite-bandwidth probe",
        "safe_claim": (
            "H2 magnetically-odd mean-current amplitude changes ~3% GH3->GH5; "
            "phase <0.05 rad. Full Jones, ICS, higher Jo, and A1 A22/F22 are not closed."
        ),
        "cep_pi_v2_gh5_archive": {
            "max_e_abs_hard": cep_abs,
            "max_e_rel_strong_hard": cep_rel,
            "pass": cep_abs <= CEP_V2_ABS_ATOL and cep_rel <= CEP_V2_REL_ATOL,
            "scope": "Jones_only",
            "metadata_provenance_rechecked": False,
            "not_a_full_production_audit": True,
        },
        "headline": headline,
        "states": states,
    }
    text = json.dumps(report, indent=2) + "\n"
    slim = {k: report[k] for k in report if k != "states"}
    print(json.dumps(slim, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

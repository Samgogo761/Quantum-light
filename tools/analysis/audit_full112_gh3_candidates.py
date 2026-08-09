#!/usr/bin/env python3
"""Audit full112 GH3 ±N candidates (job 27992).

ICS/CS at Fortran harmonic_fft_index bins.
CEP/P gate (stored-gauge alpha pairing), both directions:

    J_{+N}(α) + J_{-N}(-α)  ~ 0
    J_{+N}(-α) + J_{-N}(α)  ~ 0

Hard gate: H2/H5/H7/H9. H10 diagnostic only.
Any INCOMPLETE state fails the process (exit != 0).
Jo continuum SNR is NOT computed from node data (marked unavailable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from hhg_fft_utils import parse_input_nml, pick_ics_cs_at_order

TWOPI = 2.0 * math.pi
ALPHA_TOL = 1.0e-10
WEIGHT_TOL = 1.0e-12
INTENSITY_RTOL = 1.0e-8
PHASE_TOL = 1.0e-6
HARD_ORDERS = (2, 5, 7, 9)
# Relative floor vs ensemble max amp at that harmonic (avoid false FAIL on α≈0).
WEAK_FRAC = 1.0e-12
META_KEYS = (
    "source_sha256",
    "binary_sha256",
    "nk",
    "dt",
    "T2_cycles",
    "wvl_nm",
    "model",
    "I_bar",
    "harmonics",
    "nodes_sha256",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing run_metadata.txt: {path}")
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_manifest(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 6:
            raise ValueError(f"{path}: expected 6 columns")
        rows.append(
            {
                "id": int(f[0]),
                "weight": float(f[1]),
                "re": float(f[2]),
                "im": float(f[3]),
                "intensity": float(f[4]),
                "phase": float(f[5]) % TWOPI,
            }
        )
    return rows


def load_node_modes(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10:
            raise ValueError(f"{path}: expected 10 columns")
        nid, order = int(f[0]), int(f[4])
        vals = [float(x) for x in f[1:4] + f[5:]]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError(f"{path}: non-finite in node {nid} H{order}")
        out[(nid, order)] = {
            "weight": float(f[1]),
            "intensity": float(f[2]),
            "phase": float(f[3]) % TWOPI,
            "jx": complex(float(f[5]), float(f[6])),
            "jy": complex(float(f[7]), float(f[8])),
            "power": float(f[9]),
        }
    return out


def amp_xy(jx: complex, jy: complex) -> float:
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def phase_diff(a: float, b: float) -> float:
    d = abs((a % TWOPI) - (b % TWOPI))
    return min(d, TWOPI - d)


def validate_modes_vs_manifest(
    modes: dict[tuple[int, int], dict],
    manifest: list[dict],
    label: str,
) -> None:
    by_id = {int(r["id"]): r for r in manifest}
    for (nid, order), mp in modes.items():
        if nid not in by_id:
            raise ValueError(f"{label}: modes has unknown node {nid}")
        ref = by_id[nid]
        if abs(mp["weight"] - float(ref["weight"])) > WEIGHT_TOL:
            raise ValueError(f"{label}: node {nid} H{order} weight != manifest")
        i_m, i_r = mp["intensity"], float(ref["intensity"])
        if abs(i_m - i_r) > INTENSITY_RTOL * max(abs(i_m), abs(i_r), 1.0):
            raise ValueError(f"{label}: node {nid} H{order} intensity != manifest")
        if phase_diff(mp["phase"], float(ref["phase"])) > PHASE_TOL:
            raise ValueError(f"{label}: node {nid} H{order} phase != manifest")


def pair_nodes_by_neg_alpha(manifest: list[dict]) -> list[dict]:
    """Unordered pairs {α, -α}; alpha=0 is self-paired."""
    by_alpha: dict[tuple[float, float], dict] = {}
    for row in manifest:
        key = (round(float(row["re"]), 12), round(float(row["im"]), 12))
        if key in by_alpha:
            raise ValueError(f"duplicate alpha key in manifest: {key}")
        by_alpha[key] = row

    pairs: list[dict] = []
    used: set[int] = set()
    for row in manifest:
        nid = int(row["id"])
        if nid in used:
            continue
        re_a, im_a = float(row["re"]), float(row["im"])
        if abs(re_a) + abs(im_a) <= ALPHA_TOL:
            partner = row
            kind = "self_alpha0"
        else:
            partner = by_alpha.get((round(-re_a, 12), round(-im_a, 12)))
            if partner is None:
                raise ValueError(f"no -alpha partner for node {nid}")
            kind = "neg_alpha"
        pid = int(partner["id"])
        if abs(float(row["weight"]) - float(partner["weight"])) > WEIGHT_TOL:
            raise ValueError(f"weight mismatch nodes {nid}/{pid}")
        i_a, i_b = float(row["intensity"]), float(partner["intensity"])
        if abs(i_a - i_b) > INTENSITY_RTOL * max(abs(i_a), abs(i_b), 1.0):
            raise ValueError(f"intensity mismatch nodes {nid}/{pid}")
        pairs.append(
            {
                "id_alpha": nid,
                "id_neg": pid,
                "kind": kind,
                "re_alpha": re_a,
                "im_alpha": im_a,
            }
        )
        used.add(nid)
        used.add(pid)
    if len(used) != len(manifest):
        raise ValueError("alpha pairing did not cover all manifest nodes")
    return pairs


def cep_pi_residual(
    jxp: complex,
    jyp: complex,
    jxm: complex,
    jym: complex,
    amp_ref: float,
) -> dict[str, float | bool]:
    """Return residual; weak pairs use amp_ref denominator, not a tiny fixed floor."""
    sx = jxp + jxm
    sy = jyp + jym
    num = math.sqrt(abs(sx) ** 2 + abs(sy) ** 2)
    amp_p = amp_xy(jxp, jyp)
    amp_m = amp_xy(jxm, jym)
    den_local = max(amp_p, amp_m)
    weak = den_local <= WEAK_FRAC * max(amp_ref, 0.0)
    if weak:
        den = max(amp_ref, 0.0)
        if den <= 0.0:
            den = 1.0
        eps = num / den
    else:
        eps = num / den_local
    return {
        "epsilon_Ppi": eps,
        "num": num,
        "amp_plus": amp_p,
        "amp_minus": amp_m,
        "weak_pair": weak,
    }


def compare_pm_metadata(meta_p: dict[str, str], meta_m: dict[str, str], stem: str) -> None:
    for key in META_KEYS:
        vp = meta_p.get(key, "")
        vm = meta_m.get(key, "")
        if not vp or not vm:
            raise ValueError(f"{stem}: missing metadata key {key} on ±N")
        if vp != vm:
            raise ValueError(f"{stem}: ±N metadata mismatch {key}: {vp!r} vs {vm!r}")


def audit_state(
    stem: str,
    outroot: Path,
    orders: list[int],
    grid: dict[str, float | int],
    cov_tol: float,
) -> dict:
    plus = outroot / f"{stem}_plusN"
    minus = outroot / f"{stem}_minusN"
    if not (plus / "SUCCESS").exists() or not (minus / "SUCCESS").exists():
        return {"state": stem, "status": "INCOMPLETE", "cep_pi_hard_pass": False}

    try:
        meta_p = read_metadata(plus / "run_metadata.txt")
        meta_m = read_metadata(minus / "run_metadata.txt")
        compare_pm_metadata(meta_p, meta_m, stem)
    except (OSError, ValueError) as exc:
        return {
            "state": stem,
            "status": "FAIL",
            "error": str(exc),
            "cep_pi_hard_pass": False,
        }

    manifest_p = plus / "nodes_manifest.input.dat"
    manifest_m = minus / "nodes_manifest.input.dat"
    manifest_sha = file_sha256(manifest_p)
    if manifest_sha != file_sha256(manifest_m):
        return {
            "state": stem,
            "status": "FAIL",
            "error": "±N manifest SHA mismatch",
            "cep_pi_hard_pass": False,
        }

    try:
        manifest = load_manifest(manifest_p)
        modes_p = load_node_modes(plus / "HHG_nodes_modes.dat")
        modes_m = load_node_modes(minus / "HHG_nodes_modes.dat")
        validate_modes_vs_manifest(modes_p, manifest, f"{stem}/+N")
        validate_modes_vs_manifest(modes_m, manifest, f"{stem}/-N")
        alpha_pairs = pair_nodes_by_neg_alpha(manifest)
    except (OSError, ValueError) as exc:
        return {
            "state": stem,
            "status": "FAIL",
            "error": str(exc),
            "cep_pi_hard_pass": False,
        }

    dt = float(grid["dt_au"])
    nt = int(grid["nt"])
    omega0 = float(grid["omega0"])

    per_order: dict[str, dict] = {}
    for order in orders:
        jox = 0.0 + 0.0j
        joy = 0.0 + 0.0j
        for (nid, o), mp in modes_p.items():
            if o != order:
                continue
            mm = modes_m[(nid, o)]
            if abs(mp["weight"] - mm["weight"]) > WEIGHT_TOL:
                return {
                    "state": stem,
                    "status": "FAIL",
                    "error": f"±N weight mismatch node {nid}",
                    "cep_pi_hard_pass": False,
                }
            jox += mp["weight"] * 0.5 * (mp["jx"] - mm["jx"])
            joy += mp["weight"] * 0.5 * (mp["jy"] - mm["jy"])
        ics_p = pick_ics_cs_at_order(plus / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
        ics_m = pick_ics_cs_at_order(minus / "HHG_ics_cs.dat", order, nt=nt, dt=dt, omega0=omega0)
        per_order[str(order)] = {
            "ICS_plus": ics_p["ICS"],
            "CS_plus": ics_p["CS"],
            "ICS_minus": ics_m["ICS"],
            "CS_minus": ics_m["CS"],
            "ics_fft_index": ics_p["fft_index"],
            "ics_h_bin": ics_p["harmonic_order_bin"],
            "Jo_jx_re": jox.real,
            "Jo_jx_im": jox.imag,
            "Jo_jy_re": joy.real,
            "Jo_jy_im": joy.imag,
            "Jo_amp": amp_xy(jox, joy),
            "Jo_snr_continuum": None,
            "Jo_snr_note": "unavailable: need full Jo(omega) from merged spectrum, not ICS proxy",
        }

    cov_by_order: dict[str, dict] = {}
    cov_pass_hard = True
    for order in orders:
        amps: list[float] = []
        for pr in alpha_pairs:
            ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
            for nid in (ia, ib):
                amps.append(amp_xy(modes_p[(nid, order)]["jx"], modes_p[(nid, order)]["jy"]))
                amps.append(amp_xy(modes_m[(nid, order)]["jx"], modes_m[(nid, order)]["jy"]))
        amp_ref = max(amps) if amps else 0.0

        errs: list[float] = []
        pair_details: list[dict] = []
        for pr in alpha_pairs:
            ia, ib = int(pr["id_alpha"]), int(pr["id_neg"])
            # Direction A: J_{+N}(α) + J_{-N}(-α)
            mp_a = modes_p[(ia, order)]
            mm_a = modes_m[(ib, order)]
            res_a = cep_pi_residual(mp_a["jx"], mp_a["jy"], mm_a["jx"], mm_a["jy"], amp_ref)
            # Direction B: J_{+N}(-α) + J_{-N}(α)
            if pr["kind"] == "self_alpha0":
                res_b = res_a
            else:
                mp_b = modes_p[(ib, order)]
                mm_b = modes_m[(ia, order)]
                res_b = cep_pi_residual(mp_b["jx"], mp_b["jy"], mm_b["jx"], mm_b["jy"], amp_ref)
            eps = max(float(res_a["epsilon_Ppi"]), float(res_b["epsilon_Ppi"]))
            errs.append(eps)
            pair_details.append(
                {
                    "id_alpha": ia,
                    "id_neg": ib,
                    "kind": pr["kind"],
                    "epsilon_Ppi_dirA": res_a["epsilon_Ppi"],
                    "epsilon_Ppi_dirB": res_b["epsilon_Ppi"],
                    "epsilon_Ppi": eps,
                    "weak_pair_A": res_a["weak_pair"],
                    "weak_pair_B": res_b["weak_pair"],
                }
            )
        eps_max = max(errs) if errs else float("nan")
        eps_med = sorted(errs)[len(errs) // 2] if errs else float("nan")
        is_hard = order in HARD_ORDERS
        passed = eps_max <= cov_tol
        if is_hard and not passed:
            cov_pass_hard = False
        cov_by_order[str(order)] = {
            "epsilon_Ppi_max": eps_max,
            "epsilon_Ppi_median": eps_med,
            "amp_ref": amp_ref,
            "n_pairs": len(alpha_pairs),
            "n_direction_checks": len(alpha_pairs) * (1 if all(p["kind"] == "self_alpha0" for p in alpha_pairs) else 2),
            "gate": "hard" if is_hard else "diagnostic",
            "pass": passed,
            "pairs": pair_details,
        }

    return {
        "state": stem,
        "status": "OK",
        "selection": "keep" if stem.startswith("sv_r2p5_") else "control_low_r",
        "manifest_sha256": manifest_sha,
        "grid": grid,
        "orders": per_order,
        "cep_pi_gate": cov_by_order,
        "cep_pi_hard_pass": cov_pass_hard,
        "cep_pi_relation": (
            "both directions: J_{+N}(α)+J_{-N}(-α) and J_{+N}(-α)+J_{-N}(α); "
            "weak pairs normalize by ensemble amp_ref"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", type=Path, required=True)
    ap.add_argument(
        "--states",
        nargs="+",
        default=["sv_r2p5_th000", "sv_r2p5_th180", "sv_r0p5_th180"],
    )
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 5, 7, 9, 10])
    ap.add_argument("--cov-tol", type=float, default=1.0e-3)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    first_plus = args.outroot / f"{args.states[0]}_plusN" / "input.nml"
    if not first_plus.is_file():
        payload = {
            "campaign": "full112_gh3_candidates_k20",
            "status": "INCOMPLETE",
            "error": f"missing {first_plus}",
            "states": [{"state": st, "status": "INCOMPLETE", "cep_pi_hard_pass": False} for st in args.states],
        }
        text = json.dumps(payload, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text, encoding="utf-8")
        print(text)
        return 1

    grid = parse_input_nml(first_plus)
    summary = [audit_state(st, args.outroot, args.orders, grid, args.cov_tol) for st in args.states]
    all_ok = all(s.get("status") == "OK" and s.get("cep_pi_hard_pass", False) for s in summary)
    payload = {
        "campaign": "full112_gh3_candidates_k20",
        "status": "PASS" if all_ok else "FAIL",
        "selection_policy": {
            "keep": ["sv_r2p5_th000", "sv_r2p5_th180"],
            "control": ["sv_r0p5_th180"],
        },
        "states": summary,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

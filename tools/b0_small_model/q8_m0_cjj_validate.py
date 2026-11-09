#!/usr/bin/env python3
"""Small-scale M0 CJJ campaign: 6-k algebra, then at most four BZ configs × ±N.

Does not run quantum pump, photon output, or a new drive protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q8_m0_cjj import (  # noqa: E402
    APPROVAL_SCOPE,
    CJJ_DEFINITION,
    DEFINITION_REVIEW,
    ORDERS,
    SNAPSHOT_FRAC,
    choose_probe_k,
    cjj_fingerprint,
    domain_combine,
    dump,
    gram_psd_residual,
    propagate_cjj,
    q4_from_schrodinger,
    undriven_trapz_check,
    undriven_y,
)
from q8_m0_classical import kappa_shift, p1_protocol, project_harmonic  # noqa: E402
from q8_m0_h2_qualify import classify_channel  # noqa: E402
from q8_m0_model import R0, build_geometry, current_ops  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/q_line/cjj_m0_p1_validation_20260916"
OLD_STATIC = REPO / "results/q_line/m0_v1_20260912/static/gap_report.json"
STOCK48 = REPO / "results/q_line/m0_p0p1_20260914/p1_A0.8_nk48_nt400/analysis.json"
ALGEBRA_TOL = 1e-8
CJJ_ABS_REPORT_RESOLUTION = 1e-12
CJJ_ABS_UNIT = "(e t_h Å/ħ)^2 per-cell source kernel d"
CJJ_ABS_FLOOR_RATIONALE = (
    "engineering reporting floor for filtered per-cell d (1e-12 in d units). "
    "Not claimed to sit above two-time Q4/Gram residuals "
    "(B0 max |Q4-Q8| ~ 9.92e-12); those checks and filtered-d errors are separate. "
    "The floor is not a proven error bound on d and is not the mean-current 1e-9 "
    "sheet-harmonic floor. Do not retune it to pass a channel."
)
CJJ_D_SHAPE = (9, 2, 9, 2)
# Only this stored hash, this current hash, and B0_sixk. Future cjj.py hashes are not auto-compatible.
DOCUMENTED_COMPATIBLE_CJJ_MODULE = {
    "398e945c645bfe02cd61921f4e58d68a39f82b490906686a8c33b0bc84d5f415": {
        "allowed_names": ["B0_sixk"],
        "allowed_current_module_sha256": [
            "a89592df0f9084dc794134862f686bd25c03b43173f371000a5f9a9f59278411",
        ],
        "reason": "n_k vs nk_lin snapshot allocation fix; Q4/Gram rechecked on saved X with current a89592df",
    }
}
CONFIGS = (
    ("V24", 24, 400, 0.0),
    ("V48", 48, 400, 0.0),
    ("V48S", 48, 400, 0.5),
    ("V48dt", 48, 800, 0.0),
)


def mu_from_static() -> float:
    return float(json.loads(OLD_STATIC.read_text(encoding="utf-8"))["mu_half_filling"])


def _save(outdir: Path, tag: str, r: dict, ident: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in r.items():
        if k == "diag" and isinstance(v, dict):
            for dk, dv in v.items():
                payload[f"diag_{dk}"] = np.asarray(dv)
            continue
        payload[k] = v
    np.savez_compressed(outdir / f"{tag}.npz", **payload)
    (outdir / f"{tag}.meta.json").write_text(json.dumps(ident, indent=2, ensure_ascii=False), encoding="utf-8")


def _load(outdir: Path, tag: str) -> dict | None:
    fn = outdir / f"{tag}.npz"
    meta = outdir / f"{tag}.meta.json"
    if not fn.exists() or not meta.exists():
        return None
    raw = np.load(fn, allow_pickle=True)
    out = {k: raw[k] for k in raw.files}
    diag = {}
    for k in list(out):
        if k.startswith("diag_"):
            diag[k[5:]] = out.pop(k)
    if diag:
        out["diag"] = diag
    return out


def pack_z(z) -> dict:
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z))}


def signed_kernel_diff(z_pumped, z_undriven) -> dict:
    """Real difference kept signed; imag is a residual, not a brightness."""
    dz = complex(z_pumped) - complex(z_undriven)
    return {
        "re": float(dz.real),
        "im": float(dz.imag),
        "abs": float(abs(dz)),
        "note": "signed real difference; imag residual; difference need not be PSD",
    }


def cache_reuse_decision(stored: dict, ident: dict, payload: dict, *, name: str, expected_kf=None) -> dict:
    """Refuse silent reuse unless identity, module, shapes, and time axis match."""
    reasons: list[str] = []
    ok = True
    documented = False
    if stored.get("cjj_identity_sha256") != ident.get("cjj_identity_sha256"):
        ok = False
        reasons.append("identity_mismatch")
    stored_mod = str(stored.get("cjj_module_sha256") or "")
    ident_mod = str(ident.get("cjj_module_sha256") or "")
    if stored_mod != ident_mod:
        rec = DOCUMENTED_COMPATIBLE_CJJ_MODULE.get(stored_mod)
        allowed_cur = rec.get("allowed_current_module_sha256", []) if rec else []
        if rec and name in rec["allowed_names"] and ident_mod in allowed_cur:
            documented = True
            reasons.append("module_hash_documented_compatible")
            reasons.append(rec["reason"])
        else:
            ok = False
            reasons.append("module_hash_mismatch_not_documented")
    try:
        kf = np.asarray(payload["kf"])
        y = np.asarray(payload["Y"])
        d = np.asarray(payload["d_namb"])
        times = np.asarray(payload["times"])
        snap = np.asarray(payload["snapshot_X"])
        n_k = int(np.asarray(payload["n_k"])) if "n_k" in payload else int(kf.shape[0])
    except (KeyError, TypeError) as exc:
        return {"reuse": False, "documented_module_compat": documented, "reasons": [f"payload_unreadable:{exc}"]}
    if kf.shape[0] != n_k:
        ok = False
        reasons.append("kf_n_k_mismatch")
    if y.shape[0] != n_k:
        ok = False
        reasons.append("Y_n_k_mismatch")
    if tuple(d.shape) != CJJ_D_SHAPE:
        ok = False
        reasons.append(f"d_shape_{tuple(d.shape)}")
    if snap.ndim >= 2 and snap.shape[1] != n_k:
        ok = False
        reasons.append("snapshot_X_n_k_mismatch")
    n_cyc = int(stored.get("protocol", {}).get("n_cycles", ident.get("protocol", {}).get("n_cycles", 8)))
    ntpc = int(stored.get("nt_per_cycle", ident.get("nt_per_cycle", 0)))
    if times.size != n_cyc * ntpc + 1:
        ok = False
        reasons.append("time_axis_length")
    if stored.get("kf_kind") == "custom" and expected_kf is not None:
        got = np.mod(np.asarray(kf, dtype=float), 1.0)
        exp = np.mod(np.asarray(expected_kf, dtype=float), 1.0)
        wrap = np.abs(np.mod(got - exp + 0.5, 1.0) - 0.5)
        if got.shape != exp.shape or float(np.max(wrap)) > 1e-12:
            ok = False
            reasons.append("custom_kf_mismatch")
    if not reasons and ok:
        reasons.append("identity_module_shape_time_ok")
    return {"reuse": bool(ok), "documented_module_compat": documented, "reasons": reasons}


def run_cached(geo, prot, mu, *, name, nk, ntpc, shift, neel, kf_frac=None, save_full_x=False, verbose=True, outdir: Path | None = None) -> dict:
    root = Path(outdir) if outdir is not None else OUT
    dest = root / name
    tag = f"{name}_N{neel:+d}_A+1"
    kf_kind = "custom" if kf_frac is not None else f"mp_nk{nk}_shift{shift}"
    ident = cjj_fingerprint(
        geo, R0, prot, nk=(len(kf_frac) if kf_frac is not None else nk),
        nt_per_cycle=ntpc, mu=mu, grid_shift=0.0 if kf_frac is not None else shift,
        neel=neel, sign=+1, kf_kind=kf_kind,
    )
    meta_path = dest / f"{tag}.meta.json"
    if meta_path.exists() and (dest / f"{tag}.npz").exists():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        loaded = _load(dest, tag)
        decision = cache_reuse_decision(stored, ident, loaded, name=name, expected_kf=kf_frac)
        if decision["reuse"]:
            print(f"[{name}] load {tag} cache={decision['reasons']}", flush=True)
            return loaded
        print(f"[{name}] cache refused {tag}: {decision['reasons']}", flush=True)
    print(f"[{name}] run {tag} nk={ident['nk']} nt/cycle={ntpc}", flush=True)
    r = propagate_cjj(
        geo, R0, prot, nk=nk, nt_per_cycle=ntpc, neel=neel, sign=+1, mu=mu,
        verbose=verbose, grid_shift=shift, kf_frac=kf_frac, save_full_x=save_full_x,
    )
    _save(dest, tag, r, ident)
    return r


def u_now(vv, vc, vv0, vc0):
    w = np.concatenate([vv, vc], axis=-1)
    w0 = np.concatenate([vv0, vc0], axis=-1)
    return w @ np.conj(np.swapaxes(w0, 1, 2))


def check_sixk(r: dict, geo, prot) -> dict:

    snap_t = np.asarray(r["snapshot_t"], dtype=float)
    snap_x = np.asarray(r["snapshot_X"])
    snap_vv = np.asarray(r["snapshot_Vv"])
    snap_vc = np.asarray(r["snapshot_Vc"])
    vv0 = np.asarray(r["vv0"])
    vc0 = np.asarray(r["vc0"])
    kf = np.asarray(r["kf"], dtype=float)
    neel = int(np.asarray(r["neel"]))
    nsnap = snap_t.size
    s_ids = [1, nsnap // 2, max(2, nsnap - 3)]
    pairs = [(s_ids[0], s_ids[1]), (s_ids[1], s_ids[2]), (s_ids[1], s_ids[1])]
    rows = []
    for si, sj in pairs:
        xa = snap_x[si]
        xb = snap_x[sj]
        c8 = np.einsum("kacv,kbcv->kab", np.conj(xa), xb)
        u_i = u_now(snap_vv[si], snap_vc[si], vv0, vc0)
        u_j = u_now(snap_vv[sj], snap_vc[sj], vv0, vc0)
        u_ij = u_i @ np.conj(np.swapaxes(u_j, 1, 2))
        rho_j = snap_vv[sj] @ np.conj(np.swapaxes(snap_vv[sj], 1, 2))
        q_j = snap_vc[sj] @ np.conj(np.swapaxes(snap_vc[sj], 1, 2))
        kap_i = geo.k_cart(kf) + kappa_shift(snap_t[si], geo, prot, +1)
        kap_j = geo.k_cart(kf) + kappa_shift(snap_t[sj], geo, prot, +1)
        ja = current_ops(geo, R0, kap_i, neel)
        jb = current_ops(geo, R0, kap_j, neel)
        c4 = {}
        c8_flat = {}
        for a, b, lab in ((0, 0, "xx"), (0, 1, "xy"), (1, 0, "yx"), (1, 1, "yy")):
            c4[lab] = q4_from_schrodinger(ja[:, a], jb[:, b], u_ij, rho_j, q_j)
            c8_flat[lab] = c8[:, a, b]
        c4_vs_q8 = {lab: float(np.max(np.abs(c4[lab] - c8_flat[lab]))) for lab in c4}
        c_yx_swap = np.einsum("kcv,kcv->k", np.conj(xb[:, 1]), xa[:, 0])
        herm_xy = float(np.max(np.abs(c8[:, 0, 1] - np.conj(c_yx_swap))))
        rows.append({
            "si": int(si),
            "sj": int(sj),
            "ti": float(snap_t[si]),
            "tj": float(snap_t[sj]),
            "q4_minus_q8": c4_vs_q8,
            "herm_Cxy_vs_Cyx": herm_xy,
        })
    dpsd = gram_psd_residual(np.asarray(r["d_namb"]))
    mean_res = float(np.asarray(r["mean_vs_rho_maxabs"]))
    herm = float(np.max(np.asarray(r["diag"]["herm"])))
    idem = float(np.max(np.asarray(r["diag"]["idem"])))
    q4_ok = all(max(row["q4_minus_q8"].values()) < ALGEBRA_TOL and row["herm_Cxy_vs_Cyx"] < ALGEBRA_TOL for row in rows)
    ok = mean_res < ALGEBRA_TOL and dpsd["psd_ok"] and herm < 1e-10 and idem < 1e-10 and q4_ok
    return {"pairs": rows, "psd": dpsd, "mean_vs_rho": mean_res, "herm": herm, "idem": idem, "pass": bool(ok)}


def d_channels(d) -> dict:
    """Compact subset used in per-job summaries. Full H1–H9 lives in wrapup."""
    d = np.asarray(d)
    out = {}
    names = {1: "H1", 2: "H2", 3: "H3", 4: "H4"}
    for n in (1, 2, 3, 4):
        for a, ax in ((0, "x"), (1, "y")):
            out[f"d_{names[n]}{ax},{names[n]}{ax}"] = pack_z(d[n - 1, a, n - 1, a])
        if n >= 2:
            out[f"d_{names[n]}x,{names[n]}y"] = pack_z(d[n - 1, 0, n - 1, 1])
    out["d_H2x,H3x"] = pack_z(d[1, 0, 2, 0])
    out["d_H3x,H4x"] = pack_z(d[2, 0, 3, 0])
    return out


CROSS_WINDOWS = ((2, 0, "H2x"), (2, 1, "H2y"), (3, 0, "H3x"), (3, 1, "H3y"), (4, 0, "H4x"), (4, 1, "H4y"))


def d_catalog(d) -> dict:
    """All H1–H9 diagonals, H2/H3/H4 x/y 6×6 cross, and H2x–H6y spotlight."""
    d = np.asarray(d)
    out = {}
    for n in ORDERS:
        for a, ax in ((0, "x"), (1, "y")):
            out[f"d_H{n}{ax},H{n}{ax}"] = pack_z(d[n - 1, a, n - 1, a])
    for n, a, lab_i in CROSS_WINDOWS:
        for m, b, lab_j in CROSS_WINDOWS:
            out[f"d_{lab_i},{lab_j}"] = pack_z(d[n - 1, a, m - 1, b])
    out["d_H2x,H6y"] = pack_z(d[1, 0, 5, 1])
    return out


def stage_sixk(geo, prot, mu) -> dict:
    probe = choose_probe_k(geo, R0, mu)
    dump(OUT / "probe_k.json", {
        "labels": probe["labels"],
        "kf": probe["kf"],
        "K_abs_g": probe["K_abs_g"],
        "min_direct_gap_on_nk48": probe["min_direct_gap_on_nk48"],
        "min_direct_gap_k_frac": probe["min_direct_gap_k_frac"],
        "snapshot_frac": list(SNAPSHOT_FRAC),
        "note": probe["note"],
    })
    runs = {}
    t0 = time.perf_counter()
    for neel in (+1, -1):
        runs[neel] = run_cached(
            geo, prot, mu, name="B0_sixk", nk=6, ntpc=400, shift=0.0,
            neel=neel, kf_frac=probe["kf"], save_full_x=True, verbose=True,
        )
    wall = time.perf_counter() - t0
    checks = {str(neel): check_sixk(runs[neel], geo, prot) for neel in (+1, -1)}
    ud_chk = {
        str(neel): undriven_trapz_check(geo, R0, prot, probe["kf"], mu, neel, 400) for neel in (+1, -1)
    }
    rec = {
        "cjj_definition": CJJ_DEFINITION,
        "definition_review": DEFINITION_REVIEW,
        "approval_scope": APPROVAL_SCOPE,
        "probe": {"labels": probe["labels"], "kf": probe["kf"].tolist()},
        "wall_seconds_pair": wall,
        "algebra": checks,
        "undriven_trapz_vs_analytic": ud_chk,
        "pass": all(v["pass"] for v in checks.values()),
        "cost_note": "per-Neel wall ~ wall_seconds_pair/2; V24 ~ 576/6 times that if CJJ cost scales with Nk",
        "m0_cjj_status": "implementation_verified" if all(v["pass"] for v in checks.values()) else "implementation_failed",
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }
    dump(OUT / "B0_sixk_report.json", rec)
    print(json.dumps({"stage": "B0", "pass": rec["pass"], "wall": wall, "status": rec["m0_cjj_status"]}, indent=2), flush=True)
    return rec, probe, runs, wall


def stage_config(geo, prot, mu, name, nk, ntpc, shift, est: dict | None) -> dict:
    t0 = time.perf_counter()
    if est:
        print(json.dumps({"stage": name, "estimated_seconds_pair": est}, indent=2), flush=True)
    runs = {}
    for neel in (+1, -1):
        runs[neel] = run_cached(
            geo, prot, mu, name=name, nk=nk, ntpc=ntpc, shift=shift,
            neel=neel, save_full_x=False, verbose=True,
        )
    wall = time.perf_counter() - t0
    d_plus = np.asarray(runs[+1]["d_namb"])
    d_minus = np.asarray(runs[-1]["d_namb"])
    comb = domain_combine(d_plus, d_minus)
    mu_p = np.asarray(runs[+1]["mu_per_cell"])
    mu_m = np.asarray(runs[-1]["mu_per_cell"])
    times = np.asarray(runs[+1]["times"])
    a_cell = float(np.asarray(runs[+1]["a_cell_ang2"]))
    je = 0.5 * (mu_p + mu_m) / a_cell
    jo = 0.5 * (mu_p - mu_m) / a_cell
    mean_h = {}
    for n in ORDERS:
        ze = project_harmonic(times, je, n * prot.omega0, "sin2")
        zo = project_harmonic(times, jo, n * prot.omega0, "sin2")
        mean_h[str(n)] = {
            "Je_x": [float(ze[0].real), float(ze[0].imag)],
            "Je_y": [float(ze[1].real), float(ze[1].imag)],
            "Jo_x": [float(zo[0].real), float(zo[0].imag)],
            "Jo_y": [float(zo[1].real), float(zo[1].imag)],
        }
    und = {
        "+1": undriven_y(geo, R0, prot, np.asarray(runs[+1]["kf"]), mu, +1),
        "-1": undriven_y(geo, R0, prot, np.asarray(runs[-1]["kf"]), mu, -1),
    }
    rec = {
        "name": name,
        "nk": nk,
        "ntpc": ntpc,
        "grid_shift": shift,
        "wall_seconds_pair": wall,
        "mean_vs_rho_maxabs": {
            "+1": float(np.asarray(runs[+1]["mean_vs_rho_maxabs"])),
            "-1": float(np.asarray(runs[-1]["mean_vs_rho_maxabs"])),
        },
        "d_plus": d_channels(d_plus),
        "d_minus": d_channels(d_minus),
        "d_dom_e": d_channels(comb["d_dom_e"]),
        "d_dom_o": d_channels(comb["d_dom_o"]),
        "psd_plus": comb["psd_plus"],
        "psd_minus": comb["psd_minus"],
        "mean_harmonics_sheet_sin2": mean_h,
        "undriven_d_plus": d_channels(und["+1"]["d_namb"]),
        "undriven_d_minus": d_channels(und["-1"]["d_namb"]),
        "pumped_minus_undriven_H2xx": {
            "+1": signed_kernel_diff(d_plus[1, 0, 1, 0], und["+1"]["d_namb"][1, 0, 1, 0]),
            "-1": signed_kernel_diff(d_minus[1, 0, 1, 0], und["-1"]["d_namb"][1, 0, 1, 0]),
        },
        "cjj_definition": CJJ_DEFINITION,
        "quantum_input": "not_run",
        "photon_output": "not_run",
    }
    dump(OUT / name / "summary.json", rec)
    print(json.dumps({"stage": name, "wall": wall, "d_H2xx_plus": rec["d_plus"]["d_H2x,H2x"]}, indent=2), flush=True)
    return rec


def stock_v48_mean_check(v48: dict) -> dict:
    if not STOCK48.exists():
        return {"available": False}
    tab = json.loads(STOCK48.read_text(encoding="utf-8"))["analysis"]["windows"]["sin2"]
    out = {"available": True, "pairs": {}}
    for n, who, comp, key in (
        (1, "J_e", "x", "Je_x"),
        (2, "J_o", "x", "Jo_x"),
        (2, "J_o", "y", "Jo_y"),
        (3, "J_e", "x", "Je_x"),
    ):
        z_stock = complex(*tab[who][str(n)][comp])
        z_new = complex(*v48["mean_harmonics_sheet_sin2"][str(n)][key])
        out["pairs"][f"{who}_{comp}_H{n}"] = {
            "stock": [z_stock.real, z_stock.imag],
            "v48": [z_new.real, z_new.imag],
            "abs_diff": abs(z_new - z_stock),
        }
    return out


def classify_d(jobs: dict, *, abs_floor: float | None = None) -> dict:
    floor = CJJ_ABS_REPORT_RESOLUTION if abs_floor is None else float(abs_floor)
    ref = jobs["V48"]
    out = {}
    for side in ("d_plus", "d_minus", "d_dom_e", "d_dom_o"):
        if side not in ref:
            continue
        out[side] = {}
        for key, pack in ref[side].items():
            z_ref = complex(pack["re"], pack["im"])
            z24 = complex(jobs["V24"][side][key]["re"], jobs["V24"][side][key]["im"])
            zs = complex(jobs["V48S"][side][key]["re"], jobs["V48S"][side][key]["im"])
            zdt = complex(jobs["V48dt"][side][key]["re"], jobs["V48dt"][side][key]["im"])
            verdict = classify_channel(z_ref, z_k=z24, z_ind=zs, z_dt=zdt, abs_floor=floor)
            out[side][key] = {
                "z_V48": pack,
                "rel_V24": abs(z24 - z_ref) / abs(z_ref) if abs(z_ref) else None,
                "rel_shift": abs(zs - z_ref) / abs(z_ref) if abs(z_ref) else None,
                "rel_dt": abs(zdt - z_ref) / abs(z_ref) if abs(z_ref) else None,
                "d_V24": abs(z24 - z_ref),
                "d_shift": abs(zs - z_ref),
                "d_dt": abs(zdt - z_ref),
                "status": verdict["status"],
                "reasons": verdict["reasons"],
                "phase_diff_deg": verdict.get("phase_diff_deg"),
                "abs_floor": floor,
                "abs_unit": CJJ_ABS_UNIT,
                "note": CJJ_ABS_FLOOR_RATIONALE,
            }
    return out


def explicit_rel_columns(plus: dict, minus: dict) -> dict:
    """Keep +N relatives separate from max-of-two-domains. JSON is the source of the table."""
    p24, s24 = plus.get("rel_V24"), plus.get("rel_shift")
    m24, sm = minus.get("rel_V24"), minus.get("rel_shift")
    mx24 = None if None in (p24, m24) else max(p24, m24)
    mxs = None if None in (s24, sm) else max(s24, sm)
    return {
        "plus_rel_V24": p24,
        "plus_rel_shift": s24,
        "minus_rel_V24": m24,
        "minus_rel_shift": sm,
        "max_of_two_domains_rel_V24": mx24,
        "max_of_two_domains_rel_shift": mxs,
        "column_rule": "plus_* is +N only; max_of_two_domains_* is worse of +N and -N; do not mix in one table cell",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("sixk", "v24", "v48", "v48s", "v48dt", "summarize", "auto"), default="auto")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    geo = build_geometry()
    prot = p1_protocol(0.8)
    mu = mu_from_static()
    header_path = OUT / "campaign_header.json"
    if not header_path.exists():
        dump(header_path, {
            "role": "launch_record",
            "cjj_definition": CJJ_DEFINITION,
            "definition_review": DEFINITION_REVIEW,
            "approval_scope": APPROVAL_SCOPE,
            "m0_cjj_status_at_launch": "running",
            "m0_cjj_status": "running",
            "canonical_status_file": "campaign_status.json",
            "quantum_input": "not_run",
            "photon_output": "not_run",
            "new_drive_protocol": "not_run",
            "snapshot_policy": "25 envelope and non-integer-cycle stations recorded in q8_m0_cjj.SNAPSHOT_FRAC",
        })
    six = None
    if args.stage in ("sixk", "auto"):
        six, probe, _runs, wall6 = stage_sixk(geo, prot, mu)
        if not six["pass"]:
            dump(OUT / "STOP.json", {"reason": "B0 algebra failed", "report": six})
            return 1
        per_k = wall6 / 2.0 / 6.0
        dump(OUT / "cost_estimate.json", {
            "seconds_per_k_per_neel_nt400": per_k,
            "V24_pair_s": per_k * 576 * 2,
            "V48_pair_s": per_k * 2304 * 2,
            "V48S_pair_s": per_k * 2304 * 2,
            "V48dt_pair_s": per_k * 2304 * 2 * 2,
            "note": "linear Nk scaling estimate from B0; CJJ extras included",
        })
    jobs = {}
    mapping = {"v24": CONFIGS[0], "v48": CONFIGS[1], "v48s": CONFIGS[2], "v48dt": CONFIGS[3]}
    run_which = list(CONFIGS) if args.stage == "auto" else ([mapping[args.stage]] if args.stage in mapping else [])
    est = None
    est_path = OUT / "cost_estimate.json"
    if est_path.exists():
        est = json.loads(est_path.read_text(encoding="utf-8"))
    for name, nk, ntpc, shift in run_which:
        key = {"V24": "V24_pair_s", "V48": "V48_pair_s", "V48S": "V48S_pair_s", "V48dt": "V48dt_pair_s"}[name]
        jobs[name] = stage_config(geo, prot, mu, name, nk, ntpc, shift, est[key] if est else None)
    if args.stage in ("summarize", "auto") or (OUT / "V48" / "summary.json").exists():
        for name, *_ in CONFIGS:
            p = OUT / name / "summary.json"
            if p.exists():
                jobs[name] = json.loads(p.read_text(encoding="utf-8"))
        if set(x[0] for x in CONFIGS) <= set(jobs):
            summary = {
                "cjj_definition": CJJ_DEFINITION,
                "definition_review": DEFINITION_REVIEW,
                "approval_scope": APPROVAL_SCOPE,
                "m0_cjj_status": "implementation_verified_small_bz_complete",
                "source_observable_convergence": classify_d(jobs),
                "v48_vs_stock_nk48_mean": stock_v48_mean_check(jobs["V48"]),
                "quantum_input": "not_run",
                "photon_output": "not_run",
                "new_drive_protocol": "not_run",
                "jobs": {k: {kk: jobs[k][kk] for kk in jobs[k] if kk not in ("mean_harmonics_sheet_sin2",)} for k in jobs},
            }
            dump(OUT / "cjj_validation_summary.json", summary)
            dump(OUT / "campaign_status.json", {
                "role": "canonical_status",
                "m0_cjj_status": summary["m0_cjj_status"],
                "cjj_definition": CJJ_DEFINITION,
                "definition_review": DEFINITION_REVIEW,
                "approval_scope": APPROVAL_SCOPE,
                "abs_floor": CJJ_ABS_REPORT_RESOLUTION,
                "abs_unit": CJJ_ABS_UNIT,
                "quantum_input": "not_run",
                "photon_output": "not_run",
                "new_drive_protocol": "not_run",
            })
            hdr = json.loads(header_path.read_text(encoding="utf-8"))
            hdr["role"] = "launch_record"
            hdr["m0_cjj_status"] = summary["m0_cjj_status"]
            hdr["canonical_status_file"] = "campaign_status.json"
            dump(header_path, hdr)
            print(json.dumps({
                "m0_cjj_status": summary["m0_cjj_status"],
                "v48_stock": summary["v48_vs_stock_nk48_mean"],
            }, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

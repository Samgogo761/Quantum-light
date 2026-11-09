#!/usr/bin/env python3
"""Finite Gamma12 input-quadrature diagnostic using saved ntpc=4096 rows.

Only the sinh resonance-core n_panel changes (96 -> 128). H2 density,
frequency endpoints, tail panels, physical model, initial state, all original
dark columns, detector, and background vertices remain frozen. This is NOT a
full input-axis convergence claim. No forward/background propagation runs.

The n_panel=96 complete coordinate/output regression is mandatory before 128.
Photon initial columns are constructed in bounded frequency blocks, weighted
once by the original attach_weights, and discarded after contraction. U0 is
recomputed from the original stationary photon map, never interpolated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import q8_m0_gamma12_local_prepare_20261007 as prep
import q8_m0_gamma12_local_runner_20261007 as local
import q8_m0_gamma12_runner_20261006 as r1
import q8_m0_output_transpose_20261008 as transpose
import q8_m0_p1_h2_finite_kcv_input_close_20261005 as q
from q8_m0_classical import p1_a2_protocol
from q8_m0_continuum_columns_20261004 import stationary_photon_columns

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/q_line/p1_h2_finite_kcv_gamma12_local_plusN_20261007"
BACKWARD = ROOT / "results/q_line/gamma12_output_transpose_20261008/ntpc_4096"
REFINED = ROOT / "results/q_line/gamma12_output_diagnostic_20261007/time_4096"
OUT = ROOT / "results/q_line/gamma12_output_transpose_input_refine_20261008"
ATOL, RTOL = 1e-10, 1e-8
RAW_KEYS = ("driven_raw_u", "driven_raw_v", "undriven_raw_u", "undriven_raw_v")


def comparison(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"allclose": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    return {"allclose": bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)),
            "maxabs": float(np.max(np.abs(a-b))) if a.size else 0.,
            "relative_fro": float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))}


def accumulate_comparison(ledger, key, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Coordinate shape mismatch for {key}")
    rec = ledger.setdefault(key, {"allclose": True, "maxabs": 0.,
                                 "difference_squared": 0., "reference_squared": 0.,
                                 "n_values": 0})
    rec["allclose"] = rec["allclose"] and bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))
    rec["maxabs"] = max(rec["maxabs"], float(np.max(abs(a-b))) if a.size else 0.)
    rec["difference_squared"] += float(np.vdot((a-b).ravel(), (a-b).ravel()).real)
    rec["reference_squared"] += float(np.vdot(b.ravel(), b.ravel()).real)
    rec["n_values"] += int(a.size)


def finalize_comparisons(ledger):
    for rec in ledger.values():
        rec["relative_fro"] = float(np.sqrt(rec.pop("difference_squared")) /
                                     max(np.sqrt(rec.pop("reference_squared")), 1e-30))
    return ledger


def load_frozen():
    manifest, cache = local._load(SOURCE)  # also verifies frozen cache file hashes
    if manifest["physical_scope"]["neel"] != 1 or manifest["physical_scope"]["nk"] != 12:
        raise ValueError("Only frozen Gamma12 +N is allowed")
    info_path = REFINED / "prepared.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    bmeta_path, brows_path = BACKWARD/"backward.json", BACKWARD/"backward.npz"
    bmeta = json.loads(bmeta_path.read_text(encoding="utf-8"))
    ident = bmeta["identity"]
    expected = {"source_manifest": manifest["manifest_sha256"], "ntpc": 4096,
                "execution_code_sha256": prep.execution_hashes(),
                "transpose_code_sha256": prep.sha256_file(Path(transpose.__file__)),
                "refined_prepared_sha256": prep.sha256_file(info_path)}
    if ident != expected or info["source_manifest"] != manifest["manifest_sha256"]:
        raise ValueError("Frozen backward/source identity differs")
    if prep.sha256_file(brows_path) != bmeta["sha256"]:
        raise ValueError("Backward array hash mismatch")
    fullmeta = json.loads((BACKWARD/"full.json").read_text(encoding="utf-8"))
    if prep.sha256_file(BACKWARD/"full.npz") != fullmeta["row_sha256"]:
        raise ValueError("Existing 4096 full output hash mismatch")
    rows = {}
    with np.load(brows_path, allow_pickle=False) as z:
        for path in ("driven", "undriven"):
            rr, ff = z[path+"_row0"], z[path+"_forcing"]
            if ff.shape != (info["nstep"], 4, 2) or rr.shape != (4, 4610):
                raise ValueError("Unexpected Gamma12 backward dimensions")
            rows[path] = transpose.BackwardRows(rr, ff, 2, int(info["nstep"]),
                                                float(info["h"]), bmeta["timing_s"][path])
    identity = {**expected, "backward_npz_sha256": bmeta["sha256"],
                "backward_json_sha256": prep.sha256_file(bmeta_path),
                "reference_npz_sha256": fullmeta["row_sha256"],
                "new_script_sha256": prep.sha256_file(Path(__file__))}
    return manifest, cache, rows, info, identity


def generated_photon_block(cache, gamma, frequencies, weights, pol, phi_of, eta):
    ph = stationary_photon_columns(cache["gaps"], cache["u0"], cache["D0"],
                                    gamma, frequencies)
    bundle = SimpleNamespace(photon=ph, point=SimpleNamespace(A=np.zeros((2, 0), complex)),
                             n_photon_columns=4*len(frequencies), n_point_columns=0,
                             point_labels=())
    q.attach_weights(bundle, {round(float(o), 12): float(w)
                              for o, w in zip(frequencies, weights)})
    u0, _ = q.u0_from_bundle(bundle, pol, phi_of, eta)
    return ph, u0


def _contract(rows, initial, omega, free, t0, frequency_block, time_block):
    result, timing = {}, {}
    for path in ("driven", "undriven"):
        ans = transpose.contract_columns(rows[path], *initial, omega, free, t0=t0,
                                          frequency_block=frequency_block, time_block=time_block)
        result[path+"_raw_u"] = ans["raw_u"]
        result[path+"_raw_v"] = ans["raw_v"]
        timing[path] = float(ans["wall_s"])
    return result, timing


def _check_96_gate(identity):
    file = OUT/"n_panel_96.json"
    gate = json.loads(file.read_text(encoding="utf-8"))
    if (not gate.get("all_checks_pass") or gate.get("smoke_only")
            or not gate.get("complete_input_rows") or gate.get("n_panel") != 96):
        raise ValueError("Complete n_panel=96 coordinate/output regression must pass first")
    if gate["identity"] != identity:
        raise ValueError("96 regression used different source, backward, or script identity")
    if prep.sha256_file(file.with_suffix(".npz")) != gate["row_sha256"]:
        raise ValueError("96 regression output hash differs")
    return prep.sha256_file(file)


def run(n_panel=96, *, frequency_block=128, time_block=2048, smoke=False):
    if n_panel not in (96, 128) or min(frequency_block, time_block) < 1:
        raise ValueError("Allowed n_panel=96/128 and positive block sizes")
    stem = f"n_panel_{n_panel}" + ("_smoke" if smoke else "")
    output = OUT/(stem+".json")
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refuse overwrite: {output}")
    start = time.perf_counter()
    manifest, cache, rows, info, identity = load_frozen()
    gate_sha = _check_96_gate(identity) if n_panel == 128 else None
    gamma = float(cache["gamma"])
    pol = np.array(cache["pol"], copy=True)
    prot = p1_a2_protocol(q.A_STAR)
    op = q.omega_grid(prot.omega0, prot.t_pulse)
    op = op[op > 0]
    _, normalizer = q.normalize_phi(op, q.phi_unnormalized(op, 2*prot.omega0, prot.t_pulse))
    if not np.isclose(normalizer, manifest["preparation_checks"]["packet_normalizer"],
                      atol=1e-12, rtol=1e-12):
        raise ValueError("Frozen photon packet normalizer differs")
    phi_of = lambda nu: q.phi_at(nu, prot, normalizer)
    eta = q.eta_of(gamma)
    oms, wts = q.input_quadrature(cache["gaps"], cache["u0"], gamma, n_panel=n_panel)
    if (np.any(wts <= 0) or np.any(np.diff(oms) <= 0)
            or len(np.unique(np.round(oms, 12))) != len(oms)):
        raise ValueError("Quadrature ordering, weight, or frequency-key collision")
    ledger = {}
    if n_panel == 96:
        accumulate_comparison(ledger, "frequency_nodes", oms, cache["oms"])
        accumulate_comparison(ledger, "dnu_weights", wts, cache["wts"])
        if not all(v["allclose"] for v in ledger.values()):
            raise ValueError("96 nodes/weights do not reproduce frozen input")
    nph = 4*len(oms)
    old_nph = manifest["physical_scope"]["n_photon_columns"]
    ndark = manifest["physical_scope"]["n_dark_columns"]
    if ndark != 1976 or cache["b_cols"].shape[0] != old_nph+ndark:
        raise ValueError("Frozen dark block count differs")
    if smoke:
        selected = np.unique(np.r_[0, 1, len(oms)//2, len(oms)-2, len(oms)-1,
                                   np.argmin(abs(oms-.05)), np.argmin(abs(oms-.1)),
                                   np.argmin(abs(oms-1.))]).astype(int)
    else:
        selected = np.arange(len(oms))
    nph_selected, total = 4*len(selected), 4*len(selected)+ndark
    ids = np.r_[(4*selected[:, None]+np.arange(4)).ravel(), np.arange(nph, nph+ndark)]
    values = {key: np.empty((2, total), complex) for key in RAW_KEYS+("u0",)}
    timings = {"stationary_assembly_s": 0., "driven_contraction_s": 0.,
               "undriven_contraction_s": 0.}
    t0 = float(manifest["time"].get("t0", 0.))
    for first in range(0, len(selected), frequency_block):
        fids = selected[first:first+frequency_block]
        t = time.perf_counter()
        ph, u0 = generated_photon_block(cache, gamma, oms[fids], wts[fids], pol, phi_of, eta)
        timings["stationary_assembly_s"] += time.perf_counter()-t
        col_ids = (4*fids[:, None]+np.arange(4)).ravel()
        dest = slice(4*first, 4*(first+len(fids)))
        if n_panel == 96:
            for name, actual in (("b_cols", ph.initial.b.T), ("bdag_cols", ph.initial.bdag.T),
                                  ("A_cols", ph.initial.A.T), ("free_A0_cols", ph.free_A0.T),
                                  ("omega", ph.omega)):
                accumulate_comparison(ledger, name, actual, cache[name][col_ids])
            accumulate_comparison(ledger, "canonical_U0", u0, cache["U0_canonical"][:, col_ids])
            if not all(v["allclose"] for v in ledger.values()):
                raise ValueError("96 regenerated input coordinate mismatch")
        initial = (ph.initial.b, ph.initial.bdag, ph.initial.A)
        contracted, times = _contract(rows, initial, ph.omega, ph.free_A0, t0,
                                       frequency_block, time_block)
        for key in RAW_KEYS:
            values[key][:, dest] = contracted[key]
        values["u0"][:, dest] = u0
        for path in ("driven", "undriven"):
            timings[path+"_contraction_s"] += times[path]
        print(f"panel={n_panel}: photon frequencies {first+len(fids)}/{len(selected)}", flush=True)
        del ph, initial, contracted
    # Original dark basis is copied in its original order, not recomputed or added twice.
    dark_block = 4*frequency_block
    for first in range(0, ndark, dark_block):
        count = min(dark_block, ndark-first)
        orig_ids = np.arange(old_nph+first, old_nph+first+count)
        initial = tuple(np.array(cache[key][orig_ids].T, copy=True)
                        for key in ("b_cols", "bdag_cols", "A_cols"))
        om = np.array(cache["omega"][orig_ids], copy=True)
        free = np.array(cache["free_A0_cols"][orig_ids].T, copy=True)
        if np.any(free != 0) or np.any(om != 0):
            raise ValueError("Original dark columns unexpectedly have forcing")
        contracted, times = _contract(rows, initial, om, free, t0, frequency_block, time_block)
        dest = slice(nph_selected+first, nph_selected+first+count)
        for key in RAW_KEYS:
            values[key][:, dest] = contracted[key]
        values["u0"][:, dest] = cache["U0_canonical"][:, orig_ids]
        for path in ("driven", "undriven"):
            timings[path+"_contraction_s"] += times[path]
    du = values["driven_raw_u"]-values["undriven_raw_u"]
    vv = values["driven_raw_v"]-values["undriven_raw_v"]
    uu = values["u0"]+du
    values.update(column_id=ids, oms=oms, wts=wts, selected_frequency_ids=selected,
                  U=uu, V=vv, delta_u=du, n=vv.conj() @ vv.T, m=uu @ vv.T,
                  m_U0VT=values["u0"] @ vv.T, m_source_source=du @ vv.T,
                  normal_contribution=uu @ uu.conj().T-vv @ vv.conj().T,
                  anomalous_contribution=uu @ vv.T-vv @ uu.T,
                  U0_gram=values["u0"] @ values["u0"].conj().T)
    checks = {}
    if n_panel == 96:
        with np.load(BACKWARD/"full.npz", allow_pickle=False) as ref:
            for key in RAW_KEYS+("u0", "U", "V", "delta_u"):
                checks[key] = comparison(values[key], ref[key][:, ids])
            ur, vr = ref["U"][:, ids], ref["V"][:, ids]
            for key, expected in (("n", vr.conj() @ vr.T), ("m", ur @ vr.T),
                                  ("normal_contribution", ur @ ur.conj().T-vr @ vr.conj().T),
                                  ("anomalous_contribution", ur @ vr.T-vr @ ur.T)):
                checks[key] = comparison(values[key], expected)
    r1._finite_arrays(values)
    OUT.mkdir(parents=True, exist_ok=True)
    r1._write_npz_atomic(output.with_suffix(".npz"), values)
    ledger = finalize_comparisons(ledger)
    passed = (all(v["allclose"] for v in ledger.values())
              and all(v["allclose"] for v in checks.values())) if n_panel == 96 else None
    meta = {"status": "resonance_core_refinement_only", "identity": identity,
            "n_panel": n_panel, "ntpc": 4096, "nstep": info["nstep"], "h": info["h"],
            "n_nodes": len(oms), "n_photon_columns": nph, "n_dark_columns": ndark,
            "n_contracted_columns": total, "all_checks_pass": passed,
            "smoke_only": smoke, "complete_input_rows": not smoke,
            "coordinate_regression": ledger, "output_regression": checks,
            "required_96_gate_sha256": gate_sha,
            "packet_mass": q.packet_mass_dnu(oms, wts, prot, normalizer),
            "U0_gram_defect": float(np.max(abs(values["U0_gram"]-np.eye(2)))),
            "n_trace": float(np.trace(values["n"]).real),
            "normal_ccr_maxabs": float(np.max(abs(values["normal_contribution"]-np.eye(2)))),
            "anomalous_maxabs": float(np.max(abs(values["anomalous_contribution"]))),
            "n_h2_frozen": int(q.N_H2), "omega_lo_frozen": .001, "omega_hi_frozen": 12.,
            "tail_panels_frozen": True, "dark_basis_read_once": True,
            "background_repropagated": False, "physical_changes": False,
            "production_qualification": False, "infinite_band_or_BZ_qualification": False,
            "frequency_block": frequency_block, "time_block": time_block,
            "timing_s": timings, "total_wall_s": time.perf_counter()-start,
            "row_sha256": prep.sha256_file(output.with_suffix(".npz")),
            "atol": ATOL, "rtol": RTOL}
    prep.write_json_atomic(output, meta)
    print(json.dumps({key: meta[key] for key in ("n_panel", "n_nodes", "all_checks_pass",
                     "n_trace", "normal_ccr_maxabs", "anomalous_maxabs", "total_wall_s")}), flush=True)
    if n_panel == 96 and not passed:
        raise AssertionError("96 coordinate/output regression failed; 128 remains blocked")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-panel", choices=(96, 128), type=int, required=True)
    ap.add_argument("--frequency-block", type=int, default=128)
    ap.add_argument("--time-block", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.n_panel, frequency_block=args.frequency_block, time_block=args.time_block,
        smoke=args.smoke)


if __name__ == "__main__":
    main()

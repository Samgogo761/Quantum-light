#!/usr/bin/env python3
"""GH7 tail-model gate. Staged: does not auto-approve k40 or full GH7.

Frozen rules (2026-08-16 follow-up):
  - expected case set from FREEZE must match the root exactly
  - SUCCESS, run_status=PASS, output_sha256.txt, provenance
  - occupation: unique complete nk×nk×nb grid, trace0 = nv·Nk, first+last snapshots
  - k20→k40 Jones: mixed abs/rel; H2/5/7/9 hard, H10 diagnostic
  - every non-PASS status returns a non-zero exit code
  - dt/2 is not an accepted stage (not implemented)
  - +N-only probes do not test GH7 CEP-π covariance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "a0_layerA_m88full" / "gh7_tail"))

from diagnose_gh5_quadrature import jones_decomp
from gh7_nodes import case_name, expected_case_names, peer_k20_name
from rank_full112_gh3_candidates import amp as jones_amp
from rank_full112_gh3_candidates import load_node_modes

OCC_LO = -1.0e-8
OCC_HI = 1.0 + 1.0e-8
TRACE_RTOL = 1.0e-8
EDGE_ABS = 1.0e-3
EDGE_REL = 0.01
N_EDGE_BANDS = 8
NV = 84
NB = 112
JONES_HARD = (2, 5, 7, 9)
JONES_RTOL = 0.05
JONES_PHASE = 0.1
JONES_ETA = 1.0e-4
JONES_ATOL = JONES_ETA * JONES_RTOL  # 5e-6; independent weak absolute threshold
EXC_FLOOR = 1.0e-6
H10_RTOL = 0.10
GATE_VERSION = "gh7_tail_model_v3_20260816"
IGNORE_ROOT_NAMES = {
    "GH7_TAIL_K20.json",
    "GH7_TAIL_K40.json",
    "job_id.txt",
}


def load_freeze(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tokens(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    for tok in ("nan", "inf"):
        if tok in text:
            raise ValueError(f"NaN/Inf token in {path}")


def expected_occ_its(nt: int, stride: int) -> list[int]:
    if nt < 1 or stride < 1:
        raise ValueError(f"bad nt/stride: nt={nt} stride={stride}")
    its = list(range(1, nt + 1, stride))
    if nt not in its:
        its.append(nt)
    return its


def parse_run_nml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def ival(name: str) -> int:
        for raw in text.splitlines():
            line = raw.split("!", 1)[0].strip()
            if line.lower().startswith(name.lower()) and "=" in line:
                rhs = line.split("=", 1)[1].strip().strip(",").strip()
                return int(float(rhs))
        raise ValueError(f"{path}: missing {name}")

    nkx = ival("nkx")
    nky = ival("nky")
    if nkx != nky:
        raise ValueError(f"{path}: nkx={nkx} != nky={nky}")
    return {"nk": nkx, "occ_stride": ival("occ_stride")}


def parse_band_occupation(
    path: Path,
    *,
    nk: int,
    nb: int = NB,
    nv: int = NV,
    expected_its: list[int] | None = None,
) -> dict:
    """Parse occupation_band_kt.dat and require a unique complete k×band grid."""
    _finite_tokens(path)
    expected_nk = nk * nk
    expected_trace0 = float(nv * expected_nk)
    expected_per_snap = expected_nk * nb
    snaps: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"{path}:{lineno}: expected 6 columns")
            it = int(fields[0])
            t_fs = float(fields[1])
            ikx = int(fields[2])
            iky = int(fields[3])
            band = int(fields[4])
            occ = float(fields[5])
            if not math.isfinite(occ):
                raise ValueError(f"non-finite occupation in {path}:{lineno}")
            if occ < OCC_LO or occ > OCC_HI:
                raise ValueError(f"occupation {occ} outside [{OCC_LO}, {OCC_HI}] in {path}:{lineno}")
            rec = snaps.setdefault(
                it,
                {
                    "t_fs": t_fs,
                    "trace": 0.0,
                    "exc": 0.0,
                    "edge": 0.0,
                    "edge_max": 0.0,
                    "keys": set(),
                    "n_rows": 0,
                },
            )
            key = (ikx, iky, band)
            if key in rec["keys"]:
                raise ValueError(f"duplicate (ikx,iky,band)={key} at it={it} in {path}")
            rec["keys"].add(key)
            rec["n_rows"] += 1
            rec["trace"] += occ
            if band > nv:
                rec["exc"] += occ
            if band > nb - N_EDGE_BANDS:
                rec["edge"] += occ
                rec["edge_max"] = max(rec["edge_max"], occ)

    if not snaps:
        raise ValueError(f"no occupation rows in {path}")

    want_grid = {(ix, iy, b) for ix in range(1, nk + 1) for iy in range(1, nk + 1) for b in range(1, nb + 1)}
    for it, rec in snaps.items():
        missing = want_grid - rec["keys"]
        extra = rec["keys"] - want_grid
        if missing or extra or rec["n_rows"] != expected_per_snap:
            raise ValueError(
                f"it={it}: occupancy grid incomplete; "
                f"rows={rec['n_rows']} expected={expected_per_snap} "
                f"missing={len(missing)} extra={len(extra)}"
            )
        rec.pop("keys")

    its = sorted(snaps)
    if expected_its is not None:
        if its != list(expected_its):
            raise ValueError(f"snapshot its={its} != expected {list(expected_its)}")
    elif len(its) < 2:
        raise ValueError(f"{path}: need first and last snapshots, got {its}")

    tr0 = snaps[its[0]]["trace"]
    if abs(tr0 - expected_trace0) / max(expected_trace0, 1.0e-300) > TRACE_RTOL:
        raise ValueError(f"initial trace {tr0} != {nv}*{expected_nk}={expected_trace0}")

    max_drift = 0.0
    max_edge_rel = 0.0
    max_edge_abs = 0.0
    for it in its:
        rec = snaps[it]
        drift = abs(rec["trace"] - tr0) / max(abs(tr0), 1.0e-300)
        max_drift = max(max_drift, drift)
        max_edge_abs = max(max_edge_abs, rec["edge_max"])
        if rec["exc"] >= EXC_FLOOR:
            rel = rec["edge"] / rec["exc"]
            max_edge_rel = max(max_edge_rel, rel)
    return {
        "n_snapshots": len(its),
        "its": its,
        "it_first": its[0],
        "it_last": its[-1],
        "n_k": expected_nk,
        "n_bands": nb,
        "rows_per_snapshot": expected_per_snap,
        "trace0": tr0,
        "expected_trace0": expected_trace0,
        "max_trace_rel_drift": max_drift,
        "max_edge_abs": max_edge_abs,
        "max_edge_rel": max_edge_rel,
        "pass_range": True,
        "pass_grid": True,
        "pass_trace": max_drift <= TRACE_RTOL,
        "pass_trace0": True,
        "pass_edge_abs": max_edge_abs <= EDGE_ABS,
        "pass_edge_rel": max_edge_rel <= EDGE_REL,
        "exc_floor": EXC_FLOOR,
    }


def jones_from_modes(path: Path) -> dict[int, tuple[complex, complex]]:
    modes = load_node_modes(path)
    out: dict[int, tuple[complex, complex]] = {}
    for (_nid, order), (jx, jy, _w) in modes.items():
        out[order] = (jx, jy)
    return out


def jones_scales_from_pairs(pairs: list[tuple[dict, dict]]) -> dict[int, float]:
    """Per-harmonic A_H = max ||J|| over the campaign pairs."""
    scales: dict[int, float] = {}
    for order in JONES_HARD + (10,):
        amps: list[float] = []
        for left, right in pairs:
            if order in left:
                amps.append(jones_amp(*left[order]))
            if order in right:
                amps.append(jones_amp(*right[order]))
        scales[order] = max(amps) if amps else 0.0
    return scales


def compare_jones(a: dict, b: dict, scales: dict[int, float] | None = None) -> dict:
    if scales is None:
        scales = jones_scales_from_pairs([(a, b)])
    rows = {}
    hard_fail = []
    for order in JONES_HARD + (10,):
        if order not in a or order not in b:
            rows[str(order)] = {"status": "MISSING"}
            if order in JONES_HARD:
                hard_fail.append(order)
            continue
        de = jones_decomp(a[order][0], a[order][1], b[order][0], b[order][1])
        rtol = H10_RTOL if order == 10 else JONES_RTOL
        a_h = float(scales.get(order, 0.0))
        ref = max(de["amp_a"], de["amp_b"])
        delta = de["vec_rel"] * max(de["amp_a"], 1.0e-300)
        strong = ref >= JONES_ETA * max(a_h, 1.0e-300)
        e_abs = delta / max(a_h, 1.0e-300)
        if strong:
            e_rel = delta / max(ref, 1.0e-300)
            ok = e_rel <= rtol and de["phase_rad"] <= JONES_PHASE
        else:
            ok = e_abs <= JONES_ATOL
        mixed = {
            "strong": strong,
            "A_H": a_h,
            "e_abs": e_abs,
            "tau_abs": JONES_ATOL,
            "pass": ok,
        }
        de["mixed"] = mixed
        de["pass"] = ok
        de["rtol"] = rtol
        de["gate"] = "mixed_abs_rel_campaign_AH"
        rows[str(order)] = de
        if order in JONES_HARD and not ok:
            hard_fail.append(order)
    return {"orders": rows, "pass": not hard_fail, "hard_fail": hard_fail}


def _kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def verify_output_sha256(path: Path) -> None:
    root = path.parent
    checked = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        target = root / name
        if not target.is_file():
            raise ValueError(f"output_sha256 lists missing file: {name}")
        got = sha256_file(target)
        if got != digest:
            raise ValueError(f"output SHA mismatch for {name}: {got} != {digest}")
        checked += 1
    if checked < 1:
        raise ValueError(f"empty output_sha256.txt: {path}")


def list_case_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    cases = []
    for path in sorted(root.iterdir()):
        if path.name in IGNORE_ROOT_NAMES or path.name.endswith(".runlock"):
            continue
        if path.is_dir():
            cases.append(path)
    return cases


def audit_k20_dir(outdir: Path, freeze: dict, probe: dict, nk: int) -> dict:
    status: dict = {
        "outdir": str(outdir),
        "case": outdir.name,
        "stage": "k20",
        "kind": probe.get("kind"),
        "id": probe.get("id"),
        "theta_deg": probe.get("theta_deg"),
    }
    try:
        if not (outdir / "SUCCESS").is_file():
            raise ValueError("missing SUCCESS")
        run_status = _kv_file(outdir / "run_status.txt")
        if run_status.get("status") != "PASS":
            raise ValueError(f"run_status={run_status.get('status')}")
        sha_list = outdir / "output_sha256.txt"
        if not sha_list.is_file():
            raise ValueError("missing output_sha256.txt")
        verify_output_sha256(sha_list)
        meta = _kv_file(outdir / "run_metadata.txt")
        want_bin = freeze["pinned_binary_sha256"]
        want_tb = freeze["tb_plus_sha256"]
        want_man = probe["manifest_sha256"]
        if meta.get("binary_sha256") != want_bin:
            raise ValueError(f"binary SHA {meta.get('binary_sha256')} != {want_bin}")
        if meta.get("tb_sha256") != want_tb:
            raise ValueError(f"TB SHA {meta.get('tb_sha256')} != {want_tb}")
        if meta.get("nodes_sha256") != want_man:
            raise ValueError(f"manifest SHA {meta.get('nodes_sha256')} != {want_man}")
        if meta.get("propagate_ids") != str(probe["id"]):
            raise ValueError(f"propagate_ids={meta.get('propagate_ids')} != {probe['id']}")
        base_head = freeze.get("freeze_pin_base_head") or freeze.get("git_head")
        if base_head and base_head != "TO_BE_PINNED":
            if meta.get("freeze_pin_base_head") not in {None, base_head}:
                raise ValueError(
                    f"freeze_pin_base_head {meta.get('freeze_pin_base_head')} != {base_head}"
                )
        want_freeze_sha = freeze.get("freeze_sha256")
        if want_freeze_sha and meta.get("freeze_sha256") not in {None, want_freeze_sha}:
            raise ValueError(f"freeze_sha256 {meta.get('freeze_sha256')} != {want_freeze_sha}")
        actual_head = meta.get("campaign_actual_head") or meta.get("git_head")
        if not actual_head or actual_head == "UNAVAILABLE":
            raise ValueError("missing campaign_actual_head/git_head in run_metadata")
        status["campaign_actual_head"] = actual_head
        status["freeze_pin_base_head"] = meta.get("freeze_pin_base_head") or base_head
        modes = outdir / "HHG_nodes_modes.dat"
        if not modes.is_file():
            raise ValueError("missing HHG_nodes_modes.dat")
        _finite_tokens(modes)
        band = outdir / "occupation_band_kt.dat"
        if not band.is_file():
            raise ValueError("missing occupation_band_kt.dat")
        nml = outdir / "input.nml"
        parsed = parse_run_nml(nml) if nml.is_file() else {"nk": nk, "occ_stride": int(freeze.get("occ_stride", 336))}
        if parsed["nk"] != nk:
            raise ValueError(f"nml nk={parsed['nk']} != {nk}")
        expected_its = None
        if nml.is_file():
            from hhg_fft_utils import parse_input_nml

            nt = int(parse_input_nml(nml)["nt"])
            expected_its = expected_occ_its(nt, parsed["occ_stride"])
        occ = parse_band_occupation(band, nk=nk, expected_its=expected_its)
        status["occupation"] = occ
        if not (occ["pass_trace"] and occ["pass_edge_abs"] and occ["pass_edge_rel"] and occ["pass_grid"]):
            raise ValueError("occupation gate failed")
    except (OSError, ValueError) as exc:
        status["status"] = "FAIL"
        status["pass"] = False
        status["reason"] = str(exc)
        if "missing" in str(exc).lower() or "incomplete" in str(exc).lower():
            status["status"] = "INCOMPLETE"
        return status
    status["status"] = "PASS"
    status["pass"] = True
    return status


def same_case_set(found: list[str], expected: list[str]) -> bool:
    return sorted(found) == sorted(expected) and len(found) == len(set(found))


def _status_from_rows(rows: list[dict], expected: list[str], found: list[str]) -> str:
    if not same_case_set(found, expected):
        return "INCOMPLETE" if set(found) != set(expected) or len(found) < len(expected) else "FAIL"
    if any(row.get("status") == "INCOMPLETE" for row in rows):
        return "INCOMPLETE"
    if rows and all(row.get("status") == "PASS" for row in rows):
        return "PASS"
    return "FAIL"


def gate_exit_code(status: str) -> int:
    return 0 if status == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("k20", "k40"), required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--k20-root", type=Path)
    ap.add_argument("--k40-root", type=Path)
    ap.add_argument("--nk", type=int, default=20)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    freeze = load_freeze(args.freeze)
    probes = freeze["plusN_probes"]
    report: dict = {
        "gate": GATE_VERSION,
        "stage": args.stage,
        "auto_chain": False,
        "cep_pi_tested": False,
        "cep_pi_note": "+N-only probes do not test GH7 CEP-π covariance",
        "dt2_stage": "not_implemented",
        "n_expected": len(probes),
        "thresholds": {
            "occ_lo": OCC_LO,
            "occ_hi": OCC_HI,
            "trace_rtol": TRACE_RTOL,
            "edge_abs": EDGE_ABS,
            "edge_rel": EDGE_REL,
            "jones_rtol_hard": JONES_RTOL,
            "jones_phase_rad": JONES_PHASE,
            "jones_eta": JONES_ETA,
            "jones_atol": JONES_ATOL,
            "exc_floor": EXC_FLOOR,
            "h10_rtol": H10_RTOL,
            "jones_gate": "mixed_abs_rel_campaign_AH",
        },
    }
    sha_file = args.freeze.with_name("FREEZE.sha256")
    if sha_file.is_file():
        freeze["freeze_sha256"] = sha_file.read_text(encoding="utf-8").strip()
    if args.stage == "k20":
        if args.k20_root is None:
            raise SystemExit("--k20-root required")
        expected = expected_case_names(probes, args.nk)
        found = [p.name for p in list_case_dirs(args.k20_root)]
        report["expected_cases"] = expected
        report["found_cases"] = found
        by_name = {case_name(p, args.nk): p for p in probes}
        rows = []
        if not same_case_set(found, expected):
            extra = sorted(set(found) - set(expected))
            missing = sorted(set(expected) - set(found))
            report["missing_cases"] = missing
            report["extra_cases"] = extra
            for name in missing:
                rows.append({"case": name, "status": "INCOMPLETE", "pass": False, "reason": "case directory missing"})
            for name in extra:
                rows.append({"case": name, "status": "FAIL", "pass": False, "reason": "unexpected case directory"})
        for name in expected:
            if name not in found:
                continue
            rows.append(audit_k20_dir(args.k20_root / name, freeze, by_name[name], args.nk))
        report["cases"] = rows
        report["status"] = _status_from_rows(rows, expected, found)
        heads = sorted({r.get("campaign_actual_head") for r in rows if r.get("campaign_actual_head")})
        report["campaign_actual_heads"] = heads
        if report["status"] == "PASS" and len(heads) != 1:
            report["status"] = "FAIL"
            report["reason"] = f"campaign_actual_head not unique: {heads}"
        report["approve_k40"] = report["status"] == "PASS"
    else:
        if args.k20_root is None or args.k40_root is None:
            raise SystemExit("--k20-root and --k40-root required")
        k20_report_path = args.k20_root / "GH7_TAIL_K20.json"
        k20 = json.loads(k20_report_path.read_text(encoding="utf-8")) if k20_report_path.is_file() else None
        if k20 is None or k20.get("status") != "PASS":
            report["status"] = "BLOCKED"
            report["reason"] = "k20 occupation gate is not PASS; do not run/score k40"
        else:
            expected = expected_case_names(probes, 40)
            found = [p.name for p in list_case_dirs(args.k40_root)]
            report["expected_cases"] = expected
            report["found_cases"] = found
            by_name = {case_name(p, 40): p for p in probes}
            pairs = []
            if not same_case_set(found, expected):
                report["missing_cases"] = sorted(set(expected) - set(found))
                report["extra_cases"] = sorted(set(found) - set(expected))
            jones_pairs: list[tuple[dict, dict]] = []
            ready: list[tuple[str, dict, dict]] = []
            for name in expected:
                d40 = args.k40_root / name
                d20 = args.k20_root / peer_k20_name(name)
                if name not in found:
                    pairs.append({"case": name, "status": "INCOMPLETE", "pass": False, "reason": "k40 directory missing"})
                    continue
                audit = audit_k20_dir(d40, freeze, by_name[name], 40)
                if audit.get("status") != "PASS":
                    pairs.append({"case": name, "k20": str(d20), **audit})
                    continue
                if not (d20 / "HHG_nodes_modes.dat").is_file():
                    pairs.append({"case": name, "k20": str(d20), "status": "INCOMPLETE", "pass": False, "reason": "k20 peer missing"})
                    continue
                j20 = jones_from_modes(d20 / "HHG_nodes_modes.dat")
                j40 = jones_from_modes(d40 / "HHG_nodes_modes.dat")
                jones_pairs.append((j20, j40))
                ready.append((name, j20, j40))
            scales = jones_scales_from_pairs(jones_pairs) if jones_pairs else {}
            report["jones_A_H"] = scales
            for name, j20, j40 in ready:
                cmpj = compare_jones(j20, j40, scales)
                pairs.append({"case": name, "k20": str(args.k20_root / peer_k20_name(name)), "status": "PASS" if cmpj["pass"] else "FAIL", **cmpj})
            report["pairs"] = pairs
            if not same_case_set(found, expected):
                report["status"] = "INCOMPLETE" if set(found) != set(expected) or len(found) < len(expected) else "FAIL"
            elif not pairs:
                report["status"] = "INCOMPLETE"
            elif any(p.get("status") == "INCOMPLETE" for p in pairs):
                report["status"] = "INCOMPLETE"
            elif all(p.get("status") == "PASS" for p in pairs):
                report["status"] = "PASS"
            else:
                report["status"] = "FAIL"
            heads = sorted({p.get("campaign_actual_head") for p in pairs if p.get("campaign_actual_head")})
            report["campaign_actual_heads"] = heads
            if report["status"] == "PASS" and len(heads) != 1:
                report["status"] = "FAIL"
                report["reason"] = f"campaign_actual_head not unique: {heads}"
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return gate_exit_code(str(report.get("status")))


if __name__ == "__main__":
    raise SystemExit(main())

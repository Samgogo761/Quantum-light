#!/usr/bin/env python3
"""k40 model vs k-convergence compare for classical HHG_complex.dat.

Computes, for channels J+, J-, Je=(J++J-)/2, Jo=(J+-J-)/2 and
orders H2/5/7/9/10 (all gated at --gate, default 0.15):

  eps_model(k40) = ||J88_k40 - J112_k40|| / (||J112_k40|| + delta)
  eps_k^M        = ||J^M_k40 - J^M_k20|| / (||J^M_k40|| + delta),  M in {88,112}

Jo UNRESOLVED uses a Jo-specific continuum background:

  1. Build the full differential spectrum Jo(ω)=(J+(ω)-J-(ω))/2 on matched bins
  2. Take the median |Jo| on off-harmonic bins as background
  3. SNR = |Jo(Hn)| / bg_Jo

Do NOT average J+/J- continuum floors for Jo SNR (that overestimates noise).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ORDERS = (2, 5, 7, 9, 10)
GATE_ORDERS = (2, 5, 7, 9, 10)  # H10 is a formal gate for this campaign
CHANNELS = ("Jplus", "Jminus", "Je", "Jo")


def load_all_bins(path: Path) -> list[tuple[float, complex, complex]]:
    rows: list[tuple[float, complex, complex]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"{path}: expected >=6 columns, got {len(fields)}")
        h = float(fields[0])
        jx = complex(float(fields[2]), float(fields[3]))
        jy = complex(float(fields[4]), float(fields[5]))
        if not all(math.isfinite(x) for x in (h, jx.real, jx.imag, jy.real, jy.imag)):
            raise ValueError(f"{path}: non-finite near harmonic {h}")
        rows.append((h, jx, jy))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def nearest(rows: list[tuple[float, complex, complex]], orders: tuple[int, ...] = ORDERS):
    selected: dict[int, dict[str, Any]] = {}
    for order in orders:
        best = min(rows, key=lambda item: abs(item[0] - float(order)))
        selected[order] = {
            "harmonic_order_bin": best[0],
            "delta_order": abs(best[0] - float(order)),
            "jx": best[1],
            "jy": best[2],
        }
    return selected


def continuum_background(rows: list[tuple[float, complex, complex]], h_max: float = 12.0) -> float:
    """Median |J| on bins far from integer harmonics (FFT continuum proxy)."""
    amps: list[float] = []
    for h, jx, jy in rows:
        if h <= 0.5 or h > h_max:
            continue
        frac = abs(h - round(h))
        if 0.30 <= frac <= 0.50:
            amps.append(math.sqrt(abs(jx) ** 2 + abs(jy) ** 2))
    if len(amps) < 8:
        all_amps = [
            math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)
            for h, jx, jy in rows
            if 0.5 < h <= h_max
        ]
        all_amps.sort()
        if not all_amps:
            return 1.0e-30
        return all_amps[max(0, len(all_amps) // 4)]
    return statistics.median(amps)


def combine_spectrum_pm(
    plus_rows: list[tuple[float, complex, complex]],
    minus_rows: list[tuple[float, complex, complex]],
    sign: float,
    *,
    bin_tol: float = 1.0e-10,
) -> list[tuple[float, complex, complex]]:
    """Build Je (sign=+1) or Jo (sign=-1) on the matched FFT grid.

    Requires identical harmonic_order bin lists (as in paired ±N classical runs).
    """
    if len(plus_rows) != len(minus_rows):
        raise SystemExit(
            f"±N FFT length mismatch: +N={len(plus_rows)} -N={len(minus_rows)}"
        )
    out: list[tuple[float, complex, complex]] = []
    for (hp, jxp, jyp), (hm, jxm, jym) in zip(plus_rows, minus_rows):
        if abs(hp - hm) > bin_tol:
            raise SystemExit(f"±N FFT bin mismatch: +N h={hp} -N h={hm}")
        out.append(
            (
                hp,
                0.5 * (jxp + sign * jxm),
                0.5 * (jyp + sign * jym),
            )
        )
    return out


def amp(jx: complex, jy: complex) -> float:
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def rel_err(a_jx: complex, a_jy: complex, b_jx: complex, b_jy: complex, delta: float) -> float:
    num = math.sqrt(abs(a_jx - b_jx) ** 2 + abs(a_jy - b_jy) ** 2)
    den = amp(b_jx, b_jy) + delta
    return num / den


def combine_pm(plus: dict[int, dict], minus: dict[int, dict], sign: float) -> dict[int, dict]:
    """sign=+1 -> Je=(Jp+Jm)/2; sign=-1 -> Jo=(Jp-Jm)/2 (nearest-integer view)."""
    out: dict[int, dict] = {}
    for order in ORDERS:
        if abs(plus[order]["harmonic_order_bin"] - minus[order]["harmonic_order_bin"]) > 1e-12:
            raise SystemExit(f"bin mismatch at H{order} for Je/Jo")
        out[order] = {
            "harmonic_order_bin": plus[order]["harmonic_order_bin"],
            "delta_order": plus[order]["delta_order"],
            "jx": 0.5 * (plus[order]["jx"] + sign * minus[order]["jx"]),
            "jy": 0.5 * (plus[order]["jy"] + sign * minus[order]["jy"]),
        }
    return out


def build_channels(plus: dict[int, dict], minus: dict[int, dict]) -> dict[str, dict[int, dict]]:
    return {
        "Jplus": plus,
        "Jminus": minus,
        "Je": combine_pm(plus, minus, +1.0),
        "Jo": combine_pm(plus, minus, -1.0),
    }


def judge_verdict(
    *,
    model_gate_worst: float,
    k_worst_m88: float,
    k_worst_f112: float,
    gate: float,
) -> str:
    """Coarse decision string for the node."""
    model_fail = model_gate_worst > gate
    k_ok_m88 = k_worst_m88 <= gate
    k_ok_f112 = k_worst_f112 <= gate
    if (not model_fail) and k_ok_m88 and k_ok_f112:
        return "PASS_model_and_k"
    if model_fail and k_ok_m88 and k_ok_f112:
        return "MODEL_SPACE_GAP_k_converged"
    if (not model_fail) and (not k_ok_m88 or not k_ok_f112):
        return "K_NOT_CONVERGED_model_ok"
    if model_fail and (not k_ok_m88 or not k_ok_f112):
        return "BOTH_MODEL_AND_K_ISSUES"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--node-id", type=int, required=True)
    ap.add_argument("--weight", type=float)
    ap.add_argument("--intensity", type=float)
    ap.add_argument("--phi-cep-deg", type=float)
    ap.add_argument("--m88-k40-plus", type=Path, required=True)
    ap.add_argument("--m88-k40-minus", type=Path, required=True)
    ap.add_argument("--f112-k40-plus", type=Path, required=True)
    ap.add_argument("--f112-k40-minus", type=Path, required=True)
    ap.add_argument("--m88-k20-plus", type=Path, required=True)
    ap.add_argument("--m88-k20-minus", type=Path, required=True)
    ap.add_argument("--f112-k20-plus", type=Path, required=True)
    ap.add_argument("--f112-k20-minus", type=Path, required=True)
    ap.add_argument("--gate", type=float, default=0.15)
    ap.add_argument("--delta", type=float, default=1.0e-30)
    ap.add_argument("--max-bin-delta", type=float, default=0.25)
    ap.add_argument("--snr-min", type=float, default=3.0, help="Jo UNRESOLVED if SNR < this")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    def load_pair(plus_p: Path, minus_p: Path):
        plus_rows = load_all_bins(plus_p)
        minus_rows = load_all_bins(minus_p)
        plus = nearest(plus_rows)
        minus = nearest(minus_rows)
        for order in ORDERS:
            if plus[order]["delta_order"] > args.max_bin_delta:
                raise SystemExit(f"{plus_p}: H{order} bin too far")
            if minus[order]["delta_order"] > args.max_bin_delta:
                raise SystemExit(f"{minus_p}: H{order} bin too far")
        jo_rows = combine_spectrum_pm(plus_rows, minus_rows, -1.0)
        je_rows = combine_spectrum_pm(plus_rows, minus_rows, +1.0)
        bg_legacy = 0.5 * (
            continuum_background(plus_rows) + continuum_background(minus_rows)
        )
        bg_jo = continuum_background(jo_rows)
        bg_je = continuum_background(je_rows)
        return (
            build_channels(plus, minus),
            {
                "bg_jo": bg_jo,
                "bg_je": bg_je,
                "bg_legacy_avg_pm": bg_legacy,
                "bg_jplus": continuum_background(plus_rows),
                "bg_jminus": continuum_background(minus_rows),
            },
        )

    m88_k40, bg_m88 = load_pair(args.m88_k40_plus, args.m88_k40_minus)
    f112_k40, bg_f112 = load_pair(args.f112_k40_plus, args.f112_k40_minus)
    m88_k20, _ = load_pair(args.m88_k20_plus, args.m88_k20_minus)
    f112_k20, _ = load_pair(args.f112_k20_plus, args.f112_k20_minus)

    # Jo SNR uses full112 Jo-specific continuum (primary); also report M88 Jo bg.
    bg_jo_f112 = bg_f112["bg_jo"]
    bg_jo_m88 = bg_m88["bg_jo"]
    legacy_overest_id = {
        "legacy_over_jo_f112": (bg_f112["bg_legacy_avg_pm"] / (bg_jo_f112 + args.delta)),
        "legacy_over_jo_m88": (bg_m88["bg_legacy_avg_pm"] / (bg_jo_m88 + args.delta)),
    }

    rows_out: list[dict[str, Any]] = []
    model_gate_worst = 0.0
    k_worst_m88 = 0.0
    k_worst_f112 = 0.0
    gate_fail = False
    unresolved_jo = False
    resolved_jo_orders: list[int] = []

    for ch in CHANNELS:
        for order in ORDERS:
            a = m88_k40[ch][order]
            b = f112_k40[ch][order]
            a20 = m88_k20[ch][order]
            b20 = f112_k20[ch][order]
            amp_m88 = amp(a["jx"], a["jy"])
            amp_f112 = amp(b["jx"], b["jy"])
            eps_model = rel_err(a["jx"], a["jy"], b["jx"], b["jy"], args.delta)
            eps_k_m88 = rel_err(a20["jx"], a20["jy"], a["jx"], a["jy"], args.delta)
            eps_k_f112 = rel_err(b20["jx"], b20["jy"], b["jx"], b["jy"], args.delta)

            status = "PASS"
            note = ""
            applies = order in GATE_ORDERS
            snr = None
            bg_used = None

            if ch == "Jo":
                # Correct: SNR vs Jo(ω) continuum, not avg of J± floors.
                bg_used = bg_jo_f112
                snr = amp_f112 / (bg_used + args.delta)
                snr_m88 = amp_m88 / (bg_jo_m88 + args.delta)
                if snr < args.snr_min:
                    status = "UNRESOLVED"
                    unresolved_jo = True
                    note = (
                        "Jo SNR < snr_min vs Jo-specific continuum; "
                        "relative %% not used for FAIL"
                    )
                else:
                    resolved_jo_orders.append(order)
                    # Resolved Jo participates in the model gate.
                    if applies and eps_model > args.gate:
                        status = "FAIL"
                        gate_fail = True
                        model_gate_worst = max(model_gate_worst, eps_model)
                    elif applies:
                        model_gate_worst = max(model_gate_worst, eps_model)
            else:
                if applies and eps_model > args.gate:
                    status = "FAIL"
                    gate_fail = True
                    model_gate_worst = max(model_gate_worst, eps_model)
                elif applies:
                    model_gate_worst = max(model_gate_worst, eps_model)

            if status != "UNRESOLVED":
                k_worst_m88 = max(k_worst_m88, eps_k_m88)
                k_worst_f112 = max(k_worst_f112, eps_k_f112)

            row: dict[str, Any] = {
                "channel": ch,
                "order": order,
                "eps_model_k40": eps_model,
                "eps_k_m88": eps_k_m88,
                "eps_k_full112": eps_k_f112,
                "amp_m88_k40": amp_m88,
                "amp_full112_k40": amp_f112,
                "bin_m88": a["harmonic_order_bin"],
                "bin_full112": b["harmonic_order_bin"],
                "gate_applies": applies,
                "status": status,
                "note": note,
            }
            if ch == "Jo":
                row.update(
                    {
                        "fft_background_jo_full112": bg_jo_f112,
                        "fft_background_jo_m88": bg_jo_m88,
                        "fft_background_legacy_avg_pm_full112": bg_f112["bg_legacy_avg_pm"],
                        "snr_full112": snr,
                        "snr_m88": snr_m88,
                        "legacy_bg_overestimate_factor_full112": legacy_overest_id[
                            "legacy_over_jo_f112"
                        ],
                    }
                )
            rows_out.append(row)

            snr_txt = f"SNR={snr:.2f}" if snr is not None else "SNR=n/a"
            print(
                f"{args.label}:{ch} H{order} "
                f"eps_model={eps_model:.6e} "
                f"eps_k_m88={eps_k_m88:.6e} eps_k_112={eps_k_f112:.6e} "
                f"amp88={amp_m88:.3e} amp112={amp_f112:.3e} "
                f"{snr_txt} [{status}]"
            )

    overall = "FAIL" if gate_fail else ("PASS_WITH_UNRESOLVED_JO" if unresolved_jo else "PASS")
    verdict = judge_verdict(
        model_gate_worst=model_gate_worst,
        k_worst_m88=k_worst_m88,
        k_worst_f112=k_worst_f112,
        gate=args.gate,
    )
    print(
        f"{args.label}_OVERALL={overall} verdict={verdict} "
        f"gate_worst_eps_model={model_gate_worst:.6e} "
        f"k_worst_m88={k_worst_m88:.6e} k_worst_full112={k_worst_f112:.6e} "
        f"gate={args.gate} H10_gated=1 "
        f"Jo_bg_f112={bg_jo_f112:.3e} Jo_resolved={resolved_jo_orders}"
    )

    payload = {
        "label": args.label,
        "node_id": args.node_id,
        "weight": args.weight,
        "intensity_Wcm2": args.intensity,
        "phi_cep_deg": args.phi_cep_deg,
        "overall": overall,
        "verdict": verdict,
        "gate": args.gate,
        "gate_orders": list(GATE_ORDERS),
        "gate_worst_eps_model": model_gate_worst,
        "k_worst_eps_m88": k_worst_m88,
        "k_worst_eps_full112": k_worst_f112,
        "jo_background": {
            "full112": bg_jo_f112,
            "m88": bg_jo_m88,
            "legacy_avg_pm_full112": bg_f112["bg_legacy_avg_pm"],
            "legacy_avg_pm_m88": bg_m88["bg_legacy_avg_pm"],
            "legacy_overestimate_factor_full112": legacy_overest_id["legacy_over_jo_f112"],
            "legacy_overestimate_factor_m88": legacy_overest_id["legacy_over_jo_m88"],
            "method": "median_|Jo|_on_off_harmonic_bins_of_Jo(omega)",
        },
        "jo_resolved_orders": resolved_jo_orders,
        "snr_min": args.snr_min,
        "rows": rows_out,
        "definitions": {
            "eps_model": "||J88_k40-J112_k40||/(||J112_k40||+delta)",
            "eps_k": "||J_k40-J_k20||/(||J_k40||+delta)",
            "Je": "(J++J-)/2",
            "Jo": "(J+-J-)/2",
            "UNRESOLVED": "Jo SNR < snr_min vs Jo-specific continuum background",
        },
        "note": "NOT_formal_GH5; H10 is a formal gate; Jo bg fixed 2026-08-04",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0 if overall in ("PASS", "PASS_WITH_UNRESOLVED_JO") else 1


if __name__ == "__main__":
    raise SystemExit(main())

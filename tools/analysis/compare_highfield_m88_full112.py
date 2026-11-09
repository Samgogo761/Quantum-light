#!/usr/bin/env python3
"""Compare M88-full vs full112 classical complex currents at target harmonics.

Reads ``HHG_complex.dat`` written by the classical path
(``bsv_enabled=.false.``, ``save_complex_hhg=.true.``):

    harmonic_order  omega  ReJx ImJx ReJy ImJy  |Jx|^2_s  |Jy|^2_s  total

For each integer order n in {2,5,7,9,10}, the FFT bin with harmonic_order
closest to n is selected.  Relative complex-vector error:

    eps_n = sqrt(|dJx|^2+|dJy|^2) / (sqrt(|J112x|^2+|J112y|^2) + delta)

H2/H5/H7/H9 use --gate (default 0.15); H10 is diagnostic only.

Also reports amplitude ratio, phase / polarization diagnostics, and optionally
compares odd-channel currents

    Jo = (J(+N) - J(-N)) / 2

when --m88-minus / --full112-minus are supplied.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ORDERS = (2, 5, 7, 9, 10)
GATE_ORDERS = (2, 5, 7, 9)


def load_complex_nearest(path: Path, orders: tuple[int, ...] = ORDERS):
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
            raise ValueError(f"{path}: non-finite value near harmonic {h}")
        rows.append((h, jx, jy))
    if not rows:
        raise ValueError(f"{path}: no data rows")

    selected: dict[int, dict] = {}
    for order in orders:
        best = min(rows, key=lambda item: abs(item[0] - float(order)))
        selected[order] = {
            "harmonic_order_bin": best[0],
            "delta_order": abs(best[0] - float(order)),
            "jx": best[1],
            "jy": best[2],
        }
    return selected


def _amp(jx: complex, jy: complex) -> float:
    return math.sqrt(abs(jx) ** 2 + abs(jy) ** 2)


def _phase_deg(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real))


def _circ_delta_deg(a: float, b: float) -> float:
    """Signed circular difference a-b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _pol_fracs(jx: complex, jy: complex, delta: float) -> tuple[float, float]:
    ax2 = abs(jx) ** 2
    ay2 = abs(jy) ** 2
    den = ax2 + ay2 + delta
    return ax2 / den, ay2 / den


def _pol_angle_deg(jx: complex, jy: complex) -> float:
    """Composition angle from intensity axes: atan2(|Jy|, |Jx|) in degrees."""
    return math.degrees(math.atan2(abs(jy), abs(jx) + 0.0))


def _pol_overlap(jx_a: complex, jy_a: complex, jx_b: complex, jy_b: complex, delta: float) -> float:
    """|J_a · J_b*| / (|J_a||J_b|); 1 = identical Jones vector up to global phase."""
    num = abs(jx_a * jx_b.conjugate() + jy_a * jy_b.conjugate())
    den = (_amp(jx_a, jy_a) * _amp(jx_b, jy_b)) + delta
    return num / den


def _cplx_dict(jx: complex, jy: complex, delta: float) -> dict[str, Any]:
    amp = _amp(jx, jy)
    fx, fy = _pol_fracs(jx, jy, delta)
    return {
        "jx_re": jx.real,
        "jx_im": jx.imag,
        "jy_re": jy.real,
        "jy_im": jy.imag,
        "amp": amp,
        "phase_x_deg": _phase_deg(jx),
        "phase_y_deg": _phase_deg(jy),
        "pol_frac_x": fx,
        "pol_frac_y": fy,
        "pol_angle_deg": _pol_angle_deg(jx, jy),
    }


def compare_pair(
    a: dict[int, dict],
    b: dict[int, dict],
    *,
    gate: float,
    delta: float,
    max_bin_delta: float,
    path_a: Path,
    path_b: Path,
    channel: str,
) -> tuple[list[dict[str, Any]], float, float, bool]:
    rows: list[dict[str, Any]] = []
    gate_worst = 0.0
    h10_eps = 0.0
    gate_fail = False
    for order in ORDERS:
        if a[order]["delta_order"] > max_bin_delta:
            raise SystemExit(
                f"{path_a}: nearest bin to H{order} is "
                f"{a[order]['harmonic_order_bin']} (delta={a[order]['delta_order']})"
            )
        if b[order]["delta_order"] > max_bin_delta:
            raise SystemExit(
                f"{path_b}: nearest bin to H{order} is "
                f"{b[order]['harmonic_order_bin']} (delta={b[order]['delta_order']})"
            )
        jx_a, jy_a = a[order]["jx"], a[order]["jy"]
        jx_b, jy_b = b[order]["jx"], b[order]["jy"]
        num = math.sqrt(abs(jx_a - jx_b) ** 2 + abs(jy_a - jy_b) ** 2)
        den = _amp(jx_b, jy_b) + delta
        eps = num / den
        amp_a = _amp(jx_a, jy_a)
        amp_b = _amp(jx_b, jy_b)
        amp_ratio = amp_a / (amp_b + delta)
        dph_x = _circ_delta_deg(_phase_deg(jx_a), _phase_deg(jx_b))
        dph_y = _circ_delta_deg(_phase_deg(jy_a), _phase_deg(jy_b))
        # Intensity-weighted mean phase of the 2D Jones vector difference signal.
        # Use global phase of the complex overlap for a single phase offset estimate.
        overlap_c = jx_a * jx_b.conjugate() + jy_a * jy_b.conjugate()
        phase_offset_deg = _phase_deg(overlap_c) if abs(overlap_c) > delta else float("nan")
        pol_a = _pol_angle_deg(jx_a, jy_a)
        pol_b = _pol_angle_deg(jx_b, jy_b)
        dpol = pol_a - pol_b
        fx_a, fy_a = _pol_fracs(jx_a, jy_a, delta)
        fx_b, fy_b = _pol_fracs(jx_b, jy_b, delta)
        in_gate = order in GATE_ORDERS
        ok = (eps <= gate) if in_gate else True
        if in_gate:
            gate_worst = max(gate_worst, eps)
            if not ok:
                gate_fail = True
        else:
            h10_eps = eps
        row = {
            "channel": channel,
            "order": order,
            "eps": eps,
            "amp_ratio_m88_over_full112": amp_ratio,
            "amp_m88": amp_a,
            "amp_full112": amp_b,
            "phase_offset_deg": phase_offset_deg,
            "dphase_x_deg": dph_x,
            "dphase_y_deg": dph_y,
            "pol_angle_m88_deg": pol_a,
            "pol_angle_full112_deg": pol_b,
            "dpol_angle_deg": dpol,
            "pol_frac_xy_m88": [fx_a, fy_a],
            "pol_frac_xy_full112": [fx_b, fy_b],
            "pol_overlap": _pol_overlap(jx_a, jy_a, jx_b, jy_b, delta),
            "m88": _cplx_dict(jx_a, jy_a, delta),
            "full112": _cplx_dict(jx_b, jy_b, delta),
            "bin_m88": a[order]["harmonic_order_bin"],
            "bin_full112": b[order]["harmonic_order_bin"],
            "gate_applies": in_gate,
            "pass": ok,
        }
        rows.append(row)
        tag = "GATE" if in_gate else "diag"
        print(
            f"{channel} H{order} eps={eps:.6e} amp_ratio={amp_ratio:.4f} "
            f"phase_off={phase_offset_deg:.2f}deg dpol={dpol:.2f}deg "
            f"pol_ov={row['pol_overlap']:.4f} "
            f"fxfy_m88=({fx_a:.3f},{fy_a:.3f}) fxfy_112=({fx_b:.3f},{fy_b:.3f}) "
            f"bin=({a[order]['harmonic_order_bin']:.4f},"
            f"{b[order]['harmonic_order_bin']:.4f}) [{tag}] "
            f"{'PASS' if ok else 'FAIL'}"
        )
    return rows, gate_worst, h10_eps, gate_fail


def _jo_from_pm(plus: dict[int, dict], minus: dict[int, dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for order in ORDERS:
        if abs(plus[order]["harmonic_order_bin"] - minus[order]["harmonic_order_bin"]) > 1e-12:
            raise SystemExit(
                f"Jo bin mismatch at H{order}: "
                f"+N={plus[order]['harmonic_order_bin']} -N={minus[order]['harmonic_order_bin']}"
            )
        out[order] = {
            "harmonic_order_bin": plus[order]["harmonic_order_bin"],
            "delta_order": plus[order]["delta_order"],
            "jx": 0.5 * (plus[order]["jx"] - minus[order]["jx"]),
            "jy": 0.5 * (plus[order]["jy"] - minus[order]["jy"]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m88", type=Path, required=True, help="M88 +N HHG_complex.dat")
    ap.add_argument("--full112", type=Path, required=True, help="full112 +N HHG_complex.dat")
    ap.add_argument("--m88-minus", type=Path, help="M88 -N HHG_complex.dat (enables Jo)")
    ap.add_argument("--full112-minus", type=Path, help="full112 -N HHG_complex.dat (enables Jo)")
    ap.add_argument("--label", default="spot")
    ap.add_argument("--node-id", type=int)
    ap.add_argument("--weight", type=float)
    ap.add_argument("--intensity", type=float, help="I_drive W/cm^2")
    ap.add_argument("--phi-cep-deg", type=float)
    ap.add_argument("--delta", type=float, default=1.0e-30)
    ap.add_argument("--gate", type=float, default=0.15, help="H2/H5/H7/H9 gate")
    ap.add_argument(
        "--max-bin-delta",
        type=float,
        default=0.25,
        help="reject if nearest FFT bin is farther than this from integer order",
    )
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    a_plus = load_complex_nearest(args.m88)
    b_plus = load_complex_nearest(args.full112)
    rows_plus, gate_worst_plus, h10_plus, fail_plus = compare_pair(
        a_plus,
        b_plus,
        gate=args.gate,
        delta=args.delta,
        max_bin_delta=args.max_bin_delta,
        path_a=args.m88,
        path_b=args.full112,
        channel=f"{args.label}:Jplus",
    )

    rows_jo: list[dict[str, Any]] = []
    gate_worst_jo = 0.0
    h10_jo = 0.0
    fail_jo = False
    jo_enabled = args.m88_minus is not None or args.full112_minus is not None
    if jo_enabled:
        if args.m88_minus is None or args.full112_minus is None:
            raise SystemExit("Jo mode requires both --m88-minus and --full112-minus")
        a_minus = load_complex_nearest(args.m88_minus)
        b_minus = load_complex_nearest(args.full112_minus)
        a_jo = _jo_from_pm(a_plus, a_minus)
        b_jo = _jo_from_pm(b_plus, b_minus)
        rows_jo, gate_worst_jo, h10_jo, fail_jo = compare_pair(
            a_jo,
            b_jo,
            gate=args.gate,
            delta=args.delta,
            max_bin_delta=args.max_bin_delta,
            path_a=args.m88_minus,
            path_b=args.full112_minus,
            channel=f"{args.label}:Jo",
        )

    overall_fail = fail_plus or fail_jo
    overall = "PASS" if not overall_fail else "FAIL"
    # Preserve prior key name as alias of gate_worst on +N, but also emit explicit fields.
    gate_worst_eps = max(gate_worst_plus, gate_worst_jo) if jo_enabled else gate_worst_plus
    h10_diag_eps = max(h10_plus, h10_jo) if jo_enabled else h10_plus

    print(
        f"{args.label}_OVERALL={overall} "
        f"gate_worst_eps={gate_worst_eps:.6e} "
        f"h10_diag_eps={h10_diag_eps:.6e} "
        f"gate={args.gate} "
        f"jo_enabled={1 if jo_enabled else 0}"
    )
    if args.node_id is not None:
        print(f"{args.label}_node_id={args.node_id}")
    if args.weight is not None:
        print(f"{args.label}_weight={args.weight:.8e}")
    if args.intensity is not None:
        print(f"{args.label}_intensity_Wcm2={args.intensity:.8e}")
    if args.phi_cep_deg is not None:
        print(f"{args.label}_phi_cep_deg={args.phi_cep_deg:.8f}")

    payload = {
        "label": args.label,
        "node_id": args.node_id,
        "weight": args.weight,
        "intensity_Wcm2": args.intensity,
        "phi_cep_deg": args.phi_cep_deg,
        "overall": overall,
        "gate": args.gate,
        "gate_worst_eps": gate_worst_eps,
        "h10_diag_eps": h10_diag_eps,
        "plusN": {
            "gate_worst_eps": gate_worst_plus,
            "h10_diag_eps": h10_plus,
            "overall": "PASS" if not fail_plus else "FAIL",
            "rows": rows_plus,
        },
        "Jo": None
        if not jo_enabled
        else {
            "gate_worst_eps": gate_worst_jo,
            "h10_diag_eps": h10_jo,
            "overall": "PASS" if not fail_jo else "FAIL",
            "rows": rows_jo,
            "definition": "Jo=(J(+N)-J(-N))/2",
        },
        # backward-compatible alias (gate harmonics only, not H10)
        "worst_eps": gate_worst_eps,
        "drive_mode": "classical_fixed_field",
        "note": (
            "Jo validated"
            if jo_enabled
            else "plus-N only; Jo not validated without ±N pair"
        ),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

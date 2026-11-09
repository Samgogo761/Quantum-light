#!/usr/bin/env python3
"""Strictly combine shared-node +N/-N layer-A HHG amplitudes.

The input files are ``HHG_nodes_modes.dat`` tables with columns

    id weight I phi order ReJx ImJx ReJy ImJy |J|^2_scaled

For every node ``s`` and harmonic ``n`` this script forms the classical
trajectory amplitudes

    Je[s,n] = (Jplus[s,n] + Jminus[s,n]) / 2
    Jo[s,n] = (Jplus[s,n] - Jminus[s,n]) / 2.

With ``--summary`` it also writes, separately for every harmonic, weighted
complex means and incoherent/coherent sums. These are layer-A classical
trajectory diagnostics; they are not photon ``g^(2)``, squeezing, or
entanglement observables.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


N_COLUMNS = 10
COL_ID = 0
COL_WEIGHT = 1
COL_INTENSITY = 2
COL_PHASE = 3
COL_ORDER = 4
COL_RE_JX = 5
COL_IM_JX = 6
COL_RE_JY = 7
COL_IM_JY = 8
COL_SCALED_POWER = 9

TWOPI = 2.0 * np.pi


def _fail(message: str) -> None:
    raise ValueError(message)


def require_identical_node_manifests(plus_nodes: Path, minus_nodes: Path) -> str:
    """Hard gate: ±N must share one byte-identical node manifest."""
    if not plus_nodes.is_file():
        _fail(f"plus nodes manifest missing: {plus_nodes}")
    if not minus_nodes.is_file():
        _fail(f"minus nodes manifest missing: {minus_nodes}")
    plus_sha = file_sha256(plus_nodes)
    minus_sha = file_sha256(minus_nodes)
    if plus_sha != minus_sha:
        _fail(
            "±N node-manifest SHA-256 mismatch: "
            f"{plus_nodes}={plus_sha}, {minus_nodes}={minus_sha}"
        )
    return plus_sha


def _phase_close(a: float, b: float, *, atol: float) -> bool:
    """Circular closeness on [0, 2π), treating 0 and 2π as identical."""
    delta = abs(((a - b + np.pi) % TWOPI) - np.pi)
    return bool(delta <= atol)


def load_node_manifest(path: Path) -> dict[int, tuple[float, float, float]]:
    """Return {id: (weight, I, phi)} from a quantum-light node manifest."""
    if not path.is_file():
        _fail(f"nodes manifest missing: {path}")
    mapping: dict[int, tuple[float, float, float]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 6:
            _fail(f"{path}:{lineno}: expected 6 columns, got {len(fields)}")
        try:
            node_id = int(fields[0])
            weight = float(fields[1])
            intensity = float(fields[4])
            phase = float(fields[5]) % TWOPI
            re_a = float(fields[2])
            im_a = float(fields[3])
        except ValueError as exc:
            _fail(f"{path}:{lineno}: invalid numeric field ({exc})")
        values = (weight, re_a, im_a, intensity, phase)
        if not all(np.isfinite(v) for v in values):
            _fail(f"{path}:{lineno}: NaN/Inf is forbidden")
        if node_id in mapping:
            _fail(f"{path}:{lineno}: duplicate node id {node_id}")
        if weight < 0.0 or intensity < 0.0:
            _fail(f"{path}:{lineno}: negative weight/intensity")
        mapping[node_id] = (weight, intensity, phase)
    if not mapping:
        _fail(f"{path}: no data rows")
    return mapping


# Keys that must agree across ±N production runs. TB hash / domain labels may differ.
REQUIRED_METADATA_MATCH_KEYS = (
    "nk",
    "dt",
    "T2_cycles",
    "wvl_nm",
    "squeeze_r",
    "squeeze_theta_deg",
    "harmonics",
    "source_sha256",
    "binary_sha256",
    "nodes_sha256",
)

# Alias map: accept either short or long names written by different scripts.
_METADATA_ALIASES = {
    "nk": ("nk", "nkx", "nky"),
    "dt": ("dt",),
    "T2_cycles": ("T2_cycles", "t2_cycles"),
    "wvl_nm": ("wvl_nm", "lambda_nm", "wavelength_nm"),
    "squeeze_r": ("squeeze_r", "bsv_squeeze_r", "r"),
    "squeeze_theta_deg": ("squeeze_theta_deg", "bsv_squeeze_theta_deg", "theta_deg"),
    "harmonics": ("harmonics", "bsv_harmonics"),
    "source_sha256": ("source_sha256",),
    "binary_sha256": ("binary_sha256",),
    "nodes_sha256": ("nodes_sha256",),
}


def load_run_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        _fail(f"run metadata missing: {path}")
    meta: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            _fail(f"{path}:{lineno}: expected key=value")
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    if not meta:
        _fail(f"{path}: empty metadata")
    return meta


def _meta_get(meta: dict[str, str], logical: str) -> str | None:
    for key in _METADATA_ALIASES[logical]:
        if key in meta:
            return meta[key]
    return None


def compare_run_metadata_pair(
    plus_meta_path: Path,
    minus_meta_path: Path,
    *,
    required_keys: Iterable[str] = REQUIRED_METADATA_MATCH_KEYS,
) -> dict[str, str]:
    """Require selected physics/provenance keys to match across ±N.

    TB hash and magnetic-domain labels are intentionally *not* required to
    match, because +N/-N use different TB files by construction.
    """
    plus = load_run_metadata(plus_meta_path)
    minus = load_run_metadata(minus_meta_path)
    matched: dict[str, str] = {}
    for logical in required_keys:
        pv = _meta_get(plus, logical)
        mv = _meta_get(minus, logical)
        if pv is None:
            _fail(f"{plus_meta_path}: missing required metadata key for {logical}")
        if mv is None:
            _fail(f"{minus_meta_path}: missing required metadata key for {logical}")
        # For nk: if both nkx and nky exist, require them equal within each file
        # and equal across domains; compare the resolved value strings.
        if logical == "nk":
            for side, meta, path in (
                ("plus", plus, plus_meta_path),
                ("minus", minus, minus_meta_path),
            ):
                if "nkx" in meta and "nky" in meta and meta["nkx"] != meta["nky"]:
                    _fail(f"{path}: nkx={meta['nkx']} != nky={meta['nky']}")
        if pv != mv:
            _fail(
                f"±N run_metadata mismatch on {logical}: "
                f"{plus_meta_path}={pv!r}, {minus_meta_path}={mv!r}"
            )
        matched[logical] = pv
    return matched


def crosscheck_modes_against_manifest(
    modes: np.ndarray,
    modes_path: Path,
    manifest: dict[int, tuple[float, float, float]],
    *,
    rtol: float,
    atol: float,
    phase_atol: float,
) -> None:
    """Require every modes row's (id,w,I,phi) to match the shared manifest."""
    mode_ids = {int(round(value)) for value in modes[:, COL_ID]}
    man_ids = set(manifest)
    if mode_ids != man_ids:
        missing = sorted(man_ids - mode_ids)
        extra = sorted(mode_ids - man_ids)
        _fail(
            f"{modes_path}: node-id set differs from manifest; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for row in modes:
        node_id = int(round(row[COL_ID]))
        weight_m, intensity_m, phase_m = manifest[node_id]
        if not _close(float(row[COL_WEIGHT]), weight_m, rtol=rtol, atol=atol):
            _fail(
                f"{modes_path}: node {node_id} weight mismatch vs manifest: "
                f"{float(row[COL_WEIGHT]):.17e} vs {weight_m:.17e}"
            )
        if not _close(float(row[COL_INTENSITY]), intensity_m, rtol=rtol, atol=atol):
            _fail(
                f"{modes_path}: node {node_id} intensity mismatch vs manifest: "
                f"{float(row[COL_INTENSITY]):.17e} vs {intensity_m:.17e}"
            )
        if not _phase_close(float(row[COL_PHASE]), phase_m, atol=phase_atol):
            _fail(
                f"{modes_path}: node {node_id} phase mismatch vs manifest: "
                f"{float(row[COL_PHASE]):.17e} vs {phase_m:.17e}"
            )


def _integer_column(values: np.ndarray, label: str, path: Path) -> np.ndarray:
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        bad = values[values != rounded][0]
        _fail(f"{path}: {label} must be integer-valued; found {bad!r}")
    return rounded.astype(np.int64)


def _close(a: float, b: float, *, rtol: float, atol: float) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def load_modes(
    path: Path,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    weight_sum_tol: float = 5.0e-8,
) -> np.ndarray:
    """Load and validate one per-node harmonic-amplitude table."""
    try:
        data = np.loadtxt(path, comments="#", ndmin=2)
    except (OSError, ValueError) as exc:
        _fail(f"cannot read {path}: {exc}")

    if data.ndim != 2 or data.shape[0] == 0:
        _fail(f"{path}: no data rows")
    if data.shape[1] != N_COLUMNS:
        _fail(
            f"{path}: expected exactly {N_COLUMNS} columns "
            f"(id weight I phi order ReJx ImJx ReJy ImJy |J|^2_scaled), "
            f"found {data.shape[1]}"
        )
    if not np.all(np.isfinite(data)):
        row, col = np.argwhere(~np.isfinite(data))[0]
        _fail(f"{path}: non-finite value at data row {row + 1}, column {col + 1}")

    ids = _integer_column(data[:, COL_ID], "node id", path)
    orders = _integer_column(data[:, COL_ORDER], "harmonic order", path)
    if np.any(orders <= 0):
        _fail(f"{path}: harmonic orders must be positive")
    if np.any(data[:, COL_WEIGHT] < 0.0):
        _fail(f"{path}: negative quadrature weight is not allowed")
    if np.any(data[:, COL_INTENSITY] < 0.0):
        _fail(f"{path}: negative drive intensity is not allowed")
    if np.any(data[:, COL_SCALED_POWER] < 0.0):
        _fail(f"{path}: negative stored |J|^2_scaled is not allowed")

    keys = [(int(node_id), int(order)) for node_id, order in zip(ids, orders)]
    seen: set[tuple[int, int]] = set()
    for key in keys:
        if key in seen:
            _fail(f"{path}: duplicate (node id, harmonic order) key {key}")
        seen.add(key)

    # Every harmonic must be evaluated on exactly the same node set.
    unique_orders = sorted(set(int(value) for value in orders))
    reference_nodes = {int(ids[i]) for i in np.flatnonzero(orders == unique_orders[0])}
    for order in unique_orders[1:]:
        nodes = {int(ids[i]) for i in np.flatnonzero(orders == order)}
        if nodes != reference_nodes:
            missing = sorted(reference_nodes - nodes)
            extra = sorted(nodes - reference_nodes)
            _fail(
                f"{path}: harmonic H{order} does not use the common node set; "
                f"missing={missing}, extra={extra}"
            )

    # A node's quadrature metadata must not depend on harmonic order.
    node_metadata: dict[int, tuple[float, float, float]] = {}
    for row, node_id in zip(data, ids):
        metadata = (
            float(row[COL_WEIGHT]),
            float(row[COL_INTENSITY]),
            float(row[COL_PHASE]),
        )
        old = node_metadata.get(int(node_id))
        if old is None:
            node_metadata[int(node_id)] = metadata
            continue
        labels = ("weight", "intensity", "phase")
        for label, old_value, new_value in zip(labels, old, metadata):
            if not _close(old_value, new_value, rtol=rtol, atol=atol):
                _fail(
                    f"{path}: node {int(node_id)} has harmonic-dependent {label}: "
                    f"{old_value:.17e} vs {new_value:.17e}"
                )

    # The main program uses raw weights, so require every harmonic's weights to
    # be normalized rather than silently renormalizing them here.
    for order in unique_orders:
        mask = orders == order
        weight_sum = float(np.sum(data[mask, COL_WEIGHT], dtype=np.float64))
        if abs(weight_sum - 1.0) > weight_sum_tol:
            _fail(
                f"{path}: H{order} quadrature weights sum to {weight_sum:.17e}, "
                f"outside 1 +/- {weight_sum_tol:.3e}"
            )

    return data


def _keyed_indices(data: np.ndarray) -> dict[tuple[int, int], int]:
    return {
        (int(round(row[COL_ID])), int(round(row[COL_ORDER]))): index
        for index, row in enumerate(data)
    }


def align_and_validate_pair(
    plus: np.ndarray,
    minus: np.ndarray,
    plus_path: Path,
    minus_path: Path,
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Return minus rows aligned to plus rows after strict metadata checks."""
    p_index = _keyed_indices(plus)
    m_index = _keyed_indices(minus)
    p_keys = set(p_index)
    m_keys = set(m_index)
    if p_keys != m_keys:
        missing = sorted(p_keys - m_keys)
        extra = sorted(m_keys - p_keys)
        _fail(
            "+N/-N node-harmonic key mismatch; "
            f"missing from {minus_path}={missing}, extra in {minus_path}={extra}"
        )

    # Preserve +N row order while allowing a harmless row-order difference.
    plus_keys = [
        (int(round(row[COL_ID])), int(round(row[COL_ORDER]))) for row in plus
    ]
    aligned_minus = minus[[m_index[key] for key in plus_keys], :]
    labels = ("weight", "drive intensity", "drive phase")
    columns = (COL_WEIGHT, COL_INTENSITY, COL_PHASE)
    for p_row, m_row in zip(plus, aligned_minus):
        key = (int(round(p_row[COL_ID])), int(round(p_row[COL_ORDER])))
        for label, column in zip(labels, columns):
            if not _close(
                float(p_row[column]), float(m_row[column]), rtol=rtol, atol=atol
            ):
                _fail(
                    f"+N/-N {label} mismatch for key {key}: "
                    f"{plus_path}={p_row[column]:.17e}, "
                    f"{minus_path}={m_row[column]:.17e}"
                )
    return aligned_minus


def _complex_components(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    jx = data[:, COL_RE_JX] + 1j * data[:, COL_IM_JX]
    jy = data[:, COL_RE_JY] + 1j * data[:, COL_IM_JY]
    return jx, jy


def _power(jx: np.ndarray | complex, jy: np.ndarray | complex) -> np.ndarray | float:
    return np.abs(jx) ** 2 + np.abs(jy) ** 2


def _weighted_mean(weights: np.ndarray, values: np.ndarray) -> complex:
    return complex(np.sum(weights * values, dtype=np.complex128))


def _nonnegative_variance(ics: float, cs: float, label: str) -> float:
    variance = float(ics - cs)
    tolerance = 2.0e-12 * max(abs(ics), abs(cs), 1.0)
    if variance < -tolerance:
        _fail(
            f"{label}: ICS-CS={variance:.17e} is negative beyond roundoff; "
            "check weights and amplitudes"
        )
    return max(variance, 0.0)


def write_node_output(
    path: Path,
    plus: np.ndarray,
    jx_even: np.ndarray,
    jy_even: np.ndarray,
    jx_odd: np.ndarray,
    jy_odd: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Shared-node +N/-N classical trajectory decomposition\n")
        stream.write("# Je=(J(+N)+J(-N))/2; Jo=(J(+N)-J(-N))/2\n")
        stream.write("# absJo2=|Jox|^2+|Joy|^2; absJe2=|Jex|^2+|Jey|^2 (unscaled)\n")
        stream.write(
            "# id weight I phi order ReJxo ImJxo ReJyo ImJyo absJo2 "
            "ReJxe ImJxe ReJye ImJye absJe2\n"
        )
        for index, row in enumerate(plus):
            jo2 = float(_power(jx_odd[index], jy_odd[index]))
            je2 = float(_power(jx_even[index], jy_even[index]))
            prefix_values: Iterable[float] = (
                row[COL_WEIGHT], row[COL_INTENSITY], row[COL_PHASE]
            )
            amplitude_values: Iterable[float] = (
                jx_odd[index].real, jx_odd[index].imag,
                jy_odd[index].real, jy_odd[index].imag, jo2,
                jx_even[index].real, jx_even[index].imag,
                jy_even[index].real, jy_even[index].imag, je2,
            )
            stream.write(
                f"{int(round(row[COL_ID])):8d} "
                + " ".join(f"{float(value):.17e}" for value in prefix_values)
                + f" {int(round(row[COL_ORDER])):6d} "
                + " ".join(f"{float(value):.17e}" for value in amplitude_values)
                + "\n"
            )


def write_summary(
    path: Path,
    plus: np.ndarray,
    jx_plus: np.ndarray,
    jy_plus: np.ndarray,
    jx_minus: np.ndarray,
    jy_minus: np.ndarray,
    jx_even: np.ndarray,
    jy_even: np.ndarray,
    jx_odd: np.ndarray,
    jy_odd: np.ndarray,
) -> None:
    """Write per-harmonic weighted layer-A domain and Je/Jo diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    orders = plus[:, COL_ORDER].astype(np.int64)
    unique_orders = sorted(set(int(value) for value in orders))

    complex_labels = [
        "mu_plus_x", "mu_plus_y", "mu_minus_x", "mu_minus_y",
        "Je_bar_x", "Je_bar_y", "Jo_bar_x", "Jo_bar_y",
    ]
    scalar_labels = [
        "domain_plus_ICS", "domain_plus_CS", "domain_plus_classical_trajectory_variance",
        "domain_minus_ICS", "domain_minus_CS", "domain_minus_classical_trajectory_variance",
        "trajectory_even_ICS", "trajectory_even_CS", "trajectory_even_classical_trajectory_variance",
        "trajectory_odd_ICS", "trajectory_odd_CS", "trajectory_odd_classical_trajectory_variance",
        "domain_mean_ICS", "even_plus_odd_ICS", "ICS_identity_abs_error",
    ]
    header_columns = ["order", "n_nodes", "weight_sum"]
    for label in complex_labels:
        header_columns.extend((f"Re_{label}", f"Im_{label}"))
    header_columns.extend(scalar_labels)

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Weighted +N/-N layer-A classical trajectory summary (unscaled current units)\n")
        stream.write("# mu_plus/minus = sum_s w_s J_s(+/-N)\n")
        stream.write("# Je_bar=(mu_plus+mu_minus)/2; Jo_bar=(mu_plus-mu_minus)/2\n")
        stream.write("# domain_ICS=sum_s w_s |J_s|^2; domain_CS=|sum_s w_s J_s|^2\n")
        stream.write("# trajectory_even/odd_ICS=sum_s w_s |Je/o_s|^2\n")
        stream.write("# All quantities are classical-mixture diagnostics, not photon statistics.\n")
        stream.write("# " + " ".join(header_columns) + "\n")

        for order in unique_orders:
            mask = orders == order
            weights = plus[mask, COL_WEIGHT]
            weight_sum = float(np.sum(weights, dtype=np.float64))

            mu_px = _weighted_mean(weights, jx_plus[mask])
            mu_py = _weighted_mean(weights, jy_plus[mask])
            mu_mx = _weighted_mean(weights, jx_minus[mask])
            mu_my = _weighted_mean(weights, jy_minus[mask])
            je_bar_x = _weighted_mean(weights, jx_even[mask])
            je_bar_y = _weighted_mean(weights, jy_even[mask])
            jo_bar_x = _weighted_mean(weights, jx_odd[mask])
            jo_bar_y = _weighted_mean(weights, jy_odd[mask])

            plus_ics = float(np.sum(weights * _power(jx_plus[mask], jy_plus[mask])))
            plus_cs = float(_power(mu_px, mu_py))
            minus_ics = float(np.sum(weights * _power(jx_minus[mask], jy_minus[mask])))
            minus_cs = float(_power(mu_mx, mu_my))
            even_ics = float(np.sum(weights * _power(jx_even[mask], jy_even[mask])))
            even_cs = float(_power(je_bar_x, je_bar_y))
            odd_ics = float(np.sum(weights * _power(jx_odd[mask], jy_odd[mask])))
            odd_cs = float(_power(jo_bar_x, jo_bar_y))

            plus_var = _nonnegative_variance(plus_ics, plus_cs, f"H{order} +N")
            minus_var = _nonnegative_variance(minus_ics, minus_cs, f"H{order} -N")
            even_var = _nonnegative_variance(even_ics, even_cs, f"H{order} even")
            odd_var = _nonnegative_variance(odd_ics, odd_cs, f"H{order} odd")
            domain_mean_ics = 0.5 * (plus_ics + minus_ics)
            even_plus_odd_ics = even_ics + odd_ics
            identity_error = abs(domain_mean_ics - even_plus_odd_ics)

            complex_values = (
                mu_px, mu_py, mu_mx, mu_my,
                je_bar_x, je_bar_y, jo_bar_x, jo_bar_y,
            )
            scalar_values = (
                plus_ics, plus_cs, plus_var,
                minus_ics, minus_cs, minus_var,
                even_ics, even_cs, even_var,
                odd_ics, odd_cs, odd_var,
                domain_mean_ics, even_plus_odd_ics, identity_error,
            )
            fields = [str(order), str(int(np.count_nonzero(mask))), f"{weight_sum:.17e}"]
            for value in complex_values:
                fields.extend((f"{value.real:.17e}", f"{value.imag:.17e}"))
            fields.extend(f"{value:.17e}" for value in scalar_values)
            stream.write(" ".join(fields) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly combine shared-node +N/-N HHG_nodes_modes.dat files."
    )
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--minus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="optional per-harmonic weighted summary output",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--weight-sum-tol", type=float, default=5.0e-8)
    parser.add_argument(
        "--plus-nodes", type=Path, default=None,
        help="plus-domain nodes_manifest.dat (SHA-256 must match --minus-nodes)",
    )
    parser.add_argument(
        "--minus-nodes", type=Path, default=None,
        help="minus-domain nodes_manifest.dat (SHA-256 must match --plus-nodes)",
    )
    args = parser.parse_args()

    if args.rtol < 0.0 or args.atol < 0.0 or args.weight_sum_tol < 0.0:
        raise SystemExit("rtol, atol, and weight-sum-tol must be non-negative")
    if (args.plus_nodes is None) ^ (args.minus_nodes is None):
        raise SystemExit("provide both --plus-nodes and --minus-nodes, or neither")

    try:
        nodes_sha = None
        if args.plus_nodes is not None and args.minus_nodes is not None:
            nodes_sha = require_identical_node_manifests(
                args.plus_nodes, args.minus_nodes
            )
            print(f"nodes_manifest_sha256={nodes_sha}")

        plus = load_modes(
            args.plus, rtol=args.rtol, atol=args.atol,
            weight_sum_tol=args.weight_sum_tol,
        )
        minus = load_modes(
            args.minus, rtol=args.rtol, atol=args.atol,
            weight_sum_tol=args.weight_sum_tol,
        )
        minus_aligned = align_and_validate_pair(
            plus, minus, args.plus, args.minus,
            rtol=args.rtol, atol=args.atol,
        )

        jx_plus, jy_plus = _complex_components(plus)
        jx_minus, jy_minus = _complex_components(minus_aligned)
        jx_even = 0.5 * (jx_plus + jx_minus)
        jy_even = 0.5 * (jy_plus + jy_minus)
        jx_odd = 0.5 * (jx_plus - jx_minus)
        jy_odd = 0.5 * (jy_plus - jy_minus)

        write_node_output(args.out, plus, jx_even, jy_even, jx_odd, jy_odd)
        print(f"wrote per-node Je/Jo: {args.out}")
        if args.summary is not None:
            write_summary(
                args.summary, plus,
                jx_plus, jy_plus, jx_minus, jy_minus,
                jx_even, jy_even, jx_odd, jy_odd,
            )
            if nodes_sha is not None:
                with args.summary.open("a", encoding="utf-8") as stream:
                    stream.write(f"# nodes_manifest_sha256 = {nodes_sha}\n")
            print(f"wrote weighted summary: {args.summary}")
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

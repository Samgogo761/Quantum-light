#!/usr/bin/env python3
"""Pure-Python GH7 ES25.17 manifests + corner / wide-axis antinode selection.

Does not hardcode node IDs. Two deterministic selectors:

- corner: absolute-max-I antipode pair (worst-field stress)
- wide: max |proj_major|, then min |proj_minor| (quadrature-weight tail)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PI = math.pi
TWOPI = 2.0 * PI
ALPHA0_TOL = 1.0e-14
I_BAR = 1.0e11
SQUEEZE_R = 2.5
EXPECTED_I_MAX = 2.813e12
I_MAX_RTOL = 5.0e-3

def _gh7_moment_matched() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Physicist GH7 abscissae with weights matched to ∫ e^{-x^2} x^{2k} dx.

    The 16-digit table copied into Fortran/make_gh_nodes.py does not sum to
    √π at 1e-5, so Q0 covariance fails the production 2e-6 gate. Manifests
    use these moment-matched weights; the solver still reads from_file.
    """
    # Keep the Fortran/make_gh_nodes.py abscissae so I_max stays 2.813e12.
    xs = (
        2.651961356835233,
        1.673551628767471,
        0.8162878828589647,
    )
    # ∫ e^{-x^2} x^{2k} dx = Γ(k+1/2)
    rhs = (
        math.sqrt(PI),
        0.5 * math.sqrt(PI),
        0.75 * math.sqrt(PI),
        1.875 * math.sqrt(PI),
    )
    a = []
    for pwr in (0, 2, 4, 6):
        row = [2.0 * (xs[i] ** pwr) for i in range(3)]
        row.append(1.0 if pwr == 0 else 0.0)
        a.append(row)
    m = [list(row) + [rhs[i]] for i, row in enumerate(a)]
    n = 4
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[piv] = m[piv], m[i]
        div = m[i][i]
        for j in range(i, n + 1):
            m[i][j] /= div
        for r in range(n):
            if r == i:
                continue
            fac = m[r][i]
            for j in range(i, n + 1):
                m[r][j] -= fac * m[i][j]
    w_pos = (m[0][4], m[1][4], m[2][4])
    w0 = m[3][4]
    x = (-xs[0], -xs[1], -xs[2], 0.0, xs[2], xs[1], xs[0])
    w = (w_pos[0], w_pos[1], w_pos[2], w0, w_pos[2], w_pos[1], w_pos[0])
    return x, w


GH7_X, GH7_W = _gh7_moment_matched()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash a text file after normalizing CRLF to LF (git/server checkout)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_VQ_sv(r: float, theta_s: float) -> tuple[tuple[float, float], tuple[float, float]]:
    vx = 0.5 * (1.0 + math.exp(-2.0 * r))
    vp = 0.5 * (1.0 + math.exp(+2.0 * r))
    c = math.cos(0.5 * theta_s)
    s = math.sin(0.5 * theta_s)
    return (
        (c * c * vx + s * s * vp, c * s * (vx - vp)),
        (c * s * (vx - vp), s * s * vx + c * c * vp),
    )


def chol2(a: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    a11 = a[0][0]
    a21 = 0.5 * (a[1][0] + a[0][1])
    a22 = a[1][1]
    l00 = math.sqrt(a11)
    l10 = a21 / l00
    l11 = math.sqrt(a22 - l10 * l10)
    return ((l00, 0.0), (l10, l11))


def alpha_to_drive(alpha: complex, i_bar: float, nbar: float) -> tuple[float, float]:
    i_mean = 2.0 * i_bar
    eabs2 = nbar + 1.0
    abs2 = abs(alpha) ** 2
    intensity = i_mean * abs2 / eabs2
    if abs2 <= ALPHA0_TOL * ALPHA0_TOL:
        return 0.0, 0.0
    phi = math.atan2(alpha.imag, alpha.real)
    if phi < 0.0:
        phi += TWOPI
    return intensity, phi


def _alpha_key(alpha: complex) -> tuple[float, float]:
    return (round(alpha.real, 12), round(alpha.imag, 12))


def enforce_antipodes(rows: list[tuple], i_bar: float, nbar: float) -> list[tuple]:
    alphas = [complex(r[2], r[3]) for r in rows]
    weights = [float(r[1]) for r in rows]
    ids = [int(r[0]) for r in rows]
    by_key: dict[tuple[float, float], int] = {}
    for i, a in enumerate(alphas):
        key = _alpha_key(a)
        if key in by_key:
            raise ValueError(f"duplicate alpha key {key}")
        by_key[key] = i
    used: set[int] = set()
    out_alpha = list(alphas)
    out_w = list(weights)
    for i, a in enumerate(alphas):
        if i in used:
            continue
        if abs(a) <= ALPHA0_TOL:
            out_alpha[i] = 0.0 + 0.0j
            used.add(i)
            continue
        j = by_key.get(_alpha_key(-a))
        if j is None:
            raise ValueError(f"no antipode for id={ids[i]}")
        src, dst = (i, j) if ids[i] <= ids[j] else (j, i)
        out_alpha[dst] = -out_alpha[src]
        w_mean = 0.5 * (out_w[src] + out_w[dst])
        out_w[src] = w_mean
        out_w[dst] = w_mean
        used.add(src)
        used.add(dst)
    if len(used) != len(rows):
        raise ValueError("antipode cover incomplete")
    wsum = sum(out_w)
    out_w = [w / wsum for w in out_w]
    new_rows = []
    for i in range(len(rows)):
        a = out_alpha[i]
        intensity, phi = alpha_to_drive(a, i_bar, nbar)
        new_rows.append((ids[i], out_w[i], a.real, a.imag, intensity, phi))
    by_key = {_alpha_key(complex(r[2], r[3])): idx for idx, r in enumerate(new_rows)}
    final = list(new_rows)
    seen: set[int] = set()
    for i, row in enumerate(new_rows):
        if i in seen:
            continue
        a = complex(row[2], row[3])
        if abs(a) <= ALPHA0_TOL:
            final[i] = (row[0], row[1], 0.0, 0.0, 0.0, 0.0)
            seen.add(i)
            continue
        j = by_key[_alpha_key(-a)]
        ia, ib = (i, j) if row[0] <= new_rows[j][0] else (j, i)
        ra, rb = final[ia], final[ib]
        phi_a = float(ra[5])
        phi_b = (phi_a + PI) % TWOPI
        final[ia] = (ra[0], float(ra[1]), ra[2], ra[3], float(ra[4]), phi_a)
        final[ib] = (rb[0], float(ra[1]), -ra[2], -ra[3], float(ra[4]), phi_b)
        seen.add(ia)
        seen.add(ib)
    return final


GH = {
    5: (
        (-2.020182870456086, -0.9585724646138185, 0.0, 0.9585724646138185, 2.020182870456086),
        (0.01995324205904591, 0.3936193231522412, 0.9453087204829419, 0.3936193231522412, 0.01995324205904591),
    ),
    7: (GH7_X, GH7_W),
}


def make_sv_nodes(r: float, theta_deg: float, i_bar: float, gh_order: int) -> list[tuple]:
    xs, ws = GH[gh_order]
    theta = math.radians(theta_deg)
    vq = build_VQ_sv(r, theta)
    ell = chol2(vq)
    nbar = math.sinh(r) ** 2
    rows = []
    k = 0
    wsum = 0.0
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            k += 1
            z0 = math.sqrt(2.0) * xi
            z1 = math.sqrt(2.0) * xj
            ww = (ws[i] / math.sqrt(PI)) * (ws[j] / math.sqrt(PI))
            eta0 = ell[0][0] * z0
            eta1 = ell[1][0] * z0 + ell[1][1] * z1
            alpha = complex(eta0, eta1) / math.sqrt(2.0)
            intensity, phi = alpha_to_drive(alpha, i_bar, nbar)
            rows.append((k, ww, alpha.real, alpha.imag, intensity, phi))
            wsum += ww
    rows = [(i, ww / wsum, ra, ia, intensity, phi) for (i, ww, ra, ia, intensity, phi) in rows]
    return enforce_antipodes(rows, i_bar, nbar)


def make_sv_nodes_gh7(r: float, theta_deg: float, i_bar: float = I_BAR) -> list[tuple]:
    return make_sv_nodes(r, theta_deg, i_bar, 7)


def write_manifest(path: Path, rows: list[tuple], r: float, theta_deg: float, i_bar: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# quantum-light node manifest (shared ±N / quadrature)\n")
        fh.write("# state_type = squeezed_vacuum\n")
        fh.write("# sampling_mode = gauss_hermite\n")
        fh.write("# precision = ES25.17\n")
        fh.write("# antipodes = enforced (alpha_bar=-alpha; I,phi from alpha; phi_bar=mod(phi+pi,2pi))\n")
        fh.write(f"# squeeze_r = {r}\n")
        fh.write(f"# squeeze_theta_deg = {theta_deg}\n")
        fh.write("# gh_order = 7\n")
        fh.write(f"# I_bar = {i_bar}\n")
        fh.write("# NOTE: equal-mean-intensity mapping; not experimental BSV calibration.\n")
        fh.write(f"# n_nodes = {len(rows)}\n")
        fh.write(f"# sum_weight = {sum(r[1] for r in rows):.16e}\n")
        fh.write("# sample_id  weight  Re(alpha)  Im(alpha)  I_drive  phi_drive\n")
        for i, ww, ra, ia, intensity, phi in rows:
            fh.write(
                f"{i:8d}{ww:25.17E}{ra:25.17E}{ia:25.17E}{intensity:25.17E}{phi:25.17E}\n"
            )


def e2_axes(r: float, theta_rad: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return orthonormal (major, minor) unit axes of VQ.

    Minor is constructed as a +90° rotation of major so the pair is exactly
    orthogonal even when the off-diagonal of VQ is a tiny float residue.
    Near-diagonal VQ is aligned to the coordinate axes.
    """
    ((a, b), (_, c)) = build_VQ_sv(r, theta_rad)
    tr = a + c
    det = a * c - b * b
    disc = max(tr * tr - 4.0 * det, 0.0)
    l_hi = 0.5 * (tr + math.sqrt(disc))
    if abs(b) < 1.0e-12:
        if abs(l_hi - a) <= abs(l_hi - c):
            major = (1.0, 0.0)
        else:
            major = (0.0, 1.0)
    else:
        vx, vy = l_hi - c, b
        nrm = math.hypot(vx, vy)
        if nrm <= 0.0:
            raise ValueError("degenerate VQ major axis")
        major = (vx / nrm, vy / nrm)
    minor = (-major[1], major[0])
    return major, minor


def _pair_payload(
    left: tuple,
    right: tuple,
    major: tuple[float, float],
    minor: tuple[float, float],
    theta_deg: float,
    *,
    kind: str,
    note: str,
    extra: dict | None = None,
) -> dict:
    i_left = float(left[4])
    i_glob = extra.get("I_global_max", i_left) if extra else i_left
    payload = {
        "kind": kind,
        "theta_deg": theta_deg,
        "ids": [int(left[0]), int(right[0])],
        "I_max": i_left,
        "I_over_Imax": i_left / max(i_glob, 1.0e-300),
        "weights": [float(left[1]), float(right[1])],
        "alphas": [[float(left[2]), float(left[3])], [float(right[2]), float(right[3])]],
        "major_axis": list(major),
        "minor_axis": list(minor),
        "axis_dot": major[0] * minor[0] + major[1] * minor[1],
        "note": note,
    }
    if extra:
        payload.update(extra)
    return payload


def _antipode_row(rows: list[tuple], row: tuple) -> tuple:
    by_key = {_alpha_key(complex(item[2], item[3])): item for item in rows}
    partner = by_key.get(_alpha_key(-complex(row[2], row[3])))
    if partner is None:
        raise ValueError(f"no antipode for id={int(row[0])}")
    ids = sorted((int(row[0]), int(partner[0])))
    by_id = {int(row[0]): row, int(partner[0]): partner}
    return by_id[ids[0]], by_id[ids[1]]


def select_corner_antinode_pair(rows: list[tuple], r: float, theta_deg: float) -> dict:
    """Absolute-max-I corner antinode pair (worst-field stress), lowest IDs."""
    major, minor = e2_axes(r, math.radians(theta_deg))
    i_peak = max(float(row[4]) for row in rows)
    hot = [row for row in rows if abs(float(row[4]) - i_peak) <= I_MAX_RTOL * i_peak]
    if len(hot) < 2:
        raise ValueError(f"fewer than 2 max-I nodes at theta={theta_deg}")
    by_key = {_alpha_key(complex(row[2], row[3])): row for row in hot}
    pairs = []
    used: set[int] = set()
    for row in hot:
        nid = int(row[0])
        if nid in used:
            continue
        partner = by_key.get(_alpha_key(-complex(row[2], row[3])))
        if partner is None:
            continue
        ids = sorted((nid, int(partner[0])))
        used.add(ids[0])
        used.add(ids[1])
        pairs.append((ids, row, partner))
    if not pairs:
        raise ValueError(f"no antipode pair at I_max for theta={theta_deg}")
    pairs.sort(key=lambda item: item[0][0])
    ids, a, b = pairs[0]
    by_id = {int(a[0]): a, int(b[0]): b}
    left, right = by_id[ids[0]], by_id[ids[1]]
    if abs(float(left[4]) - EXPECTED_I_MAX) / EXPECTED_I_MAX > I_MAX_RTOL:
        raise ValueError(f"I_max={left[4]:.6e} not within {I_MAX_RTOL} of {EXPECTED_I_MAX:.6e}")
    return _pair_payload(
        left,
        right,
        major,
        minor,
        theta_deg,
        kind="corner",
        note="corner antinode: absolute max I (worst-field); not a pure wide-axis node",
        extra={"I_global_max": i_peak, "n_maxI_nodes": len(hot), "n_maxI_pairs": len(pairs)},
    )


def select_wide_axis_antinode_pair(rows: list[tuple], r: float, theta_deg: float) -> dict:
    """Pure wide-axis antinode: max |proj_major|, then min |proj_minor|."""
    major, minor = e2_axes(r, math.radians(theta_deg))
    i_peak = max(float(row[4]) for row in rows)

    def score(row: tuple) -> tuple[float, float, int]:
        ax, ay = float(row[2]), float(row[3])
        pmaj = abs(ax * major[0] + ay * major[1])
        pmin = abs(ax * minor[0] + ay * minor[1])
        return (pmaj, -pmin, -int(row[0]))

    best = max(rows, key=score)
    left, right = _antipode_row(rows, best)
    return _pair_payload(
        left,
        right,
        major,
        minor,
        theta_deg,
        kind="wide",
        note="pure wide-axis antinode: max |proj_major|, then min |proj_minor|; IDs from manifest",
        extra={"I_global_max": i_peak},
    )


def case_name(probe: dict, nk: int) -> str:
    return "sv_r2p5_th{:03d}_plusN_id{:02d}_{}_k{}".format(
        int(probe["theta_deg"]), int(probe["id"]), probe["kind"], int(nk)
    )


def peer_k20_name(k40_name: str) -> str:
    if not k40_name.endswith("_k40"):
        raise ValueError(f"not a k40 case name: {k40_name}")
    return k40_name[: -len("_k40")] + "_k20"


def expected_case_names(probes: list[dict], nk: int) -> list[str]:
    names = [case_name(p, nk) for p in probes]
    if len(set(names)) != len(names):
        raise ValueError("duplicate expected case names")
    return names


def source_tree_digest(repo: Path, *, src_from_git_head: bool = False) -> str:
    """SHA256 of Makefile + src/*.f90 listings (CRLF normalized to LF).

    ``src_from_git_head=True`` hashes ``HEAD:src/*.f90`` so a dirty worktree
    cannot poison the freeze pin. Makefile is always taken from the filesystem
    (the copy being committed with this checkpoint).
    """
    import subprocess

    src_names = sorted(path.name for path in (repo / "src").glob("*.f90"))
    rels = ["Makefile"] + [f"src/{name}" for name in src_names]
    chunks: list[str] = []
    for rel in rels:
        if src_from_git_head and rel.startswith("src/"):
            raw = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{rel}"])
        else:
            raw = (repo / rel).read_bytes()
        raw = raw.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(raw).hexdigest()
        chunks.append(f"{digest}  {rel}\n")
    return hashlib.sha256("".join(chunks).encode("ascii")).hexdigest()


def load_rows(path: Path) -> list[tuple]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        rows.append((int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    return rows


def select_from_manifest(path: Path, r: float, theta_deg: float) -> dict:
    return select_wide_axis_antinode_pair(load_rows(path), r, theta_deg)


def dump_selection(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-digest", type=Path, metavar="REPO")
    args = ap.parse_args(argv)
    if args.source_digest is not None:
        print(source_tree_digest(args.source_digest))
        return 0
    ap.error("specify --source-digest REPO")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

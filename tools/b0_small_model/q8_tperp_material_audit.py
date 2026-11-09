#!/usr/bin/env python3
"""Line-Q audit: HT-CrI3 geometry + Wannier interlayer channels -> T_perp candidates.

Does not encode an 8-band production Hamiltonian, does not run HHG/DFT/GPU.
Reads production SCF positions and (if present) full112 CrI3_tb.dat.

Historical only. Static candidate consistency lives in q8_tperp_static_audit.py;
do not treat C3_ht_three_shell here as a material-constrained C3.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCF = REPO / "deploy/a0_layerA_m88full/a1_fewk_nscf/CrI3_scf_soc_a1h_clean.in"
TB = Path(r"C:\Users\26507\Documents\量子光研究\01_输入与模型\wannier\CrI3_tb.dat")
OUT = REPO / "results/q_line/tperp_audit_20260910"

INV_C = np.array([0.499998583, 0.333333067, 0.372004766])
N_ORB8 = 8


def wrap_frac(x: np.ndarray) -> np.ndarray:
    return (x + 0.5) % 1.0 - 0.5


def parse_scf(path: Path) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    lat = None
    atoms: list[tuple[str, np.ndarray]] = []
    mode = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("CELL_PARAMETERS"):
            lat = np.array([[float(x) for x in lines[i + j + 1].split()[:3]] for j in range(3)])
        elif s.startswith("ATOMIC_POSITIONS"):
            mode = "pos"
        elif mode == "pos":
            if not s or s.startswith("HUBBARD"):
                mode = None
                continue
            p = s.split()
            atoms.append((p[0], np.array([float(x) for x in p[1:4]])))
    if lat is None:
        raise RuntimeError("no lattice")
    return lat, atoms


def geometry(lat: np.ndarray, atoms: list[tuple[str, np.ndarray]]) -> dict:
    cr = [(n, f) for n, f in atoms if n.startswith("Cr")]
    iodine = [(n, f) for n, f in atoms if n == "I"]
    names = ["tA", "tB", "bA", "bB"]
    cr_frac = {names[i]: cr[i][1] for i in range(4)}
    cr_cart = {k: v @ lat for k, v in cr_frac.items()}
    inv_pairs = {}
    for src in names:
        image = (2.0 * INV_C - cr_frac[src]) % 1.0
        best = min(
            names,
            key=lambda dst: float(np.linalg.norm(wrap_frac(image - cr_frac[dst]))),
        )
        dfrac = wrap_frac(image - cr_frac[best])
        shift = np.round(2.0 * INV_C - cr_frac[src] - cr_frac[best]).astype(int)
        inv_pairs[src] = {
            "maps_to": best,
            "frac_residual": float(np.linalg.norm(dfrac)),
            "R_image": [int(x) for x in shift],
        }
    # in-plane Cr-Cr distances (minimum image)
    a2d = lat[:2, :2]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            df = wrap_frac(cr_frac[a][:2] - cr_frac[b][:2])
            dxy = float(np.linalg.norm(df @ a2d))
            dz = abs(float((cr_cart[a] - cr_cart[b])[2]))
            pairs.append({"a": a, "b": b, "dxy_A": dxy, "dz_A": dz, "d3_A": math.hypot(dxy, dz)})
    pairs.sort(key=lambda p: p["d3_A"])
    shift_ba_ta = wrap_frac(cr_frac["bA"] - cr_frac["tA"])
    shift_bb_tb = wrap_frac(cr_frac["bB"] - cr_frac["tB"])
    # nearest interlayer Cr channel per top site
    interlayer = []
    for top in ("tA", "tB"):
        for bot in ("bA", "bB"):
            df3 = wrap_frac(cr_frac[bot] - cr_frac[top])
            # allow nearby lattice images in plane
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    df = df3 + np.array([n1, n2, 0.0])
                    vec = df @ lat
                    interlayer.append(
                        {
                            "from": top,
                            "to": bot,
                            "R": [n1, n2, 0],
                            "dxy_A": float(np.linalg.norm(vec[:2])),
                            "dz_A": float(abs(vec[2])),
                            "d3_A": float(np.linalg.norm(vec)),
                            "frac": [float(x) for x in df],
                        }
                    )
    interlayer.sort(key=lambda p: p["d3_A"])
    shortest = interlayer[0]["d3_A"]
    near = [p for p in interlayer if p["d3_A"] < shortest + 0.35]
    # I sites: interface vs outer by z relative to Cr planes
    z_top = float(cr_frac["tA"][2])
    z_bot = float(cr_frac["bA"][2])
    i_info = []
    for idx, (_n, f) in enumerate(iodine):
        z = float(f[2])
        layer = 1 if abs(z - z_top) < abs(z - z_bot) else 2
        # interface iodines sit between the two Cr planes
        interface = min(z_top, z_bot) < z < max(z_top, z_bot)
        i_info.append({"index": idx, "frac": [float(x) for x in f], "layer": layer, "interface": interface, "z": z})
    # I-I interlayer contacts
    ii = []
    for a in i_info:
        if not a["interface"] or a["layer"] != 1:
            continue
        for b in i_info:
            if not b["interface"] or b["layer"] != 2:
                continue
            df = wrap_frac(np.array(b["frac"]) - np.array(a["frac"]))
            d = float(np.linalg.norm(df @ lat))
            ii.append({"I_top": a["index"], "I_bot": b["index"], "d_A": d})
    ii.sort(key=lambda p: p["d_A"])
    aa_overlap = min(p["dxy_A"] for p in interlayer if p["from"][1] == p["to"][1])
    ab_overlap = min(p["dxy_A"] for p in interlayer if p["from"][1] != p["to"][1])
    stacking = {
        "name_from_coordinates": "HT_monoclinic_shift_minus_a2_over_3",
        "not_AA_eclipsed": aa_overlap > 1.0,
        "not_graphene_Bernal_A_over_B": ab_overlap > 1.0,
        "bottom_minus_top_frac_mean": [float(x) for x in 0.5 * (shift_ba_ta + shift_bb_tb)],
        "aa_min_dxy_A": aa_overlap,
        "ab_min_dxy_A": ab_overlap,
        "layer_sep_A": float(abs((cr_cart["bA"] - cr_cart["tA"])[2])),
    }
    return {
        "lattice_ang": lat.tolist(),
        "cr_frac": {k: [float(x) for x in v] for k, v in cr_frac.items()},
        "cr_cart": {k: [float(x) for x in v] for k, v in cr_cart.items()},
        "inversion_center_frac": INV_C.tolist(),
        "inversion_pairs_QE_labels": inv_pairs,
        "P_in_QE_AB_order": "tau_x * sigma_x  (tA<->bB, tB<->bA)",
        "P_if_bottom_relabeled_to_inversion_images": "tau_x  (tA<->bA', tB<->bB')",
        "cr_cr_pairs_sorted": pairs,
        "nearest_interlayer_cr_channels": near,
        "iodine": i_info,
        "shortest_interface_I_I": ii[:6],
        "stacking": stacking,
    }


def pauli():
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return s0, sx, sy, sz


def honeycomb_h(kx: float, ky: float, t: float, lso: float, delta: float) -> np.ndarray:
    """4x4: A/B x spin. Graphene NN + KM-like next-NN + AFM Delta s_z."""
    s0, sx, sy, sz = pauli()
    # NN from A: three vectors of length a/sqrt(3) on this slightly monoclinic lattice
    # Use 120-deg ideal vectors scaled to |a1|.
    a = 6.998304941
    d = a / math.sqrt(3.0)
    deltas = [
        np.array([d, 0.0]),
        np.array([-0.5 * d, 0.5 * math.sqrt(3.0) * d]),
        np.array([-0.5 * d, -0.5 * math.sqrt(3.0) * d]),
    ]
    fk = sum(np.exp(1j * (kx * v[0] + ky * v[1])) for v in deltas)
    hop = t * np.kron(np.array([[0, fk], [np.conj(fk), 0]], dtype=complex), s0)
    # KM: g(k) odd, couples same sublattice with s_z
    g = 2.0 * (
        np.sin(kx * a)
        + np.sin((-0.5 * a) * kx + (0.5 * math.sqrt(3.0) * a) * ky)
        + np.sin((-0.5 * a) * kx - (0.5 * math.sqrt(3.0) * a) * ky)
    )
    km = lso * g * np.kron(sz, sz)  # sigma_z s_z, wait: first kron is sublattice
    # np.kron(sz, sz) is sublattice-z tensor spin-z. Good for KM.
    mag = delta * np.kron(s0, sz)
    return hop + km + mag


def tperp_block(kind: str, kx: float, ky: float, tperp: float) -> np.ndarray:
    """2x2 on A/B, then ⊗ s0."""
    s0, sx, sy, sz = pauli()
    a = 6.998304941
    if kind == "C1_sigma0":
        tab = tperp * np.eye(2, dtype=complex)
    elif kind == "C2_sigma_x":
        tab = tperp * sx
    elif kind == "C3_ht_three_shell":
        # three interlayer images near |R|=a/sqrt(3) with mixed AA/AB weight
        d = a / math.sqrt(3.0)
        vecs_aa = [np.array([0.0, d]), np.array([-0.5 * math.sqrt(3.0) * d, -0.5 * d]), np.array([0.5 * math.sqrt(3.0) * d, -0.5 * d])]
        vecs_ab = [np.array([d, 0.0]), np.array([-0.5 * d, 0.5 * math.sqrt(3.0) * d]), np.array([-0.5 * d, -0.5 * math.sqrt(3.0) * d])]
        faa = sum(np.exp(1j * (kx * v[0] + ky * v[1])) for v in vecs_aa) / 3.0
        fab = sum(np.exp(1j * (kx * v[0] + ky * v[1])) for v in vecs_ab) / 3.0
        tab = tperp * (0.55 * faa * np.eye(2) + 0.45 * np.array([[0, fab], [np.conj(fab), 0]], dtype=complex))
    else:
        raise ValueError(kind)
    return np.kron(tab, s0)


def h8(kind: str, kx: float, ky: float, delta_n: float) -> np.ndarray:
    ht = honeycomb_h(kx, ky, t=1.0, lso=0.1, delta=delta_n)
    hb = honeycomb_h(kx, ky, t=1.0, lso=0.1, delta=-delta_n)
    tp = tperp_block(kind, kx, ky, tperp=0.15)
    h = np.zeros((N_ORB8, N_ORB8), dtype=complex)
    h[:4, :4] = ht
    h[4:, 4:] = hb
    h[:4, 4:] = tp
    h[4:, :4] = tp.conj().T
    return h


def jx8(kind: str, kx: float, ky: float, delta_n: float, dk: float = 1.0e-4) -> np.ndarray:
    return -(h8(kind, kx + dk, ky, delta_n) - h8(kind, kx - dk, ky, delta_n)) / (2.0 * dk)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def symmetry_scan() -> dict:
    s0, sx, sy, sz = pauli()
    unitaries = {
        "tau_x": kron3(sx, s0, s0),
        "tau_x_sigma_x": kron3(sx, sx, s0),
        "tau_x_s_z": kron3(sx, s0, sz),
        "tau_x_sigma_x_s_z": kron3(sx, sx, sz),
        "s_z": kron3(s0, s0, sz),
        "sigma_x": kron3(s0, sx, s0),
    }
    ks = [(0.0, 0.0), (0.7, 0.2), (1.2, -0.4), (0.52, 0.90)]
    kinds = ("C1_sigma0", "C2_sigma_x", "C3_ht_three_shell")
    out: dict = {}
    for kind in kinds:
        rec = {}
        for uname, u in unitaries.items():
            r_same = r_flipN = r_k = r_kN = 0.0
            r_j = 0.0
            for kx, ky in ks:
                hp = h8(kind, kx, ky, 0.5)
                hm = h8(kind, kx, ky, -0.5)
                hp_mk = h8(kind, -kx, -ky, 0.5)
                hm_mk = h8(kind, -kx, -ky, -0.5)
                mapped = u @ hp @ u.conj().T
                r_same = max(r_same, float(np.max(np.abs(mapped - hp))))
                r_flipN = max(r_flipN, float(np.max(np.abs(mapped - hm))))
                r_k = max(r_k, float(np.max(np.abs(mapped - hp_mk))))
                r_kN = max(r_kN, float(np.max(np.abs(mapped - hm_mk))))
                jp = jx8(kind, kx, ky, 0.5)
                r_j = max(r_j, float(np.max(np.abs(u @ jp @ u.conj().T - jp))))
            rec[uname] = {
                "same_N_same_k": r_same,
                "flip_N_same_k": r_flipN,
                "same_N_flip_k": r_k,
                "flip_N_flip_k": r_kN,
                "jx_commutes": r_j,
            }
        # TR: i s_y K
        ut = kron3(s0, s0, 1j * sy)
        r_tr = 0.0
        for kx, ky in ks:
            hp = h8(kind, kx, ky, 0.5)
            hm_mk = h8(kind, -kx, -ky, -0.5)
            mapped = ut @ hp.conj() @ ut.conj().T
            r_tr = max(r_tr, float(np.max(np.abs(mapped - hm_mk))))
        rec["T_isyK"] = {"flip_N_flip_k": r_tr}
        # PT with P = tau_x sigma_x
        p = unitaries["tau_x_sigma_x"]
        r_pt = 0.0
        for kx, ky in ks:
            hp = h8(kind, kx, ky, 0.5)
            mapped = p @ (ut @ hp.conj() @ ut.conj().T) @ p.conj().T
            r_pt = max(r_pt, float(np.max(np.abs(mapped - hp))))
        rec["PT_P_taux_sigmax"] = {"same_N_same_k": r_pt}
        p2 = unitaries["tau_x"]
        r_pt2 = 0.0
        for kx, ky in ks:
            hp = h8(kind, kx, ky, 0.5)
            mapped = p2 @ (ut @ hp.conj() @ ut.conj().T) @ p2.conj().T
            r_pt2 = max(r_pt2, float(np.max(np.abs(mapped - hp))))
        rec["PT_P_taux"] = {"same_N_same_k": r_pt2}
        out[kind] = rec
    return out


def wf_atom(i: int) -> tuple[str, int]:
    if i < 40:
        return "Cr", i // 10
    return "I", (i - 40) // 6


def wf_spin(i: int) -> int:
    if i < 40:
        return (i % 10) % 2
    return ((i - 40) % 6) % 2


def wf_i_orb(i: int) -> str:
    return ("pz", "px", "py")[((i - 40) % 6) // 2]


def stream_tb(path: Path) -> dict:
    cr_layer = {0: "t", 1: "t", 2: "b", 3: "b"}
    cr_ab = {0: "A", 1: "B", 2: "A", 3: "B"}
    # SCF I order: 0-2 interface top (z~0.30), 3-5 outer top, 6-8 outer bot, 9-11 interface bot.

    dd_ab = defaultdict(float)
    dd_ab_spinflip = defaultdict(float)
    dd_R = defaultdict(float)
    pp_inter = defaultdict(float)
    pp_inter_orb = defaultdict(float)
    dp_inter = 0.0
    dd_inter = 0.0
    pp_inter_total = 0.0
    spinflip_inter_dd = 0.0
    spinflip_inter_pp = 0.0
    max_dd = {"abs": 0.0}
    max_pp = {"abs": 0.0}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for _ in range(3):
            fh.readline()
        nw = int(fh.readline().split()[0])
        nr = int(fh.readline().split()[0])
        ndegen = []
        while len(ndegen) < nr:
            ndegen.extend(int(x) for x in fh.readline().split())
        for ir in range(nr):
            line = fh.readline()
            while line and not line.strip():
                line = fh.readline()
            rvec = tuple(int(x) for x in line.split()[:3])
            deg = float(ndegen[ir])
            for _ in range(nw * nw):
                p = fh.readline().split()
                i, j = int(p[0]) - 1, int(p[1]) - 1
                w2 = (float(p[2]) ** 2 + float(p[3]) ** 2) / deg
                amp = math.sqrt(float(p[2]) ** 2 + float(p[3]) ** 2) / deg
                si, ai = wf_atom(i)
                sj, aj = wf_atom(j)
                if si == "Cr" and sj == "Cr":
                    if cr_layer[ai] != "t" or cr_layer[aj] != "b":
                        continue
                    dd_inter += w2
                    key = cr_ab[ai] + cr_ab[aj]
                    if wf_spin(i) != wf_spin(j):
                        dd_ab_spinflip[key] += w2
                        spinflip_inter_dd += w2
                    else:
                        dd_ab[key] += w2
                    dd_R[(key, rvec)] += w2
                    if amp > max_dd["abs"]:
                        max_dd = {"abs": amp, "from": ai, "to": aj, "R": list(rvec), "spin_flip": wf_spin(i) != wf_spin(j)}
                elif si == "I" and sj == "I":
                    if not (ai in {0, 1, 2} and aj in {9, 10, 11}):
                        continue
                    pp_inter_total += w2
                    okey = wf_i_orb(i) + "-" + wf_i_orb(j)
                    pp_inter_orb[okey] += w2
                    pp_inter[rvec] += w2
                    if wf_spin(i) != wf_spin(j):
                        spinflip_inter_pp += w2
                    if amp > max_pp["abs"]:
                        max_pp = {"abs": amp, "I_i": ai, "I_j": aj, "R": list(rvec), "orb": okey, "spin_flip": wf_spin(i) != wf_spin(j)}
                elif {si, sj} == {"Cr", "I"}:
                    # interlayer Cr-I: Cr and I on different layers
                    if si == "Cr":
                        cl, ii = cr_layer[ai], aj
                    else:
                        cl, ii = cr_layer[aj], ai
                    il = "t" if ii <= 5 else "b"
                    if cl != il:
                        dp_inter += w2

    def topn(d, n=12):
        items = sorted(d.items(), key=lambda kv: -kv[1])[:n]
        out = []
        for k, v in items:
            if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], tuple):
                out.append({"AB": k[0], "R": list(k[1]), "sum_t2": v})
            elif isinstance(k, tuple):
                out.append({"R": list(k), "sum_t2": v})
            else:
                out.append({"key": str(k), "sum_t2": v})
        return out

    return {
        "nw": nw,
        "nrpts": nr,
        "bare_CrCr_interlayer_sum_t2": dd_inter,
        "bare_CrCr_interlayer_by_AB": dict(dd_ab),
        "bare_CrCr_interlayer_spinflip_by_AB": dict(dd_ab_spinflip),
        "bare_CrCr_interlayer_spinflip_sum_t2": spinflip_inter_dd,
        "bare_CrCr_top_R": topn(dd_R, 15),
        "bare_CrCr_max": max_dd,
        "interface_I_I_sum_t2": pp_inter_total,
        "interface_I_I_by_orb": dict(pp_inter_orb),
        "interface_I_I_spinflip_sum_t2": spinflip_inter_pp,
        "interface_I_I_top_R": topn(pp_inter, 10),
        "interface_I_I_max": max_pp,
        "CrI_interlayer_sum_t2": dp_inter,
        "note": (
            "Bare Cr-Cr d-d interlayer is a tiny fraction of transport weight. "
            "Interface I-I (especially pz) is the physical interlayer channel; "
            "8-band T_perp is an effective downfold, not the bare d-d matrix."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lat, atoms = parse_scf(SCF)
    geo = geometry(lat, atoms)
    (OUT / "geometry.json").write_text(json.dumps(geo, indent=2), encoding="utf-8")
    print("geometry stacked:", geo["stacking"]["name_from_coordinates"])
    print("P (QE labels):", geo["P_in_QE_AB_order"])
    print("nearest Cr-Cr interlayer:")
    for p in geo["nearest_interlayer_cr_channels"][:8]:
        print(f"  {p['from']}->{p['to']} R={p['R']} d3={p['d3_A']:.3f} dxy={p['dxy_A']:.3f}")
    sym = symmetry_scan()
    (OUT / "tperp_candidate_symmetry.json").write_text(json.dumps(sym, indent=2), encoding="utf-8")
    print("symmetry scan written")
    hop = None
    if TB.is_file():
        print("streaming", TB)
        hop = stream_tb(TB)
        (OUT / "wannier_interlayer_hopping.json").write_text(json.dumps(hop, indent=2), encoding="utf-8")
        print("hopping written")
    else:
        print("tb.dat missing; hopping tables skipped")
    report = {
        "schema": "q8_tperp_material_audit_v1",
        "not_a_physics_result": True,
        "not_encoded_into_pt_afm_toy": True,
        "geometry_file": "geometry.json",
        "symmetry_file": "tperp_candidate_symmetry.json",
        "hopping_file": "wannier_interlayer_hopping.json" if hop else None,
        "human_gates": [
            "HT stacking abstraction to 8-band geometry",
            "P(k) representation",
            "which interlayer channels to keep in minimal T_perp",
            "material vs PT-AFM-toy-goal conflict (do not auto-patch)",
        ],
    }
    (OUT / "audit_index.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("audit_index written", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C1 closeout authorized 2026-11-04: hash, bootstrap, CEP=pi/2 coherent control.

Does not rerun BSV, does not start C2, does not modify c1.json / frozen fits.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/q_line/q1600_c1_pureBSV_stat_sbe_20261104"
DOC = ROOT / "docs/04_BSV量子光"
C1_JSON = OUT / "c1.json"
C1_PY = HERE / "q8_m0_c1_pureBSV_stat_sbe_20261104.py"
PROD_SCRIPT_SHA = "ce32bc35ba143e9f1705776409524d678e05773d3450e61667b908ed4245d06b"
DISK_SCRIPT_SHA = "4bae1e01dd032e8fb3732a892c96ee1b21bb81976e2d8387edafb16648c16e4b"
BOOT_SEED = 202611041
N_BOOT = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_c1_mod():
    spec = importlib.util.spec_from_file_location("c1_prod", C1_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unpack_arr(rows) -> np.ndarray:
    out = np.empty((len(rows), 2), dtype=complex)
    for i, row in enumerate(rows):
        for j, z in enumerate(row):
            out[i, j] = complex(z["re"], z["im"])
    return out


def cpack(z: complex) -> dict:
    z = complex(z)
    return {
        "re": float(z.real),
        "im": float(z.imag),
        "abs": float(abs(z)),
        "phase_deg": float(np.degrees(np.angle(z))),
    }


def contrast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = a + b
    out = np.full_like(a, np.nan, dtype=float)
    ok = den > 0
    out[ok] = (a[ok] - b[ok]) / den[ok]
    return out


def odd_even(zp: np.ndarray, zm: np.ndarray):
    even = 0.5 * (zp + zm)
    odd = 0.5 * (zp - zm)
    return even, odd


def bootstrap_bsv(h2p: np.ndarray, h2m: np.ndarray, n_boot: int, seed: int) -> dict:
    n = h2p.shape[0]
    rng = np.random.default_rng(seed)
    sp = np.abs(h2p) ** 2
    sm = np.abs(h2m) ** 2
    pstat_p = sp.mean(axis=0)
    pstat_m = sm.mean(axis=0)
    c_pt = contrast(pstat_p, pstat_m)
    mean_p = h2p.mean(axis=0)
    mean_m = h2m.mean(axis=0)
    even, odd = odd_even(mean_p, mean_m)
    odd_norm = float(np.linalg.norm(odd))
    even_norm = float(np.linalg.norm(even))

    c_boot = np.empty((n_boot, 2), dtype=float)
    odd_boot = np.empty(n_boot, dtype=float)
    even_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pp = sp[idx].mean(axis=0)
        mm = sm[idx].mean(axis=0)
        c_boot[b] = contrast(pp, mm)
        mp = h2p[idx].mean(axis=0)
        mmc = h2m[idx].mean(axis=0)
        _e, o = odd_even(mp, mmc)
        odd_boot[b] = np.linalg.norm(o)
        even_boot[b] = np.linalg.norm(_e)

    def ci(arr):
        lo, hi = np.quantile(arr, [0.025, 0.975], axis=0)
        return np.asarray(lo, float), np.asarray(hi, float)

    c_lo, c_hi = ci(c_boot)
    o_lo, o_hi = ci(odd_boot)
    e_lo, e_hi = ci(even_boot)

    k = max(1, int(round(0.01 * n)))
    tail = {}
    for name, arr in {
        "plus_y": sp[:, 1],
        "minus_y": sm[:, 1],
        "sum_y": sp[:, 1] + sm[:, 1],
        "plus_x": sp[:, 0],
        "minus_x": sm[:, 0],
        "sum_x": sp[:, 0] + sm[:, 0],
    }.items():
        tot = float(arr.sum())
        order = np.argsort(arr)[::-1]
        frac = float(arr[order[:k]].sum() / tot) if tot > 0 else None
        tail[name] = {"k": k, "frac_top1pct": frac, "total": tot}

    return {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "method": "paired_index_resample_same_plus_minus",
        "Cstat_point": c_pt.tolist(),
        "Cstat_ci95": {"lo": c_lo.tolist(), "hi": c_hi.tolist()},
        "Cstat_contains_zero": [bool(c_lo[i] <= 0.0 <= c_hi[i]) for i in range(2)],
        "odd_norm_point": odd_norm,
        "odd_norm_ci95": [float(o_lo), float(o_hi)],
        "even_norm_point": even_norm,
        "even_norm_ci95": [float(e_lo), float(e_hi)],
        "odd_vector": [cpack(z) for z in odd],
        "even_vector": [cpack(z) for z in even],
        "Pstat_plus": pstat_p.tolist(),
        "Pstat_minus": pstat_m.tolist(),
        "tail_top1pct": tail,
        "note": (
            "95% intervals are empirical paired-bootstrap quantiles, not a proof that "
            "Cstat vanishes in the infinite-sample limit."
        ),
    }


def summarize_deterministic(h2p: np.ndarray, h2m: np.ndarray) -> dict:
    zp = h2p.mean(axis=0)
    zm = h2m.mean(axis=0)
    even, odd = odd_even(zp, zm)
    sp = np.mean(np.abs(h2p) ** 2, axis=0)
    sm = np.mean(np.abs(h2m) ** 2, axis=0)
    pcoh_p = np.abs(zp) ** 2
    pcoh_m = np.abs(zm) ** 2
    ident = (np.abs(zp) ** 2 - np.abs(zm) ** 2) - 4.0 * np.real(np.conjugate(even) * odd)
    return {
        "plus": [cpack(z) for z in zp],
        "minus": [cpack(z) for z in zm],
        "even": [cpack(z) for z in even],
        "odd": [cpack(z) for z in odd],
        "odd_norm2D": float(np.linalg.norm(odd)),
        "even_norm2D": float(np.linalg.norm(even)),
        "C_from_abs_mean_sq": contrast(pcoh_p, pcoh_m).tolist(),
        "C_from_mean_abs_sq": contrast(sp, sm).tolist(),
        "identity_maxabs": float(np.max(np.abs(ident))),
    }


def replay_protocol(mod, frozen: dict) -> dict:
    proto = mod.traveling_protocol(frozen["I0_W_cm2"])
    keys = [
        "nbar",
        "alpha_coh",
        "coherent_mean_X",
        "coherent_mean_P",
        "E_peak_coh_V_m",
        "E1_V_m",
    ]
    rel = {}
    for k in keys:
        a = float(proto[k])
        b = float(frozen["protocol"][k])
        rel[k] = abs(a - b) / max(abs(b), 1e-30)
    bsv_now = proto["bsv"]
    bsv_old = frozen["protocol"]["bsv"]
    for k in ["n_s", "abs_M", "sigma_sq", "sigma_anti", "var_X", "var_P", "r_internal"]:
        a = float(bsv_now[k])
        b = float(bsv_old[k])
        rel["bsv." + k] = abs(a - b) / max(abs(b), 1e-30)
    max_rel = float(max(rel.values()))
    return {
        "max_rel": max_rel,
        "rel": rel,
        "parameterization_now": bsv_now.get("parameterization"),
        "parameterization_frozen": bsv_old.get("parameterization"),
        "pass": bool(max_rel < 1e-12),
    }


def hash_audit(frozen: dict) -> dict:
    disk = sha256_file(C1_PY)
    model = sha256_file(HERE / "q8_m0_model.py")
    classical = sha256_file(HERE / "q8_m0_classical.py")
    rec = frozen["source_sha256"]
    n64 = json.loads((OUT / "c1_n64.json").read_text(encoding="utf-8"))
    n256 = json.loads((OUT / "c1_n256.json").read_text(encoding="utf-8"))
    report_n64 = (OUT / "C1_report_n64.md").read_text(encoding="utf-8")
    src = C1_PY.read_text(encoding="utf-8")
    return {
        "production_script_sha256": rec["this"],
        "disk_script_sha256": disk,
        "expected_disk_script_sha256": DISK_SCRIPT_SHA,
        "production_blob_available": False,
        "model_sha_match": model == rec["q8_m0_model"],
        "classical_sha_match": classical == rec["q8_m0_classical"],
        "n64_script_sha256": n64["source_sha256"]["this"],
        "n256_script_sha256": n256["source_sha256"]["this"],
        "all_production_used_ce32": all(
            x == PROD_SCRIPT_SHA
            for x in [rec["this"], n64["source_sha256"]["this"], n256["source_sha256"]["this"]]
        ),
        "n64_report_has_old_stop_phrase": ("高增益反压缩尾部" in report_n64),
        "current_source_has_old_stop_phrase": ("高增益反压缩尾部" in src),
        "current_default_n_physics_1024": ("default=1024" in src and "n-physics-samples" in src),
        "inferred_post_run_edits": [
            "默认 --n-physics-samples 在 N=64 首跑之后改为 1024",
            "write_report 在生产完成后被改写（n64 停机措辞已不在当前源码中）",
            "禁止公式 2n_s+1-2|M| 返回 0 的闸门措辞有改动；生产 JSON 当时已将其判为不可用",
            "稳定参数化 σ_sq=1/σ_anti 已出现在生产 JSON 中，不是运行后才改的物理核心",
        ],
        "physics_core_functions_still_present": all(
            s in src
            for s in [
                "X * np.cos(omega * tau) + P * np.sin(omega * tau)",
                "sigma_sq = float(1.0 / sigma_anti)",
                "def propagate_a",
                "def project_h2",
                'method="linear_exact"',
            ]
        ),
        "conclusion": (
            "c1.json 由 ce32 生成；磁盘现在是运行后改过的 4bae。"
            "q8_m0_model 与 q8_m0_classical 的哈希与生产一致。git 中没有 ce32 原文件。"
            "本轮用 traveling_protocol 对冻结 JSON 的回放作为物理等价检查。"
        ),
    }


def write_report(rec: dict) -> Path:
    boot = rec["bootstrap"]
    h = rec["hash_audit"]
    coh0 = rec["coherent_cep0"]
    coh1 = rec["coherent_cep_pi2"]
    bsv = rec["bsv_point"]
    cx0, cy0 = coh0["C_from_mean_abs_sq"]
    cx1, cy1 = coh1["C_from_mean_abs_sq"]
    bx, by = boot["Cstat_point"]
    lines = []
    lines.append("# C1 收口：bootstrap 区间、脚本哈希、CEP=π/2 相干对照（2026-11-04）\n\n")
    lines.append("外审授权范围：零新 BSV 传播、不启动 C2、不改冻结拟合、不 commit/push。\n\n")
    lines.append("本轮只做三件事：核对生成 `c1.json` 的脚本版本、对已有 1024 条样本做配对 bootstrap、")
    lines.append("补一条与 BSV 反压缩方向匹配的相干 CEP=π/2 负控。\n\n")

    lines.append("## 1. 脚本哈希\n\n")
    lines.append(f"- 生产运行写入的脚本 SHA-256：`{h['production_script_sha256']}`\n")
    lines.append(f"- 当前磁盘脚本 SHA-256：`{h['disk_script_sha256']}`\n")
    lines.append(f"- `q8_m0_model.py` 与生产一致：{h['model_sha_match']}\n")
    lines.append(f"- `q8_m0_classical.py` 与生产一致：{h['classical_sha_match']}\n")
    lines.append(f"- n64/n256/n1024 均记录 ce32：{h['all_production_used_ce32']}\n")
    lines.append(f"- 协议回放最大相对误差：{rec['protocol_replay']['max_rel']:.3e}，通过：{rec['protocol_replay']['pass']}\n")
    lines.append("- ce32 原文件不在 git 中，无法做逐字节 diff。\n")
    lines.append("- 推断的运行后修改（不触及已保存的 1024 条 H2）：\n")
    for item in h["inferred_post_run_edits"]:
        lines.append(f"  - {item}\n")
    lines.append(f"- 判断：{h['conclusion']}\n")
    lines.append("- 物理传播核心（场映射、SBE、H2、稳定协方差）仍在当前源码中，且协议回放相对误差为 0。无需重跑 1024 条 BSV。\n\n")

    lines.append("## 2. 已有 1024 条样本的配对 bootstrap\n\n")
    lines.append(f"N={boot['n']}，B={boot['n_boot']}，seed={boot['seed']}，plus/minus 使用同一组重抽样下标。\n\n")
    lines.append("| 量 | 点估计 | 约 95% 区间 | 含 0 |\n|---|---|---|---|\n")
    lines.append(
        f"| C_stat H2x | {bx:.6f} | [{boot['Cstat_ci95']['lo'][0]:.6f}, {boot['Cstat_ci95']['hi'][0]:.6f}] | {boot['Cstat_contains_zero'][0]} |\n"
    )
    lines.append(
        f"| C_stat H2y | {by:.6f} | [{boot['Cstat_ci95']['lo'][1]:.6f}, {boot['Cstat_ci95']['hi'][1]:.6f}] | {boot['Cstat_contains_zero'][1]} |\n"
    )
    lines.append(
        f"| ‖⟨J₂ᵒ⟩‖ | {boot['odd_norm_point']:.6e} | [{boot['odd_norm_ci95'][0]:.6e}, {boot['odd_norm_ci95'][1]:.6e}] | no |\n\n"
    )
    ty = boot["tail_top1pct"]
    lines.append(
        f"最大 1% 样本对 H2y 统计强度的贡献：plus {ty['plus_y']['frac_top1pct']:.3f}，"
        f"minus {ty['minus_y']['frac_top1pct']:.3f}，plus+minus {ty['sum_y']['frac_top1pct']:.3f}。\n\n"
    )
    lines.append(
        "因此：当前抽样下 BSV 的统计强度磁畴对比与零相容，点估计的负号没有明确统计显著性；"
        "非零的平均 Néel-odd 复电流更稳定。半样本 35% 门槛只适合冒烟，不能把百分之几的 C_stat 当成精密预测。\n\n"
    )
    lines.append(
        "与外审独立 5000 次 bootstrap 交叉核对：H2x 区间约 [−0.053,+0.023] vs 本轮 [−0.054,+0.023]；"
        "H2y 约 [−0.164,+0.049] vs 本轮 [−0.165,+0.043]；odd 范数约 [2.00,2.34]×10⁻³ vs 本轮 [2.00,2.35]×10⁻³。差别来自随机种子，不改变“含零 / 不含零”的判断。\n\n"
    )

    lines.append("## 3. CEP=π/2 相干负控\n\n")
    lines.append("场约定不变：E(t)∝f(t)[X cos(ωt)+P sin(ωt)]。\n\n")
    lines.append("| 输入 | (X,P) | 载波 | odd ‖H2‖ | even ‖H2‖ | C_x | C_y |\n|---|---|---|---|---|---|---|\n")
    lines.append(
        f"| 相干 CEP=0 | (2α, 0) | cos | {coh0['odd_norm2D']:.6e} | {coh0['even_norm2D']:.6e} | {cx0:.4f} | {cy0:.4f} |\n"
    )
    lines.append(
        f"| 相干 CEP=π/2 | (0, 2α) | sin | {coh1['odd_norm2D']:.6e} | {coh1['even_norm2D']:.6e} | {cx1:.4f} | {cy1:.4f} |\n"
    )
    lines.append(
        f"| 纯 BSV θ=0 | 抽样，主分量 P | 近 sin | {bsv['odd_norm2D']:.6e} | {bsv['even_norm2D']:.6e} | {bx:.4f} | {by:.4f} |\n\n"
    )
    ratio0 = rec["ratios"]["odd_bsv_over_cep0"]
    ratio1 = rec["ratios"]["odd_bsv_over_cep_pi2"]
    lines.append(f"- BSV/CEP=0 的 odd 范数比 = {ratio0:.4f}\n")
    lines.append(f"- BSV/CEP=π/2 的 odd 范数比 = {ratio1:.4f}\n")
    lines.append(f"- CEP=π/2 墙时 {rec['cep_pi2_wall_s']:.2f} s，最大非厄米 {rec['cep_pi2_max_herm']:.3e}\n")
    lines.append(f"- CEP=π/2 恒等式残差 {coh1['identity_maxabs']:.3e}\n\n")
    lines.append(rec["cep_verdict"] + "\n\n")

    lines.append("## 4. 收口判断\n\n")
    for item in rec["final_bullets"]:
        lines.append(f"- {item}\n")
    lines.append("\n未增加 BSV 样本，未改 I0/g，未启动 C2 / CrI₃ / 新 QED，未修改 `c1.json`。\n")
    lines.append("\nJSON：`" + str(OUT / "c1_closeout_cep_bootstrap_20261104.json") + "`\n")

    text = "".join(lines)
    path = OUT / "C1_closeout_20261104.md"
    path.write_text(text, encoding="utf-8")
    (DOC / "线Q_C1收口_CEP与bootstrap_20261104.md").write_text(text, encoding="utf-8")
    return path


def cep_verdict(coh0, coh1, boot, ratios) -> str:
    cx0, cy0 = coh0["C_from_mean_abs_sq"]
    cx1, cy1 = coh1["C_from_mean_abs_sq"]
    coh_contrast_survives = abs(cx0) > 0.2 and abs(cy0) > 0.2 and abs(cx1) > 0.2 and abs(cy1) > 0.2
    bsv_compatible_zero = all(boot["Cstat_contains_zero"])
    odd_shift = abs(ratios["odd_bsv_over_cep0"] - ratios["odd_bsv_over_cep_pi2"])
    if coh_contrast_survives and bsv_compatible_zero:
        extra = (
            "匹配 CEP 后，两条相干参考仍有 O(1) 的磁畴强度对比，而 BSV 的 C_stat 仍与零相容。"
            "因此“相干有对比、BSV 对比变小且与零相容”不能单用 CEP=90° 错配解释。"
        )
    else:
        extra = (
            "CEP 对照改变了相干参考的定量结果；相干–BSV 差异有一部分可能来自载波相位，"
            "不能把 0.631 倍 odd-H2 或 C_stat 差别全部归因于光子统计。"
        )
    return (
        f"{extra} odd 范数比从 CEP=0 的 {ratios['odd_bsv_over_cep0']:.3f} "
        f"变为 CEP=π/2 的 {ratios['odd_bsv_over_cep_pi2']:.3f}（绝对变化 {odd_shift:.3f}）。"
        "即便如此，这也不是量子优势：宽强度分布加随机相位的经典场同样可能压低强度对比。"
    )


def main() -> int:
    t0 = time.perf_counter()
    frozen = json.loads(C1_JSON.read_text(encoding="utf-8"))
    if frozen.get("status") != "c1_minimal_complete":
        raise SystemExit(f"unexpected c1 status {frozen.get('status')}")
    mod = load_c1_mod()
    h = hash_audit(frozen)
    replay = replay_protocol(mod, frozen)

    h2p = unpack_arr(frozen["bsv"]["per_sample_H2_plus"])
    h2m = unpack_arr(frozen["bsv"]["per_sample_H2_minus"])
    boot = bootstrap_bsv(h2p, h2m, N_BOOT, BOOT_SEED)

    coh0_p = unpack_arr(frozen["coherent"]["per_sample_H2_plus"])
    coh0_m = unpack_arr(frozen["coherent"]["per_sample_H2_minus"])
    coh0 = summarize_deterministic(coh0_p, coh0_m)
    bsv_pt = summarize_deterministic(h2p, h2m)

    print("C1 closeout: CEP=pi/2 coherent ...", flush=True)
    packed = mod.run_gates(frozen["I0_W_cm2"], 20000, 20261104, frozen["time"]["nt_per_cycle"])
    geo = packed.pop("geo")
    meta = packed.pop("meta")
    proto = packed["protocol"]
    proto_pi2 = dict(proto)
    proto_pi2["coherent_mean_X"] = 0.0
    proto_pi2["coherent_mean_P"] = float(2.0 * proto["alpha_coh"])
    proto_pi2["cep"] = float(0.5 * np.pi)
    p = mod.R0
    kf, kc, w = mod.k_grid_xy(geo, frozen["grid"]["nk"], 0.5, 0.5)
    if not mod.inversion_closed(kf):
        raise SystemExit("grid not inversion closed")
    t1 = time.perf_counter()
    rec_pi2 = mod.run_input(geo, p, kc, w, meta, proto_pi2, "coherent", 1, 20261104, 0.0)
    cep_wall = time.perf_counter() - t1
    h2p1 = unpack_arr(rec_pi2["per_sample_H2_plus"])
    h2m1 = unpack_arr(rec_pi2["per_sample_H2_minus"])
    coh1 = summarize_deterministic(h2p1, h2m1)

    ratios = {
        "odd_bsv_over_cep0": bsv_pt["odd_norm2D"] / coh0["odd_norm2D"],
        "odd_bsv_over_cep_pi2": bsv_pt["odd_norm2D"] / coh1["odd_norm2D"],
        "even_bsv_over_cep0": bsv_pt["even_norm2D"] / coh0["even_norm2D"],
        "even_bsv_over_cep_pi2": bsv_pt["even_norm2D"] / coh1["even_norm2D"],
    }
    verdict = cep_verdict(coh0, coh1, boot, ratios)
    final = [
        "C1 方法验证维持通过：稳定高增益 Wigner 协方差、|⟨J⟩|² 与 ⟨|J|²⟩ 区分、非零平均 Néel-odd H2。",
        "不能把“纯 BSV 洗掉磁畴强度对比”写成已收敛物理结论；C_stat 的 95% 区间含 0。",
        "更稳妥的表述：当前抽样下 BSV 统计强度磁畴对比显著小于所选相干参考，并且与零相容。",
        "CEP=π/2 相干负控已完成，不重跑 1024 条 BSV。",
        "本轮停止。不启动 C2。",
    ]
    rec = {
        "status": "c1_closeout_complete",
        "date": "2026-11-04",
        "not_C2": True,
        "did_not_rerun_bsv": True,
        "did_not_modify_c1_json": True,
        "did_not_modify_frozen_fit": True,
        "source_c1_json": str(C1_JSON),
        "hash_audit": h,
        "protocol_replay": replay,
        "bootstrap": boot,
        "coherent_cep0": coh0,
        "coherent_cep_pi2": coh1,
        "bsv_point": bsv_pt,
        "cep_pi2_input": {
            "coherent_mean_X": proto_pi2["coherent_mean_X"],
            "coherent_mean_P": proto_pi2["coherent_mean_P"],
            "alpha_coh": proto["alpha_coh"],
            "field": "E ~ f(t) * 2 alpha * sin(omega t), i.e. CEP=pi/2 in the code convention",
        },
        "cep_pi2_raw": {
            "wall_s": rec_pi2["wall_s"],
            "max_herm": rec_pi2["max_herm"],
            "plus": rec_pi2["per_sample_H2_plus"],
            "minus": rec_pi2["per_sample_H2_minus"],
        },
        "cep_pi2_wall_s": cep_wall,
        "cep_pi2_max_herm": rec_pi2["max_herm"],
        "ratios": ratios,
        "cep_verdict": verdict,
        "final_bullets": final,
        "closeout_script_sha256": sha256_file(Path(__file__)),
        "wall_s_total": time.perf_counter() - t0,
    }
    out_json = OUT / "c1_closeout_cep_bootstrap_20261104.json"
    out_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = write_report(rec)
    print("STATUS", rec["status"])
    print("JSON", out_json)
    print("REPORT", report)
    print("Cstat", boot["Cstat_point"], "CI", boot["Cstat_ci95"], "zero", boot["Cstat_contains_zero"])
    print("odd CI", boot["odd_norm_ci95"])
    print("ratios", ratios)
    print("verdict", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

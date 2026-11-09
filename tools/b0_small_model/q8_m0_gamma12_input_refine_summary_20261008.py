"""Source-backed summary of the bounded 96 -> 128 input-core control."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/q_line/gamma12_output_transpose_input_refine_20261008'
TARGET = BASE/'comparison.json'
DOC = ROOT/'docs/04_BSV量子光/线Q_Gamma12时间步复核与输入核心求积对照_20261008.md'


def main():
    if TARGET.exists() or DOC.exists():
        raise FileExistsError('Preserve existing summary')
    meta, rows = {}, {}
    for p in (96, 128):
        meta[p] = json.loads((BASE/f'n_panel_{p}.json').read_text(encoding='utf-8'))
        path = BASE/f'n_panel_{p}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta[p]['row_sha256']:
            raise ValueError('Output hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            rows[p] = {k: z[k] for k in ('n', 'm', 'normal_contribution', 'anomalous_contribution')}
    if meta[96]['all_checks_pass'] is not True:
        raise ValueError('96 regression failed')
    changes = {}
    for key in rows[96]:
        a, b = rows[96][key], rows[128][key]
        if key == 'normal_contribution':
            a, b = a-np.eye(2), b-np.eye(2)
        changes[key] = dict(maxabs=float(np.max(abs(b-a))),
                            relative_fro=float(np.linalg.norm(b-a)/max(np.linalg.norm(b), 1e-30)),
                            before_real=a.real.tolist(), before_imag=a.imag.tolist(),
                            after_real=b.real.tolist(), after_imag=b.imag.tolist())
    report = dict(status='finite_resonance_core_control_not_physical_qualification',
                  ntpc=4096, levels={p: {k: meta[p][k] for k in
                      ('n_nodes', 'n_contracted_columns', 'n_trace', 'normal_ccr_maxabs',
                       'anomalous_maxabs', 'U0_gram_defect', 'total_wall_s', 'row_sha256')}
                      for p in meta}, changes=changes,
                  all_original_coordinates_reproduced=meta[96]['all_checks_pass'],
                  only_resonance_core_density_changed=True, production_qualification=False,
                  original_jobs_modified=False)
    with TARGET.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    a, b = meta[96], meta[128]
    ratio = b['normal_ccr_maxabs']/a['normal_ccr_maxabs']
    interpretation = (f"正常残差从 {a['normal_ccr_maxabs']:.6f} 变为 {b['normal_ccr_maxabs']:.6f}。"
                      '本轮它增大了，说明输出对共振核心求积仍敏感，并未得到收敛。n,m 的小相对变化不能代替输出对易关系的检验；输入求积误差仍不能排除。')
    next_step = ('下一项优先保持时间网格和侧翼不变，检验共振核心更细一层能否延续下降；通过后仍须独立检查固定侧翼和有限频率范围。'
                 if ratio < .8 else
                 '下一项优先固定共振核心为 128、时间网格为 4096、频率端点不动，仅将非共振侧翼面板节点加倍，再比较完整有符号 CCR 矩阵和 n,m。复用当前反向行，不必重新传播电子背景或所有输入列。')
    text = f'''# Γ12 时间步复核与输入核心求积对照（2026-10-08）

## 当前结论

{interpretation}

本轮固定 `+N`、Γ 心 12×12、每周期 4096 步、九周期时间窗，完整内部电子坐标和 1976 条暗列。只把原频率求积中各共振核心的参数 `n_panel` 从 96 改成 128。模型、初态、D、gamma、P1、偏振、packet、输入频率端点、H2 侧翼密度与其他尾面板均未改变。

## 先复核另一位 AI 的时间阶梯结论

全列原始数据确认：`tr n` 为 73207.22294 → 78298.45774 → 79632.17712；正常 CCR 最大残差为 0.852834 → 0.851723 → 0.851325。n,m 的变化接近二阶，约 0.85 的剩余不能由已观察到的时间步修正解释。

原报告的外推约 0.8515 有符号错误，已更正为对残差矩阵作 `(4 R4096 - R2048)/3` 后的最大绝对元素 **0.851191862**。它只是三层数据下的形式诊断，不能写成连续极限证明。原 JSON 保留；修正数字见 `gamma12_output_transpose_20261008/full_time_comparison_reviewed.json`。

静态输入残差 0.03225 仅被复现，尚未排除其在驱动中的影响；共同输出误差也不能直接当成磁畴差的下界，配对误差可能相消。

## 新对照的结果

| 共振核心节点参数 | 实际频率节点 | 全部输入列 | tr n（未资格化） | 正常 CCR 最大残差 | 反常 CCR 最大残差 |
|---:|---:|---:|---:|---:|---:|
| 96 | {a['n_nodes']} | {a['n_contracted_columns']} | {a['n_trace']:.8f} | {a['normal_ccr_maxabs']:.9g} | {a['anomalous_maxabs']:.9g} |
| 128 | {b['n_nodes']} | {b['n_contracted_columns']} | {b['n_trace']:.8f} | {b['normal_ccr_maxabs']:.9g} | {b['anomalous_maxabs']:.9g} |

相对 128 的完整矩阵 Frobenius 变化：n 为 **{100*changes['n']['relative_fro']:.6g}%**，m 为 **{100*changes['m']['relative_fro']:.6g}%**。正常残差矩阵差的最大绝对元素为 **{changes['normal_contribution']['maxabs']:.9g}**；反常为 **{changes['anomalous_contribution']['maxabs']:.9g}**。不是通过归一化输出消除残差。

无泵输出 U0 的 Gram 缺陷分别为 {a['U0_gram_defect']:.6g}、{b['U0_gram_defect']:.6g}。新节点上的 U0 均从冻结的 stationary 公式重算，未插值旧行。

## 此次计算为什么可与旧作业比较

反向算法只计算固定探测泛函，与当前离散中点传播严格等价；推导另见《线Q_固定输出模式的离散转置推导_20261008.md》。本轮复用 4096 的反向行，频率列按块新建和收缩，保留完整初值及自由场强迫项。

96 先通过全部频率、权重、b/bdag/A/free_A、U0 的逐列回归，又通过完整 U,V、原始有泵/无泵行、n,m、CCR 对照，才开放 128。全部暗列读取原缓存一次，无第二套明真空。128 是求积控制计算，不是与另一套独立 128 前向全列传播的对照。

本次两次墙时为 {a['total_wall_s']:.1f} s 和 {b['total_wall_s']:.1f} s，包含输入重建与收缩；不含已有电子轨迹和反向行的创建。它只替代所选光学输出的计算，不提供完整电子末态/Pauli 诊断。

## 下一步范围与停止点

{next_step}

若这一数值轴仍不能解释残差，再依据结果分开处理有限频率端点、有限探测时间和输出公式。不要一次改变它们，也不靠修改初态、D、耗散或模型使残差变小。当前不需要新的数学物理选择；只有证据迫使改变这些物理定义或近似时，再交由本人审核。

Γ12 输出仍未通过 canonical 资格，因此本页 n,m 只是诊断值；不作光子产额、磁畴差、压缩、纠缠或 Gaussian 近似有效性的结论。也没有重新运行 -N、动态更密 BZ 或旧前向长作业。

## 产物与复现

- `results/q_line/gamma12_output_transpose_input_refine_20261008/n_panel_96.npz`、同名 JSON。
- 同目录 `n_panel_128.npz`、同名 JSON、`comparison.json`（含有符号矩阵）。
- 新脚本 `tools/b0_small_model/q8_m0_gamma12_transpose_input_refine_20261008.py`；先运行 `--n-panel 96`，通过身份/回归门后运行 `--n-panel 128`。既有产物存在时拒绝覆写。
- 本摘要由 `q8_m0_gamma12_input_refine_summary_20261008.py` 从哈希核对后的 NPZ/JSON 生成。
- 冻结 r1 和两条原始全量作业保持不变。
'''
    with DOC.open('x', encoding='utf-8') as f:
        f.write(text)
    print(json.dumps({'normal_ratio':ratio, 'n_relative':changes['n']['relative_fro'],
                      'm_relative':changes['m']['relative_fro'], 'document':str(DOC)}, indent=2))


if __name__ == '__main__':
    main()

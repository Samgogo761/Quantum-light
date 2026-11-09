"""Generate the advisor-report PPTX for the CrI3-AFM quantum-light HHG project.

Reusable: re-run after updating figures/content. Requires python-pptx + the
figures under docs/figures/{report,bsv_quadrature}/.
"""
from __future__ import annotations
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

REPO = Path(r"C:/Users/26507/Documents/New_SBEs/Quantum-light")
FIG = REPO / "docs" / "figures"
OUT = REPO / "docs" / "量子光CrI3_阶段汇报_20260615.pptx"

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
RED = RGBColor(0xC6, 0x28, 0x28)
W, H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, x, y, w, h):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def settext(p, text, size, color=DARK, bold=False, align=PP_ALIGN.LEFT, italic=False):
    p.text = text
    p.alignment = align
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color


def header(s, title, kicker=None):
    bar = s.shapes.add_shape(1, 0, 0, W, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    tf = box(s, Inches(0.45), Inches(0.12), Inches(12.4), Inches(0.85))
    settext(tf.paragraphs[0], title, 26, RGBColor(0xFF, 0xFF, 0xFF), bold=True)
    if kicker:
        p = tf.add_paragraph()
        settext(p, kicker, 13, RGBColor(0xCF, 0xDD, 0xEE))


def bullets(s, items, x=Inches(0.55), y=Inches(1.3), w=Inches(7.2), h=Inches(5.6), size=15):
    tf = box(s, x, y, w, h)
    first = True
    for lvl, txt, *rest in items:
        col = rest[0] if rest else DARK
        bold = rest[1] if len(rest) > 1 else False
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        prefix = ("" if lvl == 0 else "   " * lvl)
        mark = ["▸ ", "– ", "· "][min(lvl, 2)]
        settext(p, prefix + mark + txt, size - lvl, col, bold=bold)
        p.space_after = Pt(5)


def pic(s, path, x, y, w=None, h=None):
    if Path(path).exists():
        s.shapes.add_picture(str(path), x, y, width=w, height=h)


def caption(s, text, x, y, w):
    tf = box(s, x, y, w, Inches(0.4))
    settext(tf.paragraphs[0], text, 11, GREY, italic=True, align=PP_ALIGN.CENTER)


# ============================ 1. Title =====================================
s = slide()
band = s.shapes.add_shape(1, 0, Inches(2.4), W, Inches(2.7))
band.fill.solid(); band.fill.fore_color.rgb = BLUE; band.line.fill.background()
tf = box(s, Inches(0.8), Inches(2.7), Inches(11.7), Inches(2.0))
settext(tf.paragraphs[0], "量子光驱动双层 CrI₃ 反铁磁高次谐波", 36,
        RGBColor(0xFF, 0xFF, 0xFF), bold=True)
p = tf.add_paragraph()
settext(p, "理论框架、求解器与 BSV 量子光阶段进展", 22, RGBColor(0xDD, 0xE8, 0xF4))
tf2 = box(s, Inches(0.8), Inches(5.4), Inches(11.7), Inches(1.2))
settext(tf2.paragraphs[0], "阶段汇报 · 2026-06", 16, GREY)
p = tf2.add_paragraph()
settext(p, "SBE/HHG 求解器 · AFM 选择定则 · 量子几何/自旋流 · BSV 确定性求积 · Tier-2 量子签名路线",
        13, GREY)

# ============================ 2. Background =================================
s = slide(); header(s, "研究背景与定位", "为什么是双层 CrI₃ + 量子光")
bullets(s, [
    (0, "材料: 双层 CrI₃ — 2D 磁性绝缘体, 强 SOC, 层间反铁磁(AFM)", DARK, True),
    (1, "对称性: P 破缺 + T 破缺 + PT 守恒 → 偶次谐波可由 Néel 序参量 N 调制"),
    (1, "实验背景: Sun et al., Nature 2019 — CrI₃ 巨大非互易二次谐波(磁性 SHG)"),
    (0, "方法背景: 量子光驱动 HHG 是新兴方向", DARK, True),
    (1, "Gorlach et al., Nat. Phys. 2023 — 量子光驱动 HHG, 光子统计塑形"),
    (1, "Lewenstein et al., Nat. Phys. 2021 — HHG 产生光学猫态"),
    (1, "González-Monge & Feist 2025 — 自由空间 HHG '无量子优势'(关键边界)"),
    (0, "本研究的空白点(卖点)", BLUE, True),
    (1, "双层 CrI₃ 的 HHG 计算几乎没有; 量子光驱动 + AFM 选择定则 = 空白"),
], w=Inches(12.3))

# ============================ 3. Framework =================================
s = slide(); header(s, "计算框架", "从头算 → Wannier → 实时 SBE")
bullets(s, [
    (0, "电子结构: Quantum ESPRESSO (SOC, 非共线, Hubbard U), 双层 AFM 初态", DARK, True),
    (0, "紧束缚: Wannier90 — 112 带 d+p 旋量模型(完整 PT 完备流形)", DARK, True),
    (0, "实时动力学: 自研长度规范协变 SBE 求解器 (lg_cov)", DARK, True),
    (1, "dρ/dt = -i[H₀+E·D, ρ] + E·∇ₖρ(平行输运协变导数), RK4 积分"),
    (1, "瞬时缀饰基 T2 退相干; FFT(Hann 窗) → HHG 谱"),
    (1, "OpenMP over k 点; 预计算规范矩阵(BSV 多样本复用)"),
    (0, "生产参数", GREEN, True),
    (1, "lg_cov / 40×40 / nb1-104(经济)或 112 / T2=1 cycle / λ=3200 nm / I=2×10¹¹"),
], w=Inches(12.3))

# ============================ 4. Classical validation =====================
s = slide(); header(s, "经典光验证: 求解器可信度", "进入量子光前的全套检查")
bullets(s, [
    (0, "规范: lg_cov 是物理正确生产规范", DARK, True),
    (1, "速度规范截断破坏 f-sum 规则 → 价带背景虚电流, 已排除"),
    (0, "收敛: k 网格 40×40 (H1-H15 变化 0-2%), 60×60 确认", DARK, True),
    (0, "带窗: nb1-104(全价带+截高导带) ≈ full112, 经济且无虚电流", DARK, True),
    (0, "退相干: T2=1 cycle(=10.7 fs, 文献 2D-AFM 标准), 峰 FWHM 0.32 阶", DARK, True),
    (0, "硬验证(全部通过)", GREEN, True),
    (1, "初始电流 |J(0)| ~ 10⁻¹⁸ (机器零)"),
    (1, "弱场标度: H1/H2/H3 接近 1/2/3 阶响应"),
], w=Inches(6.4))
pic(s, FIG / "report" / "figA_classical_hhg.png", Inches(7.0), Inches(1.45), w=Inches(6.0))
caption(s, "经典 HHG: T2=1 cycle 下峰锐且分立", Inches(7.0), Inches(5.7), Inches(6.0))

# ============================ 5. Selection rule ============================
s = slide(); header(s, "核心方法贡献: AFM 选择定则闭环", "偶次谐波 ↔ Néel 序参量")
bullets(s, [
    (0, "判据: Jₙ(-N) = (-1)ⁿ⁺¹ Jₙ(N) 需 +N/-N 严格 P 共轭", DARK, True),
    (0, "诊断: 两次独立 DFT 的 +N/-N 不配对", RED, True),
    (1, "SCF 总能差 2.18 mRy; Ω_I 差 0.13 Ų; 本征值差 50-110 meV"),
    (1, "→ H2 主要落在错误通道, leakage ≈ 13"),
    (0, "解决: 从 +N TB 后处理构造严格 P 共轭 -N", GREEN, True),
    (1, "原子置换 + 轨道宇称 + R→-R + 位置块 r→-r"),
    (1, "本征值 E₋ₙ(k)=E₊ₙ(-k) 配对到 10⁻¹⁴(机器精度)"),
    (0, "结果: H2 leakage 13.1 → 0.32(定性翻转)", BLUE, True),
], w=Inches(6.4))
pic(s, FIG / "report" / "figB_selrule.png", Inches(7.0), Inches(1.55), w=Inches(6.0))
caption(s, "严格 P 配对后偶次谐波回到正确通道", Inches(7.0), Inches(5.8), Inches(6.0))

# ============================ 6. Tier0/1 observables =======================
s = slide(); header(s, "求解器可观测量增强 (Tier 0/1)", "让结果更有说服力")
bullets(s, [
    (0, "量子几何: Berry 曲率 Ω_n(k) + 量子度规 g_n(k)", DARK, True),
    (1, "PT 对称 ⟹ 占据流形 Ω_occ ≈ 0.08-0.32 bohr²(强压制)"),
    (1, "⟹ 反常(Berry)电流 j_anom 可忽略, 偶次谐波由带间极化 j_pol 主导"),
    (1, "陷阱已解决: 每带 Ω 受 Kramers 简并发散, 须看占据流形和"),
    (0, "量子度规 g(PT-even): 偶次谐波的几何起源(新颖叙事)", DARK, True),
    (0, "k 空间占据快照 ρ_nn(k,t): 激发动力学可视化", DARK, True),
    (0, "自旋分辨电流(可插拔 S_z): 名义自旋已过 AFM 自检", DARK, True),
    (1, "价带 Σ⟨S_z⟩=0.002≈0, 各带强极化 → 严格版待 .spn"),
], w=Inches(12.3), size=15)

# ============================ 7. BSV pilot -> quadrature ===================
s = slide(); header(s, "BSV 量子光: 蒙特卡洛 → 确定性求积", "方法学升级")
bullets(s, [
    (0, "BSV(亮压缩真空)= 随机相位下指数强度分布 Q(I)=e^{-I/I₀}/I₀", DARK, True),
    (0, "Pilot(51 样本 MC): 看到增强但相对误差 18-44%", RED, True),
    (1, "MC 对指数分布低效; 压到 10% 需 ~970 样本(~24 天)"),
    (0, "升级: BSV 平均 = 一维积分 → Gauss-Legendre 求积", GREEN, True),
    (1, "16 个确定性强度节点; 复现矩 ⟨xᵖ⟩=p! 到 ≤0.3%(p≤7)"),
    (1, "零噪声, 便宜 >100×(等效 >10⁴ MC 样本)"),
    (0, "四项验证全过: 归一化 1.000 / 截断 0% / dt 0.9% / 与 MC 一致 2σ", BLUE, True),
], w=Inches(6.3))
pic(s, FIG / "report" / "figC_quad_vs_mc.png", Inches(6.9), Inches(1.5), w=Inches(6.2))
caption(s, "求积(蓝)与 MC(橙±误差)一致, 但无噪声", Inches(6.9), Inches(5.7), Inches(6.2))

# ============================ 8. BSV spectra ===============================
s = slide(); header(s, "BSV 主结果(一): 增强谱", "BSV 平均 vs 经典")
pic(s, FIG / "bsv_quadrature" / "fig1_spectra.png", Inches(1.4), Inches(1.35), w=Inches(8.2))
bullets(s, [
    (0, "BSV 平均谱整体抬升", BLUE, True),
    (0, "高次谐波尤其显著", DARK),
    (0, "谱形干净、可发表", DARK),
    (0, "(16 节点求积, 零 MC 噪声)", GREY),
], x=Inches(9.9), y=Inches(2.3), w=Inches(3.2), size=15)

# ============================ 9. BSV enhancement ===========================
s = slide(); header(s, "BSV 主结果(二): 增强曲线", "随阶数单调上升, 但远低于 n!")
pic(s, FIG / "bsv_quadrature" / "fig2_enhancement.png", Inches(1.4), Inches(1.35), w=Inches(8.2))
bullets(s, [
    (0, "H1:1.2 → H5:10", DARK, True),
    (0, "→ H11:60 → H13:363×", DARK, True),
    (0, "灰线 = n! 微扰上限", GREY),
    (0, "H3-H4 起偏离 n!", RED, True),
    (1, "= 微扰→平台过渡"),
    (0, "偶次 H4 被压(选择定则)", DARK),
], x=Inches(9.9), y=Inches(1.9), w=Inches(3.2), size=15)

# ============================ 10. Physics reading ==========================
s = slide(); header(s, "物理解读", "增强从哪来, 为什么饱和")
bullets(s, [
    (0, "机制: 指数强度分布的高强度尾延展 HHG cutoff", DARK, True),
    (1, "越靠近/越过经典 cutoff 的谐波, 越被高 I 稀有事件加权 → 增强越大"),
    (0, "为何远低于 n!(关键观察)", BLUE, True),
    (1, "n! 是微扰极限 ⟨Iⁿ⟩/⟨I⟩ⁿ; 平台谐波非微扰, I-标度远弱于 Iⁿ"),
    (1, "偏离 n! 的拐点(H3-H4)本身标志微扰→平台过渡 — 有物理含义"),
    (0, "偶/奇结构: 偶次 H4 被压(AFM 选择定则 + 饱和)", DARK, True),
    (0, "求积纠正了 MC 偏差: MC 偏高 ~1.5σ(抽到的高 I 偏多)", DARK, True),
    (1, "H13: 求积 363× vs MC(噪声)949× — 求积为真值"),
], w=Inches(12.3))

# ============================ 11. Tradeoff / positioning ===================
s = slide(); header(s, "物理定位与利弊权衡", "诚实评估")
tf = box(s, Inches(0.5), Inches(1.25), Inches(12.4), Inches(0.6))
settext(tf.paragraphs[0],
        "当前 BSV 增强 = 强度涨落平均(半经典/Gorlach 路线), 可被经典强度分布复现 → 属 Feist '无量子优势'区。",
        15, DARK, bold=True)
# table
rows = [
    ("层级", "物理内容", "利", "弊/风险"),
    ("R1 强度平均(现在)", "BSV 平均谱 + 增强 + 选择定则", "双层 CrI₃ HHG+量子光本身空白, 论文1 站得住; 数据已齐", "非'量子优势'; 可经典复现"),
    ("R2 输出光非经典性", "QRT 多时关联 → 压缩/纠缠/光子统计 + 磁序指纹", "与 Feist 不冲突; 增量大且可发表; 架构已设计", "T2≈10fs 压制信号, 需先估量级"),
    ("R3 严格量子优势", "证明经典分布不可复现某观测量/估计精度", "原创性最高", "凸优化/矩问题, 难度大, 长线"),
]
left, top = Inches(0.5), Inches(2.0)
tw, th = Inches(12.3), Inches(4.6)
tbl = s.shapes.add_table(len(rows), 4, left, top, tw, th).table
tbl.columns[0].width = Inches(2.3); tbl.columns[1].width = Inches(3.6)
tbl.columns[2].width = Inches(3.4); tbl.columns[3].width = Inches(3.0)
for j, txt in enumerate(rows[0]):
    c = tbl.cell(0, j); c.text = txt
    c.fill.solid(); c.fill.fore_color.rgb = BLUE
    for p in c.text_frame.paragraphs:
        for r in p.runs: r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
for i in range(1, len(rows)):
    for j, txt in enumerate(rows[i]):
        c = tbl.cell(i, j); c.text = txt
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(11)
                r.font.color.rgb = (GREEN if j == 2 else (RED if j == 3 else DARK))

# ============================ 12. Tier 2 roadmap ===========================
s = slide(); header(s, "Tier 2: 真正的量子签名路线", "如何区别于'无量子优势'")
bullets(s, [
    (0, "核心新对象: 二阶电流关联 C^{ab}(t,t') = ⟨Ĵ^a(t)Ĵ^b(t')⟩", DARK, True),
    (1, "量子回归定理(QRT): 复用现有传播子, 额外演化辅助算符 v̂ρ"),
    (1, "频域 C(ω_q,ω_q') → 输出多模高斯态(均值+协方差矩阵)"),
    (0, "由 C 直接后处理出(R2 全部观测量)", BLUE, True),
    (1, "谐波强度 ⟨n_q⟩, 二阶相干 g²(q,q'), 压缩谱, 谐波间纠缠 E_N"),
    (1, "Néel 宇称分解 C = C_ee + |N|²C_oo: C_oo 是 AFM 独有的'磁性标记关联'"),
    (0, "关键: 相位平均杀不死 BSV 非经典性", RED, True),
    (1, "BSV 光子数只有偶数占据(SU(1,1) 配对), P(奇)=0 — 经典混合不可复现"),
    (1, "突破口: 反常关联 C_A ∝ e^{iθ}sinh r cosh r → 谱对压缩角 θ 的依赖"),
    (1, "(c 数强度采样里根本没有 θ 这个参数)"),
], w=Inches(12.3), size=14)

# ============================ 13. Literature map ===========================
s = slide(); header(s, "文献地图", "理论框架要结合的现有工作")
bullets(s, [
    (0, "量子光 HHG", BLUE, True),
    (1, "Gorlach Nat.Phys.2023(量子光HHG); Lewenstein Nat.Phys.2021(光学猫态)"),
    (1, "Stammer & Rivera-Dean 2504.13287(多时关联框架, 首要精读)"),
    (1, "Lange-Hansen-Madsen PRL 135,043603(2025)(Mott 量子光HHG, 最接近)"),
    (1, "González-Monge & Feist 2512.06602(无量子优势; 找其'仅依赖瞬时强度'假设=反例入口)"),
    (0, "数学/量子信息工具", BLUE, True),
    (1, "Glauber/Sudarshan P 表示(非经典性的精确定义)"),
    (1, "Weedbrook RMP 84,621(2012)(高斯量子信息总评)"),
    (1, "Simon/Duan PRL 84(2000)(高斯纠缠判据); Vidal-Werner(对数负性)"),
    (1, "Shchukin-Vogel(非经典性矩判据层级); 高斯 QFI 闭式(Pinel 2013)"),
    (1, "Gardiner-Zoller《Quantum Noise》(压缩库 Bloch 方程 = Tier2 Step2 原型)"),
    (0, "材料", BLUE, True),
    (1, "Sun et al. Nature 2019(CrI₃ 非互易 SHG)"),
], w=Inches(12.3), size=13)

# ============================ 14. Next steps ===============================
s = slide(); header(s, "现状小结与下一步建议")
bullets(s, [
    (0, "完整且干净的工作链(本阶段成果)", GREEN, True),
    (1, "经典光验证 → AFM 选择定则闭环 → Tier0/1 可观测量 → BSV pilot → 确定性求积生产谱"),
    (1, "论文 1(R1)核心数据已齐: 经典谱 + BSV 平均谱 + 增强曲线 + 选择定则 + 几何/自旋"),
    (0, "建议下一步", BLUE, True),
    (1, "① 论文 1 开始成文(双层 CrI₃-AFM 的 BSV-HHG, R1 层定位)", GREEN, True),
    (1, "② 并行按文献地图推 Tier 2 / R2(输出光非经典性 + 磁性指纹)"),
    (1, "③ (可选)严格自旋流: 服务器生成 .spn → 升级名义自旋"),
    (0, "权衡: R1 稳健可发但非优势; R2 增量大但需估 T2 压制下的量级", DARK, True),
], w=Inches(12.3))

prs.save(str(OUT))
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))

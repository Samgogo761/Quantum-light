"""Build a self-contained HTML work report (figures embedded) for the advisor.

Reads the curated figures, inlines SVGs and base64-encodes PNGs, and emits one
standalone .html. Re-run after updating figures/content.
"""
from __future__ import annotations
import base64
from pathlib import Path

REPO = Path(r"C:/Users/26507/Documents/New_SBEs/Quantum-light")
FIG = REPO / "docs" / "figures"
OUT = REPO / "docs" / "量子光CrI3_工作报告_20260615.html"


def embed(path: Path, max_w="100%") -> str:
    p = Path(path)
    if not p.exists():
        return f'<div class="missing">[缺图: {p.name}]</div>'
    if p.suffix == ".svg":
        svg = p.read_text(encoding="utf-8", errors="replace")
        i = svg.find("<svg")
        return f'<div class="svgwrap" style="max-width:{max_w}">{svg[i:]}</div>'
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img style="max-width:{max_w}" src="data:image/png;base64,{data}"/>'


def fig(path, cap, w="780px"):
    return f'<figure>{embed(path, w)}<figcaption>{cap}</figcaption></figure>'


CV = FIG / "classical_validation"
RP = FIG / "report"
BQ = FIG / "bsv_quadrature"

SECTIONS = []

SECTIONS.append(("研究背景与科学问题", "1", f"""
<ul>
<li><b>材料</b>:双层 CrI₃,2D 磁性绝缘体,强 SOC,层间反铁磁(AFM)。对称性 = P 破缺 + T 破缺 +
<b>PT 守恒</b> ⟹ 偶次谐波可被 Néel 序参量 <b>N</b> 调制。</li>
<li><b>实验锚点</b>:Sun et al., <i>Nature</i> 563, 47 (2019) — CrI₃ 巨大非互易二次谐波(磁性 SHG)。</li>
<li><b>方法前沿</b>:量子光驱动 HHG 是新兴方向(Gorlach, <i>Nat. Phys.</i> 2023;Lewenstein, <i>Nat. Phys.</i> 2021)。</li>
<li><b>空白点(卖点)</b>:双层 CrI₃ 的 HHG 计算几乎没有;量子光驱动 + AFM 选择定则 = 完全空白。</li>
</ul>
<div class="callout">核心问题:量子光(BSV)驱动下,AFM 序参量如何编码在 HHG 谱与谐波光子统计中?</div>
"""))

SECTIONS.append(("计算框架", "2", f"""
<p>链路:<b>Quantum ESPRESSO(SOC+U,AFM 初态)→ Wannier90(112 带 d+p spinor)→ 自研 lg_cov SBE 求解器</b>。</p>
<ul>
<li><b>112 带 = 化学完整母模型</b>(非随意大窗口):4 Cr × 5 d + 12 I × 3 p = 56 轨道 × 2(自旋/SOC)= <b>112</b>。
保留 Cr-d/I-p 杂化、SOC 自旋混合、深价带对 Berry 联络/偶极/速度矩阵元的贡献(84 价带 + 28 导带)。</li>
<li><b>运动方程(长度规范协变)</b>:
<div class="eq">dρ/dt = −i[H₀ + E·D, ρ] + E·∇<sub>k</sub>ρ</div>
平行输运 5 点有限差分协变导数,RK4 积分,瞬时缀饰基 T2 退相干,Hann 窗 FFT → HHG 谱。</li>
<li>工程:OpenMP over k;规范矩阵预计算(BSV 多样本复用)。</li>
</ul>
"""))

SECTIONS.append(("求解器验证(经典光)", "3", f"""
<p class="lead">进入量子光前,把"求解器可信"做扎实:规范 / 带窗 / k 网格 / 退相干 + 硬指标。</p>

<h3>3.1 规范选择:速度规范截断为何失败</h3>
<ul><li>Peierls 速度规范 + 对称带窗截断:H1 在 20 / 40 / 60 带 = 48 / 110 / 168,而 full112 = 0.082
(<b>暴涨 3 个量级</b>)→ 截掉深价带破坏速度规范的背景电流抵消(f-sum 规则)。</li>
<li>结论:<b>lg_cov(长度规范协变)是物理正确的生产规范</b>;VG 截断不可用于生产。</li></ul>
{fig(CV/"fig01_peierls_vg_symmetric_truncation_failure.svg","fig01 · Peierls 速度规范对称截断的低阶虚信号(故失败)")}

<h3>3.2 带窗收敛:经济配置 nb1-104 ≈ full112</h3>
<ul><li>不对称截断必须在 lg_cov 中验证。<b>nb1-104</b>(全 84 价带 + 截高 20 导带)相对 full112:
H1-H11 在 <b>1%-4%</b> 内,偶次在 <b>1%-3%</b> 内,|J(0)|≈4.42×10⁻¹⁸(同 full112 一样干净)。</li>
<li>结论:<b>full112 = 绝对基准;nb1-104 = 经济生产配置</b>(BSV 用它)。</li></ul>
<div class="figrow">
{fig(CV/"fig13_hhg_spectrum_lgcov_fullvalence_T2_0p5cycle.svg","fig13 · nb1-104 vs full112 谱","520px")}
{fig(CV/"fig14_hhg_ratio_lgcov_nb104_full112_T2_0p5cycle.svg","fig14 · H1-H11 比值","520px")}
</div>

<h3>3.3 k 网格收敛</h3>
<ul><li>full112:30×30 → 40×40,H1-H15 变化 0%-2%;40×40 vs 60×60 确认收敛 ⟹ <b>40×40 足够</b>。</li></ul>
{fig(CV/"fig16_hhg_ratio_lgcov_full112_k60_k40_T2_0p5cycle.svg","fig16 · k40 vs k60 谱比值(full112)","640px")}

<h3>3.4 退相干 T2</h3>
<ul><li>扫了 T2 = 0.5/1/2/5 fs / 0.5 cycle / no-deph。选定 <b>T2 = 1 光周期 = 10.7 fs</b>(文献 2D-AFM 常用),
峰 FWHM ≈ 0.32 阶(峰锐且分立)。</li></ul>
<div class="figrow">
{fig(CV/"fig18_hhg_ratio_lgcov_full112_T2_to_nodeph.svg","fig18 · T2 扫描相对无退相干的比值","520px")}
{fig(RP/"figA_classical_hhg.png","figA · 经典 HHG @ T2=1cyc(峰 FWHM 0.32)","520px")}
</div>

<h3>3.5 硬指标</h3>
<ul><li>所有生产配置 <b>|J(0)| ~ 10⁻¹⁸(机器零)</b>,初态干净;截价带窗口会产生非零初始电流故不用。</li>
<li>弱场标度(I=2×10¹⁰):Jt 投影的 H1/H2/H3 接近 1/2/3 阶响应 ⟹ 非线性阶次正确。</li></ul>
<p class="note">说明:经典验证共 22 张归档图,这里只选必要的 6 张(规范 / 带窗谱+比值 / k / T2);
其余中间过程图(fig02/05/06/07/08/09/10/12/19/20/21 等)归档不入报告。</p>
"""))

SECTIONS.append(("核心方法贡献:AFM 选择定则闭环", "4", f"""
<ul>
<li><b>判据</b>:<span class="eq-inline">Jₙ(−N) = (−1)ⁿ⁺¹ Jₙ(N)</span>,要求 +N/−N 是严格空间反演(P)共轭对。</li>
<li><b>诊断(失败负例)</b>:两次独立 DFT 的 +N/−N <b>不配对</b> —— SCF 总能差 2.18 mRy(0.03 eV);
Wannier Ω<sub>I</sub> 差 0.13 Ų;Γ/K/M 本征值差 50-110 meV ⟹ H2 落错通道,leakage ≈ <b>13</b>。</li>
<li><b>解决(方法创新)</b>:从 +N TB <b>后处理构造</b>严格 P 共轭 −N(原子置换 + 轨道宇称 d:+1/p:−1
+ R→−R + 位置块 r→−r)。验证 E₋ₙ(k)=E₊ₙ(−k) 配对到 <b>10⁻¹⁴</b>(4 道验证门全过)。</li>
<li><b>结果</b>:H2 leakage <b>13.1 → 0.32</b>(从错误 evenN 通道回到正确 oddN 通道)。</li>
<li><b>补强</b>:偶次谐波 intra/inter 分解显示偶次由<b>带间极化电流主导</b>(inter/tot 0.78-0.99)。</li>
</ul>
{fig(RP/"figB_selrule.png","figB · 严格 P 配对后 H2 leakage 13.1 → 0.32","680px")}
"""))

SECTIONS.append(("求解器可观测量增强(Tier 0/1)", "5", f"""
<ul>
<li><b>量子几何</b>:Berry 曲率 Ω<sub>n</sub>(k) + 量子度规 g<sub>n</sub>(k)。PT 对称 ⟹ 占据流形
<b>Ω<sub>occ</sub> ≈ 0.08-0.32 bohr²</b>(强压制)⟹ 反常(Berry)电流 j<sub>anom</sub> 可忽略 ⟹ 偶次由带间极化主导。
<br><span class="warn">关键陷阱</span>:每带 Ω 受 Kramers 简并发散(~10⁶),必须看占据流形和。
量子度规 g(PT-even,存活)= 偶次谐波的几何起源(新颖叙事)。</li>
<li><b>原生可视化导出</b>:k 空间占据 ρ_nn(k,t)、interband coherence、场 Et.dat ⟹ 为 HHG-XR 科研可视化
提供真实计算数据(而非从 input 重建)。</li>
<li><b>自旋分辨电流(可插拔 S_z)</b>:名义自旋已过 AFM 自检(价带 Σ⟨S_z⟩=0.002≈0,各带强极化);
严格版待 .spn。</li>
</ul>
"""))

SECTIONS.append(("BSV 量子光:蒙特卡洛 → 确定性求积", "6", f"""
<ul>
<li>BSV(亮压缩真空)随机相位下 = 指数强度分布 <span class="eq-inline">Q(I)=e<sup>−I/I₀</sup>/I₀</span>。</li>
<li><b>Pilot(51 样本 MC)</b>:看到增强但相对误差 18-44%;压到 10% 需 ~970 样本(~24 天)⟹ MC 不可行。</li>
<li><b>升级</b>:BSV 平均 = 一维积分 ⟹ <b>Gauss-Legendre 求积(16 节点)</b>,复现矩 ⟨xᵖ⟩=p! 到 ≤0.3%(p≤7),
<b>零噪声,便宜 &gt;100×</b>(等效 &gt;10⁴ MC 样本)。</li>
<li><b>四项验证全过</b>:归一化 1.000 / 截断敏感度 0% / dt 收敛 0.9%(最强节点 3.55×10¹²)/ 与 MC 一致 2σ。</li>
</ul>
{fig(RP/"figC_quad_vs_mc.png","figC · 求积(蓝)与 51 样本 MC(橙 ± 误差)一致,但无噪声","680px")}
"""))

SECTIONS.append(("BSV 主结果", "7", f"""
<ul>
<li><b>增强曲线</b>:BSV/经典 = H1:1.2 → H5:10 → H11:60 → <b>H13:363×</b>,随阶数单调上升。</li>
<li><b>物理解读</b>:① 机制 = 指数强度尾<b>延展 HHG cutoff</b>;② 增强<b>远低于 n!</b>(微扰上限)——
平台谐波非微扰,I-标度远弱于 Iⁿ;偏离 n! 的拐点(H3-H4)标志<b>微扰→平台过渡</b>;
③ 偶次 H4 被压(AFM 选择定则 + 饱和);④ 求积纠正 MC 偏差(H13 求积 363× vs MC 噪声 949×)。</li>
</ul>
<div class="figrow">
{fig(BQ/"fig1_spectra.png","fig1 · BSV 平均谱 vs 经典谱","520px")}
{fig(BQ/"fig2_enhancement.png","fig2 · 增强曲线 + n! 微扰上限","520px")}
</div>
{fig(BQ/"fig3_scaling.png","fig3 · 各谐波单强度标度 S_n(I)(16 节点)","640px")}
"""))

SECTIONS.append(("物理定位与利弊权衡", "8", """
<div class="callout">诚实界定:当前 BSV 增强 = 强度涨落平均(半经典/Gorlach 路线),<b>可被经典强度分布复现</b>
→ 属 González-Monge/Feist "无量子优势"区。</div>
<table class="tbl">
<tr><th>层级</th><th>物理内容</th><th class="good">利</th><th class="bad">弊 / 风险</th></tr>
<tr><td><b>R1 强度平均(现在)</b></td><td>BSV 平均谱 + 增强 + 选择定则</td>
<td class="good">双层 CrI₃ HHG+量子光本身空白,论文1 站得住;数据已齐</td><td class="bad">非"量子优势";可经典复现</td></tr>
<tr><td><b>R2 输出光非经典性</b></td><td>QRT 多时关联 → 压缩/纠缠/光子统计 + 磁序指纹</td>
<td class="good">与 Feist 不冲突;增量大且可发表;架构已设计</td><td class="bad">T2≈10fs 压制信号,需先估量级</td></tr>
<tr><td><b>R3 严格量子优势</b></td><td>证明经典分布不可复现某观测量/估计精度</td>
<td class="good">原创性最高</td><td class="bad">凸优化/矩问题,难度大,长线</td></tr>
</table>
"""))

SECTIONS.append(("Tier 2 路线:真正的量子签名", "9", """
<ul>
<li><b>核心新对象</b>:二阶电流关联 <span class="eq-inline">C<sup>ab</sup>(t,t') = ⟨Ĵ<sup>a</sup>(t)Ĵ<sup>b</sup>(t')⟩</span>,
用<b>量子回归定理</b>(复用现有传播子,额外演化辅助算符 v̂ρ)→ 频域 → 输出多模高斯态(均值 β_q + 协方差 σ)。</li>
<li><b>由 C 后处理</b>:⟨n_q⟩、g²(q,q')、压缩谱、谐波间纠缠 E_N;Néel 宇称分解 C = C_ee + |N|²C_oo,
<b>C_oo 是 AFM 独有的"磁性标记关联"</b>。</li>
<li><b>关键论点</b>:<b>相位平均杀不死 BSV 非经典性</b> —— BSV 光子数只有偶数占据(SU(1,1) 配对),
P(奇)=0,经典相干态混合不可复现。突破口 = 反常关联 <span class="eq-inline">C_A ∝ e<sup>iθ</sup> sinh r cosh r</span>
→ 谱/偏振对<b>压缩角 θ</b> 的依赖(c 数强度采样里根本没有 θ 这个自由度)。</li>
</ul>
"""))

SECTIONS.append(("文献地图", "10", """
<ul>
<li><b>量子光 HHG</b>:Gorlach <i>Nat.Phys.</i> 2023;Lewenstein <i>Nat.Phys.</i> 2021;
Stammer &amp; Rivera-Dean arXiv:2504.13287(多时关联框架,首要精读);
Lange-Hansen-Madsen <i>PRL</i> 135,043603(2025)(Mott 量子光 HHG);
<b>González-Monge &amp; Feist</b> arXiv:2512.06602(无量子优势;反例入口=其"仅依赖瞬时强度"假设)。</li>
<li><b>数学/量子信息</b>:Glauber/Sudarshan P 表示;Weedbrook <i>RMP</i> 84,621(2012);
Simon/Duan <i>PRL</i> 84(2000);Vidal-Werner(对数负性);Shchukin-Vogel(非经典性矩判据);
高斯 QFI 闭式(Pinel 2013);Gardiner-Zoller《Quantum Noise》(压缩库 Bloch 方程)。</li>
<li><b>材料</b>:Sun et al. <i>Nature</i> 563,47(2019)。</li>
</ul>
"""))

SECTIONS.append(("现状小结与下一步", "11", """
<ul>
<li><b>完整且干净的工作链</b>:经典光验证 → AFM 选择定则闭环 → Tier0/1 可观测量 → BSV pilot → 确定性求积生产谱。</li>
<li><b>论文 1(R1)数据已齐</b>:经典谱 + BSV 平均谱 + 增强曲线 + 选择定则 + 几何/自旋。</li>
<li><b>建议</b>:① 论文 1 开始成文(R1 定位);② 并行按文献地图推 Tier 2/R2;③(可选)严格自旋流(.spn)。</li>
</ul>
"""))

# ---- assemble HTML ----
nav = "".join(f'<a href="#s{n}">{n}. {t}</a>' for t, n, _ in SECTIONS)
body = ""
for title, n, html in SECTIONS:
    body += f'<section id="s{n}"><h2><span class="num">{n}</span>{title}</h2>{html}</section>\n'

DOC = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>量子光驱动双层 CrI3-AFM 高次谐波 · 工作报告</title>
<style>
:root{{--blue:#1f4e79;--accent:#2e7d32;--red:#c62828;--ink:#222;--mut:#666;--bg:#f7f8fa;}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg);line-height:1.7}}
header.hero{{background:linear-gradient(135deg,#1f4e79,#2d6aa3);color:#fff;padding:46px 8% 34px}}
header.hero h1{{margin:0;font-size:30px}}
header.hero p{{margin:8px 0 0;color:#d8e6f4;font-size:16px}}
nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e2e6ea;padding:8px 8%;
 display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;z-index:5}}
nav a{{color:var(--blue);text-decoration:none}} nav a:hover{{text-decoration:underline}}
main{{max-width:980px;margin:0 auto;padding:18px 22px 80px;background:#fff}}
section{{padding:18px 0;border-bottom:1px solid #eef1f4}}
h2{{color:var(--blue);font-size:22px;margin:6px 0 12px;display:flex;align-items:center;gap:12px}}
h2 .num{{background:var(--blue);color:#fff;width:34px;height:34px;border-radius:8px;
 display:inline-flex;align-items:center;justify-content:center;font-size:16px}}
h3{{color:#2d6aa3;font-size:16px;margin:18px 0 6px}}
ul{{margin:6px 0 6px 4px;padding-left:22px}} li{{margin:5px 0}}
.lead{{color:var(--mut);font-style:italic;margin:2px 0 10px}}
.callout{{background:#eaf3fb;border-left:4px solid var(--blue);padding:10px 14px;margin:12px 0;border-radius:0 6px 6px 0}}
.note{{color:var(--mut);font-size:13px;margin-top:10px}}
.warn{{color:var(--red);font-weight:bold}}
.eq{{background:#f0f2f5;border-radius:6px;padding:8px 12px;font-family:"Cambria Math",Georgia,serif;
 font-size:16px;margin:8px 0;display:inline-block}}
.eq-inline{{background:#f0f2f5;border-radius:4px;padding:1px 6px;font-family:Georgia,serif}}
figure{{margin:14px 0;text-align:center}}
figure img,.svgwrap{{border:1px solid #e2e6ea;border-radius:8px;background:#fff;padding:6px;display:inline-block}}
.svgwrap svg{{max-width:100%;height:auto;display:block}}
figcaption{{color:var(--mut);font-size:12.5px;margin-top:6px}}
.figrow{{display:flex;flex-wrap:wrap;gap:14px;justify-content:center}}
.figrow figure{{flex:1 1 460px;margin:8px 0}}
.tbl{{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}}
.tbl th{{background:var(--blue);color:#fff;padding:9px;text-align:left}}
.tbl td{{border:1px solid #e2e6ea;padding:9px;vertical-align:top}}
.tbl td.good{{color:var(--accent)}} .tbl td.bad{{color:var(--red)}}
.missing{{color:var(--red);padding:20px;border:1px dashed var(--red)}}
footer{{max-width:980px;margin:0 auto;padding:18px 22px 50px;color:var(--mut);font-size:12.5px}}
</style></head>
<body>
<header class="hero">
<h1>量子光驱动双层 CrI₃ 反铁磁高次谐波 · 阶段工作报告</h1>
<p>理论框架 · SBE/HHG 求解器 · AFM 选择定则 · 量子几何/自旋 · BSV 确定性求积 · Tier-2 量子签名路线 — 2026-06</p>
</header>
<nav>{nav}</nav>
<main>
<section><p class="lead">建立并验证了一套双层 CrI₃-AFM 的实时 SBE/HHG 求解器,完成从经典光到量子光(BSV)的完整计算链;
得到 AFM 选择定则闭环、量子几何/自旋可观测量、BSV 高次谐波增强谱,并诚实界定其物理层级与 Tier-2 路线。</p></section>
{body}
</main>
<footer>图:左 6 张为经典验证(共 22 张归档中精选);右 6 张为 report/BSV 结果。本报告由计算记录自动汇编,
数据可追溯至 docs/ 各专题文档。</footer>
</body></html>"""

OUT.write_text(DOC, encoding="utf-8")
print("saved:", OUT)
print("size KB:", round(len(DOC.encode("utf-8")) / 1024, 1))
print("sections:", len(SECTIONS))

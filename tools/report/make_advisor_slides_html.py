"""Build a self-contained HTML slide deck for the advisor report.

The deck is deliberately generated from one small Python file so the scientific
wording stays versioned with the solver repository. SVG figures are embedded
inline and PNG figures are base64-embedded; no external JS/CSS/assets are needed.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path


REPO = Path(r"C:/Users/26507/Documents/New_SBEs/Quantum-light")
DOCS = REPO / "docs"
FIG = DOCS / "figures"
CV = FIG / "classical_validation"
RP = FIG / "report"
BQ = FIG / "bsv_quadrature"
OUT = DOCS / "量子光CrI3_导师汇报_slides_20260615.html"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def fig(path: Path, caption: str = "", cls: str = "") -> str:
    if not path.exists():
        return f'<div class="missing">Missing figure:<br>{esc(str(path))}</div>'
    if path.suffix.lower() == ".svg":
        svg = path.read_text(encoding="utf-8", errors="replace")
        start = svg.find("<svg")
        body = svg[start:] if start >= 0 else esc(svg)
        media = f'<div class="svgbox {cls}">{body}</div>'
    else:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        media = f'<img class="{cls}" src="data:image/png;base64,{data}" alt="{esc(caption)}"/>'
    cap = f'<div class="caption">{esc(caption)}</div>' if caption else ""
    return f'<figure>{media}{cap}</figure>'


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def callout(text: str, cls: str = "") -> str:
    return f'<div class="callout {cls}">{text}</div>'


def code(text: str) -> str:
    return f"<pre>{esc(text.strip())}</pre>"


def kv(rows: list[tuple[str, str]]) -> str:
    return "<div class='kv'>" + "".join(
        f"<div class='k'>{esc(k)}</div><div class='v'>{v}</div>" for k, v in rows
    ) + "</div>"


def slide(title: str, body: str, kicker: str = "") -> str:
    k = f'<div class="kicker">{esc(kicker)}</div>' if kicker else ""
    return f"""
<section class="slide">
  {k}
  <h1>{esc(title)}</h1>
  {body}
</section>
"""


slides: list[str] = []

slides.append(slide(
    "量子光驱动双层 CrI3 反铁磁高次谐波",
    """
    <div class="titleblock">
      <p class="subtitle">lg_cov SBE 求解器验证 · AFM Néel 选择定则 · BSV 确定性求积 · Tier 2 光子统计路线</p>
      <p>阶段汇报 · 2026-06-15</p>
    </div>
    <div class="source">核心定位：经典 CrI3 HHG 已有文献；本工作推进的是量子光驱动、严格数值闭环和输出光量子态。</div>
    """,
))

slides.append(slide(
    "科学问题：磁序如何进入谐波光？",
    f"""
    <div class="two">
      <div>
        {bullets([
            "<b>材料：</b>双层 CrI3，层内铁磁、层间反铁磁，强 SOC，低能光学响应对磁序敏感。",
            "<b>对称性：</b>Néel 序使 P 和 T 分别破缺，但 PT 近似守恒；偶次谐波可携带 AFM 符号信息。",
            "<b>实验锚点：</b>Sun et al. 2019 已观察到双层 CrI3 的巨大非互易磁性 SHG。",
            "<b>我的问题：</b>把驱动光换成 BSV 后，AFM 信息如何进入平均 HHG 谱，以及谐波光子的统计关联？",
        ])}
      </div>
      <div class="bigquestion">从“谱有没有偶次峰”<br>走向<br>“输出光是什么量子态”</div>
    </div>
    """,
    "研究对象和问题",
))

slides.append(slide(
    "文献定位：不是重复 Liu 2022",
    f"""
    <p>Liu et al., EPL 140, 26005 (2022) 已经研究双层 CrI3 的经典 HHG；2024 年又有 CrX3 波长依赖的跟进工作。因此报告里不能说“CrI3 HHG 为空白”，而要说清楚我们的增量。</p>
    <table>
      <tr><th>维度</th><th>Liu et al. 2022</th><th>本工作</th></tr>
      <tr><td>驱动</td><td>经典激光</td><td>BSV 强度求积 + 后续输出光统计</td></tr>
      <tr><td>方法</td><td>WIEN2K + 速度规范 p·A；文中未显式引入 T2</td><td>Wannier 112 带 + lg_cov 长度规范 + 显式 T2</td></tr>
      <tr><td>光场</td><td>ℏω=1.0 eV，τ=60 fs，A0=0.03 Vfs/Å</td><td>3200 nm = 0.387 eV，4 cycles，深亚带隙多光子区</td></tr>
      <tr><td>选择定则</td><td>磁点群给出 AB/AA1/3 与 AFM-z/x/y 规则</td><td>严格 P 共轭 -N TB，数值验证 Jn(-N)=(-1)<sup>n+1</sup>Jn(N)</td></tr>
      <tr><td>机制</td><td>垂直偶次与 Berry 反常电流有关</td><td>当前 PT 构型 Berry 通道受压制；偶次主要指向带间极化，仍需 j_anom(t) 显式拆分</td></tr>
    </table>
    {callout("一句话：Liu 给出经典 HHG 的对称性地图；本工作把路线推进到量子光驱动、可验证求解器和输出光子统计。")}
    """,
    "必须先摆正相对已有工作的增量",
))

slides.append(slide(
    "计算链路",
    f"""
    <div class="flow">
      <div>QE<br><span>SOC+U, AFM</span></div>
      <div>Wannier90<br><span>112-band d+p spinor</span></div>
      <div>lg_cov SBE<br><span>协变长度规范</span></div>
      <div>HHG / BSV / observables<br><span>J(t), spectrum, geometry, spin, occupation</span></div>
    </div>
    {callout("主线参数：λ=3200 nm, I=2.0e11 W/cm², ncyc=4, gauge=lg_cov, main grid=40×40。经典收敛图多用 T2=0.5 cycle；BSV 求积和可观测量导出用 T2=1 cycle。")}
    """,
    "从电子结构到实时响应",
))

slides.append(slide(
    "为什么 112 带是母模型",
    f"""
    <div class="two leftwide">
      <div>
        <p>这页的图不是在证明“某个窗口最好”，而是在说明不同带窗到底覆盖了费米能上下多少能量。</p>
        {bullets([
            "4 Cr × 5d + 12 I × 3p = 56 orbitals；SOC spinor ×2 = <b>112 bands</b>。",
            "这不是任意取大的窗口，而是双层 CrI3 的 d+p 旋量自然基。",
            "图中每一条横线是一种 <b>band window</b>：左端是保留到多深的价带，右端是保留到多高的导带，单位是 eV，相对费米能 Ef。",
            "full112 覆盖约 <b>-5.01 eV 到 +3.14 eV</b>；许多对称小窗口只覆盖 -1 到 -2 eV 的价带，会丢掉深价带对偶极/Berry 联络矩阵元的贡献。",
            "当前策略：用 full112 作基准；经济配置只能采用“全价带 + 截高导带”，不能随便做对称小窗口。",
        ])}
        {callout("如果用 nb1-104，它保留 full112 的全部价带深度，只少最高 8 条导带；这和图中那些 70-99、65-104 的对称窗口不是一回事。")}
      </div>
      {fig(CV / "fig04_band_window_energy_coverage.svg", "band-window energy coverage relative to Ef", "smallplot")}
    </div>
    """,
    "带数不是随意调参",
))

slides.append(slide(
    "经典验证 I：速度规范截断失败",
    f"""
    <div class="two leftwide">
      <div>
        <p>图的横坐标 window/case 是速度规范里选取的对称带窗大小：20、30、40、50、60 表示围绕带隙附近保留对应数量的 active bands；112 是完整母模型。</p>
        <p>纵坐标是各阶谐波强度，取 log 标度。每条彩色曲线对应 H1、H3、H5、H7、H9。</p>
        {code("H1 强度示例：\n20 bands -> 48\n40 bands -> 110\n60 bands -> 168\nfull112  -> 0.082")}
        <p>这些数值不是误差百分比，而是同一个 H1 峰的绝对谱强度。截断后 H1 从 0.08 跳到几十甚至一百多，说明低阶响应被虚假背景电流污染。</p>
        {callout("结论：Peierls VG 可作小网格全带参考，但不能用截断 VG 做生产。")}
      </div>
      {fig(CV / "fig01_peierls_vg_symmetric_truncation_failure.svg", "Peierls VG symmetric-window artifacts", "smallplot")}
    </div>
    """,
    "排除错误路线",
))

slides.append(slide(
    "经典验证 II：lg_cov + 全价带截导带",
    f"""
    <div class="two">
      <div>
        {bullets([
            "lg_cov 的截断误差是虚跃迁通道不完备，不再是背景电流灾难。",
            "<b>可用截断：</b>nb_start=1, nb_end=104，保留全部 84 价带，只截掉最高 8 导带。",
            "图中绿色 nb1-104 与黑色 full112 在主峰结构上几乎重合；H1-H11 的整数峰强度在几个百分点内。",
            "H12+ 的绝对强度已经很弱，且开始对最高导带敏感；如果论文主打高阶尾部，需要用 full112 复查。",
        ])}
        {callout("这里的核心不是“104 带处处等于 112 带”，而是：当前经典基准和 BSV pilot 关注的 H1-H11 可由 nb1-104 稳定描述。")}
      </div>
      {fig(CV / "fig13_hhg_spectrum_lgcov_fullvalence_T2_0p5cycle.svg", "lg_cov nb1-104 vs full112, T2=0.5 cycle", "smallplot")}
    </div>
    """,
    "经济配置的边界",
))

slides.append(slide(
    "经典验证 III：谐波阶数要换成能量看",
    f"""
    <div class="two">
      <div>
        <p>Liu 2022 用的是 ℏω=1.0 eV，所以 H10 已经是 10 eV 左右；我们用 3200 nm，ℏω≈0.387 eV，所以 H11 只有约 4.26 eV。</p>
        {kv([
            ("Liu 2022", "1.0 eV × H10 ≈ 10 eV，近/高于带隙驱动"),
            ("本工作", "0.387 eV × H11 ≈ 4.26 eV，深亚带隙多光子驱动"),
            ("带隙尺度", "CrI3 带隙约 0.9 eV，H3 已跨过带隙能量"),
            ("结论", "不能只比较“第几阶”；要比较谐波能量和谱强度地板"),
        ])}
      </div>
      <div>
        {fig(CV / "fig14_hhg_ratio_lgcov_nb104_full112_T2_0p5cycle.svg", "H1-H11 ratio: nb1-104 / full112", "smallplot")}
      </div>
    </div>
    {callout("因此 nb1-104 的合理说法是：H1-H11 的主物理稳定；H12+ 若因 BSV 高强度尾而被重新抬升，就必须用 full112 或局部 full112 节点做交叉检查。")}
    """,
    "为什么先看 H1-H11 是合理的",
))

slides.append(slide(
    "经典验证 IV：k 网格",
    f"""
    <div class="figrow">
      {fig(CV / "fig15_hhg_spectrum_lgcov_full112_k40_k60_T2_0p5cycle.svg", "full112 k40 vs k60 spectra", "shortplot")}
      {fig(CV / "fig16_hhg_ratio_lgcov_full112_k60_k40_T2_0p5cycle.svg", "k60/k40 ratio", "shortplot")}
    </div>
    {callout("读图方式：左图看谱形是否重合；右图看整数谐波峰的 k60/k40 比值是否接近 1。当前 H1-H15 变化很小，40×40 可作为主网格；极高阶或最强 BSV 节点再做局部 k60。")}
    """,
    "40×40 的说服力",
))

slides.append(slide(
    "经典验证 V：退相干参数怎么取",
    f"""
    <div class="two">
      <div>
        {bullets([
            "测试过：T2=0.5 fs、1 fs、2 fs、5 fs、0.5 optical cycle、no dephasing。",
            "3200 nm 下 0.5 optical cycle ≈ 5.337 fs，1 optical cycle ≈ 10.674 fs。",
            "经典收敛图主要用 T2=0.5 cycle，对齐旧求解器输入语义。",
            "BSV quadrature 和可观测量导出用 T2=1 cycle，使峰形更清楚。",
            "汇报原则：每张谱图都标 T2，不把 0.5 fs 和 0.5 cycle 混在一起。",
        ])}
      </div>
      {fig(CV / "fig17_hhg_spectrum_lgcov_full112_T2_scan.svg", "full112 T2 scan", "smallplot")}
    </div>
    """,
    "只讲参数和用途，不把它讲成新物理",
))

slides.append(slide(
    "经典验证 VI：初始电流和弱场标度",
    f"""
    <div class="two leftwide">
      <div>
        <p>这页的意义是检查“初态是否干净”。绝缘体基态所有占据价带填满时，整个 BZ 积分后的基态电流必须为零：</p>
        <div class="formula small">J(0) = -Σ<sub>k,n∈occ</sub> v<sub>nn</sub>(k) ≈ 0</div>
        {bullets([
            "图中 nb1-94、nb1-104、full112 都保留全部价带，|J(0)| 在 10^-18 附近，是机器零。",
            "nb10-104、nb20-104 截掉深价带后 |J(0)| 明显升高，说明初态已经被破坏。",
            "这项检查能排除“低阶谐波其实来自伪背景电流”的风险。",
            "弱场 scaling 则用于检查非线性阶次：H1/H2/H3 在降场时向 1/2/3 阶电流响应靠近。",
        ])}
      </div>
      {fig(CV / "fig22_initial_current_lgcov_band_economy_T2_0p5cycle.svg", "initial current check", "smallplot")}
    </div>
    """,
    "初态是否真的没有伪电流",
))

slides.append(slide(
    "AFM 选择定则：真正的闭环",
    f"""
    <div class="two">
      <div>
        <p>理论基底是反演操作 P 把 +N 映到 -N，同时电流是极矢量、电场在 n 阶响应中带来 (-1)^n 因子，因此理想情况下：</p>
        <div class="formula small">J<sub>n</sub>(-N)=(-1)<sup>n+1</sup>J<sub>n</sub>(N)</div>
        {bullets([
            "定义 J_evenN=(J_N+J_-N)/2，J_oddN=(J_N-J_-N)/2。",
            "奇次谐波应该主要在 evenN 通道；偶次谐波应该主要在 oddN 通道。",
            "leakage_ratio = 禁戒通道强度 / 允许通道强度；越小，选择定则越干净。",
            "两次独立 DFT/Wannier 的 -N 不严格配对：SCF 差 2.18 mRy，本征值差 50-110 meV；因此必须构造严格 P 共轭 -N TB。",
            "P 共轭后 H2 leakage 从 13.1 降到 0.32，并从错误 evenN 通道回到 oddN 通道。",
        ])}
      </div>
      {fig(RP / "figB_selrule.png", "selection-rule leakage after P-pairing", "smallplot")}
    </div>
    {callout("验证意义：它同时检查了 TB 的 AFM 配对、k 网格对称采样、电流算符和 lg_cov 传播。此前 H2 异常来自 DFT/Wannier 配对误差，不是 SBE 求解器错误。")}
    """,
    "把对称性从口头规则变成数值检查",
))

slides.append(slide(
    "与 Liu 的机制差异：Berry 还是极化？",
    f"""
    <div class="two">
      <div>
        {bullets([
            "Liu 2022 指出某些垂直偶次谐波接近 Berry 曲率反常电流贡献。",
            "我们的当前构型做了严格 P 共轭检查，PT 对称下占据 Berry 曲率通道应被强烈压制。",
            "现有 intra/inter 分解显示偶次更像带间极化通道，但 j_inter 仍包含 j_pol + j_anom。",
            "<b>不能过度宣称：</b>在显式输出 j_anom(t) 前，只能说“证据指向极化主导”，不能说已经完全排除 Berry 通道。",
        ])}
      </div>
      <div class="formula">
        j<sub>inter</sub>(t) = j<sub>pol</sub>(t) + j<sub>anom</sub>(t)<br><br>
        下一步：单独导出 j<sub>anom</sub>(t)
      </div>
    </div>
    """,
    "一个值得保留的 TODO，而不是缺陷",
))

slides.append(slide(
    "HHG 谱从哪里来",
    f"""
    <div class="two">
      <div>
        <p>求解器首先得到时间域电流 J(t)，然后加窗做傅里叶变换。图上的“曲线”不是手动画出的离散峰，而是连续频率网格上的谱：</p>
        <div class="formula small">S(Ω) = | ∫ dt&nbsp; W(t) J(t) e<sup>iΩt</sup> |²</div>
        {kv([
            ("横坐标", "harmonic order = Ω / ω0"),
            ("纵坐标", "|J(Ω)|²，log 标度"),
            ("峰宽来源", "有限脉冲长度 + T2 退相干 + 加窗函数"),
            ("为什么比 Liu 更平滑", "窗口函数、T2 和基频不同；不应把背景平滑误读成没有谐波"),
        ])}
      </div>
      <div class="diagram spectrum">
        <div class="sine"></div>
        <div class="arrowtext">J(t) → FFT → S(Ω)</div>
        <div class="peaks">
          <span>H1</span><span>H3</span><span>H5</span><span>H7</span><span>H9</span>
        </div>
      </div>
    </div>
    {callout("读谱原则：整数谐波峰用于定量；峰间连续背景主要受脉冲、窗口和退相干影响。报告时不要把连续曲线说成“噪声少所以不真实”。")}
    """,
    "先定义谱图再讨论物理",
))

slides.append(slide(
    "BSV MC pilot：为什么不能硬堆样本",
    f"""
    <div class="two">
      <div>
        <p>这里的 MC pilot 指“小样本蒙特卡洛试跑”：从 BSV 的强度分布里随机抽样 51 条经典轨迹，先看平均谱是否有信号、误差有多大。</p>
        <p>相位平均后的 BSV 强度可以写成指数分布。令 x=I/I0，其中 I0 是平均强度，则：</p>
        <div class="formula small">p(x)=e<sup>-x</sup>, &nbsp; x≥0</div>
        <p>高次 HHG 对强度非常敏感，所以指数分布的高强度长尾会主导谱尾；随机 MC 很容易抽不准这个长尾。</p>
        {kv([
            ("51 样本 MC", "高阶相对误差约 18%-44%"),
            ("硬堆到 10%", "估计约 970 样本；3 节点并行约 24 天"),
            ("决策", "MC 只做 pilot 和交叉验证；生产换确定性求积"),
        ])}
      </div>
      <div class="diagram tail">
        <div class="tailcurve"></div>
        <p>长尾很少出现，<br>但一旦出现就强烈贡献高阶谐波。</p>
      </div>
    </div>
    """,
    "R1 的计算瓶颈",
))

slides.append(slide(
    "BSV 确定性强度求积",
    f"""
    <div class="two">
      <div>
        <p>因为当前 R1 只需要 BSV 的强度平均，随机相位积分可化成一维强度积分：</p>
        <div class="formula small">S<sub>BSV</sub>(ω)=∫<sub>0</sub><sup>∞</sup> e<sup>-x</sup>S(ω; I0x) dx</div>
        {bullets([
            "做法：选 16 个强度节点 x_i，每个节点跑一次普通经典 SBE，再按权重 w_i 相加。",
            "节点验证：复现指数矩 <x^p>=p! 到 p≤7 的 0.3% 以内。",
            "与 51 样本 MC 在约 2σ 内一致，但没有 MC 噪声。",
            "成本：16 个确定性节点替代约上千个随机样本；这就是 BSV 生产可行的关键。",
            "配置：lg_cov / 40×40 / nb1-104 / T2_cycles=1.0 / ncyc=4 / φ_CEP=90°。",
        ])}
      </div>
      {fig(RP / "figC_quad_vs_mc.png", "quadrature vs MC pilot", "smallplot")}
    </div>
    """,
    "16 个 run 抵掉上千个随机样本",
))

slides.append(slide(
    "BSV 平均谱",
    f"""
    <div class="two">
      <div>
        <p>图中灰色是同一平均强度下的经典光基准，蓝色是 BSV 强度平均。纵轴是 HHG 谱强度，log 标度；横轴是谐波阶数。</p>
        {bullets([
            "低阶 H1 只增强约 1.2×，说明线性响应主要看平均强度。",
            "靠近 cutoff 的 H11/H13 增强到 60×/363×，说明高强度尾部在抬升谱尾。",
            "H4≈0.7× 是重要提醒：偶次磁性通道不是简单随阶数单调增强，可能受 AFM 选择定则、相位和饱和共同影响。",
            "这仍是 R1：强度涨落平均，可由经典强度分布复现；不能称为严格量子优势。",
        ])}
      </div>
      {fig(BQ / "fig1_spectra.png", "BSV quadrature spectrum vs classical baseline", "smallplot")}
    </div>
    """,
    "T2=1 cycle / nb1-104 / 40×40",
))

slides.append(slide(
    "BSV 增强曲线",
    f"""
    <div class="two">
      {fig(BQ / "fig2_enhancement.png", "enhancement vs harmonic order", "smallplot")}
      <div>
        <p>读图方式：每个点是 BSV 平均谱与经典基准在整数谐波峰上的比值；虚线 n! 是微扰极限的参照，不是拟合目标。</p>
        <div class="formula small">若 S_n(I) ∝ I<sup>p</sup>，则 enhancement = ⟨I<sup>p</sup>⟩/⟨I⟩<sup>p</sup> = Γ(p+1)</div>
        <p>完全微扰的 n 光子过程常近似 p≈n，因此给出 n! 量级。实际数据低于 n!，说明体系进入非微扰平台/饱和区。</p>
        {callout("物理含义：增强随阶数上升，但不是无限按 n! 爆炸；这反而是固体 HHG 平台物理的表现。")}
      </div>
    </div>
    """,
    "别把 n! 当成必须达到的目标",
))

slides.append(slide(
    "单强度标度：为什么 cutoff 被抬高",
    f"""
    <div class="two">
      <div>
        <p>理论框架是把每一阶谐波看成强度的函数 S_n(I)，再对 BSV 的指数强度分布做平均。</p>
        <div class="formula small">S̄_n = ∫ e<sup>-x</sup>S_n(I0x)dx</div>
        {bullets([
            "低阶：S_n(I) 在弱场端近似幂律，平均后只给出温和增强。",
            "高阶：经典基准接近 cutoff，少数高 I 节点让 cutoff 外移，因此增强很大。",
            "强场端：S_n(I) 开始饱和，所以增强又低于 n! 微扰上限。",
        ])}
        {callout("时间成本风险也在这里：高阶由尾部分布控制，随机 MC 必须抽到足够多尾部样本；确定性求积则主动覆盖尾部。")}
      </div>
      {fig(BQ / "fig3_scaling.png", "single-intensity scaling of selected harmonics", "smallplot")}
    </div>
    """,
    "R1 的机制图像",
))

slides.append(slide(
    "Tier 2 ①：换一个问题",
    f"""
    <p>到这里，我们算的是平均谱。平均谱增强能被经典强度涨落复现，所以不能把它叫“严格量子优势”。下一步要问另一个问题：</p>
    <div class="bigquestion">输出谐波光本身<br>是什么量子态？</div>
    {bullets([
        "它是否被压缩？",
        "不同谐波模式之间是否纠缠？",
        "光子数统计是超泊松还是亚泊松？",
        "这些关联里有没有只随 AFM 磁序打开的通道？",
    ])}
    {callout("这不和“平均谱无量子优势”的观点矛盾：那讨论输入统计对平均谱的影响；这里讨论输出光场的量子态。")}
    """,
    "不要把 R1 说成 R2",
))

slides.append(slide(
    "Tier 2 ②：物理链条",
    f"""
    <div class="flow thin">
      <div>⟨J(t)⟩<br><span>一阶</span></div>
      <div>βq<br><span>谐波相干振幅</span></div>
      <div>C(t,t')<br><span>二阶电流关联</span></div>
      <div>Σout<br><span>输出光协方差</span></div>
    </div>
    <p>材料里的含时电流是谐波场的源。一阶均值给出现在已经算到的 HHG 谱；二阶涨落关联决定输出光的压缩和纠缠。</p>
    <div class="formula">C<sub>ab</sub>(t,t') = ⟨ΔJ<sub>a</sub>(t) ΔJ<sub>b</sub>(t')⟩</div>
    {callout("一句话：从算一个平均电流，升级为算双时电流关联。")}
    """,
    "从 SBE 到量子光学",
))

slides.append(slide(
    "Tier 2 ③：能算出什么",
    f"""
    <div class="cards">
      <div><b>压缩谱</b><br>某个正交分量噪声低于真空</div>
      <div><b>谐波纠缠</b><br>对数负性 / Duan-Simon 判据</div>
      <div><b>光子统计</b><br>g²(q,q')，聚束或反聚束</div>
      <div><b>估计精度</b><br>Gaussian QFI 与磁序敏感度</div>
    </div>
    <p>CrI3 的特色不是“也能算这些量”，而是可以把关联按 Néel 奇偶拆开。若 C_oo 通道只在磁有序态打开，它就是谐波光子的磁性指纹。</p>
    <div class="formula">C = C<sub>ee</sub> + |N|² C<sub>oo</sub> + ...</div>
    """,
    "把磁序推进到光子关联层",
))

slides.append(slide(
    "Tier 2 ④：怎么做、风险和 R3",
    f"""
    <div class="two">
      <div>
        {bullets([
            "<b>实现：</b>用量子回归定理/辅助算符传播复用现有 lg_cov 传播器，不需要推倒重写求解器。",
            "<b>风险：</b>T2≈10 fs 会压制关联积累，必须先做数量级估计。",
            "<b>可发表路径：</b>若压缩/纠缠可分辨，R2 就是独立物理结果；若太弱，也能作为原理与约束分析。",
            "<b>R3 长线：</b>BSV 偶光子配对和压缩角 θ 依赖是经典强度采样没有的入口，但需要更严格的不可复现性判据。",
        ])}
      </div>
      {kv([
          ("R1", "强度平均：当前结果，干净但可经典复现"),
          ("R2", "输出光非经典性：压缩/纠缠/C_oo 磁性指纹"),
          ("R3", "严格量子优势：经典不可复现，长线问题"),
      ])}
    </div>
    """,
    "给导师看的清晰路线",
))

slides.append(slide(
    "利弊权衡",
    """
    <table>
      <tr><th>路线</th><th>价值</th><th>风险</th><th>建议</th></tr>
      <tr><td>R1 论文</td><td>量子光驱动双层 CrI3 HHG；求解器和选择定则闭环完整</td><td>不是严格量子优势</td><td>可以先成文，但措辞要诚实</td></tr>
      <tr><td>j_anom 拆分</td><td>回应 Liu 2022 的 Berry 机制；坐实本构型极化主导</td><td>需要增加一项输出/后处理</td><td>优先补，成本不高、物理收益大</td></tr>
      <tr><td>R2 Tier 2</td><td>输出光压缩/纠缠/磁性指纹，增量最大</td><td>T2 可能压低信号</td><td>并行推进，先估数量级</td></tr>
      <tr><td>R3</td><td>严格量子优势</td><td>理论证明难，需要凸优化/矩问题</td><td>作为长线，不影响当前汇报</td></tr>
    </table>
    """,
    "不要把所有野心塞进第一篇",
))

slides.append(slide(
    "总结和下一步",
    bullets([
        "经典光 lg_cov SBE 已完成规范、带窗、k 网格、T2、弱场和初始电流验证。",
        "AFM 选择定则异常已用严格 P 共轭 -N TB 闭合，排除了求解器 bug。",
        "BSV 从 MC pilot 升级为 16 节点确定性求积，得到无噪声 R1 平均谱。",
        "相对 Liu 2022 的立足点：量子光驱动 + lg_cov 数值闭环 + 输出光子统计路线。",
        "最近的具体任务：补 j_anom(t) 拆分；把 R1 写成一篇诚实的量子光 HHG 论文；并行启动 Tier 2 双时关联数量级估计。",
    ]) + callout("汇报时最后一句：我现在已经不是在调代码，而是在从 R1 平均谱推进到 R2 输出光非经典性。"),
    "当前阶段的判断",
))


CSS = r"""
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #0f1216;
  color: #eef2f5;
  font-family: "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", Arial, sans-serif;
}
.deck { width: 100vw; height: 100vh; overflow: hidden; }
.slide {
  display: none;
  width: 100vw;
  height: 100vh;
  padding: 5.2vh 6vw 6vh;
  position: relative;
  background:
    linear-gradient(90deg, rgba(44, 88, 112, 0.16), transparent 45%),
    linear-gradient(180deg, #171b21 0%, #101318 100%);
}
.slide.active { display: block; }
.kicker {
  color: #89c2d9;
  font-size: 1.0rem;
  letter-spacing: 0.02em;
  margin-bottom: 1.0vh;
}
h1 {
  margin: 0 0 3vh;
  font-size: clamp(2.0rem, 3.2vw, 3.4rem);
  line-height: 1.08;
  font-weight: 760;
}
p, li, td, th {
  font-size: clamp(1.0rem, 1.38vw, 1.36rem);
  line-height: 1.52;
}
ul { margin: 0; padding-left: 1.25em; }
li { margin: 0.45em 0; }
.subtitle { font-size: clamp(1.25rem, 2.0vw, 1.9rem); color: #dcecf3; }
.titleblock {
  margin-top: 20vh;
  max-width: 1100px;
}
.source {
  position: absolute;
  left: 6vw;
  bottom: 7vh;
  color: #adc2ca;
  font-size: 1.05rem;
}
.two {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 3vw;
  align-items: start;
}
.two.leftwide { grid-template-columns: minmax(0, 1.16fr) minmax(0, 0.84fr); }
.widefig figure { width: 88vw; max-width: 1320px; margin: 0 auto; }
.figrow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 2vw;
  align-items: start;
}
figure { margin: 0; }
.svgbox {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  background: transparent;
  padding: 0;
  overflow: visible;
}
.svgbox svg,
img {
  display: block;
  width: 100%;
  max-height: 60vh;
  object-fit: contain;
  background: #ffffff;
  border-radius: 8px;
  padding: 8px;
}
.svgbox.smallplot { justify-content: flex-start; }
.svgbox.smallplot svg,
img.smallplot {
  width: 92%;
  max-height: 52vh;
}
.svgbox.shortplot svg,
img.shortplot {
  max-height: 43vh;
}
.figrow .caption { font-size: 0.82rem; }
.figrow figure { min-width: 0; }
.figrow .svgbox svg,
.figrow img { max-height: 42vh; }
.widefig .svgbox svg { max-height: 62vh; }
.two .svgbox svg,
.two img { max-height: 54vh; }
.two.leftwide .svgbox svg,
.two.leftwide img { max-height: 50vh; }
.two.leftwide .caption { max-width: 92%; }
.formula.small {
  font-size: clamp(1.05rem, 1.45vw, 1.45rem);
  padding: 1rem;
  margin: 1.4vh 0;
  text-align: left;
}
.caption {
  margin-top: 0.55rem;
  color: #b5c5cc;
  font-size: 0.92rem;
}
.callout {
  margin-top: 2.2vh;
  padding: 1.0rem 1.15rem;
  border-left: 5px solid #f2b84b;
  background: rgba(242, 184, 75, 0.12);
  border-radius: 6px;
  font-size: clamp(1.0rem, 1.35vw, 1.35rem);
  line-height: 1.5;
}
pre {
  white-space: pre-wrap;
  background: #0a0d10;
  color: #cfe8d5;
  padding: 1rem;
  border-radius: 7px;
  font-size: clamp(0.95rem, 1.25vw, 1.2rem);
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1.5vh;
  background: rgba(255,255,255,0.035);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  border-bottom: 1px solid rgba(255,255,255,0.13);
  padding: 0.65rem 0.75rem;
  vertical-align: top;
}
th { color: #9bd1e5; text-align: left; background: rgba(137,194,217,0.08); }
.flow {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.2vw;
  margin: 10vh 0 6vh;
}
.flow.thin { margin-top: 5vh; }
.flow div, .cards div, .kv .k, .kv .v {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.13);
  border-radius: 8px;
}
.flow div {
  min-height: 15vh;
  padding: 1.2rem;
  font-size: clamp(1.1rem, 1.55vw, 1.55rem);
  font-weight: 720;
}
.flow span { display: block; margin-top: 0.6rem; color: #bac7ce; font-size: 0.95rem; font-weight: 400; }
.bigquestion {
  min-height: 35vh;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #fff3c4;
  font-size: clamp(2.0rem, 3.2vw, 3.4rem);
  line-height: 1.25;
  border: 1px solid rgba(255,243,196,0.25);
  background: rgba(255,243,196,0.08);
  border-radius: 12px;
  padding: 2rem;
}
.formula {
  color: #eaf6ff;
  background: rgba(137,194,217,0.10);
  border: 1px solid rgba(137,194,217,0.22);
  border-radius: 10px;
  padding: 2rem;
  font-size: clamp(1.6rem, 2.4vw, 2.4rem);
  line-height: 1.4;
  text-align: center;
}
.cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.2vw;
  margin: 4vh 0;
}
.cards div { padding: 1.1rem; min-height: 18vh; }
.kv {
  display: grid;
  grid-template-columns: 0.75fr 1.25fr;
  gap: 0.6rem;
}
.kv .k, .kv .v { padding: 0.8rem; }
.kv .k { color: #9bd1e5; font-weight: 720; }
.diagram {
  min-height: 36vh;
  padding: 2rem;
  border: 1px solid rgba(137,194,217,0.25);
  background: rgba(137,194,217,0.08);
  border-radius: 10px;
}
.axis {
  position: relative;
  height: 8vh;
  margin-top: 5vh;
}
.axis span {
  position: absolute;
  transform: translateX(-50%);
  white-space: nowrap;
  color: #eaf6ff;
  font-size: clamp(0.8rem, 1.1vw, 1.05rem);
}
.barline {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #6aaed6, #f2b84b);
  margin: 0 2%;
}
.ticks {
  position: relative;
  height: 9vh;
  margin: 1.2vh 2% 0;
}
.ticks b {
  position: absolute;
  transform: translateX(-50%);
  color: #fff3c4;
  font-size: clamp(0.82rem, 1.1vw, 1.05rem);
}
.tail {
  position: relative;
  overflow: hidden;
}
.tailcurve {
  width: 90%;
  height: 18vh;
  margin: 5vh auto 3vh;
  border-left: 3px solid #9bd1e5;
  border-bottom: 3px solid #9bd1e5;
  border-radius: 0 0 0 4px;
  background:
    radial-gradient(ellipse at 18% 20%, rgba(242,184,75,0.8), transparent 9%),
    radial-gradient(ellipse at 36% 34%, rgba(242,184,75,0.45), transparent 7%),
    radial-gradient(ellipse at 70% 70%, rgba(242,184,75,0.28), transparent 5%);
  clip-path: polygon(0 0, 12% 12%, 23% 25%, 35% 42%, 52% 63%, 76% 82%, 100% 92%, 100% 100%, 0 100%);
}
.tail p {
  text-align: center;
  color: #fff3c4;
}
.spectrum {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 2.2vh;
}
.sine {
  height: 15vh;
  border-radius: 8px;
  background:
    repeating-linear-gradient(90deg, transparent 0 7%, rgba(154,209,229,0.22) 7% 7.5%, transparent 7.5% 14%),
    linear-gradient(180deg, transparent 48%, rgba(242,184,75,0.9) 49%, rgba(242,184,75,0.9) 51%, transparent 52%);
  position: relative;
  overflow: hidden;
}
.sine::after {
  content: "";
  position: absolute;
  left: -5%;
  top: 18%;
  width: 110%;
  height: 64%;
  border-radius: 50%;
  border-top: 5px solid #9bd1e5;
  transform: skewX(-18deg);
}
.arrowtext {
  text-align: center;
  font-size: clamp(1.2rem, 1.8vw, 1.8rem);
  color: #fff3c4;
  font-weight: 720;
}
.peaks {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.8vw;
  align-items: end;
  height: 16vh;
}
.peaks span {
  display: flex;
  align-items: end;
  justify-content: center;
  color: #111;
  font-weight: 760;
  border-radius: 5px 5px 0 0;
  background: linear-gradient(180deg, #f2b84b, #ffffff);
  padding-bottom: 0.5rem;
}
.peaks span:nth-child(1) { height: 95%; }
.peaks span:nth-child(2) { height: 70%; }
.peaks span:nth-child(3) { height: 46%; }
.peaks span:nth-child(4) { height: 31%; }
.peaks span:nth-child(5) { height: 22%; }
.missing {
  padding: 2rem;
  background: #4b1d1d;
  color: #ffd5d5;
  border-radius: 8px;
}
.pager {
  position: fixed;
  right: 2vw;
  bottom: 2vh;
  color: #b7c4ca;
  font-size: 0.95rem;
}
.bar {
  position: fixed;
  left: 0;
  bottom: 0;
  height: 5px;
  background: #f2b84b;
  width: 0;
  transition: width 0.15s linear;
}
.hint {
  position: fixed;
  left: 2vw;
  bottom: 2vh;
  color: #7e8e96;
  font-size: 0.9rem;
}
@media print {
  body { background: white; }
  .slide { display: block !important; page-break-after: always; color: #111; background: white; }
  .pager, .bar, .hint { display: none; }
}
"""


def render() -> str:
    total = len(slides)
    rendered: list[str] = []
    for idx, raw in enumerate(slides, 1):
        cls = "slide active" if idx == 1 else "slide"
        raw = raw.replace('<section class="slide">', f'<section class="{cls}" data-page="{idx}">', 1)
        raw = raw.replace("</section>", f'<div class="pager">{idx} / {total}</div></section>', 1)
        rendered.append(raw)
    body = "\n".join(rendered)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>量子光 CrI3 导师汇报 slides 20260615</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="deck">{body}</div>
  <div class="hint">← / → 翻页，Home/End 跳转</div>
  <div class="bar" id="bar"></div>
  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    let i = 0;
    function show(n) {{
      i = Math.max(0, Math.min(slides.length - 1, n));
      slides.forEach((s, k) => s.classList.toggle('active', k === i));
      document.getElementById('bar').style.width = ((i + 1) / slides.length * 100) + '%';
    }}
    document.addEventListener('keydown', (e) => {{
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) show(i + 1);
      if (['ArrowLeft', 'PageUp'].includes(e.key)) show(i - 1);
      if (e.key === 'Home') show(0);
      if (e.key === 'End') show(slides.length - 1);
    }});
    show(0);
  </script>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(render(), encoding="utf-8")
    print(f"[written] {OUT}")
    print(f"[slides] {len(slides)}")


if __name__ == "__main__":
    main()

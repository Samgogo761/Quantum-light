import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "C:/Users/26507/Documents/New_SBEs/Quantum-light";
const OUT = path.join(ROOT, "docs", "量子光CrI3_阶段汇报_重做版_20260702.pptx");
const WORK = path.join(ROOT, "tmp", "presentations", "cri3_stage_report_20260702");
const QA = path.join(WORK, "qa");

const assets = {
  classical: path.join(ROOT, "docs", "figures", "report", "figA_classical_hhg.png"),
  selrule: path.join(ROOT, "docs", "figures", "report", "figB_selrule.png"),
  quad: path.join(ROOT, "docs", "figures", "report", "figC_quad_vs_mc.png"),
  bsvSpectra: path.join(ROOT, "docs", "figures", "bsv_quadrature", "fig1_spectra.png"),
  bsvEnh: path.join(ROOT, "docs", "figures", "bsv_quadrature", "fig2_enhancement.png"),
  bsvScale: path.join(ROOT, "docs", "figures", "bsv_quadrature", "fig3_scaling.png"),
};

const W = 1280;
const H = 720;
const page = { left: 70, top: 54, width: 1140, height: 602 };
const colors = {
  bg: "#f7f8fb",
  ink: "#172033",
  muted: "#5b6475",
  light: "#e6eaf1",
  line: "#cfd6e3",
  blue: "#2357a5",
  cyan: "#1e87a8",
  green: "#3b7d4d",
  red: "#b33b3b",
  amber: "#a56a1e",
  purple: "#6d54a3",
  white: "#ffffff",
};

async function img(pathName) {
  return new Uint8Array(await fs.readFile(pathName));
}

function addText(slide, text, x, y, w, h, opt = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opt.size ?? 24,
    bold: opt.bold ?? false,
    color: opt.color ?? colors.ink,
    alignment: opt.align ?? "left",
  };
  return shape;
}

function addBox(slide, x, y, w, h, opt = {}) {
  return slide.shapes.add({
    geometry: opt.geometry ?? "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill: opt.fill ?? colors.white,
    line: { style: "solid", fill: opt.line ?? colors.line, width: opt.lineWidth ?? 1 },
    borderRadius: opt.radius ?? "rounded-lg",
    shadow: opt.shadow ?? "none",
  });
}

function addPill(slide, text, x, y, w, color) {
  addBox(slide, x, y, w, 34, { fill: color, line: color, radius: "rounded-xl" });
  addText(slide, text, x + 14, y + 7, w - 28, 22, { size: 13, bold: true, color: colors.white, align: "center" });
}

function addHeader(slide, title, section = "") {
  slide.background.fill = colors.bg;
  if (section) addText(slide, section, page.left, 22, 520, 24, { size: 12, bold: true, color: colors.blue });
  addText(slide, title, page.left, page.top, 930, 58, { size: 34, bold: true, color: colors.ink });
  slide.shapes.add({
    geometry: "rect",
    position: { left: page.left, top: 116, width: page.width, height: 1 },
    fill: colors.line,
    line: { style: "solid", fill: colors.line, width: 0 },
  });
}

function addFooter(slide, n, note = "Bilayer AFM-CrI3 quantum-light HHG stage report") {
  addText(slide, note, page.left, 685, 740, 22, { size: 10, color: "#7b8494" });
  addText(slide, String(n).padStart(2, "0"), 1160, 685, 50, 22, { size: 11, color: "#7b8494", align: "right" });
}

function bulletText(items) {
  return items.map((x) => "• " + x).join("\n");
}

function addBullets(slide, items, x, y, w, h, opt = {}) {
  return addText(slide, bulletText(items), x, y, w, h, { size: opt.size ?? 18, color: opt.color ?? colors.ink });
}

async function addImage(slide, file, x, y, w, h, alt, fit = "contain") {
  slide.images.add({
    blob: await img(file),
    contentType: "image/png",
    alt,
    fit,
    position: { left: x, top: y, width: w, height: h },
  });
}

function addMetric(slide, value, label, x, y, w, color = colors.blue) {
  addBox(slide, x, y, w, 92, { fill: colors.white, line: "#d9dfeb", shadow: "shadow-sm" });
  addText(slide, value, x + 18, y + 14, w - 36, 34, { size: 26, bold: true, color });
  addText(slide, label, x + 18, y + 52, w - 36, 26, { size: 13, color: colors.muted });
}

function addFormula(slide, text, x, y, w, h, color = "#eef3ff") {
  addBox(slide, x, y, w, h, { fill: color, line: "#cbd8f0", radius: "rounded-lg" });
  addText(slide, text, x + 18, y + 14, w - 36, h - 28, { size: 18, color: colors.ink });
}

function addSectionSlide(prs, n, section, title, points, color) {
  const slide = prs.slides.add();
  slide.background.fill = color;
  addText(slide, section, page.left, 80, 900, 28, { size: 16, bold: true, color: "#eef6ff" });
  addText(slide, title, page.left, 150, 950, 120, { size: 46, bold: true, color: colors.white });
  addText(slide, points.join("\n"), page.left, 320, 950, 180, { size: 22, color: "#eef6ff" });
  addFooter(slide, n, "");
  return slide;
}

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(QA, { recursive: true });

  const prs = Presentation.create({ slideSize: { width: W, height: H } });

  // 1
  {
    const slide = prs.slides.add();
    slide.background.fill = "#101827";
    addText(slide, "双层 AFM-CrI3 中量子光驱动 HHG", 70, 74, 980, 74, { size: 44, bold: true, color: colors.white });
    addText(slide, "阶段汇报：经典光求解器验证、BSV 平均谱与下一步量子关联路线", 74, 160, 940, 42, { size: 22, color: "#dbe7ff" });
    addBox(slide, 74, 260, 1050, 210, { fill: "#172235", line: "#33425d", radius: "rounded-xl" });
    addText(slide, "核心说法", 104, 290, 260, 34, { size: 22, bold: true, color: "#8ecae6" });
    addText(slide, "现在已经有一条可复查的计算链：\nlg_cov 长度规范 SBE 通过经典光、带窗、k 网格、AFM 选择定则与外部 benchmark 检验；BSV 平均谱可由确定性强度求积高效得到。下一步不再只问谱强度，而是计算输出谐波光的双时电流关联、协方差和 QFI。", 104, 335, 980, 112, { size: 24, color: colors.white });
    addPill(slide, "QE + Wannier90 112-band", 78, 535, 250, colors.blue);
    addPill(slide, "lg_cov SBE", 354, 535, 150, colors.green);
    addPill(slide, "P-conjugate N test", 530, 535, 210, colors.purple);
    addPill(slide, "BSV quadrature", 766, 535, 190, colors.amber);
    addFooter(slide, 1, "Stage report rebuilt from 汇报.pdf, old slides, and six literature supplement pages");
  }

  // 2
  {
    const slide = prs.slides.add();
    addHeader(slide, "科学问题不是“CrI3 有没有 HHG”，而是磁序如何写进量子光输出", "1. 研究命题");
    addBullets(slide, [
      "双层 AFM-CrI3 中 P 与 T 各自破缺，PT 保持；Néel 矢量 N 为非线性响应提供可翻转的磁序标签。",
      "经典 HHG 已经可作为磁构型指纹；本工作把驱动从经典光推进到 BSV，并进一步关心出射光子的统计性质。",
      "平均谱只能回答强度问题；双时电流关联 Cconn(t,t') 才能回答压缩、纠缠、g2 与磁序传感的问题。",
    ], 78, 150, 580, 220);
    addFormula(slide, "J_e/o(t) = [J(N,t) ± J(-N,t)] / 2\nC_ab(t,t') = <ΔJ_a(t) ΔJ_b(t')>\nΣ -> squeezing / entanglement / QFI", 720, 152, 430, 166);
    addBox(slide, 720, 354, 430, 142, { fill: "#fff7ed", line: "#f0d0aa" });
    addText(slide, "汇报主线", 744, 378, 160, 28, { size: 18, bold: true, color: colors.amber });
    addText(slide, "先证明求解器和经典光基准可信；再展示 BSV 平均谱增强；最后说明如何进入输出光量子统计。", 744, 416, 380, 66, { size: 19, color: colors.ink });
    addFooter(slide, 2);
  }

  // 3
  {
    const slide = prs.slides.add();
    addHeader(slide, "文献定位要说准：Liu 已做经典 CrI3 HHG，我们做量子光与关联", "1. 研究命题");
    const rows = [
      ["维度", "Liu et al. 2022", "本工作"],
      ["驱动", "经典激光", "BSV 量子光 + 输出光统计"],
      ["规范", "速度规范 p·A", "协变长度规范 lg_cov + T2"],
      ["问题", "AFM 构型的 HHG 指纹", "N 宇称分辨的平均谱与双时关联"],
      ["验证", "对称性选择定则", "严格 P 共轭 -N + leakage 定量"],
    ];
    const x = 84, y = 150, col = [150, 360, 480], rh = 54;
    rows.forEach((r, i) => {
      let xx = x;
      r.forEach((c, j) => {
        addBox(slide, xx, y + i * rh, col[j], rh, { fill: i === 0 ? "#eaf0fb" : colors.white, line: "#d7deeb", radius: "rounded-none" });
        addText(slide, c, xx + 14, y + i * rh + 13, col[j] - 28, 24, { size: i === 0 ? 16 : 15, bold: i === 0, color: colors.ink });
        xx += col[j];
      });
    });
    addBox(slide, 84, 460, 990, 86, { fill: "#f0f7f5", line: "#c5ddd4" });
    addText(slide, "因此“空白点”要改成：双层 AFM-CrI3 的量子光驱动 HHG 与出射谐波光统计，而不是 CrI3 HHG 本身。", 112, 486, 934, 36, { size: 22, bold: true, color: colors.green });
    addFooter(slide, 3, "Sources used in report framing: Liu et al., EPL 140, 26005 (2022); Yue & Gaarde JOSA B 2022; Sun et al. Nature 2019");
  }

  // 4
  addSectionSlide(prs, 4, "2. 求解器与材料模型", "先把经典光问题算干净", [
    "核心方程：dρ/dt = -i[H0 + E·D,ρ] + E·Dkρ",
    "112-band Wannier 母模型保留 Cr-d / I-p 与 SOC 杂化",
    "生产路径：lg_cov / k40 / nb1-104；full112 作基准",
  ], colors.blue);

  // 5
  {
    const slide = prs.slides.add();
    addHeader(slide, "112 带不是任意大窗口，而是 CrI3 的化学完整母模型", "2. 求解器与材料模型");
    addMetric(slide, "112", "4 Cr × 5d + 12 I × 3p，再乘 spinor/SOC", 80, 150, 300, colors.blue);
    addMetric(slide, "84 + 28", "84 条价带 + 28 条导带；HOMO=84，LUMO=85", 420, 150, 300, colors.green);
    addMetric(slide, "nb1-104", "保留全部价带，只截最高 8 条导带", 760, 150, 300, colors.amber);
    addFormula(slide, "length gauge SBE:\ndρ/dt = -i[H0 + E·D, ρ] + E·Dkρ\n\ncurrent:\nJ_a(t) = -Σ_k Tr[P_a(k)ρ(k,t)] / (Nk A_cell)", 92, 300, 540, 210);
    addBullets(slide, [
      "D(k) 是 Berry connection / dipole matrix，协变差分保证 gauge-covariant k 空间传播。",
      "Peierls velocity gauge 截断会破坏 f-sum 抵消，因此只作为诊断分支。",
      "nb1-104 对 H1-H11 足够经济；讨论 H12+ 或 cutoff 时回到 full112。",
    ], 690, 312, 430, 168, { size: 18 });
    addFooter(slide, 5);
  }

  // 6
  {
    const slide = prs.slides.add();
    addHeader(slide, "速度规范截断失败，数学根源是 f-sum 抵消被破坏", "3. 经典光验证");
    addText(slide, "观察", 86, 150, 180, 26, { size: 18, bold: true, color: colors.red });
    addBullets(slide, [
      "Peierls VG 对称带窗下，H1 从 full112 的 0.082 被抬到 48、110、168 等量级。",
      "带窗变大反而不单调收敛，说明不是物理平台，而是截断背景电流。",
    ], 86, 188, 510, 100);
    addFormula(slide, "Σ_occ v_nn(k) = 0  需要完整价带抵消\n截掉深价带 -> 残余 ground-state current -> H1 伪增强", 86, 330, 510, 120, "#fff1f1");
    addBox(slide, 650, 156, 440, 300, { fill: "#ffffff", line: "#d7deeb" });
    addText(slide, "因此生产路径不再使用 VG 截断", 682, 190, 378, 40, { size: 26, bold: true, color: colors.red });
    addText(slide, "改用 lg_cov：没有需要巨大抗磁项抵消的结构，截断误差表现为高能虚跃迁通道逐步缺失，而不是低阶伪电流爆炸。", 682, 250, 370, 100, { size: 20, color: colors.ink });
    addFooter(slide, 6);
  }

  // 7
  {
    const slide = prs.slides.add();
    addHeader(slide, "经典基准已经通过带窗、k 网格和初态三项检查", "3. 经典光验证");
    await addImage(slide, assets.classical, 72, 132, 650, 352, "Classical validation summary", "contain");
    addMetric(slide, "1-4%", "nb1-104 与 full112 在 H1-H11 的典型偏差", 760, 150, 300, colors.green);
    addMetric(slide, "1.8e-5", "k60/full112 vs k40/full112 的 H1-H15 线性 L2 误差", 760, 260, 300, colors.blue);
    addMetric(slide, "4e-18", "full112 与 nb1-104 的 |J(0)| 机器零", 760, 370, 300, colors.purple);
    addText(slide, "结论：经典光绝对基准用 full112；经济计算用 nb1-104。高阶 H13+ 已接近数值地板，若研究 cutoff 必须回 full112 复核。", 92, 520, 1030, 54, { size: 21, bold: true, color: colors.ink });
    addFooter(slide, 7);
  }

  // 8
  {
    const slide = prs.slides.add();
    addHeader(slide, "AFM 选择定则用严格 P 共轭 -N 关上了关键缺口", "3. 经典光验证");
    await addImage(slide, assets.selrule, 72, 132, 640, 352, "AFM selection rule leakage", "contain");
    addFormula(slide, "J_e = [J(N)+J(-N)]/2\nJ_o = [J(N)-J(-N)]/2\n\n奇次 -> N-even\n偶次 -> N-odd", 750, 150, 330, 145, "#eef7ff");
    addMetric(slide, "13.1 -> 0.32", "H2 forbidden-channel leakage 改善 41 倍", 750, 330, 330, colors.red);
    addMetric(slide, "8.9e-6", "H3 leakage，说明奇次通道很干净", 750, 440, 330, colors.green);
    addText(slide, "旧独立 -N 的异常来自 DFT/Wannier 配对误差；P 共轭 -N 后选择定则恢复，说明 SBE 路径和 N 宇称分析本身没有问题。", 86, 535, 1030, 46, { size: 20, bold: true, color: colors.ink });
    addFooter(slide, 8);
  }

  // 9
  {
    const slide = prs.slides.add();
    addHeader(slide, "外部 benchmark 用来回答“求解器是否正常”，不是转移课题", "3. 经典光验证");
    const data = [
      ["ZnO", "QE -> Wannier -> SBE 链路检查", "4 原子无 SOC；20 WF；J(0) 随 k 收敛到机器零", "只作 2D kz=0 切片健康检查"],
      ["hBN", "文献式二带模型 benchmark", "解析 tb.dat；λ=1600 nm，I=3.5 TW/cm²；odd/even 分量结构正常", "first-principles slab 因真空态 spread 过大暂停"],
      ["CrI3", "目标体系", "112-band lg_cov、P 共轭选择定则、BSV 求积全部围绕它", "H12+ 与 Tier 2 仍需专项验证"],
    ];
    data.forEach((r, i) => {
      const y = 150 + i * 132;
      addBox(slide, 90, y, 1050, 104, { fill: colors.white, line: "#d8deea" });
      addText(slide, r[0], 112, y + 22, 115, 32, { size: 28, bold: true, color: [colors.blue, colors.green, colors.purple][i] });
      addText(slide, r[1], 260, y + 18, 300, 28, { size: 19, bold: true, color: colors.ink });
      addText(slide, r[2], 260, y + 54, 450, 28, { size: 16, color: colors.muted });
      addText(slide, r[3], 780, y + 36, 320, 38, { size: 16, color: colors.ink });
    });
    addFooter(slide, 9);
  }

  // 10
  addSectionSlide(prs, 10, "4. BSV 平均谱", "量子光平均谱可以高效算出来", [
    "BSV 平均谱 = 对指数强度分布的加权经典谱",
    "MC pilot 证明增强存在，但方差太大",
    "确定性求积把数十天的计算压到半天量级",
  ], colors.green);

  // 11
  {
    const slide = prs.slides.add();
    addHeader(slide, "BSV 平均谱的关键公式很简单：经典谱做指数权重平均", "4. BSV 平均谱");
    addFormula(slide, "<S(ω)>_BSV = ∫_0^∞ exp(-x) S(ω; x Ibar) dx", 110, 150, 700, 82, "#f0f7f5");
    addText(slide, "这一步只描述平均谱。它可以由经典强度分布复现，所以不能把它称为“严格量子优势”。它的价值是：在双层 AFM-CrI3 中建立量子光驱动 HHG 的第一组干净、可复查数据。", 110, 270, 700, 98, { size: 22, color: colors.ink });
    addBox(slide, 870, 150, 260, 250, { fill: "#ffffff", line: "#d7deeb" });
    addText(slide, "强度统计", 900, 180, 200, 28, { size: 22, bold: true, color: colors.green });
    addText(slide, "BSV 的高强度尾会更频繁地采到强场事件，因此高阶谐波被抬升。", 900, 230, 190, 84, { size: 20, color: colors.ink });
    addText(slide, "下一层量子性要看 Cconn、协方差和 QFI。", 900, 336, 190, 42, { size: 17, color: colors.muted });
    addFooter(slide, 11);
  }

  // 12
  {
    const slide = prs.slides.add();
    addHeader(slide, "MC pilot 给出信号，但确定性求积才是生产路线", "4. BSV 平均谱");
    await addImage(slide, assets.quad, 70, 132, 660, 366, "Quadrature versus Monte Carlo", "contain");
    addMetric(slide, "51", "MC pilot 样本数：能看趋势，误差仍大", 770, 150, 310, colors.amber);
    addMetric(slide, "16", "确定性强度节点：复现指数矩，零采样噪声", 770, 260, 310, colors.green);
    addMetric(slide, ">100x", "相对暴力 MC 的有效成本优势", 770, 370, 310, colors.blue);
    addText(slide, "判断：BSV 平均谱的数值瓶颈不是单条轨迹，而是采样策略。求积路线已经把问题从“随机噪声”变成“可控积分误差”。", 86, 525, 1030, 50, { size: 21, bold: true, color: colors.ink });
    addFooter(slide, 12);
  }

  // 13
  {
    const slide = prs.slides.add();
    addHeader(slide, "BSV 平均谱整体抬升，高次谐波增强更明显", "4. BSV 平均谱");
    await addImage(slide, assets.bsvSpectra, 70, 138, 520, 330, "BSV and classical spectra", "contain");
    await addImage(slide, assets.bsvEnh, 632, 138, 520, 330, "BSV enhancement", "contain");
    addMetric(slide, "H13 ≈ 363x", "高阶增强代表高强度尾对 cutoff 区的贡献", 118, 500, 300, colors.red);
    addMetric(slide, "H11 ≈ 60x", "已远高于经典固定强度谱", 490, 500, 300, colors.amber);
    addMetric(slide, "H1 ≈ 1.2x", "低阶仍接近经典响应", 860, 500, 300, colors.blue);
    addFooter(slide, 13);
  }

  // 14
  {
    const slide = prs.slides.add();
    addHeader(slide, "p! 标尺把增强机制讲清：低阶近微扰，高阶进入平台", "4. BSV 平均谱");
    await addImage(slide, assets.bsvScale, 74, 136, 690, 390, "Harmonic scaling with intensity", "contain");
    addFormula(slide, "若 S_n(I) ∝ I^p，则\n<S_n>_BSV / S_n(Ibar) = ∫e^{-x}x^p dx = p!", 800, 156, 330, 115, "#fff7ed");
    addBullets(slide, [
      "数据低于 p! 上限，说明很快离开纯微扰幂律。",
      "高阶增强来自强度分布高尾对 cutoff 的延展，而不是简单 p! 放大。",
      "这给后续 BSV 文章提供可解释的理论标尺。",
    ], 810, 304, 320, 150, { size: 18 });
    addFooter(slide, 14);
  }

  // 15
  {
    const slide = prs.slides.add();
    addHeader(slide, "当前结果的位置：R1 已经够成文，R2 是下一篇的核心增量", "5. 后续理论路线");
    const cols = [
      ["R1 平均谱", "BSV 强度平均 + HHG 增强", "数据已齐，最稳妥", "可由经典强度分布复现"],
      ["R2 输出光非经典性", "Cconn -> covariance -> squeezing / entanglement / QFI", "增量大，和 Feist 边界不冲突", "T2 会压制信号，先估量级"],
      ["R3 严格量子优势", "经典矩问题 / 正定 P / 凸可行性", "原创性最高", "数学和计算难度最大"],
    ];
    cols.forEach((c, i) => {
      const x = 80 + i * 370;
      addBox(slide, x, 150, 330, 330, { fill: colors.white, line: "#d6ddeb", radius: "rounded-xl" });
      addText(slide, c[0], x + 24, 180, 260, 32, { size: 24, bold: true, color: [colors.green, colors.blue, colors.purple][i] });
      addText(slide, c[1], x + 24, 232, 280, 70, { size: 18, color: colors.ink });
      addText(slide, "利： " + c[2], x + 24, 328, 280, 44, { size: 16, color: colors.green });
      addText(slide, "风险： " + c[3], x + 24, 392, 280, 54, { size: 16, color: colors.red });
    });
    addText(slide, "建议：R1 先写清楚；R2 同步做最小模型验证；R3 等 R2 量级出来后再投入。", 110, 535, 1000, 34, { size: 23, bold: true, color: colors.ink });
    addFooter(slide, 15);
  }

  // 16
  {
    const slide = prs.slides.add();
    addHeader(slide, "Tier 2 的计算对象是双时连通电流关联", "5. 后续理论路线");
    addFormula(slide, "C_ab(t,t') = <ΔJ_a(t) ΔJ_b(t')>\n\nmode projection:\nΣ_qq'  <-  ∫dt dt' u_q(t) C(t,t') u_q'(t')", 78, 150, 520, 140, "#eef7ff");
    addBullets(slide, [
      "平均电流 J(t) 给出 HHG 强度；双时关联给出光子涨落。",
      "协方差矩阵 Σ 可进一步计算压缩谱、双模纠缠、g2 与 QFI。",
      "这才是“输出谐波光是什么量子态”的问题。",
    ], 78, 330, 520, 140);
    addBox(slide, 690, 150, 410, 280, { fill: colors.white, line: "#d6ddeb" });
    addText(slide, "QRT/辅助算符实现", 720, 180, 320, 28, { size: 22, bold: true, color: colors.blue });
    addText(slide, "在 t' 用电流算符“敲一下”ρ，得到辅助算符 Λ；然后用同一个 SBE 传播子把 Λ 推进到 t，再取 Tr[P_a Λ_b]。", 720, 232, 330, 92, { size: 20, color: colors.ink });
    addText(slide, "好处：不重写求解器，只增加一个非厄米辅助传播通道。", 720, 356, 330, 46, { size: 18, color: colors.green });
    addFooter(slide, 16);
  }

  // 17
  {
    const slide = prs.slides.add();
    addHeader(slide, "N 宇称分解要从平均电流推广到关联函数", "5. 后续理论路线");
    addFormula(slide, "C_ee, C_oo, C_eo, C_oe\n\nJ_e/o = [J(N) ± J(-N)]/2", 84, 150, 360, 130, "#f0f7f5");
    addText(slide, "物理含义", 84, 322, 180, 30, { size: 22, bold: true, color: colors.green });
    addBullets(slide, [
      "Cee：普通晶格背景涨落。",
      "Coo：只在磁有序态打开的 N-奇量子涨落。",
      "Ceo/Coe：磁性与晶格通道之间的混合关联。",
    ], 84, 365, 420, 120);
    addBox(slide, 610, 150, 500, 300, { fill: "#fffdf5", line: "#ead7a6" });
    addText(slide, "为什么它值得做", 642, 182, 300, 30, { size: 24, bold: true, color: colors.amber });
    addText(slide, "平均 H2/H4 只是磁序的一阶指纹；Coo 则是磁序写入输出光量子涨落的二阶指纹。它直接连接压缩、纠缠和磁序 QFI。", 642, 242, 420, 100, { size: 22, color: colors.ink });
    addText(slide, "如果 Coo 可观测，论文的重心就能从“量子光增强 HHG”升级到“磁序量子增强传感”。", 642, 370, 420, 48, { size: 18, color: colors.red });
    addFooter(slide, 17);
  }

  // 18
  {
    const slide = prs.slides.add();
    addHeader(slide, "最小模型先行：4-band 拟合模型用来降低 Tier 2 风险", "5. 后续理论路线");
    addBullets(slide, [
      "112-band 第一性原理模型适合最终验证，但不适合一开始就调 QRT、协方差和 QFI。",
      "先从 112-WF 拟合 4-band 最小模型：保留带边、SOC/磁序矩阵元、N 宇称结构。",
      "在 4-band 上跑通 Cconn -> Σ -> squeezing / entanglement / QFI，再迁移回 112-band。",
    ], 86, 150, 590, 180);
    const steps = [
      ["112-WF", "提取带边与矩阵元"],
      ["4-band", "跑通 QRT 与关联"],
      ["112-band", "最终数值确认"],
      ["paper", "R1/R2 分层成文"],
    ];
    steps.forEach((s, i) => {
      const x = 110 + i * 260;
      addBox(slide, x, 400, 180, 90, { fill: colors.white, line: "#d6ddeb" });
      addText(slide, s[0], x + 20, 420, 140, 26, { size: 22, bold: true, color: [colors.blue, colors.green, colors.purple, colors.amber][i] });
      addText(slide, s[1], x + 20, 456, 140, 22, { size: 14, color: colors.muted });
      if (i < steps.length - 1) addText(slide, "→", x + 198, 423, 42, 34, { size: 30, bold: true, color: colors.line });
    });
    addFooter(slide, 18);
  }

  // 19
  {
    const slide = prs.slides.add();
    addHeader(slide, "后续计划按风险分层推进", "6. 计划与取舍");
    const items = [
      ["短期：把 R1 写成论文骨架", "固定 lg_cov、nb1-104/full112、P 共轭选择定则、BSV 求积结果；明确它是平均谱增强，不声称严格量子优势。"],
      ["短期并行：补 j_anom(t) 分解", "与 Liu 的 Berry 机制对接，确认 PT 构型下偶次是否由带间极化主导。"],
      ["中期：实现 QRT 辅助算符通道", "先 4-band，后 112-band；输出 Cconn、Σ、压缩谱、纠缠和 QFI。"],
      ["长线：经典不可复现判据", "在 R2 出现可观测信号后，再投入矩问题/正定 P/凸优化判据。"],
    ];
    items.forEach((it, i) => {
      const y = 145 + i * 105;
      addBox(slide, 90, y, 1040, 78, { fill: colors.white, line: "#d7deeb" });
      addText(slide, it[0], 118, y + 18, 350, 30, { size: 20, bold: true, color: [colors.green, colors.amber, colors.blue, colors.purple][i] });
      addText(slide, it[1], 500, y + 17, 580, 38, { size: 16, color: colors.ink });
    });
    addFooter(slide, 19);
  }

  // 20
  {
    const slide = prs.slides.add();
    addHeader(slide, "本阶段结论：经典基准可信，BSV 平均谱有结果，下一步转向关联", "6. 计划与取舍");
    addBox(slide, 90, 150, 1040, 250, { fill: "#172235", line: "#253752", radius: "rounded-xl" });
    addText(slide, "可以向导师明确汇报的三句话", 128, 180, 600, 34, { size: 24, bold: true, color: "#8ecae6" });
    addText(slide, "1. 求解器的经典光路径已经经过规范、带窗、k 网格、初态和选择定则验证。\n2. BSV 平均谱不是靠高噪声 MC，而是可以用确定性强度求积稳定得到。\n3. 真正的新物理在下一层：N 宇称分辨的双时电流关联与输出光量子统计。", 128, 238, 940, 108, { size: 25, color: colors.white });
    addMetric(slide, "R1", "平均谱论文：可整理成文", 110, 450, 270, colors.green);
    addMetric(slide, "R2", "Cconn/QFI：下一阶段主攻", 470, 450, 270, colors.blue);
    addMetric(slide, "R3", "量子优势判据：长线", 830, 450, 270, colors.purple);
    addFooter(slide, 20);
  }

  // QA exports
  for (const [index, slide] of prs.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await prs.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(QA, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA, `${stem}.layout.json`), await layout.text());
  }
  const montage = await prs.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(QA, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(prs);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

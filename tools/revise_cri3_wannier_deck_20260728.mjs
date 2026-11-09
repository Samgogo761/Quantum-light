import fs from "node:fs/promises";
import path from "node:path";
import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
} from "file:///C:/Users/26507/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/artifact_tool_utils.mjs";

const repo = "C:/Users/26507/Documents/New_SBEs/Quantum-light";
const workspace = path.join(repo, ".tmp_ppt_revision_20260728");
const starter = path.join(workspace, "template-starter.pptx");
const output = path.join(
  repo,
  "docs/07_理论框架与汇报/CrI3_Wannier模型选取与HHG生产方案_导师讨论_20260728_讲解增强版.pptx",
);

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(starter));

function slideAt(number) {
  return presentation.slides.items[number - 1];
}

function shapeByName(slide, name) {
  const shape = slide.shapes.items.find((item) => item.name === name);
  if (!shape) throw new Error(`Missing shape "${name}" on output slide ${presentation.slides.items.indexOf(slide) + 1}`);
  return shape;
}

function imageByName(slide, name) {
  const image = slide.images.items.find((item) => item.name === name);
  if (!image) throw new Error(`Missing image "${name}" on output slide ${presentation.slides.items.indexOf(slide) + 1}`);
  return image;
}

function frame(left, top, width, height) {
  return { left, top, width, height };
}

function setFrame(item, left, top, width, height) {
  item.frame = frame(left, top, width, height);
}

function setText(slide, name, value, options = {}) {
  const shape = shapeByName(slide, name);
  shape.text = value;
  if (options.frame) shape.frame = options.frame;
  if (options.fontSize) shape.text.style = { fontSize: options.fontSize };
  return shape;
}

async function replaceImage(slide, name, relativePath, alt, targetFrame) {
  const image = imageByName(slide, name);
  const ext = path.extname(relativePath).toLowerCase();
  const contentType =
    ext === ".svg"
      ? "image/svg+xml"
      : ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : "image/png";
  image.replace({
    blob: await fs.readFile(path.join(repo, relativePath)),
    contentType,
    alt,
    fit: "contain",
  });
  image.crop = { left: 0, top: 0, right: 0, bottom: 0 };
  image.fit = "contain";
  if (targetFrame) image.frame = targetFrame;
}

function deleteImage(slide, name) {
  imageByName(slide, name).delete();
}

function setNotes(slide, sources, talk = []) {
  const lines = ["[Sources]", ...sources.map((s) => `- ${s}`)];
  if (talk.length) lines.push("", "[讲解要点]", ...talk.map((s) => `- ${s}`));
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

function setPage(slide, name, number) {
  setText(slide, name, String(number).padStart(2, "0"));
}

// Slide 4 — band-fit evidence before the M40 failure argument.
{
  const slide = slideAt(4);
  setText(slide, "矩形 31", "能带拟合审计 · 每个候选模型均先优化 Wannier 参数");
  setText(slide, "矩形 32", "以 M40(d-only) 为最严格例：带隙可复现，但纯 Cr-d 子空间仍存在不可消除的结构性缺失");
  setPage(slide, "矩形 34", 4);

  setFrame(shapeByName(slide, "圆角矩形 35"), 60, 176, 1160, 342);
  setFrame(shapeByName(slide, "圆角矩形 38"), 660, 176, 550, 342);
  await replaceImage(
    slide,
    "图片 67",
    "docs/08_QE与Wannier/figures/CrI3_qe200_wannier112_donly40opt1_1x3_highCB.svg",
    "QE200、full112 与 M40-opt1 的近带隙和高导带能带比较",
    frame(76, 181, 1128, 295),
  );
  deleteImage(slide, "图片 68");

  setText(
    slide,
    "矩形 37",
    "三栏：QE200 ｜ full112 ｜ M40-opt1；Γ/K 带隙均复现为 0.667 / 1.124 eV",
    { frame: frame(80, 480, 600, 30), fontSize: 14 },
  );
  setText(
    slide,
    "矩形 40",
    "能量误差 L_E 均过闸：M52 8.7、M64 6.4、M88 7.5、M40-opt1 20.2 meV（阈值 50 meV）",
    { frame: frame(680, 480, 520, 30), fontSize: 14 },
  );

  setFrame(shapeByName(slide, "圆角矩形 41"), 72, 538, 520, 102);
  setText(slide, "矩形 42", "M40 参数确已充分搜索", { frame: frame(90, 549, 484, 26), fontSize: 17 });
  setText(
    slide,
    "矩形 43",
    "pilot：近隙 MAE 6.4 meV，但 Ω/WF=10.68 Å²；\nopt1：Ω/WF 降至 6.57 Å²，中窗 RMS 18.4 meV。",
    { frame: frame(90, 579, 484, 52), fontSize: 16 },
  );

  setFrame(shapeByName(slide, "圆角矩形 44"), 628, 538, 580, 102);
  setText(slide, "矩形 45", "能带闸门通过 ≠ 模型最终通过", { frame: frame(646, 549, 544, 26), fontSize: 17 });
  setText(
    slide,
    "矩形 46",
    "两套 40-WF 都把可表示的 d-like 能带拟合到极限；\n但密集 I-p 价带与部分高导带不在该 Hilbert 子空间中。",
    { frame: frame(646, 579, 544, 52), fontSize: 16 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "docs/08_QE与Wannier/CrI3_40WF审计第一轮_20260714.md"),
      path.join(repo, "docs/08_QE与Wannier/CrI3_模型评分与nv12验收_20260715.md"),
      path.join(repo, "docs/08_QE与Wannier/CrI3_M52_M64_M88验收与评分_20260716.md"),
      path.join(repo, "docs/08_QE与Wannier/figures/CrI3_qe200_wannier112_donly40opt1_1x3_highCB.svg"),
    ],
    [
      "先强调我们不是因为参数没调好而淘汰 M40；pilot 与 opt1 分别优化近隙误差和局域化。",
      "能量误差只是第一道闸门。所有 reduced models 都先通过 L_E，再比较子空间、位置矩阵和 HHG。",
    ],
  );
}

// Slide 5 — projectability and HHG: why energy fitting is insufficient.
{
  const slide = slideAt(5);
  setText(slide, "矩形 31", "M40 负面对照 · 能带吻合是必要条件，但不是 HHG 充分条件");
  setText(slide, "矩形 32", "M40 能拟合 Eₙ(k)，却无法表示 I-p 价带波函数；偶极、位置矩阵与相干路径随之失真");
  setPage(slide, "矩形 34", 5);
  await replaceImage(
    slide,
    "图片 67",
    "docs/08_QE与Wannier/figures/CrI3_qe_crd_projectability.png",
    "QE 各能带在 Cr-d 子空间中的 projectability",
    frame(92, 188, 505, 284),
  );
  await replaceImage(
    slide,
    "图片 68",
    "docs/08_QE与Wannier/figures/CrI3_HHG_peak_ratios_vs_nb104.svg",
    "M40-pilot 相对 nb1-104 参考的 HHG 峰高比",
    frame(680, 188, 510, 284),
  );
  setText(
    slide,
    "矩形 37",
    "p_d,n(k)=Σα∈Cr-3d |⟨φαk|ψn,k^QE⟩|²；红=占据、蓝=导带，误差棒为 k 范围",
    { frame: frame(82, 476, 526, 38), fontSize: 14 },
  );
  setText(
    slide,
    "矩形 40",
    "图中为 M40-pilot：H1=0.727×、H6=1.733×、H10=6.638×；opt1 另为 H1=0.344×、H6=6.593×",
    { frame: frame(672, 476, 526, 38), fontSize: 14 },
  );
  setFrame(shapeByName(slide, "圆角矩形 41"), 72, 538, 520, 102);
  setText(slide, "矩形 42", "左图表现与原因", { frame: frame(90, 549, 484, 26), fontSize: 17 });
  setText(
    slide,
    "矩形 43",
    "VBM 的 p_d 仅 3–6%，第一导带 40–47%，第二导带 81–86%。\n价带顶约 94% 以上不是 Cr-d，d-only 无法表示初始波函数。",
    { frame: frame(90, 579, 484, 52), fontSize: 16 },
  );
  setFrame(shapeByName(slide, "圆角矩形 44"), 628, 538, 580, 102);
  setText(slide, "矩形 45", "右图表现与结论", { frame: frame(646, 549, 544, 26), fontSize: 17 });
  setText(
    slide,
    "矩形 46",
    "R_q=P_q^M40/P_q^ref；虚线 1 才是正确参考。\n低阶被压低、高阶伪增强，说明 Dₘₙ(k)、相位与 sum rule 已失真。",
    { frame: frame(646, 579, 544, 52), fontSize: 16 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "docs/08_QE与Wannier/CrI3_40WF审计第一轮_20260714.md"),
      path.join(repo, "docs/08_QE与Wannier/CrI3_模型评分与nv12验收_20260715.md"),
      path.join(repo, "docs/08_QE与Wannier/figures/CrI3_qe_crd_projectability.png"),
      path.join(repo, "docs/08_QE与Wannier/figures/CrI3_HHG_peak_ratios_vs_nb104.svg"),
    ],
    [
      "左图纵轴接近 1 表示 QE 态可由 Cr-d 子空间表示；接近 0 表示该态主要来自被删掉的 I-p。",
      "右图的柱状数据属于 M40-pilot；旧版把 opt1 数值写到 pilot 图旁，现已纠正。",
      "核心句：Eₙ(k) 接近只约束本征值，不保证本征矢、偶极矩阵和 HHG 相位正确。",
    ],
  );
}

// Slide 6 — keep model-definition content, only renumber.
setPage(slideAt(6), "矩形 41", 6);

// Slide 7 — enlarge and explain the static score radar.
{
  const slide = slideAt(7);
  setText(slide, "矩形 27", "静态闸门 · 雷达图怎样读，为什么 M88 胜出");
  setText(slide, "矩形 28", "四个方向均为归一化惩罚量：越靠中心越好；带数不是唯一决定因素");
  setPage(slide, "矩形 30", 7);
  setFrame(shapeByName(slide, "圆角矩形 31"), 60, 178, 610, 355);
  setFrame(shapeByName(slide, "圆角矩形 34"), 690, 178, 530, 355);
  await replaceImage(
    slide,
    "图片 57",
    "docs/08_QE与Wannier/figures/CrI3_model_score_radar_20260716.png",
    "full112、M40、M52、M64、M88 的静态评分雷达图",
    frame(82, 184, 565, 330),
  );
  deleteImage(slide, "图片 58");
  setText(
    slide,
    "矩形 33",
    "半径越小越好：M88 是 reduced models 中面积最小、最均衡者",
    { frame: frame(72, 501, 586, 24), fontSize: 15 },
  );
  setFrame(shapeByName(slide, "圆角矩形 37"), 710, 198, 490, 285);
  setText(slide, "矩形 38", "四个轴的物理意义", { frame: frame(730, 212, 450, 30), fontSize: 18 });
  setText(
    slide,
    "矩形 39",
    "L_E：QE–Wannier 能量 RMS 误差\nL_sub：目标能窗内的子空间/态数失配\nL_loc：Wannier spread，衡量局域化\nL_hop：相对 full112 丢失的 hopping 权重\n\nM88 四项同时较小；M64 虽有 64 WF，方向选错仍失败。",
    { frame: frame(730, 252, 450, 215), fontSize: 17 },
  );
  setText(
    slide,
    "矩形 36",
    "静态评分只做预筛；最终必须由普通 HHG 与磁性 Jₒ 动力学验收",
    { frame: frame(710, 495, 490, 26), fontSize: 15 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "docs/08_QE与Wannier/CrI3_M52_M64_M88验收与评分_20260716.md"),
      path.join(repo, "docs/08_QE与Wannier/figures/CrI3_model_score_radar_20260716.png"),
    ],
    [
      "所有轴都是惩罚量，因此不能按普通能力雷达图理解成越外越好。",
      "M88 的优势来自同时保留面内 I-px/py 与 Cr-d 杂化，不是单纯因为 88 大于 64。",
    ],
  );
}

// Slide 8 — enlarge and explain the ordinary-HHG ratio.
{
  const slide = slideAt(8);
  setText(slide, "矩形 27", "动力学闸门 1 · 普通 HHG 比值怎样读");
  setText(slide, "矩形 28", "横向比较同一谐波阶次；R_q=1 表示待检验模型与 full112 的强度完全一致");
  setPage(slide, "矩形 30", 8);
  setFrame(shapeByName(slide, "圆角矩形 31"), 60, 178, 730, 355);
  setFrame(shapeByName(slide, "圆角矩形 34"), 820, 178, 400, 355);
  deleteImage(slide, "图片 57");
  await replaceImage(
    slide,
    "图片 58",
    "results/magnetic_hhg_ablation_pmN_20260719/figures/figB_hhg_ratio_H1_H9.png",
    "M52、M64、M88 的普通 HHG 强度相对 full112 的比值",
    frame(72, 205, 706, 270),
  );
  setText(
    slide,
    "矩形 36",
    "R_q=I_q^(M)/I_q^(112)，I_q∝|J_q|²；虚线 1 是理想参考",
    { frame: frame(72, 490, 706, 30), fontSize: 16 },
  );
  setFrame(shapeByName(slide, "圆角矩形 37"), 840, 198, 360, 305);
  setText(slide, "矩形 38", "符号与读图结论", { frame: frame(860, 213, 320, 30), fontSize: 18 });
  setText(
    slide,
    "矩形 39",
    "q：谐波阶数\nJ_q：qω₀ 处的复电流\nI_q：第 q 阶谐波强度\nM：待检验模型；112：full112\n\nM88 的 H1/H3/H5/H7/H9=\n0.994/0.974/1.012/0.923/0.975。\nM52 系统偏弱；M64 高阶异常。",
    { frame: frame(860, 252, 320, 235), fontSize: 17 },
  );
  setText(slide, "矩形 33", "结论：准确性不随 52→64→88 简单单调变化；轨道方向更重要", {
    frame: frame(840, 506, 360, 20),
    fontSize: 14,
  });
  setNotes(
    slide,
    [
      path.join(repo, "docs/09_少带模型/给场外AI_pmN消融完成后现状_20260719.md"),
      path.join(repo, "results/magnetic_hhg_ablation_pmN_20260719/figures/figB_hhg_ratio_H1_H9.png"),
    ],
    [
      "横轴是谐波阶数，纵轴是强度比；每个点只与同阶 full112 比较。",
      "M88 基本贴住 1，证明保留全部 I-px/py 对 x 偏振普通 HHG 是关键。",
    ],
  );
}

// Slide 9 — define Je/Jo and explain the Jo amplitude-ratio chart.
{
  const slide = slideAt(9);
  setText(slide, "矩形 27", "磁序奇通道 1/2 · Jₒ 是什么，幅值比怎样读");
  setText(slide, "矩形 28", "±N 磁畴差分消除与磁畴无关的背景；Jₒ 是复电流振幅，不是谐波强度");
  setPage(slide, "矩形 30", 9);
  setFrame(shapeByName(slide, "圆角矩形 31"), 60, 178, 700, 355);
  setFrame(shapeByName(slide, "圆角矩形 34"), 790, 178, 430, 355);
  await replaceImage(
    slide,
    "图片 57",
    "results/magnetic_hhg_ablation_pmN_20260719/figures/figC_Jo_ablation_vs_full112.png",
    "M52、M64、M88 的磁序奇电流幅值相对 full112 的比值",
    frame(70, 218, 680, 260),
  );
  deleteImage(slide, "图片 58");
  setText(
    slide,
    "矩形 33",
    "Rₒ,q^(M/112)=|Jₒ,q^M|/|Jₒ,q^112|；水平线 1 表示幅值与 full112 一致",
    { frame: frame(72, 490, 676, 32), fontSize: 15 },
  );
  setText(
    slide,
    "矩形 36",
    "Jₑ,q = [J_q(+N)+J_q(−N)] / 2\nJₒ,q = [J_q(+N)−J_q(−N)] / 2",
    { frame: frame(812, 195, 386, 78), fontSize: 19 },
  );
  setFrame(shapeByName(slide, "圆角矩形 37"), 810, 286, 390, 220);
  setText(slide, "矩形 38", "下标与物理意义", { frame: frame(830, 301, 350, 30), fontSize: 18 });
  setText(
    slide,
    "矩形 39",
    "q：谐波阶数；J_q：复 Fourier 振幅\n+N/−N：一对反向 Néel 磁畴\ne：even，磁畴反转后不变\no：odd，磁畴反转后变号\n|Jₒ,q|：磁序敏感幅值；arg Jₒ,q：相位\n\nM88：H2/H4/H6/H8=\n1.007/1.000/0.940/0.967。",
    { frame: frame(830, 340, 350, 150), fontSize: 16 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "docs/09_少带模型/给场外AI_pmN消融完成后现状_20260719.md"),
      path.join(repo, "results/magnetic_hhg_ablation_pmN_20260719/figures/figC_Jo_ablation_vs_full112.png"),
    ],
    [
      "J_e 保留磁畴无关背景；J_o 提取在 N→−N 时变号的磁序敏感通道。",
      "M64 的 H10 看似增强 5.27 倍，但该模型已在位置矩阵和高阶健康检查中失败，不能解释成新物理。",
    ],
  );
}

// Slide 10 — define and explain the Jo phase error.
{
  const slide = slideAt(10);
  setText(slide, "矩形 27", "磁序奇通道 2/2 · 复相位误差怎样定义");
  setText(slide, "矩形 28", "量子关联和模式干涉依赖复相位；只比较 |Jₒ| 或强度不够");
  setPage(slide, "矩形 30", 10);
  setFrame(shapeByName(slide, "圆角矩形 31"), 60, 178, 740, 355);
  setFrame(shapeByName(slide, "圆角矩形 34"), 830, 178, 390, 355);
  deleteImage(slide, "图片 57");
  await replaceImage(
    slide,
    "图片 58",
    "results/magnetic_hhg_P0_JeJo_complex_20260719/figures/figP0_Jo_amp_phase_error_M52_M88.png",
    "M52 与 M88 的磁序奇复幅值和相位误差",
    frame(70, 214, 720, 268),
  );
  setText(
    slide,
    "矩形 36",
    "M88 在 H2–H8 同时保住幅值和相位；H10 开始进入高导带敏感边界",
    { frame: frame(72, 490, 716, 32), fontSize: 15 },
  );
  setText(
    slide,
    "矩形 33",
    "Δφₒ,q^(M/112)\n= arg[Jₒ,q^M (Jₒ,q^112)*]",
    { frame: frame(850, 198, 350, 75), fontSize: 20 },
  );
  setFrame(shapeByName(slide, "圆角矩形 37"), 850, 288, 340, 215);
  setText(slide, "矩形 38", "上标、符号与判据", { frame: frame(870, 303, 300, 30), fontSize: 18 });
  setText(
    slide,
    "矩形 39",
    "M/112：模型 M 相对 full112\n*：复共轭；arg：复数相位\nΔφ=0：发射相位完全一致\n\n相位偏差会改变谐波间干涉；\nH10 的敏感性不能被强度相近掩盖。",
    { frame: frame(870, 342, 300, 145), fontSize: 16 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "results/magnetic_hhg_P0_JeJo_complex_20260719/figures/figP0_Jo_amp_phase_error_M52_M88.png"),
      path.join(repo, "docs/09_少带模型/给场外AI_pmN消融完成后现状_20260719.md"),
    ],
    [
      "相位误差用模型复幅值乘参考复共轭来定义，避免分别取相位造成分支跳跃。",
      "M88 H2–H8 的幅值和相位同时通过，才有资格做后续磁性模式关联的材料 parent。",
    ],
  );
}

// Slide 11 — define the S2 active-space cases and explain the spectrum overlay.
{
  const slide = slideAt(11);
  setText(slide, "矩形 27", "S2 截断 1/2 · 固定 parent 后还能删多少导带？");
  setText(slide, "矩形 28", "S2 是项目内部的第二阶段 active-space 测试：不重新 Wannier 化，只改变 SBE 中保留的 Bloch 带");
  setPage(slide, "矩形 30", 11);
  setFrame(shapeByName(slide, "圆角矩形 31"), 60, 178, 720, 355);
  setFrame(shapeByName(slide, "圆角矩形 34"), 810, 178, 410, 355);
  await replaceImage(
    slide,
    "图片 57",
    "results/S2_truncation_pmN_20260720/figures/fig1_S2_hhg_overlay.png",
    "M88 与 M52 不同 active Bloch 带数下的 HHG 光谱叠加",
    frame(70, 220, 700, 270),
  );
  deleteImage(slide, "图片 58");
  setText(
    slide,
    "矩形 33",
    "谱线越重合越可靠：M88-c20 紧贴 parent；c10 已损失 H1；M52-c6 出现伪峰",
    { frame: frame(72, 495, 696, 28), fontSize: 15 },
  );
  setText(
    slide,
    "矩形 36",
    "命名规则：Mxx 是 Wannier parent；cN 表示保留最低 N 条导带",
    { frame: frame(830, 195, 370, 42), fontSize: 16 },
  );
  setFrame(shapeByName(slide, "圆角矩形 37"), 830, 248, 370, 260);
  setText(slide, "矩形 38", "S2 的六种情况", { frame: frame(850, 263, 330, 30), fontSize: 18 });
  setText(
    slide,
    "矩形 39",
    "M88-full = 58v+30c = 88\nM88-c20 = 58v+20c = 78\nM88-c10 = 58v+10c = 68\nM52-full = 24v+28c = 52\nM52-c10 = 24v+10c = 34\nM52-c6  = 24v+6c  = 30\n\nv：全部占据价带；c：最低空导带。",
    { frame: frame(850, 302, 330, 190), fontSize: 16 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "docs/09_少带模型/给场外AI_S2截断与F0诊断全量报告_20260720.md"),
      path.join(repo, "results/S2_truncation_pmN_20260720/figures/fig1_S2_hhg_overlay.png"),
    ],
    [
      "parent/full 是该 Wannier 模型的全部 Bloch 带；full112 才是最终材料参考。",
      "S2 的科学问题是：在轨道子空间已经选对之后，还能安全删除多少高导带。",
    ],
  );
}

// Slide 12 — explain the S2 truncation-to-parent ratios and the strict-30 failure.
{
  const slide = slideAt(12);
  setText(slide, "矩形 27", "S2 截断 2/2 · 为什么裸 30 带失败，而 78 带通过");
  setText(slide, "矩形 28", "先与各自 parent 比较截断误差，再检查 parent 是否接近 full112；两个闸门缺一不可");
  setPage(slide, "矩形 30", 12);
  setFrame(shapeByName(slide, "圆角矩形 31"), 60, 178, 720, 355);
  setFrame(shapeByName(slide, "圆角矩形 34"), 810, 178, 410, 355);
  deleteImage(slide, "图片 57");
  await replaceImage(
    slide,
    "图片 58",
    "results/S2_truncation_pmN_20260720/figures/fig2_S2_hhg_ratio_vs_parent.png",
    "不同 active-space 截断相对各自 parent 的 HHG 峰高比",
    frame(70, 214, 700, 280),
  );
  setText(
    slide,
    "矩形 36",
    "R_q^(trunc/parent)=I_q^trunc/I_q^parent；R_q=1 表示截断后完全复现 parent",
    { frame: frame(72, 495, 696, 28), fontSize: 15 },
  );
  setText(
    slide,
    "矩形 33",
    "图中蓝柱在 1.5 处截断；真实的 M52-c6 H5=16.27×、H9=8.05×",
    { frame: frame(830, 195, 370, 46), fontSize: 16 },
  );
  setFrame(shapeByName(slide, "圆角矩形 37"), 830, 252, 370, 256);
  setText(slide, "矩形 38", "逐项比较结果", { frame: frame(850, 267, 330, 30), fontSize: 18 });
  setText(
    slide,
    "矩形 39",
    "M52-c6=30：高阶伪增强，淘汰\nM52-c10=34：恢复 M52 parent，\n但 M52 parent 本身低估 full112\nM88-c10=68：H1 已明显损失\nM88-c20=78：H1–H9 保持 parent\n\n结论：78 带是最低通过候选；\n失败的是裸 24v+6c，不是否定 downfolding。",
    { frame: frame(850, 306, 330, 186), fontSize: 16 },
  );
  setNotes(
    slide,
    [
      path.join(repo, "docs/09_少带模型/给场外AI_S2截断与F0诊断全量报告_20260720.md"),
      path.join(repo, "results/S2_truncation_pmN_20260720/figures/fig2_S2_hhg_ratio_vs_parent.png"),
    ],
    [
      "M52-c10 通过 parent 闸门但不通过 full112 闸门，因此只能是机制模型。",
      "M52-c6 的异常来自导带不完整破坏通道抵消；不能把图中被截短的柱高误读为约 1.5。",
    ],
  );
}

// Slide 13 — P2 band-truncation error definition.
{
  const slide = slideAt(13);
  setText(slide, "矩形 37", "P2 最终验收 1/2 · 带截断复误差怎样定义");
  setText(slide, "矩形 38", "同一 k60 网格下比较 M88-c20(78) 与 M88-full(88)：同时检验幅值和相位");
  setPage(slide, "矩形 40", 13);
  setFrame(shapeByName(slide, "圆角矩形 41"), 60, 178, 690, 350);
  setFrame(shapeByName(slide, "圆角矩形 44"), 780, 178, 440, 350);
  await replaceImage(
    slide,
    "图片 69",
    "results/P2_k60_accept_20260721/figures/fig2_eps_band_k60.png",
    "M88-c20 相对 M88-full 的普通与磁序奇通道复误差",
    frame(70, 214, 670, 265),
  );
  deleteImage(slide, "图片 72");
  setText(
    slide,
    "矩形 43",
    "红虚线：普通通道 10%、磁序奇通道 15% 的验收阈值",
    { frame: frame(72, 490, 666, 26), fontSize: 15 },
  );
  setText(
    slide,
    "矩形 46",
    "ε_band^J(q)=|J_q^78−J_q^88|/(|J_q^88|+δ)\nε_band^o(q)=|Jₒ,q^78−Jₒ,q^88|/(|Jₒ,q^88|+δ)\n\n上标 J：普通电流；o：磁序奇通道\n下标 band：误差来自 active-band 截断\n78/88：active 带数；q：谐波阶数；δ>0：数值底",
    { frame: frame(805, 195, 390, 305), fontSize: 17 },
  );
  setText(slide, "矩形 48", "≤7.5%");
  setText(slide, "矩形 49", "普通 H1–H9");
  setText(slide, "矩形 51", "≤6.7%");
  setText(slide, "矩形 52", "Jₒ H2–H8");
  setText(slide, "矩形 54", "46%");
  setText(slide, "矩形 55", "Jₒ H10 复误差");
  setText(slide, "矩形 57", "0.43 rad");
  setText(slide, "矩形 58", "Jₒ H10 相位差");
  setNotes(
    slide,
    [
      path.join(repo, "docs/09_少带模型/P2_k60_验收报告_20260721.md"),
      path.join(repo, "results/P2_k60_accept_20260721/figures/fig2_eps_band_k60.png"),
    ],
    [
      "ε_band 比较的是两个复数的距离，所以会同时惩罚振幅和相位偏差。",
      "H10 的 J_o 虽弱，但 46% 复误差和 0.43 rad 相位差都明确失败；H2–H8 通过。",
    ],
  );
}

// Slide 14 — P2 k-mesh error definition.
{
  const slide = slideAt(14);
  setText(slide, "矩形 37", "P2 最终验收 2/2 · k 网格误差怎样定义");
  setText(slide, "矩形 38", "固定模型，比较 k40 与 k60；目标通道变化小于约 1%，说明当前主误差来自删带而非 k 网格");
  setPage(slide, "矩形 40", 14);
  setFrame(shapeByName(slide, "圆角矩形 41"), 60, 178, 690, 350);
  setFrame(shapeByName(slide, "圆角矩形 44"), 780, 178, 440, 350);
  deleteImage(slide, "图片 69");
  await replaceImage(
    slide,
    "图片 72",
    "results/P2_k60_accept_20260721/figures/fig3_eps_k_mesh.png",
    "M88-full 与 M88-c20 从 k40 到 k60 的普通和磁序奇通道复误差",
    frame(70, 214, 670, 265),
  );
  setText(
    slide,
    "矩形 46",
    "k40=40×40×1，k60=60×60×1；目标通道均已进入平台区",
    { frame: frame(72, 490, 666, 26), fontSize: 15 },
  );
  setText(
    slide,
    "矩形 43",
    "ε_k^X(q;M)=|J_X,q^(M,k60)−J_X,q^(M,k40)|\n                 /(|J_X,q^(M,k60)|+δ)\n\nX=J：普通电流；X=o：磁序奇通道\nM：M88-full 或 M88-c20\n下标 k：误差来自 k-mesh；q：谐波阶数\nδ>0：防止极弱参考信号使比值发散",
    { frame: frame(805, 195, 390, 305), fontSize: 17 },
  );
  setText(slide, "矩形 48", "k40");
  setText(slide, "矩形 49", "40×40×1");
  setText(slide, "矩形 51", "k60");
  setText(slide, "矩形 52", "60×60×1");
  setText(slide, "矩形 54", "X=J 或 o");
  setText(slide, "矩形 55", "普通 / 磁序奇");
  setText(slide, "矩形 57", "无需 k80");
  setText(slide, "矩形 58", "当前目标已收敛");
  setNotes(
    slide,
    [
      path.join(repo, "docs/09_少带模型/P2_k60_验收报告_20260721.md"),
      path.join(repo, "results/P2_k60_accept_20260721/figures/fig3_eps_k_mesh.png"),
    ],
    [
      "k40/k60 的差别通常小于 1%，所以 M88-c20@k40 可用于日常 H1–H9 与 J_o H2–H8。",
      "这里的结论不能外推到 H11+、不同波长、不同强度或量子双时关联。",
    ],
  );
}

// Slides 15–16 — keep content, update page markers.
setPage(slideAt(15), "矩形 32", 15);
setPage(slideAt(16), "矩形 47", 16);

await fs.mkdir(path.dirname(output), { recursive: true });
const exported = await PresentationFile.exportPptx(presentation);
await exported.save(output);
console.log(JSON.stringify({ output, slides: presentation.slides.items.length }, null, 2));

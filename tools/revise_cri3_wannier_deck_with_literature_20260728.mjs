import path from "node:path";
import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
} from "file:///C:/Users/26507/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/artifact_tool_utils.mjs";

const repo = "C:/Users/26507/Documents/New_SBEs/Quantum-light";
const workspace = path.join(repo, ".tmp_ppt_literature_20260728");
const starter = path.join(workspace, "template-starter.pptx");
const output = path.join(
  repo,
  "docs",
  "07_理论框架与汇报",
  "CrI3_Wannier模型选取与HHG生产方案_导师讨论_20260728_文献论证版.pptx",
);

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(
  await FileBlob.load(starter),
);

function slideAt(number) {
  return presentation.slides.items[number - 1];
}

function shapeById(slide, id) {
  const shape = slide.shapes.items.find((item) => String(item.id) === String(id));
  if (!shape) {
    throw new Error(
      `Missing shape id ${id} on slide ${presentation.slides.items.indexOf(slide) + 1}`,
    );
  }
  return shape;
}

function setText(slide, id, value, options = {}) {
  const shape = shapeById(slide, id);
  shape.text = value;
  if (options.frame) shape.frame = options.frame;
  const style = {};
  if (options.fontSize) style.fontSize = options.fontSize;
  if (options.bold !== undefined) style.bold = options.bold;
  if (options.color) style.color = options.color;
  if (options.alignment) style.alignment = options.alignment;
  if (Object.keys(style).length) shape.text.style = style;
  return shape;
}

function frame(left, top, width, height) {
  return { left, top, width, height };
}

function setPageNumber(slide, pageNumber) {
  const marker = slide.shapes.items.find((item) => {
    const text = item.text?.toString?.().trim() ?? "";
    const f = item.frame;
    return (
      f &&
      f.left >= 1160 &&
      f.top <= 80 &&
      /^\d{2}$/.test(text)
    );
  });
  if (marker) marker.text = String(pageNumber).padStart(2, "0");
}

function setNotes(slide, sources, talk = []) {
  const lines = ["[Sources]", ...sources.map((source) => `- ${source}`)];
  if (talk.length) {
    lines.push("", "[讲解要点]", ...talk.map((item) => `- ${item}`));
  }
  slide.speakerNotes.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

function appendNotes(slide, sources, talk = []) {
  const base = (slide.speakerNotes.text ?? "").trim();
  const lines = [];
  if (sources.length) {
    lines.push("[新增文献依据]", ...sources.map((source) => `- ${source}`));
  }
  if (talk.length) {
    lines.push("", "[新增讲解要点]", ...talk.map((item) => `- ${item}`));
  }
  slide.speakerNotes.setText(
    [base, lines.join("\n")].filter(Boolean).join("\n\n"),
  );
  slide.speakerNotes.setVisible(true);
}

const refs = {
  R1: "R1 — “Orbital perspective on high-harmonic generation from solids,” Nature Communications 14, 8421 (2023), https://doi.org/10.1038/s41467-023-44041-0",
  R2: "R2 — G. P. Zhang and Y. H. Bai, Physical Review B 99, 094313 (2019), https://doi.org/10.1103/PhysRevB.99.094313",
  R3: "R3 — G. P. Zhang and Y. H. Bai, Physical Review B 101, 081412(R) (2020), https://doi.org/10.1103/PhysRevB.101.081412",
  R4: "R4 — S. Gholam-Mirzaei et al., arXiv:2503.05073 (2025), preprint, https://arxiv.org/abs/2503.05073",
  R5: "R5 — P. G. Hawkins, M. Yu. Ivanov, and V. S. Yakovlev, Physical Review A 91, 013405 (2015), https://doi.org/10.1103/PhysRevA.91.013405",
  R6: "R6 — A. Taghizadeh, F. Hipólito, and T. G. Pedersen, Physical Review B 96, 195413 (2017), https://doi.org/10.1103/PhysRevB.96.195413",
  R7: "R7 — R. E. F. Silva, F. Martín, and M. Ivanov, Physical Review B 100, 195201 (2019), https://doi.org/10.1103/PhysRevB.100.195201",
  R8: "R8 — L. Yue and M. B. Gaarde, Physical Review A 101, 053411 (2020), https://doi.org/10.1103/PhysRevA.101.053411",
  R9: "R9 — D. Soriano et al., npj Computational Materials 7, 162 (2021), https://doi.org/10.1038/s41524-021-00631-4",
  R10: "R10 — H. Liu et al., EPL 140, 26005 (2022), https://doi.org/10.1209/0295-5075/ac9c26",
};

// Slide 2: connect the literature guardrails to the internal evidence chain.
setText(
  slideAt(2),
  49,
  "文献先给方法护栏，本项目再用数据决定带数：先保留正确轨道，再在正确 parent 上截空带。[R1–R9]",
);
appendNotes(
  slideAt(2),
  [refs.R1, refs.R5, refs.R6, refs.R7, refs.R8, refs.R9],
  [
    "老师以往约30带的经验在孤立少带流形或低阶目标中可以成立，但不能跨材料直接移植。",
    "文献只告诉我们哪些自由度不能先验删除；CrI3 的具体带数仍由 full112→M40/M52/M64/M88→S2/P2 数据决定。",
  ],
);

// Slide 3: full112 benchmark and literature-scale context.
setText(
  slideAt(3),
  32,
  "基准 · 112 条不是任意大窗口；多带传播先例见 [R1–R3]",
  { frame: frame(70, 42, 820, 28) },
);
appendNotes(
  slideAt(3),
  [refs.R1, refs.R2, refs.R3, refs.R9],
  [
    "不要说“文献证明 CrI3 必须112带”；准确说法是，44、106和160带的直接传播说明复杂材料使用大空间并不异常。",
    "本项目的112来自4个Cr的d旋量与12个I的p旋量完整计数；是否能缩减，要由后面的本材料验收决定。",
  ],
);

// Slide 4: literature precedent, reusing the four-card model-definition layout.
{
  const slide = slideAt(4);
  setText(slide, 39, "文献对照 · 复杂材料 HHG 如何处理多带问题");
  setText(
    slide,
    40,
    "常见策略是保留物理完整的多轨道或连续能窗，再按目标谐波做受控缩减",
  );

  setText(slide, 44, "ReS₂ · 44 MLWF", {
    frame: frame(92, 201, 250, 34),
    fontSize: 24,
  });
  setText(slide, 45, "Nat. Commun. 14, 8421 (2023) [R1]", {
    frame: frame(305, 205, 275, 28),
    fontSize: 14,
  });
  setText(slide, 46, "完整传播 44×44 密度矩阵；50×50 k 网格", {
    frame: frame(92, 247, 490, 28),
    fontSize: 16,
  });
  setText(
    slide,
    47,
    "动力学后再做原子/轨道分辨电流；未先压成二带或四带。",
    { frame: frame(92, 286, 490, 48), fontSize: 15 },
  );

  setText(slide, 49, "BaTiS₃ · 106 bands", {
    frame: frame(666, 201, 270, 34),
    fontSize: 24,
  });
  setText(slide, 50, "Phys. Rev. B 99, 094313 (2019) [R2]", {
    frame: frame(900, 205, 255, 28),
    fontSize: 14,
  });
  setText(slide, 51, "直接传播 DFT 第 41–146 带的连续大能窗", {
    frame: frame(666, 247, 490, 28),
    fontSize: 16,
  });
  setText(
    slide,
    52,
    "改变所取能窗后定性结论稳定；重点是覆盖目标发射能区。",
    { frame: frame(666, 286, 490, 48), fontSize: 15 },
  );

  setText(slide, 54, "solid C₆₀ · 160 bands", {
    frame: frame(92, 406, 285, 34),
    fontSize: 24,
  });
  setText(slide, 55, "Phys. Rev. B 101, 081412(R) (2020) [R3]", {
    frame: frame(360, 410, 220, 28),
    fontSize: 13,
  });
  setText(slide, 56, "120 价带 + 40 导带；完整多带 Liouville", {
    frame: frame(92, 452, 490, 28),
    fontSize: 16,
  });
  setText(
    slide,
    57,
    "用多带密度矩阵处理占据与 Pauli 约束，不采用单活性电子近似。",
    { frame: frame(92, 491, 490, 48), fontSize: 15 },
  );

  setText(slide, 59, "Ag · 9 MLWF", {
    frame: frame(666, 406, 220, 34),
    fontSize: 24,
  });
  setText(slide, 60, "arXiv:2503.05073 (2025, preprint) [R4]", {
    frame: frame(835, 410, 320, 28),
    fontSize: 13,
  });
  setText(slide, 61, "5s + 5p + 4d：化学上完整的小轨道组", {
    frame: frame(666, 452, 490, 28),
    fontSize: 16,
  });
  setText(
    slide,
    62,
    "H7/H9 仍依赖 d 电子；少带可行来自简单原胞，不是“9带普适”。",
    { frame: frame(666, 491, 490, 48), fontSize: 15 },
  );

  setText(
    slide,
    63,
    "共同策略：完整物理空间作 reference → 按目标谐波受控缩减。full112 是参考，78 带是经 S2/P2 验收的生产空间。\n注：各论文的带/轨道与 spin 计数不同，数字只用于比较处理策略。",
    { frame: frame(78, 603, 1120, 55), fontSize: 17 },
  );
  setNotes(
    slide,
    [refs.R1, refs.R2, refs.R3, refs.R4],
    [
      "ReS2与CrI3最接近：作者保留44个化学Wannier轨道，完整传播后再做轨道电流分解。",
      "BaTiS3和C60说明，106或160带的Liouville传播已有公开先例；上百带不是方法失败。",
      "Ag只有9个轨道，是因为一个简单原胞的5s/5p/4d已构成化学完整窗口；它反而说明少带必须有材料依据。",
      "本页不证明CrI3必须112带，而是反驳“复杂材料原则上只能用约30带”的先验。",
    ],
  );
}

// Slide 5: energy fitting is necessary but not sufficient.
setText(slideAt(5), 46, "能带闸门通过 ≠ 模型最终通过 [R7]");
appendNotes(
  slideAt(5),
  [refs.R7, refs.R9],
  [
    "Silva等人的Wannier-HHG研究说明，相近能带并不自动保证相近HHG；偶极矩阵、长程跃迁和轨道空间同样重要。",
    "因此先强调pilot/opt1已经充分优化，再说明M40失败来自Hilbert子空间缺少I-p，而不是参数没有调好。",
  ],
);

// Slide 6: d-only failure.
setText(slideAt(6), 46, "右图表现与结论 [R5,R7,R9]");
appendNotes(
  slideAt(6),
  [refs.R5, refs.R7, refs.R9],
  [
    "左图证明价带顶几乎不是Cr-d；右图则证明这个波函数缺失已传递到偶极矩阵与HHG。",
    "Hawkins说明更高导带可实质改变强场电流；Silva说明能带拟合不能替代光学算符验收。",
  ],
);

// Slide 7: orbital-ablation models.
setText(
  slideAt(7),
  63,
  "物理依据：界面 pz 捕获约 70–72% 层间通道；px/py 捕获约 81% 面内 d-p 通道。[R9]",
);
appendNotes(
  slideAt(7),
  [refs.R9],
  [
    "M52/M64/M88不是凑带数，而是围绕CrI3配体轨道做材料内消融。",
    "Soriano等人的多轨道交换研究支持I-p参与磁性机制；本项目进一步用HHG数据区分px/py与pz的作用。",
  ],
);

// Slide 9: ordinary HHG convergence.
setText(
  slideAt(9),
  28,
  "动力学闸门 1 · 普通 HHG 必须逐阶验收 [R5,R6,R8]",
  { frame: frame(70, 42, 790, 28) },
);
appendNotes(
  slideAt(9),
  [refs.R5, refs.R6, refs.R8],
  [
    "文献不支持跨材料预设统一30带上限；所需带数依赖场参数、频段和所用规范。",
    "本页的逐谐波比值是把这一方法学要求落实到CrI3，而不是只看一张能带图。",
  ],
);

// Slides 10–11: magnetic amplitude and phase.
setText(
  slideAt(10),
  28,
  "磁序奇通道 1/2 · Jₒ 是什么，幅值比怎样读 [R8]",
  { frame: frame(70, 42, 760, 28) },
);
appendNotes(
  slideAt(10),
  [refs.R8, refs.R10],
  [
    "Jₒ由±N磁畴复电流差分得到，保留的是磁序敏感的复振幅，而不是强度。",
    "这一复量后续还要比较相位，因为Berry connection和跃迁偶极相位会进入非线性发射。",
  ],
);

setText(
  slideAt(11),
  28,
  "磁序奇通道 2/2 · 复相位误差怎样定义 [R8]",
  { frame: frame(70, 35, 800, 24) },
);
appendNotes(
  slideAt(11),
  [refs.R8],
  [
    "Yue与Gaarde展示，忽略Berry connection或跃迁偶极相位会改变谱与选择定则。",
    "因此M88与full112必须比较复相位，不能只看|Jₒ|或强度是否接近。",
  ],
);

// Slides 12–15: S2/P2 methodology.
setText(
  slideAt(12),
  28,
  "S2 截断 1/2 · 固定 parent 后还能删多少导带？[R5,R6,R8]",
  { frame: frame(70, 42, 820, 28) },
);
appendNotes(
  slideAt(12),
  [refs.R5, refs.R6, refs.R8],
  [
    "S2只改变同一parent内的Bloch active space，因此把Wannier模型误差与导带截断误差分开。",
    "长度规范通常比速度规范更耐截断，但并不使24v+6c这样的硬截断自动可靠。",
  ],
);

setText(
  slideAt(13),
  28,
  "S2 截断 2/2 · 为什么裸 30 带失败，而 78 带通过 [R5–R8]",
  { frame: frame(70, 42, 860, 28) },
);
appendNotes(
  slideAt(13),
  [refs.R5, refs.R6, refs.R7, refs.R8],
  [
    "M52-c6的16.27倍伪峰是本项目数据；文献提供的是风险机制：被删导带、偶极矩阵和规范求和可能破坏相消。",
    "结论只否定裸24v+6c；若导师坚持≤30带，需要downfolding并同步重整化有效偶极/电流算符。",
  ],
);

setText(
  slideAt(14),
  38,
  "P2 最终验收 1/2 · 带截断复误差怎样定义 [R6,R8]",
  { frame: frame(70, 42, 800, 28) },
);
appendNotes(
  slideAt(14),
  [refs.R6, refs.R8],
  [
    "P2比较的是复数距离，因此振幅和相位同时进入误差；这比只比较强度更严格。",
    "H10的46%复误差来自振幅与约0.43 rad相位偏差共同作用，不能被“信号弱”掩盖。",
  ],
);

appendNotes(
  slideAt(15),
  [refs.R8],
  [
    "P2第二部分固定模型、只比较k40与k60，证明主要目标的误差来自带空间而不是k网格。",
    "因此不引入未经测试的k50：日常用k40，最终抽查用k60。",
  ],
);

// Slide 16: final hierarchy.
setText(
  slideAt(16),
  30,
  "冻结后的统一参数 · 文献护栏 + 本项目收敛 [R1–R9]",
  { frame: frame(70, 42, 850, 28) },
);
appendNotes(
  slideAt(16),
  Object.values(refs).slice(0, 9),
  [
    "文献建立“完整物理子空间作reference、缩减模型按目标响应验收”的方法论。",
    "本项目数据才确定具体分工：full112是真值参考，M88-full是可操作parent，M88-c20是H1–H9/JₒH2–H8生产空间。",
  ],
);

// Slide 17: advisor decisions and literature boundary.
setText(
  slideAt(17),
  63,
  "不是为了“多用带”，而是保留 Cr-d/I-p 杂化与复相位；78 带是当前目标下最低已验证空间。[R1–R9]",
  { fontSize: 17 },
);
appendNotes(
  slideAt(17),
  Object.values(refs).slice(0, 9),
  [
    "请导师拍板的不是“是否永远使用112带”，而是同意分层模型及其明确适用范围。",
    "30带不是被经验否定，而是被本项目的M52-c6动力学结果否定；未来若做≤30，应改用downfolding路线。",
  ],
);

// Slide 18: references appendix.
{
  const slide = slideAt(18);
  setText(slide, 39, "附录 · 多带传播、规范与 Wannier 模型的文献依据");
  setText(
    slide,
    40,
    "论文给出方法学边界；full112、M88-full 与 78 带的具体分工来自本工作 D1–D3",
  );

  setText(slide, 44, "多带实际传播 [R1–R3]", {
    frame: frame(92, 201, 300, 34),
    fontSize: 22,
  });
  setText(slide, 45, "ReS₂ 44 MLWF ｜ BaTiS₃ 106 bands", {
    frame: frame(92, 247, 490, 28),
    fontSize: 15,
  });
  setText(slide, 46, "solid C₆₀：120v + 40c = 160 bands", {
    frame: frame(92, 282, 490, 28),
    fontSize: 15,
  });
  setText(
    slide,
    47,
    "Nat. Commun. 14, 8421；PRB 99, 094313；PRB 101, 081412(R).",
    { frame: frame(92, 315, 490, 26), fontSize: 13 },
  );

  setText(slide, 49, "少带成立的边界 [R4,R5]", {
    frame: frame(666, 201, 320, 34),
    fontSize: 22,
  });
  setText(slide, 50, "Ag：9 个 5s/5p/4d MLWF", {
    frame: frame(666, 247, 490, 28),
    fontSize: 15,
  });
  setText(slide, 51, "强 mid-IR：多导带可显著改变 HHG", {
    frame: frame(666, 282, 490, 28),
    fontSize: 15,
  });
  setText(
    slide,
    52,
    "arXiv:2503.05073；PRA 91, 013405 (2015).",
    { frame: frame(666, 315, 490, 26), fontSize: 13 },
  );

  setText(slide, 54, "有限带与规范 [R6,R8]", {
    frame: frame(92, 406, 320, 34),
    fontSize: 22,
  });
  setText(slide, 55, "截断会破坏规范等价与光学求和", {
    frame: frame(92, 452, 490, 28),
    fontSize: 15,
  });
  setText(slide, 56, "Berry connection / 偶极相位不可省略", {
    frame: frame(92, 487, 490, 28),
    fontSize: 15,
  });
  setText(
    slide,
    57,
    "PRB 96, 195413 (2017)；PRA 101, 053411 (2020).",
    { frame: frame(92, 520, 490, 26), fontSize: 13 },
  );

  setText(slide, 59, "Wannier 与 CrI₃ [R7,R9,R10]", {
    frame: frame(666, 406, 360, 34),
    fontSize: 22,
  });
  setText(slide, 60, "能带拟合 ≠ HHG 算符完备", {
    frame: frame(666, 452, 490, 28),
    fontSize: 15,
  });
  setText(slide, 61, "I-p 参与交换；已有双层 CrI₃ HHG", {
    frame: frame(666, 487, 490, 28),
    fontSize: 15,
  });
  setText(
    slide,
    62,
    "PRB 100, 195201；npj CM 7, 162；EPL 140, 26005.",
    { frame: frame(666, 520, 490, 26), fontSize: 13 },
  );

  setText(
    slide,
    63,
    "完整 DOI 与链接已写入讲者备注。R1–R10 支撑方法边界，D1–D3 决定本项目模型层级。",
    { frame: frame(78, 610, 1120, 42), fontSize: 17 },
  );
  setNotes(
    slide,
    Object.values(refs),
    [
      "本页作为备查，不必在主汇报中逐条念。",
      "若导师追问“别人为什么能用少带”，优先回答：Ag的小空间来自简单原胞的化学完整s+p+d；它不是跨材料的带数上限。",
    ],
  );
}

for (let page = 1; page <= presentation.slides.items.length; page += 1) {
  setPageNumber(slideAt(page), page);
}

const exported = await PresentationFile.exportPptx(presentation);
await exported.save(output);
console.log(
  JSON.stringify(
    { output, slides: presentation.slides.items.length },
    null,
    2,
  ),
);

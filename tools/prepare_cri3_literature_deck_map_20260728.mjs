import fs from "node:fs/promises";
import path from "node:path";

const repo = "C:/Users/26507/Documents/New_SBEs/Quantum-light";
const workspace = path.join(repo, ".tmp_ppt_literature_20260728");
const inspectPath = path.join(
  workspace,
  "template-inspect",
  "template-inspect.ndjson",
);

const lines = (await fs.readFile(inspectPath, "utf8"))
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const editableIds = new Map();
for (const item of lines) {
  if (!Number.isInteger(item.slide)) continue;
  if (!["shape", "textbox"].includes(item.kind)) continue;
  const ids = editableIds.get(item.slide) ?? [];
  ids.push(item.id);
  editableIds.set(item.slide, ids);
}

const sourceSlides = [
  1, 2, 3,
  6, // New literature precedent slide.
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  6, // New references appendix slide.
];

const narrativeRoles = [
  "opening thesis",
  "evidence map",
  "full112 band benchmark",
  "multiband HHG literature precedent",
  "M40 Wannier tuning and band-fit evidence",
  "M40 projectability and HHG failure",
  "orbital-ablation model definitions",
  "static model-score radar explanation",
  "ordinary HHG ratio explanation",
  "Je and Jo definitions plus amplitude ratios",
  "Jo phase-error definition and interpretation",
  "S2 active-space definitions and spectrum overlay",
  "S2 truncation-to-parent ratio and strict-30 result",
  "P2 active-band truncation error definition",
  "P2 k-mesh error definition",
  "frozen model hierarchy",
  "advisor decisions",
  "literature references appendix",
];

const outputSlides = sourceSlides.map((sourceSlide, index) => ({
  outputSlide: index + 1,
  sourceSlide,
  narrativeRole: narrativeRoles[index],
  reuseMode: "duplicate-slide",
  editTargets: [
    {
      action: "rewrite",
      sourceElementIds: editableIds.get(sourceSlide) ?? [],
    },
  ],
}));

const frameMap = {
  outputSlides,
  omittedSourceSlides: [],
};

await fs.writeFile(
  path.join(workspace, "template-frame-map.json"),
  `${JSON.stringify(frameMap, null, 2)}\n`,
  "utf8",
);

await fs.writeFile(
  path.join(workspace, "template-audit.txt"),
  [
    "Source: 16-slide CrI3 Wannier/HHG advisor deck, 1280x720.",
    "Visual system: dark green field, pale-yellow titles, cyan section labels, muted green/yellow/purple/red evidence accents.",
    "Typography: Arial/Calibri inherited from source; titles about 31 pt, body 16-20 pt, page markers 13 pt.",
    "Reusable patterns: four-card model-definition slide (source 6) and existing evidence/result slides.",
    "Insertion contract: duplicate source slide 6 for the multiband-literature evidence slide and the references appendix.",
    "All original 16 source slides remain in the narrative, with page numbers shifted after the literature insertion.",
    "No master/layout restyling. Existing charts and images remain untouched.",
    "",
  ].join("\n"),
  "utf8",
);

await fs.writeFile(
  path.join(workspace, "deviation-log.txt"),
  [
    "Output slide 4: source slide 6 reused as a four-case literature comparison.",
    "Output slide 18: source slide 6 reused as a four-block references appendix.",
    "Output slides 5-18: page markers renumbered after insertion.",
    "Selected existing conclusion text boxes gain compact [R#] citations; no font family or visual language change.",
    "",
  ].join("\n"),
  "utf8",
);

await fs.writeFile(
  path.join(workspace, "source-notes.txt"),
  [
    "Primary external sources used in the deck:",
    "R1 Jimenez-Galan et al., Nature Communications 14, 8421 (2023), ReS2 44-Wannier-orbital HHG.",
    "R2 Zhang and Bai, Physical Review B 99, 094313 (2019), BaTiS3 106-band Liouville propagation.",
    "R3 Zhang and Bai, Physical Review B 101, 081412(R) (2020), solid C60 160-band propagation.",
    "R4 Gholam-Mirzaei et al., arXiv:2503.05073 (2025), Ag 9-Wannier-orbital s+p+d model (preprint).",
    "R5 Hawkins, Ivanov, and Yakovlev, Physical Review A 91, 013405 (2015), multiple conduction bands in HHG.",
    "R6 Taghizadeh, Hipolito, and Pedersen, Physical Review B 96, 195413 (2017), basis truncation and gauge dependence.",
    "R7 Silva, Martin, and Ivanov, Physical Review B 100, 195201 (2019), MLWF-based HHG and optical-matrix completeness.",
    "R8 Yue and Gaarde, Physical Review A 101, 053411 (2020), structure/laser gauges, Berry connection and dipole phases.",
    "",
    "Internal evidence:",
    "D1 docs/08_QE与Wannier/CrI3_40WF审计第一轮_20260714.md",
    "D2 docs/09_少带模型/给场外AI_S2截断与E0诊断全量报告_20260720.md",
    "D3 docs/09_少带模型/P2_k60_验收报告_20260721.md",
    "",
  ].join("\n"),
  "utf8",
);

console.log(
  JSON.stringify(
    {
      map: path.join(workspace, "template-frame-map.json"),
      slides: outputSlides.length,
    },
    null,
    2,
  ),
);

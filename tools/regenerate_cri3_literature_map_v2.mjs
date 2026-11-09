import fs from "node:fs/promises";
import path from "node:path";

const repo = "C:/Users/26507/Documents/New_SBEs/Quantum-light";
const workspace = path.join(repo, ".tmp_ppt_literature_20260728");
const editableIds = new Map();

for (let slideNumber = 1; slideNumber <= 16; slideNumber += 1) {
  const layoutPath = path.join(
    workspace,
    "template-inspect",
    "layouts",
    `source-slide-${String(slideNumber).padStart(2, "0")}.layout.json`,
  );
  const layout = JSON.parse(await fs.readFile(layoutPath, "utf8"));
  editableIds.set(
    slideNumber,
    layout.elements
      .filter((item) => ["shape", "textbox"].includes(item.kind))
      .map((item) => item.aid),
  );
}

const sourceSlides = [
  1, 2, 3,
  6,
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  6,
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
      sourceElementIds: editableIds.get(sourceSlide),
    },
  ],
}));

const frameMap = { outputSlides, omittedSourceSlides: [] };
const outputPath = path.join(workspace, "template-frame-map.json");
await fs.writeFile(outputPath, `${JSON.stringify(frameMap, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ outputPath, slides: outputSlides.length }, null, 2));

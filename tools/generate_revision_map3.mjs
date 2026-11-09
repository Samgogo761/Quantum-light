import fs from "node:fs";
import path from "node:path";

const workspace = "C:/Users/26507/Documents/New_SBEs/Quantum-light/.tmp_ppt_revision_20260728";
const layoutDir = path.join(workspace, "template-inspect", "layouts");

const plans = [
  { outputSlide: 1, sourceSlide: 1, narrativeRole: "opening thesis" },
  { outputSlide: 2, sourceSlide: 2, narrativeRole: "evidence map" },
  { outputSlide: 3, sourceSlide: 3, narrativeRole: "full112 band benchmark", keepImages: true },
  { outputSlide: 4, sourceSlide: 4, narrativeRole: "M40 Wannier tuning and band-fit evidence", addZones: [{ left: 76, top: 181, width: 1128, height: 295 }] },
  { outputSlide: 5, sourceSlide: 4, narrativeRole: "M40 projectability and HHG failure", addZones: [{ left: 92, top: 188, width: 505, height: 284 }, { left: 680, top: 188, width: 510, height: 284 }] },
  { outputSlide: 6, sourceSlide: 5, narrativeRole: "orbital-ablation model definitions" },
  { outputSlide: 7, sourceSlide: 6, narrativeRole: "static model-score radar explanation", addZones: [{ left: 82, top: 184, width: 565, height: 330 }] },
  { outputSlide: 8, sourceSlide: 6, narrativeRole: "ordinary HHG ratio explanation", addZones: [{ left: 72, top: 205, width: 706, height: 270 }] },
  { outputSlide: 9, sourceSlide: 7, narrativeRole: "Je and Jo definitions plus amplitude ratios", addZones: [{ left: 70, top: 218, width: 680, height: 260 }] },
  { outputSlide: 10, sourceSlide: 7, narrativeRole: "Jo phase-error definition and interpretation", addZones: [{ left: 70, top: 214, width: 720, height: 268 }] },
  { outputSlide: 11, sourceSlide: 8, narrativeRole: "S2 active-space definitions and spectrum overlay", addZones: [{ left: 70, top: 220, width: 700, height: 270 }] },
  { outputSlide: 12, sourceSlide: 8, narrativeRole: "S2 truncation-to-parent ratio and strict-30 result", addZones: [{ left: 70, top: 214, width: 700, height: 280 }] },
  { outputSlide: 13, sourceSlide: 9, narrativeRole: "P2 active-band truncation error definition", addZones: [{ left: 70, top: 214, width: 670, height: 265 }] },
  { outputSlide: 14, sourceSlide: 9, narrativeRole: "P2 k-mesh error definition", addZones: [{ left: 70, top: 214, width: 670, height: 265 }] },
  { outputSlide: 15, sourceSlide: 10, narrativeRole: "frozen model hierarchy" },
  { outputSlide: 16, sourceSlide: 11, narrativeRole: "advisor decisions" },
];

function readLayout(slide) {
  return JSON.parse(
    fs.readFileSync(
      path.join(layoutDir, `source-slide-${String(slide).padStart(2, "0")}.layout.json`),
      "utf8",
    ),
  );
}

function targets(plan) {
  const layout = readLayout(plan.sourceSlide);
  const shapeIds = layout.elements.filter((e) => e.kind === "shape").map((e) => e.aid);
  const imageIds = layout.elements.filter((e) => e.kind === "image").map((e) => e.aid);
  const editTargets = [{ action: "rewrite", sourceElementIds: shapeIds }];
  if (imageIds.length) {
    editTargets.push({
      action: plan.keepImages ? "replace" : "delete",
      sourceElementIds: imageIds,
    });
  }
  for (const zone of plan.addZones ?? []) {
    editTargets.push({
      action: "add",
      newPrimitiveAllowed: true,
      mustNotOverlapInherited: true,
      zone,
      reason:
        "Insert an independent image asset inside the inherited evidence frame after deleting the inherited image, preventing shared-media mutation between duplicated slides.",
    });
  }
  return editTargets;
}

const map = {
  outputSlides: plans.map((plan) => ({
    outputSlide: plan.outputSlide,
    sourceSlide: plan.sourceSlide,
    narrativeRole: plan.narrativeRole,
    reuseMode: "duplicate-slide",
    editTargets: targets(plan),
  })),
  omittedSourceSlides: [],
};

fs.writeFileSync(path.join(workspace, "template-frame-map.json"), JSON.stringify(map, null, 2), "utf8");
console.log(JSON.stringify({ slides: map.outputSlides.length }, null, 2));

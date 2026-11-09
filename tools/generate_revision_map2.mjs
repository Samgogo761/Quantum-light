import fs from "node:fs";
import path from "node:path";

const workspace = "C:/Users/26507/Documents/New_SBEs/Quantum-light/.tmp_ppt_revision_20260728";
const layoutDir = path.join(workspace, "template-inspect", "layouts");

const plans = [
  { outputSlide: 1, sourceSlide: 1, narrativeRole: "opening thesis" },
  { outputSlide: 2, sourceSlide: 2, narrativeRole: "evidence map" },
  { outputSlide: 3, sourceSlide: 3, narrativeRole: "full112 band benchmark", images: { left: "replace", right: "replace" } },
  { outputSlide: 4, sourceSlide: 4, narrativeRole: "M40 Wannier tuning and band-fit evidence", images: { left: "replace", right: "delete" } },
  { outputSlide: 5, sourceSlide: 4, narrativeRole: "M40 projectability and HHG failure", images: { left: "replace", right: "replace" } },
  { outputSlide: 6, sourceSlide: 5, narrativeRole: "orbital-ablation model definitions" },
  { outputSlide: 7, sourceSlide: 6, narrativeRole: "static model-score radar explanation", images: { left: "replace", right: "delete" } },
  { outputSlide: 8, sourceSlide: 6, narrativeRole: "ordinary HHG ratio explanation", images: { left: "delete", right: "replace" } },
  { outputSlide: 9, sourceSlide: 7, narrativeRole: "Je and Jo definitions plus amplitude ratios", images: { left: "replace", right: "delete" } },
  { outputSlide: 10, sourceSlide: 7, narrativeRole: "Jo phase-error definition and interpretation", images: { left: "delete", right: "replace" } },
  { outputSlide: 11, sourceSlide: 8, narrativeRole: "S2 active-space definitions and spectrum overlay", images: { left: "replace", right: "delete" } },
  { outputSlide: 12, sourceSlide: 8, narrativeRole: "S2 truncation-to-parent ratio and strict-30 result", images: { left: "delete", right: "replace" } },
  { outputSlide: 13, sourceSlide: 9, narrativeRole: "P2 active-band truncation error definition", images: { left: "replace", right: "delete" } },
  { outputSlide: 14, sourceSlide: 9, narrativeRole: "P2 k-mesh error definition", images: { left: "delete", right: "replace" } },
  { outputSlide: 15, sourceSlide: 10, narrativeRole: "frozen model hierarchy" },
  { outputSlide: 16, sourceSlide: 11, narrativeRole: "advisor decisions" },
];

function readLayout(slide) {
  const p = path.join(layoutDir, `source-slide-${String(slide).padStart(2, "0")}.layout.json`);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function buildTargets(plan) {
  const layout = readLayout(plan.sourceSlide);
  const shapes = layout.elements.filter((e) => e.kind === "shape").map((e) => e.aid);
  const images = layout.elements.filter((e) => e.kind === "image");
  const targets = [];
  if (shapes.length) targets.push({ action: "rewrite", sourceElementIds: shapes });
  if (images.length) {
    const actions = plan.images ?? {};
    const left = images[0];
    const right = images[1];
    if (left) targets.push({ action: actions.left ?? "replace", sourceElementIds: [left.aid] });
    if (right) targets.push({ action: actions.right ?? "replace", sourceElementIds: [right.aid] });
  }
  return targets;
}

const map = {
  outputSlides: plans.map((plan) => ({
    outputSlide: plan.outputSlide,
    sourceSlide: plan.sourceSlide,
    narrativeRole: plan.narrativeRole,
    reuseMode: "duplicate-slide",
    editTargets: buildTargets(plan),
  })),
  omittedSourceSlides: [],
};

fs.writeFileSync(path.join(workspace, "template-frame-map.json"), JSON.stringify(map, null, 2), "utf8");
console.log(JSON.stringify({ slides: map.outputSlides.length }, null, 2));

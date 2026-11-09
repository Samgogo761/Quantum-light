import fs from "node:fs/promises";

const mapPath =
  "C:/Users/26507/Documents/New_SBEs/Quantum-light/.tmp_ppt_revision_20260728/template-frame-map.json";
const map = JSON.parse(await fs.readFile(mapPath, "utf8"));

const formulaZones = new Map([
  [
    10,
    {
      left: 850,
      top: 214,
      width: 350,
      height: 42,
      reason: "Add a readable phase-error formula textbox after deleting the inherited hidden/white formula shape.",
    },
  ],
  [
    14,
    {
      left: 812,
      top: 204,
      width: 374,
      height: 284,
      reason: "Add a readable k-mesh-error formula and symbol-definition textbox inside the inherited right explanation panel.",
    },
  ],
]);

for (const slide of map.outputSlides) {
  const item = formulaZones.get(slide.outputSlide);
  if (!item) continue;
  const { reason, ...zone } = item;
  slide.editTargets.push({
    action: "add",
    newPrimitiveAllowed: true,
    mustNotOverlapInherited: false,
    zone,
    reason,
  });
}

await fs.writeFile(mapPath, JSON.stringify(map, null, 2), "utf8");
console.log(mapPath);

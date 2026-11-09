import fs from "node:fs";

const input = "C:/Users/26507/Documents/New_SBEs/Quantum-light/tools/revise_cri3_wannier_deck_20260728.mjs";
const output = "C:/Users/26507/Documents/New_SBEs/Quantum-light/tools/revise_cri3_wannier_deck_20260728_v2.mjs";
let source = fs.readFileSync(input, "utf8");

const helper = `
async function addIndependentImage(slide, inheritedName, relativePath, alt, targetFrame) {
  imageByName(slide, inheritedName).delete();
  const ext = path.extname(relativePath).toLowerCase();
  const contentType =
    ext === ".svg"
      ? "image/svg+xml"
      : ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : "image/png";
  return slide.images.add({
    blob: await fs.readFile(path.join(repo, relativePath)),
    contentType,
    alt,
    fit: "contain",
    position: targetFrame,
    crop: { left: 0, top: 0, right: 0, bottom: 0 },
  });
}

`;

source = source.replace("function deleteImage(slide, name) {", `${helper}function deleteImage(slide, name) {`);
source = source.replaceAll("await replaceImage(", "await addIndependentImage(");
fs.writeFileSync(output, source, "utf8");
console.log(output);

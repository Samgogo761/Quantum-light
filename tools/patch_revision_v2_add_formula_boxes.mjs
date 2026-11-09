import fs from "node:fs/promises";

const file =
  "C:/Users/26507/Documents/New_SBEs/Quantum-light/tools/revise_cri3_wannier_deck_20260728_v2.mjs";

let source = await fs.readFile(file, "utf8");

source = source.replace(
  "function deleteImage(slide, name) {\n  imageByName(slide, name).delete();\n}\n",
  [
    "function deleteImage(slide, name) {",
    "  imageByName(slide, name).delete();",
    "}",
    "",
    "function addTextBox(slide, value, targetFrame, options = {}) {",
    "  const shape = slide.shapes.add({",
    '    geometry: "textbox",',
    "    position: targetFrame,",
    '    fill: "none",',
    '    line: { style: "solid", fill: "none", width: 0 },',
    "  });",
    "  shape.text = value;",
    "  shape.text.style = {",
    "    fontSize: options.fontSize ?? 18,",
    "    bold: options.bold ?? false,",
    '    color: options.color ?? "#263B33",',
    '    alignment: options.alignment ?? "left",',
    "  };",
    "  return shape;",
    "}",
    "",
  ].join("\n"),
);

source = source.replace(
  '  setText(\n    slide,\n    "矩形 33",\n    "Δφₒ,q^(M/112)\\n= arg[Jₒ,q^M (Jₒ,q^112)*]",\n    { frame: frame(850, 198, 350, 75), fontSize: 20, color: "#263B33", bold: true },\n  );',
  [
    '  shapeByName(slide, "矩形 33").delete();',
    "  addTextBox(",
    "    slide,",
    '    "Δφₒ,q^(M/112) = arg[Jₒ,q^M (Jₒ,q^112)*]",',
    "    frame(850, 214, 350, 42),",
    '    { fontSize: 19, color: "#263B33", bold: true, alignment: "center" },',
    "  );",
  ].join("\n"),
);

source = source.replace(
  '  setText(\n    slide,\n    "矩形 43",\n    "ε_k^X(q;M)=|J_X,q^(M,k60)−J_X,q^(M,k40)|\\n                 /(|J_X,q^(M,k60)|+δ)\\n\\nX=J：普通电流；X=o：磁序奇通道\\nM：M88-full 或 M88-c20\\n下标 k：误差来自 k-mesh；q：谐波阶数\\nδ>0：防止极弱参考信号使比值发散",\n    { frame: frame(805, 195, 390, 305), fontSize: 17, color: "#263B33" },\n  );',
  [
    '  shapeByName(slide, "矩形 43").delete();',
    "  addTextBox(",
    "    slide,",
    '    "ε_k^X(q;M) = |J_X,q^(M,k60) − J_X,q^(M,k40)|\\n/(|J_X,q^(M,k60)| + δ)\\n\\nX=J：普通总电流；X=o：磁序奇通道\\nM：M88-full 或 M88-c20\\n下标 k：误差来自 k 网格；q：谐波阶数\\nδ>0：数值底，防止弱信号比值发散",',
    "    frame(812, 204, 374, 284),",
    '    { fontSize: 16, color: "#263B33", alignment: "left" },',
    "  );",
  ].join("\n"),
);

await fs.writeFile(file, source, "utf8");
console.log(file);

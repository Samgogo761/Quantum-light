import fs from "node:fs/promises";

const file =
  "C:/Users/26507/Documents/New_SBEs/Quantum-light/tools/revise_cri3_wannier_deck_20260728_v2.mjs";

let source = await fs.readFile(file, "utf8");

source = source.replace(
  '  if (options.fontSize) shape.text.style = { fontSize: options.fontSize };',
  [
    "  const style = {};",
    "  if (options.fontSize) style.fontSize = options.fontSize;",
    "  if (options.color) style.color = options.color;",
    "  if (options.bold !== undefined) style.bold = options.bold;",
    "  if (Object.keys(style).length) shape.text.style = style;",
  ].join("\n"),
);

source = source.replace(
  '    frame(82, 184, 565, 330),',
  '    frame(82, 184, 565, 300),',
);
source = source.replace(
  '{ frame: frame(72, 501, 586, 24), fontSize: 15 },',
  '{ frame: frame(72, 496, 586, 28), fontSize: 15 },',
);

source = source.replace(
  '  setText(slide, "矩形 27", "磁序奇通道 2/2 · 复相位误差怎样定义");\n' +
    '  setText(slide, "矩形 28", "量子关联和模式干涉依赖复相位；只比较 |Jₒ| 或强度不够");',
  '  setText(slide, "矩形 27", "磁序奇通道 2/2 · 复相位误差怎样定义", { frame: frame(70, 35, 720, 24) });\n' +
    '  setText(slide, "矩形 28", "量子关联和模式干涉依赖复相位；只比较 |Jₒ| 或强度不够", { frame: frame(70, 72, 1120, 62) });',
);

source = source.replace(
  '{ frame: frame(850, 198, 350, 75), fontSize: 20 },',
  '{ frame: frame(850, 198, 350, 75), fontSize: 20, color: "#263B33", bold: true },',
);
source = source.replace(
  '{ frame: frame(805, 195, 390, 305), fontSize: 17 },',
  '{ frame: frame(805, 195, 390, 305), fontSize: 17, color: "#263B33" },',
);

await fs.writeFile(file, source, "utf8");
console.log(file);

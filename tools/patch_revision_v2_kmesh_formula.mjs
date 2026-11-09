import fs from "node:fs/promises";

const file =
  "C:/Users/26507/Documents/New_SBEs/Quantum-light/tools/revise_cri3_wannier_deck_20260728_v2.mjs";

let source = await fs.readFile(file, "utf8");
const anchor = '  setText(slide, "矩形 48", "k40");';

if (!source.includes("K_MESH_FORMULA_BOX")) {
  source = source.replace(
    anchor,
    [
      "  // K_MESH_FORMULA_BOX",
      '  shapeByName(slide, "矩形 43").delete();',
      "  addTextBox(",
      "    slide,",
      '    "ε_k^X(q;M) = |J_X,q^(M,k60) − J_X,q^(M,k40)|\\n/(|J_X,q^(M,k60)| + δ)\\n\\nX=J：普通总电流；X=o：磁序奇通道\\nM：M88-full 或 M88-c20\\n下标 k：误差来自 k 网格；q：谐波阶数\\nδ>0：数值底，防止弱信号比值发散",',
      "    frame(812, 204, 374, 284),",
      '    { fontSize: 16, color: "#263B33", alignment: "left" },',
      "  );",
      anchor,
    ].join("\n"),
  );
}

await fs.writeFile(file, source, "utf8");
console.log(file);

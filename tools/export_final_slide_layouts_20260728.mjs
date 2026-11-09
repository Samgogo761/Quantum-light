import fs from "node:fs/promises";
import path from "node:path";
import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
} from "file:///C:/Users/26507/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/artifact_tool_utils.mjs";

const repo = "C:/Users/26507/Documents/New_SBEs/Quantum-light";
const workspace = path.join(repo, ".tmp_ppt_revision_20260728");
const input = path.join(workspace, "final_ascii.pptx");
const outputDir = path.join(workspace, "final-layout-v5");

await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const presentation = await PresentationFile.importPptx(await FileBlob.load(input));

await fs.mkdir(outputDir, { recursive: true });
for (let i = 0; i < presentation.slides.items.length; i += 1) {
  const slide = presentation.slides.items[i];
  const layout = await presentation.export({ slide, format: "layout" });
  const padded = String(i + 1).padStart(2, "0");
  await fs.writeFile(
    path.join(outputDir, `final-slide-${padded}.layout.json`),
    JSON.stringify(layout, null, 2),
    "utf8",
  );
}

console.log(JSON.stringify({ slides: presentation.slides.items.length, outputDir }, null, 2));

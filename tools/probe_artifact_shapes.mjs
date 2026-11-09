import {
  ensureArtifactToolWorkspace,
  importArtifactTool,
} from "file:///C:/Users/26507/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/artifact_tool_utils.mjs";

const workspace = "C:/Users/26507/Documents/New_SBEs/Quantum-light/.tmp_ppt_revision_20260728";
await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const p = await PresentationFile.importPptx(
  await FileBlob.load(`${workspace}/template-starter.pptx`),
);
const slide = p.slides.items[6];
const shape = slide.shapes.items.find((x) => x.name === "矩形 39");
const image = slide.images.items[0];
console.log(JSON.stringify({
  shapeKeys: Object.keys(shape),
  shapeFrame: shape.frame,
  shapePosition: shape.position,
  textKeys: Object.keys(shape.text),
  textStyle: shape.text.style,
  imageKeys: Object.keys(image),
  imageFrame: image.frame,
  imagePosition: image.position,
}, null, 2));

import { pathToFileURL } from "node:url";

const artifactPath = "C:/Users/yzska/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { FileBlob, PresentationFile } = await import(pathToFileURL(artifactPath).href);

const source = "E:/SSD/Data/program/python/DRL_FinalProjectProposal/outputs/manual-20260607-ppo-smc-v2/presentations/ppo-smc-report/output/smc-ppo-trading-report-v2.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
console.log({
  count: presentation.slides.count,
  slideKeys: Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides)),
  presentationKeys: Object.getOwnPropertyNames(Object.getPrototypeOf(presentation)),
});

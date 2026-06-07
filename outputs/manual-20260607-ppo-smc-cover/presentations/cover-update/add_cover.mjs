import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const artifactPath = "C:/Users/yzska/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { FileBlob, PresentationFile } = await import(pathToFileURL(artifactPath).href);

const source = "E:/SSD/Data/program/python/DRL_FinalProjectProposal/outputs/manual-20260607-ppo-smc-v2/presentations/ppo-smc-report/output/smc-ppo-trading-report-v2.pptx";
const output = "E:/SSD/Data/program/python/DRL_FinalProjectProposal/outputs/manual-20260607-ppo-smc-cover/presentations/cover-update/output/smc-ppo-trading-report-v3-cover.pptx";
const preview = "E:/SSD/Data/program/python/DRL_FinalProjectProposal/outputs/manual-20260607-ppo-smc-cover/presentations/cover-update/preview/cover.png";

const COLORS = {
  bg: "#F7F8F4",
  ink: "#071018",
  muted: "#4B5C68",
  navy: "#123A56",
  teal: "#356B85",
  pale: "#D8E6E8",
  pale2: "#EEF4F3",
  orange: "#FF725F",
  green: "#2E8B57",
  line: "#C4D1D5",
  transparent: "#00000000",
};

function line(fill = COLORS.transparent, width = 0, style = "solid") {
  return { style, fill, width };
}

function shape(slide, x, y, w, h, fill, stroke = COLORS.transparent, geometry = "rect") {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line(stroke, stroke === COLORS.transparent ? 0 : 1),
  });
}

function text(slide, value, x, y, w, h, opts = {}) {
  const s = shape(slide, x, y, w, h, opts.fill ?? COLORS.transparent, opts.stroke ?? COLORS.transparent);
  s.text = value;
  s.text.fontSize = opts.size ?? 24;
  s.text.color = opts.color ?? COLORS.ink;
  s.text.bold = Boolean(opts.bold);
  s.text.typeface = opts.mono ? "Cascadia Mono" : "Microsoft JhengHei";
  s.text.alignment = opts.align ?? "left";
  s.text.verticalAlignment = opts.valign ?? "top";
  s.text.insets = opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 };
  return s;
}

async function saveBlob(blob, filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const slide = presentation.slides.add();
slide.moveTo(0);

shape(slide, 0, 0, 1280, 720, COLORS.bg);
shape(slide, 72, 54, 34, 6, COLORS.orange);
text(slide, "FINAL PROJECT REPORT", 118, 42, 420, 28, {
  size: 16,
  bold: true,
  color: COLORS.teal,
  insets: { left: 0, right: 0, top: 2, bottom: 0 },
});

text(slide, "SMC + PPO 多股票動態資金配置系統", 72, 132, 1040, 76, {
  size: 44,
  bold: true,
  color: COLORS.ink,
});
text(slide, "Smart Money Concept × Proximal Policy Optimization", 76, 220, 760, 36, {
  size: 24,
  color: COLORS.muted,
});

shape(slide, 72, 305, 1136, 4, COLORS.navy);
shape(slide, 72, 360, 335, 132, COLORS.pale2, COLORS.line);
shape(slide, 472, 360, 335, 132, "#FFFFFF", COLORS.line);
shape(slide, 872, 360, 335, 132, COLORS.pale2, COLORS.line);

text(slide, "組員", 100, 390, 130, 28, { size: 22, bold: true, color: COLORS.navy });
text(slide, "游宗勝\n李承育\n史福隆", 100, 426, 250, 58, { size: 20, color: COLORS.ink });

text(slide, "課堂授課教授", 500, 390, 180, 28, { size: 22, bold: true, color: COLORS.navy });
text(slide, "陳煥 老師", 500, 432, 230, 32, { size: 24, bold: true, color: COLORS.orange });

text(slide, "報告主題", 900, 390, 160, 28, { size: 22, bold: true, color: COLORS.navy });
text(slide, "以 SMC 特徵建立 PPO 交易環境，驗證 Pair / Basket 台股資金配置策略。", 900, 426, 245, 48, {
  size: 16,
  color: COLORS.muted,
});

text(slide, "Demo: Streamlit App  |  Video: YouTube", 72, 596, 680, 28, {
  size: 18,
  color: COLORS.teal,
  bold: true,
});
shape(slide, 72, 676, 1136, 1, COLORS.line);
text(slide, "DRL Final Project Proposal", 72, 686, 420, 18, { size: 11, color: COLORS.muted });
text(slide, "01", 1160, 684, 48, 20, { size: 12, bold: true, color: COLORS.navy, mono: true, align: "right" });

await fs.mkdir(path.dirname(output), { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);

const coverPng = await presentation.export({ slide, format: "png", scale: 1 });
await saveBlob(coverPng, preview);

console.log(JSON.stringify({ output, preview, slideCount: presentation.slides.count }, null, 2));

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = path.resolve(process.cwd());
const OUTPUT_DIR = path.join(
  ROOT,
  "outputs",
  "01a01df7-0fae-7d13-a014-5625551d60ff",
);
const PREVIEW_DIR = path.join(OUTPUT_DIR, "previews");

const SOURCES = {
  a10_caid2: {
    archive: "release/b2_caid2_final_results_20260825.tar.gz",
    expectedArchiveSha256:
      "3a398b74f9887b34a1bc530edabedae81da046efa6473f8a30c9f9ff5235d535",
    report:
      "artifacts/b5_locked_sources_20260826/a10_caid2/outputs/analysis/b2_caid2_fair_comparison/formal/b2_report.json",
  },
  a10_caid3: {
    archive: "release/a10_caid3_final_results_20260825.tar.gz",
    expectedArchiveSha256:
      "93b906a45339c918d00d4e670da4c5a792395fe4570b8b45c12b7e4da68f15c8",
    report:
      "artifacts/b5_locked_sources_20260826/b1/outputs/analysis/b1_caid3_fair_comparison/b1_report.json",
  },
  lora: {
    archive: "release/b4_lora_caid23_results_20260826.tar.gz",
    expectedArchiveSha256:
      "2ec8d02222975d26db42956d14a8343c4145933c77839e5ef98feb7c4c986c49",
    reports: {
      CAID2:
        "artifacts/b5_locked_sources_20260826/lora/outputs/analysis/b4_baseline_reproduction/caid2/evaluation/lora_650m_disprot7_caid2_report.json",
      CAID3:
        "artifacts/b5_locked_sources_20260826/lora/outputs/analysis/b4_baseline_reproduction/caid3/evaluation/lora_650m_disprot7_caid3_report.json",
    },
  },
  punch2_light: {
    archive: "release/b4_punch2_light_caid23_results_20260826.tar.gz",
    expectedArchiveSha256:
      "830b88bd0f14843e040c5f4ef5135066cbf1befa8ea353704777eaef06780df3",
    reportRoot:
      "artifacts/b5_locked_sources_20260826/punch2_light/outputs/analysis/b4_baseline_reproduction/punch2_light",
  },
};

const DATASETS = ["CAID2", "CAID3"];
const TRACKS = ["disorder_nox", "disorder_pdb"];
const METHODS = [
  "A10-locked",
  "LoRA-DR-Suite 650M",
  "PUNCH2-Light Paper-8",
  "PUNCH2-Light Released-13",
];
const METHOD_COLORS = {
  "A10-locked": "#1F4E78",
  "LoRA-DR-Suite 650M": "#C55A11",
  "PUNCH2-Light Paper-8": "#70AD47",
  "PUNCH2-Light Released-13": "#8064A2",
};

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function abs(relativePath) {
  return path.join(ROOT, ...relativePath.split("/"));
}

async function sha256File(relativePath) {
  const bytes = await fs.readFile(abs(relativePath));
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

async function readJson(relativePath) {
  return JSON.parse(await fs.readFile(abs(relativePath), "utf8"));
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function parseNdjson(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function metricRow(dataset, track, model, variant, metrics, reportPath, reportSha256) {
  return {
    dataset,
    track,
    model,
    variant,
    roc_auc_caid_compatible: metrics.roc_auc_caid_compatible,
    roc_auc_full_precision: metrics.roc_auc_full_precision,
    aucpr_trapezoid: metrics.aucpr_trapezoid,
    aps: metrics.aps,
    f1_at_0_5: metrics.f1_at_0_5,
    mcc_at_0_5: metrics.mcc_at_0_5,
    fmax: metrics.fmax,
    mcc_at_fmax: metrics.mcc_at_fmax,
    fmax_threshold: metrics.fmax_threshold,
    reference_proteins: metrics.reference_proteins,
    predicted_proteins: metrics.predicted_proteins,
    protein_coverage: metrics.protein_coverage,
    reference_residues: metrics.reference_residues,
    predicted_residues: metrics.predicted_residues,
    residue_coverage: metrics.residue_coverage,
    known_residues: metrics.known_residues,
    positive_labels: metrics.positive_labels,
    negative_labels: metrics.negative_labels,
    source_report: reportPath.replaceAll("\\", "/"),
    source_report_sha256: reportSha256,
  };
}

async function verifyArchiveSource(sourceName, source) {
  const actual = await sha256File(source.archive);
  assert(
    actual === source.expectedArchiveSha256,
    `${sourceName} archive SHA256 mismatch: ${actual}`,
  );
  return {
    archive: source.archive,
    expected_sha256: source.expectedArchiveSha256,
    actual_sha256: actual,
    match: true,
  };
}

async function loadRows() {
  const archives = {};
  for (const [name, source] of Object.entries(SOURCES)) {
    archives[name] = await verifyArchiveSource(name, source);
  }

  const rows = [];

  const b2 = await readJson(SOURCES.a10_caid2.report);
  assert(b2.status === "pass", "B2 report did not pass");
  assert(b2.a10_locked_before_caid2_label_access === true, "A10 was not locked before CAID2 labels");
  assert(b2.caid2_labels_used_for_training_or_tuning === false, "CAID2 labels contaminated model selection");
  const b2Sha = await sha256File(SOURCES.a10_caid2.report);
  for (const track of TRACKS) {
    const metric = b2.metrics.find(
      (row) => row.method === "A10-locked" && row.track === track,
    );
    assert(metric, `missing A10 CAID2 ${track}`);
    rows.push(metricRow("CAID2", track, METHODS[0], "30-model uniform logit mean", metric, SOURCES.a10_caid2.report, b2Sha));
  }

  const b1 = await readJson(SOURCES.a10_caid3.report);
  assert(b1.status === "pass", "B1 report did not pass");
  assert(b1.a10_locked_before_caid_label_access === true, "A10 was not locked before CAID3 labels");
  assert(b1.caid_labels_used_for_training_or_tuning === false, "CAID3 labels contaminated model selection");
  const b1Sha = await sha256File(SOURCES.a10_caid3.report);
  for (const track of TRACKS) {
    const metric = b1.metrics.find(
      (row) => row.method === "A10-locked" && row.track === track,
    );
    assert(metric, `missing A10 CAID3 ${track}`);
    rows.push(metricRow("CAID3", track, METHODS[0], "30-model uniform logit mean", metric, SOURCES.a10_caid3.report, b1Sha));
  }

  for (const dataset of DATASETS) {
    const reportPath = SOURCES.lora.reports[dataset];
    const report = await readJson(reportPath);
    assert(report.status === "pass", `${dataset} LoRA report did not pass`);
    assert(report.model_locked_before_label_access === true, `${dataset} LoRA model was not locked`);
    assert(report.caid_labels_used_for_training_or_tuning === false, `${dataset} LoRA label contamination`);
    const reportSha = await sha256File(reportPath);
    for (const track of TRACKS) {
      rows.push(metricRow(dataset, track, METHODS[1], "CQSB/esm2_650M-LoRA-ID-DisProt7", report.tracks[track], reportPath, reportSha));
    }
  }

  for (const dataset of DATASETS) {
    const datasetLower = dataset.toLowerCase();
    for (const [variant, model] of [
      ["paper8", METHODS[2]],
      ["released13", METHODS[3]],
    ]) {
      const reportPath = `${SOURCES.punch2_light.reportRoot}/${datasetLower}/evaluation/punch2_light_${variant}_${datasetLower}_report.json`;
      const report = await readJson(reportPath);
      assert(report.status === "pass", `${dataset} ${variant} PUNCH2-Light report did not pass`);
      assert(report.model_selection_completed_before_label_access === true, `${dataset} ${variant} was not locked`);
      assert(report.caid_labels_used_for_training_or_tuning === false, `${dataset} ${variant} label contamination`);
      assert(report.prediction_lock?.all_entries_match === true, `${dataset} ${variant} prediction lock mismatch`);
      assert(report.prediction_lock?.predictions_covered === true, `${dataset} ${variant} predictions not covered by lock`);
      const reportSha = await sha256File(reportPath);
      for (const track of TRACKS) {
        rows.push(metricRow(dataset, track, model, variant, report.tracks[track], reportPath, reportSha));
      }
    }
  }

  const ordered = [];
  for (const dataset of DATASETS) {
    for (const track of TRACKS) {
      const group = rows.filter((row) => row.dataset === dataset && row.track === track);
      assert(group.length === 4, `${dataset} ${track} expected four models, found ${group.length}`);
      const baseline = group[0];
      for (const row of group) {
        for (const key of [
          "reference_proteins",
          "reference_residues",
          "known_residues",
          "positive_labels",
          "negative_labels",
        ]) {
          assert(row[key] === baseline[key], `${dataset} ${track} ${key} mismatch for ${row.model}`);
        }
        assert(row.protein_coverage === 1, `${dataset} ${track} incomplete protein coverage for ${row.model}`);
        assert(row.residue_coverage === 1, `${dataset} ${track} incomplete residue coverage for ${row.model}`);
      }
      for (const model of METHODS) {
        ordered.push(group.find((row) => row.model === model));
      }
    }
  }
  assert(ordered.every(Boolean), "ordered metric matrix is incomplete");

  return { rows: ordered, archives };
}

function winnerFor(rows, dataset, track, metric) {
  const group = rows.filter((row) => row.dataset === dataset && row.track === track);
  return group.reduce((best, row) => (row[metric] > best[metric] ? row : best));
}

function markdownReport(rows) {
  const columns = [
    ["roc_auc_caid_compatible", "ROC-AUC"],
    ["aucpr_trapezoid", "AUPRC"],
    ["aps", "APS"],
    ["f1_at_0_5", "F1@0.5"],
    ["mcc_at_0_5", "MCC@0.5"],
    ["fmax", "Fmax"],
  ];
  const out = [
    "# B5 locked external comparison",
    "",
    "All models were locked before CAID2/CAID3 label access. Values are post-lock external-test results with 100% protein and residue coverage.",
    "",
  ];
  for (const dataset of DATASETS) {
    for (const track of TRACKS) {
      out.push(`## ${dataset} ${track}`);
      out.push("");
      out.push(`| Model | ${columns.map(([, label]) => label).join(" | ")} |`);
      out.push(`| --- | ${columns.map(() => "---:").join(" | ")} |`);
      const group = rows.filter((row) => row.dataset === dataset && row.track === track);
      const maxima = Object.fromEntries(
        columns.map(([key]) => [key, Math.max(...group.map((row) => row[key]))]),
      );
      for (const row of group) {
        const cells = columns.map(([key]) => {
          const value = Number(row[key]).toFixed(3);
          return row[key] === maxima[key] ? `**${value}**` : value;
        });
        out.push(`| ${row.model} | ${cells.join(" | ")} |`);
      }
      out.push("");
    }
  }
  out.push("## Interpretation");
  out.push("");
  out.push("- LoRA-DR-Suite 650M has the highest ROC-AUC on both NOX tracks.");
  out.push("- A10-locked has the highest ROC-AUC on both PDB tracks and exceeds PUNCH2-Light on all four ROC-AUC tracks.");
  out.push("- PUNCH2-Light Released-13 is the stronger PUNCH2-Light variant and approaches A10 on the PDB tracks.");
  out.push("- These external labels must not be reused to modify or retune A10; any improved successor requires a newly locked development protocol.");
  out.push("");
  return out.join("\n");
}

function applyTitle(range) {
  range.format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "center",
  };
}

function applyHeader(range) {
  range.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D9E2F3" },
  };
}

function applySection(range) {
  range.format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#17365D", size: 12 },
    borders: { preset: "outside", style: "thin", color: "#9EADBA" },
  };
}

function worksheetMetricRow(row) {
  return [
    row.dataset,
    row.track,
    row.model,
    row.variant,
    row.roc_auc_caid_compatible,
    row.roc_auc_full_precision,
    row.aucpr_trapezoid,
    row.aps,
    row.f1_at_0_5,
    row.mcc_at_0_5,
    row.fmax,
    row.mcc_at_fmax,
    row.fmax_threshold,
    row.reference_proteins,
    row.predicted_proteins,
    row.protein_coverage,
    row.reference_residues,
    row.predicted_residues,
    row.residue_coverage,
    row.known_residues,
    row.positive_labels,
    row.negative_labels,
    row.source_report,
    row.source_report_sha256,
  ];
}

async function buildWorkbook(rows, archives) {
  const workbook = Workbook.create();
  const results = workbook.worksheets.add("Results");
  const metrics = workbook.worksheets.add("Metrics");
  const deltas = workbook.worksheets.add("Deltas");
  const protocol = workbook.worksheets.add("Protocol");
  for (const sheet of [results, metrics, deltas, protocol]) {
    sheet.showGridLines = false;
  }

  const metricHeaders = [
    "Dataset", "Track", "Model", "Variant", "ROC-AUC (CAID)", "ROC-AUC (full)",
    "AUPRC", "APS", "F1@0.5", "MCC@0.5", "Fmax", "MCC@Fmax", "Fmax threshold",
    "Reference proteins", "Predicted proteins", "Protein coverage", "Reference residues",
    "Predicted residues", "Residue coverage", "Known residues", "Positive labels",
    "Negative labels", "Source report", "Source report SHA256",
  ];
  metrics.getRange("A1:X1").values = [metricHeaders];
  metrics.getRange("A2:X17").values = rows.map(worksheetMetricRow);
  applyHeader(metrics.getRange("A1:X1"));
  metrics.getRange("A1:X17").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  metrics.getRange("E2:M17").format.numberFormat = "0.000";
  metrics.getRange("P2:P17").format.numberFormat = "0.0%";
  metrics.getRange("S2:S17").format.numberFormat = "0.0%";
  metrics.getRange("N2:O17").format.numberFormat = "0";
  metrics.getRange("Q2:R17").format.numberFormat = "0";
  metrics.getRange("T2:V17").format.numberFormat = "0";
  metrics.getRange("W2:X17").format.wrapText = true;
  metrics.getRange("A1:X17").format.rowHeight = 22;
  metrics.getRange("A:A").format.columnWidth = 10;
  metrics.getRange("B:B").format.columnWidth = 17;
  metrics.getRange("C:C").format.columnWidth = 27;
  metrics.getRange("D:D").format.columnWidth = 36;
  metrics.getRange("E:M").format.columnWidth = 14;
  metrics.getRange("N:V").format.columnWidth = 16;
  metrics.getRange("W:W").format.columnWidth = 62;
  metrics.getRange("X:X").format.columnWidth = 66;
  metrics.freezePanes.freezeRows(1);
  metrics.freezePanes.freezeColumns(4);
  metrics.tables.add("A1:X17", true, "B5MetricsTable");

  results.getRange("A1:Q2").merge();
  results.getRange("A1:Q2").values = [["B5 locked CAID2/CAID3 model comparison"]];
  applyTitle(results.getRange("A1:Q2"));
  results.getRange("A3:Q3").merge();
  results.getRange("A3:Q3").values = [[
    "Post-lock external evaluation only · CAID-compatible metrics · 100% protein/residue coverage · no CAID labels used for training or tuning",
  ]];
  results.getRange("A3:Q3").format = {
    fill: "#EAF2F8",
    font: { italic: true, color: "#404040" },
    wrapText: true,
  };

  results.getRange("A5:E5").values = [["Track", ...METHODS]];
  applyHeader(results.getRange("A5:E5"));
  const matrixFormulas = [];
  let metricSheetRow = 2;
  for (const dataset of DATASETS) {
    for (const track of TRACKS) {
      matrixFormulas.push([
        `="${dataset} ${track.replace("disorder_", "").toUpperCase()}"`,
        `=Metrics!$E$${metricSheetRow}`,
        `=Metrics!$E$${metricSheetRow + 1}`,
        `=Metrics!$E$${metricSheetRow + 2}`,
        `=Metrics!$E$${metricSheetRow + 3}`,
      ]);
      metricSheetRow += 4;
    }
  }
  results.getRange("A6:E9").formulas = matrixFormulas;
  results.getRange("B6:E9").format.numberFormat = "0.000";
  results.getRange("A5:E9").format.borders = { preset: "all", style: "thin", color: "#B4C6E7" };
  results.getRange("A6:A9").format.font = { bold: true, color: "#17365D" };
  for (let r = 0; r < 4; r += 1) {
    const group = rows.slice(r * 4, r * 4 + 4);
    const bestIndex = group.reduce((best, row, idx) =>
      row.roc_auc_caid_compatible > group[best].roc_auc_caid_compatible ? idx : best, 0);
    results.getCell(5 + r, 1 + bestIndex).format = {
      fill: "#E2F0D9",
      font: { bold: true, color: "#375623" },
      borders: { preset: "all", style: "medium", color: "#70AD47" },
    };
  }
  results.getRange("A11:I11").merge();
  results.getRange("A11:I11").values = [["Complete metrics (formula-linked to the audited Metrics sheet)"]];
  applySection(results.getRange("A11:I11"));
  results.getRange("A12:I12").values = [[
    "Dataset", "Track", "Model", "ROC-AUC", "AUPRC", "APS", "F1@0.5", "MCC@0.5", "Fmax",
  ]];
  applyHeader(results.getRange("A12:I12"));
  const detailFormulas = rows.map((_, index) => {
    const sourceRow = index + 2;
    return [
      `=Metrics!$A$${sourceRow}`,
      `=Metrics!$B$${sourceRow}`,
      `=Metrics!$C$${sourceRow}`,
      `=Metrics!$E$${sourceRow}`,
      `=Metrics!$G$${sourceRow}`,
      `=Metrics!$H$${sourceRow}`,
      `=Metrics!$I$${sourceRow}`,
      `=Metrics!$J$${sourceRow}`,
      `=Metrics!$K$${sourceRow}`,
    ];
  });
  results.getRange("A13:I28").formulas = detailFormulas;
  results.getRange("D13:I28").format.numberFormat = "0.000";
  results.getRange("A12:I28").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  for (let groupIndex = 0; groupIndex < 4; groupIndex += 1) {
    for (let metricOffset = 0; metricOffset < 6; metricOffset += 1) {
      const key = ["roc_auc_caid_compatible", "aucpr_trapezoid", "aps", "f1_at_0_5", "mcc_at_0_5", "fmax"][metricOffset];
      const group = rows.slice(groupIndex * 4, groupIndex * 4 + 4);
      const bestIndex = group.reduce((best, row, idx) => row[key] > group[best][key] ? idx : best, 0);
      results.getCell(12 + groupIndex * 4 + bestIndex, 3 + metricOffset).format = {
        fill: "#E2F0D9",
        font: { bold: true, color: "#375623" },
      };
    }
  }
  results.getRange("A30:I30").merge();
  results.getRange("A30:I30").values = [["Interpretation"]];
  applySection(results.getRange("A30:I30"));
  results.getRange("A31:I34").merge(true);
  results.getRange("A31:A34").values = [
    ["• LoRA-DR-Suite 650M leads ROC-AUC on CAID2/CAID3 NOX."],
    ["• A10-locked leads ROC-AUC on CAID2/CAID3 PDB and beats both PUNCH2-Light variants on all four ROC-AUC tracks."],
    ["• PUNCH2-Light Released-13 is stronger than Paper-8 and nearly matches A10 on PDB."],
    ["• CAID2/CAID3 labels are now test-only evidence; they must not be used to retune A10."],
  ];
  results.getRange("A31:I34").format = { wrapText: true, fill: "#F7F9FB" };
  results.getRange("A:A").format.columnWidth = 12;
  results.getRange("B:B").format.columnWidth = 17;
  results.getRange("C:C").format.columnWidth = 29;
  results.getRange("D:I").format.columnWidth = 13;
  results.getRange("J:Q").format.columnWidth = 12;
  results.freezePanes.freezeRows(4);

  const chart = results.charts.add("bar", results.getRange("A5:E9"));
  chart.title = "ROC-AUC across four locked external tracks";
  chart.titleTextStyle.fontSize = 12;
  chart.hasLegend = true;
  chart.setPosition("K5", "Q20");
  chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart.yAxis = { numberFormatCode: "0.000", min: 0, max: 1 };
  chart.series.items.forEach((series, index) => {
    series.fill = METHOD_COLORS[METHODS[index]];
  });

  deltas.getRange("A1:J2").merge();
  deltas.getRange("A1:J2").values = [["Metric deltas versus A10-locked"]];
  applyTitle(deltas.getRange("A1:J2"));
  deltas.getRange("A4:J4").values = [[
    "Dataset", "Track", "Model", "ROC Δ", "AUPRC Δ", "APS Δ", "F1 Δ", "MCC Δ", "Fmax Δ", "Interpretation",
  ]];
  applyHeader(deltas.getRange("A4:J4"));
  const deltaRows = [];
  for (let index = 0; index < rows.length; index += 1) {
    const sourceRow = index + 2;
    const a10SourceRow = Math.floor(index / 4) * 4 + 2;
    deltaRows.push([
      `=Metrics!$A$${sourceRow}`,
      `=Metrics!$B$${sourceRow}`,
      `=Metrics!$C$${sourceRow}`,
      `=Metrics!$E$${sourceRow}-Metrics!$E$${a10SourceRow}`,
      `=Metrics!$G$${sourceRow}-Metrics!$G$${a10SourceRow}`,
      `=Metrics!$H$${sourceRow}-Metrics!$H$${a10SourceRow}`,
      `=Metrics!$I$${sourceRow}-Metrics!$I$${a10SourceRow}`,
      `=Metrics!$J$${sourceRow}-Metrics!$J$${a10SourceRow}`,
      `=Metrics!$K$${sourceRow}-Metrics!$K$${a10SourceRow}`,
      index % 4 === 0 ? '="Reference"' : `=IF(D${index + 5}>0,"Higher ROC-AUC","Lower ROC-AUC")`,
    ]);
  }
  deltas.getRange("A5:J20").formulas = deltaRows;
  deltas.getRange("D5:I20").format.numberFormat = "+0.000;-0.000;0.000";
  deltas.getRange("A4:J20").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  deltas.getRange("D5:I20").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFFFFF", "#D9EAD3"],
    thresholds: ["min", { type: "num", value: 0 }, "max"],
  });
  deltas.getRange("A:A").format.columnWidth = 12;
  deltas.getRange("B:B").format.columnWidth = 17;
  deltas.getRange("C:C").format.columnWidth = 29;
  deltas.getRange("D:I").format.columnWidth = 13;
  deltas.getRange("J:J").format.columnWidth = 20;
  deltas.freezePanes.freezeRows(4);

  protocol.getRange("A1:F2").merge();
  protocol.getRange("A1:F2").values = [["B5 provenance and reproducibility protocol"]];
  applyTitle(protocol.getRange("A1:F2"));
  protocol.getRange("A4:F4").values = [["Source", "Archive", "Expected SHA256", "Actual SHA256", "Match", "Purpose"]];
  applyHeader(protocol.getRange("A4:F4"));
  const archiveRows = Object.entries(archives).map(([name, entry]) => [
    name,
    entry.archive,
    entry.expected_sha256,
    entry.actual_sha256,
    entry.match,
    name.startsWith("a10") ? "Locked A10 external evaluation" : "Official baseline reproduction",
  ]);
  protocol.getRange("A5:F8").values = archiveRows;
  protocol.getRange("A4:F8").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  protocol.getRange("E5:E8").format = { fill: "#E2F0D9", font: { bold: true, color: "#375623" } };
  protocol.getRange("A10:F10").merge();
  protocol.getRange("A10:F10").values = [["Guardrails"]];
  applySection(protocol.getRange("A10:F10"));
  protocol.getRange("A11:B16").values = [
    ["A10 lock", "30 members: 15 A2 + 15 A8; seeds 17/29/43; folds 0-4; uniform logit mean"],
    ["External-label policy", "CAID2/CAID3 labels used only after model locking, solely for evaluation"],
    ["Coverage", "All four compared models have 100% protein and residue coverage on every track"],
    ["Output", "One classic-IDR probability per residue; no soft-disorder output"],
    ["PUNCH2-Light", "Both Paper-8 and Released-13 official checkpoint ensembles are reported"],
    ["Retuning prohibition", "Do not use CAID2/CAID3 outcomes to modify A10; a successor needs a newly locked protocol"],
  ];
  protocol.getRange("A11:B16").format = { wrapText: true, borders: { preset: "all", style: "thin", color: "#D9E2F3" } };
  protocol.getRange("A18:F18").merge();
  protocol.getRange("A18:F18").values = [["Metric definitions"]];
  applySection(protocol.getRange("A18:F18"));
  protocol.getRange("A19:B24").values = [
    ["ROC-AUC", "CAID-compatible ROC area; ranking metric used in headline comparisons"],
    ["AUPRC", "Trapezoidal area under the precision-recall curve"],
    ["APS", "Average precision score"],
    ["F1@0.5", "F1 score at the fixed probability threshold 0.5"],
    ["MCC@0.5", "Matthews correlation coefficient at threshold 0.5"],
    ["Fmax", "Maximum F1 over the predefined threshold sweep; descriptive, not a locked deployment threshold"],
  ];
  protocol.getRange("A19:B24").format = { wrapText: true, borders: { preset: "all", style: "thin", color: "#D9E2F3" } };
  protocol.getRange("A:A").format.columnWidth = 24;
  protocol.getRange("B:B").format.columnWidth = 76;
  protocol.getRange("C:D").format.columnWidth = 66;
  protocol.getRange("E:E").format.columnWidth = 10;
  protocol.getRange("F:F").format.columnWidth = 34;
  protocol.getRange("A1:F24").format.verticalAlignment = "top";
  protocol.freezePanes.freezeRows(4);

  const inspection = {
    workbook: parseNdjson((await workbook.inspect({ kind: "workbook,sheet,table,drawing", maxChars: 8000, tableMaxRows: 4, tableMaxCols: 8 })).ndjson),
    results_region: parseNdjson((await workbook.inspect({ kind: "region", sheetId: "Results", range: "A1:Q34", maxChars: 12000 })).ndjson),
    formulas: parseNdjson((await workbook.inspect({ kind: "formula", sheetId: "Results", range: "A5:I28", maxChars: 12000, options: { maxResults: 200 } })).ndjson),
  };

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  for (const sheetName of ["Results", "Metrics", "Deltas", "Protocol"]) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: sheetName === "Metrics" ? 0.75 : 1, format: "png" });
    await fs.writeFile(
      path.join(PREVIEW_DIR, `${sheetName.toLowerCase()}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(path.join(OUTPUT_DIR, "b5_unified_model_comparison_20260826.xlsx"));
  await fs.writeFile(
    path.join(OUTPUT_DIR, "b5_workbook_inspection.json"),
    `${JSON.stringify(inspection, null, 2)}\n`,
    "utf8",
  );
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const { rows, archives } = await loadRows();

  const csvHeaders = Object.keys(rows[0]);
  const csv = [
    csvHeaders.map(csvCell).join(","),
    ...rows.map((row) => csvHeaders.map((key) => csvCell(row[key])).join(",")),
  ].join("\r\n");
  await fs.writeFile(path.join(OUTPUT_DIR, "b5_unified_metrics.csv"), `${csv}\r\n`, "utf8");
  await fs.writeFile(path.join(OUTPUT_DIR, "b5_manuscript_table.md"), markdownReport(rows), "utf8");

  const winners = {};
  for (const dataset of DATASETS) {
    winners[dataset] = {};
    for (const track of TRACKS) {
      winners[dataset][track] = {};
      for (const metric of ["roc_auc_caid_compatible", "aucpr_trapezoid", "aps", "f1_at_0_5", "mcc_at_0_5", "fmax"]) {
        const winner = winnerFor(rows, dataset, track, metric);
        winners[dataset][track][metric] = { model: winner.model, value: winner[metric] };
      }
    }
  }
  const report = {
    schema_version: 1,
    experiment: "b5_locked_model_comparison",
    status: "pass",
    generated_at: new Date().toISOString(),
    rows: rows.length,
    datasets: DATASETS,
    tracks: TRACKS,
    models: METHODS,
    archive_integrity: archives,
    validations: {
      all_archive_hashes_match: true,
      all_reports_pass: true,
      all_models_locked_before_label_access: true,
      caid_labels_used_for_training_or_tuning: false,
      full_protein_and_residue_coverage: true,
      shared_reference_counts_match: true,
      single_classic_idr_output: true,
    },
    winners,
    metrics: rows,
  };
  await fs.writeFile(path.join(OUTPUT_DIR, "b5_report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");

  await buildWorkbook(rows, archives);
  console.log(JSON.stringify({
    status: "pass",
    rows: rows.length,
    output_dir: path.relative(ROOT, OUTPUT_DIR).replaceAll("\\", "/"),
    workbook: "b5_unified_model_comparison_20260826.xlsx",
    previews: ["results.png", "metrics.png", "deltas.png", "protocol.png"],
  }, null, 2));
}

await main();

import fs from "node:fs/promises";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath =
  "outputs/01a01df7-0fae-7d13-a014-5625551d60ff/b5_unified_model_comparison_20260826.xlsx";
const blob = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(blob);
const expectedSheets = ["Results", "Metrics", "Deltas", "Protocol"];
const errorTokens = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A", "#NUM!", "#NULL!"];
const errors = [];
const sheetChecks = {};

for (const sheetName of expectedSheets) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  const values = used.values;
  const formulas = used.formulas;
  for (let row = 0; row < values.length; row += 1) {
    for (let col = 0; col < values[row].length; col += 1) {
      const value = values[row][col];
      if (typeof value === "string" && errorTokens.some((token) => value.includes(token))) {
        errors.push({ sheet: sheetName, row: row + 1, col: col + 1, value });
      }
    }
  }
  sheetChecks[sheetName] = {
    rows: values.length,
    columns: Math.max(...values.map((row) => row.length)),
    formulas: formulas.flat().filter((value) => typeof value === "string" && value.startsWith("=")).length,
    charts: sheet.charts.items.length,
    tables: sheet.tables.items.length,
  };
}

const results = workbook.worksheets.getItem("Results");
const rocMatrix = results.getRange("A5:E9").values;
const expectedMatrix = [
  ["Track", "A10-locked", "LoRA-DR-Suite 650M", "PUNCH2-Light Paper-8", "PUNCH2-Light Released-13"],
  ["CAID2 NOX", 0.781, 0.835, 0.778, 0.779],
  ["CAID2 PDB", 0.958, 0.92, 0.949, 0.95],
  ["CAID3 NOX", 0.852, 0.86, 0.826, 0.84],
  ["CAID3 PDB", 0.954, 0.92, 0.951, 0.953],
];
if (JSON.stringify(rocMatrix) !== JSON.stringify(expectedMatrix)) {
  throw new Error(`Results ROC matrix mismatch: ${JSON.stringify(rocMatrix)}`);
}
if (errors.length) {
  throw new Error(`formula errors found: ${JSON.stringify(errors)}`);
}
if (sheetChecks.Results.charts !== 1 || sheetChecks.Metrics.tables !== 1) {
  throw new Error(`drawing/table verification failed: ${JSON.stringify(sheetChecks)}`);
}

const result = {
  status: "pass",
  workbook: workbookPath,
  expected_sheets_present: true,
  formula_error_scan: "pass",
  formula_errors: errors,
  roc_matrix_matches_locked_reports: true,
  native_chart_present: true,
  metrics_table_present: true,
  sheets: sheetChecks,
};
await fs.writeFile(
  "outputs/01a01df7-0fae-7d13-a014-5625551d60ff/b5_workbook_verification.json",
  `${JSON.stringify(result, null, 2)}\n`,
  "utf8",
);
console.log(JSON.stringify(result, null, 2));

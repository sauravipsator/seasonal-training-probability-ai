import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const summary = JSON.parse(await fs.readFile("tmp/monthly_report_summary.json", "utf8"));
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Monthly Top Sellers");
sheet.showGridLines = false;

sheet.mergeCells("A1:G1");
sheet.getRange("A1").values = [["Monthly Top Seller Report"]];
sheet.getRange("A1:G1").format = {
  fill: "#123047", font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center", verticalAlignment: "center",
};
sheet.getRange("A1:G1").format.rowHeight = 30;

sheet.mergeCells("A2:G2");
sheet.getRange("A2").values = [["Observed production sales only - delivered and partially delivered orders"]];
sheet.getRange("A2:G2").format = {
  fill: "#E7F0F7", font: { italic: true, color: "#36576B" }, horizontalAlignment: "center",
};

sheet.getRange("A4:B5").values = [["Observed months", summary.observed_months], ["Delivered units analysed", summary.total_delivered_units]];
sheet.getRange("A4:A5").format = { fill: "#DCEAF3", font: { bold: true, color: "#123047" } };
sheet.getRange("B4:B5").format = { fill: "#F4F8FB", font: { bold: true, color: "#123047" }, horizontalAlignment: "right", numberFormat: "#,##0" };

const headerRow = 7;
const headers = [["Month", "Rank", "Top seller", "Units sold", "Order lines", "Active vendors", "Active stations"]];
const values = summary.rows.map(r => [r.month, r.rank, r.top_seller, r.units_sold, r.order_lines, r.active_vendors, r.active_stations]);
sheet.getRange(`A${headerRow}:G${headerRow}`).values = headers;
sheet.getRange(`A${headerRow}:G${headerRow}`).format = {
  fill: "#1F5A7A", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center",
};
sheet.getRange(`A${headerRow + 1}:G${headerRow + values.length}`).values = values;
sheet.getRange(`B${headerRow + 1}:B${headerRow + values.length}`).format.horizontalAlignment = "center";
sheet.getRange(`D${headerRow + 1}:G${headerRow + values.length}`).format = { horizontalAlignment: "right", numberFormat: "#,##0" };
sheet.getRange(`A${headerRow}:G${headerRow + values.length}`).format.borders = { preset: "outside", style: "thin", color: "#9DB8C8" };
sheet.getRange(`A${headerRow}:G${headerRow + values.length}`).format.borders = { preset: "inside", style: "thin", color: "#D7E2E9" };
sheet.getRange(`D${headerRow + 1}:D${headerRow + values.length}`).conditionalFormats.add("dataBar", { color: "#4C9B85", gradient: true });
sheet.tables.add(`A${headerRow}:G${headerRow + values.length}`, true, "MonthlyTopSellers");
sheet.freezePanes.freezeRows(headerRow);

sheet.getRange(`A${headerRow + values.length + 3}:G${headerRow + values.length + 3}`).merge();
sheet.getRange(`A${headerRow + values.length + 3}`).values = [[`Sources: ${summary.source_files.join(", ")}`]];
sheet.getRange(`A${headerRow + values.length + 3}:G${headerRow + values.length + 3}`).format = { font: { italic: true, color: "#5B6B75" } };

sheet.getRange("A:A").format.columnWidth = 24;
sheet.getRange("B:B").format.columnWidth = 9;
sheet.getRange("C:C").format.columnWidth = 28;
sheet.getRange("D:G").format.columnWidth = 17;

await fs.mkdir("output", { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save("output/monthly_top_seller_report.xlsx");

const check = await workbook.inspect({ kind: "table", range: `Monthly Top Sellers!A1:G${headerRow + values.length}`, include: "values,formulas", tableMaxRows: 20, tableMaxCols: 7 });
console.log(check.ndjson);
const preview = await workbook.render({ sheetName: "Monthly Top Sellers", range: `A1:G${headerRow + values.length + 3}`, scale: 1.5 });
await fs.writeFile("tmp/monthly_top_seller_report_preview.png", new Uint8Array(await preview.arrayBuffer()));

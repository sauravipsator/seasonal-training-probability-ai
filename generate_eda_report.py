"""Create a compact, demo-ready PDF from the latest station forecast output."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).parent
INPUT = ROOT / "output" / "station-demand-dashboard.json"
OUTPUT = ROOT / "reports" / "EDA_Report.pdf"


def money(value: int | float) -> str:
    return f"INR {value / 100000:.2f}L"


def main() -> None:
    data = json.loads(INPUT.read_text())
    summary = sorted(data["stationForecasts"], key=lambda row: row["forecastOrders"], reverse=True)
    routes = sorted(data["routeForecasts"], key=lambda row: row["forecastOrders"], reverse=True)
    evaluation = data["evaluation"]
    metadata = data["metadata"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=15 * mm, bottomMargin=14 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], textColor=colors.HexColor("#123B73"), fontSize=23, leading=27, spaceAfter=7)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], textColor=colors.HexColor("#123B73"), fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=14, textColor=colors.HexColor("#334155"))
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#64748B"))
    story = [
        Paragraph("AI Seasonal Demand Forecasting", title),
        Paragraph("Exploratory analysis and model-readiness report", ParagraphStyle("subtitle", parent=styles["Normal"], textColor=colors.HexColor("#64748B"), fontSize=11, leading=15)),
        Spacer(1, 7),
        Paragraph(f"Forecast window: <b>{metadata['forecastStart']}</b> to <b>{metadata['forecastEnd']}</b> | Model: <b>XGBoost daily station demand</b>", body),
        Spacer(1, 10),
    ]

    kpis = [["Chronological holdout", "Training coverage", "Forecast coverage"], [
        f"MAE {evaluation['mae']} orders\nRMSE {evaluation['rmse']}\nR2 {evaluation['r2']}",
        f"{evaluation['trainingRows']:,} training rows\n{len(summary)} stations\nCalendar-aware features",
        f"{metadata['forecastDays']} days\n{len(routes):,} route allocations\n{len(data['eventAlerts'])} event alerts",
    ]]
    kpi_table = Table(kpis, colWidths=[58 * mm, 58 * mm, 58 * mm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#123B73")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")), ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([kpi_table, Paragraph("What the model uses", heading), Paragraph(
        "Delivered and partially delivered orders are deduplicated from line-item exports before aggregation. The daily model uses station, weekday, month, weekend, known festival/holiday/local-event flags, prior-day demand, a 7-day rolling average, and observed vendor capacity. Features are restricted to information available before the prediction day.", body
    ), Paragraph("Operational findings", heading)])

    top_rows = [["Station", "30-day orders", "Forecast revenue", "Readiness", "Risk"]]
    for row in summary[:8]:
        top_rows.append([row["stationName"], f"{row['forecastOrders']:,}", money(row["forecastRevenue"]), f"{row['readinessScore']}/100", row["riskLevel"]])
    stations_table = Table(top_rows, colWidths=[60 * mm, 30 * mm, 33 * mm, 25 * mm, 28 * mm])
    stations_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B73")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([stations_table, Paragraph("Route and event intelligence", heading)])
    route_lines = "; ".join(f"{route['trainName']} ({route['trainNo']}): {route['forecastOrders']:,} orders" for route in routes[:3]) or "No qualifying route history."
    alerts = data["eventAlerts"]
    alert_lines = "; ".join(f"{alert['eventName']} on {alert['date']} ({alert['daysUntil']} days into the model window)" for alert in alerts) or "No known event in the current model window."
    story.extend([
        Paragraph(f"<b>Highest forecast routes:</b> {route_lines}.", body),
        Spacer(1, 4),
        Paragraph(f"<b>Known calendar alerts:</b> {alert_lines}.", body),
        Paragraph("Method and limitations", heading),
        Paragraph("Route figures are derived by allocating each station's XGBoost forecast using its trailing 90-day train-route demand share. Readiness compares predicted volume with a 10% buffer over recent daily baseline capacity; risk labels are Normal, Medium, High and Critical. These forecasts guide preparation only. Weather, train disruptions, vendor availability, new promotions and local events can change actual demand. Verify local holiday dates and live operations before acting.", body),
        Spacer(1, 8), Paragraph("Reproduce: activate .venv, run python train_station_demand_forecast.py, then python generate_eda_report.py. Supporting notebooks are in notebooks/.", small),
    ])
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

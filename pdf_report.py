"""
pdf_report.py — Generate a downloadable PDF analysis report for a stock.

Uses reportlab to build an in-memory PDF (no temp files), returned as bytes
for st.download_button. Includes: header, key metrics, valuation/DCF summary,
scoring, and a disclaimer.
"""

import io
import datetime


def _fmt_num(v, prefix="", suffix="", dec=2):
    try:
        f = float(v)
        return f"{prefix}{f:,.{dec}f}{suffix}"
    except Exception:
        return "—"


def _fmt_pct(v, dec=1):
    try:
        return f"{float(v)*100:+.{dec}f}%"
    except Exception:
        return "—"


def build_pdf_report(ticker: str, info: dict, scoring: dict, dcf: dict,
                     valuation: dict, sentiment: dict, deep: dict) -> bytes:
    """Build a one-to-two page PDF analysis report. Returns PDF bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=16*mm, rightMargin=16*mm)

    styles = getSampleStyleSheet()
    accent = colors.HexColor("#2563eb")
    dark   = colors.HexColor("#111318")
    muted  = colors.HexColor("#64748b")

    h_title = ParagraphStyle("h_title", parent=styles["Title"], fontSize=22,
                             textColor=dark, spaceAfter=2, alignment=TA_LEFT,
                             fontName="Helvetica-Bold")
    h_sub   = ParagraphStyle("h_sub", parent=styles["Normal"], fontSize=11,
                             textColor=muted, spaceAfter=10)
    h_sec   = ParagraphStyle("h_sec", parent=styles["Heading2"], fontSize=13,
                             textColor=accent, spaceBefore=12, spaceAfter=6,
                             fontName="Helvetica-Bold")
    body    = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5,
                             textColor=dark, leading=14)
    small   = ParagraphStyle("small", parent=styles["Normal"], fontSize=8,
                             textColor=muted, leading=11)

    story = []
    name = info.get("shortName") or info.get("longName") or ticker

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(f"{ticker} — {name}", h_title))
    sector_bits = " · ".join([b for b in [info.get("sector"), info.get("industry"),
                                            info.get("country")] if b])
    story.append(Paragraph(sector_bits, h_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e4e4e7")))
    story.append(Spacer(1, 8))

    # ── Score banner ──────────────────────────────────────────────────────────
    if scoring:
        comp = scoring.get("composite", 0)
        sig  = scoring.get("signal", "")
        story.append(Paragraph(
            f"<b>Composite Score:</b> {comp:.1f} / 10 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Signal:</b> {sig}", h_sec))

    # ── Key metrics table ──────────────────────────────────────────────────────
    cp = info.get("currentPrice") or info.get("regularMarketPrice")
    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Price", _fmt_num(cp, prefix="$"),
         "Market Cap", _fmt_num((info.get("marketCap") or 0)/1e9, prefix="$", suffix="B", dec=1)],
        ["Trailing P/E", _fmt_num(info.get("trailingPE"), dec=1),
         "Forward P/E", _fmt_num(info.get("forwardPE"), dec=1)],
        ["Revenue Growth", _fmt_pct(info.get("revenueGrowth")),
         "Profit Margin", _fmt_pct(info.get("profitMargins"))],
        ["ROE", _fmt_pct(info.get("returnOnEquity")),
         "Debt/Equity", _fmt_num(info.get("debtToEquity"), dec=1)],
        ["Dividend Yield", _fmt_pct(info.get("dividendYield")),
         "Beta", _fmt_num(info.get("beta"), dec=2)],
    ]
    t = Table(metrics_data, colWidths=[42*mm, 42*mm, 42*mm, 42*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f4f4f5")),
        ("TEXTCOLOR", (0,0), (-1,0), muted),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,1), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,1), (0,-1), muted),
        ("TEXTCOLOR", (2,1), (2,-1), muted),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e4e4e7")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # ── DCF / valuation ─────────────────────────────────────────────────────────
    if dcf:
        story.append(Paragraph("Valuation (DCF)", h_sec))
        fv = dcf.get("fair_value") or dcf.get("intrinsic_value")
        rows = [["Model", "Fair Value", "Upside vs Price"]]
        models = dcf.get("models", {})
        if isinstance(models, dict) and models:
            for mname, mval in list(models.items())[:5]:
                fvv = mval.get("fair_value") if isinstance(mval, dict) else mval
                up = ""
                try:
                    if fvv and cp:
                        up = f"{(float(fvv)/float(cp)-1)*100:+.1f}%"
                except Exception:
                    pass
                rows.append([str(mname), _fmt_num(fvv, prefix="$"), up])
        elif fv:
            up = f"{(float(fv)/float(cp)-1)*100:+.1f}%" if (fv and cp) else ""
            rows.append(["DCF Fair Value", _fmt_num(fv, prefix="$"), up])

        if len(rows) > 1:
            dt = Table(rows, colWidths=[68*mm, 50*mm, 50*mm])
            dt.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f4f4f5")),
                ("TEXTCOLOR", (0,0), (-1,0), muted),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 9),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e4e4e7")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
                ("TOPPADDING", (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(dt)
            story.append(Spacer(1, 6))

    # ── Deep analysis highlights ────────────────────────────────────────────────
    if deep:
        story.append(Paragraph("Financial Health", h_sec))
        piotroski = deep.get("piotroski", {})
        altman    = deep.get("altman", {})
        bits = []
        if piotroski.get("score") is not None:
            bits.append(f"Piotroski F-Score: <b>{piotroski['score']}/9</b>")
        if altman.get("z_score") is not None:
            bits.append(f"Altman Z-Score: <b>{altman['z_score']:.2f}</b> ({altman.get('zone','')})")
        if bits:
            story.append(Paragraph(" &nbsp;|&nbsp; ".join(bits), body))
            story.append(Spacer(1, 4))

    # ── Analyst sentiment ───────────────────────────────────────────────────────
    if sentiment:
        tgt = info.get("targetMeanPrice")
        if tgt and cp:
            try:
                upside = (float(tgt)/float(cp) - 1) * 100
                story.append(Paragraph("Analyst Consensus", h_sec))
                story.append(Paragraph(
                    f"Mean price target: <b>${float(tgt):,.2f}</b> "
                    f"(implied {upside:+.1f}%) across "
                    f"{info.get('numberOfAnalystOpinions','—')} analysts.", body))
            except Exception:
                pass

    # ── Footer / disclaimer ─────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e4e4e7")))
    story.append(Spacer(1, 4))
    gen_date = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    story.append(Paragraph(
        f"Generated by Stock Analyzer on {gen_date}. Data: Yahoo Finance. "
        "This report is for informational purposes only and does not constitute "
        "investment advice. Valuations are model estimates based on assumptions "
        "that may not hold. Always do your own research.", small))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

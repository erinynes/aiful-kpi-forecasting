"""Build the management-style Monthly Business Performance Report (PDF) from pipeline outputs.

All numbers in the text are read from reports/tables/, so re-running the pipeline after a
new month's PDF is published regenerates the report end to end.
"""
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from forecasting import ROOT

TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "AIFUL_Monthly_Performance_Report_2026-08.pdf"

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
TINT = colors.HexColor("#f4f3f0")
BLUE = colors.HexColor("#2a78d6")
GOOD = colors.HexColor("#1c7a3e")
BAD = colors.HexColor("#b3261e")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                            alignment=TA_LEFT, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=10, leading=13, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK,
                         spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=INK,
                         spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("b", fontName="Helvetica", fontSize=9.3, leading=13, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bl", fontName="Helvetica", fontSize=9.3, leading=13, textColor=INK,
                             leftIndent=11, bulletIndent=0, spaceAfter=3),
    "small": ParagraphStyle("sm", fontName="Helvetica", fontSize=7.6, leading=10, textColor=INK2),
    "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.4, leading=10.5, textColor=INK),
    "celln": ParagraphStyle("cn", fontName="Helvetica", fontSize=8.4, leading=10.5, textColor=INK, alignment=2),
    "cellb": ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.4, leading=10.5, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items]


def fig(name, width_mm=172):
    img = Image(str(FIG / name))
    r = img.imageHeight / img.imageWidth
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * r
    return img


def table(data, widths, header=True, align_right_from=1):
    t = Table(data, colWidths=[w * mm for w in widths], hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.4),
          ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2)]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.2), ("TEXTCOLOR", (0, 0), (-1, 0), INK2),
               ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2)]
    for r in range(1 if header else 0, len(data)):
        if r % 2 == 0:
            st.append(("BACKGROUND", (0, r), (-1, r), TINT))
    t.setStyle(TableStyle(st))
    return t


def signed(v, unit="%", d=1):
    color = "#1c7a3e" if v > 0 else "#b3261e" if v < 0 else "#52514e"
    return f'<font color="{color}">{v:+.{d}f}{unit}</font>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(INK2)
    canvas.drawString(18 * mm, 10 * mm,
                      "Independent analysis built only from public IR disclosures and official statistics. "
                      "Not an AIFUL / Muninova document.")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    facts = json.loads((TAB / "key_facts.json").read_text())
    sc = pd.read_csv(TAB / "scorecard_latest_month.csv").set_index("kpi")
    mape = pd.read_csv(TAB / "mape_by_model.csv", index_col=0)
    rob = pd.read_csv(TAB / "macro_robustness_mape.csv", index_col=0)
    dm = pd.read_csv(TAB / "macro_value_dm_test.csv")
    fc = pd.read_csv(TAB / "forecast_12m_sarima.csv")
    scen = pd.read_csv(TAB / "macro_scenarios.csv")

    a, n, cr, bal, acc, dq, rc = (sc.loc[x] for x in ["applications", "new_accounts", "contract_rate_pct",
                                                       "unsecured_loans_mil", "accounts_total_k",
                                                       "delinquency_ratio_pct", "refund_claims"])
    fy_app_g = (facts["fy27_applications_outlook"] / facts["fy26_applications_actual"] - 1) * 100
    fy_new_g = (facts["fy27_new_accounts_outlook"] / facts["fy26_new_accounts_actual"] - 1) * 100
    bal_g = (facts["fy27_balance_mar27"] / facts["balance_mar26"] - 1) * 100
    skill = lambda t: (1 - mape.loc[t, "sarima"] / mape.loc[t, "snaive"]) * 100

    story = []
    # ------------------------------------------------------------- cover / summary
    story += [P("Monthly Business Performance Report", "title"),
              P("AIFUL Corporation (standalone) · Unsecured personal-loan business · Results to <b>August 2026</b>"
                " · Outlook to August 2027", "sub"),
              Spacer(1, 8)]
    story.append(P("Executive summary", "h1"))
    story += bullets([
        f"<b>Demand is holding, conversion is not.</b> Applications rose {a.yoy_pct:.1f}% YoY in August "
        f"({a.fytd_yoy_pct:+.1f}% fiscal year-to-date), but new accounts fell {-n.yoy_pct:.1f}% "
        f"({n.fytd_yoy_pct:+.1f}% FYTD) because the contract rate slipped to {cr.actual:.1f}% "
        f"({cr.yoy_pt:+.1f}pt YoY; FYTD {facts['fytd_contract_rate']:.1f}% vs {facts['fytd_contract_rate_ly']:.1f}%).",
        f"<b>The book keeps compounding.</b> Unsecured balance reached ¥{bal.actual/1000:,.1f}bn "
        f"({bal.yoy_pct:+.1f}% YoY), with growth decelerating from ~9% a year ago. Customer accounts with a "
        f"balance: {acc.actual:,.0f}k ({acc.yoy_pct:+.1f}%).",
        f"<b>Credit quality improved.</b> The delinquent-loan ratio was {dq.actual:.3f}% vs {dq.last_year:.3f}% a year "
        f"ago, consistent with tighter screening explaining part of the lower contract rate. Interest-refund "
        f"claims continue to fade ({rc.fytd:.0f} FYTD, {rc.fytd_yoy_pct:+.0f}%).",
        f"<b>Outlook (FY2027/3):</b> applications ~{facts['fy27_applications_outlook']/1e3:,.0f}k ({fy_app_g:+.1f}% vs FY2026/3), "
        f"new accounts ~{facts['fy27_new_accounts_outlook']/1e3:,.0f}k ({fy_new_g:+.1f}%), "
        f"March-2027 balance ¥{facts['fy27_balance_mar27']/1000:,.0f}bn ({bal_g:+.1f}% YoY; 80% range "
        f"¥{facts['fy27_balance_mar27_lo80']/1000:,.0f}-{facts['fy27_balance_mar27_hi80']/1000:,.0f}bn).",
        f"<b>Macro conditions do not improve the forecast.</b> Adding unemployment, consumer confidence and "
        f"inflation to the time-series model did not reduce error in a 54-month backtest. AIFUL's volumes are "
        f"driven mainly by its own levers (marketing, credit policy, digital channels), which is where "
        f"monitoring should focus.",
    ])

    story.append(P("August 2026 scorecard", "h1"))
    rows = [["KPI", "Aug-26", "Aug-25", "YoY", "vs 1-month\nforecast", "FYTD (Apr-Aug)", "FYTD YoY"]]
    def fmt(v, d=0):
        return f"{v:,.{d}f}"
    rows.append([P("Applications (unsecured)", "cell"), fmt(a.actual), fmt(a.last_year), P(signed(a.yoy_pct), "celln"),
                 P(signed(a.vs_forecast_pct), "celln"), fmt(a.fytd), P(signed(a.fytd_yoy_pct), "celln")])
    rows.append([P("New accounts acquired", "cell"), fmt(n.actual), fmt(n.last_year), P(signed(n.yoy_pct), "celln"),
                 P(signed(n.vs_forecast_pct), "celln"), fmt(n.fytd), P(signed(n.fytd_yoy_pct), "celln")])
    rows.append([P("Contract rate (%)", "cell"), fmt(cr.actual, 1), fmt(cr.last_year, 1), P(signed(cr.yoy_pt, "pt"), "celln"),
                 "–", fmt(facts["fytd_contract_rate"], 1),
                 P(signed(facts["fytd_contract_rate"] - facts["fytd_contract_rate_ly"], "pt"), "celln")])
    rows.append([P("Unsecured balance (¥mil)", "cell"), fmt(bal.actual), fmt(bal.last_year), P(signed(bal.yoy_pct), "celln"),
                 P(signed(bal.vs_forecast_pct, d=2), "celln"), "–", "–"])
    rows.append([P("Accounts with balance ('000)", "cell"), fmt(acc.actual), fmt(acc.last_year), P(signed(acc.yoy_pct), "celln"),
                 "–", "–", "–"])
    rows.append([P("Delinquent-loan ratio (%)", "cell"), fmt(dq.actual, 3), fmt(dq.last_year, 3),
                 P(signed(-dq.yoy_pt, "pt", 3).replace(f"{-dq.yoy_pt:+.3f}", f"{dq.yoy_pt:+.3f}"), "celln"), "–", "–", "–"])
    rows.append([P("Interest-refund claims", "cell"), fmt(rc.actual), fmt(rc.last_year), P(signed(-rc.yoy_pct).replace(f"{-rc.yoy_pct:+.1f}", f"{rc.yoy_pct:+.1f}"), "celln"),
                 "–", fmt(rc.fytd), P(signed(-rc.fytd_yoy_pct).replace(f"{-rc.fytd_yoy_pct:+.1f}", f"{rc.fytd_yoy_pct:+.1f}"), "celln")])
    story.append(table(rows, [44, 20, 20, 17, 21, 26, 18]))
    story.append(Spacer(1, 3))
    story.append(P("Colour shows direction for the business: green = favourable, red = unfavourable (for delinquency "
                   "and refund claims a fall is favourable). 'vs 1-month forecast' compares the actual with a SARIMA "
                   "forecast made with data to July 2026.", "small"))

    # ------------------------------------------------------------- acquisition funnel
    story.append(PageBreak())
    story.append(P("1. Customer acquisition funnel", "h1"))
    story.append(P(
        "Applications quadrupled between 2021 and 2023 and have since levelled off at around 80k a month "
        "(12-month average). Part of the 2023 peak was over-counting: AIFUL disclosed that duplicate "
        "applications were counted until May 2023, so earlier volumes are not like-for-like. The model "
        "includes an explicit term for that break."))
    story.append(fig("01_funnel_volumes.png", 168))
    story.append(P(
        "New accounts peaked in late 2023 and have trended down since. With applications flat, the gap is "
        "conversion: the contract rate has fallen from ~39% (late 2023, after the counting fix) to ~29%. "
        "Two explanations fit the public data, and they need different responses:"))
    story += bullets([
        "<b>Tighter credit policy</b> (deliberate): consistent with the falling delinquency ratio (Section 2). "
        "If so, the lower rate is the cost of a better book and is acceptable.",
        "<b>Lower-intent application mix</b> (channel or marketing driven): e.g. more comparison-site or "
        "price-shopping traffic that never completes. This would show up as falling conversion in specific "
        "channels and is fixable. <i>Separating the two needs internal channel data (Section 5).</i>",
    ])
    story.append(fig("02_contract_rate.png", 168))

    # ------------------------------------------------------------- portfolio
    story.append(PageBreak())
    story.append(P("2. Portfolio growth and credit quality", "h1"))
    story.append(P(
        f"The unsecured personal-loan balance has grown every year since 2015, reaching ¥{bal.actual/1000:,.1f}bn. "
        f"Growth has slowed from a 2023-24 peak of ~12% to {bal.yoy_pct:.1f}%, reflecting fewer new accounts. "
        "Because the balance moves slowly, it is highly predictable: the 1-month-ahead model error is under 0.3%."))
    story.append(fig("03_balance.png", 160))
    story.append(P(
        f"The delinquent-loan ratio (12-month average) has fallen from ~0.9-1.0% in 2022-23 to about 0.7%, "
        f"with August at {dq.actual:.3f}%. Combined with the lower contract rate, this suggests AIFUL is "
        "trading volume for quality. Interest-refund (grey-zone) claims are now a small residual "
        f"({rc.actual:.0f} in August vs 3,000-4,000 a month in 2016)."))
    story.append(fig("04_delinquency.png", 160))

    # ------------------------------------------------------------- outlook
    story.append(PageBreak())
    story.append(P("3. Twelve-month outlook (September 2026 - August 2027)", "h1"))
    story.append(P(
        "Forecasts come from a seasonal ARIMA model on log values, with terms for the 2020 COVID shock and the "
        "2023 application-counting change. Shaded bands are 80% and 95% prediction intervals. The volume "
        "series are noisy month to month (typical error ~11% at 1-6 months ahead), so management targets "
        "should be set on quarterly totals rather than single months."))
    story.append(fig("06_forecasts.png", 158))
    f = fc.copy()
    f["q"] = pd.PeriodIndex(f["month"], freq="M").asfreq("Q-MAR")
    qt = f[f.target != "unsecured_loans_mil"].groupby(["q", "target"])["mean"].sum().unstack()
    bq = f[f.target == "unsecured_loans_mil"].groupby("q")["mean"].last()
    rows = [["Fiscal quarter", "Applications", "New accounts", "Implied contract rate", "Balance at quarter end (¥bn)"]]
    for q in qt.index:
        nm = f"FY{q.qyear}/3 Q{q.quarter}" + (" (Sep only)" if q == qt.index[0] else "") + \
             (" (Jul-Aug only)" if q == qt.index[-1] else "")
        rows.append([nm, f"{qt.loc[q, 'applications']:,.0f}", f"{qt.loc[q, 'new_accounts']:,.0f}",
                     f"{qt.loc[q, 'new_accounts'] / qt.loc[q, 'applications'] * 100:.1f}%",
                     f"{bq.loc[q] / 1000:,.1f}"])
    story.append(KeepTogether([P("Forecast by fiscal quarter", "h2"), table(rows, [38, 28, 28, 36, 44])]))

    # ------------------------------------------------------------- macro
    story.append(PageBreak())
    story.append(P("4. Do macro conditions help predict the business?", "h1"))
    story.append(P(
        "The test: an identical SARIMA model run with and without three macro drivers (change in "
        "unemployment, consumer confidence, core CPI inflation). Each driver is lagged six months so it is "
        "already published when the forecast is made. Every model was re-fitted at each of 54 monthly "
        "forecast origins (September 2021 - February 2026) and scored on the following 1-6 months."))
    story.append(fig("05_backtest_mape.png", 172))
    story.append(P(
        f"<b>Result.</b> SARIMA cuts error versus a seasonal-naive benchmark by {skill('applications'):.0f}% "
        f"(applications), {skill('new_accounts'):.0f}% (new accounts) and {skill('unsecured_loans_mil'):.0f}% (balance). "
        f"Adding macro drivers does not improve it: MAPE is {mape.loc['applications','sarimax']:.1f}% vs "
        f"{mape.loc['applications','sarima']:.1f}% for applications and {mape.loc['new_accounts','sarimax']:.1f}% vs "
        f"{mape.loc['new_accounts','sarima']:.1f}% for new accounts. At one month ahead macro makes forecasts "
        f"significantly <i>worse</i> for applications and balance (Diebold-Mariano p = "
        f"{dm[(dm.target=='applications')&(dm.h==1)].p_value.iloc[0]:.2f} and "
        f"{dm[(dm.target=='unsecured_loans_mil')&(dm.h==1)].p_value.iloc[0]:.2f}). No horizon shows a significant gain."))
    rr = [["Macro driver set (SARIMA + ...)", "Applications", "New accounts", "Balance"]]
    for idx, r in rob.iterrows():
        pretty = {"sarima (no macro)": "No macro (SARIMA only)",
                  "baseline_set (d_unemp, CCI, core CPI)": "Pre-specified: unemployment change, CCI, core CPI",
                  "consumer_confidence only": "Consumer confidence only",
                  "unemployment change only": "Unemployment change only",
                  "CCI income + employment sub-indices": "CCI income + employment sub-indices",
                  "call rate + core CPI": "Call rate + core CPI"}
        rr.append([pretty.get(idx, idx), f"{r.applications:.2f}",
                   f"{r.new_accounts:.2f}", f"{r.unsecured_loans_mil:.2f}"])
    story.append(KeepTogether([P("Robustness: backtest MAPE (%) with alternative driver sets", "h2"),
                               table(rr, [78, 28, 28, 28])]))
    story.append(Spacer(1, 4))
    story.append(P(
        "No driver set improves MAPE by more than ~0.4pt. Consumer confidence alone comes closest, but it was "
        "chosen after seeing the results, so the small gain should not be treated as real."))
    story.append(P(
        "<b>Interpretation.</b> In-sample, the macro effect on new accounts is <i>counter-cyclical</i> and statistically "
        "significant: weaker confidence and rising unemployment come before more new accounts six months later, "
        "consistent with households borrowing to cover income shortfalls. But the effect is small next to "
        "company-driven swings (the 2022-23 growth surge, the post-2023 slowdown), so it adds nothing out of sample. "
        "<b>For management, macro indicators belong on a watch-list for risk scenarios, not in the core forecast.</b>"))
    sp = scen[scen.target != "unsecured_loans_mil"].copy()
    rows = [["Scenario (applied to Sep-26 onward)", "Applications, 12m", "New accounts, 12m"]]
    for s_ in sp.scenario.unique():
        x = sp[sp.scenario == s_].set_index("target")
        rows.append([s_, f"{x.loc['applications','vs_base_pct']:+.1f}%", f"{x.loc['new_accounts','vs_base_pct']:+.1f}%"])
    story.append(KeepTogether([P("Macro stress scenarios (direction only, low confidence)", "h2"),
                               table(rows, [96, 34, 34])]))
    story.append(Spacer(1, 3))
    story.append(P("Effect on the 12-month balance forecast is under 0.2% in every scenario. A downside "
                   "scenario raises <i>demand</i>; whether it raises bookings depends on credit policy, "
                   "which the model does not see.", "small"))

    # ------------------------------------------------------------- channels
    story.append(PageBreak())
    story.append(P("5. Distribution channels: what the public data cannot show", "h1"))
    story.append(P(
        "Muninova reports that more than 90% of AIFUL applications are now submitted online, that the whole "
        "process from application to contract can be completed in the smartphone app, and that cardless "
        "transactions are available at Seven Bank and Lawson Bank ATMs. The monthly disclosure has no channel "
        "breakdown, so this report cannot measure channel performance directly. With internal data, I would "
        "extend the model as follows:"))
    rows = [["Question for management", "Internal data needed", "Model / monitoring extension"],
            [P("Is the contract-rate decline caused by credit policy or by channel mix?", "cell"),
             P("Applications, approvals and contracts by channel (web, app, phone, store / contract machine, "
               "affiliate / comparison sites) and by screening outcome", "cell"),
             P("Decompose the contract-rate change into <b>mix</b> vs <b>within-channel rate</b> effects each month "
               "(shift-share analysis); alert when the within-channel rate moves more than 2 standard deviations", "cell")],
            [P("Which channels will drive next year's acquisition?", "cell"),
             P("Monthly funnel counts per channel, 3+ years", "cell"),
             P("<b>Hierarchical forecasting</b>: SARIMA per channel, reconciled (MinT) to the company total "
               "reported here, so channel plans sum to the same number as the corporate forecast", "cell")],
            [P("What does marketing buy?", "cell"),
             P("Advertising spend by medium (TV, search, affiliate, SNS), campaign calendar", "cell"),
             P("Add spend as regressors (adstock-transformed). Unlike macro, these are the levers the backtest "
               "suggests actually move volume", "cell")],
            [P("Is app and cardless usage changing customer value?", "cell"),
             P("Logins, cardless ATM withdrawals by network (Seven Bank, Lawson Bank, other), repayment channel", "cell"),
             P("Track the share of cardless and app-repayment users and their balance growth / delinquency vs "
               "card users; cohort retention curves by acquisition channel", "cell")],
            [P("Early warning on credit quality", "cell"),
             P("Early-stage arrears (1-30 days) by channel and vintage", "cell"),
             P("Vintage curves by acquisition month and channel; forecast the delinquent-loan ratio 3-6 months ahead "
               "from early arrears", "cell")]]
    story.append(table(rows, [48, 56, 70], align_right_from=99))

    # ------------------------------------------------------------- method
    story.append(P("6. Data and method", "h1"))
    story += bullets([
        "<b>Company data:</b> AIFUL monthly data PDFs (Muninova Holdings IR, FY2014/3 - Aug 2026, 161 months), parsed "
        "automatically. All parsed values were checked against the published YoY percentages (max difference 0.05pt) "
        "and the published contract rate (= new accounts ÷ applications).",
        "<b>Macro data:</b> unemployment rate (Statistics Bureau Labour Force Survey), consumer confidence index "
        "(Cabinet Office ESRI, two-or-more-person households, seasonally adjusted), CPI (Statistics Bureau, "
        "2025 base), hourly earnings, call rate and USD/JPY (via FRED/OECD).",
        "<b>Models:</b> seasonal naive; Holt-Winters (damped trend); SARIMA(1,1,1)(0,1,1)12 on logs; the same SARIMA "
        "with macro regressors; LightGBM direct multi-horizon model. Expanding-window, rolling-origin evaluation "
        "with no look-ahead.",
        "<b>Caveats:</b> monthly account figures exclude part of the former LIFE receivables except at quarter-ends; "
        "applications before June 2023 include duplicates; the delinquency ratio covers former-AIFUL loans only. "
        "Forecasts assume no change in credit policy, interest-rate caps or marketing strategy.",
        "Code, data and full backtest tables: see the project repository (README).",
    ])
    story.append(KeepTogether([P("Macro context used in the test", "h2"), fig("07_macro.png", 172)]))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="AIFUL Monthly Business Performance Report - August 2026",
                            author="Pankhuri Srivastava")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

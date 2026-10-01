"""Final 12-month forecasts, August-2026 scorecard, macro stress scenarios and report figures."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

import forecasting as F

FIG = F.ROOT / "reports" / "figures"
TAB = F.ROOT / "reports" / "tables"

# palette (validated reference palette, light mode)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, BAND, BAND2 = "#2a78d6", "#eb6834", "#cde2fb", "#e6f0fc"
LABELS = {"applications": "Applications", "new_accounts": "New accounts",
          "unsecured_loans_mil": "Unsecured loan balance (¥bn)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 200,
    "lines.linewidth": 2,
})


def ts(idx):
    return idx.to_timestamp()


def scale(target, v):
    return v / 1000 if target == "unsecured_loans_mil" else v


def thousands(ax):
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    k, m = F.load()
    last = k.index.max()  # 2026-08
    facts = {"last_month": str(last)}

    # ------------------------------------------------ scorecard: latest month
    def one_step(target, origin):
        y = k[target].loc[:origin]
        idx = pd.period_range(origin + 1, periods=1, freq="M")
        return float(F.f_sarima(y, 1, target=target, fc_index=idx)[0])

    rows = []
    fytd = pd.period_range(f"{last.year}-04", last, freq="M")
    fytd_ly = fytd - 12
    for t, unit in [("applications", "count"), ("new_accounts", "count"), ("contract_rate_pct", "%"),
                    ("unsecured_loans_mil", "¥mil"), ("accounts_total_k", "'000"),
                    ("delinquency_ratio_pct", "%"), ("refund_claims", "count")]:
        a, ly = k.loc[last, t], k.loc[last - 12, t]
        r = {"kpi": t, "unit": unit, "actual": a, "last_year": ly,
             "yoy_pct": (a / ly - 1) * 100 if unit != "%" else None,
             "yoy_pt": a - ly if unit == "%" else None}
        if t in F.TARGETS:
            fc = one_step(t, last - 1)
            r["forecast_1m"] = fc
            r["vs_forecast_pct"] = (a / fc - 1) * 100
        if t in ("applications", "new_accounts", "refund_claims"):
            r["fytd"] = k.loc[fytd, t].sum()
            r["fytd_ly"] = k.loc[fytd_ly, t].sum()
            r["fytd_yoy_pct"] = (r["fytd"] / r["fytd_ly"] - 1) * 100
        rows.append(r)
    sc = pd.DataFrame(rows)
    sc.to_csv(TAB / "scorecard_latest_month.csv", index=False)
    facts["fytd_contract_rate"] = float(k.loc[fytd, "new_accounts"].sum() / k.loc[fytd, "applications"].sum() * 100)
    facts["fytd_contract_rate_ly"] = float(k.loc[fytd_ly, "new_accounts"].sum() / k.loc[fytd_ly, "applications"].sum() * 100)

    # ------------------------------------------------ final forecasts (SARIMA = production model)
    fcs = {}
    for t in F.TARGETS:
        fcs[t] = F.final_forecast(k, m, t, model="sarima", horizon=12).table
    out = pd.concat({t: f for t, f in fcs.items()}, names=["target", "month"]).round(1)
    out.to_csv(TAB / "forecast_12m_sarima.csv")

    # macro stress scenarios on SARIMAX (shock applied to macro values after last published month)
    scen = {"base (macro held flat)": None,
            "downside: unemp +0.7pp, CCI -6, core CPI +1pp": {"d_unemp": 0.7, "consumer_confidence": -6, "core_cpi_yoy": 1.0},
            "upside: unemp -0.3pp, CCI +3": {"d_unemp": -0.3, "consumer_confidence": 3}}
    srows = []
    for t in F.TARGETS:
        for name, s in scen.items():
            f = F.final_forecast(k, m, t, model="sarimax", horizon=12, scenario=s).table
            srows.append((t, name, f["mean"].sum() if t != "unsecured_loans_mil" else f["mean"].iloc[-1]))
    sdf = pd.DataFrame(srows, columns=["target", "scenario", "value_12m"])
    base = sdf[sdf.scenario.str.startswith("base")].set_index("target")["value_12m"]
    sdf["vs_base_pct"] = sdf.apply(lambda r: (r.value_12m / base[r.target] - 1) * 100, axis=1)
    sdf.round(2).to_csv(TAB / "macro_scenarios.csv", index=False)

    # FY2027/3 outlook = actual Apr-Aug + forecast Sep-Mar
    fy_months = pd.period_range("2026-09", "2027-03", freq="M")
    for t in ("applications", "new_accounts"):
        facts[f"fy27_{t}_outlook"] = float(k.loc[fytd, t].sum() + fcs[t].loc[fy_months, "mean"].sum())
        facts[f"fy26_{t}_actual"] = float(k.loc["2025-04":"2026-03", t].sum())
    facts["fy27_balance_mar27"] = float(fcs["unsecured_loans_mil"].loc[pd.Period("2027-03", "M"), "mean"])
    facts["fy27_balance_mar27_lo80"] = float(fcs["unsecured_loans_mil"].loc[pd.Period("2027-03", "M"), "lo80"])
    facts["fy27_balance_mar27_hi80"] = float(fcs["unsecured_loans_mil"].loc[pd.Period("2027-03", "M"), "hi80"])
    facts["balance_mar26"] = float(k.loc["2026-03", "unsecured_loans_mil"])
    (TAB / "key_facts.json").write_text(json.dumps(facts, indent=2))

    # ------------------------------------------------ figures
    k_plot = k.loc["2014-04":]

    # 1. funnel volumes: two panels, shared x, separate y (no dual axis)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, t, c in [(axes[0], "applications", BLUE), (axes[1], "new_accounts", BLUE)]:
        s = k_plot[t]
        ax.plot(ts(s.index), s.values, color=c, linewidth=1.4, alpha=0.35)
        ax.plot(ts(s.index), s.rolling(12).mean().values, color=c, linewidth=2)
        ax.set_title(f"{LABELS[t]} per month  (thin: monthly, bold: 12-month average)")
        thousands(ax)
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-30"), color=GRID, alpha=0.8, lw=0)
    axes[0].axvline(pd.Timestamp("2023-06-01"), color=MUTED, lw=1, ls="--")
    axes[0].text(pd.Timestamp("2023-07-15"), 30000, "Counting method changed\n(duplicates removed), Jun-23",
                 fontsize=7.5, color=INK2, va="bottom")
    axes[1].text(pd.Timestamp("2020-05-01"), axes[1].get_ylim()[1] * 0.92, "COVID", fontsize=7.5, color=INK2, ha="center")
    save(fig, "01_funnel_volumes.png")

    # 2. contract rate
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    cr = k_plot["new_accounts"].rolling(3).sum() / k_plot["applications"].rolling(3).sum() * 100
    ax.plot(ts(cr.index), cr.values, color=BLUE)
    ax.set_title("Contract rate: new accounts ÷ applications, 3-month rolling (%)")
    ax.axvline(pd.Timestamp("2023-06-01"), color=MUTED, lw=1, ls="--")
    ax.annotate(f"{cr.iloc[-1]:.1f}%", (ts(cr.index)[-1], cr.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK)
    save(fig, "02_contract_rate.png")

    # 3. balance and growth (two panels)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    b = k_plot["unsecured_loans_mil"] / 1000
    axes[0].plot(ts(b.index), b.values, color=BLUE)
    axes[0].set_title("Unsecured personal-loan balance (¥bn)")
    g = (k["unsecured_loans_mil"].pct_change(12) * 100).loc["2014-04":]
    axes[1].bar(ts(g.index), g.values, width=24, color=BLUE)
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[1].set_title("Year-on-year growth (%)")
    save(fig, "03_balance.png")

    # 4. delinquency
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    d = k_plot["delinquency_ratio_pct"]
    ax.plot(ts(d.index), d.values, color=BLUE, lw=1.2, alpha=0.35)
    ax.plot(ts(d.index), d.rolling(12).mean().values, color=BLUE)
    ax.set_title("Unsecured delinquent-loan ratio, monthly (%)  (bold: 12-month average)")
    save(fig, "04_delinquency.png")

    # 5. backtest accuracy
    mape = pd.read_csv(TAB / "mape_by_model.csv", index_col=0)
    order = ["snaive", "lgbm", "ets", "sarimax", "sarima"]
    names = {"snaive": "Seasonal naive", "lgbm": "LightGBM + macro", "ets": "Holt-Winters",
             "sarimax": "SARIMA + macro", "sarima": "SARIMA"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, t in zip(axes, F.TARGETS):
        v = mape.loc[t, order]
        cols = [ORANGE if o == "sarimax" else (BLUE if o == "sarima" else "#b9b8b2") for o in order]
        ax.barh([names[o] for o in order], v.values, color=cols, height=0.62)
        for i, x in enumerate(v.values):
            ax.text(x, i, f" {x:.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_title(LABELS[t].replace(" (¥bn)", ""), fontsize=9)
        ax.set_xlim(0, v.max() * 1.3)
        ax.grid(axis="y", visible=False)
        if t != F.TARGETS[0]:
            ax.set_yticklabels([])
        ax.set_xlabel("MAPE %, h=1-6")
    save(fig, "05_backtest_mape.png")

    # 6. forecast fans
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6))
    for ax, t in zip(axes, F.TARGETS):
        hist = scale(t, k[t].loc["2022-09":])
        f = fcs[t]
        x = ts(f.index)
        ax.fill_between(x, scale(t, f.lo95), scale(t, f.hi95), color=BAND2, lw=0, label="95% interval")
        ax.fill_between(x, scale(t, f.lo80), scale(t, f.hi80), color=BAND, lw=0, label="80% interval")
        ax.plot(ts(hist.index), hist.values, color=BLUE, label="Actual")
        ax.plot([ts(hist.index)[-1], x[0]], [hist.iloc[-1], scale(t, f["mean"].iloc[0])], color=ORANGE, ls="--", lw=1.5)
        ax.plot(x, scale(t, f["mean"]), color=ORANGE, ls="--", label="SARIMA forecast")
        ax.set_title(f"{LABELS[t]}: actual and 12-month forecast")
        thousands(ax)
    axes[0].legend(loc="upper left", ncol=4, fontsize=7.5, frameon=False)
    save(fig, "06_forecasts.png")

    # 7. macro context (small multiples, own y-scale each)
    mm = m.loc["2014-04":]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2), sharex=True)
    for ax, (c, ttl) in zip(axes, [("unemployment_rate", "Unemployment rate (%)"),
                                   ("consumer_confidence", "Consumer confidence index"),
                                   ("core_cpi_yoy", "Core CPI, YoY (%)")]):
        s = mm[c].dropna()
        ax.plot(ts(s.index), s.values, color=BLUE, lw=1.6)
        ax.set_title(ttl, fontsize=8.5)
        ax.tick_params(labelsize=7)
    save(fig, "07_macro.png")

    print(sc.round(2).to_string())
    print(sdf.round(2).to_string())
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()

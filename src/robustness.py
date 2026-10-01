"""Robustness: does ANY reasonable macro set help SARIMA? Re-run SARIMAX with alternative driver sets."""
import pandas as pd
import forecasting as F

SETS = {
    "baseline_set (d_unemp, CCI, core CPI)": ["d_unemp", "consumer_confidence", "core_cpi_yoy"],
    "consumer_confidence only": ["consumer_confidence"],
    "unemployment change only": ["d_unemp"],
    "CCI income + employment sub-indices": ["cc_income_growth", "cc_employment"],
    "call rate + core CPI": ["call_rate", "core_cpi_yoy"],
}

def main():
    k, m = F.load()
    base = F.backtest(k, m, start="2021-09", models=["sarima"])
    out = {"sarima (no macro)": base.groupby("target").ape.mean()}
    for name, feats in SETS.items():
        F.MACRO_FEATURES = feats
        bt = F.backtest(k, m, start="2021-09", models=["sarimax"])
        out[name] = bt.groupby("target").ape.mean()
    res = pd.DataFrame(out).T.round(2)
    res.to_csv(F.ROOT / "reports" / "tables" / "macro_robustness_mape.csv")
    print(res)

if __name__ == "__main__":
    main()

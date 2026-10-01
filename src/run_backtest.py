"""Run the rolling-origin backtest and write accuracy tables to reports/tables/."""
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import H, PROC, ROOT, backtest, diebold_mariano, load

TABLES = ROOT / "reports" / "tables"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    k, m = load()
    bt = backtest(k, m, start="2021-09")
    bt.to_csv(PROC / "backtest_forecasts.csv", index=False)

    # MAPE by target x model (avg over h=1..6), and by horizon
    mape = bt.groupby(["target", "model"]).ape.mean().unstack().round(2)
    mape_h = bt.groupby(["target", "model", "h"]).ape.mean().unstack("h").round(2)
    # MASE-style skill vs seasonal naive
    skill = (1 - mape.div(mape["snaive"], axis=0)).mul(100).round(1)
    mape.to_csv(TABLES / "mape_by_model.csv")
    mape_h.to_csv(TABLES / "mape_by_horizon.csv")
    skill.to_csv(TABLES / "skill_vs_snaive_pct.csv")

    # Does macro help? SARIMAX vs SARIMA, Diebold-Mariano per horizon
    rows = []
    for t in bt.target.unique():
        for h in range(1, H + 1):
            a = bt[(bt.target == t) & (bt.h == h)].pivot_table(index="origin", columns="model",
                                                                values=["actual", "forecast"])
            act = a["actual"]["sarima"].values
            e0 = np.log(a["forecast"]["sarima"].values) - np.log(act)
            e1 = np.log(a["forecast"]["sarimax"].values) - np.log(act)
            dm, p = diebold_mariano(e1, e0, h)
            rows.append((t, h, np.mean(np.abs(e0)) * 100, np.mean(np.abs(e1)) * 100, dm, p))
    dm = pd.DataFrame(rows, columns=["target", "h", "sarima_mae_log_pct", "sarimax_mae_log_pct",
                                     "dm_stat", "p_value"]).round(3)
    dm.to_csv(TABLES / "macro_value_dm_test.csv", index=False)

    print("MAPE (%), avg h=1..6\n", mape, "\n")
    print("Skill vs seasonal naive (%)\n", skill, "\n")
    print("MAPE by horizon\n", mape_h, "\n")
    print("SARIMAX vs SARIMA (DM test; negative stat = macro better)\n", dm)


if __name__ == "__main__":
    main()

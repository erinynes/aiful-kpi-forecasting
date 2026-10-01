"""Build a monthly panel of Japanese macro drivers.

Sources (all public, raw files in data/raw/):
  - Unemployment rate, SA (%)            FRED LRHUTTTTJPM156S  (Statistics Bureau Labour Force Survey via OECD)
  - Hourly earnings, manufacturing YoY % FRED LCEAMN01JPM659S  (MHLW Monthly Labour Survey via OECD)
  - Call rate (%)                        FRED IRSTCI01JPM156N  (Bank of Japan via OECD)
  - USD/JPY                              FRED EXJPUS
  - Consumer Confidence Index, SA        Cabinet Office ESRI Consumer Confidence Survey (e-Stat long-term table)
  - CPI YoY % (all items / core)         Statistics Bureau, 2025-base long-term series (e-Stat)

Output: data/processed/macro_monthly.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

FRED = {
    "LRHUTTTTJPM156S": "unemployment_rate",
    "LCEAMN01JPM659S": "wage_growth_yoy",
    "IRSTCI01JPM156N": "call_rate",
    "EXJPUS": "usdjpy",
}


def fred() -> pd.DataFrame:
    frames = []
    for sid, name in FRED.items():
        df = pd.read_csv(RAW / f"aiful_macro_fred_{sid}.csv", na_values=".")
        df["month"] = pd.PeriodIndex(pd.to_datetime(df["observation_date"]), freq="M")
        frames.append(df.set_index("month")[sid].rename(name))
    return pd.concat(frames, axis=1)


def consumer_confidence() -> pd.DataFrame:
    x = pd.read_excel(RAW / "aiful_macro_consumer_confidence_SA.xlsx", header=None)
    hdr_en = x.iloc[5]
    rows = x[x[0].astype(str).str.fullmatch(r"\d{10}")]
    out = pd.DataFrame({
        "month": pd.PeriodIndex(
            [f"{c[:4]}-{c[-2:]}" for c in rows[0].astype(str)], freq="M"),
        "consumer_confidence": pd.to_numeric(rows[4], errors="coerce").values,
    })
    # sub-indices: income growth and employment outlook are the most relevant for credit demand
    for col, name in [("Income growth", "cc_income_growth"), ("Employment", "cc_employment")]:
        idx = [i for i, h in hdr_en.items() if isinstance(h, str) and h.endswith(col)]
        if idx:
            out[name] = pd.to_numeric(rows[idx[0]], errors="coerce").values
    return out.set_index("month")


def cpi() -> pd.DataFrame:
    path = RAW / "aiful_macro_cpi_yoy_middle_class.csv"
    df = pd.read_csv(path, encoding="cp932", header=None, dtype=str, skiprows=1)
    en = df.iloc[0].tolist()
    data = df[df[0].str.fullmatch(r"\d{6}", na=False)]
    col_all = en.index("All items")
    col_core = en.index("All items, less fresh food")
    out = pd.DataFrame({
        "month": pd.PeriodIndex([f"{s[:4]}-{s[4:]}" for s in data[0]], freq="M"),
        "cpi_yoy": pd.to_numeric(data[col_all], errors="coerce").values,
        "core_cpi_yoy": pd.to_numeric(data[col_core], errors="coerce").values,
    })
    return out.set_index("month")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = fred().join(consumer_confidence(), how="outer").join(cpi(), how="outer")
    m = m.loc["2010-01":]
    # real wage growth proxy = nominal manufacturing hourly earnings growth minus CPI inflation
    m["real_wage_growth"] = m["wage_growth_yoy"] - m["cpi_yoy"]
    m.to_csv(OUT / "macro_monthly.csv", index_label="month")
    print(m.tail(4).round(2).to_string())
    print("last valid:", {c: str(m[c].last_valid_index()) for c in m})


if __name__ == "__main__":
    main()

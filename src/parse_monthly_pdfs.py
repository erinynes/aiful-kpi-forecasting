"""Parse AIFUL / Muninova 'Monthly Data' PDFs into a tidy monthly KPI table.

Source: https://www.muninova.co.jp/en/ir/finance/monthly_data.html
Each PDF covers one fiscal year (April-March). Older files (to FY2018/3) are
landscape-rotated; pdfplumber normalises coordinates so one parser handles both.

Output: data/processed/aiful_monthly_kpis.csv (one row per month)
        data/processed/aiful_monthly_yoy_reported.csv (published YoY %, for validation)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

NUM = re.compile(r"^[-▲△]?[\d,]+(\.\d+)?$")
MONTH = re.compile(r"^(\d{2})/(\d{1,2})$")
PERIOD_HEADERS = ("上期計", "下期計", "通期計", "累計")


def _lines(page, tol=2.5):
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
    words.sort(key=lambda w: (round(w["top"]), w["x0"]))
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]


def _to_num(s: str) -> float:
    s = s.replace(",", "").replace("▲", "-").replace("△", "-")
    return float(s)


def classify(label: str, after_accounts: bool) -> str | None:
    """Map a Japanese row label to a canonical metric name."""
    l = label.replace(" ", "")
    if "営業債権合計" in l:
        return "total_receivables_mil"
    if "営業貸付金残高" in l:
        return "loans_outstanding_mil"
    if "口座数" in l:
        return "accounts_total_k"
    if "移管" in l:
        return "transferred_loans_mil"
    if "解約発生率" in l:
        return "delinquency_ratio_pct"
    if "成約率" in l and "<" not in l and "＜" not in l:
        return "contract_rate_pct"
    if "申込件数" in l:
        return "applications"
    if "新規獲得件数" in l:
        return "new_accounts"
    if "利息返還請求件数" in l:
        return "refund_claims"
    if "利息返還金" in l:
        return "refund_cash_mil"
    if "割賦売掛金" in l:
        return "installment_mil"
    if "支払承諾見返" in l:
        return "guarantee_mil"
    if "事業者" in l:
        return "smallbiz_accounts_k" if after_accounts else "smallbiz_loans_mil"
    if "有担保" in l:
        return "secured_accounts_k" if after_accounts else "secured_loans_mil"
    if "無担保" in l:
        return "unsecured_accounts_k" if after_accounts else "unsecured_loans_mil"
    return None


def parse_pdf(path: Path):
    page = pdfplumber.open(path).pages[0]
    lines = _lines(page)

    # 1) column headers: month labels and period-total labels with their x-centres
    cols: list[tuple[float, str]] = []
    for line in lines:
        months = [w for w in line if MONTH.match(w["text"])]
        if len(months) >= 12:
            for w in months:
                yy, mm = MONTH.match(w["text"]).groups()
                cols.append(((w["x0"] + w["x1"]) / 2, f"20{yy}-{int(mm):02d}"))
            break
    for line in lines:
        for w in line:
            if w["text"] in PERIOD_HEADERS:
                cols.append(((w["x0"] + w["x1"]) / 2, "TOTAL"))
    if not cols:
        raise ValueError(f"no month header in {path.name}")
    month_xs = sorted(c for c in cols if c[1] != "TOTAL")
    spacing = (month_xs[-1][0] - month_xs[0][0]) / (len(month_xs) - 1)

    def column_of(w):
        xc = (w["x0"] + w["x1"]) / 2
        best = min(cols, key=lambda c: abs(c[0] - xc))
        # numbers are right-aligned, so allow ~ half a column of slack
        return best[1] if abs(best[0] - xc) < spacing * 0.6 else None

    levels: dict[str, dict[str, float]] = {}
    yoys: dict[str, dict[str, float]] = {}
    after_accounts = False
    last_metric = None
    pending_label = None
    for line in lines:
        texts = [w["text"] for w in line]
        nums = [w for w in line if NUM.match(w["text"])]
        label = "".join(t for t in texts if not NUM.match(t))
        is_yoy = "yoy" in label.lower()
        metric = None if is_yoy else classify(label, after_accounts)
        if metric == "accounts_total_k":
            after_accounts = True
        if not nums:
            if metric:
                pending_label = metric  # label and values sit on different lines
            continue
        if not is_yoy and metric is None and pending_label:
            metric = pending_label
        pending_label = None
        values = {}
        for w in nums:
            c = column_of(w)
            if c and c != "TOTAL":
                values[c] = _to_num(w["text"])
        if len(values) < 3:  # partial-year files can have as few as 3 real months
            continue
        if is_yoy:
            if last_metric:
                yoys[last_metric] = values
        elif metric:
            levels[metric] = values
            last_metric = metric
    return levels, yoys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lv_rows, yy_rows = [], []
    for pdf in sorted(RAW.glob("aiful_MD_*.pdf")):
        levels, yoys = parse_pdf(pdf)
        for m, d in levels.items():
            for month, v in d.items():
                lv_rows.append((month, m, v, pdf.name))
        for m, d in yoys.items():
            for month, v in d.items():
                yy_rows.append((month, m, v))
        print(f"{pdf.name}: {len(levels)} metrics, months {min(next(iter(levels.values())))}..{max(next(iter(levels.values())))}")

    lv = pd.DataFrame(lv_rows, columns=["month", "metric", "value", "source"])
    wide = lv.pivot_table(index="month", columns="metric", values="value", aggfunc="first")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    # future months in the partial-year file are published as 0
    wide = wide[wide["total_receivables_mil"] > 0].sort_index()
    wide.to_csv(OUT / "aiful_monthly_kpis.csv", index_label="month")

    yy = pd.DataFrame(yy_rows, columns=["month", "metric", "yoy_pct"])
    yy.pivot_table(index="month", columns="metric", values="yoy_pct", aggfunc="first").to_csv(
        OUT / "aiful_monthly_yoy_reported.csv", index_label="month")
    print(wide.shape, wide.index.min(), wide.index.max())


if __name__ == "__main__":
    main()

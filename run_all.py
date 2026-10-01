"""Run the full pipeline: parse PDFs -> macro panel -> backtest -> robustness -> forecasts & figures -> PDF report.

Usage:  python run_all.py            (full run, ~15 min; the backtests dominate)
        python run_all.py --fast     (skip both backtests, reuse existing tables)
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
steps = ["parse_monthly_pdfs.py", "build_macro.py", "run_backtest.py", "robustness.py",
         "make_report_assets.py", "build_report.py"]
if "--fast" in sys.argv:
    steps = [s for s in steps if s not in ("run_backtest.py", "robustness.py")]

for s in steps:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, s], cwd=SRC, check=True)

"""
Keep outputs/live_predicted.csv: every day-ahead NSW price forecast next to the
actual price, one row per half-hour. Run after dashboard.py (run_daily_forecast.bat
does both at 21:00).

Each run
  1. adds (or re-issues) tomorrow's predictions from outputs/forecast_d1/;
  2. fills actual prices: AEMO's clean daily file wherever it has the bin
     (actual_source = aemo, final), otherwise today's 5-min dispatch averaged
     to 30 min (actual_source = dispatch, provisional, overwritten later).
     The newest dispatch bin may still be partial, so it is left out.

    .venv/Scripts/python.exe record_live.py
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data_loader

D1_DIR = config.OUTPUT_DIR / "forecast_d1"
SOURCE_CSV = config.OUTPUT_DIR / "dashboard" / "today_source.csv"
TABLE_CSV = config.OUTPUT_DIR / "live_predicted.csv"
PAGE_TEMPLATE = config.PROJECT_ROOT / "live_template.html"
PAGE_HTML = config.PROJECT_ROOT / "live.html"
PRED_COLS = ["naive_pred", "arima_pred", "hybrid_pred", "aemo_pd"]
ERR_COLS = {c: c.replace("_pred", "") + "_err" for c in PRED_COLS}
COLS = ["timestamp", "forecast_date", "issued_at", *PRED_COLS,
        "actual_price", "actual_source", *ERR_COLS.values()]


def load_table() -> pd.DataFrame:
    if TABLE_CSV.exists():
        return pd.read_csv(TABLE_CSV, parse_dates=["timestamp"])
    return pd.DataFrame(columns=COLS).astype({"timestamp": "datetime64[ns]"})


def add_forecasts(table: pd.DataFrame) -> pd.DataFrame:
    """Add day-ahead files not yet in the table; the newest is always re-read,
    so a re-run for the same day replaces the earlier issuance."""
    files = sorted(D1_DIR.glob("forecast_price_*.csv"))
    if not files:
        print("[record] no day-ahead price forecast found")
        return table
    have = set(table["forecast_date"].astype(str))
    for i, f in enumerate(files):
        day = f.stem.removeprefix("forecast_price_")
        if day in have and i < len(files) - 1:
            continue
        fc = pd.read_csv(f, parse_dates=["timestamp"])
        new = fc[["timestamp", "issued_at", *PRED_COLS]].copy()
        new.insert(1, "forecast_date", day)
        table = table[table["forecast_date"].astype(str) != day]
        table = pd.concat([table, new], ignore_index=True) if len(table) else new
        print(f"[record] predictions for {day} (issued {new['issued_at'].iloc[0]})")
    return table


def fill_actuals(table: pd.DataFrame) -> pd.DataFrame:
    table = table.set_index("timestamp")
    for c in ("actual_price", "actual_source"):
        if c not in table:
            table[c] = np.nan
    table["actual_source"] = table["actual_source"].astype(object)

    if SOURCE_CSV.exists():
        src = pd.read_csv(SOURCE_CSV, index_col=0, parse_dates=True)
        act = src[src["source"] == "actual"].iloc[:-1]["price"]
        idx = table.index.intersection(act.index)
        idx = idx[table.loc[idx, "actual_source"] != "aemo"]
        table.loc[idx, "actual_price"] = act[idx]
        table.loc[idx, "actual_source"] = "dispatch"

    live = data_loader.read_live()["price"]
    idx = table.index.intersection(live.dropna().index)
    table.loc[idx, "actual_price"] = live[idx]
    table.loc[idx, "actual_source"] = "aemo"

    for c, err in ERR_COLS.items():
        table[err] = table[c] - table["actual_price"]
    return table.reset_index()


def write_page(table: pd.DataFrame) -> None:
    """Render live.html: my model (hybrid) vs AEMO pre-dispatch vs actual price."""
    def num(v):
        return None if pd.isna(v) else round(float(v), 2)
    rows = [{"t": ts.strftime("%Y-%m-%d %H:%M"), "model": num(m), "aemo": num(a), "actual": num(x)}
            for ts, m, a, x in zip(pd.to_datetime(table["timestamp"]), table["hybrid_pred"],
                                   table["aemo_pd"], table["actual_price"])]
    data = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "rows": rows}
    html = PAGE_TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", json.dumps(data))
    PAGE_HTML.write_text(html, encoding="utf-8")
    print(f"[record] wrote {PAGE_HTML}")


def main() -> None:
    table = fill_actuals(add_forecasts(load_table()))
    table = table.sort_values("timestamp")[COLS]
    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_CSV, index=False, float_format="%.2f")

    n = table.groupby("forecast_date")["actual_price"].count()
    for day, k in n.items():
        print(f"[record] {day}: {k}/48 actuals")
    print(f"[record] wrote {TABLE_CSV}")
    write_page(table)


if __name__ == "__main__":
    main()

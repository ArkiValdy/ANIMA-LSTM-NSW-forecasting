"""
AEMO pre-dispatch forecast downloader + parser (plan Sec 5 benchmark).

Pre-dispatch runs are published roughly 40 minutes before each trading day.
The MMSDM archive keeps PREDISPATCHPRICE (RRP forecast) and
PREDISPATCHREGIONSUM (demand forecast) per 30-minute period with a
PREDISPATCHSEQNO / DATETIME layout.

For the day-ahead benchmark we take, for each target day, the forecast row
whose DATETIME falls inside that day (AEMO pre-dispatch is a "perfect
scheduling" forecast run over the whole day) — the standard "pre-dispatch
issued ~1 day ahead" comparison.

Outputs (eval window only):
  data/raw/predispatch_price_30min.csv   (forecast RRP per 30-min period)
  data/raw/predispatch_demand_30min.csv  (forecast TOTALDEMAND per 30-min)
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_aemo import RAW_DIR, REGION, download, iter_months, parse_aemo_csv, table_url  # noqa: E402

TABLES = {
    "price": "PREDISPATCHPRICE",
    "demand": "PREDISPATCHREGIONSUM",
}
VALUE_COLUMNS = {"price": "RRP", "demand": "TOTALDEMAND"}


def load_month(year: int, month: int, kind: str) -> pd.DataFrame:
    table = TABLES[kind]
    zip_path = RAW_DIR / f"{table}_{year:04d}{month:02d}.zip"
    download(table_url(year, month, table), zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        df = parse_aemo_csv(zf.read(zf.namelist()[0]))

    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df = df[(df["REGIONID"] == REGION) & (df["INTERVENTION"] == "0")]
    df = df.dropna(subset=["DATETIME"])

    df = pd.DataFrame({
        "DATETIME": df["DATETIME"],
        "value": pd.to_numeric(df[VALUE_COLUMNS[kind]], errors="coerce"),
    }).dropna()
    df = df.set_index("DATETIME")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.resample("30min").mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01", help="YYYY-MM inclusive")
    ap.add_argument("--end", default="2026-06", help="YYYY-MM inclusive")
    args = ap.parse_args()

    t0 = time.perf_counter()
    for kind in ("price", "demand"):
        parts = []
        for y, m in iter_months(args.start, args.end):
            print(f"[{kind}] {y:04d}-{m:02d}")
            parts.append(load_month(y, m, kind))
        full = pd.concat(parts)
        full = full[~full.index.duplicated(keep="first")].sort_index()
        full.index.name = "timestamp"
        out = RAW_DIR / f"predispatch_{kind}_30min.csv"
        full.to_csv(out)
        print(f"[{kind}] Wrote {len(full)} rows to {out} "
              f"({full.index.min()} .. {full.index.max()})")

    print(f"[predispatch] Elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

"""
NSW dashboard (AEMO NEM-dashboard style) with a true day-ahead forecast for TOMORROW.

    .venv/Scripts/python.exe dashboard.py --refresh      # full run (~10 min)
    .venv/Scripts/python.exe dashboard.py --skip_models  # re-render, reuse forecasts

Steps
  1. (--refresh) update the live AEMO / Open-Meteo files (data/update_live.py).
  2. Download today's 5-min dispatch and the latest AEMO
     pre-dispatch run from NEMWeb (data/nemweb_live.py).
  3. Build a nowcast file: NSW actuals through the latest dispatch interval,
     AEMO pre-dispatch for the rest of today. The live file stays untouched.
  4. Forecast tomorrow (price and demand) with naive / ARIMA / hybrid, the same
     protocol as forecast.py, and store AEMO's pre-dispatch for tomorrow next
     to it as a benchmark.
  5. Score earlier day-ahead forecasts against clean actuals (live file only).
  6. Write outputs/dashboard/index.html (self-contained; data embedded).

Outputs
  outputs/forecast_d1/forecast_{target}_{date}.csv   issued the day before,
      with an extra aemo_pd column; scores_{target}.csv once actuals exist
  outputs/dashboard/price_demand_nowcast.csv
  outputs/dashboard/index.html
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_loader

sys.path.insert(0, str(config.DATA_DIR))
import nemweb_live  # noqa: E402

D1_DIR = config.OUTPUT_DIR / "forecast_d1"
DASH_DIR = config.OUTPUT_DIR / "dashboard"
TEMPLATE = config.PROJECT_ROOT / "dashboard_template.html"
TARGETS = ("price", "demand")
MODELS = ("naive", "arima", "hybrid", "aemo_pd")
# Positive MWFLOW runs from the first region to the second.
IC_ENDS = {
    "N-Q-MNSP1": ("NSW", "QLD", "Terranora"),
    "NSW1-QLD1": ("NSW", "QLD", "QNI"),
    "VIC1-NSW1": ("VIC", "NSW", "VIC–NSW"),
    "NSW1-SA1": ("NSW", "SA", "EnergyConnect"),
    "V-SA": ("VIC", "SA", "Heywood"),
    "V-S-MNSP1": ("VIC", "SA", "Murraylink"),
    "T-V-MNSP1": ("TAS", "VIC", "Basslink"),
}


def _ms(idx) -> list[int]:
    """NEM-time timestamps as epoch ms, read as if UTC (the page formats in UTC)."""
    return (pd.DatetimeIndex(idx).asi8 // 1_000_000).tolist()


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ----------------------------------------------------------------------------- #
# Forecasting
# ----------------------------------------------------------------------------- #
def run_models(nowcast_path: Path, tomorrow: pd.Timestamp, pd_bins: pd.DataFrame) -> None:
    import forecast

    D1_DIR.mkdir(parents=True, exist_ok=True)
    fidx = pd.date_range(tomorrow, periods=config.PERIODS_PER_DAY, freq="30min")
    for target in TARGETS:
        cached = config.OUTPUT_DIR / "forecast" / f"arima_order_{target}.json"
        if cached.exists() and not (D1_DIR / cached.name).exists():
            shutil.copy(cached, D1_DIR / cached.name)
        forecast.run_forecast(
            price_demand_path=nowcast_path,
            temperature_path=config.LIVE_TEMPERATURE_CSV,
            target=target,
            forecast_date=str(tomorrow.date()),
            out_dir=D1_DIR,
            score=False,
            extra_columns={"aemo_pd": pd_bins[target].reindex(fidx).values},
        )


def score_d1(target: str, live: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE of each day-ahead forecast whose day now has 48 actuals."""
    rows = []
    for f in sorted(D1_DIR.glob(f"forecast_{target}_*.csv")):
        fc = pd.read_csv(f, index_col=0, parse_dates=True)
        act = live[target].reindex(fc.index)
        if act.notna().sum() != config.PERIODS_PER_DAY:
            continue
        row = {"date": str(fc.index[0].date()), "issued_at": fc["issued_at"].iloc[0]}
        for m in MODELS:
            col = next((c for c in (f"{m}_pred", m) if c in fc), None)
            if col is not None:
                err = act.values - fc[col].values
                row[f"{m}_RMSE"] = float(np.sqrt(np.nanmean(err ** 2)))
                row[f"{m}_MAE"] = float(np.nanmean(np.abs(err)))
        rows.append(row)
    scores = pd.DataFrame(rows)
    if len(scores):
        scores.to_csv(D1_DIR / f"scores_{target}.csv", index=False)
    return scores


# ----------------------------------------------------------------------------- #
# Payload
# ----------------------------------------------------------------------------- #
def build_payload(disp, pd_run, today_src, tomorrow, live) -> dict:
    today = tomorrow - pd.Timedelta(days=1)
    yesterday = today - pd.Timedelta(days=1)
    last5 = disp["price"].index.max()

    rid = nemweb_live.REGION
    p, d = disp["price"][rid].dropna(), disp["demand"][rid].dropna()
    now = {
        "price": _r(p.iloc[-1]), "demand": _r(d.iloc[-1], 0),
        "avg_today": _r(p.mean()), "max_today": _r(p.max()), "min_today": _r(p.min()),
        "max_at": _ms([p.idxmax()])[0], "min_at": _ms([p.idxmin()])[0],
        "peak_demand": _r(d.max(), 0), "peak_demand_at": _ms([d.idxmax()])[0],
        "price5": {"t": _ms(p.index), "v": [_r(v) for v in p.values]},
        "demand5": {"t": _ms(d.index), "v": [_r(v, 0) for v in d.values]},
    }

    # Interconnectors touching NSW, signed as flow INTO NSW (+ import, - export)
    ics = []
    for _, r in disp["interconnectors"].iterrows():
        if r["INTERCONNECTORID"] not in IC_ENDS or "NSW" not in IC_ENDS[r["INTERCONNECTORID"]][:2]:
            continue
        a, b, name = IC_ENDS[r["INTERCONNECTORID"]]
        mw = float(r["MWFLOW"])
        frm, to = (a, b) if mw >= 0 else (b, a)
        limit = abs(float(r["EXPORTLIMIT"] if mw >= 0 else r["IMPORTLIMIT"]))
        ics.append({"id": r["INTERCONNECTORID"], "name": name, "from": frm, "to": to,
                    "mw": _r(abs(mw), 0), "limit": _r(limit, 0),
                    "into_nsw": _r(abs(mw) if to == "NSW" else -abs(mw), 0)})
    now["net_import"] = _r(sum(i["into_nsw"] for i in ics), 0)

    nsw = {}
    hist_idx = pd.date_range(yesterday, today - pd.Timedelta(minutes=30), freq="30min")
    act_today = today_src[today_src["source"] == "actual"]
    pd_today = today_src[today_src["source"] == "predispatch"]
    for target in TARGETS:
        fc = pd.read_csv(D1_DIR / f"forecast_{target}_{tomorrow.date()}.csv", index_col=0, parse_dates=True)
        hist = pd.concat([live[target].reindex(hist_idx), act_today[target]])
        pd_series = pd.concat([pd_today[target], fc["aemo_pd"]])
        nsw[target] = {
            "actual": {"t": _ms(hist.index), "v": [_r(v) for v in hist.values]},
            "aemo_pd": {"t": _ms(pd_series.index), "v": [_r(v) for v in pd_series.values]},
            **{m: {"t": _ms(fc.index), "v": [_r(v) for v in fc[f"{m}_pred"].values]}
               for m in ("naive", "arima", "hybrid")},
            "issued_at": fc["issued_at"].iloc[0],
            "temperature": {"t": _ms(fc.index), "v": [_r(v, 1) for v in fc["temperature_fc"].values]},
        }

    scores = {}
    for target in TARGETS:
        d1 = score_d1(target, live)
        same = config.OUTPUT_DIR / "forecast" / f"scores_{target}.csv"
        scores[target] = {
            "d1": json.loads(d1.to_json(orient="records")) if len(d1) else [],
            "same_day": json.loads(pd.read_csv(same).to_json(orient="records")) if same.exists() else [],
        }

    return {
        "generated_nem": str(nemweb_live.nem_now().replace(microsecond=0)),
        "last_dispatch": str(last5),
        "pd_run": f"{pd_run[:4]}-{pd_run[4:6]}-{pd_run[6:8]} {pd_run[8:10]}:{pd_run[10:12]}",
        "today": str(today.date()),
        "tomorrow": str(tomorrow.date()),
        "today_actual_bins": int(len(act_today)),
        "now": now,
        "interconnectors": ics,
        "nsw": nsw,
        "scores": scores,
    }


def render(payload: dict) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    out = DASH_DIR / "index.html"
    out.write_text(html.replace("/*__DATA__*/null", blob), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="update live AEMO/Open-Meteo files first")
    ap.add_argument("--skip_models", action="store_true", help="reuse tomorrow's saved forecasts")
    args = ap.parse_args()

    if args.refresh:
        subprocess.run([sys.executable, str(config.DATA_DIR / "update_live.py")], check=True)

    today = pd.Timestamp(nemweb_live.nem_now().date())
    tomorrow = today + pd.Timedelta(days=1)
    disp = nemweb_live.fetch_dispatch_today(today)
    pd_df, pd_run = nemweb_live.fetch_predispatch()
    pd_bins = nemweb_live.predispatch_bins(pd_df)
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    nowcast = DASH_DIR / "price_demand_nowcast.csv"
    today_src = nemweb_live.build_nowcast(config.LIVE_PRICE_DEMAND_CSV, disp, pd_bins, nowcast, today)
    today_src.to_csv(DASH_DIR / "today_source.csv", index_label="timestamp")

    if not args.skip_models:
        run_models(nowcast, tomorrow, pd_bins)

    live = data_loader.read_live()
    out = render(build_payload(disp, pd_run, today_src, tomorrow, live))
    print(f"[dashboard] wrote {out}")


if __name__ == "__main__":
    main()

"""
Pipeline orchestrator for the ARIMA-LSTM hybrid electricity forecaster.

Headline evaluation (plan Sec 3): rolling day-ahead walk-forward.

    python main.py --mode walkforward --target demand
    python main.py --mode walkforward --target price

Quick smoke test (static chronological split, not the headline result):

    python main.py --mode static --target demand

Live forecast of the next day (fetches the latest data first):

    python main.py --mode forecast --target demand --refresh
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np

# Reproducibility — seed before importing tensorflow indirectly through lstm_model
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quiet TF info/threadpool noise
random.seed(42)
np.random.seed(42)

import config
import data_loader
import preprocessing
import arima_model
import lstm_model
import evaluation
import walk_forward


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARIMA-LSTM hybrid electricity forecaster")
    p.add_argument(
        "--price_demand",
        type=str,
        default=str(config.PRICE_DEMAND_CSV),
        help="Path to price/demand CSV (consolidated 30-min or raw AEMO).",
    )
    p.add_argument(
        "--temperature",
        type=str,
        default=str(config.TEMPERATURE_CSV),
        help="Path to temperature CSV (hourly/daily; synthetic used if missing).",
    )
    p.add_argument(
        "--target",
        type=str,
        default=config.TARGET_COLUMN,
        choices=["demand", "price"],
        help="Forecast target (demand is primary, price secondary).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="walkforward",
        choices=["walkforward", "static", "forecast"],
        help="walkforward = headline rolling day-ahead; static = smoke test; "
             "forecast = live forecast of the next day (live data files).",
    )
    p.add_argument(
        "--forecast_date",
        type=str,
        default=None,
        help="Forecast mode: day to forecast (default: the day after the last "
             "complete day of data). A past date is saved as a hindcast.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Forecast mode: run data/update_live.py first to fetch the latest data.",
    )
    p.add_argument(
        "--eval_start",
        type=str,
        default=config.EVAL_START,
        help="Inclusive eval start date (YYYY-MM-DD) for walk-forward.",
    )
    p.add_argument(
        "--max_eval_days",
        type=int,
        default=None,
        help="Cap the number of evaluated forecast days (smoke testing).",
    )
    p.add_argument(
        "--fday_demand",
        action="store_true",
        help="Price target only: add forecast-day demand (actual as perfect-"
             "forecast proxy) as a per-period LSTM input. Outputs go to "
             "outputs/walkforward_fday_demand/ so the baseline is kept.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override the walk-forward output folder (e.g. for live-data runs).",
    )
    return p.parse_args()


def run_walkforward_mode(args: argparse.Namespace) -> None:
    config.PRICE_USE_FDAY_DEMAND = args.fday_demand
    if args.fday_demand and args.target != "price":
        raise SystemExit("--fday_demand only applies to --target price")
    out_dir = config.OUTPUT_DIR / (
        "walkforward_fday_demand" if args.fday_demand else "walkforward"
    )
    if args.out_dir:
        out_dir = Path(args.out_dir)
    print(f"\n[main] TARGET = {config.TARGET_COLUMN}  (mode = walkforward"
          f"{', + forecast-day demand' if args.fday_demand else ''})")

    df = data_loader.load_dataset(
        price_demand_path=args.price_demand,
        temperature_path=args.temperature,
    )
    df_features = preprocessing.build_features(df)
    print(f"[main] Features: {len(df_features)} rows, "
          f"range {df_features.index.min()} .. {df_features.index.max()}")

    result = walk_forward.run_walkforward(
        df_features=df_features,
        target=config.TARGET_COLUMN,
        eval_start=args.eval_start,
        max_eval_days=args.max_eval_days,
        out_dir=out_dir,
    )

    # Aggregate metrics + report figures from the tidy predictions frame
    evaluation.evaluate_walkforward(
        predictions=result.predictions,
        target=config.TARGET_COLUMN,
        out_dir=out_dir,
    )

    # AEMO pre-dispatch benchmark (plan Sec 5) — best-effort: skipped if the
    # archived predispatch file is unavailable.
    try:
        import aemo_benchmark

        aemo_benchmark.aemo_metrics(
            predictions=result.predictions,
            target=config.TARGET_COLUMN,
            out_dir=out_dir,
        )
    except FileNotFoundError:
        print("[main] AEMO pre-dispatch data not available — skipping benchmark.")
    print(f"\n[main] Eval window: {result.eval_start.date()} .. {result.eval_end.date()} "
          f"({result.n_days} days)")
    print(f"[main] Predictions: "
          f"{out_dir / f'predictions_{config.TARGET_COLUMN}.csv'}")


def run_forecast_mode(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    import forecast

    if args.fday_demand:
        raise SystemExit("--fday_demand needs actual forecast-day demand, which is "
                         "unknown in forecast mode")
    if args.refresh:
        subprocess.run(
            [sys.executable, str(config.DATA_DIR / "update_live.py")], check=True
        )
    # Default inputs switch to the live files; explicit paths are respected
    price_demand = args.price_demand
    if price_demand == str(config.PRICE_DEMAND_CSV):
        price_demand = config.LIVE_PRICE_DEMAND_CSV
    temperature = args.temperature
    if temperature == str(config.TEMPERATURE_CSV):
        temperature = config.LIVE_TEMPERATURE_CSV
    forecast.run_forecast(
        price_demand_path=Path(price_demand),
        temperature_path=Path(temperature),
        target=args.target,
        forecast_date=args.forecast_date,
        out_dir=Path(args.out_dir) if args.out_dir else config.OUTPUT_DIR / "forecast",
    )


def run_static_mode(args: argparse.Namespace) -> None:
    import hybrid_model

    print(f"\n[main] STATIC SMOKE TEST — target = {config.TARGET_COLUMN}")

    df = data_loader.load_dataset(
        price_demand_path=args.price_demand,
        temperature_path=args.temperature,
    )
    df_features = preprocessing.build_features(df)
    splits = preprocessing.prepare(df_features)

    arima_res = arima_model.fit_and_predict(
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
        model_path=config.MODEL_DIR / "arima.joblib",
    )

    lstm_res = lstm_model.train_and_predict(
        X_train=splits.X_train,
        residuals_train=arima_res.train_residuals,
        X_val=splits.X_val,
        residuals_val=arima_res.val_residuals,
        X_test=splits.X_test,
        weights_path=config.MODEL_DIR / "lstm.weights.h5",
    )

    hybrid = hybrid_model.combine(
        arima_train=arima_res.train_pred,
        arima_val=arima_res.val_pred,
        arima_test=arima_res.test_pred,
        lstm_train=lstm_res.train_pred,
        lstm_val=lstm_res.val_pred,
        lstm_test=lstm_res.test_pred,
        target_scaler=splits.target_scaler,
    )

    timings = {
        "arima_fit": arima_res.train_time_seconds,
        "arima_walk_forward": arima_res.walkforward_time_seconds,
        "lstm_train": lstm_res.train_time_seconds,
    }
    evaluation.evaluate(
        idx_test=splits.idx_test,
        y_test=splits.y_test_raw,
        arima_pred=hybrid.arima_test,
        hybrid_pred=hybrid.test,
        spike_test=splits.spike_test,
        timings=timings,
        out_dir=config.OUTPUT_DIR,
    )


def main() -> None:
    args = _parse_args()
    config.TARGET_COLUMN = args.target
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()
    if args.mode == "walkforward":
        run_walkforward_mode(args)
    elif args.mode == "forecast":
        run_forecast_mode(args)
    else:
        run_static_mode(args)
    print(f"\n[main] Total execution time: {time.perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    main()

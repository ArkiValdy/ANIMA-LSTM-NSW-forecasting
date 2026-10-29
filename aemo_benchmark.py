"""
AEMO pre-dispatch benchmark (plan Sec 5).

Loads the archived AEMO pre-dispatch price/demand forecasts
(data/raw/predispatch_{price,demand}_30min.csv) and evaluates them with the
SAME metrics on the SAME days as the walk-forward models.

Fairness note (state in Method): the MMSDM archive stores the FINAL
pre-dispatch run for each period (e.g. the 04:30 forecast was issued ~04:02
the same day), so AEMO's numbers here benefit from near-real-time information
and are expected to be optimistic versus a strictly day-ahead comparison. The
gap analysis in the Discussion should account for this horizon asymmetry.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_predispatch(kind: str) -> pd.Series:
    """Load archived pre-dispatch forecast for `kind` ('price' or 'demand')."""
    path = config.RAW_DATA_DIR / f"predispatch_{kind}_30min.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["value"].astype(float)


def merge_aemo(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Add an `aemo_pred` column to the walk-forward predictions frame, aligned
    on the forecast timestamps.
    """
    out = predictions.copy()
    aemo = load_predispatch(target)
    out["aemo_pred"] = aemo.reindex(out.index)
    return out


def aemo_metrics(predictions: pd.DataFrame, target: str, out_dir: Path) -> pd.DataFrame:
    """Evaluate AEMO pre-dispatch with the same segmented metrics."""
    import evaluation as ev

    df = merge_aemo(predictions, target)
    df = df.dropna(subset=["aemo_pred"])

    if df.empty:
        print("[aemo] No overlapping pre-dispatch rows — skipping benchmark.")
        return pd.DataFrame()

    models = [m for m in ("naive", "arima", "hybrid") if f"{m}_pred" in df.columns]
    table = ev._segmented_metrics(df, models + ["aemo"])
    ev._print_table(f"AEMO pre-dispatch benchmark — target: {target}", table)

    out_dir = ev._ensure_outdir(out_dir)
    table.to_csv(out_dir / f"aemo_benchmark_{target}.csv")
    return table


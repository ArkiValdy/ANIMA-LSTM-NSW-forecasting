"""
Compare two walk-forward runs of the same target on the same days, e.g. the
price hybrid with and without forecast-day demand.

    python compare_runs.py outputs/live/price_no_demand outputs/live/price_fday_demand \
        --labels no_demand fday_demand --target price

Prints overall / normal / spike RMSE and MAE for both hybrids plus the shared
naive and ARIMA baselines, and a paired t-test on daily RMSE differences.
Writes comparison_{target}.csv to the second run's folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def _segment_metrics(df: pd.DataFrame, col: str) -> dict:
    err = df["actual"] - df[col]
    out = {}
    for seg, mask in (
        ("overall", np.ones(len(df), bool)),
        ("normal", df["is_spike"].values == 0),
        ("spike", df["is_spike"].values == 1),
    ):
        e = err[mask]
        out[f"{seg}_RMSE"] = _rmse(e) if len(e) else np.nan
        out[f"{seg}_MAE"] = float(e.abs().mean()) if len(e) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--labels", nargs=2, default=["a", "b"])
    ap.add_argument("--target", default="price")
    args = ap.parse_args()
    la, lb = args.labels

    a = pd.read_csv(Path(args.run_a) / f"predictions_{args.target}.csv", index_col=0, parse_dates=True)
    b = pd.read_csv(Path(args.run_b) / f"predictions_{args.target}.csv", index_col=0, parse_dates=True)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    assert np.allclose(a["actual"], b["actual"]), "runs disagree on actuals"

    df = a[["actual", "is_spike", "naive_pred", "arima_pred"]].copy()
    df[f"hybrid_{la}"] = a["hybrid_pred"]
    df[f"hybrid_{lb}"] = b["hybrid_pred"]

    models = ["naive_pred", "arima_pred", f"hybrid_{la}", f"hybrid_{lb}"]
    table = pd.DataFrame({m.replace("_pred", ""): _segment_metrics(df, m) for m in models}).T

    n_days = df.index.normalize().nunique()
    print(f"\nTarget: {args.target}   window: {common.min().date()} .. {common.max().date()}  "
          f"({n_days} days, {int(df['is_spike'].sum())} spike periods)\n")
    print(table.round(2).to_string())

    # Paired t-test on daily RMSE: hybrid_b vs hybrid_a and vs baselines
    day = df.index.normalize()
    daily = {
        m: df.groupby(day).apply(lambda g, m=m: _rmse(g["actual"] - g[m]))
        for m in models
    }
    print("\nPaired t-tests on daily RMSE (mean_diff < 0 means the first model is better):")
    for x, y in (
        (f"hybrid_{lb}", f"hybrid_{la}"),
        (f"hybrid_{lb}", "naive_pred"),
        (f"hybrid_{lb}", "arima_pred"),
    ):
        d = daily[x] - daily[y]
        t, p = stats.ttest_rel(daily[x], daily[y])
        wins = int((d < 0).sum())
        print(f"  {x:>20} vs {y:<20} mean_diff={d.mean():+8.2f}  t={t:6.2f}  p={p:.4f}  "
              f"first better on {wins}/{len(d)} days")

    out = Path(args.run_b) / f"comparison_{args.target}.csv"
    table.to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

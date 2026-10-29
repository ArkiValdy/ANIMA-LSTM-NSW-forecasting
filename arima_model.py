"""
Auto-ARIMA fitting and rolling DAY-AHEAD prediction.

Day-ahead protocol (plan Sec 0/3): at the end of day D, forecast all
HORIZON_PERIODS values of day D+1 BEFORE seeing any of tomorrow's actuals,
then update the model state with the full observed day.

Refit schedule (compute reality):
  - order selection (auto_arima): quarterly
  - parameter refit: weekly (fresh fit with the fixed order on the window)
  - between refits: model.update(day_actuals, maxiter=0) absorbs observations
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pmdarima as pm
from sklearn.preprocessing import StandardScaler

import config

# maxiter=0 state updates intentionally skip optimisation, which makes
# statsmodels emit "ML failed to converge" warnings on every update. Suppress.
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", message="m \\(%i\\) set for non-seasonal fit")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")


def _auto_arima(y_train: np.ndarray):
    """auto_arima with the project's search ranges."""
    return pm.auto_arima(
        y_train,
        start_p=config.ARIMA_START_P,
        max_p=config.ARIMA_MAX_P,
        start_q=config.ARIMA_START_Q,
        max_q=config.ARIMA_MAX_Q,
        max_d=config.ARIMA_MAX_D,
        seasonal=config.ARIMA_SEASONAL,
        m=config.ARIMA_M,
        information_criterion=config.ARIMA_INFORMATION_CRITERION,
        stepwise=config.ARIMA_STEPWISE,
        trace=config.ARIMA_TRACE,
        suppress_warnings=config.ARIMA_SUPPRESS_WARNINGS,
        error_action="ignore",
        random_state=config.RANDOM_SEED,
    )


def select_order(y_train: np.ndarray) -> tuple:
    """Run auto_arima order selection (expensive — call quarterly)."""
    model = _auto_arima(y_train)
    order = tuple(model.order)
    print(f"[arima] auto_arima selected order {order} (AIC {model.aic():.2f})")
    return order


def select_order_scaled(y: np.ndarray) -> tuple:
    """select_order on a raw series, standard-scaled over its own window."""
    return select_order(StandardScaler().fit_transform(y.reshape(-1, 1)).ravel())


def fit_fixed_order(y_train: np.ndarray, order: tuple):
    """Fit ARIMA with a fixed order (weekly refit — fast)."""
    model = pm.ARIMA(order=order)
    model.fit(y_train)
    return model


def predict_day_ahead(model, n_periods: int = config.HORIZON_PERIODS) -> np.ndarray:
    """Forecast the full next day from the current model state."""
    return np.asarray(model.predict(n_periods=n_periods), dtype=np.float64)


def update_with_day(model, day_actuals: np.ndarray) -> None:
    """Absorb one observed day into the model state (maxiter=0 fast update)."""
    model.update(list(np.asarray(day_actuals).ravel()), maxiter=0)


# ----------------------------------------------------------------------------- #
# Static-split compatibility (smoke test only)
# ----------------------------------------------------------------------------- #
@dataclass
class ArimaResult:
    model: pm.arima.ARIMA
    order: tuple
    aic: float
    train_pred: np.ndarray
    val_pred: np.ndarray
    test_pred: np.ndarray
    train_residuals: np.ndarray
    val_residuals: np.ndarray
    train_time_seconds: float
    walkforward_time_seconds: float


def _walk_forward(
    model: pm.arima.ARIMA,
    actual: np.ndarray,
    refit_interval: int = 0,
) -> np.ndarray:
    """One-step-ahead rolling forecast (used by the static smoke test only)."""
    preds = np.empty(len(actual), dtype=np.float64)
    for i, y_obs in enumerate(actual):
        preds[i] = float(model.predict(n_periods=1)[0])
        if refit_interval and (i + 1) % refit_interval == 0:
            model.update([y_obs])
        else:
            model.update([y_obs], maxiter=0)
    return preds


def fit_and_predict(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_path: Optional[Path] = None,
) -> ArimaResult:
    """Static-split ARIMA (smoke test only; headline path is walk_forward.py)."""
    print("\n" + "=" * 60)
    print("ARIMA (static smoke test) — auto-selecting order")
    print("=" * 60)

    t0 = time.perf_counter()
    model = _auto_arima(y_train)
    train_time = time.perf_counter() - t0
    order = model.order
    aic = float(model.aic())
    print(f"\nSelected order : {order}")
    print(f"AIC            : {aic:.2f}")
    print(f"Fit time       : {train_time:.1f}s")

    train_pred = np.asarray(model.predict_in_sample())
    pad = len(y_train) - len(train_pred)
    if pad:
        # Back-fill the missing leading in-sample values with the first one
        first_valid = train_pred[0] if len(train_pred) else 0.0
        train_pred = np.concatenate([np.full(pad, first_valid), train_pred])

    print("\n[arima] Walk-forward forecasting on validation set...")
    t0 = time.perf_counter()
    val_pred = _walk_forward(model, y_val)
    print("[arima] Walk-forward forecasting on test set...")
    test_pred = _walk_forward(model, y_test)
    walk_time = time.perf_counter() - t0

    train_residuals = y_train - train_pred
    val_residuals = y_val - val_pred

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"[arima] Model saved to {model_path}")

    train_rmse = float(np.sqrt(np.mean(train_residuals ** 2)))
    val_rmse = float(np.sqrt(np.mean(val_residuals ** 2)))
    test_rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))
    print(
        f"[arima] RMSE (scaled): train={train_rmse:.4f}  "
        f"val={val_rmse:.4f}  test={test_rmse:.4f}"
    )

    return ArimaResult(
        model=model,
        order=order,
        aic=aic,
        train_pred=train_pred.astype(np.float32),
        val_pred=val_pred.astype(np.float32),
        test_pred=test_pred.astype(np.float32),
        train_residuals=train_residuals.astype(np.float32),
        val_residuals=val_residuals.astype(np.float32),
        train_time_seconds=train_time,
        walkforward_time_seconds=walk_time,
    )

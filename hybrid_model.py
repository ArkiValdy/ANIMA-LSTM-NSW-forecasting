"""
Combine ARIMA point forecasts with LSTM residual forecasts and inverse-scale
the result back to AUD/MWh.

hybrid_forecast = arima_prediction + lstm_residual_prediction   (in scaled space)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class HybridForecast:
    # Hybrid predictions in original AUD/MWh scale
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    # ARIMA-only predictions in original scale (for comparison)
    arima_train: np.ndarray
    arima_val: np.ndarray
    arima_test: np.ndarray


def _inverse(scaler: StandardScaler, arr: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(arr).reshape(-1, 1)).ravel()


def combine(
    arima_train: np.ndarray,
    arima_val: np.ndarray,
    arima_test: np.ndarray,
    lstm_train: np.ndarray,
    lstm_val: np.ndarray,
    lstm_test: np.ndarray,
    target_scaler: StandardScaler,
) -> HybridForecast:
    """
    Sum the ARIMA and LSTM components in scaled space, then inverse-transform
    both the hybrid result and the ARIMA-only baseline.
    """
    hybrid_train_scaled = arima_train + lstm_train
    hybrid_val_scaled = arima_val + lstm_val
    hybrid_test_scaled = arima_test + lstm_test

    return HybridForecast(
        train=_inverse(target_scaler, hybrid_train_scaled),
        val=_inverse(target_scaler, hybrid_val_scaled),
        test=_inverse(target_scaler, hybrid_test_scaled),
        arima_train=_inverse(target_scaler, arima_train),
        arima_val=_inverse(target_scaler, arima_val),
        arima_test=_inverse(target_scaler, arima_test),
    )

"""
Sequence LSTM that learns to predict the (scaled) residuals left over by ARIMA.

Architecture (day-ahead, plan Sec 4 "optional LSTM upgrade"):
  - Input A: (samples, 48, n_seq)  — the previous day's 48 feature vectors
  - Input B: (samples, n_ctx)      — forecast-day context (calendar + weather),
                                    broadcast across timesteps via RepeatVector
  - LSTM(64) -> Dropout -> Dense(48)  — one residual value per forecast period

Residual targets AND both input tensors are scaled per training window with
StandardScalers fit on the window only (no future data); raw inputs mix
demand in MW (~10^4) with calendar flags (0/1), which otherwise lets the
largest-magnitude features dominate. Monthly warm-start refits reuse weights.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Determinism: seed everything BEFORE importing tensorflow
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, models, optimizers, callbacks  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402


def _set_seeds(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class LstmSequenceModel:
    model: tf.keras.Model
    residual_scaler: StandardScaler
    train_time_seconds: float
    seq_scaler: StandardScaler
    ctx_scaler: StandardScaler


def _fit_feature_scaler(X: np.ndarray) -> StandardScaler:
    """Per-feature scaler over all (sample, period) rows of a 3-D tensor."""
    return StandardScaler().fit(X.reshape(-1, X.shape[-1]))


def _scale(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)


def _compile(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _callbacks(weights_path: Optional[Path]) -> list:
    """Early stopping on val_loss (+ best-weights checkpoint if a path is given)."""
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.LSTM_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        cb_list.append(
            callbacks.ModelCheckpoint(
                filepath=str(weights_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )
        )
    return cb_list


@dataclass
class LstmResult:
    model: tf.keras.Model
    history: dict
    train_pred: np.ndarray
    val_pred: np.ndarray
    test_pred: np.ndarray
    train_time_seconds: float


def build_model(n_seq_features: int, n_context_features: int) -> tf.keras.Model:
    """Two-input sequence model -> (samples, HORIZON) residual predictions.

    - sequence input: (B, 48, n_seq) — the previous day's 48 feature vectors
    - context input:  (B, 48, n_ctx+2) — forecast-day calendar + weather
      broadcast per period, plus an explicit per-period time-of-day encoding
      (built in numpy by preprocessing)

    The LSTM summarises the previous day; the summary is broadcast across the
    horizon, concatenated with the per-period context, and a shared
    TimeDistributed head predicts one residual per period (so the model always
    knows WHICH period of the day it is predicting).
    """
    seq_in = layers.Input(shape=(config.LSTM_TIMESTEPS, n_seq_features), name="sequence")
    ctx_in = layers.Input(shape=(config.HORIZON_PERIODS, n_context_features), name="context")

    x = layers.LSTM(config.LSTM_HIDDEN_UNITS, return_sequences=False)(seq_in)
    x = layers.Dropout(config.LSTM_DROPOUT)(x)

    summary = layers.RepeatVector(config.HORIZON_PERIODS)(x)   # (B, 48, H)

    x = layers.Concatenate(axis=-1)([summary, ctx_in])          # (B, 48, H+C)
    x = layers.TimeDistributed(layers.Dense(16, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)              # (B, 48, 1)
    out = layers.Flatten()(x)                                   # (B, 48)

    model = models.Model(inputs=[seq_in, ctx_in], outputs=out)
    return _compile(model)


def train_sequence_model(
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
    residuals: np.ndarray,
    X_seq_val: np.ndarray,
    X_ctx_val: np.ndarray,
    residuals_val: np.ndarray,
    weights_path: Optional[Path] = None,
    initial_weights: Optional[np.ndarray] = None,
    verbose: int = 1,
) -> LstmSequenceModel:
    """
    Train the sequence LSTM on (previous-day sequence + forecast-day context)
    -> day residuals. Residuals are standard-scaled per window.

    `initial_weights` enables warm-started monthly refits.
    """
    _set_seeds()

    # Residuals arrive as (n_days, HORIZON). Scale per value but keep the
    # day structure so X (n_days, ...) and y (n_days, HORIZON) line up.
    scaler = StandardScaler()
    scaler.fit(residuals.reshape(-1, 1))
    y = scaler.transform(residuals.reshape(-1, 1)).reshape(residuals.shape)
    y_val = scaler.transform(residuals_val.reshape(-1, 1)).reshape(residuals_val.shape)

    # Input scalers fit on the training tensors only, then applied to val
    seq_scaler = _fit_feature_scaler(X_seq)
    ctx_scaler = _fit_feature_scaler(X_ctx)
    X_seq, X_seq_val = _scale(seq_scaler, X_seq), _scale(seq_scaler, X_seq_val)
    X_ctx, X_ctx_val = _scale(ctx_scaler, X_ctx), _scale(ctx_scaler, X_ctx_val)

    model = build_model(
        n_seq_features=X_seq.shape[2],
        n_context_features=X_ctx.shape[2],
    )
    if initial_weights is not None:
        model.set_weights(initial_weights)

    cb_list = _callbacks(weights_path)

    t0 = time.perf_counter()
    history = model.fit(
        [X_seq, X_ctx],
        y,
        validation_data=([X_seq_val, X_ctx_val], y_val),
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=cb_list,
        shuffle=False,          # keep chronological order
        verbose=verbose,
    )
    train_time = time.perf_counter() - t0

    return LstmSequenceModel(
        model=model,
        residual_scaler=scaler,
        train_time_seconds=train_time,
        seq_scaler=seq_scaler,
        ctx_scaler=ctx_scaler,
    )


def predict_sequence(model: LstmSequenceModel, X_seq: np.ndarray, X_ctx: np.ndarray) -> np.ndarray:
    """Predict a day's residuals in RAW residual units (inverse-scaled)."""
    X_seq = _scale(model.seq_scaler, X_seq)
    X_ctx = _scale(model.ctx_scaler, X_ctx)
    # predict_on_batch: same output as predict() without its per-call
    # dataset/callback overhead (~40x faster for a single day)
    pred_scaled = np.asarray(model.model.predict_on_batch([X_seq, X_ctx])).reshape(-1, 1)
    return model.residual_scaler.inverse_transform(pred_scaled).ravel()


# ----------------------------------------------------------------------------- #
# Static-split compatibility (smoke test only)
# ----------------------------------------------------------------------------- #
def _to_3d(X: np.ndarray) -> np.ndarray:
    """Reshape (samples, features) -> (samples, timesteps=1, features)."""
    if X.ndim == 2:
        return X.reshape((X.shape[0], 1, X.shape[1]))
    return X


def build_model_static(n_features: int) -> tf.keras.Model:
    """Single-input LSTM for the static smoke test (timesteps=1)."""
    model = models.Sequential(
        [
            layers.Input(shape=(1, n_features)),
            layers.LSTM(config.LSTM_HIDDEN_UNITS),
            layers.Dropout(config.LSTM_DROPOUT),
            layers.Dense(1),
        ]
    )
    return _compile(model)


def train_and_predict(
    X_train: np.ndarray,
    residuals_train: np.ndarray,
    X_val: np.ndarray,
    residuals_val: np.ndarray,
    X_test: np.ndarray,
    weights_path: Optional[Path] = None,
    verbose: int = 0,
) -> LstmResult:
    """Static smoke-test path: single-step LSTM on residuals."""
    _set_seeds()

    Xtr = _to_3d(X_train)
    Xva = _to_3d(X_val)
    Xte = _to_3d(X_test)

    model = build_model_static(n_features=X_train.shape[1])

    cb_list = _callbacks(weights_path)

    t0 = time.perf_counter()
    history = model.fit(
        Xtr,
        residuals_train,
        validation_data=(Xva, residuals_val),
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=cb_list,
        shuffle=False,
        verbose=verbose,
    )
    train_time = time.perf_counter() - t0

    train_pred = model.predict(Xtr, verbose=0).ravel()
    val_pred = model.predict(Xva, verbose=0).ravel()
    test_pred = model.predict(Xte, verbose=0).ravel()

    return LstmResult(
        model=model,
        history=history.history,
        train_pred=train_pred.astype(np.float32),
        val_pred=val_pred.astype(np.float32),
        test_pred=test_pred.astype(np.float32),
        train_time_seconds=train_time,
    )

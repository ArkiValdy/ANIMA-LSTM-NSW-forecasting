"""
Central configuration for the ARIMA-LSTM hybrid electricity forecasting model.

Design (per IMPLEMENTATION_PLAN.md):
  - Rolling-origin DAY-AHEAD forecasting: at end of day D, forecast all 48
    half-hourly periods of day D+1, slide forward one day.
  - Option A features: only >=24h lags + calendar + weather (no <24h lags),
    directly comparable to AEMO pre-dispatch.
  - Demand is the primary target, price secondary; target selectable via CLI.
  - Per-window StandardScaler (never fit on future data); no reverse-chrono
    demo hacks.
"""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Default input files
PRICE_DEMAND_CSV = RAW_DATA_DIR / "price_demand_30min.csv"
TEMPERATURE_CSV = RAW_DATA_DIR / "temperature_hourly.csv"
# Historical + live files written by data/update_live.py (forecast mode default)
LIVE_PRICE_DEMAND_CSV = RAW_DATA_DIR / "price_demand_30min_live.csv"
LIVE_TEMPERATURE_CSV = RAW_DATA_DIR / "temperature_hourly_live.csv"

# Region of interest
REGION = "NSW1"

# Half-hourly periods per day
PERIODS_PER_DAY = 48

# --------------------------------------------------------------------------- #
# Forecast task: rolling day-ahead
# --------------------------------------------------------------------------- #
# Window of history used for training/refitting (in days).
TRAIN_DAYS = 365
# Horizon: forecast the full next day (48 periods) before seeing any actuals.
HORIZON_PERIODS = PERIODS_PER_DAY
# Eval window (inclusive start). End = last available day in the data.
EVAL_START = "2026-01-01"
# Refit schedule (compute reality, plan Sec 3):
#   - ARIMA order selection: quarterly (or once — see ORDER_SELECT_ONCE)
#   - ARIMA parameter refit: weekly
#   - LSTM warm-start refit: monthly
ARIMA_ORDER_REFIT_DAYS = 91
# auto_arima on a 17.5k-point half-hourly window is pathologically slow
# (~1.5-96 min depending on convergence retries). The plan (Sec 3) allows
# selecting the order on the FIRST window only; True = select once and keep
# the order fixed for the whole evaluation.
ORDER_SELECT_ONCE = True
ARIMA_REFIT_DAYS = 7
LSTM_REFIT_DAYS = 30
# Length of the ARIMA day-ahead residual history replayed to build LSTM
# training samples. Replayed with weekly ARIMA refits (cheap), so a larger
# window gives the LSTM more day-pairs without much cost.
LSTM_RESIDUAL_DAYS = 250

# Base forecast the LSTM residual model is trained on top of:
#   "arima" -> hybrid = ARIMA + LSTM(ARIMA day-ahead residuals)  (plan Sec 4)
#   "naive" -> hybrid = naive + LSTM(naive residuals)            (empirically
#              stronger: ARIMA reverts to a level at 48 steps, so the LSTM
#              learns weather/weekday corrections to same-period-yesterday)
HYBRID_BASE = "naive"

# --------------------------------------------------------------------------- #
# Target / spike configuration
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "demand"          # "demand" (primary) or "price" (secondary)
# Spike threshold (AUD/MWh) for spike-period evaluation of the price target.
SPIKE_THRESHOLD = 130.0
# For the demand target, "spike" periods are the top-quantile demand periods.
PEAK_DEMAND_QUANTILE = 0.98

# Outlier detection (variable-threshold sliding window)
OUTLIER_WINDOW = 144      # 3 days * 48 half-hours
OUTLIER_K = 2.5           # number of standard deviations

# Heating / cooling degree base temperature (Celsius)
BASE_TEMPERATURE = 18.0

# --------------------------------------------------------------------------- #
# Features (Option A: only >=24h lags + calendar + weather)
# --------------------------------------------------------------------------- #
# Lag offsets in DAYS. All are >= 24h so they are known at day-ahead time.
PRICE_LAG_DAYS = [1, 2, 7, 14, 21]
DEMAND_LAG_DAYS = [1, 2, 7, 14, 21]

# Rolling-statistic windows (in days), computed on data up to the previous day.
ROLLING_WINDOWS_DAYS = [7, 14]

# Features that are common to both targets (calendar + weather).
CALENDAR_FEATURES = ["day_of_week", "month", "is_weekend", "is_holiday"]
WEATHER_FEATURES = ["hdd", "cdd"]
# Time-of-day cyclic features (sin/cos of half-hour index).
TIME_FEATURES = ["hour_sin", "hour_cos"]

# Features used for the LSTM sequence input per period (excluding target-day
# context, which is broadcast across the sequence). Built dynamically in
# preprocessing based on the active target.
SEQ_FEATURE_SUFFIXES = (
    [f"price_lag_{d}d" for d in PRICE_LAG_DAYS]
    + [f"demand_lag_{d}d" for d in DEMAND_LAG_DAYS]
    + [f"price_roll_mean_{d}d" for d in ROLLING_WINDOWS_DAYS]
    + [f"price_roll_std_{d}d" for d in ROLLING_WINDOWS_DAYS]
    + [f"demand_roll_mean_{d}d" for d in ROLLING_WINDOWS_DAYS]
    + [f"demand_roll_std_{d}d" for d in ROLLING_WINDOWS_DAYS]
    + TIME_FEATURES
)
# Target-day context broadcast across the sequence: calendar + weather of the
# forecast day (weather = actual as "perfect forecast" proxy, stated in Method).
CONTEXT_FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES

# Forecast-day demand as a per-period input to the PRICE model (CLI:
# --fday_demand). Uses actual demand as a "perfect forecast" proxy, the same
# assumption as weather, so results are an upper bound on the value of demand
# information. Both features are scale-free:
#   fday_demand_delta_gw: demand minus same period yesterday, in GW
#   fday_demand_z:        demand standardised by the 7-day rolling mean/std
#                         known at the end of the previous day
PRICE_USE_FDAY_DEMAND = False
FDAY_DEMAND_FEATURES = ["fday_demand_delta_gw", "fday_demand_z"]

# --------------------------------------------------------------------------- #
# ARIMA settings (auto_arima search ranges)
# --------------------------------------------------------------------------- #
ARIMA_START_P = 0
ARIMA_MAX_P = 5
ARIMA_START_Q = 0
ARIMA_MAX_Q = 5
ARIMA_MAX_D = 2
ARIMA_SEASONAL = False        # set True if you want SARIMA; expensive on half-hourly
ARIMA_M = 48                  # daily seasonality (used only if ARIMA_SEASONAL=True)
ARIMA_INFORMATION_CRITERION = "aic"
ARIMA_STEPWISE = True
ARIMA_TRACE = False
ARIMA_SUPPRESS_WARNINGS = True

# --------------------------------------------------------------------------- #
# LSTM settings (sequence model)
# --------------------------------------------------------------------------- #
LSTM_HIDDEN_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 60
LSTM_BATCH_SIZE = 256
LSTM_PATIENCE = 8
# Sequence length in periods = the full previous day (48 half-hours).
LSTM_TIMESTEPS = PERIODS_PER_DAY

# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
PLOT_DPI = 120
PLOT_FIGSIZE = (12, 5)

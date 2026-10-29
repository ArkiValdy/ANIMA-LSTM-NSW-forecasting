# Implementation Plan — ARIMA-LSTM Capstone (Supervisor Feedback, Aug 2026)

**Scope:** Turn the current single-month April 2026 pipeline into a rolling one-year, day-ahead
forecasting study for **demand (primary target) and price (secondary target)**, with exogenous
weather and lagged-price features, benchmarked against AEMO's own pre-dispatch forecasts.

This maps the three supervisor points onto the existing codebase
(`config.py`, `data_loader.py`, `preprocessing.py`, `arima_model.py`, `lstm_model.py`,
`hybrid_model.py`, `evaluation.py`, `main.py`).

---

## 0. Define the forecasting task precisely (do this first — everything depends on it)

The supervisor's "use a year of data and predict the next day, continuing on" is
**rolling-origin day-ahead forecasting**:

> At the end of day *D*, using the previous 365 days of half-hourly data, forecast all
> 48 half-hourly values of day *D+1*. Slide forward one day. Repeat over the whole
> evaluation period.

Two consequences worth writing into the Method section now:

1. **The current pipeline is one-step-ahead (30 min), not day-ahead.**
   `arima_model._walk_forward()` predicts one period then absorbs the actual via
   `model.update()`. For day-ahead you predict `n_periods=48` *before* seeing any of
   tomorrow's actuals, then update with the full observed day.
2. **Feature availability changes.** A 1-hour price lag is only known for the first
   forecast step of the day. For a day-ahead horizon, usable inputs are: lags of 24h or
   older, calendar features, and weather. Two clean options — pick one and state it:
   - **Option A (recommended): pure day-ahead.** Use only ≥24h lags + calendar + weather.
     Directly comparable to AEMO pre-dispatch. Drop the 1h lag from the day-ahead model.
   - **Option B: dual setup.** Report both a 1-step (30-min) model — where the 1h lag is
     legitimate — and the day-ahead model. More work, but shows the supervisor's
     "24h/1h ago" idea in the setting where each lag is valid.

**Decision needed from you/supervisor:** A or B. The plan below assumes A, with B as a
stretch goal.

---

## 1. Data acquisition (blocking — start immediately)

| Dataset | Source | Use |
|---|---|---|
| Half-hourly price + demand, NSW1, ~18 months (e.g. Jan 2025 → Jun 2026; 12 months training warm-up + ≥6 months rolling evaluation) | AEMO Aggregated Price & Demand (monthly CSVs, same schema as your current file) | Model inputs + actuals |
| Hourly/half-hourly temperature, Sydney Observatory Hill (066214) or airport AWS | BOM half-hourly observations (or NEMWeb `ROOFTOP_PV`/weather alternatives) | Replaces the daily-max broadcast — your proposal already flags this limitation; supervisor's "integrate weather" point |
| AEMO pre-dispatch forecasts (price: `PREDISPATCHPRICE`; demand: `PREDISPATCHREGIONSUM`) for the same evaluation window | NEMWeb MMS archive (monthly zips) | The "compare with AEMO" benchmark |

Notes:
- Concatenating monthly AEMO CSVs needs only a thin loader change — `load_price_demand()`
  already handles the schema; add a "load a folder of CSVs" path.
- For each target day, extract the AEMO pre-dispatch run issued ~1 day ahead (e.g. the
  12:30 pm run for the next trading day) so the comparison horizon matches yours.
- Keep raw downloads in `data/raw/` untouched; document retrieval dates (rubric:
  data recorded accurately, artefacts kept for verification).

## 2. Generalise the target: demand primary, price secondary

- `config.py`: replace the single `TARGET_COLUMN = "price"` with a `TARGET` setting
  (`"demand"` or `"price"`) selectable from the CLI (`main.py --target demand`).
- `preprocessing.py`: build lag/rolling features for *both* series (mostly already done);
  when target = demand, price lags (≥24h) become exogenous inputs — exactly the
  supervisor's "price of electricity 24 hrs ago" point. Spike flags stay price-based for
  the price runs; for demand runs, report peak-demand periods instead.
- Everything downstream already works off scaled arrays, so this is mostly config work.
- Run the full study twice: demand (primary, headline results) and price (secondary,
  connects to the proposal's literature review).

## 3. Walk-forward engine (the core build)

New module `walk_forward.py` orchestrating the rolling loop; `main.py` gains
`--mode walkforward`.

- **Windows:** train = 365 days, horizon = 48 periods, step = 1 day. With 6 months of
  evaluation that's ~180 forecast days ≈ 8,640 forecast points per target.
- **Refit schedule (compute reality):** a full `auto_arima` + LSTM fit per day × 180 days
  is not feasible on a laptop. Standard compromise, worth stating in Method:
  - Order selection: run `auto_arima` once per quarter (or on the first window only);
    keep the order fixed in between.
  - ARIMA parameters: refit weekly; between refits, roll the window forward with
    `model.update(day_actuals)` (your `ARIMA_REFIT_INTERVAL` config hook already
    anticipates this).
  - LSTM: refit monthly with warm-started weights; predict daily.
  - Log wall-clock per component — this becomes the computational-cost table (Table 3 of
    the proposal) with real numbers.
- **Scaling fix:** scalers must be refit on each training window only (never on data from
  the future). Current `prepare()` fits once globally — refactor so the walk-forward loop
  owns scaler fitting. Consider `StandardScaler` over `MinMaxScaler`: a min-max fitted on
  a window breaks when a bigger spike arrives later.
- **Remove `REVERSE_CHRONO_FOR_SPIKE_DEMO`** and the static-split path for headline
  results (keep the static mode only as a quick smoke test). A year of data makes the
  hack unnecessary, and reversed-time evaluation would be indefensible in the report.
- **Restore full lags** (`[1, 2, 7, 14, 21]` days) and rolling windows (`[7, 14]` days)
  now that warm-up is affordable; add the 24h lag explicitly (already `lag_1d`).
- Persist per-day forecasts to `outputs/walkforward/predictions.parquet`
  (timestamp, actual, arima, hybrid, aemo, target, spike/peak flag) — one tidy file that
  every report figure is generated from.

## 4. Exogenous features (weather + lagged price)

- **LSTM side (easy):** hourly temperature, HDD/CDD, ≥24h price and demand lags, calendar
  features — extend `FEATURE_COLUMNS`; the residual-LSTM design is unchanged.
- **ARIMA side (choose one, justify in Method):**
  - **Keep ARIMA univariate** (current behaviour): the hybrid's non-linear/exogenous
    effects all live in the LSTM. Defensible, matches Kollia & Kollias, zero extra code.
  - **Upgrade to SARIMAX with exog** (`pm.auto_arima(y, X=exog)`): weather enters the
    linear component too; day-ahead prediction then needs day-ahead exog values — use
    actual temperature as a "perfect weather forecast" proxy and state the assumption
    (standard practice in the load-forecasting literature).
  - Recommendation: start univariate, add SARIMAX as a comparison run if time allows —
    that contrast is itself a reportable finding.
- **Optional LSTM upgrade** (stretch): move from `timesteps=1` feature vectors to true
  sequence input (e.g. last 48 periods). Only attempt after the walk-forward engine works.

## 5. AEMO benchmark comparison

New module `aemo_benchmark.py`:

- Parse pre-dispatch archives, align each target day's ~day-ahead forecast to your
  forecast timestamps.
- Evaluate AEMO's forecasts with **the same metrics on the same days** as ARIMA, hybrid
  and (optionally) a naive seasonal baseline (demand at same period yesterday/last week —
  cheap and makes every comparison more credible).
- Headline report table: rows = Naive / ARIMA / Hybrid / AEMO pre-dispatch;
  columns = RMSE, MAE, MAPE × overall / peak / spike segments.
- Framing for the report: AEMO's demand forecasts are strong — *on par* is already a
  solid result for a capstone, and *why* the gap exists (or doesn't) is Discussion gold.
  Don't over-promise "better"; promise a fair, well-analysed comparison.

## 6. Evaluation & report outputs

Extend `evaluation.py` to aggregate across walk-forward days:

- Metrics by month/season and by weekday/weekend → robustness dimension from the proposal.
- Error-by-hour-of-day profile (demand forecasting classic; shows where the LSTM helps).
- Rolling 30-day RMSE line per model → shows stability over the year (adaptive-retraining
  story from the proposal's "improvements" paragraph).
- Forecast-vs-actual overlays for best/worst days; scatter plots (existing code reusable).
- Keep Diebold-Mariano or at least paired-difference significance testing in mind for
  "the two models are statistically close" claims — one t-test on daily RMSE differences
  is cheap and lifts the Findings rigour noticeably.
- Every figure saved from the tidy predictions file with a fixed style — matches the
  rubric's "figures integrated, referred to, APA-captioned".

## 7. Report mapping (write as you go)

- **Method:** the precise task definition (§0), data provenance (§1), walk-forward
  protocol + refit schedule (§3), feature sets (§4), benchmark alignment (§5). Step-by-step
  is exactly what the rubric's Method row asks for.
- **Findings:** the §5 headline table + §6 figures, demand first, price second.
- **Discussion:** why the hybrid does/doesn't beat ARIMA and AEMO under NEM conditions;
  contrast with Kollia & Kollias' 85% claim; context-dependence thesis from your lit review.
- **Pass/fail criteria:** log every supervisor meeting (this one included — these three
  points *are* your "seek and respond to feedback" evidence; write down what you changed
  in response) and keep the communication log current.

## Suggested order of work

1. Data acquisition (§1) — start downloads now; everything else blocks on it.
2. Target generalisation + config cleanup (§2, removing the demo-reverse hack).
3. Walk-forward engine with ARIMA only (§3) — get the loop right before adding parts.
4. LSTM residual component inside the loop; exogenous features (§4).
5. AEMO benchmark (§5), then evaluation aggregation (§6).
6. Price as secondary target = re-run, not re-build.
7. Report drafting (§7) in parallel from step 3 onward.

## Risks

- **Compute:** 180 walk-forward days × two targets is hours, not minutes. The refit
  schedule in §3 is the mitigation; also cache fitted models per window.
- **Pre-dispatch data wrangling** is the fiddliest data task (MMS table formats). Budget
  a full day; fall back to demand-only benchmark if price pre-dispatch parsing drags.
- **MinMax scaler under regime change** — switch to standard scaling per window (§3).

---

## Completion status (Aug 2026) — core §1–§6 implemented

Everything below is built, verified end-to-end, and reproducible via `main.py`:

| Item | Status |
|---|---|
| §1 Data: AEMO price/demand 2025-01→2026-07 (`data/raw/price_demand_30min.csv`, 26,209 rows), Open-Meteo hourly temp (`temperature_hourly.csv`), AEMO pre-dispatch (`predispatch_{price,demand}_30min.csv`), raw zips archived in `data/raw/zips/` | ✅ |
| §2 Target generalisation: `--target demand|price`, target-aware spike/peak flags, reverse-chrono hack removed | ✅ |
| §3 Walk-forward engine (`walk_forward.py`): 365d train, 48-period horizon, weekly ARIMA refit + daily update, monthly warm-start LSTM, per-window StandardScaler, tidy predictions + timings CSVs | ✅ |
| §4 Features (Option A): ≥24h lags [1,2,7,14,21]d, rolling [7,14]d, calendar + weather + time-of-day, LSTM upgraded to true 48-period sequence input with per-period encoding and TimeDistributed head | ✅ |
| §5 Benchmarks: seasonal naive + AEMO pre-dispatch compared with same metrics (fairness caveat: archive stores final pre-dispatch runs) | ✅ |
| §6 Evaluation: monthly / weekday-weekend / error-by-hour / rolling-30d RMSE, paired t-tests, figures | ✅ |

**Result — demand (primary target), 182 rolling day-ahead days:**
`hybrid RMSE 730.4` vs `naive 758.9` vs `arima 1206.9` — hybrid beats naive (p=0.035) and ARIMA (p<0.0001) on paired daily RMSE differences. AEMO final pre-dispatch 250.9 (uses near-real-time info, not strictly day-ahead).

**Result — price (secondary target), same window:**
`arima RMSE 172.7` best on RMSE, `naive MAE 32.4` best on MAE; the naive-base hybrid overcorrects on spike days (RMSE 253). Matches the plan's expectation that price is the harder, spike-dominated target; the hybrid-vs-baseline contrast is itself the Discussion finding.

**Key design decision made during implementation (deviates from §4's "ARIMA base" default):**
ARIMA(2,1,1) reverts to a level over a 48-period horizon, so the residual LSTM was being asked to reconstruct the entire daily pattern. Switching the residual base to the seasonal naive (`HYBRID_BASE = "naive"` in config) makes the LSTM learn weather/weekday corrections instead — this is what delivers the significant demand win.

**Compute notes (Table 3 material):** auto_arima order selection on a 17.5k-point window is pathologically slow (up to ~96 min on some windows) → `ORDER_SELECT_ONCE = True` selects on the first window only; full demand run then takes ~7 min, price ~6 min. ARIMA daily state update ~1.4s/day is the dominant per-day cost.
- **Scope:** Option B (dual horizon), SARIMAX comparison and sequence-LSTM are stretch
  goals. The §0–§6 core alone satisfies all three supervisor points.

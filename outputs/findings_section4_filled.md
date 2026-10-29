# 4. Findings

<!-- Filled from the archived predictions files on 2026-10-13. Items marked CHECK are places where the
     archived results disagree with the draft wording, and items marked PENDING have no data yet. -->

This section reports the results of the two evaluation modes defined in Section 3.1, with the backtest results in Sections 4.1 to 4.3, the live out of sample results in Section 4.4, and computational cost in Section 4.5. Backtest findings carry the statistical comparisons that address O2 and O3, live findings address generalisation, and every figure and table is generated from the archived predictions files so the exhibits can be reproduced without re-running the pipeline.

## 4.1 Demand Forecasting

The hybrid is the most accurate of the three models built in this project on demand, with an overall RMSE of 724 MW against 759 MW for the seasonal naive baseline and 1,208 MW for ARIMA(2,1,1), but its margin over the naive baseline is not statistically significant and AEMO pre-dispatch is far ahead of all three at 251 MW. Table [4] reports RMSE, MAE, and MAPE for the seasonal naive baseline, ARIMA(2,1,1), the hybrid, and AEMO pre-dispatch on identical forecast days, segmented into overall, normal, and peak periods. The table covers 181 forecast days from 1 January to 30 June 2026 and 8,679 half hourly periods, of which 94 are peak periods.

| Model | Segment | RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|---|---|
| Seasonal naive | Overall | 759.3 | 499.1 | 6.81 |
| | Normal | 744.4 | 490.2 | 6.76 |
| | Peak | 1,618.4 | 1,312.2 | 11.08 |
| ARIMA(2,1,1) | Overall | 1,207.5 | 928.0 | 12.72 |
| | Normal | 1,147.1 | 897.7 | 12.52 |
| | Peak | 3,802.0 | 3,701.8 | 31.29 |
| Hybrid | Overall | 724.0 | 492.0 | 6.69 |
| | Normal | 712.4 | 485.1 | 6.66 |
| | Peak | 1,432.5 | 1,122.9 | 9.47 |
| AEMO pre-dispatch | Overall | 250.9 | 193.5 | 2.57 |
| | Normal | 248.7 | 192.0 | 2.57 |
| | Peak | 406.8 | 332.2 | 2.83 |

The paired t tests on daily RMSE described in Section 3.8 confirm two of the three differences among the project's own models at the 5% level. The hybrid is better than ARIMA by a mean of 529 MW per day (t = −17.36, p < 0.001, n = 181) and ARIMA is worse than the naive baseline by 510 MW (t = 17.17, p < 0.001), but the hybrid's mean advantage of 19 MW over the naive baseline is not significant (t = −1.16, p = 0.249), and the hybrid has the lower daily RMSE on only 91 of 181 days. AEMO pre-dispatch is significantly better than every model, with a mean daily advantage of 379 MW over the hybrid (t = −14.04, p < 0.001). The pre-dispatch series is the final run for each period and not a day ahead issue, so this gap overstates the difference at a matched horizon. The hybrid's gain is largest in the peak segment, where its RMSE of 1,433 MW is 11% below the naive baseline.

Figure [4] plots rolling 30 day RMSE for each model across the evaluation window, which shows that ARIMA is the worst model throughout while the ordering of the hybrid and the naive baseline is season dependent. The hybrid sits below the naive baseline in 65% of the rolling windows, trailing it through January and running level with it by the end of June. Figure [5] profiles mean absolute error by hour of day, which locates each model's errors within the daily cycle, and the hybrid improves on the naive baseline in every hour from 05:00 to 19:00, with the largest gains of 67 to 83 MW across the morning ramp from 06:00 to 08:00 and smaller gains of 7 to 27 MW across the evening peak from 17:00 to 19:00. The hybrid is worse than the naive baseline overnight, by 13 to 78 MW between 20:00 and 04:00. Table [5] reports RMSE by month, which carries the robustness dimension of O4 and feeds the seasonal caveat at the end of this section.

| Month | Seasonal naive | ARIMA(2,1,1) | Hybrid | AEMO pre-dispatch |
|---|---|---|---|---|
| January | 867.2 | 1,461.6 | 876.7 | 259.4 |
| February | 1,036.2 | 1,331.6 | 1,002.2 | 270.9 |
| March | 730.9 | 1,103.4 | 662.8 | 256.1 |
| April | 609.4 | 1,254.7 | 525.5 | 246.5 |
| May | 597.4 | 985.0 | 553.4 | 220.2 |
| June | 638.5 | 1,050.9 | 621.3 | 251.3 |

The hybrid has the lowest RMSE of the project's models in five of the six months, with January the exception.

## 4.2 Price Forecasting

<!-- CHECK: the draft's headline "the hybrid performs worst of the compared models" holds on MAE and MAPE
     and among the three project models on RMSE, but AEMO pre-dispatch has the highest overall RMSE
     (325.0). The sentence below is worded to match the numbers.
     CHECK: the price ARIMA order selected by auto_arima was (0,1,5), not (2,1,1). (2,1,1) is the demand order. -->

The price results follow the same layout, and the headline finding is stated plainly, which is that the hybrid performs worst of the project's three models on every overall metric and significantly worse than the naive base it corrects. Table [6] reports the same metric set over the same forecast days, segmented into overall, normal, and spike periods as defined in Section 3.3. The table covers the same 8,679 periods, of which 562 are spike periods.

| Model | Segment | RMSE (A$/MWh) | MAE (A$/MWh) | MAPE (%) |
|---|---|---|---|---|
| Seasonal naive | Overall | 237.2 | 32.4 | 121.0 |
| | Normal | 171.4 | 25.8 | 126.8 |
| | Spike | 667.0 | 127.5 | 36.8 |
| ARIMA(0,1,5) | Overall | 172.8 | 35.9 | 248.0 |
| | Normal | 45.1 | 28.9 | 262.3 |
| | Spike | 656.9 | 136.7 | 41.8 |
| Hybrid | Overall | 249.1 | 68.9 | 399.5 |
| | Normal | 191.5 | 63.4 | 423.5 |
| | Spike | 654.5 | 147.7 | 52.5 |
| AEMO pre-dispatch | Overall | 325.0 | 24.3 | 54.6 |
| | Normal | 194.6 | 15.1 | 55.1 |
| | Spike | 1,041.1 | 157.7 | 46.5 |

The paired t tests confirm the hybrid's deficit. Its daily RMSE is higher than that of its own seasonal naive base by a mean of 36.8 A$/MWh (t = 10.72, p < 0.001, n = 181), higher than ARIMA by 46.4 A$/MWh (t = 4.55, p < 0.001), and higher than AEMO pre-dispatch by 40.6 A$/MWh (t = −2.37 for pre-dispatch minus hybrid, p = 0.019), and the hybrid beats the naive baseline on only 22 of 181 days. The difference between ARIMA and the naive baseline is not significant (p = 0.326), and neither are the differences between pre-dispatch and ARIMA (p = 0.667) or pre-dispatch and the naive baseline (p = 0.816). AEMO pre-dispatch has the lowest MAE of any model at 24.3 A$/MWh and the lowest daily RMSE on most days, but it has the highest overall RMSE at 325.0 A$/MWh, because its errors in the extreme intervals of January and February are larger than those of any other model. The MAPE values are inflated for every model by prices near zero and are reported for completeness.

This ordering means the LSTM residual component subtracted value on price over this window, since the hybrid sits below the base it corrects, and the result is reported as found rather than softened, because a negative finding under a fair protocol is itself an answer to the research question posed in Section 1.2. Figure [6] plots rolling 30 day RMSE, Figure [7] profiles error by hour of day, and Table [7] reports monthly RMSE, which together show that the deficit is spread evenly across the day and the year and does not concentrate in spike periods. The hybrid's rolling RMSE is above the naive baseline's in every window and it is the worst of the three models in 83% of them. Its mean absolute error exceeds the naive baseline's in every hour of the day by a near constant 28 to 45 A$/MWh. The deficit arises in normal periods, where the hybrid's MAE of 63.4 A$/MWh is more than double the naive baseline's 25.8 A$/MWh, while in spike periods its RMSE of 654.5 A$/MWh is marginally below the naive baseline's 667.0 A$/MWh. In relative terms the hybrid is weakest in the calm months of March and June, where its RMSE is roughly two and a half times the naive baseline's, and the LSTM correction over the window behaves as a level offset, averaging about +34 A$/MWh in January to March and −35 A$/MWh in June.

| Month | Seasonal naive | ARIMA(0,1,5) | Hybrid | AEMO pre-dispatch |
|---|---|---|---|---|
| January | 486.5 | 351.4 | 499.1 | 598.9 |
| February | 314.0 | 226.8 | 333.8 | 534.6 |
| March | 28.1 | 32.9 | 69.0 | 19.8 |
| April | 25.3 | 36.1 | 30.6 | 20.3 |
| May | 43.0 | 41.1 | 54.4 | 17.4 |
| June | 27.1 | 28.8 | 72.0 | 16.9 |

The January and February values are dominated by a small number of extreme days, with half hourly prices reaching 10,047 A$/MWh on 10 January and 6,342 A$/MWh on 5 February.

## 4.3 Effect of Exogenous Demand on Price Forecasts

<!-- CHECK: the 50.3 to 46.0 figures in the draft come from the July to October walk forward on live data
     (outputs/live/), not from the January to June backtest, and the difference is not significant
     (p = 0.232). On the backtest window the effect is 249.1 to 246.1 with a worse MAE. The draft's
     "improves price accuracy materially" is not supported by either window, so this subsection is
     rewritten around what the two runs show. -->

Adding the forecast day's demand as an exogenous input does not produce a reliable improvement in price accuracy. On the backtest window the hybrid's overall RMSE falls from 249.1 to 246.1 A$/MWh, a reduction of 1.2%, but its MAE rises from 69.0 to 77.2 A$/MWh and its daily RMSE is higher by a mean of 8.9 A$/MWh, a difference that is significant in the wrong direction (t = 2.81, p = 0.006, n = 181). On the walk forward over 1 July to 4 October 2026 the overall RMSE falls from 50.3 to 46.0 A$/MWh, a reduction of 8.5%, and MAE falls from 33.2 to 31.8 A$/MWh, but the paired t test on daily RMSE does not confirm the difference (mean difference −1.5 A$/MWh, t = −1.20, p = 0.232, n = 96), and the demand informed hybrid has the lower daily RMSE on only 46 of 96 days. In both windows the hybrid with demand remains less accurate than the naive baseline, which records 237.1 A$/MWh on the backtest window and 32.6 A$/MWh on the later one.

These runs gave the price model the actual demand of the day being forecast, so the figures are an upper bound on the achievable gain rather than an operational result, because a deployed forecaster would have to use a demand forecast in place of the actual and would inherit that forecast's errors. The size of the bound still carries information, since it measures how much of the price error is attributable to not knowing demand, and the answer over these windows is that little of it is. It also connects the two forecasting tasks in this report, because the demand accuracy reported in Section 4.1 could only reduce a gain that perfect demand information does not deliver consistently. The chained run using the hybrid's own demand forecast as the exogenous input was not performed.

## 4.4 Live Out of Sample Results

<!-- CHECK: the scheduled 21:00 job started on 12 October. The only scored day ahead issue is 10 October,
     which was issued at 19:28 on 9 October, not 21:00. No day ahead forecast exists for 11 or 12 October.
     PENDING: actuals for 13 October arrive with tonight's 21:00 run (provisional) and tomorrow's (final). -->

The live track provides the out of sample evidence that no backtest can, for the reason given in Section 3.6, and its results are descriptive per Section 3.8. Table [8] reports daily RMSE and MAE for each live forecast day from the first true day ahead issue onward, for the hybrid and the pre-dispatch benchmark captured at the same issue time, and states whether each day is scored against provisional or final actuals. At the time of writing the track holds two issues, of which one has been scored.

| Date | Issued | Hybrid RMSE | Hybrid MAE | Pre-dispatch RMSE | Pre-dispatch MAE | Actuals |
|---|---|---|---|---|---|---|
| 10 October | 19:28, 9 October | 606.8 | 186.9 | 594.8 | 177.0 | final |
| 13 October | 21:03, 12 October | [pending] | [pending] | [pending] | [pending] | [pending] |

Values are in A$/MWh. On 10 October both forecasters are dominated by a single interval at 07:00 that settled at 4,091 A$/MWh, against a hybrid forecast of 35 A$/MWh and a pre-dispatch value of 43 A$/MWh. With that interval excluded the hybrid's RMSE is 160.8 A$/MWh and pre-dispatch's is 112.2 A$/MWh. The hybrid also carried a 972 A$/MWh value at 05:00 that it inherited from the previous day's spike through its naive base, where the actual was 114 A$/MWh. Figure [8] overlays forecast and actual price for the live window, which shows that the hybrid tracked the shape of the afternoon and evening on 10 October while sitting 20 to 80 A$/MWh below the actual price, and that pre-dispatch sat 130 to 200 A$/MWh above it from 16:30 onward.

The 12 to 13 October episode serves as the live case study, because its intervals reached the extreme prices that the backtest identifies as the hardest segment. On 12 October the half hourly price rose to 4,384 A$/MWh at 14:00, 13,017 A$/MWh at 14:30, 5,212 A$/MWh at 15:00, and 2,064 A$/MWh at 15:30, and then settled between 179 and 395 A$/MWh from 16:00 to 20:00. No day ahead forecast was issued for 12 October, because the scheduled job began that evening, so the episode is assessed on the forecast for 13 October issued at 21:03 on 12 October. The hybrid forecast repeats the previous afternoon almost exactly, with 4,475, 13,109, 5,305, and 2,159 A$/MWh from 14:00 to 15:30 and between 249 and 370 A$/MWh across the evening, because its naive base copies the previous day and the LSTM correction adds only 60 to 95 A$/MWh. Pre-dispatch at the same issue time carried 199 to 299 A$/MWh across that afternoon window and placed the scarcity in the evening, with 20,111 A$/MWh at 18:00, 18:30, and 19:30 and about 1,100 A$/MWh at 17:30 and 19:00. ARIMA stayed between 32 and 171 A$/MWh all day. [PENDING: actual prices for 13 October and the resulting errors for each forecaster.] The episode is informative in both directions, since it shows how a forecaster issued before an extreme event behaves when the event arrives, and it is exactly the class of interval that Section 1.2 argues aggregate metrics conceal. [PENDING: the conclusion the episode supports, which depends on whether the 13 October extreme intervals fell in the afternoon as the hybrid assumed, in the evening as pre-dispatch indicated, or did not occur.]

## 4.5 Computational Cost

<!-- CHECK: the archived timing logs show auto_arima taking 94 s (demand) and 46 s (price), not 96 min.
     The 96 min figure appears only as an "up to" note in config.py and IMPLEMENTATION_PLAN.md.
     CHECK: the ARIMA daily update is not timed separately (the logger records 0.0 s for it), so it is
     bounded by the per day total. -->

Table [9] reports the wall clock cost of every pipeline component, logged as described in Section 3.5, which completes the O4 evidence and grounds the refit schedule's justification in measured numbers.

| Component | Frequency | Demand | Price |
|---|---|---|---|
| auto_arima order selection | once, first window | 94 s | 46 s |
| ARIMA weekly refit | weekly, 26 refits | 3.8 s mean (1.2 to 5.6 s) | 5.3 s mean (3.6 to 8.3 s) |
| ARIMA daily update | daily | not logged separately | not logged separately |
| LSTM warm start refit | every 30 days, 6 refits | 6.1 s mean (4.4 to 7.1 s) | 7.5 s mean (5.6 to 10.2 s) |
| LSTM cold start fit (reference) | once | 8.5 s | 8.5 s |
| Daily forecast issue, backtest | per day without a refit | 1.6 s median | 0.8 s median |
| Full backtest, 181 days | once | 468 s | 387 s |
| Live job end to end | per evening, both targets | 3 min 53 s | |

The full refit schedule keeps the daily cost of a backtest issue under 3 seconds on days without a refit and the whole 181 day backtest under 8 minutes per target, and the evening live job, which adds the data download, both targets, and the dashboard, completed in under 4 minutes, all on an AMD Ryzen 7 7800X3D with 32 GB of memory. The live figure is from the single evening run logged so far, on 12 October.

One caveat applies across this whole section. The backtest window and the live track cover different parts of the year, with the backtest spanning January to June 2026, which is summer, autumn, and the first month of winter, while the live track runs in October, which is spring, so the two sets of results are drawn from different demand and price regimes, and differences between backtest and live accuracy partly reflect season rather than model behaviour. Within track comparisons are unaffected, because every model in a track faces the same days, but cross track comparisons and the monthly results in Tables [5] and [7] should be read with this in mind.

# Model card

Purpose: next-day system-wide rental forecasting from information available after the prior day closes. Intended use: historical methodology review. Not intended for live staffing, station replenishment or pricing.

Model: linear ridge regression with an unpenalized intercept and training-only feature scaling. Predictors are four count lags, two shifted rolling averages, cyclical calendar terms, holiday flag and time trend. Predictions are clipped at zero. Penalties are 1, 10 and 100; select once using validation MAE. Baselines are yesterday, last week and the mean of four earlier same-weekday observations.

Training is repeated daily using actual observations up to the forecast cutoff. No held-out-day actual enters its own forecast. Earlier test outcomes legitimately become history for later one-day forecasts. The fixed uncertainty band is descriptive; it assumes residual behaviour transfers and its measured coverage shows that assumption is imperfect.

All errors are reported in rental units or weighted absolute percentage error. WAPE is absolute error sum divided by actual count sum, not mean daily percentage error. Positive bias means overforecasting. Deployment would require current data, observed data-availability latency, station-level constraints, weather forecasts available at the actual cutoff and a cost-based capacity decision.

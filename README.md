# Filtered (Residual) Volatility Forecasting with Machine Learning and Trading

## Overview

This project implements a **hybrid volatility forecasting framework**  (day-ahead) in Germany and France power markets. Please see mainfile_plus_demo.ipynb for functions and full demonstration with graphics and diagnostics output.

The approach combines:

1. **Rolling  HAR-RV** to model systematic volatility dynamics  
2. **XGBoost in rolling or expanding windows** to forecast residual volatility  
3. **Hidden Markov Model** -based regime detection
4. **Rank-based primary metrics, validation and trading implementation**

The objective is **relative ranking of volatility (cross-sectional signal extraction)** , not precise level prediction.

> Research-grade project, work in progress! Right now all the functions and methods and analysis are in one notebook, which will be changed soon. Results are non-optimal at this stage either, and active investigation is ongoing.

## WHAT'S NEW

- **03.03.2026** 
- Tested **rolling-window** vs expanding window CV, now a new function incorporating both functionalities is in place. Motivation -- potentially better performance of rolling windows on multi-regime data, where regimes are persistent. Markov regime detection currently does not work as intended, therefore analyzing without it for now. **Result**: improved IC by 0.3 points (pooled and per-country) compared to exp. window.
- Optuna optimizer for XGBoost hyperparameters, simple version trained/tested on post-2022 regime, not on full data and not in the main CV. Per-country optimization. Result: minor/no improvement in metrics, but substantial improvement of overfitting gap (FR, DE 0.16, 0.11 vs 0.21, 0.16). However, optimization ideally needs to run in the main CV function.

**01.03.2026** -- This a **major update** of the previous work. Most significant changes:
- Proper **historical data** (previously -- pre-aggregated small data sample from a ML competition), spanning from **01.2015 to 02.2026**.  
- GARCH(1,1) was previously used as a baseline model, however, it did not yield satisfactory results on the new dataset. Replaced with **HAR-RV** for further use; previous results kept for demostration and comparison.
- All the modeling execution is now parallelized.
---

## Data

- **ENTSO-e**: 15-min-frequency generation and load data
- **Copernicus**: daily  weather data (rain, wind, temperature)
- **EMBER**: hourly electricity price data (found it easier to retrieve than directly from ENTSO-e)

All data is then re-aggregated into daily. Realized volatility and log prices are buit from price data. Physical units are kept for the variables. For more details and instructions for data retrieval and processing please check data_processing.ipynb.


---

## Objective & Evaluation

### Primary Metric
- **Spearman rank correlation (Information Coefficient)**
- The model is evaluated on its ability to produce correct cross-sectional rank ordering. 
- RMSE and MAE are also implemented, but are currently **not used as an absolute metric**, rather as a relative metric for tracking model improvements. They might also be used directly in the future project updates.

### Validation Layers
- Statistical validation (rank IC, residual ACF, stability checks)
- Economic validation (market-neutral trading strategy with transaction costs)

---

## Model Architecture

### 1. Baseline

For each country:

- Rolling **HAR-RV** model
- Different estimation windows for DE and FR to better capture their specifics
- One-step-ahead volatility forecast

Purpose:
- Capture autoregressive volatility structure
- Remove common dynamics before ML step

---

### 2. ML on Residual Volatility

Machine learning is applied to **residual volatility** (after baseline subtraction) 

- Model: **XGBoost**
- **Currently used**: rolling window of 1200 days
- Can be also used in expanding window mode, exp. window usually 252-280 days  
- Prediction horizon: 21 days  

Why XGBoost:
- Captures nonlinear regime effects
- Handles threshold behavior typical in power markets
- Flexible in terms of regularization
---

### 3. Feature Engineering

Features are designed to reflect structural system regimes. Currently very minimalist, but can expand the feature pool later.

**Volatility persistence**
- Lagged log prices
- Rolling volatility statistics

**Cross-border features**
- Load imbalance 
- Generation imbalance

**Rolling load and weather**
- Seasonality
- Load history

All features are constructed to avoid forward-looking bias.

---

### Regime detection

- **Hidden Markov Model (HMM)**-based regime classifier
- Unsupervised n-state (n=2) Gaussian HMM
- Probabilistic regime assignment --> detecting high/low vol states

## Results (Baseline + ML in expanding window walk-forward CV)

| Metric | Value |
|--------|-------|
| Pooled IC (ML) | 0.39 |
| DE IC (ML) | 0.41 |
| FR IC (ML) | 0.34 |
| DE IC (baseline) | 0.60 |
| FR IC (baseline) | 0.56 |

- Baseline captures the dynamics
- Persistent predictive power
- Results are economically exploitable

---

## Trading Framework

- Cross-country market-neutral framework
- OOS backtest
- Rolling z-scores, no lookahead
- Adaptive thresholding (static/quantile/vol-based)
- Daily cross-sectional neutrality enforced
- Options: EWMA signal smoothing, rebalance thresholds, vol targeting, regime-based exposure shrinkage
- PnL transaction-cost-adjusted

---

### Trading Results (OOS, incl. costs)

| Metric               | Value |
|----------------------|-------|
| Sharpe Ratio         | 1.69  |
| Max Drawdown         |-5.91 |
| Avg Daily Turnover   | 0.84 |
| Win Rate | 45.85% |
| Cost coverage ratio | 7.85 |
---

# Current issues and next steps:

- Mostly raw and a few basic engineered features are used for XGBoost right now -- add and test more engineered features
-  Regime detection inside the CV function currently mixes up regimes -- need more robust implementation
- Trading strategy: still sensitive to extremes, more realistic execution modeling (slippage, realistic fees) can be done
- Improve diagnostic plots to make them more informative, esp. noisy per-fold ones
---


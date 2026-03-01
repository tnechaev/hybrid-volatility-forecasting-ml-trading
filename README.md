# Filtered (Residual) Volatility Forecasting with Machine Learning and Trading

## Overview

This project implements a **hybrid volatility forecasting framework**  (day-ahead) in Germany and France power markets. Please see mainfile_plus_demo.ipynb for functions and full demonstration with graphics and diagnostics output.

The approach combines:

1. **Rolling  HAR-RV** to model systematic volatility dynamics  
2. **XGBoost** to forecast residual volatility  
3. **Hidden Markov Model** -based regime detection
4. **Rank-based primary metrics, validation and trading implementation**

The objective is **relative ranking of volatility (cross-sectional signal extraction)** , not precise level prediction.

> Research-grade project, work in progress! Right now all the functions and methods and analysis are in one notebook, which will be changed soon. Results are non-optimal at this stage either, and active investigation is ongoing.

## IMPORTANT UPDATE!

**01.03.2026** -- This a **major update** of the previous work. Most significant changes:
- Proper **historical data** (previously -- pre-aggregated small data sample from a ML competition), spanning currently from **01.2017 to 12.2023**. Soon the extended period from 01.2015 till now will be included.  
- GARCH(1,1) was previously used as a baseline model, however, it did not yield satisfactory results on the new dataset. Replaced with **HAR-RV** for further use; previous results kept for demostration and comparison.
- All the modeling execution is now parallelized.
---

## Data

- **ENTSO-e**: hourly generation and load data
- **Copernicus**: hourly  weather data (rain, wind, temperature)
- **EMBER**: electricity hourly price data (found it easier to retrieve than directly from ENTSO-e)

All data is then re-aggregated into daily. Realized volatility and log prices are buit from price data. Physical units are kept for the variables.

For more details and instructions for data retrieval and processing please check data_processing.ipynb.
---

## Objective & Evaluation

### Primary Metric
- **Spearman rank correlation (Information Coefficient)**
-The model is evaluated on its ability to produce correct cross-sectional rank ordering. 
- RMSE and MAE are also implemented, but are currently not useful per-se. Staying there for the future upgrades and as a relative metric (track model improvements).

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
- Expanding training window: 280 days  
- Prediction horizon: 21 days  

Why XGBoost:
- Captures nonlinear regime effects
- Handles threshold behavior typical in power markets
- Flexible in terms of regularization
---

### 3. Feature Engineering

Features are designed to reflect structural system regimes. Currently very minimalist, but can expand the otpions later.

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

## Results (Rolling Backtest)

| Metric | Value |
|--------|-------|
| Pooled IC (ML) | 0.37 |
| DE IC (ML) | 0.38 |
| FR IC (ML) | 0.32 |
| DE IC (baseline) | 0.53 |
| FR IC (baseline) | 0.44 |

- Baseline captures the dynamics
- Moderate but persistent predictive power
- Results are economically exploitable

---

## Trading Framework

- Cross-country market-neutral framework

- Rolling z-scores, no lookahead
- Adaptive thresholding (static/quantile/vol-based)
- Daily cross-sectional neutrality enforced
- Options: EWMA signal smoothing, rebalance thresholds, vol targeting, regime-based exposure shrinkage
- PnL transaction-cost-adjusted

---

### Trading Results (OOS, incl. costs)

| Metric               | Value |
|----------------------|-------|
| Sharpe Ratio         | 0.94  |
| Max Drawdown         |-7.31  |
| Avg Daily Turnover   | 0.78  |
---

# Current issues and next steps:

- Baseline currently not regime-aware, could still be improved overall
- Substantial overfitting (train/val gap of ~0.2) --> improve feature engineering and hyperparameter optimizaton (can be e.g. Bayesian)
-  Regime detection may be utilized sub-optimally, separate per-regime model implementation can be considered
- Trading strategy: still sensitive to extremes, more realistic execution modeling can be done. Cost estimate is very optimistic, need more research of what real values are
---


# Filtered (Residual) Volatility Forecasting with Machine Learning and Trading

- [Overview](#Overview)
- [WHAT'S NEW](#WHAT'S NEW)
- [Data](#Data)
- [Model Architecture](#Model Architecture)
- [Results (Baseline + ML in rolling window walk-forward CV)](#Results (Baseline + ML in rolling window walk-forward CV))
- [Trading Framework](#Trading Framework)
- [Trading Results (OOS, incl. costs)](#Trading Results (OOS, incl. costs))
- [Current issues and next steps](#Current issues and next steps)

## Overview

This project implements a **hybrid volatility forecasting framework**  (day-ahead) in Germany and France power markets. Please see **demo.ipynb** for the full demonstration with graphics and diagnostics output.

The approach combines:

1. **Rolling  HAR-RV** to model systematic volatility dynamics  
2. **XGBoost in rolling or expanding windows** to forecast residual volatility  
3. **Hidden Markov Model**- or vol z-score-based regime detection
4. **Rank-based primary metrics, validation and trading implementation**

The objective is **relative ranking of volatility (cross-sectional signal extraction)** , not precise level prediction.

### Research-grade project, work in progress! 

## WHAT'S NEW

**19.03.2026** 
- Optuna hyperparam optimization runs in folds in main CV function. Further improvement in overfitting gap, which is now 0.11-0.12 for DE-FR (70-78% retention rate).

**15.03.2026** 
-  Added more engineered features, reduced redundancy in final selection
- Mostly fixed weekly autocorrelation error pattern

 **03.03.2026** 
-  **rolling-window** XGBoost (expanding window functionality kept). Motivation: when regimes change persistently, better not to overuse old data. **Result**: improved IC by 0.3 points (pooled and per-country) compared to exp. window.

**01.03.2026** -- This a **major update** of the previous work. Most significant changes:
- **Historical data** from **01.2015 to 02.2026** (previously -- pre-aggregated small data sample from a ML competition).  
- GARCH(1,1), previously used as a baseline, did not perform on the new dataset --> replaced with **HAR-RV**.
- Most of the code is parallelized.

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

### Validation Layers
- Statistical validation (rank IC, residual ACF, overfit and stability checks)
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

Features are designed to reflect structural system regimes. Currently very minimalist, to be expanded.

**Volatility persistence**
- Lagged log prices
- Rolling volatility statistics

**Cross-border features**
- Load imbalance 
- Generation imbalance

**Rolling load and weather**
- Seasonality
- Load history

**Residual load** = Load - Wind Generation - Solar Generation

All features are constructed to avoid forward-looking bias.

---

### Regime detection

- **Hidden Markov Model (HMM)**-based regime classifier
- Unsupervised n-state (n=2) Gaussian HMM
- Probabilistic regime assignment --> detecting high/low vol states
- Or alternatively **vol z-score**-based regime classification

## Results (Baseline + ML in rolling window walk-forward CV)

| Metric | Value |
|--------|-------|
| Pooled IC (ML) | 0.34 |
| DE IC (ML) | 0.39 |
| FR IC (ML) | 0.29 |
| DE IC (baseline) | 0.67 |
| FR IC (baseline) | 0.67 |

- Baseline captures the dynamics
- Persistent predictive power
- Results are economically exploitable

---

## Trading Framework

- Cross-country 'market-neutral'-style framework
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
| Sharpe Ratio         | 2.4  |
| Max Drawdown         |-3.8 |
| Avg Daily Turnover   | 0.9 |
| Win Rate | 44.8% |
| Cost coverage ratio | 9.3 |
---

# Current issues and next steps

- Add generation **forecasts**/outages / errors as features -- makes it more realistic
- More realistic execution modeling (slippage, realistic fees) for trading strategy

---


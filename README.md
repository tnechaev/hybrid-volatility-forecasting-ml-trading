# Regime-Conditional Volatility Forecasting with ML; Trading framework

- [Overview](#overview)
- [What's New](#whats-new)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Trading Framework](#trading-framework)
- [Current Issues and Next Steps](#current-issues-and-next-steps)


## Overview

This project implements a **hybrid volatility forecasting framework** for day-ahead power markets in Germany and France. Please see **demo.ipynb** for the full demonstration with graphics and diagnostics output.

The approach combines:

1. **Rolling HAR-RV** to model systematic volatility dynamics
2. **XGBoost in rolling windows** for main forecasting, with **regime-conditional targets** (HAR-RV residual in high-vol, log(RV) level in calm regimes)
3. **Regime detection** via rolling vol z-score or Gaussian-mix HMM
4. **Rank-based primary metrics, validation, and trading implementation**
5. **Market-neutral two-regime trading strategy**: regime-conditional, rolling Sharpe drawdown control, no-lookahead position sizing

The objective is **relative ranking of volatility (cross-sectional signal extraction)**, not precise level prediction.

### Research-grade project, work in progress! 

---

## WHAT'S NEW

**04.04.2026**
- Regime-conditional CV targets: high-vol regime -- predict HAR-RV residual; calm -- predict log(RV) level directly. Materially improved IC and subsequently trading results in calm regime.
- EUR-denominated PnL with notional conversion implemented. 
- Rolling Sharpe drawdown control added to trading strategy, improved Calmar ratio.
- Rolling IC-based calm position sizing: calm regime positions sized using Kelly-proportional scaling derived from past fold IC ratios. 

**19.03.2026** 
- Optuna hyperparameter optimization runs in folds in main CV function. Further improvement in overfitting gap, which is now 0.11-0.12 for DE-FR (70-78% retention rate).

**15.03.2026** 
-  Added more engineered features, reduced redundancy in final selection
- Mostly fixed weekly autocorrelation error pattern

 **03.03.2026** 
-  **rolling-window** XGBoost (expanding window functionality kept). Motivation: when regimes change persistently, better not to overuse old data. **Result**: improved IC by 0.3 points (pooled and per-country) compared to exp. window.

**01.03.2026: major update**
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
**Spearman rank correlation (Information Coefficient, IC)**: the model is evaluated on its ability to produce correct cross-sectional rank ordering of volatility across DE and FR.

### Validation Layers
- Statistical: rank IC, IC stability across folds and regimes, residual ACF, overfitting gap, IC retention rate
- Economic: market-neutral trading strategy with transaction costs estimate, Sharpe, Sortino, Calmar, cost coverage


---

## Model Architecture

### 1. Baseline: Rolling HAR-RV

For each country:
- Rolling OLS estimation of HAR-RV(1,5,22) on log(RV)
- Estimation windows: DE=500 days, FR=200 days (calibrated to regime sensitivity)
- One-step-ahead volatility forecast in log space, exponentiated back to levels

Purpose:
- Capture autoregressive volatility structure (daily, weekly, monthly components)
- Remove common predictable dynamics before ML step

---

### 2. ML (XGBoost) as main forecaster

Why XGBoost:
- Captures nonlinear regime effects
- Handles threshold behavior typical in power markets
- Flexible in terms of regularization

XGBoost is used for forecasting **regime-conditional** targets:
- High-vol regime: `log(RV_t+1) - log(HAR_t+1)` -- HAR-RV errors can be large and are also predictable from features
- Calm regime: log(RV) directly (NO HAR-RV subtraction) -- residuals close to zero, thus vol level is more predictable than residual

- Rolling window: 1200 days (expanding window available)
- Prediction horizon: 21-day test windows in walk-forward CV
- 178 folds total (107 pre-2022, 71 post-2022)
- Per-fold Optuna hyperparameter search on inner val split

### 3. Feature Engineering

**Volatility persistence**
- Lagged log returns and realized vol
- Short-to-long vol ratio (regime signal)

**Cross-border features**
- Load imbalance (DE_load - FR_load)
- Wind imbalance
- VOL_SPREAD and VOL_RATIO

**Residual load** per country (load - wind - solar)

**Generation mix**
- Wind onshore/offshore, solar, hydro reservoir, pumped storage
- French nuclear share and deviation from 90-day rolling mean
- Fossil gas and hard coal generation

**Rolling load and weather**
- 7-day and 30-day rolling means and std of load and generation
- Temperature, wind speed, precipitation

All features are constructed to avoid lookahead.

### 4. Regime Detection

**Vol z-score** (currently used, consistent between CV and trading):
```
zscore_t = (log(RV_t) - rolling_mean(252d)) / rolling_std(252d)

```
Computed with `shift(1)`, avoids lookahead.

**HMM** (available, currently not used):
- Gaussian mixture (2 components per state) emission is currently implemented to account for distribution tails
- Filtered (forward-pass only) probabilities implemented to avoid lookahead from smoother

---

## Results

### Baseline (HAR-RV)

| Metric | DE | FR |
|--------|----|----|
| Spearman IC | 0.67 | 0.67 |

### ML (XGBoost) on conditional target (rolling 1200-day window, walk-forward CV)

| Metric | Value |
|--------|-------|
| Pooled IC (OOS) | 0.67 |
| DE IC (OOS) | 0.65 |
| FR IC (OOS) | 0.70 |
| Overfitting gap DE | 0.1 |
| Overfitting gap FR | 0.1 |

---

## Trading Framework

### Architecture

Two-regime market-neutral strategy:
- High-vol: predicted residual converted to z-score used as a signal
- Calm:  predicted log(RV), supporting size reduction by rolling IC ratio, used as a signal

Cross-country daily market neutrality enforced. Positions sum to zero each day.

### Position Sizing

**Calm regime scale**: computed from rolling IC ratio of past 20 folds only:
```
calm_scale_t = clip(calm_IC[t-20..t-1] / hv_IC[t-20..t-1], 0.05, 1.0)
```
Bets proportional to edge. Mean rolling scale: 0.87 (range 0.37–1.00).

**Rolling Sharpe drawdown control**: positions scaled down when 63-day rolling Sharpe falls below threshold:
```
sharpe_scale_t = clip(rolling_sharpe[t-63..t-1] / sharpe_floor, 0, 1)
```

### Cost Model

(Guess)timates for EEX baseload futures (one-way):
- Bid-ask spread: ~15 bps
- Exchange + clearing fees: ~4 bps
- Market impact: ~6 bps (superlinear)
- **Total used in backtest: 50 bps** (larger than estimated, because likely too optimistic)

### Trading Results (OOS, 100 MW notional, 50 bps TC)

| Metric | Value |
|--------|-------|
| Sharpe | 2.86 |
| Sharpe (before TC) | 3.06 |
| Sortino | 4.42 |
| Calmar | 1.19 |
| Win Rate (active days) | 55.96% |
| Avg Daily Turnover | 0.70 |
| Cost Coverage | 15.47x |
| Total PnL (10yr) | 23.5M EUR |
| Annualised PnL | 1.58M EUR/year |
| Max Drawdown EUR | -1.41M EUR |

### Regime Breakdown

| Regime | Sharpe | PnL (EUR) | Days |
|--------|--------|-----------|------|-----------|
| Calm (65%) | 2.40 | 11.7M | 2429 | 
| High-vol (35%) | 3.58 | 11.8M | 1309 | 


### Cost Sensitivity

| Cost (bps) | Sharpe | Coverage |
|------------|--------|----------|
| 5 | 3.04 | 154.7x |
| 15 | 3.00 | 51.6x |
| 25 | 2.96 | 30.9x |
| 50 | 2.86 | 15.5x |
| 100 | 2.66 | 7.7x |

### Notional Sensitivity

| Notional | Total PnL | Ann. PnL | Sharpe | Max DD |
|----------|-----------|----------|--------|--------|
| 10 MW | 2.35M EUR | 158k EUR | 2.90 | -141k EUR |
| 25 MW | 5.88M EUR | 396k EUR | 2.90 | -353k EUR |
| 50 MW | 11.75M EUR | 792k EUR | 2.90 | -706k EUR |
| 100 MW | 23.50M EUR | 1,585k EUR | 2.90 | -1,412k EUR |
| 150 MW | 35.26M EUR | 2,377k EUR | 2.90 | -2,118k EUR |


---

# Current issues and next steps

- Test statistical jump models as regime classifiers
- Add cross-border flow features (interconnector congestion)
- Calm strategy performance structurally limited by low daily-frequency signal in calm periods
- Implement **intraday strategy**: shorter horizon, potential to overcome calm regime limitations; add **forecast errors** as features

---


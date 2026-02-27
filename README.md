# GARCH-Filtered Volatility Forecasting with Machine Learning (XGBoost)

## IMPORTANT UPDATE!

**26.02.2026** -- I found a significant structural issue in my data reconstruction. Not only the dataset is small, but, due to this structural issue, the modeling technique used does not correspond properly to the data. All results here to be taken very cautiously. I am working on proper historical (ENTSO-e, Copernicus, EMBER) data analysis that spans several years. The main model idea (GARCH + XGBoost) will stay the same, with more complexity required for work with long-history data spanning several regimes. The cross-country trading strategy will stay the same, with two separate signal series for FR and DE available for use. This repo will be likely deleted or wiped clean and updated with the new project, with all the functions of data retrieval and extended modeling/validation steps for full reproduction. I plan to post the updated project within 2-4 days since this announcement. 


## Overview

This project implements a **hybrid volatility forecasting framework** for 24-hour electricity futures in the **Germany–France (DE–FR) coupled power market**.

The approach combines:

1. **Rolling GARCH(1,1)** to model systematic volatility dynamics  
2. **XGBoost** to forecast residual volatility  
3. **Rank-based validation and trading implementation**

The objective is **relative ranking of volatility (cross-sectional signal extraction)** rather than precise level prediction.

> This is a research-grade project and work in progress.

---

## Data

- Source: ENS machine learning challenge  
  https://challengedata.ens.fr/participants/challenges/97/
- Only **TRAIN data** is used
- A **rolling forecasting setup** is implemented
- Full realized volatility is available for evaluation
- No challenge TEST data is used
- The original challenge focused on price explanation, not forecasting

Due to potential restrictions, the dataset is not included in the repository.

---

## Objective & Evaluation

### Primary Metric
- **Spearman rank correlation (Information Coefficient)**

The model is evaluated on its ability to correctly rank relative volatility between DE and FR.

### Validation Layers
- Statistical validation (rank IC, stability checks)
- Economic validation (market-neutral trading strategy with transaction costs)

---

## Model Architecture

### 1. GARCH Baseline

For each country:

- Rolling **GARCH(1,1)** model
- 500-day estimation window
- One-step-ahead volatility forecast
- Models estimated independently for DE and FR

Purpose:
- Capture autoregressive volatility structure
- Remove common dynamics before ML step

---

### 2. ML on Residual Volatility

Machine learning is applied to **GARCH residual volatility**.

- Model: **XGBoost**
- Rolling training window: 280 days  
- Prediction horizon: 21 days  
- Empirical hyperparameter selection

Why XGBoost:
- Captures nonlinear regime effects
- Handles threshold behavior typical in power markets
- Robust to small, noisy datasets (with regularization)

---

### 3. Feature Engineering

Features are designed to reflect structural system regimes.

**Volatility persistence**
- Lagged volatility
- Rolling volatility statistics

**System stress**
- Residual load measures
- Stress indicators

**Cross-border congestion**
- Flow imbalance variables

**Fuel & merit order**
- Gas–coal spreads
- Carbon pressure proxies

**Renewables regime**
- Relative renewable penetration

Feature selection to be used for training is now automatic and is not perfect. Might adjust it later.

Key observation:
> Structural regime variables dominate predictive power. Many exogenous effects are absorbed by lagged volatility and stress indicators.

All features are constructed to avoid forward-looking bias.

---

## Results (Rolling Backtest)

| Metric | Value |
|--------|-------|
| Pooled IC (ML residual) | 0.32 |
| DE IC (ML residual) | 0.12 |
| FR IC (ML residual) | 0.18 |
| DE IC (GARCH baseline) | 0.08 |
| FR IC (GARCH baseline) | 0.07 |
| DE Rank autocorr. | 0.78 |
| FR Rank autocorr. | 0.68 |

Interpretation:
- ML improves upon a weak GARCH baseline
- Predictive power is moderate but persistent
- Results are economically exploitable

---

## Trading Framework

Signals are converted into a **relative volatility spread strategy**:

- Cross-sectional normalization
- Rolling z-scores
- Threshold-based entry rules
- Hypothetical transaction costs included
- Market-neutral across DE–FR

The strategy targets **relative volatility mispricing**, not a directional bet.

---

### Trading Results (With Costs)

| Metric               | Value |
|----------------------|-------|
| Sharpe Ratio         | 0.8   |
| Max Drawdown         |-13.8  |
| Avg Daily Turnover   | 0.48  |

---

# Limitations

This project has important constraints:

### 1. Small dataset
- Limited time span
- Low number of independent regimes
- Risk of overfitting despite rolling validation
- Results may not generalize to new structural regimes

### 2. Simple baseline
- GARCH(1,1) is a simple benchmark and has quite a weak signal level here
- Incremental ML improvement should be interpreted cautiously

### 3. Trading assumptions
- Simplified execution model
- No liquidity constraints
- No slippage modeling
- Capacity limits not considered

---

## Next Steps

- Stronger volatility benchmarks (HAR, regime-switching models)
- Bayesian hyperparameter optimization
- External dataset validation
- More realistic execution modeling

---

## Takeaway

This project demonstrates:

- A structured approach to hybrid volatility modeling  
- Rank-based evaluation aligned with trading applications  
- Integration of statistical validation and trading backtesting  

It should be viewed as a **research prototype**, not a production-ready trading system.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")

#--------------------------------
# "Market-neutral"-style cross-country strategy
# ---------------------------------------
def compute_market_neutral_pnl_adaptive(
    bt,
    threshold=0.5,
    cost_bps=0.0005,
    lookback=30,
    # ----- adaptive thresholding -----
    adaptive_threshold: str = "static",   # {"static","quantile","vol"}
    quantile: float = 0.85,               # used when adaptive_threshold=="quantile"
    vol_k: float = 1.0,                   # multiplier when adaptive_threshold=="vol"
    min_threshold: float = 0.1,           # floor for adaptive thresholds
    # ----- smoothing & rebalancing -----
    ewma_alpha: float = None,             # if set, apply EWMA smoothing to raw signals (0 < alpha <=1)
    rebalance_thr: float = None,          # absolute position change threshold (fraction of exposure) to actually trade; if None no threshold
    # ----- vol targeting -----
    enable_vol_target: bool = False,
    target_ann_vol: float = 0.10,         # target annual vol (e.g. 0.10 = 10%)
    vol_window: int = 126,                # realized vol calculation window (days)
    vol_cap: float = 2.0,                 # cap on vol scaling factor
    vol_floor: float = 0.1,               # floor on vol scaling factor
    # ----- regime shrinkage / hedging -----
    use_regime_prob: dict = None,
        # e.g. {'DE': 'DE_regime_prob_1', 'FR': 'FR_regime_prob_1'}
        # expects column in bt (per row) with probability of "bad" regime (0..1)
    regime_shrink_to: float = 0.3,        # shrink exposure to this fraction when prob=1
    # ----- other -----
    clip_signal: float = 2.0,             # clip raw signal magnitude as before
    z_fillna: float = 0.0,                # value to use for z when missing
    verbose: bool = False
):
    """
    Adaptive market-neutral pnl calculator.
      - z-scores per country (rolling lookback, past-only shift(1))
      - thresholding to form signals
      - cross-sectional demeaning and normalization so daily abs exposure = 1
      - pnl computed on `true` residuals; transaction costs applied using turnover

    Extra features (optional):
      - `adaptive_threshold`: compute threshold from historical |z| quantiles or short-run volatility
      - `ewma_alpha`: smooth signals via EWMA to reduce turnover
      - `rebalance_thr`: only execute changes larger than this threshold (reduces turnover)
      - `enable_vol_target`: scale exposures to target realized volatility
      - `use_regime_prob`: optional map of country -> column name (0..1) to shrink exposure when regime risk high

    Output columns: `position`, `executed_position`, `pnl_before_cost`, `turnover`, `tcost`, `pnl`
    """
    df = bt.copy()
    # Normalize column expectations
    df = df.sort_values(['DATE', 'COUNTRY']).reset_index(drop=True)

    # --- 1) rolling mean/std of preds (past-only) and z ---
    # rolling mean/std per country, then shift(1) to avoid lookahead
    grp = df.groupby('COUNTRY')['pred']
    df['pred_mean_past'] = grp.transform(lambda x: x.rolling(lookback, min_periods=5).mean().shift(1))
    df['pred_std_past']  = grp.transform(lambda x: x.rolling(lookback, min_periods=5).std().shift(1))

    # fallback medians for small-sample stds (same approach as you had)
    df['pred_std_past'] = df['pred_std_past'].fillna(df.groupby('COUNTRY')['pred_std_past'].transform('median'))
    df['pred_std_past'] = df['pred_std_past'].replace(0, 1e-6)

    # z
    df['z'] = (df['pred'] - df['pred_mean_past']) / df['pred_std_past']
    df['z'] = df['z'].fillna(z_fillna)

    # --- 2) adaptive thresholding (per country, past-only) ---
    # compute a threshold series per row and store threshold_used
    df['threshold_used'] = np.nan

    if adaptive_threshold == "static":
        df['threshold_used'] = threshold
    else:
        # compute rolling reference of |z| on training history (past-only)
        absz = df.groupby('COUNTRY')['z'].transform(lambda x: x.abs().rolling(lookback, min_periods=5).quantile(quantile).shift(1) if adaptive_threshold=='quantile' else x.abs().rolling(lookback, min_periods=5).std().shift(1))
        if adaptive_threshold == 'quantile':
            # quantile already computed above
            df['threshold_used'] = absz.fillna(threshold)
        elif adaptive_threshold == 'vol':
            # vol_k * rolling std of z
            df['threshold_used'] = (vol_k * absz).fillna(threshold)
        else:
            df['threshold_used'] = threshold
    # enforce min threshold
    df['threshold_used'] = df['threshold_used'].fillna(threshold).clip(lower=min_threshold)

    # --- 3) form raw signals (only signal beyond threshold) ---
    df['signal_raw'] = np.where(df['z'].abs() > df['threshold_used'], df['z'], 0.0)

    # clipping
    df['signal'] = df['signal_raw'].clip(-clip_signal, clip_signal)

    # optional EWMA smoothing (per-country) to reduce noise/turnover
    if ewma_alpha is not None and 0.0 < ewma_alpha <= 1.0:
        df['signal'] = df.groupby('COUNTRY')['signal'].transform(lambda s: s.fillna(0).ewm(alpha=ewma_alpha, adjust=False).mean())

    # --- 4) initial cross-sectional market-neutralization (make signals sum to 0 each day) ---
    df['signal_cs'] = df['signal'] - df.groupby('DATE')['signal'].transform('mean')

    # --- 5) apply per-row scale factors: regime shrink + vol target (both optional) ---
    # regime shrink: if use_regime_prob provided and column exists, map to value; else 0
    df['regime_prob_used'] = 0.0
    if use_regime_prob is not None:
        # allow dict or mapping; for each country get column name
        for c in df['COUNTRY'].unique():
            colname = use_regime_prob.get(c) if isinstance(use_regime_prob, dict) else None
            if colname and colname in df.columns:
                mask = df['COUNTRY'] == c
                df.loc[mask, 'regime_prob_used'] = df.loc[mask, colname].fillna(0.0).clip(0.0, 1.0)
            else:
                # keep default 0.0 for that country
                pass

    # compute scale_regime = 1 - prob * (1 - regime_shrink_to)
    df['scale_regime'] = 1.0 - df['regime_prob_used'] * (1.0 - regime_shrink_to)
    df['scale_regime'] = df['scale_regime'].clip(lower=regime_shrink_to, upper=1.0)

    # vol targeting: compute realized vol of actual residuals ('true') per country, past-only
    df['scale_vol_target'] = 1.0
    if enable_vol_target:
        # compute realized daily std of `true` per country, using trailing vol_window, shift(1)
        df['realized_vol'] = df.groupby('COUNTRY')['true'].transform(
            lambda x: x.rolling(vol_window, min_periods=10).std().shift(1)
        ).replace(0, np.nan)
        # convert to annualized
        df['realized_ann_vol'] = df['realized_vol'] * np.sqrt(252)
        # scaling factor
        df['scale_vol_target'] = (target_ann_vol / (df['realized_ann_vol'].replace(0, np.nan))).fillna(1.0)
        df['scale_vol_target'] = df['scale_vol_target'].clip(lower=vol_floor, upper=vol_cap)

    # final scale applied to signal_cs before normalization
    df['signal_scaled'] = df['signal_cs'] * df['scale_regime'] * df['scale_vol_target']

    # --- 6) normalize scaled signals so total absolute exposure = 1 per day (market neutral) ---
    # compute sum abs per day
    df['daily_abs_sum'] = df.groupby('DATE')['signal_scaled'].transform(lambda x: x.abs().sum())
    df['daily_abs_sum'] = df['daily_abs_sum'].replace(0, np.nan)  # avoid div0
    df['position_desired'] = df['signal_scaled'] / df['daily_abs_sum']
    df['position_desired'] = df['position_desired'].fillna(0.0)

    # At this point positions are market-neutral by construction (sum ~0) and abs-sum = 1 per day.
    # We still may want to apply a rebalance threshold to reduce small trades (turnover).
    # Compute the exec. position by comparing to previous exec.position and only change when delta > rebalance_thr.
    df['prev_executed_position'] = df.groupby('COUNTRY')['position_desired'].shift(1).fillna(0.0)

    # executed_position initialization = desired, "freeze" small changes
    if rebalance_thr is None:
        df['executed_position'] = df['position_desired']
    else:
        # vectorized application by DATE: preserve cross-sectional neutrality after thresholding
        # Approach: for each DATE, start from prev_executed (previous day) and apply changes >= thr; then renormalize.
        executed_list = []
        dates = pd.to_datetime(df['DATE']).unique()
        # build a helper mapping date -> slice
        df_idx = df.reset_index()
        for dt in dates:
            mask_dt = df['DATE'] == dt
            rows = df.loc[mask_dt].copy()
            # previous executed positions (per row) are prev_executed_position
            prev = rows['prev_executed_position'].values.astype(float)
            desired = rows['position_desired'].values.astype(float)
            delta = desired - prev
            # keep changes only where abs(delta) >= rebalance_thr
            keep_mask = np.abs(delta) >= rebalance_thr
            executed = prev.copy()
            executed[keep_mask] = desired[keep_mask]
            # renormalize executed so abs sum = 1 and mean = 0 (only if not all zero)
            abs_sum = np.abs(executed).sum()
            if abs_sum > 0:
                executed = executed / abs_sum
                # ensure sum zero; due to rounding, subtract mean approx
                executed = executed - np.mean(executed)
            else:
                executed = executed  # keep zeros
            executed_list.append(pd.Series(executed, index=rows.index))
        executed_ser = pd.concat(executed_list).sort_index()
        df['executed_position'] = executed_ser.reindex(df.index).fillna(0.0)

    # compute turnover and costs relative to prev_executed_position grouped by COUNTRY
    df['prev_position_used'] = df.groupby('COUNTRY')['executed_position'].shift(1).fillna(0.0)
    df['turnover'] = (df['executed_position'] - df['prev_position_used']).abs()
    df['tcost'] = df['turnover'] * cost_bps

    # pnl before cost on raw 'true' residuals
    df['pnl_before_cost'] = df['executed_position'] * df['true']
    df['pnl'] = df['pnl_before_cost'] - df['tcost']

    # copy back into column names 
    df['position'] = df['executed_position']

    # basic check: positions sum to ~0 each day
    daily_pos_sum = df.groupby('DATE')['position'].sum()
    max_dev = daily_pos_sum.abs().max()
    if verbose and not np.isfinite(max_dev):
        print("Warning: daily position sums contain non-finite values.")
    if verbose and max_dev > 1e-6:
        print(f"Warning: positions not perfectly market neutral after processing. Max daily sum deviation = {max_dev:.3e}")

    return df


#----------------------------
# Performance evaluation
#----------------------------

def evaluate_trading_performance(bt):
    """
    Evaluate trading strategy performance
    
    Metrics:
    - Sharpe ratio
    - Max drawdown
    - Turnover
    - Win rate
    """
    # Daily PnL
    daily_pnl = bt.groupby('DATE')['pnl'].sum()
    
    # Sharpe (annualized, assuming 252 trading days)
    sharpe = (daily_pnl.mean() / (daily_pnl.std() + 1e-9)) * np.sqrt(252)
    
    # Max drawdown
    cum_pnl = daily_pnl.cumsum()
    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl - roll_max
    max_dd = drawdown.min()
    
    # Turnover
    avg_daily_turnover = bt.groupby('DATE')['turnover'].sum().mean()
    
    # Win rate
    win_rate = (daily_pnl > 0).mean()
    
    # PnL statistics
    total_pnl = daily_pnl.sum()
    total_cost = bt['tcost'].sum()
    
    metrics = {
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_daily_turnover': avg_daily_turnover,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_cost': total_cost,
        'pnl_before_cost': bt['pnl_before_cost'].sum(),
        'n_days': len(daily_pnl)
    }
    
    print("=== Trading Performance ===")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Daily Turnover: {avg_daily_turnover:.4f}")
    print(f"Total PnL: {total_pnl:.6f}")
    print(f"Total Cost: {total_cost:.6f}")
    print(f"PnL Before Cost: {metrics['pnl_before_cost']:.6f}")
    
    return metrics, daily_pnl
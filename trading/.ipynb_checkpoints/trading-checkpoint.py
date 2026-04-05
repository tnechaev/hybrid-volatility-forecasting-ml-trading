import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")


def build_price_panel(
    price_file_de: str,
    price_file_fr: str,
    datetime_col: str = 'Datetime (Local)',
    price_col: str = 'Price (EUR/MWhe)',
) -> pd.DataFrame:
    """
    Build daily price panel from EMBER hourly price files.
    Returns DataFrame with columns: DATE, DE_price, FR_price
    """
    def _load_daily(path, prefix):
        raw = pd.read_csv(path, parse_dates=[datetime_col])
        raw = raw[[datetime_col, price_col]].copy()
        raw.columns = ['datetime', 'price']
        raw = raw.dropna(subset=['price'])
        raw['price'] = pd.to_numeric(raw['price'], errors='coerce')
        raw = raw[raw['price'] > 0]
        raw = raw.sort_values('datetime')
        raw['date'] = pd.to_datetime(raw['datetime']).dt.normalize()

        daily = raw.groupby('date')['price'].agg(
            mean='mean',
            count='count',
        ).reset_index()
        daily.columns = ['DATE', f'{prefix}_price', f'{prefix}_price_hours']
        daily = daily[daily[f'{prefix}_price_hours'] >= 20]
        daily['DATE'] = pd.to_datetime(daily['DATE'])
        return daily.drop(columns=[f'{prefix}_price_hours'])

    de = _load_daily(price_file_de, 'DE')
    fr = _load_daily(price_file_fr, 'FR')

    combined = de.merge(fr, on='DATE', how='inner')
    combined = combined.sort_values('DATE').reset_index(drop=True)

    print(f"Price panel: {len(combined)} days  "
          f"{combined['DATE'].min().date()} to {combined['DATE'].max().date()}")
    print(f"  DE: mean={combined['DE_price'].mean():.1f}  "
          f"median={combined['DE_price'].median():.1f}  "
          f"max={combined['DE_price'].max():.1f} EUR/MWh")
    print(f"  FR: mean={combined['FR_price'].mean():.1f}  "
          f"median={combined['FR_price'].median():.1f}  "
          f"max={combined['FR_price'].max():.1f} EUR/MWh")
    return combined


"""
Market-neutral vol trading strategy

NO-LOOKAHEAD checks
------------------------
1. Regime detection: rolling z-score with shift(1) — uses only past data
2. Signal z-score: rolling mean/std with shift(1)
3. calm_position_scale: rolling IC ratio from past folds only (no in-sample estimates)
4. Rolling Sharpe drawdown control: uses past 63 days only, shift(1)
5. EUR scaling: fixed median price (computed once from full sample — should be acceptable
   since median price is a constant, not a signal)

NUMERICAL CHOICES 
-----------------------------------
vol_regime_window = 252
    One trading year, balances stability vs adaptivity. Shorter = more reactive but noisier.

vol_regime_threshold = 0.5
    ~69th percentile of normal distribution, flags ~30% of days as high-vol.
    Somewhat matches empirical ~26% high-vol frequency in the DE/FR data.

ic_lookback_folds = 20
    20 folds × 21 days = ~420 days ≈ 1.5 years of IC history.
    Should be sufficient to estimate a stable IC ratio and to adapt to regime shifts.

scale_floor = 0.05
    Never go below 5% of full size even if calm IC estimate is near zero.
    Prevents completely shutting down based on a noisy short-history estimate.

rolling_sharpe_window = 63
    ~3 months of daily returns. Short enough to detect deteriorating conditions
    quickly; long enough to avoid overreacting to daily noise.
    63 days gives roughly 15 independent 21-day periods for estimation.

sharpe_floor = 0.3
    Minimum rolling Sharpe to maintain full exposure.
    Below this, positions scale down linearly toward zero.
    0.3 of the long-run average, picked as an eyeballed 
    "strategy is not working" signal. To be adjusted based on risk tolerances/needs.

sharpe_scale_cap = 1.0
    Maximum position scale from Sharpe control = 1.0.
    The Sharpe control only reduces exposure, not amplifies it, so
    no leverage here.

cost_bps = 50 bps
    Eyeball EEX power futures estimate:
    spread 15-20 bps + fees 4-5 bps + impact 6-10 bps = ~25-35 bps.
    50 bps to be up from potentially very optimistic estimate.
"""

# =============================================================================
# Rolling calm scale from past fold ICs
# =============================================================================

def compute_rolling_calm_scale(
    fold_stats: pd.DataFrame,
    ic_lookback_folds: int = 20,
    min_folds: int = 5,
    scale_floor: float = 0.05,
    scale_cap: float = 1.0,
    verbose: bool = True,
) -> pd.Series:
    """
    For each fold t, uses IC estimates from folds max(0, t-ic_lookback_folds)..t-1;
    No current or future fold information is used.

    Returns Series indexed by test_from date. Merged into bt by DATE
    before calling compute_market_neutral_pnl_adaptive.

    ic_lookback_folds=20: ~1.5 years of IC history.
    scale_floor=0.05: always keep at least 5% of full size.
    """
    fs = fold_stats.copy().sort_values('test_from').reset_index(drop=True)
    fs['test_from'] = pd.to_datetime(fs['test_from'])

    if 'fold_regime_de' not in fs.columns or 'fold_regime_fr' not in fs.columns:
        if verbose:
            print("WARNING: fold_regime_de/fr not in fold_stats. "
                  "Using constant scale=1.0. Rerun CV with full_pipeline_v2.py.")
        return pd.Series(1.0, index=fs['test_from'], name='calm_scale_rolling')

    # A fold is high-vol if at least one country is in high-vol regime
    fs['fold_is_hv'] = (
        (fs['fold_regime_de'].fillna(0) + fs['fold_regime_fr'].fillna(0)) >= 1
    ).astype(int)

    # Use pooled_ic as the IC estimate per fold
    ic_col = 'pooled_ic' if 'pooled_ic' in fs.columns else 'spearman_ic_de'

    scales = []
    for t in range(len(fs)):
        if t < min_folds:
            # Not enough history yet — use full size (default)
            scales.append(1.0)
            continue

        start_idx = max(0, t - ic_lookback_folds)
        history   = fs.iloc[start_idx:t]   # strictly past folds only

        hv_folds   = history[history['fold_is_hv'] == 1][ic_col].dropna()
        calm_folds = history[history['fold_is_hv'] == 0][ic_col].dropna()

        if len(hv_folds) < 3 or len(calm_folds) < 3:
            scales.append(1.0)
            continue

        hv_ic   = float(max(hv_folds.mean(),   0.0))
        calm_ic = float(max(calm_folds.mean(), 0.0))

        if hv_ic < 1e-4:
            scales.append(1.0)
            continue

        # Kelly-proportional: bet proportional to edge relative to high-vol edge
        scale = float(np.clip(calm_ic / hv_ic, scale_floor, scale_cap))
        scales.append(scale)

    result = pd.Series(scales, index=fs['test_from'], name='calm_scale_rolling')

    if verbose:
        print(f"\nRolling calm_position_scale (from past {ic_lookback_folds} folds):")
        print(f"  Mean={np.mean(scales):.3f}  Std={np.std(scales):.3f}  "
              f"Min={np.min(scales):.3f}  Max={np.max(scales):.3f}")
        print(f"  First {min_folds} folds use default scale=1.0 (insufficient history)")

    return result


def merge_rolling_scale_into_bt(
    bt: pd.DataFrame,
    fold_stats: pd.DataFrame,
    ic_lookback_folds: int = 20,
    min_folds: int = 5,
    scale_floor: float = 0.05,
    scale_cap: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling scale and merge into bt by DATE.
    Each test date gets the scale computed from folds before it.
    Dates within a fold's test period all get the same scale (forward filled).
    """
    rolling_scale = compute_rolling_calm_scale(
        fold_stats, ic_lookback_folds, min_folds, scale_floor, scale_cap, verbose
    )

    # Build date-level mapping with forward fill
    scale_df = rolling_scale.reset_index()
    scale_df.columns = ['DATE', 'calm_scale_rolling']
    scale_df['DATE'] = pd.to_datetime(scale_df['DATE']).dt.normalize()

    bt = bt.copy()
    bt['DATE'] = pd.to_datetime(bt['DATE']).dt.normalize()

    all_dates = pd.DataFrame({'DATE': sorted(bt['DATE'].unique())})
    scale_full = (all_dates
                  .merge(scale_df, on='DATE', how='left')
                  .sort_values('DATE'))
    scale_full['calm_scale_rolling'] = scale_full['calm_scale_rolling'].ffill().bfill()

    bt = bt.merge(scale_full, on='DATE', how='left')
    return bt


# =============================================================================
# Main trading strategy
# =============================================================================

def compute_market_neutral_pnl_adaptive(
    bt: pd.DataFrame,
    # ---- regime detection (must match CV settings exactly) ----
    vol_regime_window: int = 252,
    vol_regime_threshold: float = 0.5,
    # ---- high-vol signal ----
    threshold: float = 0.5,
    lookback: int = 30,
    clip_signal: float = 2.0,
    z_fillna: float = 0.0,
    # ---- position sizing ----
    # calm_position_scale: if None, uses 'calm_scale_rolling' column if present,
    # else falls back to scalar 1.0. Set a float to override completely.
    calm_position_scale: float = None,
    high_vol_position_scale: float = 1.0,
    # ---- rolling Sharpe drawdown control ----
    # Scales ALL positions down when recent rolling Sharpe is weak.
    # Implements: scale = clip(rolling_sharpe / sharpe_floor, 0, 1)
    # so at Sharpe = sharpe_floor -- get 100% of positions;
    # at Sharpe = 0 -- get 0% (flat); never leverages above 1.
    use_sharpe_sizing: bool = True,
    rolling_sharpe_window: int = 63,    # ~3 months — see module docstring
    sharpe_floor: float = 0.5,          # scale to 0 below this — see module docstring
    sharpe_scale_cap: float = 1.0,      # never amplify above 1
    sharpe_warmup_days: int = 63,       # days before Sharpe control activates
    # ---- costs ----
    cost_bps: float = 0.005,           # 50 bps — see module docstring
    # ---- EUR conversion ----
    price_panel: pd.DataFrame = None,
    notional_mw: float = 10.0,
    hours_per_day: int = 24,
    eur_scaling_method: str = 'fixed',  # 'fixed' | 'regime' | 'variable'
    # ---- legacy Markov ----
    use_regime_prob: dict = None,
    regime_shrink_to: float = 0.3,
    # ---- output ----
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Market-neutral vol trading strategy.

    POSITION SIZING HIERARCHY:
    1. Base signal: z-score of XGBoost prediction, thresholded
    2. Regime scale: calm periods sized by rolling IC ratio (past folds only)
    3. Rolling Sharpe control: scale down when recent performance is poor
    4. Normalize to unit abs exposure per day (market neutral by construction)
    5. Costs applied on turnover

    All scaling uses only past information. No in-sample estimates.
    """
    df = bt.copy().sort_values(['DATE', 'COUNTRY']).reset_index(drop=True)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.normalize()

    # ----------------------------------------------------------------
    # Per-country regime flag from log(realized_vol) z-score
    # ----------------------------------------------------------------
    df['vol_regime_flag']   = 0
    df['vol_regime_zscore'] = 0.0

    for pref in ['DE', 'FR']:
        rv_col = f'{pref}_realized_vol'
        if rv_col not in df.columns:
            continue
        mask   = df['COUNTRY'] == pref
        log_rv = np.log(df.loc[mask, rv_col].astype(float).clip(lower=1e-12))
        mu  = log_rv.rolling(vol_regime_window, min_periods=30).mean().shift(1)
        sig = log_rv.rolling(vol_regime_window, min_periods=30).std().shift(1).clip(lower=1e-8)
        zs  = (log_rv - mu) / sig
        df.loc[mask, 'vol_regime_zscore'] = zs.values
        df.loc[mask, 'vol_regime_flag']   = (zs > vol_regime_threshold).astype(int).values

    # ----------------------------------------------------------------
    # Resolve calm_position_scale (no lookahead)
    # ----------------------------------------------------------------
    if calm_position_scale is not None:
        # Explicit scalar override — use as-is
        df['_calm_scale'] = float(calm_position_scale)
        if verbose:
            print(f"calm_position_scale: fixed at {calm_position_scale:.3f} (explicit override)")
    elif 'calm_scale_rolling' in df.columns:
        # Pre-computed rolling scale merged in by merge_rolling_scale_into_bt()
        df['_calm_scale'] = df['calm_scale_rolling'].fillna(1.0)
        if verbose:
            print(f"calm_position_scale: rolling (from past fold ICs)  "
                  f"mean={df['_calm_scale'].mean():.3f}  "
                  f"min={df['_calm_scale'].min():.3f}  "
                  f"max={df['_calm_scale'].max():.3f}")
    else:
        # No scale available — default to 1.0 with warning
        df['_calm_scale'] = 1.0
        if verbose:
            print("WARNING: calm_position_scale not set and 'calm_scale_rolling' "
                  "not in bt. Using 1.0. Call merge_rolling_scale_into_bt() first.")

    # ----------------------------------------------------------------
    # XGBoost prediction z-score signal
    # ----------------------------------------------------------------
    grp = df.groupby('COUNTRY')['pred']
    df['pred_mean_past'] = grp.transform(
        lambda x: x.rolling(lookback, min_periods=5).mean().shift(1)
    )
    df['pred_std_past'] = grp.transform(
        lambda x: x.rolling(lookback, min_periods=5).std().shift(1)
    ).fillna(
        grp.transform(
            lambda x: x.rolling(lookback, min_periods=5).std().shift(1)
        ).groupby(df['COUNTRY']).transform('median')
    ).replace(0, 1e-6)

    df['z_pred'] = (
        (df['pred'] - df['pred_mean_past']) / df['pred_std_past']
    ).fillna(z_fillna)

    df['signal_raw'] = np.where(df['z_pred'].abs() > threshold, df['z_pred'], 0.0)
    df['signal']     = df['signal_raw'].clip(-clip_signal, clip_signal)

    # Cross-sectional demean (market neutral before scaling)
    df['signal_cs'] = df['signal'] - df.groupby('DATE')['signal'].transform('mean')
    abs_cs = df.groupby('DATE')['signal_cs'].transform(
        lambda x: x.abs().sum()
    ).replace(0, np.nan)
    df['signal_norm'] = (df['signal_cs'] / abs_cs).fillna(0.0)

    # ----------------------------------------------------------------
    # Regime-dependent sizing (per country, no lookahead)
    # ----------------------------------------------------------------
    df['position_pre_sharpe'] = np.where(
        df['vol_regime_flag'] == 1,
        df['signal_norm'] * high_vol_position_scale,
        df['signal_norm'] * df['_calm_scale'],
    )

    # Re-demean after scaling (regime flags can differ between DE and FR
    # on the same day, breaking neutrality)
    df['position_pre_sharpe'] = (
        df['position_pre_sharpe']
        - df.groupby('DATE')['position_pre_sharpe'].transform('mean')
    )
    abs_pre = df.groupby('DATE')['position_pre_sharpe'].transform(
        lambda x: x.abs().sum()
    ).replace(0, np.nan)
    df['position_pre_sharpe'] = (df['position_pre_sharpe'] / abs_pre).fillna(0.0)

    # ----------------------------------------------------------------
    # Legacy Markov shrinkage (backward compat, optional)
    # ----------------------------------------------------------------
    df['scale_regime'] = 1.0
    if use_regime_prob is not None:
        df['regime_prob_used'] = 0.0
        for c in df['COUNTRY'].unique():
            colname = use_regime_prob.get(c) if isinstance(use_regime_prob, dict) else None
            if colname and colname in df.columns:
                mask = df['COUNTRY'] == c
                df.loc[mask, 'regime_prob_used'] = (
                    df.loc[mask, colname].fillna(0.0).clip(0.0, 1.0)
                )
        df['scale_regime'] = (
            1.0 - df['regime_prob_used'] * (1.0 - regime_shrink_to)
        ).clip(lower=regime_shrink_to, upper=1.0)
        df['position_pre_sharpe'] = df['position_pre_sharpe'] * df['scale_regime']

    # ----------------------------------------------------------------
    # Rolling Sharpe drawdown control (no lookahead)
    # ----------------------------------------------------------------
    # Compute paper PnL series using position_pre_sharpe × true
    # then compute rolling Sharpe on that series (past only, shift(1))
    # and scale positions down when rolling Sharpe is poor.
    #
    # Implementation:
    #   paper_pnl_t = position_pre_sharpe_t × true_t  (no cost yet)
    #   daily_paper = sum of paper_pnl over countries per date
    #   rolling_sharpe_t = mean(paper_pnl[t-window..t-1]) /
    #                      std(paper_pnl[t-window..t-1]) × sqrt(252)
    #   sharpe_scale_t = clip(rolling_sharpe_t / sharpe_floor, 0, sharpe_scale_cap)
    #   position_t = position_pre_sharpe_t × sharpe_scale_t

    df['_paper_pnl_row'] = df['position_pre_sharpe'] * df['true']
    daily_paper = df.groupby('DATE')['_paper_pnl_row'].sum()

    roll_mu  = daily_paper.rolling(rolling_sharpe_window, min_periods=10).mean().shift(1)
    roll_sig = daily_paper.rolling(rolling_sharpe_window, min_periods=10).std().shift(1).clip(lower=1e-9)
    roll_sharpe = (roll_mu / roll_sig) * np.sqrt(252)

    if use_sharpe_sizing:
        # Scale: 0 when rolling Sharpe <= 0, 1 when rolling Sharpe >= sharpe_floor
        # Linear interpolation between 0 and sharpe_floor
        sharpe_scale_series = (roll_sharpe / sharpe_floor).clip(lower=0.0, upper=sharpe_scale_cap)
        # Warmup: don't apply before sharpe_warmup_days (use scale=1 during warmup)
        warmup_dates = set(sorted(daily_paper.index)[:sharpe_warmup_days])
        sharpe_scale_series[sharpe_scale_series.index.isin(warmup_dates)] = 1.0
        sharpe_scale_series = sharpe_scale_series.fillna(1.0)

        # Map scale back to bt rows by DATE
        sharpe_scale_map = sharpe_scale_series.to_dict()
        df['sharpe_scale'] = df['DATE'].map(sharpe_scale_map).fillna(1.0)

        if verbose:
            pct_full = (sharpe_scale_series >= sharpe_scale_cap).mean()
            pct_reduced = (sharpe_scale_series < 1.0).mean()
            pct_flat = (sharpe_scale_series < 0.1).mean()
            print(f"Sharpe control: full size {pct_full:.1%} of days | "
                  f"reduced {pct_reduced:.1%} | near-flat {pct_flat:.1%}")
    else:
        df['sharpe_scale'] = 1.0

    df['position'] = df['position_pre_sharpe'] * df['sharpe_scale']

    # Re-normalise after Sharpe scaling
    # (Sharpe scale is date-level so neutrality is preserved, but abs-sum changes)
    abs_final = df.groupby('DATE')['position'].transform(
        lambda x: x.abs().sum()
    ).replace(0, np.nan)
    df['position'] = (df['position'] / abs_final).fillna(0.0)

    # ----------------------------------------------------------------
    # Transaction costs
    # ----------------------------------------------------------------
    df['prev_pos'] = df.groupby('COUNTRY')['position'].shift(1).fillna(0.0)
    df['turnover'] = (df['position'] - df['prev_pos']).abs()
    df['tcost']    = df['turnover'] * cost_bps

    # ----------------------------------------------------------------
    # Dimensionless P&L
    # ----------------------------------------------------------------
    df['pnl_before_cost'] = df['position'] * df['true']
    df['pnl']             = df['pnl_before_cost'] - df['tcost']
    df['executed_position'] = df['position']

    # ----------------------------------------------------------------
    # EUR conversion
    # ----------------------------------------------------------------
    df['pnl_eur']             = np.nan
    df['pnl_before_cost_eur'] = np.nan
    df['tcost_eur']           = np.nan
    df['price_t']             = np.nan
    df['eur_scale']           = np.nan

    if price_panel is not None:
        px = price_panel.copy()
        px['DATE'] = pd.to_datetime(px['DATE']).dt.normalize()

        # Forward+backward fill so all BT dates have a price
        all_dates = pd.DataFrame(
            {'DATE': sorted(set(df['DATE'].unique()) | set(px['DATE'].unique()))}
        )
        px_full = (all_dates
                   .merge(px[['DATE', 'DE_price', 'FR_price']], on='DATE', how='left')
                   .sort_values('DATE'))
        px_full[['DE_price', 'FR_price']] = px_full[['DE_price', 'FR_price']].ffill().bfill()
        df = df.merge(px_full, on='DATE', how='left')

        df['price_t'] = np.where(
            df['COUNTRY'] == 'DE', df['DE_price'], df['FR_price']
        )

        if eur_scaling_method == 'fixed':
            # Single median price — Sharpe preserved
            # Median is a stable constant (not a signal),
            # so multiplying by it should not change the return distribution shape.
            median_price = df['price_t'].median()
            df['eur_scale'] = notional_mw * hours_per_day * median_price
            if verbose:
                print(f"EUR: fixed scale, median_price={median_price:.1f} EUR/MWh, "
                      f"scale={df['eur_scale'].iloc[0]:,.0f} EUR/unit")

        elif eur_scaling_method == 'regime':
            # Separate median per regime — captures regime specifics
            # but stable-ish within each regime (not capturing so much the daily noise)
            for regime_val in [0, 1]:
                mask_r  = df['vol_regime_flag'] == regime_val
                med_px  = df.loc[mask_r, 'price_t'].median() if mask_r.sum() > 0 else df['price_t'].median()
                df.loc[mask_r, 'eur_scale'] = notional_mw * hours_per_day * med_px
            if verbose:
                med_calm = df.loc[df['vol_regime_flag'] == 0, 'price_t'].median()
                med_hv   = df.loc[df['vol_regime_flag'] == 1, 'price_t'].median()
                print(f"EUR: regime scale, calm_median={med_calm:.1f}, "
                      f"hv_median={med_hv:.1f} EUR/MWh")

        elif eur_scaling_method == 'variable':
            # Per-row actual price — most realistic, Sharpe will change
            df['eur_scale'] = notional_mw * hours_per_day * df['price_t']
            if verbose:
                print(f"EUR: variable per-row price. EUR Sharpe != dimensionless Sharpe.")

        df['pnl_eur']             = df['pnl']             * df['eur_scale']
        df['pnl_before_cost_eur'] = df['pnl_before_cost'] * df['eur_scale']
        df['tcost_eur']           = df['tcost']            * df['eur_scale']

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    if verbose:
        daily_pos_sum = df.groupby('DATE')['position'].sum()
        max_dev = daily_pos_sum.abs().max()
        status  = f"OK: {max_dev:.2e}" if max_dev < 1e-4 else f"WARNING: {max_dev:.3e}"
        print(f"Market neutrality: {status}")

        for pref in ['DE', 'FR']:
            mask  = df['COUNTRY'] == pref
            n_hv  = int(df.loc[mask, 'vol_regime_flag'].sum())
            n_tot = int(mask.sum())
            print(f"{pref} high-vol: {n_hv}/{n_tot} ({n_hv/n_tot:.1%})")

        print(f"Cost rate: {cost_bps*1e4:.1f} bps per unit turnover")

    # Cleanup internal columns
    df.drop(columns=[c for c in df.columns if c.startswith('_')],
            inplace=True, errors='ignore')

    return df


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_trading_performance(
    bt: pd.DataFrame,
    cost_bps_label: str = None,
    notional_mw: float = None,
) -> tuple:
    """
    Full performance evaluation.
    EUR metrics auto-included if pnl_eur column present.
    Notional sensitivity table included if notional_mw provided.
    """
    daily_pnl    = bt.groupby('DATE')['pnl'].sum()
    daily_pnl_bc = bt.groupby('DATE')['pnl_before_cost'].sum()

    sharpe    = (daily_pnl.mean()    / (daily_pnl.std()    + 1e-9)) * np.sqrt(252)
    sharpe_bc = (daily_pnl_bc.mean() / (daily_pnl_bc.std() + 1e-9)) * np.sqrt(252)
    downside  = daily_pnl[daily_pnl < 0].std() + 1e-9
    sortino   = (daily_pnl.mean() / downside) * np.sqrt(252)
    cum_pnl   = daily_pnl.cumsum()
    max_dd    = (cum_pnl - cum_pnl.cummax()).min()
    ann_pnl   = daily_pnl.mean() * 252
    calmar    = ann_pnl / abs(max_dd) if max_dd != 0 else np.nan

    avg_turnover  = bt.groupby('DATE')['turnover'].sum().mean()
    total_pnl     = daily_pnl.sum()
    total_cost    = bt['tcost'].sum()
    pnl_bc        = bt['pnl_before_cost'].sum()
    cost_coverage = pnl_bc / (total_cost + 1e-9)

    win_rate       = (daily_pnl > 0).mean()
    active_days    = bt.groupby('DATE').apply(
        lambda x: (x['position'].abs() > 1e-6).any()
    )
    win_rate_active = (daily_pnl[active_days] > 0).mean()

    # Sharpe control diagnostics
    has_sharpe_scale = 'sharpe_scale' in bt.columns
    if has_sharpe_scale:
        daily_scale = bt.groupby('DATE')['sharpe_scale'].mean()

    # EUR metrics
    has_eur = 'pnl_eur' in bt.columns and bt['pnl_eur'].notna().any()
    eur_metrics = {}
    if has_eur:
        daily_eur     = bt.groupby('DATE')['pnl_eur'].sum()
        daily_eur_bc  = bt.groupby('DATE')['pnl_before_cost_eur'].sum()
        sharpe_eur    = (daily_eur.mean()    / (daily_eur.std()    + 1e-9)) * np.sqrt(252)
        sharpe_eur_bc = (daily_eur_bc.mean() / (daily_eur_bc.std() + 1e-9)) * np.sqrt(252)
        corr_px       = bt.groupby('DATE')['price_t'].mean().corr(daily_pnl)
        max_dd_eur    = (daily_eur.cumsum() - daily_eur.cumsum().cummax()).min()
        eur_metrics   = {
            'sharpe_eur':     sharpe_eur,
            'sharpe_eur_bc':  sharpe_eur_bc,
            'total_eur':      daily_eur.sum(),
            'ann_eur':        daily_eur.mean() * 252,
            'max_loss_eur':   daily_eur.min(),
            'max_gain_eur':   daily_eur.max(),
            'max_dd_eur':     max_dd_eur,
            'corr_price_pnl': corr_px,
        }

    # Regime breakdown — date-level majority vote
    regime_stats = {}
    if 'vol_regime_flag' in bt.columns:
        day_regime  = bt.groupby('DATE')['vol_regime_flag'].mean().ge(0.5).astype(int)
        daily_price = bt.groupby('DATE')['price_t'].mean() if has_eur else None
        daily_eur_s = bt.groupby('DATE')['pnl_eur'].sum() if has_eur else None

        for rv, label in [(0, 'calm'), (1, 'high_vol')]:
            mask_r  = day_regime == rv
            sub_pnl = daily_pnl[mask_r]
            if len(sub_pnl) == 0:
                continue
            s    = (sub_pnl.mean() / (sub_pnl.std() + 1e-9)) * np.sqrt(252)
            s_bc = (daily_pnl_bc[mask_r].mean() /
                    (daily_pnl_bc[mask_r].std() + 1e-9)) * np.sqrt(252)
            dd_r = (sub_pnl.cumsum() - sub_pnl.cumsum().cummax()).min()
            entry = {
                'sharpe':         s,
                'sharpe_bc':      s_bc,
                'mean_daily_pnl': sub_pnl.mean(),
                'win_rate':       (sub_pnl > 0).mean(),
                'total_pnl':      sub_pnl.sum(),
                'max_dd':         dd_r,
                'n_days':         len(sub_pnl),
                'pct_days':       len(sub_pnl) / len(daily_pnl),
            }
            if has_eur:
                sub_eur = daily_eur_s[mask_r]
                s_eur   = (sub_eur.mean() / (sub_eur.std() + 1e-9)) * np.sqrt(252)
                entry['sharpe_eur'] = s_eur
                entry['total_eur']  = sub_eur.sum()
                entry['avg_price']  = daily_price[mask_r].mean()
            regime_stats[label] = entry

    metrics = {
        'sharpe': sharpe, 'sharpe_before_cost': sharpe_bc,
        'sortino': sortino, 'calmar': calmar, 'max_drawdown': max_dd,
        'avg_daily_turnover': avg_turnover,
        'win_rate_daily': win_rate, 'win_rate_active': win_rate_active,
        'total_pnl': total_pnl, 'total_cost': total_cost,
        'pnl_before_cost': pnl_bc, 'cost_coverage': cost_coverage,
        'n_days': len(daily_pnl), 'regime_stats': regime_stats,
        **eur_metrics,
    }

    notional_str = f" | {notional_mw:.0f} MW" if notional_mw else ""
    label_str    = f" [{cost_bps_label}]" if cost_bps_label else ""
    print(f"\n=== Trading Performance{label_str}{notional_str} ===")
    print(f"Sharpe:             {sharpe:.4f}  (before cost: {sharpe_bc:.4f})")
    print(f"Sortino:            {sortino:.4f}")
    print(f"Calmar:             {calmar:.4f}")
    print(f"Max Drawdown:       {max_dd:.4f}")
    print(f"Win Rate (all):     {win_rate:.2%}")
    print(f"Win Rate (active):  {win_rate_active:.2%}")
    print(f"Avg Daily Turnover: {avg_turnover:.4f}")
    print(f"Total PnL:          {total_pnl:.4f}")
    print(f"Total Cost:         {total_cost:.4f}")
    print(f"PnL Before Cost:    {pnl_bc:.4f}")
    print(f"Cost Coverage:      {cost_coverage:.2f}x")

    if has_sharpe_scale:
        pct_full    = (daily_scale >= 0.99).mean()
        pct_reduced = (daily_scale < 0.99).mean()
        pct_flat    = (daily_scale < 0.10).mean()
        print(f"\nSharpe control:     full {pct_full:.1%} | "
              f"reduced {pct_reduced:.1%} | near-flat {pct_flat:.1%}")

    if has_eur:
        print(f"\n--- EUR metrics ---")
        print(f"Sharpe EUR:         {eur_metrics['sharpe_eur']:.4f}  "
              f"(before cost: {eur_metrics['sharpe_eur_bc']:.4f})")
        print(f"Total P&L:          {eur_metrics['total_eur']:,.0f} EUR  "
              f"({eur_metrics['total_eur']/1e6:.2f}M)")
        print(f"Annualised P&L:     {eur_metrics['ann_eur']:,.0f} EUR/year")
        print(f"Max Drawdown EUR:   {eur_metrics['max_dd_eur']:,.0f} EUR")
        print(f"Max daily loss:     {eur_metrics['max_loss_eur']:,.0f} EUR")
        print(f"Max daily gain:     {eur_metrics['max_gain_eur']:,.0f} EUR")
        print(f"Corr(price, PnL):   {eur_metrics['corr_price_pnl']:.3f}  "
              f"({'EUR Sharpe > dim' if eur_metrics['corr_price_pnl'] > 0 else 'EUR Sharpe < dim'})")

    if regime_stats:
        print(f"\nRegime breakdown:")
        for label, rs in regime_stats.items():
            eur_str = (f"  EUR_Sharpe={rs['sharpe_eur']:.3f}"
                       f"  avg_px={rs['avg_price']:.0f}"
                       f"  EUR={rs['total_eur']/1e3:,.0f}k"
                       if 'sharpe_eur' in rs else "")
            print(f"  {label:10s}  Sharpe={rs['sharpe']:.3f}"
                  f"  PnL={rs['total_pnl']:.3f}"
                  f"  MaxDD={rs['max_dd']:.3f}"
                  f"  WinRate={rs['win_rate']:.2%}"
                  f"  days={rs['n_days']} ({rs['pct_days']:.1%})"
                  + eur_str)

    print(f"\nCost sensitivity:")
    daily_turn = bt.groupby('DATE')['turnover'].sum()
    for test_bps in [0.0005, 0.0015, 0.0025, 0.0050, 0.0100]:
        hyp_pnl = daily_pnl_bc - daily_turn * test_bps
        s   = (hyp_pnl.mean() / (hyp_pnl.std() + 1e-9)) * np.sqrt(252)
        cov = pnl_bc / (daily_turn.sum() * test_bps + 1e-9)
        print(f"  {test_bps*1e4:5.1f} bps:  Sharpe={s:.3f}  coverage={cov:.1f}x")

    if notional_mw and has_eur:
        print(f"\nNotional sensitivity:")
        daily_eur_base = bt.groupby('DATE')['pnl_eur'].sum()
        for mw in [10, 25, 50, 100, 150]:
            ratio      = mw / notional_mw
            scaled     = daily_eur_base * ratio
            s          = (scaled.mean() / (scaled.std() + 1e-9)) * np.sqrt(252)
            max_dd_mw  = (scaled.cumsum() - scaled.cumsum().cummax()).min()
            print(f"  {mw:5.0f} MW:  Total={scaled.sum()/1e3:,.0f} kEUR  "
                  f"Ann={scaled.mean()*252/1e3:,.0f} kEUR  "
                  f"Sharpe={s:.3f}  MaxDD={max_dd_mw/1e3:,.0f} kEUR")

    return metrics, daily_pnl
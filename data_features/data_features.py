# data_features.py
import numpy as np
import pandas as pd
import warnings
import gc


# ------------------------------------
# Loads the data into a common panel
# --------------------------------------
def load_coupled_panel(germany_file: str, france_file: str) -> pd.DataFrame:
    """
    Load Germany and France CSVs.
    Output: one row per DATE, with DE_* and FR_* columns in same row.
    """
    def sanitize_col(c: str) -> str:
        """Sanitize a column name"""
        return str(c).strip().replace(' ', '_').replace('/', '_').replace('-', '_')
    g = pd.read_csv(germany_file, parse_dates=[0])
    f = pd.read_csv(france_file, parse_dates=[0])

    g.rename(columns={g.columns[0]: 'DATE'}, inplace=True)
    f.rename(columns={f.columns[0]: 'DATE'}, inplace=True)

    g['DATE'] = pd.to_datetime(g['DATE']).dt.normalize()
    f['DATE'] = pd.to_datetime(f['DATE']).dt.normalize()

    def prefix(df, code):
        df = df.copy()
        for c in df.columns:
            if c == 'DATE':
                continue
            df.rename(columns={c: f"{code}_{sanitize_col(c)}"}, inplace=True)
        return df

    g = prefix(g, "DE")
    f = prefix(f, "FR")

    panel = pd.merge(g, f, on="DATE", how="inner")
    panel = panel.sort_values("DATE").reset_index(drop=True)

    return panel

#------------------------------------------------
# Feature engineering
# ------------------------------------------------------
################### THIS FUNCTION NOW DOES LAGS WITH RESPECT TO TIME T+1!!!! TARGET IS AT T+1! SO LAG-1 IS T, LAG-7 
################### IS T-6 !! CAREFUL WITH HOW THE TARGET IS DEFINED TO AVOID LOOKAHEAD!!!!!
def build_features(panel: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    df = panel.copy()

    prefixes = ['DE', 'FR']
    core = ['load_MW', 'wind_total', 'solar_total',
            'wind_share', 'solar_share',
            'temp_C', 'wind_speed', 'precip',
            'daily_log_return', 'realized_vol']

    # Lags
    for pref in prefixes:
        for base in core:
            col = f"{pref}_{base}"
            if col not in df.columns:
                continue
            #df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag7"] = df[col].shift(6)

    # Rolling means/std 
    for pref in prefixes:
        for base in ['load_MW', 'wind_total', 'solar_total']:
            col = f"{pref}_{base}"
            if col not in df.columns:
                continue
#            df[f"{col}_rm_7"] = df[col].shift(1).rolling(7, min_periods=2).mean()
#            df[f"{col}_std_7"] = df[col].shift(1).rolling(7, min_periods=2).std()
#            df[f"{col}_rm_30"] = df[col].shift(1).rolling(30, min_periods=5).mean()
            df[f"{col}_rm_7"] = df[col].rolling(7, min_periods=2).mean()
            df[f"{col}_std_7"] = df[col].rolling(7, min_periods=2).std()
            df[f"{col}_rm_30"] = df[col].rolling(30, min_periods=5).mean()


    # Volatility features
    for pref in prefixes:
        rcol = f"{pref}_daily_log_return"
        if rcol not in df.columns:
            continue
        #df[f"{rcol}_vol_14"] = df[rcol].shift(1).rolling(14, min_periods=5).std()
        df[f"{rcol}_vol_14"] = df[rcol].rolling(14, min_periods=5).std()

    # Cross-country features 
    if 'DE_load_MW' in df.columns and 'FR_load_MW' in df.columns:
        df['LOAD_IMBALANCE'] = df['DE_load_MW'] - df['FR_load_MW']

    if 'DE_wind_total' in df.columns and 'FR_wind_total' in df.columns:
        df['WIND_IMBALANCE'] = df['DE_wind_total'] - df['FR_wind_total']

    return df

#------------------------------------------------------
# Shifting target to t+1, winsorization (optional), zero-clipping
# -------------------------------------------------------
def make_t_plus_1_targets_winsorized(
    panel: pd.DataFrame,
    de_real_col: str = "DE_realized_vol",
    de_garch_col: str = "DE_garch_sigma",   # works for both GARCH and HAR-RV (same name)
    fr_real_col: str = "FR_realized_vol",
    fr_garch_col: str = "FR_garch_sigma",
    winsor_pct: float = 0.01,               # symmetric winsorization at [pct, 1-pct]
    clip_zeros: bool = True,                # replace realized_vol == 0 with NaN before log
    verbose: bool = True
) -> pd.DataFrame:
    """
    1. clip_zeros: set realized_vol == 0 (or very close) to NaN before taking log.
    2. winsor_pct: after computing log-residuals, symmetrically winsorize
       at the [winsor_pct, 1-winsor_pct] quantiles (default 1%/99%).
       This is applied BEFORE the t+1 shift
    """
    EPS = 1e-12

    df = panel.copy().sort_values("DATE").reset_index(drop=True)

    def _safe_log_resid(real_col, baseline_col):
        rv  = df[real_col].astype(float).copy()
        sig = df[baseline_col].astype(float).copy()

        if clip_zeros:
            # treat very-near-zero realized vol as missing
            # (log of 0 is -inf; these days carry no usable information)
            rv[rv < 1e-6] = np.nan

        log_rv  = np.log(rv  + EPS)
        log_sig = np.log(sig + EPS)
        resid   = log_rv - log_sig

        if winsor_pct > 0.0:
            lo = resid.quantile(winsor_pct)
            hi = resid.quantile(1.0 - winsor_pct)
            n_clipped = int(((resid < lo) | (resid > hi)).sum())
            resid = resid.clip(lower=lo, upper=hi)
            if verbose:
                prefix = real_col.split("_")[0]
                print(
                    f"[winsor] {prefix}: clipped {n_clipped} observations "
                    f"| lo={lo:.3f}  hi={hi:.3f}"
                )

        return resid

    de_resid = _safe_log_resid(de_real_col, de_garch_col)
    fr_resid = _safe_log_resid(fr_real_col, fr_garch_col)

    # shift -1: target at row t is the residual at row t+1
    df["DE_residual_target"] = de_resid.shift(-1)
    df["FR_residual_target"] = fr_resid.shift(-1)

    if verbose:
        for col in ["DE_residual_target", "FR_residual_target"]:
            s = df[col].dropna()
            print(
                f"[target stats] {col}: n={len(s)}  "
                f"mean={s.mean():.3f}  std={s.std():.3f}  "
                f"skew={s.skew():.3f}  "
                f"[{s.quantile(0.01):.2f}, {s.quantile(0.99):.2f}]"
            )

    return df
import numpy as np
import pandas as pd

def build_features(X, Y=None, add_target=False, eps=1e-8):
    """
    Build features for electricity ML, fully look-ahead-free.
    
    Parameters
    ----------
    X : pd.DataFrame
        Raw features with ID, DAY_ID, COUNTRY, etc.
    Y : pd.DataFrame, optional
        Target dataframe with 'ID' and 'TARGET'
    add_target : bool
        Whether to merge Y as volatility
    eps : float
        Small value to prevent division by zero
    """
    df = X.copy()

    # ---- merge target if needed ----
    if add_target:
        if Y is None:
            raise ValueError("Y must be provided when add_target=True")
        Xs = df.sort_values('ID').reset_index(drop=True)
        Ys = Y.sort_values('ID').reset_index(drop=True)
        if not (Xs['ID'].values == Ys['ID'].values).all():
            raise ValueError("X and Y IDs do not match")
        df = Xs.copy()
        df['volatility'] = Ys['TARGET'].values

    # ---- DAY_ID handling ----
    if not np.issubdtype(df['DAY_ID'].dtype, np.number):
        try:
            df['DAY_ID'] = pd.to_datetime(df['DAY_ID'])
        except Exception:
            pass

    # ---- sort for time operations ----
    df = df.sort_values(['COUNTRY','DAY_ID']).reset_index(drop=True)

    # ---- VOLATILITY features (lagged) ----
    if add_target:
        for lag in [1,3,7]:
            df[f'vol_lag{lag}'] = df.groupby('COUNTRY')['volatility'].shift(lag)

        for w in [7,30]:
            df[f'vol_roll_std_{w}'] = df.groupby('COUNTRY')['volatility'] \
                .transform(lambda x: x.rolling(w, min_periods=3).std().shift(1))

            df[f'vol_roll_mean_{w}'] = df.groupby('COUNTRY')['volatility'] \
                .transform(lambda x: x.rolling(w, min_periods=3).mean().shift(1))

    # ---- COUNTRY FLAG ----
    df['IS_FR'] = (df['COUNTRY'] == 'FR').astype(int)

    # ---- CREATE LAGGED RAW FEATURES TO AVOID LOOKAHEAD ----
    raw_cols = ['DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD','DE_WINDPOW','FR_WINDPOW',
                'DE_SOLAR','FR_SOLAR','FR_NUCLEAR','DE_NUCLEAR','DE_FR_EXCHANGE',
                'FR_DE_EXCHANGE','GAS_RET','COAL_RET','CARBON_RET','DE_COAL','DE_LIGNITE',
                'DE_TEMP','FR_TEMP','DE_WIND','FR_WIND','DE_RAIN','FR_RAIN',
                'DE_CONSUMPTION','FR_CONSUMPTION']

    for col in raw_cols:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('COUNTRY')[col].shift(1)

    # ---- SPREADS / FLAGS using lagged versions ----
    df['LOAD_IMBALANCE'] = df['DE_RESIDUAL_LOAD_lag1'] - df['FR_RESIDUAL_LOAD_lag1']
    df['WIND_IMBALANCE'] = df['DE_WINDPOW_lag1'] - df['FR_WINDPOW_lag1']
    df['SOLAR_IMBALANCE'] = df['DE_SOLAR_lag1'] - df['FR_SOLAR_lag1']
    df['NUCLEAR_IMBALANCE'] = df['FR_NUCLEAR_lag1'] - df['DE_NUCLEAR_lag1']

    df['FLOW_PRESSURE'] = df['DE_FR_EXCHANGE_lag1'] - df['FR_DE_EXCHANGE_lag1']
    df['TOTAL_FLOW'] = df['DE_FR_EXCHANGE_lag1'].abs() + df['FR_DE_EXCHANGE_lag1'].abs()

    df['DE_RESIDUAL_STRESS'] = df['DE_RESIDUAL_LOAD_lag1'] / (df['DE_CONSUMPTION_lag1'] + eps)
    df['FR_RESIDUAL_STRESS'] = df['FR_RESIDUAL_LOAD_lag1'] / (df['FR_CONSUMPTION_lag1'] + eps)

    df['GAS_COAL_SPREAD'] = df['GAS_RET_lag1'] - df['COAL_RET_lag1']
    df['CARBON_PRESSURE'] = df['CARBON_RET_lag1'] * (df['DE_COAL_lag1'] + df['DE_LIGNITE_lag1'])

    # ---- REGIME features ----
    if 'GAS_RET_lag1' in df.columns:
        df['GAS_RET_30m'] = df.groupby('COUNTRY')['GAS_RET_lag1'] \
            .transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1))

    if 'DE_RESIDUAL_LOAD_lag1' in df.columns:
        df['LOAD_TREND_30'] = df.groupby('COUNTRY')['DE_RESIDUAL_LOAD_lag1'] \
            .transform(lambda x: x.rolling(30, min_periods=10).mean().shift(1))

    df['REL_RENEWABLE'] = (df['DE_WINDPOW_lag1'] + df['DE_SOLAR_lag1']) - \
                          (df['FR_WINDPOW_lag1'] + df['FR_SOLAR_lag1'])

    # ---- WEATHER anomalies ----
    weather_cols = ['DE_TEMP','FR_TEMP','DE_WIND','FR_WIND','DE_RAIN','FR_RAIN']
    for c in weather_cols:
        if f'{c}_lag1' in df.columns:
            m = df.groupby('COUNTRY')[f'{c}_lag1'].transform(lambda x: x.rolling(7, min_periods=3).mean().shift(1))
            df[f'{c}_ANOM'] = df[f'{c}_lag1'] - m

    # ---- EXTRA ROLLING STATS ----
    rolling_cols = ['DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD','DE_WINDPOW','FR_WINDPOW',
                    'DE_CONSUMPTION','FR_CONSUMPTION']
    for col in rolling_cols:
        if f'{col}_lag1' in df.columns:
            for w in [3,7,30]:
                df[f'{col}_rm_{w}'] = df.groupby('COUNTRY')[f'{col}_lag1'] \
                    .transform(lambda x: x.rolling(w, min_periods=2).mean().shift(1))
                df[f'{col}_std_{w}'] = df.groupby('COUNTRY')[f'{col}_lag1'] \
                    .transform(lambda x: x.rolling(w, min_periods=2).std().shift(1))

    # ---- INTERACTION ----
    df['LOADxGAS'] = df['LOAD_IMBALANCE'] * df['GAS_RET_30m']

    # ---- DAY-RANK features ----
    rank_cols = ['DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD','LOAD_IMBALANCE','TOTAL_FLOW']
    for c in rank_cols:
        if f'{c}_lag1' in df.columns:
            df[f'{c}_day_rank'] = df.groupby('DAY_ID')[f'{c}_lag1'] \
                .transform(lambda x: x.rank(method='average') / len(x))

    return df
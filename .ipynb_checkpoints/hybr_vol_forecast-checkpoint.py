# hybr_vol_forecast.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
from arch import arch_model
import xgboost as xgb
from tqdm import tqdm
import time
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau, ks_2samp
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed
from typing import Sequence, List, Dict, Any, Tuple, Optional
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import gc
import optuna
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
EPS = 1e-12


class HybrVolClass:

    def __init__(self):
        # state
        self.panel: Optional[pd.DataFrame] = None
        self.selected_features: List[str] = []
        self.diagnostics: Dict[str, Any] = {}
        self.model_results: Dict[str, Any] = {}


    @staticmethod
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


    ################### THIS FUNCTION NOW DOES LAGS WITH RESPECT TO TIME T+1!!!! TARGET IS AT T+1! SO LAG-1 IS T, LAG-7 
    ################### IS T-6 !! CAREFUL WITH HOW THE TARGET IS DEFINED TO AVOID LOOKAHEAD!!!!!
    @staticmethod
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
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from arch import arch_model
import xgboost as xgb
from tqdm import tqdm
import time
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed, parallel_backend
from typing import Sequence, List, Dict, Any, Tuple, Optional
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
import gc
import optuna
import logging


import warnings
warnings.filterwarnings("ignore")


SEED = 42
np.random.seed(SEED)

# ----------------
# GARCH (1,1) model -- currently not used
# -----------------
def add_garch_parallel(panel: pd.DataFrame,
                       min_obs=250,
                       window=150,          # can be int or dict
                       p=1, q=1,
                       n_jobs=4):
    """
    Fit GARCH(p,q) models per country in parallel,
    store sigma forecasts in panel, and compute metrics
    vs realized volatility columns (DE_realized_vol, FR_realized_vol).

    window:
        - int → same window for all countries
        - dict → e.g. {"DE": 500, "FR": 150}
    """
    df = panel.copy()
    metrics = {}

    # allow per-country window
    if isinstance(window, dict):
        window_map = window
    else:
        window_map = {"DE": window, "FR": window}

    def fit_garch_for_country(pref):
        ret_col = f"{pref}_daily_log_return"
        sigma_col = f"{pref}_garch_sigma"
        realized_col = f"{pref}_realized_vol"

        country_window = window_map.get(pref, window)

        sigmas = np.full(len(df), np.nan)
        rets = df[ret_col].values

        for i in range(len(df)):
            if i < min_obs:
                continue

            hist = rets[max(0, i-country_window):i]
            hist = hist[np.isfinite(hist)]

            effective_min_obs = min(min_obs, country_window)
            
            if len(hist) < effective_min_obs:
                continue

            try:
                am = arch_model(hist, vol='Garch', p=p, q=q, dist='t', rescale=True)
                res = am.fit(disp='off')
                fcast = res.forecast(horizon=1, reindex=False)
                var = fcast.variance.values[-1, 0]
                sigmas[i] = np.sqrt(max(var, 1e-16))
            except:
                continue

        # Compute metrics vs existing realized volatility column
        mask = ~np.isnan(sigmas) & ~np.isnan(df[realized_col])
        if mask.sum() > 0:
            y_pred = sigmas[mask]
            y_true = df.loc[mask, realized_col].values
            ic = spearmanr(y_pred, y_true).correlation
            rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
            mae = float(np.mean(np.abs(y_pred - y_true)))
        else:
            ic = rmse = mae = np.nan

        return pref, sigmas, {'spearman_ic': ic, 'rmse': rmse, 'mae': mae}

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_garch_for_country)(pref) for pref in ['DE', 'FR']
    )

    # Assign computed sigmas back to main df
    for pref, sigmas, m in results:
        df[f"{pref}_garch_sigma"] = sigmas
        metrics[pref] = m

    return df, metrics

# ----------------
# HAR-RV model
# -----------------

def add_har_rv_parallel(
    panel: pd.DataFrame,
    min_obs: int = 22,
    window: object = 150,
    n_jobs: int = -1
) -> tuple:

    df = panel.copy()

    if isinstance(window, dict):
        window_map = window
    else:
        window_map = {"DE": window, "FR": window}

    def fit_har_for_country(pref: str):

        rv_col = f"{pref}_realized_vol"
        sigma_col = f"{pref}_garch_sigma"
        realized_col = rv_col

        country_window = window_map.get(pref, window if isinstance(window, int) else 150)

        rv_raw = df[rv_col].astype(float)

        # log RV (safe)
        rv = np.log(np.clip(rv_raw, 1e-8, None))

        n = len(rv)

        # ------------------------------------------------------------------
        # PRECOMPUTE HAR FEATURES (NO LOOKAHEAD)
        # ------------------------------------------------------------------

        rv_d = rv.shift(1)
        rv_w = rv.shift(1).rolling(5).mean()
        rv_m = rv.shift(1).rolling(22).mean()

        X_full = pd.concat(
            [
                pd.Series(1.0, index=rv.index),
                rv_d,
                rv_w,
                rv_m
            ],
            axis=1
        )

        X_full.columns = ["const", "rv_d", "rv_w", "rv_m"]

        forecasts = np.full(n, np.nan)

        X_np = X_full.values
        y_np = rv.values

        # ------------------------------------------------------------------
        # ROLLING OLS
        # ------------------------------------------------------------------

        for i in range(n):

            if i < max(min_obs, 22):
                continue

            start = max(0, i - country_window)

            X = X_np[start:i]
            y = y_np[start:i]

            valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

            X = X[valid]
            y = y[valid]

            if len(y) < 10:
                forecasts[i] = np.nanmean(y_np[start:i])
                continue

            try:

                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

                x_f = X_np[i]

                if not np.isfinite(x_f).all():
                    forecasts[i] = np.nanmean(y_np[start:i])
                    continue

                pred = beta @ x_f

                # prevent extreme values before exponentiating
                pred = max(pred, -20)

                forecasts[i] = np.exp(pred)

            except Exception:
                forecasts[i] = np.nanmean(y_np[start:i])

        # ------------------------------------------------------------------
        # DIAGNOSTICS 
        # ------------------------------------------------------------------

        mask = ~np.isnan(forecasts) & ~np.isnan(df[realized_col])

        if mask.sum() > 0:

            y_pred = forecasts[mask]
            y_true = df.loc[mask, realized_col].values

            ic = spearmanr(y_pred, y_true).correlation
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae = float(np.mean(np.abs(y_pred - y_true)))

        else:

            ic = rmse = mae = np.nan

        return pref, forecasts, {"spearman_ic": ic, "rmse": rmse, "mae": mae}

    # ----------------------------------------------------------------------
    # PARALLEL EXECUTION
    # ----------------------------------------------------------------------

    with parallel_backend("loky"):
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(fit_har_for_country)(pref) for pref in ["DE", "FR"]
        )

    # ----------------------------------------------------------------------
    # COLLECT RESULTS
    # ----------------------------------------------------------------------

    for pref, forecasts, _ in results:
        df[f"{pref}_garch_sigma"] = forecasts

    metrics = {pref: m for pref, _, m in results}

    return df, metrics


#---------------------------------------
# Feature selector -- regime-aware with z-score vol as indicator, rolling windows
# ----------------------------------------


# Calendar stems — forced into union, excluded from selection pool
_CALENDAR_STEMS = (
    #[f"dow_{d}" for d in range(7)]
     #["week_sin", "week_cos"] # "doy_sin", "doy_cos"]
)

# Shared unprefixed feature stems considered for both country models
_SHARED_STEMS = [
    "LOAD_IMBALANCE", "WIND_IMBALANCE",
    "VOL_SPREAD", "VOL_RATIO",
]


def feature_selection(
    df: pd.DataFrame,
    features: List[str] = None,
    target_de: str = "DE_residual_target",
    target_fr: str = "FR_residual_target",
    # --- window ---
    train_window: int = 1200,
    min_train_days: int = 280,
    test_horizon: int = 21,
    gap_days: int = 1,
    # --- selection ---
    prefilter_top_k: int = 20,
    max_features: int = 10,
    freq_threshold: float = 0.70,
    freq_threshold_regime: float = 0.45,
    # --- vol_zscore regime splitting (optional) ---
    # When True, each fold is split into high/low vol regime halves using
    # a rolling zscore of log(realized_vol). Importance scores are collected
    # separately per regime so the diagnostics show which features matter
    # in each regime. The zscore is NOT added as a feature.
    use_vol_zscore: bool = True,
    vol_zscore_window: int = 252,
    vol_zscore_threshold: float = 0.5,   # zscore > threshold => high-vol regime
    # --- forced features (beyond calendar) ---
    force_features: List[str] = None,
    # --- model ---
    xgb_params_prefilter: Dict[str, Any] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:

    if xgb_params_prefilter is None:
        xgb_params_prefilter = {
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "max_depth":        3,
            "eta":              0.03,
            "subsample":        0.6,
            "colsample_bytree": 0.5,
            "lambda":           300,
            "alpha":            10,
            "min_child_weight": 30,
            "seed":             SEED,
            "nthread":          1,
            "verbosity":        0,
        }

    if force_features is None:
        force_features = []

    # ---- Calendar: only PREFIXED versions, excluded from selection pool ------
    calendar_forced: List[str] = []
    for stem in _CALENDAR_STEMS:
        for pref in ["DE", "FR"]:
            col = f"{pref}_{stem}"
            if col in df.columns:
                calendar_forced.append(col)
    calendar_forced = list(dict.fromkeys(calendar_forced))

    all_forced = list(dict.fromkeys(calendar_forced + force_features))

    # Set of column names to exclude from selection candidates
    # (all calendar variants — prefixed and unprefixed)
    calendar_exclude = set()
    for stem in _CALENDAR_STEMS:
        calendar_exclude.add(stem)
        for pref in ["DE", "FR"]:
            calendar_exclude.add(f"{pref}_{stem}")

    # ---- Feature autodetection -----------------------------------------------
    exclude_exact = {
        "DATE", target_de, target_fr,
        "DE_garch_sigma", "FR_garch_sigma",
        "DE_realized_vol","FR_realized_vol"
        "COUNTRY", "COUNTRY_FLAG",
        "DE_vol_zscore", "FR_vol_zscore",   # regime signal, not a feature
        "DE_load_MW", "FR_load_MW",
        "DE_Wind_Offshore_Actual_Aggregated", "DE_Wind_Onshore_Actual_Aggregated",
        "FR_Wind_Onshore_Actual_Aggregated","FR_Wind_Offshore_Actual_Aggregated",
        "FR_doy_cos","FR_doy_sin","FR_week_cos","FR_week_sin", 
        "DE_doy_cos","DE_doy_sin","DE_week_cos","DE_week_sin",
        "DE_wind_total", "DE_wind_share", "DE_Hydro_Pumped_Storage_Actual_Consumption"
        
        
    } | calendar_exclude

    if features is None:
        candidate_de = [
            c for c in df.columns
            if c.startswith("DE_") and c not in exclude_exact
        ]
        candidate_fr = [
            c for c in df.columns
            if c.startswith("FR_") and c not in exclude_exact
        ]
        shared_in_df = [
            c for c in _SHARED_STEMS
            if c in df.columns and c not in exclude_exact
        ]
        candidate_de = list(dict.fromkeys(candidate_de + shared_in_df))
        candidate_fr = list(dict.fromkeys(candidate_fr + shared_in_df))
    else:
        candidate_de = [
            f for f in features
            if f.startswith("DE_") and f not in exclude_exact
        ]
        candidate_fr = [
            f for f in features
            if f.startswith("FR_") and f not in exclude_exact
        ]
        shared_in_feats = [
            f for f in features
            if not f.startswith("DE_") and not f.startswith("FR_")
            and f not in exclude_exact
        ]
        candidate_de = list(dict.fromkeys(candidate_de + shared_in_feats))
        candidate_fr = list(dict.fromkeys(candidate_fr + shared_in_feats))

    df = df.sort_values("DATE").reset_index(drop=True)
    unique_dates = np.sort(df["DATE"].unique())

    starts = list(range(
        min_train_days,
        len(unique_dates) - gap_days - test_horizon + 1,
        test_horizon,
    ))
    if not starts:
        raise ValueError("Date range too small for min_train_days/test_horizon.")

    # ---- Helpers -------------------------------------------------------------
    def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 5:
                    return 0.0
                xs, ys = x[mask], y[mask]
                if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
                    return 0.0
                r = spearmanr(xs, ys).correlation
                return 0.0 if not np.isfinite(r) else float(r)
            except Exception:
                return 0.0

    def evaluate_preds(y_true: np.ndarray, preds: np.ndarray) -> dict:
        mask = np.isfinite(y_true) & np.isfinite(preds)
        if mask.sum() == 0:
            return {"n": 0, "pooled_ic": np.nan, "rmse": np.nan, "mae": np.nan}
        y, p = y_true[mask], preds[mask]
        return {
            "n":         int(mask.sum()),
            "pooled_ic": safe_spearman(p, y),
            "rmse":      float(np.sqrt(mean_squared_error(y, p))),
            "mae":       float(mean_absolute_error(y, p)),
        }

    # ---- Per-fold vol zscore (training window only) --------------------------
    def _compute_fold_zscore(fold_df, train_mask, pref):
        """
        Returns a boolean Series (same index as fold_df) where True = high-vol
        regime row. Computed from training window statistics only — no lookahead.
        Used to split training data for regime-aware importance, NOT as a feature.
        """
        rv_col = f"{pref}_realized_vol"
        if rv_col not in fold_df.columns:
            return pd.Series(False, index=fold_df.index)

        rv_tr = fold_df.loc[train_mask, rv_col].astype(float).clip(lower=1e-12)
        log_rv_tr = np.log(rv_tr)
        mu  = log_rv_tr.rolling(vol_zscore_window, min_periods=30).mean()
        sig = log_rv_tr.rolling(vol_zscore_window, min_periods=30).std().clip(lower=1e-8)
        zscore_tr = (log_rv_tr - mu) / sig

        high_vol = pd.Series(False, index=fold_df.index)
        high_vol.loc[train_mask] = zscore_tr > vol_zscore_threshold
        return high_vol

    # ---- Fold worker ---------------------------------------------------------
    def run_fold(i: int) -> Optional[dict]:
        train_start = max(0, i - train_window) if train_window is not None else 0
        train_dates = unique_dates[train_start:i]
        test_dates  = unique_dates[i + gap_days: i + gap_days + test_horizon]
        if len(train_dates) < min_train_days or len(test_dates) == 0:
            return None

        train_mask = df["DATE"].isin(train_dates)
        test_mask  = df["DATE"].isin(test_dates)
        fold_df    = df   # read-only, no copy needed

        # Regime masks (only used when use_vol_zscore=True)
        regime_masks = {}
        if use_vol_zscore:
            for pref in ["DE", "FR"]:
                high = _compute_fold_zscore(fold_df, train_mask, pref)
                regime_masks[pref] = {
                    "high": train_mask & high,
                    "low":  train_mask & ~high,
                }

        out = {
            "fold_start_idx": i,
            "train_from":  train_dates[0],
            "train_until": train_dates[-1],
            "test_from":   test_dates[0],
            "test_to":     test_dates[-1],
            "DE": {},
            "FR": {},
        }

        def _fit_booster(X_tr, y_tr):
            """Fit one XGBoost with internal val split. Returns (booster, importance) or None."""
            valid = y_tr.notna() & np.isfinite(y_tr)
            if valid.sum() < 40:
                return None, {}
            X_c = X_tr.loc[valid]; y_c = y_tr.loc[valid].astype(float)
            val_size = max(int(0.15 * len(X_c)), 15)
            if len(X_c) <= val_size + 15:
                return None, {}
            X_t, X_v = X_c.iloc[:-val_size], X_c.iloc[-val_size:]
            y_t, y_v = y_c.iloc[:-val_size], y_c.iloc[-val_size:]
            try:
                dtrain = xgb.DMatrix(X_t, label=y_t, missing=np.nan)
                dval   = xgb.DMatrix(X_v, label=y_v, missing=np.nan)
                bst = xgb.train(
                    xgb_params_prefilter, dtrain,
                    num_boost_round=800,
                    evals=[(dval, "val")],
                    early_stopping_rounds=40,
                    verbose_eval=False,
                )
                imp = (bst.get_score(importance_type="gain")
                       or bst.get_score(importance_type="weight")
                       or {})
                return bst, imp
            except Exception:
                return None, {}

        def _country_select(candidates: List[str], target_col: str,
                             pref: str) -> dict:
            res = {
                "prefilter_scores":    [],
                "prefilter_feats":     [],
                "selected_feats":      [],
                "importance":          {},
                "importance_high_vol": {},
                "importance_low_vol":  {},
                "train_metrics":       None,
                "test_metrics":        None,
            }
            if target_col not in fold_df.columns:
                return res

            train_sub = fold_df.loc[train_mask]
            y_tr_full = train_sub[target_col].values.astype(float)

            # Univariate Spearman prefilter on full training window
            scores = []
            for feat in candidates:
                if feat not in train_sub.columns:
                    scores.append((feat, 0.0))
                    continue
                x = train_sub[feat].values.astype(float)
                scores.append((feat, safe_spearman(x, y_tr_full)))

            scores.sort(key=lambda kv: abs(kv[1]), reverse=True)
            res["prefilter_scores"] = scores
            prefilter_feats = [f for f, _ in scores[:prefilter_top_k]]
            res["prefilter_feats"] = prefilter_feats
            if not prefilter_feats:
                return res

            X_tr_all = train_sub.reindex(columns=prefilter_feats)
            y_tr_ser = train_sub[target_col]

            # Full-window model (always fitted)
            bst_full, imp_full = _fit_booster(X_tr_all, y_tr_ser)
            res["importance"] = imp_full

            # Regime-split models (only when use_vol_zscore=True)
            if use_vol_zscore and pref in regime_masks:
                high_mask = regime_masks[pref]["high"]
                low_mask  = regime_masks[pref]["low"]

                X_high = fold_df.loc[high_mask].reindex(columns=prefilter_feats)
                y_high = fold_df.loc[high_mask, target_col]
                X_low  = fold_df.loc[low_mask].reindex(columns=prefilter_feats)
                y_low  = fold_df.loc[low_mask, target_col]

                _, imp_high = _fit_booster(X_high, y_high)
                _, imp_low  = _fit_booster(X_low,  y_low)

                res["importance_high_vol"] = imp_high
                res["importance_low_vol"]  = imp_low

            # Select top-k by full-window importance
            if imp_full:
                feat_sorted = sorted(
                    prefilter_feats,
                    key=lambda f: imp_full.get(f, 0.0),
                    reverse=True,
                )
                res["selected_feats"] = feat_sorted[:max_features]

            # Diagnostics
            if bst_full is not None:
                preds_tr = bst_full.predict(xgb.DMatrix(X_tr_all, missing=np.nan))
                res["train_metrics"] = evaluate_preds(y_tr_full, preds_tr)

                test_sub   = fold_df.loc[test_mask]
                X_test     = test_sub.reindex(columns=prefilter_feats)
                y_test     = test_sub[target_col].values.astype(float)
                preds_test = bst_full.predict(xgb.DMatrix(X_test, missing=np.nan))
                res["test_metrics"] = evaluate_preds(y_test, preds_test)

            return res

        out["DE"] = _country_select(candidate_de, target_de, "DE")
        out["FR"] = _country_select(candidate_fr, target_fr, "FR")
        return out

    # ---- Run folds -----------------------------------------------------------
    #results = Parallel(n_jobs=n_jobs)(delayed(run_fold)(i) for i in starts)
    with parallel_backend("loky"):
        results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_fold)(i) for i in starts
    )
    results = [r for r in results if r is not None]
    n_folds = len(results)
    if n_folds == 0:
        raise ValueError("No folds produced.")

    # ---- Aggregate -----------------------------------------------------------
    REGIME_BOUNDARY = pd.Timestamp("2022-01-01")

    sel_counts_de:  Dict[str, int]   = {}
    sel_counts_fr:  Dict[str, int]   = {}
    sel_pre_de:     Dict[str, int]   = {}
    sel_post_de:    Dict[str, int]   = {}
    sel_pre_fr:     Dict[str, int]   = {}
    sel_post_fr:    Dict[str, int]   = {}

    agg_imp_de:          Dict[str, float] = {}
    agg_imp_fr:          Dict[str, float] = {}
    agg_imp_pre_de:      Dict[str, float] = {}
    agg_imp_post_de:     Dict[str, float] = {}
    agg_imp_pre_fr:      Dict[str, float] = {}
    agg_imp_post_fr:     Dict[str, float] = {}
    agg_imp_high_de:     Dict[str, float] = {}
    agg_imp_low_de:      Dict[str, float] = {}
    agg_imp_high_fr:     Dict[str, float] = {}
    agg_imp_low_fr:      Dict[str, float] = {}

    fold_level_metrics = []
    n_pre = n_post = 0

    for r in results:
        is_post = int(pd.Timestamp(r["test_from"]) >= REGIME_BOUNDARY)
        n_post += is_post; n_pre += 1 - is_post

        de, fr = r["DE"], r["FR"]
        fold_level_metrics.append({
            "train_from":    r["train_from"],
            "train_until":   r["train_until"],
            "test_from":     r["test_from"],
            "test_to":       r["test_to"],
            "DE_train_ic":   de.get("train_metrics", {}).get("pooled_ic", np.nan),
            "DE_train_rmse": de.get("train_metrics", {}).get("rmse",      np.nan),
            "DE_test_ic":    de.get("test_metrics",  {}).get("pooled_ic", np.nan),
            "DE_test_rmse":  de.get("test_metrics",  {}).get("rmse",      np.nan),
            "FR_train_ic":   fr.get("train_metrics", {}).get("pooled_ic", np.nan),
            "FR_train_rmse": fr.get("train_metrics", {}).get("rmse",      np.nan),
            "FR_test_ic":    fr.get("test_metrics",  {}).get("pooled_ic", np.nan),
            "FR_test_rmse":  fr.get("test_metrics",  {}).get("rmse",      np.nan),
        })

        def _accum(sel_counts, sel_pre, sel_post, agg_imp, agg_pre, agg_post,
                   agg_high, agg_low, country_res):
            for f in country_res.get("selected_feats", []):
                sel_counts[f] = sel_counts.get(f, 0) + 1
                if is_post: sel_post[f] = sel_post.get(f, 0) + 1
                else:        sel_pre[f]  = sel_pre.get(f,  0) + 1
            for k, v in (country_res.get("importance") or {}).items():
                v = float(v)
                agg_imp[k]  = agg_imp.get(k, 0.0)  + v
                agg_post[k] = agg_post.get(k, 0.0) + v * is_post
                agg_pre[k]  = agg_pre.get(k,  0.0) + v * (1 - is_post)
            for k, v in (country_res.get("importance_high_vol") or {}).items():
                agg_high[k] = agg_high.get(k, 0.0) + float(v)
            for k, v in (country_res.get("importance_low_vol") or {}).items():
                agg_low[k] = agg_low.get(k, 0.0) + float(v)

        _accum(sel_counts_de, sel_pre_de, sel_post_de,
               agg_imp_de, agg_imp_pre_de, agg_imp_post_de,
               agg_imp_high_de, agg_imp_low_de, de)
        _accum(sel_counts_fr, sel_pre_fr, sel_post_fr,
               agg_imp_fr, agg_imp_pre_fr, agg_imp_post_fr,
               agg_imp_high_fr, agg_imp_low_fr, fr)

    # Normalise
    def _norm(d, n): return {k: v / n for k, v in d.items()}
    mean_imp_de      = _norm(agg_imp_de,      n_folds)
    mean_imp_fr      = _norm(agg_imp_fr,      n_folds)
    mean_imp_pre_de  = _norm(agg_imp_pre_de,  max(n_pre,  1))
    mean_imp_post_de = _norm(agg_imp_post_de, max(n_post, 1))
    mean_imp_pre_fr  = _norm(agg_imp_pre_fr,  max(n_pre,  1))
    mean_imp_post_fr = _norm(agg_imp_post_fr, max(n_post, 1))
    mean_imp_high_de = _norm(agg_imp_high_de, n_folds)
    mean_imp_low_de  = _norm(agg_imp_low_de,  n_folds)
    mean_imp_high_fr = _norm(agg_imp_high_fr, n_folds)
    mean_imp_low_fr  = _norm(agg_imp_low_fr,  n_folds)

    # Frequencies
    all_feats_de = list(dict.fromkeys(candidate_de))
    all_feats_fr = list(dict.fromkeys(candidate_fr))

    freq_de      = {k: sel_counts_de.get(k, 0) / n_folds       for k in all_feats_de}
    freq_fr      = {k: sel_counts_fr.get(k, 0) / n_folds       for k in all_feats_fr}
    freq_pre_de  = {k: sel_pre_de.get(k, 0)  / max(n_pre, 1)   for k in all_feats_de}
    freq_post_de = {k: sel_post_de.get(k, 0) / max(n_post, 1)  for k in all_feats_de}
    freq_pre_fr  = {k: sel_pre_fr.get(k, 0)  / max(n_pre, 1)   for k in all_feats_fr}
    freq_post_fr = {k: sel_post_fr.get(k, 0) / max(n_post, 1)  for k in all_feats_fr}

    # Regime-aware stability: global OR per-regime threshold
    def _stable(f, freq, freq_pre, freq_post):
        return (freq.get(f, 0)      >= freq_threshold
                or freq_pre.get(f, 0)  >= freq_threshold_regime
                or freq_post.get(f, 0) >= freq_threshold_regime)

    stable_de = [f for f in all_feats_de if _stable(f, freq_de, freq_pre_de, freq_post_de)]
    stable_fr = [f for f in all_feats_fr if _stable(f, freq_fr, freq_pre_fr, freq_post_fr)]

    def _fallback(freq_map, imp_map, k):
        return [f for f, _ in sorted(
            freq_map.items(),
            key=lambda kv: (kv[1], imp_map.get(kv[0], 0.0)),
            reverse=True,
        )[:k]]

    if not stable_de: stable_de = _fallback(freq_de, mean_imp_de, max_features)
    if not stable_fr: stable_fr = _fallback(freq_fr, mean_imp_fr, max_features)

    stable_de = sorted(stable_de, key=lambda f: mean_imp_de.get(f, 0), reverse=True)[:max_features]
    stable_fr = sorted(stable_fr, key=lambda f: mean_imp_fr.get(f, 0), reverse=True)[:max_features]

    # Final union: selected + forced calendar (prefixed only) + extra forced
    selected_union = list(dict.fromkeys(stable_de + stable_fr + all_forced))

    # ---- Diagnostics ---------------------------------------------------------
    diagnostics = {
        "n_folds":           n_folds,
        "n_pre_folds":       n_pre,
        "n_post_folds":      n_post,
        "folds":             pd.DataFrame(fold_level_metrics),
        "freq_de":           freq_de,
        "freq_fr":           freq_fr,
        "freq_pre_de":       freq_pre_de,
        "freq_post_de":      freq_post_de,
        "freq_pre_fr":       freq_pre_fr,
        "freq_post_fr":      freq_post_fr,
        "mean_imp_de":       mean_imp_de,
        "mean_imp_fr":       mean_imp_fr,
        "mean_imp_pre_de":   mean_imp_pre_de,
        "mean_imp_post_de":  mean_imp_post_de,
        "mean_imp_pre_fr":   mean_imp_pre_fr,
        "mean_imp_post_fr":  mean_imp_post_fr,
        "mean_imp_high_de":  mean_imp_high_de,
        "mean_imp_low_de":   mean_imp_low_de,
        "mean_imp_high_fr":  mean_imp_high_fr,
        "mean_imp_low_fr":   mean_imp_low_fr,
        "stable_de":         stable_de,
        "stable_fr":         stable_fr,
        "calendar_forced":   calendar_forced,
        "selected_union":    selected_union,
        "raw_fold_results":  results,
        # legacy keys
        "mean_importance_de": mean_imp_de,
        "mean_importance_fr": mean_imp_fr,
    }

    # ---- Verbose output ------------------------------------------------------
    if verbose:
        folds_df = diagnostics["folds"]
        def _cm(col):
            return float(folds_df[col].dropna().mean()) if col in folds_df else np.nan

        raw = results
        avg_pf_de = np.mean([len(r["DE"].get("prefilter_feats", [])) for r in raw])
        avg_sl_de = np.mean([len(r["DE"].get("selected_feats",  [])) for r in raw])
        avg_pf_fr = np.mean([len(r["FR"].get("prefilter_feats", [])) for r in raw])
        avg_sl_fr = np.mean([len(r["FR"].get("selected_feats",  [])) for r in raw])

        print("\n=== FEATURE SELECTION DIAGNOSTICS ===")
        print(f"Folds: {n_folds}  (pre-2022: {n_pre}, post-2022: {n_post})")
        print(f"Train window: {train_window} days ({'rolling' if train_window else 'expanding'})")
        print(f"Freq threshold: global={freq_threshold}, per-regime={freq_threshold_regime}")
        print(f"Vol zscore regime split: {use_vol_zscore}")
        print(f"Avg prefilter kept: DE={avg_pf_de:.1f}, FR={avg_pf_fr:.1f}")
        print(f"Avg selected per fold: DE={avg_sl_de:.1f}, FR={avg_sl_fr:.1f}")
        print(f"Stable DE: {len(stable_de)}  Stable FR: {len(stable_fr)}")
        print(f"Calendar features forced (prefixed only): {len(calendar_forced)}")
        print(f"Union total: {len(selected_union)}")

        print("\n--- Fold-averaged metrics ---")
        print(f"DE: train IC={_cm('DE_train_ic'):.4f}  test IC={_cm('DE_test_ic'):.4f}  "
              f"gap={_cm('DE_train_ic') - _cm('DE_test_ic'):.4f}")
        print(f"FR: train IC={_cm('FR_train_ic'):.4f}  test IC={_cm('FR_test_ic'):.4f}  "
              f"gap={_cm('FR_train_ic') - _cm('FR_test_ic'):.4f}")

        print("\nStable DE features:", stable_de)
        print("Stable FR features:", stable_fr)
        print("Forced calendar (prefixed):", calendar_forced)
        print("Final union:", selected_union)

        def _top(d, n=10):
            return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

        print("\nTop importance DE (all folds):")
        for f, v in _top(mean_imp_de): print(f"  {f}: {v:.4f}")

        print("\nTop importance FR — post-2022:")
        for f, v in _top(mean_imp_post_fr): print(f"  {f}: {v:.4f}")

        print("\nTop importance FR — pre-2022:")
        for f, v in _top(mean_imp_pre_fr): print(f"  {f}: {v:.4f}")

        if use_vol_zscore:
            print("\nTop importance DE — high-vol regime:")
            for f, v in _top(mean_imp_high_de): print(f"  {f}: {v:.4f}")
            print("\nTop importance DE — low-vol regime:")
            for f, v in _top(mean_imp_low_de): print(f"  {f}: {v:.4f}")
            print("\nTop importance FR — high-vol regime:")
            for f, v in _top(mean_imp_high_fr): print(f"  {f}: {v:.4f}")
            print("\nTop importance FR — low-vol regime:")
            for f, v in _top(mean_imp_low_fr): print(f"  {f}: {v:.4f}")

        shared_selected = [f for f in selected_union if f in _SHARED_STEMS]
        if shared_selected:
            print("\nShared cross-country features selected:", shared_selected)

        # FR nuclear check
        nuc_feats = [f for f in selected_union if "nuclear" in f.lower()]
        print(f"\nFR nuclear in union: {nuc_feats if nuc_feats else 'NONE — check freq_post_fr'}")
        if not nuc_feats:
            nuc_candidates = {k: v for k, v in freq_post_fr.items() if "nuclear" in k.lower()}
            print(f"  Post-2022 freq for nuclear features: {nuc_candidates}")

    return selected_union, diagnostics
    

#################################################
#################################################
# MAIN XGBOOST FUNCTION
#################################################
#################################################

# =============================================================================
# detect_markov_regime — filtered probs
# =============================================================================

def detect_markov_regime(
    df: pd.DataFrame,
    residual_col: str,
    n_states: int = 2,
    scale: bool = True,
    random_state: int = 42,
) -> tuple:
    """
    Gaussian HMM regime detection using FILTERED (causal) probabilities.

    The original used model.predict_proba() which calls the forward-backward
    algorithm and returns SMOOTHED posteriors P(state_t | obs_1..obs_T).
    These use future observations — lookahead.

    This version replaces that with the FORWARD PASS ONLY, computing
    filtered posteriors P(state_t | obs_1..obs_t) via the alpha recursion.
    No future observation influences the probability at time t.

    Output contract: identical to the original.
        data_df        : DataFrame, same index as input (after dropna),
                         columns: regime, regime_prob_0, regime_prob_1, ...
        diagnostics    : dict with transition_matrix, means (original scale),
                         covariances, state_counts
    State alignment: state_1 is always the HIGH-volatility regime (larger mean).
    """
    data = df[[residual_col]].dropna().copy()
    X = data.values.astype(float)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X.copy()
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state,
    )"""

    model = GMMHMM(
        n_components=n_states,      # number of hidden states
        n_mix=2,                    # Gaussian mixture components per state
        covariance_type="full",
        n_iter=1000,
        random_state=random_state,
    )
    
    model.fit(X_fit)

    # ------------------------------------------------------------------
    # Forward pass (alpha recursion) — strictly causal
    # log-domain for numerical stability
    # ------------------------------------------------------------------
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))   # (S,)
    log_transmat  = np.log(np.clip(model.transmat_,  1e-300, None))   # (S, S)
    log_emission  = model._compute_log_likelihood(X_fit)               # (T, S)

    T, S = log_emission.shape
    log_alpha = np.full((T, S), -np.inf)
    log_alpha[0] = log_startprob + log_emission[0]

    for t in range(1, T):
        # log_alpha[t, s] = log_emission[t, s]
        #                   + logsumexp_over_prev(log_alpha[t-1] + log_transmat[:, s])
        log_alpha[t] = (
            np.logaddexp.reduce(
                log_alpha[t - 1, :, None] + log_transmat,   # (S, S)
                axis=0,                                       # sum over previous states
            )
            + log_emission[t]
        )

    # Normalise each row → P(state_t | obs_1..t)
    log_norm      = np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)  # (T, 1)
    filtered_probs = np.exp(log_alpha - log_norm)                           # (T, S)

    # Viterbi path for the discrete label (uses full sequence but only for
    # the hard assignment, not for the probability values)
    hidden_states = model.predict(X_fit).astype(int)

    # ------------------------------------------------------------------
    # State alignment: state index 1 = HIGH vol (larger mean)
    # ------------------------------------------------------------------
    if scaler is not None:
        means_orig = scaler.inverse_transform(
            model.means_.reshape(S, -1)
        ).flatten()
    else:
        means_orig = model.means_.flatten().copy()

    if n_states == 2 and float(means_orig[0]) > float(means_orig[1]):
        # state 0 is high-vol — swap so that state 1 = high-vol
        filtered_probs = filtered_probs[:, [1, 0]]
        hidden_states  = 1 - hidden_states
        means_orig     = means_orig[[1, 0]]

    # ------------------------------------------------------------------
    # Build output DataFrame — same column names as original
    # ------------------------------------------------------------------
    data["regime"] = hidden_states
    for i in range(n_states):
        data[f"regime_prob_{i}"] = filtered_probs[:, i]

    diagnostics = {
        "transition_matrix": model.transmat_,
        "means":             means_orig,
        "covariances":       model.covars_.flatten(),
        "state_counts":      np.bincount(hidden_states, minlength=n_states),
        "filtered":          True,   # confirms no lookahead
    }

    return data, diagnostics


# =============================================================================
# add_regime_probs_to_panel — fit once, attach to panel before CV
# =============================================================================

def add_regime_probs_to_panel(
    panel: pd.DataFrame,
    countries: list = None,
    rv_col_template: str = "{pref}_realized_vol",
    n_states: int = 2,
    random_state: int = 42,
) -> tuple:
    """
    Fit detect_markov_regime ONCE per country on the full panel,
    then attach filtered regime_prob columns to the panel.

    Has to be called before CV loop. 
    
    Returns
    -------
    panel_out    : DataFrame with added columns {pref}_regime_prob_0/1
    regime_diags : dict  {pref: diagnostics_dict}

    """
    if countries is None:
        countries = ['DE', 'FR']

    panel_out    = panel.copy().sort_values('DATE').reset_index(drop=True)
    regime_diags = {}

    for pref in countries:
        rv_col = rv_col_template.format(pref=pref)
        if rv_col not in panel_out.columns:
            print(f"[add_regime_probs] WARNING: {rv_col} not found — skipping {pref}")
            continue

        # log(realized_vol) — same transformation used in the original per-fold code
        rv_raw = panel_out.set_index('DATE')[rv_col].astype(float)
        rv_raw = rv_raw.where(rv_raw >= 1e-6)                          # NaN out zeros
        log_rv = np.log(rv_raw.clip(lower=1e-12)).dropna()

        log_rv_df = log_rv.to_frame(name=rv_col)
        data_df, diag = detect_markov_regime(
            df=log_rv_df,
            residual_col=rv_col,
            n_states=n_states,
            scale=True,
            random_state=random_state,
        )

        # Merge filtered probs back into panel by date
        data_df.index = pd.to_datetime(data_df.index)
        panel_dates   = pd.to_datetime(panel_out['DATE'])

        for i in range(n_states):
            col = f"{pref}_regime_prob_{i}"
            panel_out[col] = (
                data_df[f"regime_prob_{i}"]
                .reindex(panel_dates.values)
                .values
            )

        regime_diags[pref] = diag
        print(
            f"[{pref}] HMM fitted on full panel. "
            f"means={np.round(diag['means'], 3)}  "
            f"state_counts={diag['state_counts']}  "
            f"filtered=True (no lookahead)"
        )

    return panel_out, regime_diags


# =============================================================================
# Regime detection vol z-score (shared between CV and trading)
# =============================================================================

def compute_vol_regime_flag(
    panel: pd.DataFrame,
    rv_col_template: str = '{pref}_realized_vol',
    window: int = 252,          
    threshold: float = 0.5,     
    min_periods: int = 30,
    prefixes: list = None,
) -> pd.DataFrame:
    """
    Compute per-country vol regime flag from log(realized_vol) z-score.
    
    Consistent with the trading strategy regime detection.
    Both use: zscore = (log_rv - rolling_mean(window)) / rolling_std(window)
    Both use shift(1) for no-lookahead.
    
    Returns panel with added columns:
        {pref}_vol_regime_zscore  : raw z-score
        {pref}_vol_regime_flag    : 1 = high-vol, 0 = calm
    
    Pass this panel to two_regime_rolling_cv_per_country so the CV sees
    the same regime labels as the trading strategy.
    """
    if prefixes is None:
        prefixes = ['DE', 'FR']
    
    df = panel.copy()
    df = df.sort_values('DATE').reset_index(drop=True)
    
    for pref in prefixes:
        rv_col = rv_col_template.format(pref=pref)
        if rv_col not in df.columns:
            print(f"WARNING: {rv_col} not found — skipping {pref}")
            continue
        
        log_rv = np.log(df[rv_col].astype(float).clip(lower=1e-12))
        mu  = log_rv.rolling(window, min_periods=min_periods).mean().shift(1)
        sig = log_rv.rolling(window, min_periods=min_periods).std().shift(1).clip(lower=1e-8)
        zs  = (log_rv - mu) / sig
        
        df[f'{pref}_vol_regime_zscore'] = zs
        df[f'{pref}_vol_regime_flag']   = (zs > threshold).astype(int)
        
        pct_high = (df[f'{pref}_vol_regime_flag'] == 1).mean()
        print(f"{pref}: {pct_high:.1%} of days flagged as high-vol "
              f"(threshold={threshold}, window={window})")
    
    return df


# =============================================================================
# Regime-conditional CV
# Two targets: high-vol uses HAR-RV residual, calm uses log(rv) level
# =============================================================================

def two_regime_rolling_cv_per_country(
    df: pd.DataFrame,
    features: List[str],
    target_de: str = "DE_residual_target",
    target_fr: str = "FR_residual_target",
    # --- window ---
    train_window: int = 1200,
    min_train_days: int = 280,
    test_horizon: int = 21,
    gap_days: int = 0,
    # --- model ---
    xgb_params: Dict[str, Any] = None,
    n_jobs: int = -1,
    min_samples_per_regime: int = 50,
    # --- regime-conditional targets ---
    # When True, calm-period folds train on log(rv) level instead of HAR residual.
    # This directly targets the predictable component in calm markets.
    # The predictions are stored in separate columns: pred_hv / pred_calm.
    use_regime_targets: bool = True,
    vol_regime_window: int = 252,
    vol_regime_threshold: float = 0.5,
    calm_target_de: str = "DE_log_rv_target",   # log(rv_t+1) 
    calm_target_fr: str = "FR_log_rv_target",
    # --- Markov ---
    use_markov: Dict[str, bool] = None,
    k_regimes: int = 2,
    use_per_regime_models: bool = False,
    # --- Vol zscore ---
    use_vol_zscore: Dict[str, bool] = None,
    vol_zscore_window: int = 252,
    # --- Recency weighting ---
    use_recency_weights: bool = False,
    recency_halflife: int = 180,
    # --- Optuna ---
    use_optuna: bool = True,
    n_optuna_trials: int = 25,
    optuna_val_frac: float = 0.15,
    detect_kwargs: Dict[str, Any] = None,
) -> tuple:
    """
    Rolling walk-forward CV with optional regime-conditional targets.
    
    REGIME-CONDITIONAL TARGETS (use_regime_targets=True)
    
    High-vol folds: predict HAR-RV residual (original target)
        Edge: XGBoost corrects HAR-RV systematic errors during crises
    Calm folds: predict log(rv_t+1) directly
        Edge: generation mix / weather predict vol level in calm markets
              even when the HAR residual is near-zero noise
    
    The regime flag for each fold is computed from the training window only.
    A fold is "high-vol" if the test period's regime,
    estimated from training data, suggests high-vol.
    
    Output: all_preds contains pred_hv (HAR residual prediction) and
    pred_calm (log-rv prediction) as separate columns, plus pred which
    is the regime-appropriate prediction for that fold.

    SHARED (UNPREFIXED) FEATURES
        Features in `features` that do not start with 'DE_' or 'FR_' are
        treated as shared and added to BOTH country feature lists.
        Example: VOL_SPREAD, LOAD_IMBALANCE, VOL_RATIO.

    OPTUNA (use_optuna=True)
        Each fold runs a fast Optuna study on its own training window.
        The search space covers max_depth, eta, subsample, colsample_bytree,
        lambda, alpha, min_child_weight. The objective is Spearman IC on the
        inner val split (last optuna_val_frac of the training window).
        Best params are used to train the final model on the full training
        window. 
        Warning: adds ~n_optuna_trials * 2 XGBoost fits per fold.
        With n_jobs=-1 and 178 folds this can get comp. expensive.
        Set use_optuna=False to use xgb_params directly (faster, default).

    MARKOV
        Pre-computed filtered regime probabilities are read from columns
        {pref}_regime_prob_0 / _1 already in df. No HMM is fitted here.
        Call add_regime_probs_to_panel() once before this function.
        The CV fold simply reads the prob value for each training/test row;
        test rows are filled with the last training-period value (causal fill).
        If use_per_regime_models=True, two XGBoost models are trained per
        country, soft-weighted by the regime probability.   
    """

    if not _OPTUNA_AVAILABLE and use_optuna:
        raise ImportError("pip install optuna")

    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': 3, 'eta': 0.02, 'subsample': 0.6,
            'colsample_bytree': 0.6, 'lambda': 20, 'alpha': 5,
            'min_child_weight': 30, 'seed': SEED, 'nthread': 1, 'verbosity': 0,
        }
    if use_markov is None:
        use_markov = {'DE': False, 'FR': False}
    if use_vol_zscore is None:
        use_vol_zscore = {'DE': False, 'FR': False}

    df = df.sort_values('DATE').reset_index(drop=True)
    unique_dates = np.sort(df['DATE'].unique())

    features_de_base     = [f for f in features if f.startswith("DE_")]
    features_fr_base     = [f for f in features if f.startswith("FR_")]
    features_shared_base = [
        f for f in features
        if not f.startswith("DE_") and not f.startswith("FR_")
        and f in df.columns
    ]

    if not features_de_base or not features_fr_base:
        raise ValueError("features must contain both DE_ and FR_ prefixed columns.")
    if features_shared_base:
        print(f"[CV] Shared features: {features_shared_base}")

    for pref in ['DE', 'FR']:
        if use_markov.get(pref, False):
            for i in range(k_regimes):
                col = f"{pref}_regime_prob_{i}"
                if col not in df.columns:
                    raise ValueError(f"Call add_regime_probs_to_panel() first.")

    def safe_spearman(x, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 5: return 0.0
                xs, ys = x[mask], y[mask]
                if np.std(xs) < 1e-12 or np.std(ys) < 1e-12: return 0.0
                r = spearmanr(xs, ys).correlation
                return 0.0 if not np.isfinite(r) else float(r)
            except Exception:
                return 0.0

    def _recency_weights(n, halflife):
        decay = np.log(2) / halflife
        w = np.exp(-decay * np.arange(n - 1, -1, -1, dtype=float))
        return w / w.mean()

    def _optuna_search(X_tr_full, y_tr_full):
        valid = y_tr_full.notna() & np.isfinite(y_tr_full)
        X_c = X_tr_full.loc[valid]; y_c = y_tr_full.loc[valid].astype(float)
        val_size = max(int(optuna_val_frac * len(X_c)), 20)
        if len(X_c) <= val_size + 20:
            return xgb_params
        X_t, X_v = X_c.iloc[:-val_size], X_c.iloc[-val_size:]
        y_t, y_v = y_c.iloc[:-val_size], y_c.iloc[-val_size:]
        dtrain_o = xgb.DMatrix(X_t, label=y_t, missing=np.nan)
        dval_o   = xgb.DMatrix(X_v, label=y_v, missing=np.nan)
        y_v_arr  = y_v.values

        def objective(trial):
            params = {
                'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                'seed': SEED, 'nthread': 1, 'verbosity': 0,
                'max_depth':        trial.suggest_int('max_depth', 2, 5),
                'eta':              trial.suggest_float('eta', 0.005, 0.05, log=True),
                'subsample':        trial.suggest_float('subsample', 0.4, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
                'lambda':           trial.suggest_float('lambda', 5.0, 100.0, log=True),
                'alpha':            trial.suggest_float('alpha', 0.5, 20.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 10, 60),
            }
            try:
                bst = xgb.train(params, dtrain_o, num_boost_round=1500,
                                evals=[(dval_o, 'val')], early_stopping_rounds=40,
                                verbose_eval=False)
                preds_v = bst.predict(dval_o)
                ic = safe_spearman(preds_v, y_v_arr)
                return -ic if np.isfinite(ic) else 0.0
            except Exception:
                return 0.0

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
        best = study.best_params
        return {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                'seed': SEED, 'nthread': 1, 'verbosity': 0, **best}

    def train_xgb_safe(X, y, sample_weight=None, params_override=None):
        if X.shape[0] == 0: return None
        y_ser = pd.Series(y.values, index=X.index)
        valid = y_ser.notna() & np.isfinite(y_ser)
        if valid.sum() < min_samples_per_regime: return None
        X_c = X.loc[valid]; y_c = y_ser.loc[valid].astype(float)
        w_c = sample_weight[valid.values] if sample_weight is not None else None
        val_size = max(int(0.15 * len(X_c)), 20)
        if len(X_c) <= val_size + 10: return None
        X_tr, X_val = X_c.iloc[:-val_size], X_c.iloc[-val_size:]
        y_tr, y_val = y_c.iloc[:-val_size], y_c.iloc[-val_size:]
        w_tr = w_c[:-val_size] if w_c is not None else None
        p = params_override if params_override is not None else xgb_params
        dtrain = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, weight=w_tr)
        dval   = xgb.DMatrix(X_val, label=y_val, missing=np.nan)
        try:
            m = xgb.train(p, dtrain, num_boost_round=2000,
                          evals=[(dval, 'val')], early_stopping_rounds=50,
                          verbose_eval=False)
            return {'model': m, 'train_pred': m.predict(dtrain),
                    'val_pred': m.predict(dval),
                    'y_tr': y_tr.values, 'y_val': y_val.values}
        except Exception:
            return None

    def run_fold(i):
        train_start = max(0, i - train_window) if train_window is not None else 0
        train_dates = unique_dates[train_start:i]
        test_dates  = unique_dates[i + gap_days: i + gap_days + test_horizon]
        if len(train_dates) < min_train_days or len(test_dates) == 0:
            return None

        train_mask = df['DATE'].isin(train_dates)
        test_mask  = df['DATE'].isin(test_dates)

        features_de = features_de_base + features_shared_base
        features_fr = features_fr_base + features_shared_base
        fold_df = df.copy()

        # ---- Regime flag for this fold (from training window only) ----
        # Per-country: flag = 1 if most recent training zscore > threshold
        fold_regime = {}
        if use_regime_targets:
            for pref in ['DE', 'FR']:
                rv_col = f'{pref}_realized_vol'
                if rv_col not in fold_df.columns:
                    fold_regime[pref] = 1  # default to high-vol
                    continue
                rv_tr = fold_df.loc[train_mask, rv_col].astype(float).clip(lower=1e-12)
                log_rv = np.log(rv_tr)
                mu  = log_rv.rolling(vol_regime_window, min_periods=30).mean()
                sig = log_rv.rolling(vol_regime_window, min_periods=30).std().clip(lower=1e-8)
                zs  = (log_rv - mu) / sig
                # Use last training-window zscore to determine test period regime
                last_zs = zs.dropna().iloc[-1] if zs.dropna().size else 0.0
                fold_regime[pref] = int(last_zs > vol_regime_threshold)

        # ---- Markov ----
        for pref, feats in [('DE', features_de), ('FR', features_fr)]:
            if not use_markov.get(pref, False): continue
            last_train_iloc = fold_df.loc[train_mask].index[-1]
            for i_state in range(k_regimes):
                col = f"{pref}_regime_prob_{i_state}"
                last_val = fold_df.loc[last_train_iloc, col]
                fill_val = float(last_val) if np.isfinite(last_val) else 0.5
                fold_df.loc[test_mask, col] = fill_val
                if col not in feats: feats.append(col)

        # ---- Vol zscore ----
        for pref, feats in [('DE', features_de), ('FR', features_fr)]:
            if not use_vol_zscore.get(pref, False): continue
            rv_col = f"{pref}_realized_vol"; zs_col = f"{pref}_vol_zscore"
            if rv_col not in fold_df.columns: continue
            rv_tr = fold_df.loc[train_mask, rv_col].astype(float).clip(lower=1e-12)
            log_rv_tr = np.log(rv_tr)
            mu  = log_rv_tr.rolling(vol_zscore_window, min_periods=30).mean()
            sig = log_rv_tr.rolling(vol_zscore_window, min_periods=30).std().clip(lower=1e-8)
            last_mu  = float(mu.dropna().iloc[-1])  if mu.dropna().size  else 0.0
            last_sig = float(sig.dropna().iloc[-1]) if sig.dropna().size else 1.0
            fold_df.loc[train_mask, zs_col] = ((log_rv_tr - mu) / sig).values
            rv_te = fold_df.loc[test_mask, rv_col].astype(float).clip(lower=1e-12)
            fold_df.loc[test_mask, zs_col] = (np.log(rv_te) - last_mu) / last_sig
            if zs_col not in feats: feats.append(zs_col)

        # ---- Select targets based on regime ----
        def _get_target(pref, default_target, calm_target):
            if not use_regime_targets:
                return default_target
            regime = fold_regime.get(pref, 1)
            if regime == 1:
                return default_target   # high-vol: predict HAR residual
            else:
                return calm_target      # calm: predict log(rv) level

        tgt_de = _get_target('DE', target_de, calm_target_de)
        tgt_fr = _get_target('FR', target_fr, calm_target_fr)

        # Use default target if calm target not available
        if tgt_de not in fold_df.columns: tgt_de = target_de
        if tgt_fr not in fold_df.columns: tgt_fr = target_fr

        def _Xy(feats, tgt, mask):
            X = fold_df.loc[mask, feats].copy()
            y = (fold_df.loc[mask, tgt].copy() if tgt in fold_df.columns
                 else pd.Series(np.nan, index=X.index))
            return X, y

        X_tr_de, y_tr_de = _Xy(features_de, tgt_de, train_mask)
        X_te_de, y_te_de = _Xy(features_de, tgt_de, test_mask)
        X_tr_fr, y_tr_fr = _Xy(features_fr, tgt_fr, train_mask)
        X_te_fr, y_te_fr = _Xy(features_fr, tgt_fr, test_mask)

        params_de = xgb_params; params_fr = xgb_params
        if use_optuna:
            params_de = _optuna_search(X_tr_de, y_tr_de)
            params_fr = _optuna_search(X_tr_fr, y_tr_fr)

        def _train_predict(X_tr, y_tr, X_te, pref, params):
            n = len(X_tr)
            w_base = _recency_weights(n, recency_halflife) if use_recency_weights else None
            prob_col_high = f"{pref}_regime_prob_1"
            has_regime = (use_per_regime_models and use_markov.get(pref, False)
                          and prob_col_high in fold_df.columns)
            if not has_regime:
                res = train_xgb_safe(X_tr, y_tr, sample_weight=w_base, params_override=params)
                if res is None: return np.full(len(X_te), np.nan), res
                return res['model'].predict(xgb.DMatrix(X_te, missing=np.nan)), res
            p_high = fold_df.loc[train_mask, prob_col_high].fillna(0.5).values[:n]
            p_low  = 1.0 - p_high
            base   = w_base if w_base is not None else np.ones(n)
            w_high = base * p_high; w_high /= (w_high.mean() + 1e-12)
            w_low  = base * p_low;  w_low  /= (w_low.mean()  + 1e-12)
            res_high = train_xgb_safe(X_tr, y_tr, sample_weight=w_high, params_override=params)
            res_low  = train_xgb_safe(X_tr, y_tr, sample_weight=w_low,  params_override=params)
            p_high_test = fold_df.loc[test_mask, prob_col_high].fillna(0.5).values
            dtest = xgb.DMatrix(X_te, missing=np.nan)
            ph = res_high['model'].predict(dtest) if res_high else np.zeros(len(X_te))
            pl = res_low['model'].predict(dtest)  if res_low  else np.zeros(len(X_te))
            return p_high_test * ph + (1 - p_high_test) * pl, res_high

        preds_de, res_de = _train_predict(X_tr_de, y_tr_de, X_te_de, 'DE', params_de)
        preds_fr, res_fr = _train_predict(X_tr_fr, y_tr_fr, X_te_fr, 'FR', params_fr)

        def _diag(res):
            if res is None:
                return {'train_rmse_exp': np.nan, 'val_rmse_exp': np.nan,
                        'train_ic_exp': np.nan, 'val_ic_exp': np.nan, 'fi_exp': {}}
            tp, vp, ty, vy = res['train_pred'], res['val_pred'], res['y_tr'], res['y_val']
            return {'train_rmse_exp': float(np.sqrt(np.mean((tp-ty)**2))) if tp.size else np.nan,
                    'val_rmse_exp':   float(np.sqrt(np.mean((vp-vy)**2))) if vp.size else np.nan,
                    'train_ic_exp':   safe_spearman(tp, ty),
                    'val_ic_exp':     safe_spearman(vp, vy),
                    'fi_exp':         res['model'].get_score(importance_type='gain')}

        diag_de = _diag(res_de); diag_fr = _diag(res_fr)

        rows = []
        for j, idx in enumerate(X_te_de.index):
            rows.append({'DATE': fold_df.loc[idx, 'DATE'], 'COUNTRY': 'DE',
                         'pred': float(preds_de[j]) if np.isfinite(preds_de[j]) else np.nan,
                         'true': float(y_te_de.loc[idx]) if np.isfinite(y_te_de.loc[idx]) else np.nan,
                         'fold_regime_de': fold_regime.get('DE', 1),
                         'target_used_de': tgt_de})
        for j, idx in enumerate(X_te_fr.index):
            rows.append({'DATE': fold_df.loc[idx, 'DATE'], 'COUNTRY': 'FR',
                         'pred': float(preds_fr[j]) if np.isfinite(preds_fr[j]) else np.nan,
                         'true': float(y_te_fr.loc[idx]) if np.isfinite(y_te_fr.loc[idx]) else np.nan,
                         'fold_regime_fr': fold_regime.get('FR', 1),
                         'target_used_fr': tgt_fr})
        pred_df = pd.DataFrame(rows)

        def safe_metrics(p, t):
            p, t = np.asarray(p, float), np.asarray(t, float)
            m = np.isfinite(p) & np.isfinite(t)
            if not m.any(): return np.nan, np.nan, np.nan
            return (safe_spearman(p[m], t[m]),
                    float(np.sqrt(np.mean((p[m]-t[m])**2))),
                    float(np.mean(np.abs(p[m]-t[m]))))

        ic_de=rmse_de=mae_de=ic_fr=rmse_fr=mae_fr=np.nan; n_test_de=n_test_fr=0
        if not pred_df.empty:
            for c, is_de in [('DE', True), ('FR', False)]:
                sub = pred_df[pred_df['COUNTRY']==c]
                if len(sub):
                    iv, rv2, mv = safe_metrics(sub['pred'], sub['true'])
                    if is_de:
                        ic_de=iv; rmse_de=rv2; mae_de=mv; n_test_de=int(sub['true'].notna().sum())
                    else:
                        ic_fr=iv; rmse_fr=rv2; mae_fr=mv; n_test_fr=int(sub['true'].notna().sum())

        pm = pred_df['pred'].notna() & pred_df['true'].notna()
        pooled_ic   = safe_spearman(pred_df.loc[pm,'pred'], pred_df.loc[pm,'true'])
        pooled_rmse = float(np.sqrt(np.mean((pred_df.loc[pm,'pred']-pred_df.loc[pm,'true'])**2))) if pm.any() else np.nan
        pooled_mae  = float(np.mean(np.abs(pred_df.loc[pm,'pred']-pred_df.loc[pm,'true']))) if pm.any() else np.nan

        fold_stat = {
            'train_until': train_dates[-1], 'test_from': test_dates[0],
            'test_to': test_dates[-1], 'n_train': int(train_mask.sum()),
            'n_test_de': n_test_de, 'n_test_fr': n_test_fr,
            'spearman_ic_de': float(ic_de) if np.isfinite(ic_de) else np.nan,
            'rmse_de': float(rmse_de) if np.isfinite(rmse_de) else np.nan,
            'mae_de':  float(mae_de)  if np.isfinite(mae_de)  else np.nan,
            'spearman_ic_fr': float(ic_fr) if np.isfinite(ic_fr) else np.nan,
            'rmse_fr': float(rmse_fr) if np.isfinite(rmse_fr) else np.nan,
            'mae_fr':  float(mae_fr)  if np.isfinite(mae_fr)  else np.nan,
            'pooled_ic': pooled_ic, 'pooled_rmse': pooled_rmse, 'pooled_mae': pooled_mae,
            'train_rmse_pre_exp':    float(diag_de.get('train_rmse_exp', np.nan)),
            'val_rmse_pre_exp':      float(diag_de.get('val_rmse_exp',   np.nan)),
            'train_ic_pre_exp':      float(diag_de.get('train_ic_exp',   np.nan)),
            'val_ic_pre_exp':        float(diag_de.get('val_ic_exp',     np.nan)),
            'train_rmse_post_exp':   np.nan, 'val_rmse_post_exp': np.nan,
            'train_ic_post_exp':     np.nan, 'val_ic_post_exp':   np.nan,
            'n_pre_train': np.nan, 'n_post_train': np.nan,
            'train_rmse_pre_exp_de': float(diag_de.get('train_rmse_exp', np.nan)),
            'val_rmse_pre_exp_de':   float(diag_de.get('val_rmse_exp',   np.nan)),
            'train_ic_pre_exp_de':   float(diag_de.get('train_ic_exp',   np.nan)),
            'val_ic_pre_exp_de':     float(diag_de.get('val_ic_exp',     np.nan)),
            'train_rmse_pre_exp_fr': float(diag_fr.get('train_rmse_exp', np.nan)),
            'val_rmse_pre_exp_fr':   float(diag_fr.get('val_rmse_exp',   np.nan)),
            'train_ic_pre_exp_fr':   float(diag_fr.get('train_ic_exp',   np.nan)),
            'val_ic_pre_exp_fr':     float(diag_fr.get('val_ic_exp',     np.nan)),
            'markov_de': {'fitted': use_markov.get('DE', False), 'source': 'pre_computed'},
            'markov_fr': {'fitted': use_markov.get('FR', False), 'source': 'pre_computed'},
            'fi_exp_de': diag_de.get('fi_exp', {}), 'fi_exp_fr': diag_fr.get('fi_exp', {}),
            'train_window_used': len(train_dates),
            'fold_regime_de': fold_regime.get('DE', 1),
            'fold_regime_fr': fold_regime.get('FR', 1),
            'target_used_de': tgt_de, 'target_used_fr': tgt_fr,
            'optuna_params_de': params_de if use_optuna else {},
            'optuna_params_fr': params_fr if use_optuna else {},
        }

        gc.collect()
        return pred_df, fold_stat

    starts = list(range(min_train_days,
                        len(unique_dates) - gap_days - test_horizon + 1,
                        test_horizon))
    with parallel_backend("loky"):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_fold)(i) for i in starts)
    results = [r for r in results if r is not None]
    if not results: raise ValueError("No folds produced.")

    all_preds  = pd.concat([r[0] for r in results], ignore_index=True)
    fold_stats = pd.DataFrame([r[1] for r in results])

    pm_all = all_preds['pred'].notna() & all_preds['true'].notna()
    pooled_ic   = safe_spearman(all_preds.loc[pm_all,'pred'], all_preds.loc[pm_all,'true'])
    pooled_rmse = float(np.sqrt(np.mean((all_preds.loc[pm_all,'pred']-all_preds.loc[pm_all,'true'])**2))) if pm_all.any() else np.nan
    pooled_mae  = float(np.mean(np.abs(all_preds.loc[pm_all,'pred']-all_preds.loc[pm_all,'true']))) if pm_all.any() else np.nan

    per_country_ic={}; per_country_rmse={}; per_country_mae={}
    for c in ['DE','FR']:
        sub = all_preds[all_preds['COUNTRY']==c]
        mc  = sub['pred'].notna() & sub['true'].notna()
        if mc.any():
            per_country_ic[c]   = safe_spearman(sub.loc[mc,'pred'], sub.loc[mc,'true'])
            per_country_rmse[c] = float(np.sqrt(np.mean((sub.loc[mc,'pred']-sub.loc[mc,'true'])**2)))
            per_country_mae[c]  = float(np.mean(np.abs(sub.loc[mc,'pred']-sub.loc[mc,'true'])))
        else:
            per_country_ic[c]=per_country_rmse[c]=per_country_mae[c]=np.nan

    rolling_ic={}; rolling_rmse={}; rolling_mae={}; rolling_pred_mean={}; rolling_pred_std={}
    for c in ['DE','FR']:
        sub = all_preds[all_preds['COUNTRY']==c].set_index('DATE').sort_index()
        if not len(sub):
            for d in [rolling_ic,rolling_rmse,rolling_mae,rolling_pred_mean,rolling_pred_std]:
                d[c]=pd.Series(dtype=float)
            continue
        rp=sub['pred'].rank(); rt=sub['true'].rank()
        rolling_ic[c]        = rp.rolling(126,min_periods=10).corr(rt)
        rolling_rmse[c]      = ((sub['pred']-sub['true'])**2).rolling(126,min_periods=10).mean().apply(np.sqrt)
        rolling_mae[c]       = (sub['pred']-sub['true']).abs().rolling(126,min_periods=10).mean()
        rolling_pred_mean[c] = sub['pred'].rolling(126,min_periods=10).mean()
        rolling_pred_std[c]  = sub['pred'].rolling(126,min_periods=10).std()

    ps = all_preds.set_index('DATE').sort_index()
    rolling_pooled_ic = ps['pred'].rank().rolling(126,min_periods=10).corr(ps['true'].rank())
    pooled_pred_mean  = ps['pred'].rolling(126,min_periods=10).mean()
    pooled_pred_std   = ps['pred'].rolling(126,min_periods=10).std()

    overall = {
        'mean_fold_ic_de':     float(fold_stats['spearman_ic_de'].dropna().mean()),
        'mean_fold_ic_fr':     float(fold_stats['spearman_ic_fr'].dropna().mean()),
        'mean_fold_pooled_ic': float(fold_stats['pooled_ic'].dropna().mean()),
        'pooled_ic': pooled_ic, 'pooled_rmse': pooled_rmse, 'pooled_mae': pooled_mae,
        'per_country_ic': per_country_ic, 'per_country_rmse': per_country_rmse,
        'per_country_mae': per_country_mae,
        'mean_rmse_fold': float(fold_stats[['rmse_de','rmse_fr']].stack().dropna().mean()),
        'mean_mae_fold':  float(fold_stats[['mae_de','mae_fr']].stack().dropna().mean()),
        'n_folds': len(fold_stats),
        'n_regime_detected_folds_de': int(fold_stats['markov_de'].apply(
            lambda x: bool(x.get('fitted')) if isinstance(x,dict) else False).sum()),
        'n_regime_detected_folds_fr': int(fold_stats['markov_fr'].apply(
            lambda x: bool(x.get('fitted')) if isinstance(x,dict) else False).sum()),
    }

    if use_regime_targets:
        n_hv_de = int(fold_stats['fold_regime_de'].sum())
        n_hv_fr = int(fold_stats['fold_regime_fr'].sum())
        n_tot   = len(fold_stats)
        print(f"Regime-conditional targets: "
              f"DE high-vol folds={n_hv_de}/{n_tot}, "
              f"FR high-vol folds={n_hv_fr}/{n_tot}")

    print(f"=== SUMMARY ===")
    print(f"Pooled IC: {pooled_ic:.4f}")
    print(f"Per-country IC: {per_country_ic}")
    print(f"Mean fold ICs — DE: {overall['mean_fold_ic_de']:.4f}  FR: {overall['mean_fold_ic_fr']:.4f}")

    return all_preds, {
        'folds': fold_stats, 'overall': overall,
        'rolling_ic': rolling_ic, 'rolling_pooled_ic': rolling_pooled_ic,
        'rolling_rmse': rolling_rmse, 'rolling_mae': rolling_mae,
        'rolling_pred_mean': rolling_pred_mean, 'rolling_pred_std': rolling_pred_std,
        'pooled_pred_mean': pooled_pred_mean, 'pooled_pred_std': pooled_pred_std,
    }



    
#----------------------------------------------------------
# Optuna XGBoost hyperparam optimizer
#-----------------------------------------------------------

def optimise_xgb_params_optuna(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_cutoff: str,
    val_cutoff: str,
    n_trials: int = 80,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Bayesian XGBoost hyperparameter search. Optimises Spearman IC."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    df = df.sort_values('DATE').reset_index(drop=True)
    feat_cols = [f for f in features if f in df.columns]

    tr_mask  = df['DATE'] <  pd.Timestamp(train_cutoff)
    val_mask = (df['DATE'] >= pd.Timestamp(train_cutoff)) & \
               (df['DATE'] <  pd.Timestamp(val_cutoff))

    y_tr  = df.loc[tr_mask,  target]
    y_val = df.loc[val_mask, target]
    ok_tr  = y_tr.notna()  & np.isfinite(y_tr)
    ok_val = y_val.notna() & np.isfinite(y_val)

    dtrain = xgb.DMatrix(df.loc[tr_mask,  feat_cols].loc[ok_tr],
                         label=y_tr.loc[ok_tr], missing=np.nan)
    dval   = xgb.DMatrix(df.loc[val_mask, feat_cols].loc[ok_val],
                         label=y_val.loc[ok_val], missing=np.nan)

    def objective(trial):
        p = {
            'objective':        'reg:squarederror',
            'eval_metric':      'rmse',
            'seed':             seed, 'nthread': 1, 'verbosity': 0,
            'max_depth':        trial.suggest_int('max_depth', 2, 5),
            'eta':              trial.suggest_float('eta', 0.005, 0.1, log=True),
            'subsample':        trial.suggest_float('subsample', 0.4, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'lambda':           trial.suggest_float('lambda', 1.0, 500.0, log=True),
            'alpha':            trial.suggest_float('alpha', 0.1, 50.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
        }
        m = xgb.train(p, dtrain, num_boost_round=1000,
                      evals=[(dval,'val')], early_stopping_rounds=40,
                      verbose_eval=False)
        preds  = m.predict(dval)
        y_true = y_val.loc[ok_val].values
        mask   = np.isfinite(preds) & np.isfinite(y_true)
        if np.std(preds) < 0.01:
            return 0.0
        if mask.sum() < 5:
            return 0.0
        ic = float(spearmanr(preds[mask], y_true[mask]).correlation)
        return ic if np.isfinite(ic) else 0.0


    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params.copy()
    best.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                 'seed': seed, 'nthread': 1, 'verbosity': 0})
    print(f"Best IC: {study.best_value:.4f}  |  Params: {best}")
    return best


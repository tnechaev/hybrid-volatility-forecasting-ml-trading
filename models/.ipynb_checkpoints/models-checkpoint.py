import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from arch import arch_model
import xgboost as xgb
from tqdm import tqdm
import time
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed
from typing import Sequence, List, Dict, Any, Tuple, Optional
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import gc
import optuna

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
    min_obs: int = 22,          # minimum history before first HAR forecast
    window: object = 150,       # rolling OLS window; int or dict {"DE": 500, "FR": 150}
    n_jobs: int = 4
) -> tuple:
    """
    Fit rolling HAR-RV models per country in parallel and store
    one-step-ahead forecasts in the panel.

    HAR-RV model (Corsi 2009):
        RV_{t+1} = beta0
                 + beta1 * RV_d_t          (daily:   last observation)
                 + beta2 * RV_w_t          (weekly:  mean of last 5 obs)
                 + beta3 * RV_m_t          (monthly: mean of last 22 obs)
                 + epsilon

    Rolling OLS is estimated on a window of observations.

    OUTPUT COLUMNS (identical names to add_garch_parallel):
        {pref}_garch_sigma  — HAR-RV forecast of RV at t+1
                              (column intentionally keeps the garch_sigma name
                               so the rest of the code works no matter the function used)
    """

    df = panel.copy()

    # Allow per-country window
    if isinstance(window, dict):
        window_map = window
    else:
        window_map = {"DE": window, "FR": window}

    def fit_har_for_country(pref: str):
        rv_col = f"{pref}_realized_vol"
        sigma_col = f"{pref}_garch_sigma"   # <-- intentional: keep existing name
        realized_col = rv_col

        country_window = window_map.get(pref, window if isinstance(window, int) else 150)

        rv = df[rv_col].values.astype(float)
        n = len(rv)
        forecasts = np.full(n, np.nan)

        for i in range(n):
            # Forecast RV at position i using only data up to i-1
            # (the forecast is "what will RV be at t=i, given info through t=i-1")
            if i < max(min_obs, 22):          # need at least 22 obs for monthly component
                continue

            # history: positions 0 … i-1
            start = max(0, i - country_window)
            hist = rv[start:i]                # shape (k,)
            hist_full = rv[max(0, i - 22):i]  # up to 22 obs for monthly component

            # build HAR regressors on historical window for fitting
            # each row j in [start+22 : i] has:
            #   y_j     = rv[j]          (target in-sample)
            #   x1_j    = rv[j-1]        (daily)
            #   x2_j    = mean(rv[j-5:j])  (weekly)
            #   x3_j    = mean(rv[j-22:j]) (monthly)
            fit_start = start + 22            # need 22 lags to build monthly
            fit_end = i                       # exclusive: fit on rv[fit_start..i-1]

            if fit_end - fit_start < 10:      # too few obs → fallback to rolling mean
                forecasts[i] = np.nanmean(hist)
                continue

            # build design matrix for OLS fit
            rows_y = []
            rows_X = []
            for j in range(fit_start, fit_end):
                y_j = rv[j]
                if not np.isfinite(y_j):
                    continue
                x1 = rv[j - 1]
                x2_slice = rv[j - 5:j]
                x3_slice = rv[j - 22:j]
                if not np.isfinite(x1):
                    continue
                x2 = np.nanmean(x2_slice) if len(x2_slice) > 0 else np.nan
                x3 = np.nanmean(x3_slice) if len(x3_slice) > 0 else np.nan
                if not (np.isfinite(x2) and np.isfinite(x3)):
                    continue
                rows_y.append(y_j)
                rows_X.append([1.0, x1, x2, x3])

            if len(rows_y) < 5:               # degenerate — fallback
                forecasts[i] = np.nanmean(hist)
                continue

            Y = np.array(rows_y)
            X = np.array(rows_X)

            try:
                # OLS via normal equations (fast for small matrices)
                beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            except Exception:
                forecasts[i] = np.nanmean(hist)
                continue

            # forecast for position i using rv at i-1 (the last known value)
            x1_f = rv[i - 1]
            x2_f = np.nanmean(rv[max(0, i - 5):i])
            x3_f = np.nanmean(rv[max(0, i - 22):i])

            if not (np.isfinite(x1_f) and np.isfinite(x2_f) and np.isfinite(x3_f)):
                forecasts[i] = np.nanmean(hist)
                continue

            pred = beta[0] + beta[1] * x1_f + beta[2] * x2_f + beta[3] * x3_f
            # clamp: RV forecasts should be non-negative
            forecasts[i] = max(pred, 1e-12)

        # Compute metrics vs realized volatility (identical to add_garch_parallel)
        mask = ~np.isnan(forecasts) & ~np.isnan(df[realized_col])
        if mask.sum() > 0:
            y_pred = forecasts[mask]
            y_true = df.loc[mask, realized_col].values
            ic   = spearmanr(y_pred, y_true).correlation
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mae  = float(np.mean(np.abs(y_pred - y_true)))
        else:
            ic = rmse = mae = np.nan

        return pref, forecasts, {"spearman_ic": ic, "rmse": rmse, "mae": mae}

    # Parallel computation — identical to add_garch_parallel function
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_har_for_country)(pref) for pref in ["DE", "FR"]
    )

    for pref, forecasts, m in results:
        df[f"{pref}_garch_sigma"] = forecasts    # same column name as GARCH version
        metrics = {}                             

    # rebuild metrics dict in one pass (avoids scoping issue)
    metrics = {pref: m for pref, _, m in results}

    return df, metrics


#---------------------------------------
# Feature selector -- obsolete, logic needs updating, does not work properly for FR
# ----------------------------------------

def feature_selection(
    df: pd.DataFrame,
    features: List[str] = None,
    target_de: str = "DE_residual_target",
    target_fr: str = "FR_residual_target",
    min_train_days: int = 280,
    test_horizon: int = 21,
    gap_days: int = 1,
    prefilter_top_k: int = 15,
    max_features: int = 10,
    xgb_params_prefilter: Dict[str, Any] = None,
    n_jobs: int = -1,
    freq_threshold: float = 0.7,   # keep features appearing in >= fraction of folds
    verbose: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Expanding-window stability selection for prefixed DE_/FR_ features.
    Returns (selected_union_prefixed, diagnostics_dict).
    diagnostics contains per-fold train/test metrics, selection counts, mean importances.

    NOTE: If `features` is None, features are autodetected from input dataframe by excluding
    a small set of leak / meta columns (see exclude_prefixes).
    """
    # default xgb params (regularized)
    if xgb_params_prefilter is None:
        xgb_params_prefilter = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 3,
            "eta": 0.03,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
            "lambda": 300,
            "alpha": 10,
            "min_child_weight": 30,
            "seed": SEED,
            "nthread": 1,
            "verbosity": 0
        }

    # ----------------------------
    # FEATURE AUTODETECTION (only change)
    # ----------------------------
    # If user passed features explicitly, respect it. Otherwise build from df columns
    # by excluding likely leakage / meta columns.
    if features is None:
        exclude_prefixes = [
            # general meta
            "DATE",
            # targets
            target_de, target_fr,
            "DE_garch_sigma", "FR_garch_sigma",
            #"DE_realized_vol", "FR_realized_vol",
            # country flag etc
            "COUNTRY", "COUNTRY_FLAG"
        ]

        # Candidate feature columns: any column not starting with any exclude token
        candidate_cols = []
        for col in df.columns:
            skip = False
            for ex in exclude_prefixes:
                # match exact column names or prefix matches where useful
                if col == ex or col.startswith(ex):
                    skip = True
                    break
            if not skip:
                candidate_cols.append(col)

        # keep only prefixed DE_/FR_ columns
        features_de = [c for c in candidate_cols if c.startswith("DE_")]
        features_fr = [c for c in candidate_cols if c.startswith("FR_")]
        features = features_de + features_fr
    else:
        features_de = [f for f in features if f.startswith("DE_")]
        features_fr = [f for f in features if f.startswith("FR_")]

    # prepare
    df = df.sort_values("DATE").reset_index(drop=True)
    unique_dates = np.sort(df["DATE"].unique())
    starts = list(range(min_train_days, len(unique_dates) - gap_days - test_horizon + 1, test_horizon))
    if len(starts) == 0:
        raise ValueError("Date range too small or min_train_days/test_horizon settings incompatible.")

    # helpers
    def safe_spearman(x_arr, y_arr):
        try:
            if len(x_arr)==0 or len(y_arr)==0: return 0.0
            if np.nanstd(x_arr) < 1e-12 or np.nanstd(y_arr) < 1e-12: return 0.0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                r = spearmanr(x_arr, y_arr).correlation
            return 0.0 if np.isnan(r) else float(r)
        except Exception:
            return 0.0

    def evaluate_preds(y_true, preds):
        mask = (~np.isnan(y_true)) & (~np.isnan(preds))
        if mask.sum() == 0:
            return {"n":0, "pooled_ic":np.nan, "rmse":np.nan, "mae":np.nan}
        y = np.array(y_true)[mask]; p = np.array(preds)[mask]
        return {"n": len(y), "pooled_ic": safe_spearman(p, y), "rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p))}

    # fold worker (per fold do both countries)
    def run_fold(i):
        train_dates = unique_dates[:i]
        test_dates = unique_dates[i + gap_days : i + gap_days + test_horizon]
        if len(test_dates) == 0:
            return None
        train_mask = df["DATE"].isin(train_dates)
        test_mask = df["DATE"].isin(test_dates)

        out = {
            "fold_start_idx": i,
            "train_from": train_dates[0],
            "train_until": train_dates[-1],
            "test_from": test_dates[0],
            "test_to": test_dates[-1],
            "DE": {},
            "FR": {}
        }

        # per-country routine
        def _country_select(prefix_feats, target_col):
            res = {
                "prefilter_scores": [],
                "prefilter_feats": [],
                "selected_feats": [],
                "importance": {},
                "train_metrics": None,
                "test_metrics": None
            }
            if target_col not in df.columns:
                return res

            # compute pooled univariate ic on train only
            scores = []
            train_df = df.loc[train_mask]
            for feat in prefix_feats:
                x = train_df[feat].values if feat in train_df.columns else np.array([])
                y = train_df[target_col].values
                mask = np.isfinite(x) & np.isfinite(y)
                ic = safe_spearman(x[mask], y[mask]) if mask.sum()>0 else 0.0
                scores.append((feat, ic))
            scores.sort(key=lambda kv: abs(kv[1]), reverse=True)
            res["prefilter_scores"] = scores
            prefilter_feats = [f for f,_ in scores[:prefilter_top_k]]
            res["prefilter_feats"] = prefilter_feats

            # if no features or no train labels, return
            if len(prefilter_feats) == 0:
                return res
            y_train = df.loc[train_mask, target_col]
            mask_train = y_train.notna()
            X_train = df.loc[train_mask, prefilter_feats].loc[mask_train].copy()
            y_train = y_train.loc[mask_train].copy()
            if len(X_train) == 0:
                return res

            # train xgboost for fold selection
            try:
                dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
                booster = xgb.train(xgb_params_prefilter, dtrain, num_boost_round=600, verbose_eval=False)
                importance = booster.get_score(importance_type="gain") or booster.get_score(importance_type="weight")
            except Exception as e:
                # training failed for this fold
                return res

            res["importance"] = importance
            # rank by importance pick top max_features
            feat_import_sorted = sorted(prefilter_feats, key=lambda fv: importance.get(fv, 0.0), reverse=True)
            selected = feat_import_sorted[:max_features]
            res["selected_feats"] = selected

            # train & evaluate on train (pooled) for diagnostics
            preds_train = booster.predict(xgb.DMatrix(X_train, missing=np.nan))
            res["train_metrics"] = evaluate_preds(y_train.values, preds_train)

            # evaluate on fold test (rows where target exists)
            y_test = df.loc[test_mask, target_col]

            # IMPORTANT: booster was trained on prefilter_feats, so we must
            # pass the same feature set at prediction time.
            
            if len(prefilter_feats) > 0:
                X_test_full = df.loc[test_mask, prefilter_feats].copy()
            
                # ensure same column order as training
                X_test_full = X_test_full.reindex(columns=prefilter_feats)
            
                preds_test = booster.predict(
                    xgb.DMatrix(X_test_full, missing=np.nan)
                )
            
                res["test_metrics"] = evaluate_preds(y_test.values, preds_test)
            else:
                res["test_metrics"] = {"n":0, "pooled_ic":np.nan, "rmse":np.nan, "mae":np.nan}

            return res

        out["DE"] = _country_select(features_de, target_de)
        out["FR"] = _country_select(features_fr, target_fr)
        return out

    # run folds (parallel)
    results = Parallel(n_jobs=n_jobs)(delayed(run_fold)(i) for i in starts)
    results = [r for r in results if r is not None]
    n_folds = len(results)
    if n_folds == 0:
        raise ValueError("No folds produced.")

    # aggregate selections and importances
    sel_counts_de = {}
    sel_counts_fr = {}
    agg_importance_de = {}
    agg_importance_fr = {}
    fold_level_metrics = []

    for r in results:
        # DE
        de = r["DE"]
        # record fold metrics
        fold_level_metrics.append({
            "train_until": r["train_until"],
            "test_from": r["test_from"],
            "test_to": r["test_to"],
            "DE_train_n": de.get("train_metrics", {}).get("n", 0),
            "DE_train_ic": de.get("train_metrics", {}).get("pooled_ic", np.nan),
            "DE_train_rmse": de.get("train_metrics", {}).get("rmse", np.nan),
            "DE_test_n": de.get("test_metrics", {}).get("n", 0),
            "DE_test_ic": de.get("test_metrics", {}).get("pooled_ic", np.nan),
            "DE_test_rmse": de.get("test_metrics", {}).get("rmse", np.nan),
        })
        for f in de.get("selected_feats", []):
            sel_counts_de[f] = sel_counts_de.get(f, 0) + 1
        for k,v in (de.get("importance") or {}).items():
            agg_importance_de[k] = agg_importance_de.get(k, 0.0) + float(v)

        # FR
        fr = r["FR"]
        fold_level_metrics[-1].update({
            "FR_train_n": fr.get("train_metrics", {}).get("n", 0),
            "FR_train_ic": fr.get("train_metrics", {}).get("pooled_ic", np.nan),
            "FR_train_rmse": fr.get("train_metrics", {}).get("rmse", np.nan),
            "FR_test_n": fr.get("test_metrics", {}).get("n", 0),
            "FR_test_ic": fr.get("test_metrics", {}).get("pooled_ic", np.nan),
            "FR_test_rmse": fr.get("test_metrics", {}).get("rmse", np.nan),
        })
        for f in fr.get("selected_feats", []):
            sel_counts_fr[f] = sel_counts_fr.get(f, 0) + 1
        for k,v in (fr.get("importance") or {}).items():
            agg_importance_fr[k] = agg_importance_fr.get(k, 0.0) + float(v)

    # normalize aggregated importance by number of folds where present
    mean_importance_de = {k: v / float(n_folds) for k,v in agg_importance_de.items()}
    mean_importance_fr = {k: v / float(n_folds) for k,v in agg_importance_fr.items()}

    # compute selection frequency
    freq_de = {k: sel_counts_de.get(k, 0) / float(n_folds) for k in features_de}
    freq_fr = {k: sel_counts_fr.get(k, 0) / float(n_folds) for k in features_fr}

    # choose stable features based on freq_threshold or at least top by frequency if none meet threshold
    stable_de = [f for f,frq in freq_de.items() if frq >= freq_threshold]
    stable_fr = [f for f,frq in freq_fr.items() if frq >= freq_threshold]

    # fallback: if stable empty, pick top-k by frequency up to max_features
    def fallback_top(freq_map, k):
        # choose correct importance map depending on feature origin
        importance_map = mean_importance_de if any(k in mean_importance_de for k in freq_map) else mean_importance_fr
        sorted_by_freq = sorted(freq_map.items(), key=lambda kv: (kv[1], importance_map.get(kv[0],0.0)), reverse=True)
        return [kv[0] for kv in sorted_by_freq[:k]]

    if len(stable_de) == 0:
        stable_de = fallback_top(freq_de, max_features)
    if len(stable_fr) == 0:
        stable_fr = fallback_top(freq_fr, max_features)

    # cap to max_features each
    stable_de = stable_de[:max_features]
    stable_fr = stable_fr[:max_features]

    # union
    selected_union = list(dict.fromkeys(stable_de + stable_fr))

    diagnostics = {
        "n_folds": n_folds,
        "folds": pd.DataFrame(fold_level_metrics),
        "freq_de": freq_de,
        "freq_fr": freq_fr,
        "mean_importance_de": mean_importance_de,
        "mean_importance_fr": mean_importance_fr,
        "stable_de": stable_de,
        "stable_fr": stable_fr,
        "selected_union": selected_union,
        "raw_fold_results": results
    }



    # ---------- print diagnostics ----------
    if verbose:
        folds_df = diagnostics["folds"]
        n_folds = diagnostics["n_folds"]

        # avg prefilter kept / avg selected per fold (DE / FR) from raw_fold_results
        raw = diagnostics.get("raw_fold_results", [])
        pref_counts_de = []
        sel_counts_de = []
        pref_counts_fr = []
        sel_counts_fr = []
        for r in raw:
            de = r.get("DE", {})
            fr = r.get("FR", {})
            pref_counts_de.append(len(de.get("prefilter_feats", [])))
            sel_counts_de.append(len(de.get("selected_feats", [])))
            pref_counts_fr.append(len(fr.get("prefilter_feats", [])))
            sel_counts_fr.append(len(fr.get("selected_feats", [])))

        avg_prefilter_de = float(np.mean(pref_counts_de)) if len(pref_counts_de)>0 else 0
        avg_selected_de = float(np.mean(sel_counts_de)) if len(sel_counts_de)>0 else 0
        avg_prefilter_fr = float(np.mean(pref_counts_fr)) if len(pref_counts_fr)>0 else 0
        avg_selected_fr = float(np.mean(sel_counts_fr)) if len(sel_counts_fr)>0 else 0

        # fold-level aggregated train / test metrics (mean across folds)
        def _col_mean(df, col):
            return float(df[col].dropna().mean()) if (col in df and df[col].notna().sum()>0) else np.nan

        de_train_ic_mean = _col_mean(folds_df, "DE_train_ic")
        de_test_ic_mean  = _col_mean(folds_df, "DE_test_ic")
        de_train_rmse_mean = _col_mean(folds_df, "DE_train_rmse")
        de_test_rmse_mean  = _col_mean(folds_df, "DE_test_rmse")

        fr_train_ic_mean = _col_mean(folds_df, "FR_train_ic")
        fr_test_ic_mean  = _col_mean(folds_df, "FR_test_ic")
        fr_train_rmse_mean = _col_mean(folds_df, "FR_train_rmse")
        fr_test_rmse_mean  = _col_mean(folds_df, "FR_test_rmse")

        # stable selections & union
        stable_de = diagnostics.get("stable_de", [])
        stable_fr = diagnostics.get("stable_fr", [])
        selected_union = diagnostics.get("selected_union", [])

        # mean importance (sorted) helper
        mean_imp_de = diagnostics.get("mean_importance_de", {})
        mean_imp_fr = diagnostics.get("mean_importance_fr", {})
        top_imp_de = sorted(mean_imp_de.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_imp_fr = sorted(mean_imp_fr.items(), key=lambda kv: kv[1], reverse=True)[:10]

        print("\n=== FEATURE SELECTION DIAGNOSTICS ===")
        print("Folds:", n_folds)
        print(f"Avg prefilter kept: DE={avg_prefilter_de:.1f}, FR={avg_prefilter_fr:.1f}")
        print(f"Avg selected (per fold): DE={avg_selected_de:.1f}, FR={avg_selected_fr:.1f}")
        print("Union selected count:", len(selected_union))

        print("\n--- Fold-averaged pooled metrics (mean over folds) ---")
        print("DE: train pooled IC / RMSE  :", np.round(de_train_ic_mean, 4), np.round(de_train_rmse_mean, 4))
        print("DE: test  pooled IC / RMSE  :", np.round(de_test_ic_mean, 4), np.round(de_test_rmse_mean, 4))
        print("FR: train pooled IC / RMSE  :", np.round(fr_train_ic_mean, 4), np.round(fr_train_rmse_mean, 4))
        print("FR: test  pooled IC / RMSE  :", np.round(fr_test_ic_mean, 4), np.round(fr_test_rmse_mean, 4))

        print("\nTop stable DE selected features (stable_de):", stable_de)
        print("Top stable FR selected features (stable_fr):", stable_fr)
        print("Final union (selected_union):", selected_union)

        print("\nTop mean importance (DE):")
        for f, v in top_imp_de:
            print(" ", f, ":", float(v))
        print("\nTop mean importance (FR):")
        for f, v in top_imp_fr:
            print(" ", f, ":", float(v))


    return selected_union, diagnostics

#################################################
#################################################
# MAIN XGBOOST FUNCTION
#################################################
#################################################

# =============================================================================
# 1. detect_markov_regime — filtered probs
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

    model = GaussianHMM(
        n_components=n_states,
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
# 2. add_regime_probs_to_panel — fit once, attach to panel before CV
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
# 3. two_regime_rolling_cv_per_country
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
    n_jobs: int = 4,
    min_samples_per_regime: int = 50,
    # --- Markov (pre-computed columns, no per-fold refitting) ---
    # Requires: add_regime_probs_to_panel() called before CV so that
    # {pref}_regime_prob_0 / _1 already exist as columns in df.
    use_markov: Dict[str, bool] = None,
    k_regimes: int = 2,
    use_per_regime_models: bool = False,   # soft-weighted per-regime XGBoost
    # --- Vol zscore (independent of Markov, computed per fold) ---
    # A data-driven regime signal: (log_rv - rolling_mean) / rolling_std
    # Computed from the training window only — no lookahead.
    # Can be used together with or instead of Markov.
    use_vol_zscore: Dict[str, bool] = None,
    vol_zscore_window: int = 252,
    # --- Recency weighting ---
    use_recency_weights: bool = False,
    recency_halflife: int = 180,
    # kept for backward compat, ignored
    detect_kwargs: Dict[str, Any] = None,
) -> tuple:
    """
    Rolling walk-forward CV, per-country XGBoost on HAR-RV residuals.

    MARKOV
        Pre-computed filtered regime probabilities are read from columns
        {pref}_regime_prob_0 / _1 already in df. No HMM is fitted here.
        Call add_regime_probs_to_panel() once before this function.
        The CV fold simply reads the prob value for each training/test row;
        test rows are filled with the last training-period value (causal fill).
        If use_per_regime_models=True, two XGBoost models are trained per
        country, soft-weighted by the regime probability.

    VOL ZSCORE
        Computed in each fold from the training window only.
        Formula: zscore_t = (log_rv_t - rolling_mean) / rolling_std
        where mean and std use vol_zscore_window days of training history.
        Test rows are filled with the zscore computed using the last
        training-window statistics (causal fill, same logic as Markov).
        Adds {pref}_vol_zscore as an extra feature column for that fold.

    Both options are fully independent.
    """

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

    features_de_base = [f for f in features if f.startswith("DE_")]
    features_fr_base = [f for f in features if f.startswith("FR_")]
    if not features_de_base or not features_fr_base:
        raise ValueError("features must contain both DE_ and FR_ prefixed columns.")

    # Validate Markov columns exist before starting
    for pref in ['DE', 'FR']:
        if use_markov.get(pref, False):
            for i in range(k_regimes):
                col = f"{pref}_regime_prob_{i}"
                if col not in df.columns:
                    raise ValueError(
                        f"use_markov['{pref}']=True but '{col}' not found in df. "
                        f"Call add_regime_probs_to_panel() before the CV."
                    )

    # ------------------------------------------------------------------ helpers
    def safe_spearman(a, b):
        try:
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 5:
                return np.nan
            return float(spearmanr(a[mask], b[mask]).correlation)
        except Exception:
            return np.nan

    def _recency_weights(n: int, halflife: int) -> np.ndarray:
        decay = np.log(2) / halflife
        w = np.exp(-decay * np.arange(n - 1, -1, -1, dtype=float))
        return w / w.mean()

    def train_xgb_safe(
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ):
        if X.shape[0] == 0:
            return None
        y_ser = pd.Series(y.values, index=X.index)
        valid = y_ser.notna() & np.isfinite(y_ser)
        if valid.sum() < min_samples_per_regime:
            return None

        X_c = X.loc[valid]
        y_c = y_ser.loc[valid].astype(float)
        w_c = sample_weight[valid.values] if sample_weight is not None else None

        val_size = max(int(0.15 * len(X_c)), 20)
        if len(X_c) <= val_size + 10:
            return None

        X_tr, X_val = X_c.iloc[:-val_size], X_c.iloc[-val_size:]
        y_tr, y_val = y_c.iloc[:-val_size], y_c.iloc[-val_size:]
        w_tr = w_c[:-val_size] if w_c is not None else None

        dtrain = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, weight=w_tr)
        dval   = xgb.DMatrix(X_val, label=y_val, missing=np.nan)

        try:
            m = xgb.train(
                xgb_params, dtrain,
                num_boost_round=2000,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            return {
                'model':      m,
                'train_pred': m.predict(dtrain),
                'val_pred':   m.predict(dval),
                'y_tr':       y_tr.values,
                'y_val':      y_val.values,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------ fold
    def run_fold(i: int):
        train_start = max(0, i - train_window) if train_window is not None else 0
        train_dates = unique_dates[train_start:i]
        test_dates  = unique_dates[i + gap_days: i + gap_days + test_horizon]
        if len(train_dates) < min_train_days or len(test_dates) == 0:
            return None

        train_mask = df['DATE'].isin(train_dates)
        test_mask  = df['DATE'].isin(test_dates)

        features_de = features_de_base.copy()
        features_fr = features_fr_base.copy()
        fold_df = df.copy()

        # ----------------------------------------------------------------
        # MARKOV — read pre-computed filtered probs, no fitting here.
        # Test rows are filled with the last training-period prob value.
        # ----------------------------------------------------------------
        for pref, feats in [('DE', features_de), ('FR', features_fr)]:
            if not use_markov.get(pref, False):
                continue

            last_train_iloc = fold_df.loc[train_mask].index[-1]

            for i_state in range(k_regimes):
                col = f"{pref}_regime_prob_{i_state}"
                last_val = fold_df.loc[last_train_iloc, col]
                fill_val = float(last_val) if np.isfinite(last_val) else 0.5
                fold_df.loc[test_mask, col] = fill_val
                if col not in feats:
                    feats.append(col)

        # ----------------------------------------------------------------
        # VOL ZSCORE — computed per fold from training window only.
        # Strictly independent of Markov; can be used with or without it.
        # ----------------------------------------------------------------
        for pref, feats in [('DE', features_de), ('FR', features_fr)]:
            if not use_vol_zscore.get(pref, False):
                continue

            rv_col = f"{pref}_realized_vol"
            zs_col = f"{pref}_vol_zscore"
            if rv_col not in fold_df.columns:
                continue

            rv_tr = fold_df.loc[train_mask, rv_col].astype(float).clip(lower=1e-12)
            log_rv_tr = np.log(rv_tr)

            mu  = log_rv_tr.rolling(vol_zscore_window, min_periods=30).mean()
            sig = log_rv_tr.rolling(vol_zscore_window, min_periods=30).std().clip(lower=1e-8)

            # Last training-window statistics — used to fill test rows (no lookahead)
            last_mu  = float(mu.dropna().iloc[-1])  if mu.dropna().size  else 0.0
            last_sig = float(sig.dropna().iloc[-1]) if sig.dropna().size else 1.0

            # Training rows: rolling zscore within their window
            fold_df.loc[train_mask, zs_col] = ((log_rv_tr - mu) / sig).values

            # Test rows: zscore using last training-window baseline
            rv_te = fold_df.loc[test_mask, rv_col].astype(float).clip(lower=1e-12)
            fold_df.loc[test_mask, zs_col] = (np.log(rv_te) - last_mu) / last_sig

            if zs_col not in feats:
                feats.append(zs_col)

        # ----------------------------------------------------------------
        # X / y matrices
        # ----------------------------------------------------------------
        def _Xy(feats, tgt, mask):
            X = fold_df.loc[mask, feats].copy()
            y = (fold_df.loc[mask, tgt].copy() if tgt in fold_df.columns
                 else pd.Series(np.nan, index=X.index))
            return X, y

        X_tr_de, y_tr_de = _Xy(features_de, target_de, train_mask)
        X_te_de, y_te_de = _Xy(features_de, target_de, test_mask)
        X_tr_fr, y_tr_fr = _Xy(features_fr, target_fr, train_mask)
        X_te_fr, y_te_fr = _Xy(features_fr, target_fr, test_mask)

        # ----------------------------------------------------------------
        # Train + predict
        # Single model (default) or two soft-weighted per-regime models.
        # Per-regime models use Markov prob_1 as soft weight — only
        # meaningful when use_markov is True for that country.
        # ----------------------------------------------------------------
        def _train_predict(X_tr, y_tr, X_te, pref):
            n = len(X_tr)
            w_base = _recency_weights(n, recency_halflife) if use_recency_weights else None

            prob_col_high = f"{pref}_regime_prob_1"
            has_regime = (
                use_per_regime_models
                and use_markov.get(pref, False)
                and prob_col_high in fold_df.columns
            )

            if not has_regime:
                res = train_xgb_safe(X_tr, y_tr, sample_weight=w_base)
                if res is None:
                    return np.full(len(X_te), np.nan), res
                return res['model'].predict(xgb.DMatrix(X_te, missing=np.nan)), res

            # Soft-weighted models: weight each sample by its regime prob
            p_high = fold_df.loc[train_mask, prob_col_high].fillna(0.5).values[:n]
            p_low  = 1.0 - p_high
            base   = w_base if w_base is not None else np.ones(n)
            w_high = base * p_high;  w_high /= (w_high.mean() + 1e-12)
            w_low  = base * p_low;   w_low  /= (w_low.mean()  + 1e-12)

            res_high = train_xgb_safe(X_tr, y_tr, sample_weight=w_high)
            res_low  = train_xgb_safe(X_tr, y_tr, sample_weight=w_low)

            p_high_test = fold_df.loc[test_mask, prob_col_high].fillna(0.5).values
            dtest = xgb.DMatrix(X_te, missing=np.nan)
            ph = res_high['model'].predict(dtest) if res_high else np.zeros(len(X_te))
            pl = res_low['model'].predict(dtest)  if res_low  else np.zeros(len(X_te))
            blended = p_high_test * ph + (1 - p_high_test) * pl
            return blended, res_high   # return high-regime res for diagnostics

        preds_de, res_de = _train_predict(X_tr_de, y_tr_de, X_te_de, 'DE')
        preds_fr, res_fr = _train_predict(X_tr_fr, y_tr_fr, X_te_fr, 'FR')

        # ----------------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------------
        def _diag(res):
            if res is None:
                return {'train_rmse_exp': np.nan, 'val_rmse_exp': np.nan,
                        'train_ic_exp':   np.nan, 'val_ic_exp':   np.nan, 'fi_exp': {}}
            tp, vp, ty, vy = res['train_pred'], res['val_pred'], res['y_tr'], res['y_val']
            return {
                'train_rmse_exp': float(np.sqrt(np.mean((tp - ty) ** 2))) if tp.size else np.nan,
                'val_rmse_exp':   float(np.sqrt(np.mean((vp - vy) ** 2))) if vp.size else np.nan,
                'train_ic_exp':   safe_spearman(tp, ty),
                'val_ic_exp':     safe_spearman(vp, vy),
                'fi_exp':         res['model'].get_score(importance_type='gain'),
            }

        diag_de = _diag(res_de)
        diag_fr = _diag(res_fr)

        # ----------------------------------------------------------------
        # Prediction rows
        # ----------------------------------------------------------------
        rows = []
        for j, idx in enumerate(X_te_de.index):
            rows.append({
                'DATE':    fold_df.loc[idx, 'DATE'],
                'COUNTRY': 'DE',
                'pred':    float(preds_de[j]) if np.isfinite(preds_de[j]) else np.nan,
                'true':    float(y_te_de.loc[idx]) if np.isfinite(y_te_de.loc[idx]) else np.nan,
            })
        for j, idx in enumerate(X_te_fr.index):
            rows.append({
                'DATE':    fold_df.loc[idx, 'DATE'],
                'COUNTRY': 'FR',
                'pred':    float(preds_fr[j]) if np.isfinite(preds_fr[j]) else np.nan,
                'true':    float(y_te_fr.loc[idx]) if np.isfinite(y_te_fr.loc[idx]) else np.nan,
            })
        pred_df = pd.DataFrame(rows)

        # ----------------------------------------------------------------
        # Per-fold metrics
        # ----------------------------------------------------------------
        def safe_metrics(p, t):
            p, t = np.asarray(p, float), np.asarray(t, float)
            m = np.isfinite(p) & np.isfinite(t)
            if not m.any():
                return np.nan, np.nan, np.nan
            return (safe_spearman(p[m], t[m]),
                    float(np.sqrt(np.mean((p[m] - t[m]) ** 2))),
                    float(np.mean(np.abs(p[m] - t[m]))))

        ic_de = rmse_de = mae_de = ic_fr = rmse_fr = mae_fr = np.nan
        n_test_de = n_test_fr = 0
        if not pred_df.empty:
            for c, (ic_ref, rmse_ref, mae_ref, n_ref) in [
                ('DE', ('ic_de', 'rmse_de', 'mae_de', 'n_test_de')),
                ('FR', ('ic_fr', 'rmse_fr', 'mae_fr', 'n_test_fr')),
            ]:
                sub = pred_df[pred_df['COUNTRY'] == c]
                if len(sub):
                    ic_v, rm_v, ma_v = safe_metrics(sub['pred'], sub['true'])
                    if c == 'DE':
                        ic_de = ic_v; rmse_de = rm_v; mae_de = ma_v
                        n_test_de = int(sub['true'].notna().sum())
                    else:
                        ic_fr = ic_v; rmse_fr = rm_v; mae_fr = ma_v
                        n_test_fr = int(sub['true'].notna().sum())

        pm = pred_df['pred'].notna() & pred_df['true'].notna()
        pooled_ic   = safe_spearman(pred_df.loc[pm, 'pred'], pred_df.loc[pm, 'true'])
        pooled_rmse = float(np.sqrt(np.mean((pred_df.loc[pm, 'pred'] - pred_df.loc[pm, 'true']) ** 2))) if pm.any() else np.nan
        pooled_mae  = float(np.mean(np.abs(pred_df.loc[pm, 'pred'] - pred_df.loc[pm, 'true']))) if pm.any() else np.nan

        fold_stat = {
            'train_until':           train_dates[-1],
            'test_from':             test_dates[0],
            'test_to':               test_dates[-1],
            'n_train':               int(train_mask.sum()),
            'n_test_de':             n_test_de,
            'n_test_fr':             n_test_fr,
            'spearman_ic_de':        float(ic_de)   if np.isfinite(ic_de)   else np.nan,
            'rmse_de':               float(rmse_de) if np.isfinite(rmse_de) else np.nan,
            'mae_de':                float(mae_de)  if np.isfinite(mae_de)  else np.nan,
            'spearman_ic_fr':        float(ic_fr)   if np.isfinite(ic_fr)   else np.nan,
            'rmse_fr':               float(rmse_fr) if np.isfinite(rmse_fr) else np.nan,
            'mae_fr':                float(mae_fr)  if np.isfinite(mae_fr)  else np.nan,
            'pooled_ic':             pooled_ic,
            'pooled_rmse':           pooled_rmse,
            'pooled_mae':            pooled_mae,
            'train_rmse_pre_exp':    float(diag_de.get('train_rmse_exp', np.nan)),
            'val_rmse_pre_exp':      float(diag_de.get('val_rmse_exp',   np.nan)),
            'train_ic_pre_exp':      float(diag_de.get('train_ic_exp',   np.nan)),
            'val_ic_pre_exp':        float(diag_de.get('val_ic_exp',     np.nan)),
            'train_rmse_post_exp':   np.nan,
            'val_rmse_post_exp':     np.nan,
            'train_ic_post_exp':     np.nan,
            'val_ic_post_exp':       np.nan,
            'n_pre_train':           np.nan,
            'n_post_train':          np.nan,
            'train_rmse_pre_exp_de': float(diag_de.get('train_rmse_exp', np.nan)),
            'val_rmse_pre_exp_de':   float(diag_de.get('val_rmse_exp',   np.nan)),
            'train_ic_pre_exp_de':   float(diag_de.get('train_ic_exp',   np.nan)),
            'val_ic_pre_exp_de':     float(diag_de.get('val_ic_exp',     np.nan)),
            'train_rmse_pre_exp_fr': float(diag_fr.get('train_rmse_exp', np.nan)),
            'val_rmse_pre_exp_fr':   float(diag_fr.get('val_rmse_exp',   np.nan)),
            'train_ic_pre_exp_fr':   float(diag_fr.get('train_ic_exp',   np.nan)),
            'val_ic_pre_exp_fr':     float(diag_fr.get('val_ic_exp',     np.nan)),
            # markov_de / markov_fr: kept for downstream compat
            # no longer contains per-fold HMM diag — just a status flag
            'markov_de':             {'fitted': use_markov.get('DE', False), 'source': 'pre_computed'},
            'markov_fr':             {'fitted': use_markov.get('FR', False), 'source': 'pre_computed'},
            'fi_exp_de':             diag_de.get('fi_exp', {}),
            'fi_exp_fr':             diag_fr.get('fi_exp', {}),
            'train_window_used':     len(train_dates),
        }

        gc.collect()
        return pred_df, fold_stat

    # ------------------------------------------------------------------ run
    starts = list(range(
        min_train_days,
        len(unique_dates) - gap_days - test_horizon + 1,
        test_horizon,
    ))
    results = Parallel(n_jobs=n_jobs)(delayed(run_fold)(i) for i in starts)
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No folds produced.")

    all_preds  = pd.concat([r[0] for r in results], ignore_index=True)
    fold_stats = pd.DataFrame([r[1] for r in results])

    # ------------------------------------------------------------------ aggregate
    pm_all = all_preds['pred'].notna() & all_preds['true'].notna()
    pooled_ic   = safe_spearman(all_preds.loc[pm_all, 'pred'], all_preds.loc[pm_all, 'true'])
    pooled_rmse = float(np.sqrt(np.mean((all_preds.loc[pm_all, 'pred'] - all_preds.loc[pm_all, 'true']) ** 2))) if pm_all.any() else np.nan
    pooled_mae  = float(np.mean(np.abs(all_preds.loc[pm_all, 'pred'] - all_preds.loc[pm_all, 'true']))) if pm_all.any() else np.nan

    per_country_ic = {}; per_country_rmse = {}; per_country_mae = {}
    for c in ['DE', 'FR']:
        sub = all_preds[all_preds['COUNTRY'] == c]
        mc  = sub['pred'].notna() & sub['true'].notna()
        if mc.any():
            per_country_ic[c]   = safe_spearman(sub.loc[mc, 'pred'], sub.loc[mc, 'true'])
            per_country_rmse[c] = float(np.sqrt(np.mean((sub.loc[mc, 'pred'] - sub.loc[mc, 'true']) ** 2)))
            per_country_mae[c]  = float(np.mean(np.abs(sub.loc[mc, 'pred'] - sub.loc[mc, 'true'])))
        else:
            per_country_ic[c] = per_country_rmse[c] = per_country_mae[c] = np.nan

    rolling_ic = {}; rolling_rmse = {}; rolling_mae = {}
    rolling_pred_mean = {}; rolling_pred_std = {}
    for c in ['DE', 'FR']:
        sub = all_preds[all_preds['COUNTRY'] == c].set_index('DATE').sort_index()
        if not len(sub):
            for d in [rolling_ic, rolling_rmse, rolling_mae, rolling_pred_mean, rolling_pred_std]:
                d[c] = pd.Series(dtype=float)
            continue
        rp = sub['pred'].rank(); rt = sub['true'].rank()
        rolling_ic[c]        = rp.rolling(126, min_periods=10).corr(rt)
        rolling_rmse[c]      = ((sub['pred'] - sub['true']) ** 2).rolling(126, min_periods=10).mean().apply(np.sqrt)
        rolling_mae[c]       = (sub['pred'] - sub['true']).abs().rolling(126, min_periods=10).mean()
        rolling_pred_mean[c] = sub['pred'].rolling(126, min_periods=10).mean()
        rolling_pred_std[c]  = sub['pred'].rolling(126, min_periods=10).std()

    ps = all_preds.set_index('DATE').sort_index()
    rolling_pooled_ic = ps['pred'].rank().rolling(126, min_periods=10).corr(ps['true'].rank())
    pooled_pred_mean  = ps['pred'].rolling(126, min_periods=10).mean()
    pooled_pred_std   = ps['pred'].rolling(126, min_periods=10).std()

    overall = {
        'mean_fold_ic_de':     float(fold_stats['spearman_ic_de'].dropna().mean()),
        'mean_fold_ic_fr':     float(fold_stats['spearman_ic_fr'].dropna().mean()),
        'mean_fold_pooled_ic': float(fold_stats['pooled_ic'].dropna().mean()),
        'pooled_ic':           pooled_ic,
        'pooled_rmse':         pooled_rmse,
        'pooled_mae':          pooled_mae,
        'per_country_ic':      per_country_ic,
        'per_country_rmse':    per_country_rmse,
        'per_country_mae':     per_country_mae,
        'mean_rmse_fold':      float(fold_stats[['rmse_de', 'rmse_fr']].stack().dropna().mean()),
        'mean_mae_fold':       float(fold_stats[['mae_de', 'mae_fr']].stack().dropna().mean()),
        'n_folds':             len(fold_stats),
        'n_regime_detected_folds_de': int(
            fold_stats['markov_de'].apply(
                lambda x: bool(x.get('fitted')) if isinstance(x, dict) else False
            ).sum()),
        'n_regime_detected_folds_fr': int(
            fold_stats['markov_fr'].apply(
                lambda x: bool(x.get('fitted')) if isinstance(x, dict) else False
            ).sum()),
    }

    print("=== SUMMARY ===")
    print(f"Pooled IC: {pooled_ic}")
    print(f"Per-country IC: {per_country_ic}")
    print(f"Mean fold ICs — DE: {overall['mean_fold_ic_de']:.4f}  FR: {overall['mean_fold_ic_fr']:.4f}")
    print(f"Mean RMSE (fold mean): {overall['mean_rmse_fold']:.4f}")
    print(f"Pooled RMSE / MAE: {pooled_rmse:.4f} / {pooled_mae:.4f}")

    return all_preds, {
        'folds':             fold_stats,
        'overall':           overall,
        'rolling_ic':        rolling_ic,
        'rolling_pooled_ic': rolling_pooled_ic,
        'rolling_rmse':      rolling_rmse,
        'rolling_mae':       rolling_mae,
        'rolling_pred_mean': rolling_pred_mean,
        'rolling_pred_std':  rolling_pred_std,
        'pooled_pred_mean':  pooled_pred_mean,
        'pooled_pred_std':   pooled_pred_std,
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


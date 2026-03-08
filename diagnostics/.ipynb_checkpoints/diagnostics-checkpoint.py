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
from scipy.stats import spearmanr
from scipy import stats
from statsmodels.tsa.stattools import acf
EPS = 1e-12


#----------------------------
# Baseline model diagnostics
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

EPS = 1e-12


def baseline_check(df, pref, window=126, acf_lags=40, plot=True):
    """
    Unified baseline diagnostics for har-rv vs realized volatility.

    Parameters
    ----------
    df : DataFrame
    pref : str
        Country prefix ("DE", "FR", etc.)
    window : int
        Rolling window
    acf_lags : int
        Lags for ACF plot
    plot : bool
        Whether to produce plots

    Returns
    -------
    roll_df : DataFrame
        Rolling RMSE / MAE / Spearman
    """

    sig_col = f"{pref}_garch_sigma"
    real_col = f"{pref}_realized_vol"

    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")

    sig = df[sig_col]
    real = df[real_col]

    ok = sig.notna() & real.notna()

    print(f"\n--- {pref} Baseline diagnostics ---")
    print("rows:", len(df))
    print("nan sigma:", sig.isna().sum(), f"({sig.isna().mean():.1%})")
    print("nan realized:", real.isna().sum(), f"({real.isna().mean():.1%})")
    print("valid pairs:", ok.sum())

    if ok.sum() > 10:
        r = spearmanr(np.log(real[ok] + EPS), np.log(sig[ok] + EPS)).correlation
        print(f"spearman(log real, log sigma): {r:.4f}")

    # residual
    resid = np.log(real + EPS) - np.log(sig + EPS)

    err = sig - real
    abs_err = np.abs(err)
    sq_err = err**2

    # rolling metrics
    roll_rmse = np.sqrt(sq_err.rolling(window, min_periods=10).mean())
    roll_mae = abs_err.rolling(window, min_periods=10).mean()

    ranked_sig = sig.rank()
    ranked_real = real.rank()
    roll_spear = ranked_sig.rolling(window, min_periods=10).corr(ranked_real)

    roll_df = pd.DataFrame({
        "DATE": df["DATE"],
        "roll_rmse": roll_rmse,
        "roll_mae": roll_mae,
        "roll_spearman": roll_spear,
        "resid": resid
    }).set_index("DATE")

    if plot:

        # Rolling metrics
        fig, ax = plt.subplots(2,1,figsize=(12,7),sharex=True)

        ax[0].plot(roll_df.index, roll_df["roll_spearman"], label="Rolling Spearman")
        ax[0].axhline(0,color="k",lw=0.5)
        ax[0].set_title(f"{pref} Rolling Spearman (window={window})")
        ax[0].grid()

        ax[1].plot(roll_df.index, roll_df["roll_rmse"], label="Rolling RMSE")
        ax[1].set_title("Rolling RMSE")
        ax[1].grid()

        plt.tight_layout()
        plt.show()

        # Residual diagnostics
        res = resid.dropna()

        fig, ax = plt.subplots(1,3,figsize=(14,4))

        ax[0].hist(res,bins=60,density=True)
        ax[0].set_title("Residual distribution")

        stats.probplot(res, dist="norm", plot=ax[1])
        ax[1].set_title("QQ plot")

        plot_acf(res, lags=acf_lags, ax=ax[2])
        ax[2].set_title("Residual ACF")

        plt.tight_layout()
        plt.show()

        print(
            f"residual mean={res.mean():.4f}  "
            f"std={res.std():.4f}  "
            f"skew={pd.Series(res).skew():.4f}"
        )

    return roll_df


#----------------------------------------------------------------
# Main XGBoost diagnostics
#-----------------------------------------------------------------

def plot_cv_diagnostics(preds: pd.DataFrame, metrics: Dict[str, Any], max_acf_lag: int = 30):
    """
    - Shows fold-level and rolling Spearman on a single two-panel figure (DE/FR/pooled).
    - Plots rolling RMSE/MAE per-country when available.
    - Scatter pred vs true pooled and per-country.
    - Error distributions pooled and per-country.
    - Autocorrelation of errors per-country.
    - Cross-sectional mean/std and rolling prediction mean/std.
    - Train/Val diffs (spearman & rmse) computed directly from folds
    """

    folds = metrics.get("folds", pd.DataFrame()).copy()
    overall = metrics.get("overall", {}) or {}
    rolling_ic = metrics.get("rolling_ic", {}) or {}
    rolling_rmse = metrics.get("rolling_rmse", {}) or {}
    rolling_mae = metrics.get("rolling_mae", {}) or {}
    rolling_pred_mean = metrics.get("rolling_pred_mean", {}) or {}
    rolling_pred_std = metrics.get("rolling_pred_std", {}) or {}
    pooled_pred_mean = metrics.get("pooled_pred_mean", pd.Series(dtype=float))
    pooled_pred_std = metrics.get("pooled_pred_std", pd.Series(dtype=float))

    _tmp = metrics.get("rolling_pooled_ic", pd.Series(dtype=float))
    rolling_pooled_ic = _tmp if (not hasattr(_tmp, "__len__") or len(_tmp) > 0) else pd.Series(dtype=float)

    # Prepare preds
    preds = preds.copy()
    if "DATE" in preds.columns:
        preds["DATE"] = pd.to_datetime(preds["DATE"])
    preds = preds.sort_values("DATE")

    
    # ---- 1) Fold-level IC (DE, FR, pooled) and rolling IC in single two-panel figure ----
    #fig, axes = plt.subplots(figsize=(14, 10), sharex=True)
    #ax_fold, ax_roll = axes
    """
    # Fold-level: detect columns for DE/FR/pooled IC in folds
    if not folds.empty:
        # unify train_until to datetimes if present
        if "train_until" in folds.columns:
            try:
                x_dates = pd.to_datetime(folds["train_until"])
            except Exception:
                x_dates = folds["train_until"]
        else:
            # fallback: use index
            x_dates = folds.index

        # pooled fold IC column candidates
        pooled_candidates = ["pooled_ic", "spearman_ic", "pooled_spearman", "fold_ic", "pooled"]
        pooled_col = next((c for c in pooled_candidates if c in folds.columns), None)

        # per-country candidates
        de_col_candidates = ["spearman_ic_de", "DE_fold_ic", "DE_spearman", "spearman_de"]
        fr_col_candidates = ["spearman_ic_fr", "FR_fold_ic", "FR_spearman", "spearman_fr"]
        de_col = next((c for c in de_col_candidates if c in folds.columns), None)
        fr_col = next((c for c in fr_col_candidates if c in folds.columns), None)

        # Plot pooled fold IC if found
        if pooled_col is not None:
            ax_fold.plot(x_dates, folds[pooled_col], marker="o", linestyle="-", label="pooled (fold)")
        # Plot DE/FR if found
        if de_col is not None:
            ax_fold.plot(x_dates, folds[de_col], marker="o", linestyle="-", label="DE (fold)")
        if fr_col is not None:
            ax_fold.plot(x_dates, folds[fr_col], marker="o", linestyle="-", label="FR (fold)")

        ax_fold.axhline(0, color="k", lw=0.6)
        ax_fold.set_ylabel("Fold Spearman IC")
        ax_fold.set_title("Fold-level Spearman IC (pooled & per-country)")
        ax_fold.legend()
        ax_fold.grid(True)

    else:
        ax_fold.text(0.5, 0.5, "No fold-level data available", ha="center", va="center")
        ax_fold.set_title("Fold-level Spearman IC (no data)")

    """
    plt.figure(figsize=(14, 4))
    # Rolling IC: prefer metrics['rolling_ic'] dict with keys 'DE','FR' and metrics['rolling_pooled_ic']
    has_roll_de = isinstance(rolling_ic, dict) and "DE" in rolling_ic and rolling_ic["DE"] is not None
    has_roll_fr = isinstance(rolling_ic, dict) and "FR" in rolling_ic and rolling_ic["FR"] is not None
    has_pooled_roll = isinstance(rolling_pooled_ic, (pd.Series, np.ndarray)) and len(rolling_pooled_ic) > 0

    if has_roll_de:
        s = rolling_ic["DE"].dropna().copy()
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
        plt.plot(s.index, s.values, label="DE (rolling)", linestyle="-")
    if has_roll_fr:
        s = rolling_ic["FR"].dropna().copy()
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
        plt.plot(s.index, s.values, label="FR (rolling)", linestyle="-")
    if has_pooled_roll:
        s = pd.Series(rolling_pooled_ic).dropna().copy()
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
        plt.plot(s.index, s.values, label="pooled (rolling)", linestyle="--", alpha=0.9)

    plt.ylabel("Rolling IC")
    plt.title("Rolling Spearman IC (per-country & pooled)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # ---- 2) Rolling RMSE and MAE (per-country) ----
    def plot_rolling_metric(metric_dict, title, ylabel):
        if isinstance(metric_dict, dict) and ("DE" in metric_dict or "FR" in metric_dict):
            plt.figure(figsize=(14, 4))
            plotted = False
            for pref, label in [("DE", "DE"), ("FR", "FR")]:
                s = metric_dict.get(pref)
                if s is None or len(s) == 0:
                    continue
                ser = pd.Series(s).dropna()
                try:
                    ser.index = pd.to_datetime(ser.index)
                except Exception:
                    pass
                plt.plot(ser.index, ser.values, label=f"{label} {title}")
                plotted = True
            if plotted:
                plt.title(title)
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    plot_rolling_metric(rolling_rmse, "Rolling RMSE (per-country)", "RMSE")
    plot_rolling_metric(rolling_mae, "Rolling MAE (per-country)", "MAE")




    # ---- 6) Scatter: pred vs true (pooled) + per-country ----
    from scipy.stats import linregress
    
    def plot_scatter_fit(x, y, title="Pred vs True"):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 2:
            return
    
        plt.figure(figsize=(6,6))
        plt.scatter(x, y, alpha=0.4, s=12, label="data")
    
        # Identity line y=x
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        buffer = 0.02*(xmax - xmin)
        x_line = np.array([xmin - buffer, xmax + buffer])
        plt.plot(x_line, x_line, color='k', linestyle='--', label='y = x')
    
        # Fit line (OLS)
        try:
            res = linregress(x, y)
            slope, intercept = res.slope, res.intercept
            y_fit = intercept + slope * x_line
            plt.plot(x_line, y_fit, color='tab:red', linestyle='-', 
                     label=f'fit: y={slope:.3f}x+{intercept:.3f}')
    
            txt = f"slope={slope:.3f}\nintercept={intercept:.3f}"
            plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                           va='top', ha='left', fontsize=9,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        except Exception as e:
            print("Linear fit failed:", e)
    
        plt.title(title)
        plt.xlabel('pred'); plt.ylabel('true'); plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    
    # ---- pooled ----
    pooled_mask = (~preds['pred'].isna()) & (~preds['true'].isna())
    if pooled_mask.sum() > 0:
        x = preds.loc[pooled_mask, 'pred'].values
        y = preds.loc[pooled_mask, 'true'].values
        plot_scatter_fit(x, y, title="Predicted residual vs True residual (pooled)")
    
    # ---- per-country ----
    if "COUNTRY" in preds.columns:
        for c in preds['COUNTRY'].dropna().unique():
            sub = preds[preds['COUNTRY'] == c]
            mask = (~sub['pred'].isna()) & (~sub['true'].isna())
            if mask.sum() > 0:
                x = sub.loc[mask, 'pred'].values
                y = sub.loc[mask, 'true'].values
                plot_scatter_fit(x, y, title=f'Pred vs True ({c})')






    

    # ---- 4) Prediction error distributions (pooled + per-country) ----
    if {"pred", "true"}.issubset(preds.columns):
        preds = preds.copy()
        preds["err"] = preds["pred"] - preds["true"]
        pool_err = preds["err"].dropna()
        if len(pool_err) > 0:
            plt.figure(figsize=(10, 4))
            plt.hist(pool_err.values, bins=80, density=True, alpha=0.7)
            plt.title("Prediction error distribution (pooled)")
            plt.xlabel("pred - true"); plt.ylabel("density"); plt.grid(True); plt.tight_layout(); plt.show()

        if "COUNTRY" in preds.columns:
            for c in preds["COUNTRY"].dropna().unique():
                e = preds.loc[preds["COUNTRY"] == c, "err"].dropna()
                if len(e) == 0:
                    continue
                plt.figure(figsize=(8, 3.5))
                plt.hist(e.values, bins=60, density=True, alpha=0.7)
                plt.title(f"Prediction error distribution ({c})")
                plt.xlabel("pred - true"); plt.ylabel("density"); plt.grid(True); plt.tight_layout(); plt.show()

    # ---- 5) Error autocorrelation per-country ----
    if "COUNTRY" in preds.columns and "DATE" in preds.columns:
        for c in preds["COUNTRY"].dropna().unique():
            sub = preds[preds["COUNTRY"] == c].set_index("DATE").sort_index()
            if "err" not in sub.columns:
                continue
            e = sub["err"].dropna()
            if len(e) <= 5:
                continue
            acfs = [e.autocorr(lag=lag) for lag in range(1, max_acf_lag + 1)]
            plt.figure(figsize=(12, 3))
            plt.bar(range(1, max_acf_lag + 1), acfs)
            plt.ylim(0,0.5)
            plt.title(f"Error autocorrelation ({c}) lags 1..{max_acf_lag}")
            plt.xlabel("lag"); plt.ylabel("autocorr"); plt.grid(True); plt.tight_layout(); plt.show()

    # ---- 6) Cross-sectional mean/std of predictions across countries by date ----
    if {"DATE", "COUNTRY", "pred"}.issubset(preds.columns):
        try:
            cs = preds.pivot(index="DATE", columns="COUNTRY", values="pred")
            if cs.shape[1] >= 1:
                pred_mean = cs.mean(axis=1)
                pred_std = cs.std(axis=1)

                plt.figure(figsize=(14, 3.5))
                plt.plot(pd.to_datetime(pred_mean.index), pred_mean.values, label="cross-sectional mean(pred)")
                plt.title("Cross-sectional mean of predictions (by date)")
                plt.xlabel("DATE"); plt.ylabel("mean(pred)"); plt.grid(True); plt.tight_layout(); plt.show()

                plt.figure(figsize=(14, 3.5))
                plt.plot(pd.to_datetime(pred_std.index), pred_std.values, label="cross-sectional std(pred)")
                plt.title("Cross-sectional std of predictions (by date)")
                plt.xlabel("DATE"); plt.ylabel("std(pred)"); plt.grid(True); plt.tight_layout(); plt.show()
        except Exception:
            pass

    # ---- 7) Rolling predicted mean ± std (per-country) ----
    if isinstance(rolling_pred_mean, dict) and rolling_pred_mean:
        for c, s_mean in rolling_pred_mean.items():
            if s_mean is None or getattr(s_mean, "empty", True):
                continue
            s_mean = pd.Series(s_mean).dropna()
            s_std = rolling_pred_std.get(c, pd.Series(dtype=float)).dropna()
            try:
                s_mean.index = pd.to_datetime(s_mean.index)
                s_std.index = pd.to_datetime(s_std.index)
            except Exception:
                pass
            plt.figure(figsize=(14, 3.5))
            plt.plot(s_mean.index, s_mean.values, label=f"{c} rolling pred mean")
            if not s_std.empty:
                low = s_mean.values - s_std.reindex(s_mean.index).values
                high = s_mean.values + s_std.reindex(s_mean.index).values
                plt.fill_between(s_mean.index, low, high, alpha=0.2, label="±1 std")
            plt.title(f"Rolling predicted mean ± std ({c})")
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # ---- 8) Train-Val diffs: Spearman and RMSE (fixed logic using folds) ----
    if not folds.empty:
        # Helper to find train/val ic columns for a country
        def find_col(prefixes):
            for p in prefixes:
                if p in folds.columns:
                    return p
            return None

        # find columns for train_ic and val_ic per country (many possible names)
        train_ic_de = find_col(["train_ic_pre_exp_de", "train_ic_de", "train_ic_pre_exp_de"])
        val_ic_de = find_col(["val_ic_pre_exp_de", "val_ic_de", "val_ic_pre_exp_de"])
        train_ic_fr = find_col(["train_ic_pre_exp_fr", "train_ic_fr", "train_ic_pre_exp_fr"])
        val_ic_fr = find_col(["val_ic_pre_exp_fr", "val_ic_fr", "val_ic_pre_exp_fr"])

        # compute diffs where possible (train - val)
        has_de_ic = (train_ic_de is not None and val_ic_de is not None)
        has_fr_ic = (train_ic_fr is not None and val_ic_fr is not None)

        if has_de_ic or has_fr_ic:
            plt.figure(figsize=(14, 4))
            if has_fr_ic:
                y_fr = folds[train_ic_fr] - folds[val_ic_fr]
                plt.plot(pd.to_datetime(folds["train_until"]), y_fr, label="FR (train - val) Spearman")
            if has_de_ic:
                y_de = folds[train_ic_de] - folds[val_ic_de]
                plt.plot(pd.to_datetime(folds["train_until"]), y_de, label="DE (train - val) Spearman")
            plt.axhline(0, color="k", lw=0.6)
            plt.title("Train - Val Spearman IC difference (per-country)")
            plt.xlabel("train_until"); plt.ylabel("train - val Spearman")
            plt.legend(); plt.grid(True); plt.xticks(rotation=30); plt.tight_layout(); plt.show()

        # RMSE diff (val - train) similar detection
        train_rmse_de = find_col(["train_rmse_pre_exp_de", "train_rmse_de", "train_rmse_pre_exp_de"])
        val_rmse_de = find_col(["val_rmse_pre_exp_de", "val_rmse_de", "val_rmse_pre_exp_de"])
        train_rmse_fr = find_col(["train_rmse_pre_exp_fr", "train_rmse_fr", "train_rmse_pre_exp_fr"])
        val_rmse_fr = find_col(["val_rmse_pre_exp_fr", "val_rmse_fr", "val_rmse_pre_exp_fr"])

        has_de_rmse = (train_rmse_de is not None and val_rmse_de is not None)
        has_fr_rmse = (train_rmse_fr is not None and val_rmse_fr is not None)

        if has_de_rmse or has_fr_rmse:
            plt.figure(figsize=(14, 4))
            if has_fr_rmse:
                y_fr = folds[val_rmse_fr] - folds[train_rmse_fr]
                plt.plot(pd.to_datetime(folds["train_until"]), y_fr, label="FR (val - train) RMSE")
            if has_de_rmse:
                y_de = folds[val_rmse_de] - folds[train_rmse_de]
                plt.plot(pd.to_datetime(folds["train_until"]), y_de, label="DE (val - train) RMSE")
            plt.axhline(0, color="k", lw=0.6)
            plt.title("Val - Train RMSE difference (per-country)")
            plt.xlabel("train_until"); plt.ylabel("Val - Train RMSE")
            plt.legend(); plt.grid(True); plt.xticks(rotation=30); plt.tight_layout(); plt.show()

        # Print two summary numbers (if present) as in original code
        try:
            if has_fr_ic:
                diff_fr = float(np.nanmean(folds[train_ic_fr]) - np.nanmean(folds[val_ic_fr]))
                print("diff. IC train-val, FR:", round(diff_fr, 3))
            if has_de_ic:
                diff_de = float(np.nanmean(folds[train_ic_de]) - np.nanmean(folds[val_ic_de]))
                print("diff. IC train-val, DE:", round(diff_de, 3))
        except Exception:
            pass

    # ---- 9) Summary printout ----
    print("=== SUMMARY ===")
    if "pooled_ic" in overall:
        print("Pooled IC:", overall.get("pooled_ic"))
    if "pooled_ic_demeaned" in overall:
        print("Pooled IC (demeaned):", overall.get("pooled_ic_demeaned"))
    if "per_country_ic" in overall:
        print("Per-country IC:", overall.get("per_country_ic"))
    print("Mean fold ICs (DE/FR):", overall.get("mean_fold_ic_de"), overall.get("mean_fold_ic_fr"))
    print("Mean RMSE (fold mean):", overall.get("mean_rmse_fold"))
    print("Pooled RMSE / MAE:", overall.get("pooled_rmse"), overall.get("pooled_mae"))

    return None
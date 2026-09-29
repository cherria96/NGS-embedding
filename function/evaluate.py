"""Pooled/site-balanced metrics and site-bootstrap paired comparisons.

Design-spec item 14 (evaluation) / item 7 (reporting conventions). Per-site
R^2 is never computed/averaged (sites have 1-4 observations)."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Metrics:
    n: int
    n_sites: int
    pooled_rmse: float
    pooled_mae: float
    pooled_r2: float
    site_balanced_mse: float
    site_balanced_rmse: float
    site_balanced_mae: float
    per_site: pd.DataFrame  # site_id, n, mse, mae


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, site_ids: List[str]) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    n = len(y_true)
    pooled_rmse = float(np.sqrt(np.mean(resid ** 2))) if n > 0 else float("nan")
    pooled_mae = float(np.mean(np.abs(resid))) if n > 0 else float("nan")
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if n > 1 else float("nan")
    pooled_r2 = 1.0 - ss_res / ss_tot if ss_tot not in (0.0, float("nan")) and not np.isnan(ss_tot) else float("nan")

    df = pd.DataFrame({"site_id": site_ids, "resid": resid, "abs_resid": np.abs(resid)})
    per_site = df.groupby("site_id").agg(n=("resid", "size"), mse=("resid", lambda x: float(np.mean(x ** 2))),
                                          mae=("abs_resid", "mean")).reset_index()
    site_balanced_mse = float(per_site["mse"].mean()) if len(per_site) else float("nan")
    site_balanced_mae = float(per_site["mae"].mean()) if len(per_site) else float("nan")
    site_balanced_rmse = float(np.sqrt(site_balanced_mse)) if not np.isnan(site_balanced_mse) else float("nan")

    return Metrics(
        n=n, n_sites=per_site.shape[0], pooled_rmse=pooled_rmse, pooled_mae=pooled_mae, pooled_r2=pooled_r2,
        site_balanced_mse=site_balanced_mse, site_balanced_rmse=site_balanced_rmse, site_balanced_mae=site_balanced_mae,
        per_site=per_site,
    )


def coverage(n_scorable: int, n_total: int) -> float:
    return float(n_scorable) / float(n_total) if n_total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Classification (item 7's logistic-loss extension)
# ---------------------------------------------------------------------------

def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)


def per_sample_log_loss(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    p = _clip_prob(np.asarray(prob, dtype=float))
    y = np.asarray(y_true, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def select_threshold_on_training(y_train: np.ndarray, prob_train: np.ndarray, metric: str = "mcc") -> float:
    """Item 7: 'Select the decision threshold using training data only.'
    Scans candidate thresholds at the observed training probabilities and
    picks the one maximizing MCC (ties broken toward 0.5). Falls back to
    0.5 if the training fold has a single class (MCC undefined)."""
    from sklearn.metrics import matthews_corrcoef, fbeta_score

    y_train = np.asarray(y_train, dtype=float)
    prob_train = np.asarray(prob_train, dtype=float)
    if len(np.unique(y_train)) < 2:
        return 0.5
    candidates = sorted(set(np.clip(prob_train, 1e-9, 1 - 1e-9).tolist()) | {0.5})
    best_thr, best_score = 0.5, -np.inf
    for thr in candidates:
        pred = (prob_train >= thr).astype(float)
        if metric == "mcc":
            score = matthews_corrcoef(y_train, pred) if len(np.unique(pred)) > 1 else -1.0
        elif metric == "f2":
            score = fbeta_score(y_train, pred, beta=2.0, zero_division=0)
        else:
            raise ValueError(f"unknown threshold-selection metric {metric!r}")
        if score > best_score or (score == best_score and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score, best_thr = score, thr
    return float(best_thr)


def select_threshold_via_inner_oof_f2(y: np.ndarray, prob: np.ndarray, site_ids: List[str],
                                       grid: Optional[List[float]] = None) -> float:
    """Item 14's classification threshold rule: choose among a fixed grid
    using the SELECTED candidate's pooled inner cross-validation
    out-of-fold probabilities (never outer/test data), with equal-site
    weighted confusion counts, breaking F2 ties toward the LARGER
    threshold. Freeze the result before outer prediction."""
    from datasets.datapreprocess.preprocessing import site_balanced_weights

    grid = grid or [round(0.05 * i, 2) for i in range(1, 20)]
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    weights = site_balanced_weights(list(site_ids))
    best_thr, best_f2 = grid[0], -1.0
    for thr in grid:
        pred = (prob >= thr).astype(float)
        tp = float(np.sum(weights * ((pred == 1) & (y == 1))))
        fp = float(np.sum(weights * ((pred == 1) & (y == 0))))
        fn = float(np.sum(weights * ((pred == 0) & (y == 1))))
        beta2 = 4.0
        denom = (1 + beta2) * tp + beta2 * fn + fp
        f2 = (1 + beta2) * tp / denom if denom > 0 else 0.0
        if f2 >= best_f2:  # ascending grid + >= means a later (larger) threshold wins ties
            best_f2, best_thr = f2, thr
    return float(best_thr)


@dataclasses.dataclass
class ClassificationMetrics:
    n: int
    n_sites: int
    n_positive: int
    pooled_auc_roc: float
    pooled_auc_pr: float
    pooled_brier: float
    site_balanced_log_loss: float
    threshold_note: str
    precision: float
    recall: float
    f1: float
    f2: float
    mcc: float
    accuracy: float
    per_site: pd.DataFrame  # site_id, n, log_loss


def compute_classification_metrics(y_true: np.ndarray, prob_pred: np.ndarray, site_ids: List[str],
                                    predicted_class: np.ndarray) -> ClassificationMetrics:
    from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                                  matthews_corrcoef, fbeta_score, precision_score, recall_score,
                                  f1_score, accuracy_score)

    y_true = np.asarray(y_true, dtype=float)
    prob_pred = np.asarray(prob_pred, dtype=float)
    predicted_class = np.asarray(predicted_class, dtype=float)
    n = len(y_true)
    n_pos = int(np.sum(y_true))

    if 0 < n_pos < n:
        pooled_auc_roc = float(roc_auc_score(y_true, prob_pred))
        pooled_auc_pr = float(average_precision_score(y_true, prob_pred))
    else:
        pooled_auc_roc = float("nan")
        pooled_auc_pr = float("nan")
    pooled_brier = float(brier_score_loss(y_true, prob_pred))

    ll = per_sample_log_loss(y_true, prob_pred)
    df = pd.DataFrame({"site_id": site_ids, "log_loss": ll})
    per_site = df.groupby("site_id").agg(n=("log_loss", "size"), log_loss=("log_loss", "mean")).reset_index()
    site_balanced_log_loss = float(per_site["log_loss"].mean()) if len(per_site) else float("nan")

    mcc = float(matthews_corrcoef(y_true, predicted_class)) if len(np.unique(predicted_class)) > 1 and 0 < n_pos < n else float("nan")
    return ClassificationMetrics(
        n=n, n_sites=per_site.shape[0], n_positive=n_pos,
        pooled_auc_roc=pooled_auc_roc, pooled_auc_pr=pooled_auc_pr, pooled_brier=pooled_brier,
        site_balanced_log_loss=site_balanced_log_loss,
        threshold_note="predicted_class uses each pooled sample's own fold-specific, training-selected threshold",
        precision=float(precision_score(y_true, predicted_class, zero_division=0)),
        recall=float(recall_score(y_true, predicted_class, zero_division=0)),
        f1=float(f1_score(y_true, predicted_class, zero_division=0)),
        f2=float(fbeta_score(y_true, predicted_class, beta=2.0, zero_division=0)),
        mcc=mcc, accuracy=float(accuracy_score(y_true, predicted_class)),
        per_site=per_site,
    )


def bootstrap_paired_site_log_loss_differences(oof_a: pd.DataFrame, oof_b: pd.DataFrame, B: int = 1000,
                                                seed: int = 0) -> dict:
    """Classification analogue of bootstrap_paired_site_differences: paired
    difference in site-balanced log-loss. oof_*: columns sample_id,
    site_id, target, prediction (prediction = predicted probability)."""
    common_sites = sorted(set(oof_a["site_id"]).intersection(oof_b["site_id"]))
    if not common_sites:
        return {"note": "no common scorable sites between the two models", "diffs": []}

    def site_ll(df, site):
        sub = df[df["site_id"] == site]
        return float(np.mean(per_sample_log_loss(sub["target"].to_numpy(), sub["prediction"].to_numpy())))

    site_ll_a = {s: site_ll(oof_a, s) for s in common_sites}
    site_ll_b = {s: site_ll(oof_b, s) for s in common_sites}
    observed_diff = float(np.mean([site_ll_a[s] - site_ll_b[s] for s in common_sites]))

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        draw = rng.choice(common_sites, size=len(common_sites), replace=True)
        d = np.mean([site_ll_a[s] - site_ll_b[s] for s in draw])
        diffs.append(float(d))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_sites": len(common_sites), "n_bootstrap": B,
        "observed_mean_site_logloss_diff_a_minus_b": observed_diff,
        "ci95_low": float(lo), "ci95_high": float(hi),
        "note": "conditional on fitted OOF models; repeated CV folds are not independent samples",
    }


def bootstrap_paired_site_differences(oof_a: pd.DataFrame, oof_b: pd.DataFrame, B: int = 1000,
                                       seed: int = 0) -> dict:
    """Bootstrap sites (with replacement) for the difference in site-balanced
    MSE between two models' stored OOF predictions, using the same sampled
    sites for both. oof_*: DataFrame with columns sample_id, site_id,
    target, prediction (scorable rows only). Intervals are conditional on
    the fitted OOF models (item 14: they omit part of retraining
    uncertainty; repeated folds are not independent for an ordinary
    t-test)."""
    common_sites = sorted(set(oof_a["site_id"]).intersection(oof_b["site_id"]))
    if not common_sites:
        return {"note": "no common scorable sites between the two models", "diffs": []}

    def site_mse(df, site):
        sub = df[df["site_id"] == site]
        return float(np.mean((sub["target"] - sub["prediction"]) ** 2))

    site_mse_a = {s: site_mse(oof_a, s) for s in common_sites}
    site_mse_b = {s: site_mse(oof_b, s) for s in common_sites}
    observed_diff = float(np.mean([site_mse_a[s] - site_mse_b[s] for s in common_sites]))

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        draw = rng.choice(common_sites, size=len(common_sites), replace=True)
        d = np.mean([site_mse_a[s] - site_mse_b[s] for s in draw])
        diffs.append(float(d))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_sites": len(common_sites), "n_bootstrap": B,
        "observed_mean_site_mse_diff_a_minus_b": observed_diff,
        "ci95_low": float(lo), "ci95_high": float(hi),
        "note": "conditional on fitted OOF models; repeated CV folds are not independent samples",
    }

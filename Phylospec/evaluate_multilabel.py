#!/usr/bin/env python3
"""
evaluate_multilabel.py -- evaluation protocol for the five AD warning flags.

The task is five binary targets with prevalences spanning 4/140 to 26/140, on 140
samples drawn from only 36 independent sites. Three consequences drive everything
here:

  * A single 80/20 split cannot evaluate the rare flags. Positives per flag:
    ammonia_toxicity 26, acid_base_balance 21, buffer_capacity 9,
    biogas_quality 7, acid_accumulation 4. Repeated site-grouped CV instead.
  * A 0.5 decision threshold is meaningless at 3% prevalence. Tune one threshold
    per flag, for MCC, on data the outer test fold never saw.
  * Accuracy and micro-averages are dominated by ammonia_toxicity. Report
    per-flag first, macro-average second.

Usage:

    from evaluate_multilabel import (
        site_grouped_cv, pos_weight_from_labels, tune_thresholds_mcc,
        score_multilabel, evaluate_cv, LABEL_COLS)

    res = evaluate_cv(Y, sites, fit_predict, label_names=LABEL_COLS)
    print(res["per_label"]); print(res["macro"])
"""
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (average_precision_score, fbeta_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

LABEL_COLS = [
    "warning_acid_base_balance",
    "warning_buffer_capacity",
    "warning_acid_accumulation",
    "warning_ammonia_toxicity",
    "warning_biogas_quality",
]


# ------------------------------------------------------------------ splitting
def site_grouped_cv(y, sites, n_splits=5, n_repeats=5, seed=42):
    """Repeated multilabel-stratified k-fold over SITES, never over samples.

    The 140 samples are 36 sites sampled up to four seasons each. Two seasons of
    the same digester share a microbial community lineage and an operator, so a
    sample-level split leaks: the model can recognise the site rather than the
    instability. Folds are built on sites and expanded to samples.

    Stratification uses each site's label vector aggregated by max (a site counts
    as positive for a flag if any of its seasons was flagged), which is what keeps
    the rare flags from landing entirely in one fold.

    Yields (train_idx, test_idx) sample-index arrays, n_splits * n_repeats times.
    """
    y = np.asarray(y)
    sites = np.asarray(sites)
    uniq = np.unique(sites)
    site_rows = {s: np.flatnonzero(sites == s) for s in uniq}
    site_y = np.array([y[site_rows[s]].max(axis=0) for s in uniq])

    for rep in range(n_repeats):
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True,
                                         random_state=seed + rep)
        for tr_s, te_s in mskf.split(np.zeros((len(uniq), 1)), site_y):
            tr = np.concatenate([site_rows[uniq[i]] for i in tr_s])
            te = np.concatenate([site_rows[uniq[i]] for i in te_s])
            yield np.sort(tr), np.sort(te)


def split_train_val(y_train, sites_train, val_frac=0.2, seed=0):
    """Inner site-grouped split of the training fold, for early stopping and for
    threshold tuning. Must come out of the training sites only -- tuning a
    threshold on the outer test fold is the single easiest way to invent a result.
    """
    n_splits = max(2, int(round(1.0 / val_frac)))
    for tr, va in site_grouped_cv(y_train, sites_train, n_splits=n_splits,
                                  n_repeats=1, seed=seed):
        return tr, va


# --------------------------------------------------------------- class balance
def pos_weight_from_labels(y_train, cap=None):
    """pos_weight for torch.nn.BCEWithLogitsLoss, as a 5-element tensor.

    BCEWithLogitsLoss multiplies the positive term of label j by pos_weight[j], so
    the conventional value is n_negative / n_positive computed PER LABEL. Computed
    on the training fold only -- deriving it from the full dataset leaks the test
    fold's class balance.

    Full-dataset values for reference (they will differ per fold):
        acid_base_balance 119/21 = 5.7   buffer_capacity 131/9  = 14.6
        acid_accumulation 136/4  = 34.0  ammonia_toxicity 114/26 = 4.4
        biogas_quality    133/7  = 19.0

    A weight of 34 on four positives makes the gradient for acid_accumulation
    swamp the other four flags and is a common cause of the loss diverging or the
    model predicting that flag everywhere. Pass cap=10 (or cap="sqrt", which uses
    sqrt(neg/pos)) and report which you used -- this is a modelling choice, not a
    detail. Returns float64 array of shape (n_labels,).
    """
    y = np.asarray(y_train)
    pos = y.sum(axis=0).astype(float)
    neg = y.shape[0] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    if cap == "sqrt":
        w = np.sqrt(w)
    elif cap is not None:
        w = np.minimum(w, float(cap))
    return w


# ------------------------------------------------------------------ thresholds
def tune_thresholds_mcc(y_true, y_score, n_grid=200):
    """One decision threshold per label, chosen to maximise MCC on the data given.

    At 3-19% prevalence the default 0.5 sits far from the operating point that
    balances the confusion matrix, and it sits in a *different* place for each of
    the five flags -- so a single global threshold cannot be right for more than
    one of them. MCC is the right objective to tune against here because it is the
    one common binary summary that stays honest when the classes are this skewed:
    it uses all four cells of the confusion matrix, so a model that predicts the
    majority class everywhere scores 0, not 0.9.

    Call this on VALIDATION scores, apply the returned thresholds to test scores.
    Labels with no positives in y_true get 0.5 and should be excluded downstream.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    out = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        yt, ys = y_true[:, j], y_score[:, j]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        grid = np.unique(np.quantile(ys, np.linspace(0.01, 0.99, n_grid)))
        best, best_t = -2.0, 0.5
        for t in grid:
            m = matthews_corrcoef(yt, (ys >= t).astype(int))
            if m > best:
                best, best_t = m, float(t)
        out[j] = best_t
    return out


# --------------------------------------------------------------------- scoring
def score_multilabel(y_true, y_score, thresholds, label_names=None, beta=2.0):
    """Per-label and macro-averaged metrics.

    Threshold-free: AUPRC (average precision) with the label's prevalence as its
    own baseline -- an AUPRC of 0.15 is strong at 4% prevalence and worthless at
    20%, so the lift over baseline is the comparable number, not AUPRC itself.
    AUROC is reported too but is optimistic under this much imbalance.

    Thresholded at the tuned operating point: MCC, F_beta (beta=2 by default, which
    weights recall 4x precision -- for a digester warning system a missed
    instability costs far more than an unnecessary inspection), precision, recall.

    Macro-averages are unweighted means over labels that have at least one positive
    AND one negative in y_true; labels that do not are reported as NaN and left out
    of the macro, with n_labels_scored recording how many contributed. An
    unweighted macro is what makes the four-positive flag count as much as the
    twenty-six-positive one, which is the point -- but it also makes the macro
    noisy, so always read it beside the per-label table.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    thresholds = np.asarray(thresholds)
    names = list(label_names) if label_names is not None else \
        [f"label_{j}" for j in range(y_true.shape[1])]
    rows = []
    for j, nm in enumerate(names):
        yt, ys = y_true[:, j], y_score[:, j]
        npos = int(yt.sum())
        if npos == 0 or npos == len(yt):
            rows.append({"label": nm, "n_pos": npos, "prevalence": npos / len(yt),
                         "auprc": np.nan, "auprc_lift": np.nan, "auroc": np.nan,
                         "mcc": np.nan, f"f{beta:g}": np.nan, "precision": np.nan,
                         "recall": np.nan, "threshold": float(thresholds[j])})
            continue
        yp = (ys >= thresholds[j]).astype(int)
        prev = npos / len(yt)
        ap = average_precision_score(yt, ys)
        rows.append({
            "label": nm, "n_pos": npos, "prevalence": prev,
            "auprc": ap, "auprc_lift": ap / prev, "auroc": roc_auc_score(yt, ys),
            "mcc": matthews_corrcoef(yt, yp),
            f"f{beta:g}": fbeta_score(yt, yp, beta=beta, zero_division=0),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "threshold": float(thresholds[j]),
        })
    keys = ["auprc", "auprc_lift", "auroc", "mcc", f"f{beta:g}", "precision", "recall"]
    scored = [r for r in rows if not np.isnan(r["mcc"])]
    macro = {f"macro_{k}": float(np.mean([r[k] for r in scored])) if scored else np.nan
             for k in keys}
    macro["n_labels_scored"] = len(scored)
    return {"per_label": rows, "macro": macro}


# ------------------------------------------------------------------- driver
def evaluate_cv(y, sites, fit_predict, label_names=None, n_splits=5, n_repeats=5,
                seed=42, beta=2.0, pos_weight_cap=10.0):
    """Full protocol: site-grouped CV, pooled out-of-fold scoring, repeated.

    `fit_predict(train_idx, val_idx, test_idx, pos_weight) -> (val_scores, test_scores)`
    returns arrays of shape (len(val_idx), 5) and (len(test_idx), 5).

    Metrics are computed on predictions POOLED across the folds of one repeat, not
    fold by fold. This matters here and is not a cosmetic choice: site-grouped
    5-fold puts ~7 of the 36 sites in each test fold, so a flag whose positives sit
    in one or two sites is absent from most test folds and cannot be scored in
    them. Scoring per fold and averaging then throws away most of the evidence --
    in a synthetic cohort with this exact design, acid_accumulation was scorable in
    5 of 25 folds and buffer_capacity in 15. Because k-fold predicts every sample
    exactly once per repeat, pooling gives one complete 140-sample out-of-fold
    evaluation per repeat in which every flag carries its full positive count. The
    n_repeats repeats then give the spread.

    Thresholds are tuned once per repeat on pooled inner-validation scores and
    applied to the pooled out-of-fold test scores, so no test prediction
    contributes to the operating point that scores it.

    Returns per-label means with 95% percentile intervals across repeats, the
    macro-average, and the per-repeat records.
    """
    y = np.asarray(y)
    sites = np.asarray(sites)
    names = list(label_names) if label_names is not None else LABEL_COLS
    n, L = y.shape
    repeats = []

    for rep in range(n_repeats):
        oof = np.full((n, L), np.nan)
        val_y, val_s = [], []
        gen = site_grouped_cv(y, sites, n_splits, n_repeats=1, seed=seed + rep)
        for k, (tr_all, te) in enumerate(gen):
            tr_rel, va_rel = split_train_val(y[tr_all], sites[tr_all], seed=seed + 100 * rep + k)
            tr, va = tr_all[tr_rel], tr_all[va_rel]
            pw = pos_weight_from_labels(y[tr], cap=pos_weight_cap)
            v_s, t_s = fit_predict(tr, va, te, pw)
            oof[te] = t_s
            val_y.append(y[va]); val_s.append(np.asarray(v_s))
        thr = tune_thresholds_mcc(np.vstack(val_y), np.vstack(val_s))
        covered = ~np.isnan(oof).any(axis=1)
        repeats.append(score_multilabel(y[covered], oof[covered], thr, names, beta=beta))

    metric_keys = ["auprc", "auprc_lift", "auroc", "mcc", f"f{beta:g}", "precision",
                   "recall", "threshold"]
    per_label = []
    for j, nm in enumerate(names):
        agg = {"label": nm, "n_pos": repeats[0]["per_label"][j]["n_pos"],
               "prevalence": repeats[0]["per_label"][j]["prevalence"]}
        for k in metric_keys:
            v = np.array([r["per_label"][j][k] for r in repeats], dtype=float)
            v = v[~np.isnan(v)]
            agg[k] = float(v.mean()) if v.size else np.nan
            agg[k + "_lo"] = float(np.percentile(v, 2.5)) if v.size else np.nan
            agg[k + "_hi"] = float(np.percentile(v, 97.5)) if v.size else np.nan
        agg["n_repeats_scored"] = int(sum(
            not np.isnan(r["per_label"][j]["mcc"]) for r in repeats))
        per_label.append(agg)

    macro = {}
    for k in repeats[0]["macro"]:
        v = np.array([r["macro"][k] for r in repeats], dtype=float)
        v = v[~np.isnan(v)]
        macro[k] = float(v.mean()) if v.size else np.nan
        macro[k + "_lo"] = float(np.percentile(v, 2.5)) if v.size else np.nan
        macro[k + "_hi"] = float(np.percentile(v, 97.5)) if v.size else np.nan
    return {"per_label": per_label, "macro": macro, "n_repeats": n_repeats,
            "repeats": repeats}


def audit_site_label_support(y, sites, label_names=None):
    """How many SITES carry a positive for each flag.

    This is the number that decides whether the flag is evaluable at all. With
    site-grouped folds, a flag whose four positives all come from one site can
    only ever appear in one test fold -- every other fold scores it NaN. Run this
    before committing to n_splits and report it in the paper.
    """
    y, sites = np.asarray(y), np.asarray(sites)
    names = list(label_names) if label_names is not None else LABEL_COLS
    uniq = np.unique(sites)
    site_y = np.array([y[sites == s].max(axis=0) for s in uniq])
    return [{"label": nm, "n_pos_samples": int(y[:, j].sum()),
             "n_pos_sites": int(site_y[:, j].sum()), "n_sites": len(uniq)}
            for j, nm in enumerate(names)]

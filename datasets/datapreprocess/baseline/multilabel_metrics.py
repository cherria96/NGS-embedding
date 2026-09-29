"""Shared evaluation for multi-label (5-flag) binary warning prediction."""
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def predict_proba_matrix(clf, X):
    """Normalise the two shapes sklearn classifiers return P(class=1) in for a
    multi-label target: MultiOutputClassifier / natively-multi-output
    classifiers (RandomForest, KNeighbors) return a list of (n_samples,
    n_classes) arrays, one per label; MLPClassifier returns a single
    (n_samples, n_labels) array directly."""
    proba = clf.predict_proba(X)
    if isinstance(proba, list):
        return np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba])
    return proba


def compute_multilabel_metrics(y_true, y_prob, flag_names, threshold=0.5):
    """y_true, y_prob: (n_samples, n_flags) arrays. Returns per-flag metrics and macro averages."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    per_flag = {}
    for i, flag in enumerate(flag_names):
        yt, yp, ys = y_true[:, i], y_pred[:, i], y_prob[:, i]
        acc = accuracy_score(yt, yp)
        mcc = matthews_corrcoef(yt, yp) if len(np.unique(yp)) > 1 else 0.0
        if len(np.unique(yt)) > 1:
            roc_auc = roc_auc_score(yt, ys)
            aupr = average_precision_score(yt, ys)
        else:
            roc_auc = float("nan")
            aupr = float("nan")
        per_flag[flag] = {"accuracy": acc, "mcc": mcc, "roc_auc": roc_auc, "aupr": aupr}

    macro = {}
    for metric in ("accuracy", "mcc", "roc_auc", "aupr"):
        vals = [per_flag[f][metric] for f in flag_names]
        macro[metric] = float(np.nanmean(vals))

    return {"per_flag": per_flag, "macro": macro}


def save_results(result, model_name, domain, out_path):
    payload = {"model": model_name, "domain": domain, **result}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics -> {out_path}")
    print(json.dumps(payload["macro"], indent=2))

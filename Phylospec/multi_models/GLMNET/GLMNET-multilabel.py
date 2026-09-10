"""
GLMNET-equivalent baseline (elastic-net-penalised logistic regression, the
standard Python stand-in for R's glmnet) for multi-label warning prediction (5
flags at once). Same CLI/CSV contract as RF-multilabel.py. Like XGBoost-
multilabel.py, this fits 5 SEPARATE LogisticRegression(penalty="elasticnet")
models -- one per flag -- each with its own class_weight={0: 1, 1: min(n_neg/
n_pos, --pos-weight-cap)} computed from that flag's TRAINING fold, so
--pos-weight-cap is actually used (task Sec 5A.4) rather than accepted-and-
ignored the way RF/SVM's class_weight="balanced" does.
"""
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_preprocessing"))
from multilabel_metrics import compute_multilabel_metrics, save_results

FLAGS = [
    "acid_accumulation",
    "acid_base_balance",
    "ammonia_toxicity",
    "biogas_quality",
    "buffer_capacity",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def capped_class_weight(y_col, cap):
    n_pos = int(y_col.sum())
    n_neg = len(y_col) - n_pos
    if n_pos == 0:
        return {0: 1.0, 1: 1.0}
    w = n_neg / n_pos
    return {0: 1.0, 1: min(w, cap) if cap else w}


def fit_predict_all_flags(X_train, y_train, X_eval, pos_weight_cap, seed):
    probs = np.zeros((X_eval.shape[0], len(FLAGS)), dtype=np.float64)
    for i in range(len(FLAGS)):
        cw = capped_class_weight(y_train[:, i], pos_weight_cap)
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight=cw,
            max_iter=5000,
            random_state=seed,
        )
        clf.fit(X_train, y_train[:, i])
        probs[:, i] = clf.predict_proba(X_eval)[:, 1]
    return probs


def main(train_csv, test_csv, out_path, domain, seed=42, val_csv=None,
         scores_out_dir=None, pos_weight_cap=10.0):
    set_seed(seed)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=["SampleID"] + FLAGS).values
    y_train = train_df[FLAGS].values.astype(int)
    X_test = test_df.drop(columns=["SampleID"] + FLAGS).values
    y_test = test_df[FLAGS].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_prob = fit_predict_all_flags(X_train, y_train, X_test, pos_weight_cap, seed)

    if scores_out_dir:
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_csv:
            val_df = pd.read_csv(val_csv)
            X_val = scaler.transform(val_df.drop(columns=["SampleID"] + FLAGS).values)
            y_val = val_df[FLAGS].values
            val_prob = fit_predict_all_flags(X_train, y_train, X_val, pos_weight_cap, seed)
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), val_prob)
            np.save(os.path.join(scores_out_dir, "val_y.npy"), y_val)
        return

    result = compute_multilabel_metrics(y_test, y_prob, FLAGS)
    save_results(result, "GLMNET", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label GLMNET (elastic-net logistic regression).")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", default=None,
                         help="required unless --scores-out-dir is given")
    parser.add_argument("--val", default=None,
                         help="inner-validation CSV; with --scores-out-dir, scores "
                              "are computed for this split too")
    parser.add_argument("--scores-out-dir", default=None,
                         help="dump raw val/test score arrays here instead of "
                              "computing metrics inline (site-grouped CV path)")
    parser.add_argument("--pos-weight-cap", type=float, default=10.0,
                         help="cap on per-flag class_weight[1] = n_neg/n_pos, "
                              "computed on the training fold only (Sec 5A.4)")
    args = parser.parse_args()
    if not args.scores_out_dir and not args.out:
        parser.error("--out is required unless --scores-out-dir is given")
    main(args.train, args.test, args.out, args.domain,
         val_csv=args.val, scores_out_dir=args.scores_out_dir,
         pos_weight_cap=args.pos_weight_cap)

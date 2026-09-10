"""
Random Forest baseline for multi-label warning prediction (5 flags at once).

Unlike RF-multi.py (single multi-class "Group" column, internal 5-fold CV),
this script consumes an explicit train/test split with one 0/1 column per
flag and trains a single RandomForestClassifier on the full label matrix --
scikit-learn natively supports multi-output classification by fitting one
tree ensemble per output column internally.
"""
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def main(train_csv, test_csv, out_path, domain, seed=42, val_csv=None, scores_out_dir=None):
    set_seed(seed)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns=["SampleID"] + FLAGS).values
    y_train = train_df[FLAGS].values
    X_test = test_df.drop(columns=["SampleID"] + FLAGS).values
    y_test = test_df[FLAGS].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        max_features="log2",
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    def predict_proba_matrix(X):
        # predict_proba on a multi-output classifier returns a list (one per
        # label) of (n_samples, n_classes) arrays; take the P(class=1) column
        # of each.
        proba_list = clf.predict_proba(X)
        return np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba_list])

    y_prob = predict_proba_matrix(X_test)

    if scores_out_dir:
        # Site-grouped CV path (evaluate_multilabel.evaluate_cv): dump raw,
        # unthresholded scores for both the inner-val and outer-test splits
        # instead of scoring inline -- threshold tuning (on val) and final
        # metrics (on pooled test) are centralized in
        # run_site_grouped_cv_benchmark.py so every model is scored identically.
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_csv:
            val_df = pd.read_csv(val_csv)
            X_val = scaler.transform(val_df.drop(columns=["SampleID"] + FLAGS).values)
            y_val = val_df[FLAGS].values
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), predict_proba_matrix(X_val))
            np.save(os.path.join(scores_out_dir, "val_y.npy"), y_val)
        return

    result = compute_multilabel_metrics(y_test, y_prob, FLAGS)
    save_results(result, "RF", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label Random Forest.")
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
    parser.add_argument("--pos-weight-cap", type=float, default=None,
                         help="accepted for CLI uniformity with the other 4 "
                              "model scripts; unused -- RF's class imbalance "
                              "handling is class_weight='balanced', not a "
                              "BCEWithLogitsLoss pos_weight")
    args = parser.parse_args()
    if not args.scores_out_dir and not args.out:
        parser.error("--out is required unless --scores-out-dir is given")
    main(args.train, args.test, args.out, args.domain,
         val_csv=args.val, scores_out_dir=args.scores_out_dir)

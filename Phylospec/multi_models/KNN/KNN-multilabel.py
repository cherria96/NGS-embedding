"""
K-Nearest-Neighbors baseline for multi-label warning prediction (5 flags at
once). Same CLI/CSV contract as RF-multilabel.py -- see that file's docstring
for the shape of --train/--val/--test and the --scores-out-dir site-grouped-CV
path. KNeighborsClassifier natively supports a 2D (multi-output) y, same as
RandomForestClassifier, so predict_proba_matrix's list-of-arrays branch applies.
"""
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_preprocessing"))
from multilabel_metrics import compute_multilabel_metrics, save_results, predict_proba_matrix

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

    n_neighbors = min(5, len(X_train) - 1) or 1
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    clf.fit(X_train, y_train)

    y_prob = predict_proba_matrix(clf, X_test)

    if scores_out_dir:
        os.makedirs(scores_out_dir, exist_ok=True)
        np.save(os.path.join(scores_out_dir, "test_scores.npy"), y_prob)
        np.save(os.path.join(scores_out_dir, "test_y.npy"), y_test)
        if val_csv:
            val_df = pd.read_csv(val_csv)
            X_val = scaler.transform(val_df.drop(columns=["SampleID"] + FLAGS).values)
            y_val = val_df[FLAGS].values
            np.save(os.path.join(scores_out_dir, "val_scores.npy"), predict_proba_matrix(clf, X_val))
            np.save(os.path.join(scores_out_dir, "val_y.npy"), y_val)
        return

    result = compute_multilabel_metrics(y_test, y_prob, FLAGS)
    save_results(result, "KNN", domain, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-label KNN.")
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
                         help="accepted for CLI uniformity with the other model "
                              "scripts; unused -- KNN has no imbalance-weighting knob")
    args = parser.parse_args()
    if not args.scores_out_dir and not args.out:
        parser.error("--out is required unless --scores-out-dir is given")
    main(args.train, args.test, args.out, args.domain,
         val_csv=args.val, scores_out_dir=args.scores_out_dir)

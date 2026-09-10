"""
Build K-fold cross-validation splits for the multi-label warning benchmark.

The original example_train/example_test split is small (107-110 train, 31
test per domain) and some flags have only a handful of positives, so a
single held-out split is noisy. This pools train+test back together per
domain and re-splits with KFold, writing fold-specific train/test CSVs (for
RF/CNN/PMCNN/MetaDR) and fold-specific npy files (for DeepPhylo).

Sample order and feature order are identical between the CSV pipeline
(input_for_all_models/Warnings_<domain>/) and the DeepPhylo npy pipeline
(DeepPhylo/data/warnings_<domain>/) for all three domains (verified), so the
same KFold split indices are reused for both to keep folds identical across
all 5 models.
"""
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = os.path.join(HERE, "input_for_all_models")
DEEPPHYLO_DATA_DIR = os.path.join(HERE, "DeepPhylo", "data")
CV_DIR = os.path.join(HERE, "cv_folds")
DOMAINS = ["ARC", "BAC", "merged"]
N_SPLITS = 5
SEED = 42


def build_domain(domain, n_splits=N_SPLITS, seed=SEED):
    csv_dir = os.path.join(CSV_DATA_DIR, f"Warnings_{domain}")
    train_df = pd.read_csv(os.path.join(csv_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(csv_dir, "test.csv"))
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    dp_dir = os.path.join(DEEPPHYLO_DATA_DIR, f"warnings_{domain}")
    X_all = np.vstack([np.load(os.path.join(dp_dir, "X_train.npy")), np.load(os.path.join(dp_dir, "X_eval.npy"))])
    Y_all = np.vstack([np.load(os.path.join(dp_dir, "Y_train.npy")), np.load(os.path.join(dp_dir, "Y_eval.npy"))])
    assert len(combined_df) == len(X_all), f"{domain}: CSV rows ({len(combined_df)}) != npy rows ({len(X_all)})"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    out_csv_dir = os.path.join(CV_DIR, f"Warnings_{domain}")
    out_dp_dir = os.path.join(CV_DIR, f"DeepPhylo_warnings_{domain}")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(combined_df)):
        fold_csv_dir = os.path.join(out_csv_dir, f"fold_{fold_idx}")
        os.makedirs(fold_csv_dir, exist_ok=True)
        combined_df.iloc[train_idx].to_csv(os.path.join(fold_csv_dir, "train.csv"), index=False)
        combined_df.iloc[test_idx].to_csv(os.path.join(fold_csv_dir, "test.csv"), index=False)

        fold_dp_dir = os.path.join(out_dp_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dp_dir, exist_ok=True)
        np.save(os.path.join(fold_dp_dir, "X_train.npy"), X_all[train_idx])
        np.save(os.path.join(fold_dp_dir, "X_eval.npy"), X_all[test_idx])
        np.save(os.path.join(fold_dp_dir, "Y_train.npy"), Y_all[train_idx])
        np.save(os.path.join(fold_dp_dir, "Y_eval.npy"), Y_all[test_idx])
        shutil.copyfile(os.path.join(dp_dir, "c.npy"), os.path.join(fold_dp_dir, "c.npy"))
        shutil.copyfile(os.path.join(dp_dir, "label_names.txt"), os.path.join(fold_dp_dir, "label_names.txt"))

    print(f"[{domain}] built {n_splits} folds -> {out_csv_dir} and {out_dp_dir}")


if __name__ == "__main__":
    for domain in DOMAINS:
        build_domain(domain)

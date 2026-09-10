"""
build_site_grouped_cv_folds.py -- REPLACES build_cv_folds.py.

build_cv_folds.py uses plain sklearn `KFold(shuffle=True)` at the SAMPLE
level. That is a real correctness bug, not a style choice: the 140 samples
here are 36 sites sampled over up to four seasons, so a sample-level split
routinely puts two seasons of the same digester on opposite sides of a fold --
the model can then recognise the site's own microbial-community fingerprint
instead of the instability signal, and every number `results/cv/` currently
holds was produced this way (task Sec 5A.1).

This script instead calls evaluate_multilabel.site_grouped_cv() (already
implemented, do not reimplement), which stratifies by SITE using each site's
label vector aggregated with max, and additionally carves an INNER
train/validation split out of each outer-training fold via split_train_val()
-- needed because Sec 5A.3 requires decision thresholds to be tuned via MCC on
validation scores that never touch the outer test fold.

5 repeats x 5 outer folds = 25 (train, val, test) triples per domain, written
to cv_folds/Warnings_<domain>/rep<r>_fold<k>/{train,val,test}.csv for the
CSV-consuming models (RF/CNN/PMCNN/MetaDR) and
cv_folds/DeepPhylo_warnings_<domain>/rep<r>_fold<k>/{X,Y}_{train,val,eval}.npy
for DeepPhylo (kept the existing "eval" name for the outer-test split so
deepphylo_classification_multi_label_bce.py's --data_dir X_eval.npy/Y_eval.npy
convention needs no change; the new --val_dir points at the val split).

Requires: the ORIGINAL example_train.csv/example_test.csv split's SampleID
order to already match 1:1 with the DeepPhylo X_train.npy/X_eval.npy row
order (this was true of the pre-existing build_cv_folds.py and is verified
again here from sample_ids_{train,eval}.txt if present, rather than assumed).
"""
import csv
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(PHYLOSPEC_DIR, "..")
CSV_DATA_DIR = os.path.join(HERE, "input_for_all_models")
DEEPPHYLO_DATA_DIR = os.path.join(HERE, "DeepPhylo", "data")
CV_DIR = os.path.join(HERE, "cv_folds_site_grouped")
METADATA_PATH = os.path.join(ROOT, "data", "final", "metadata.csv")
DOMAINS = ["ARC", "BAC", "merged"]
N_SPLITS = 5
N_REPEATS = 5
SEED = 42

sys.path.insert(0, PHYLOSPEC_DIR)
from evaluate_multilabel import LABEL_COLS, site_grouped_cv, split_train_val  # noqa: E402

FLAG_COLS = [c.replace("warning_", "") for c in LABEL_COLS]


def load_site_map():
    site_map = {}
    with open(METADATA_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            site_map[row["SampleID"]] = row["Site"]
    return site_map


def _verify_row_alignment(combined_df, dp_dir):
    """The DeepPhylo X_all/Y_all built below assumes row i of combined_df is
    row i of X_all -- check that against sample_ids_{train,eval}.txt if they
    exist (written by prepare_warning_targets_data.write_deepphylo_npys),
    instead of silently trusting it."""
    train_ids_path = os.path.join(dp_dir, "sample_ids_train.txt")
    eval_ids_path = os.path.join(dp_dir, "sample_ids_eval.txt")
    if not (os.path.exists(train_ids_path) and os.path.exists(eval_ids_path)):
        return
    dp_ids = (open(train_ids_path).read().split() + open(eval_ids_path).read().split())
    csv_ids = combined_df["SampleID"].astype(str).tolist()
    assert dp_ids == csv_ids, (
        "DeepPhylo npy sample order != combined CSV sample order -- the "
        "assumption that row i of X_all/Y_all is row i of combined_df is "
        "false for this domain; fix the alignment before trusting any fold "
        "built here")


def build_domain(domain, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    csv_dir = os.path.join(CSV_DATA_DIR, f"Warnings_{domain}")
    train_df = pd.read_csv(os.path.join(csv_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(csv_dir, "test.csv"))
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    dp_dir = os.path.join(DEEPPHYLO_DATA_DIR, f"warnings_{domain}")
    X_all = np.vstack([np.load(os.path.join(dp_dir, "X_train.npy")),
                       np.load(os.path.join(dp_dir, "X_eval.npy"))])
    Y_all = np.vstack([np.load(os.path.join(dp_dir, "Y_train.npy")),
                       np.load(os.path.join(dp_dir, "Y_eval.npy"))])
    assert len(combined_df) == len(X_all), (
        f"{domain}: CSV rows ({len(combined_df)}) != npy rows ({len(X_all)})")
    _verify_row_alignment(combined_df, dp_dir)

    site_map = load_site_map()
    sites = combined_df["SampleID"].astype(str).map(site_map).values
    missing_site = pd.isna(sites)
    assert not missing_site.any(), (
        f"{domain}: {missing_site.sum()} samples have no Site in "
        f"{METADATA_PATH}: {combined_df['SampleID'][missing_site].tolist()[:5]}")

    y_for_split = combined_df[FLAG_COLS].values

    out_csv_dir = os.path.join(CV_DIR, f"Warnings_{domain}")
    out_dp_dir = os.path.join(CV_DIR, f"DeepPhylo_warnings_{domain}")
    n_written = 0

    for rep in range(n_repeats):
        gen = site_grouped_cv(y_for_split, sites, n_splits=n_splits, n_repeats=1,
                              seed=seed + rep)
        for fold, (tr_all, te) in enumerate(gen):
            tr_rel, va_rel = split_train_val(y_for_split[tr_all], sites[tr_all],
                                             seed=seed + 100 * rep + fold)
            tr, va = tr_all[tr_rel], tr_all[va_rel]

            tag = f"rep{rep}_fold{fold}"
            fold_csv_dir = os.path.join(out_csv_dir, tag)
            os.makedirs(fold_csv_dir, exist_ok=True)
            combined_df.iloc[tr].to_csv(os.path.join(fold_csv_dir, "train.csv"), index=False)
            combined_df.iloc[va].to_csv(os.path.join(fold_csv_dir, "val.csv"), index=False)
            combined_df.iloc[te].to_csv(os.path.join(fold_csv_dir, "test.csv"), index=False)

            fold_dp_dir = os.path.join(out_dp_dir, tag)
            os.makedirs(fold_dp_dir, exist_ok=True)
            np.save(os.path.join(fold_dp_dir, "X_train.npy"), X_all[tr])
            np.save(os.path.join(fold_dp_dir, "Y_train.npy"), Y_all[tr])
            np.save(os.path.join(fold_dp_dir, "X_val.npy"), X_all[va])
            np.save(os.path.join(fold_dp_dir, "Y_val.npy"), Y_all[va])
            np.save(os.path.join(fold_dp_dir, "X_eval.npy"), X_all[te])  # outer test
            np.save(os.path.join(fold_dp_dir, "Y_eval.npy"), Y_all[te])
            shutil.copyfile(os.path.join(dp_dir, "c.npy"), os.path.join(fold_dp_dir, "c.npy"))
            shutil.copyfile(os.path.join(dp_dir, "label_names.txt"),
                            os.path.join(fold_dp_dir, "label_names.txt"))
            n_written += 1

    print(f"[{domain}] wrote {n_written} (rep, fold) triples -> {out_csv_dir} and {out_dp_dir}")


if __name__ == "__main__":
    for domain in DOMAINS:
        build_domain(domain)

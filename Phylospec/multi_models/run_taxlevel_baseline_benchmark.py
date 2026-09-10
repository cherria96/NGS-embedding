#!/usr/bin/env python3
"""
run_taxlevel_baseline_benchmark.py -- classical-ML baseline comparison requested
alongside the tree-aware benchmark (run_site_grouped_cv_benchmark.py): 6 flat
feature-vector models (RF, KNN, SVM, NNET, XGBoost, GLMNET) trained on naive
taxonomy-LABEL-collapsed relative abundance (see build_taxlevel_inputs.py) at
each of Phylum / Class / Order / Family / Genus rank, x each of ARC / BAC /
merged domain -- 15 (level, domain) table pairs x 6 models.

Deliberately reuses run_site_grouped_cv_benchmark.make_csv_fit_predict (the
--train/--val/--test CSV subprocess contract all 6 of these models share, since
none of them need a tree, a c.npy patristic matrix, or Phylo-Spec's own
train/test split) plus its score_multilabel_table / load_site_map helpers and
evaluate_multilabel.evaluate_cv -- the exact same site-grouped, pooled-OOF,
MCC-tuned-threshold protocol (task Sec 5A), so these results are directly
comparable to results_<domain>_<model>.csv.

Output: results_<level>_<domain>_<model>.csv, one row per flag plus a macro
row, in Phylospec/multi_models/results/site_grouped_cv_taxlevel/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
sys.path.insert(0, PHYLOSPEC_DIR)
from run_site_grouped_cv_benchmark import (  # noqa: E402
    make_csv_fit_predict, score_multilabel_table, load_site_map,
)
from evaluate_multilabel import LABEL_COLS, audit_site_label_support, evaluate_cv  # noqa: E402

CSV_DATA_DIR = os.path.join(HERE, "input_for_all_models_taxlevel")
RESULTS_DIR = os.path.join(HERE, "results", "site_grouped_cv_taxlevel")
LEVELS = ["Phylum", "Class", "Order", "Family", "Genus"]
DOMAINS = ["ARC", "BAC", "merged"]
MODELS = ["RF", "KNN", "SVM", "NNET", "XGBoost", "GLMNET"]
POS_WEIGHT_CAP = 10.0

FLAG_COLS = sorted(c.replace("warning_", "") for c in LABEL_COLS)  # alphabetical,
# matches the FLAGS constant hardcoded in every */*-multilabel.py script


def load_table(level, domain):
    csv_dir = os.path.join(CSV_DATA_DIR, level, f"Warnings_{domain}")
    combined_df = pd.read_csv(os.path.join(csv_dir, "table.csv"))
    site_map = load_site_map()
    sites = combined_df["SampleID"].astype(str).map(site_map).values
    assert not pd.isna(sites).any(), f"{level}/{domain}: samples with no Site in metadata"
    y = combined_df[FLAG_COLS].values.astype(int)  # alphabetical order, authoritative
    return combined_df, sites, y


def build_fit_predict(model, domain, combined_df, scratch_root):
    script = os.path.join(HERE, model, f"{model}-multilabel.py")
    return make_csv_fit_predict(script, domain, combined_df, [], scratch_root)


def run_all(levels=LEVELS, domains=DOMAINS, models=MODELS, n_splits=5, n_repeats=5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scratch_root = os.path.join(RESULTS_DIR, "_scratch")
    os.makedirs(scratch_root, exist_ok=True)

    for level in levels:
        for domain in domains:
            combined_df, sites, y = load_table(level, domain)
            support = audit_site_label_support(y, sites, label_names=FLAG_COLS)
            pd.DataFrame(support).to_csv(
                os.path.join(RESULTS_DIR, f"site_label_support_{level}_{domain}.csv"), index=False)
            print(f"=== {level} / {domain} === "
                  f"({combined_df.shape[1] - 1 - len(FLAG_COLS)} features, "
                  f"{combined_df.shape[0]} samples)")

            for model in models:
                print(f"--- {level} / {domain} / {model} ---")
                fit_predict = build_fit_predict(model, domain, combined_df, scratch_root)
                res = evaluate_cv(y, sites, fit_predict, label_names=FLAG_COLS,
                                  n_splits=n_splits, n_repeats=n_repeats,
                                  pos_weight_cap=POS_WEIGHT_CAP)
                table = score_multilabel_table(res, domain, model)
                table.insert(0, "level", level)
                out_path = os.path.join(RESULTS_DIR, f"results_{level}_{domain}_{model}.csv")
                table.to_csv(out_path, index=False)
                print(table[["flag", "n_pos", "mcc", "f2", "auprc", "auprc_lift"]]
                      .to_string(index=False))
                print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    a = ap.parse_args()
    run_all(a.levels, a.domains, a.models, a.n_splits, a.n_repeats)

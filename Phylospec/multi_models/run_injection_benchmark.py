#!/usr/bin/env python3
"""
run_injection_benchmark.py -- the "with vs without cross-domain-syntrophy-
injection" arm (Phylospec/CROSS_DOMAIN_INJECTION_BY_MODEL.md), DeepPhylo only
in this pass (Sec 1 of that doc -- a drop-in embedding swap). Phylo-Spec
(Sec 2, node-weight reweighting or edge-feature columns) and MetaDR (Sec 3,
optimal-leaf-ordering seriation) are NOT wired in here; both need real
per-model surgery beyond a swapped input file and are left for a follow-up.

Injection only applies to the MERGED domain -- ARC-only/BAC-only have no
Bacteria x Archaea pair to find at all, and build_cross_domain_graph.py
itself refuses to run without both domains present (see prep_injection_
inputs.py / the build_cross_domain_graph.py runs already executed for
Warnings_merged under input_for_all_models_{genus,asvtree}).

Same train:test split as run_holdout_benchmark.py (ML/results_warnings/
split_and_cv_assignment.csv). Per the doc's explicit warning ("otherwise you
have changed two things at once"), "with" and "without" use the IDENTICAL
feature set -- build_cross_domain_graph.py's own tree-match + prevalence-
filtered `kept` subset (injection_DeepPhylo_X.npy), not the full genus-tree/
asv-tree feature set used elsewhere in this benchmark -- so this arm's numbers
are not directly comparable to results_{genustree,asvtree}_merged_DeepPhylo.
csv's feature count, only to each other (with vs without, same p).

Output: results_<arm>_merged_DeepPhylo_{without,with}.csv (point estimates,
no CI, same shape as run_holdout_benchmark.py's tables plus an `injection`
column).
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(PHYLOSPEC_DIR, "..")

sys.path.insert(0, HERE)
sys.path.insert(0, PHYLOSPEC_DIR)
from run_site_grouped_cv_benchmark import FLAG_COLS, PY, POS_WEIGHT_CAP, run_subprocess  # noqa: E402
from run_holdout_benchmark import load_split, load_site_map, result_table  # noqa: E402
from evaluate_multilabel import audit_site_label_support, evaluate_holdout  # noqa: E402

ARMS = {
    "genustree": os.path.join(HERE, "input_for_all_models_genus", "Warnings_merged"),
    "asvtree": os.path.join(HERE, "input_for_all_models_asvtree", "Warnings_merged"),
}
RESULTS_DIR = os.path.join(HERE, "results", "holdout")


def load_injection_data(arm_dir):
    """table_for_injection.csv (row order authoritative) + injection_DeepPhylo_X
    .npy (build_cross_domain_graph.py's own tree-match + prevalence-filtered,
    reordered feature subset, same row order) + this benchmark's own Y (from
    table.csv, FLAG_COLS order) -- NOT injection_DeepPhylo_y.npy, which only
    carries the single placeholder label prep_injection_inputs.py used to
    satisfy build_cross_domain_graph.py's -c format."""
    table = pd.read_csv(os.path.join(arm_dir, "table_for_injection.csv"))
    full_table = pd.read_csv(os.path.join(arm_dir, "table.csv"))
    assert (table["SampleID"].values == full_table["SampleID"].values).all(), (
        "table_for_injection.csv and table.csv row order diverged")
    X = np.load(os.path.join(arm_dir, "injection_DeepPhylo_X.npy"))
    assert len(X) == len(table), f"{arm_dir}: injection X rows != table rows"
    Y = full_table[FLAG_COLS].values.astype(int)
    sample_ids = table["SampleID"].astype(str).values
    return sample_ids, X, Y


def make_fit_predict(X, Y, embedding_path, domain, scratch_root):
    script = os.path.join(HERE, "DeepPhylo", "deepphylo_classification_multi_label_bce.py")

    def fit_predict(tr, va, te, pos_weight):
        with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
            np.save(os.path.join(tmp, "X_train.npy"), X[tr])
            np.save(os.path.join(tmp, "Y_train.npy"), Y[tr].astype(np.float32))
            np.save(os.path.join(tmp, "X_val.npy"), X[va])
            np.save(os.path.join(tmp, "Y_val.npy"), Y[va].astype(np.float32))
            np.save(os.path.join(tmp, "X_eval.npy"), X[te])
            np.save(os.path.join(tmp, "Y_eval.npy"), Y[te].astype(np.float32))
            with open(os.path.join(tmp, "label_names.txt"), "w") as f:
                f.write("\n".join(FLAG_COLS))
            scores_dir = os.path.join(tmp, "scores")
            cmd = [PY, script, "--data_dir", tmp, "--val_dir", tmp, "--domain", domain,
                   "--pos-weight-cap", str(POS_WEIGHT_CAP), "--scores-out-dir", scores_dir,
                   "--embedding_path", embedding_path]
            run_subprocess(cmd)
            val_scores = np.load(os.path.join(scores_dir, "val_scores.npy"))
            test_scores = np.load(os.path.join(scores_dir, "test_scores.npy"))
        return val_scores, test_scores
    return fit_predict


def run_arm(arm, arm_dir, split_map, out_dir, scratch_root):
    sample_ids, X, Y = load_injection_data(arm_dir)
    keep = pd.Series(sample_ids).isin(split_map).values
    sample_ids, X, Y = sample_ids[keep], X[keep], Y[keep]
    split = pd.Series(sample_ids).map(split_map).values
    train_idx = np.flatnonzero(split == "train")
    test_idx = np.flatnonzero(split == "test")

    site_map = load_site_map()
    sites = pd.Series(sample_ids).map(site_map).values
    assert not pd.isna(sites).any(), f"{arm}: samples with no Site"
    support = audit_site_label_support(Y[test_idx], sites[test_idx], label_names=FLAG_COLS)
    pd.DataFrame(support).to_csv(
        os.path.join(out_dir, f"test_label_support_{arm}_injection.csv"), index=False)
    print(f"=== {arm} / merged / injection === ({X.shape[1]} features, "
         f"{len(train_idx)} train / {len(test_idx)} test)")

    for tag, fname in [("without", "injection_treeonly_DeepPhylo_embeding.npy"),
                       ("with", "injection_DeepPhylo_embeding.npy")]:
        embedding_path = os.path.join(arm_dir, fname)
        print(f"--- {arm} / merged / DeepPhylo / {tag} injection ---")
        fit_predict = make_fit_predict(X, Y, embedding_path, "merged", scratch_root)
        res = evaluate_holdout(Y, sites, fit_predict, train_idx, test_idx,
                               label_names=FLAG_COLS, pos_weight_cap=POS_WEIGHT_CAP)
        table = result_table(res, arm, "merged", "DeepPhylo")
        table.insert(3, "injection", tag)
        out_path = os.path.join(out_dir, f"results_{arm}_merged_DeepPhylo_{tag}.csv")
        table.to_csv(out_path, index=False)
        print(table[["flag", "n_pos", "mcc", "f2", "auprc", "auprc_lift"]].to_string(index=False))
        print(f"-> {out_path}")


def run_all(arms):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scratch_root = os.path.join(RESULTS_DIR, "_scratch")
    os.makedirs(scratch_root, exist_ok=True)
    split_map = load_split()
    for arm in arms:
        run_arm(arm, ARMS[arm], split_map, RESULTS_DIR, scratch_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = ap.parse_args()
    run_all(a.arms)

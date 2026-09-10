#!/usr/bin/env python3
"""
build_inputs_from_genus_tables.py -- convert THIS repo's genus-level pipeline
output (output/genus_tree/<mode>/{arc,bac,merged}_table.csv + _tree.nwk) into
the per-domain input directories run_site_grouped_cv_benchmark.py consumes.

Why this exists: the pre-existing input_for_all_models/Warnings_<domain>/ and
DeepPhylo/data/warnings_<domain>/ directories were built by
prepare_warning_targets_data.py from RAW ASV-level features under a naive
DOMAIN_CONNECT_BRANCH_LEN=1.0 graft (see REPORT.md Sec 4 / the task doc's Sec
5.5) -- not the genus-collapsed, calibrated-graft-fallback pipeline actually
built and verified in this repo. Running the evaluation driver against the
old directories would silently evaluate the wrong feature set. This script
points it at the right one instead of overwriting the old directories (so
both remain available for comparison).

Per domain, writes to multi_models/input_for_all_models_genus/Warnings_<domain>/:
  table.csv          SampleID-indexed, genus features + 5 flag columns (0/1)
                      -- read_site_grouped_cv_benchmark.load_domain() consumes
                      this directly (no train/test split at rest; the outer
                      test/inner-val split happens inside evaluate_cv() itself)
  phylogeny.nwk       copy of <arm>_tree.nwk, for MetaDR's tree traversal
  PMCNN_list.csv      4 phylogenetic feature orderings (build_pmcnn_groups'
                      cophenetic-clustering recipe), for PMCNN's 4 conv branches
  X.npy, Y.npy, c.npy, label_names.txt   for DeepPhylo (c.npy = RAW patristic
                      distance matrix; PCA happens inside the DeepPhylo script)

Requires the genus-tree pipeline to have already been run
(run_genus_tree_pipeline.py --mode <mode>).
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.cluster.hierarchy import fcluster, linkage

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(PHYLOSPEC_DIR, "..")
OUT_DIR = os.path.join(HERE, "input_for_all_models_genus")
DOMAINS = ["arc", "bac", "merged"]

sys.path.insert(0, PHYLOSPEC_DIR)
from domain_merge_utils import fast_patristic  # noqa: E402
from evaluate_multilabel import LABEL_COLS  # noqa: E402

FLAG_COLS = sorted(c.replace("warning_", "") for c in LABEL_COLS)
PMCNN_THRESHOLDS = [0.1, 0.01, 0.3, 0.2]


def cophenetic_matrix(tree):
    terminals = tree.get_terminals()
    names = [t.name for t in terminals]
    n = len(terminals)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = tree.distance(terminals[i], terminals[j])
    return names, matrix


def pmcnn_orderings(tree, thresholds=PMCNN_THRESHOLDS):
    """Verbatim recipe from PMCNN/build_pmcnn_groups.py, factored to take an
    already-parsed tree instead of re-reading the file."""
    names, dist = cophenetic_matrix(tree)
    transformed = np.exp(-(dist * dist) / 0.5).astype(np.float32)

    def my_dist(p1, p2):
        return 1.0 - transformed[int(np.asarray(p1).item()), int(np.asarray(p2).item())]

    X = np.arange(transformed.shape[0]).reshape(-1, 1)
    linked = linkage(X, method="single", metric=my_dist)
    orderings = []
    for t in thresholds:
        clusters = fcluster(linked, t=t, criterion="distance")
        index_dict = {}
        for i, val in enumerate(clusters):
            index_dict.setdefault(val, []).append(i)
        orderings.append([names[idx] for indexes in index_dict.values() for idx in indexes])
    return orderings


def build_domain(arm, genus_dir, out_root):
    print(f"=== {arm} ===")
    table_path = os.path.join(genus_dir, f"{arm}_table.csv")
    tree_path = os.path.join(genus_dir, f"{arm}_tree.nwk")
    if not (os.path.exists(table_path) and os.path.exists(tree_path)):
        raise FileNotFoundError(
            f"{table_path} / {tree_path} not found -- run "
            f"run_genus_tree_pipeline.py first")

    multi_table = os.path.join(genus_dir, arm, "multi_output", "table.csv")
    # write_label_views() wrote this with the SampleID INDEX carrying its name
    # through to_csv(), so plain pd.read_csv() (no index_col) already gives a
    # normal "SampleID" column -- no index gymnastics needed.
    df = pd.read_csv(multi_table)
    feature_cols = [c for c in df.columns if c not in FLAG_COLS + ["SampleID"]]
    df = df[["SampleID"] + feature_cols + FLAG_COLS]

    domain_name = {"arc": "ARC", "bac": "BAC", "merged": "merged"}[arm]
    out_dir = os.path.join(out_root, f"Warnings_{domain_name}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "table.csv"), index=False)
    print(f"  table.csv: {df.shape[0]} samples x {len(feature_cols)} features")

    tree = Phylo.read(tree_path, "newick")
    shutil.copyfile(tree_path, os.path.join(out_dir, "phylogeny.nwk"))

    orderings = pmcnn_orderings(tree)
    pd.DataFrame(orderings).to_csv(os.path.join(out_dir, "PMCNN_list.csv"), index=False)
    print(f"  PMCNN_list.csv: {len(orderings)} orderings x {len(orderings[0])} features")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y = df[FLAG_COLS].to_numpy(dtype=np.float32)
    C = fast_patristic(tree, feature_cols)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "Y.npy"), Y)
    np.save(os.path.join(out_dir, "c.npy"), C)
    with open(os.path.join(out_dir, "label_names.txt"), "w") as f:
        f.write("\n".join(FLAG_COLS))
    print(f"  DeepPhylo: X {X.shape}  Y {Y.shape}  c {C.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genus-dir", default=os.path.join(ROOT, "output", "genus_tree", "genus_autorun"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    for arm in DOMAINS:
        build_domain(arm, a.genus_dir, a.out_dir)

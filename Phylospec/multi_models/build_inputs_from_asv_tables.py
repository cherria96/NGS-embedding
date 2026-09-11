#!/usr/bin/env python3
"""
build_inputs_from_asv_tables.py -- convert the pre-existing, original
ASV-level inputs at multi_models/input_for_all_models/Warnings_<domain>/
{train,test}.csv + phylogeny.nwk (raw ASV features under a naive
DOMAIN_CONNECT_BRANCH_LEN=1.0 graft, built by prepare_warning_targets_data.py)
into the same per-domain input-directory shape
run_site_grouped_cv_benchmark.py consumes for the genus-level arm (see
build_inputs_from_genus_tables.py) -- one combined table.csv (no train/test
split at rest; evaluate_cv() owns the outer test / inner val split), plus a
copy of the tree, PMCNN orderings, and DeepPhylo's X/Y/c/label_names.

Per domain, writes to multi_models/input_for_all_models_asv/Warnings_<domain>/:
  table.csv          SampleID-indexed, ASV features + 5 flag columns (0/1)
  phylogeny.nwk       copy of the original ASV-level tree
  PMCNN_list.csv      4 phylogenetic feature orderings (kept for interface
                      parity with the genus pipeline; unused by the
                      DeepPhylo/MetaDR/PhyloSpec run this was built for)
  X.npy, Y.npy, c.npy, label_names.txt   for DeepPhylo (c.npy = raw patristic
                      distance matrix; PCA happens inside the DeepPhylo script)
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from Bio import Phylo

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ASV_SRC_DIR = os.path.join(HERE, "input_for_all_models")
OUT_DIR = os.path.join(HERE, "input_for_all_models_asv")
DOMAINS = ["ARC", "BAC", "merged"]

sys.path.insert(0, PHYLOSPEC_DIR)
from domain_merge_utils import fast_patristic  # noqa: E402
from evaluate_multilabel import LABEL_COLS  # noqa: E402
from build_inputs_from_genus_tables import pmcnn_orderings  # noqa: E402

FLAG_COLS = sorted(c.replace("warning_", "") for c in LABEL_COLS)


def build_domain(domain, src_dir, out_root):
    print(f"=== {domain} ===")
    src = os.path.join(src_dir, f"Warnings_{domain}")
    train_df = pd.read_csv(os.path.join(src, "train.csv"))
    test_df = pd.read_csv(os.path.join(src, "test.csv"))
    df = pd.concat([train_df, test_df], ignore_index=True)
    feature_cols = [c for c in df.columns if c not in FLAG_COLS + ["SampleID"]]
    df = df[["SampleID"] + feature_cols + FLAG_COLS]

    out_dir = os.path.join(out_root, f"Warnings_{domain}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "table.csv"), index=False)
    print(f"  table.csv: {df.shape[0]} samples x {len(feature_cols)} features "
         f"(train {len(train_df)} + test {len(test_df)})")

    tree_path = os.path.join(src, "phylogeny.nwk")
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
    ap.add_argument("--src-dir", default=ASV_SRC_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    for domain in DOMAINS:
        build_domain(domain, a.src_dir, a.out_dir)

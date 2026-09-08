"""
Build DeepPhylo regression inputs (X/Y/C .npy triples) for the AD-digester
VFA/ALK-stability prediction task, from this project's own QIIME2 exports
and Dat_chem-derived metadata.

Target label ("Group"): eff_TVFAs / eff_ALK per SampleID, from data/final/metadata.csv.

Three experiments are produced under Phylospec/multi_models/DeepPhylo/data/:
  - vfa_alk_ARC     : archaeal-only feature table + tree
  - vfa_alk_BAC     : bacterial-only feature table + tree
  - vfa_alk_merged  : dual-domain (ARC leaves + BAC leaves concatenated),
                      trees grafted under a synthetic root joining the two
                      domain roots

Each experiment directory gets: X_train.npy, X_eval.npy, Y_train.npy,
Y_eval.npy, c.npy (leaf-by-leaf patristic distance matrix, in the same
column order as X), feature_names.txt (for traceability), and
sample_ids_{train,eval}.txt.
"""
import csv
import os

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.abspath(__file__))
QIIME_DIR = os.path.join(ROOT, "data", "qiimeresult")
METADATA_PATH = os.path.join(ROOT, "data", "final", "metadata.csv")
OUT_BASE = os.path.join(ROOT, "Phylospec", "multi_models", "DeepPhylo", "data")

PREVALENCE_THRESH = 0.15
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Branch length used to join each domain's root to the synthetic merged
# root. Chosen to be on the same order as each domain's own root-to-leaf
# depth (ARC max depth ~3.7, BAC max depth ~3.9), so the Bacteria/Archaea
# split is neither negligible nor wildly dominant in the patristic
# distances feeding the PCA embedding.
DOMAIN_CONNECT_BRANCH_LEN = 1.0


def round_no_key(sample_col):
    """'1-11b-DGYSb' -> '1-11b' (matches metadata.csv's SampleID)."""
    parts = sample_col.split("-")
    return parts[0] + "-" + parts[1]


def load_feature_table(domain):
    path = os.path.join(QIIME_DIR, domain, "table_filtered.tsv")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    df.columns = [round_no_key(c) for c in df.columns]
    return df  # features x samples, indexed by round-no key


def load_labels():
    labels = {}
    with open(METADATA_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tvfa, alk = row["eff_TVFAs"], row["eff_ALK"]
            if tvfa in ("", None) or alk in ("", None):
                continue
            try:
                tvfa_f, alk_f = float(tvfa), float(alk)
            except ValueError:
                continue
            if alk_f != 0:
                labels[row["SampleID"]] = tvfa_f / alk_f
    return labels


def load_site_map():
    """SampleID ('1-2') -> Site code ('BSN'). A site's 4 seasonal samples
    are not independent (same digester, same microbial community lineage),
    so splits must keep every season of a site on the same side."""
    site_map = {}
    with open(METADATA_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            site_map[row["SampleID"]] = row["Site"]
    return site_map


def prevalence_filter(counts_df, thresh=PREVALENCE_THRESH):
    """counts_df: features x samples raw counts. Returns filtered df."""
    prevalence = (counts_df > 0).sum(axis=1) / counts_df.shape[1]
    return counts_df.loc[prevalence >= thresh]


def relative_abundance(counts_df):
    """features x samples -> features x samples, each column sums to 1."""
    totals = counts_df.sum(axis=0)
    return counts_df.div(totals, axis=1)


def prune_tree(tree_path, keep_leaf_names):
    tree = Phylo.read(tree_path, "newick")
    keep = set(keep_leaf_names)
    drop = [c for c in tree.get_terminals() if c.name not in keep]
    for clade in drop:
        tree.prune(clade)
    remaining = {c.name for c in tree.get_terminals()}
    missing = keep - remaining
    if missing:
        raise ValueError(f"{len(missing)} requested leaves not found in tree, e.g. {list(missing)[:5]}")
    return tree


def patristic_distance_matrix(tree, ordered_leaf_names):
    """Square matrix in the given leaf-name order, using Bio.Phylo distances."""
    name_to_clade = {c.name: c for c in tree.get_terminals()}
    n = len(ordered_leaf_names)
    dist = np.zeros((n, n), dtype=np.float64)
    clades = [name_to_clade[name] for name in ordered_leaf_names]
    for i in range(n):
        ci = clades[i]
        for j in range(i + 1, n):
            d = tree.distance(ci, clades[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def graft_domain_trees(arc_tree, bac_tree, connect_len=DOMAIN_CONNECT_BRANCH_LEN):
    arc_tree.root.branch_length = connect_len
    bac_tree.root.branch_length = connect_len
    new_root = Clade(branch_length=0.0, name="domain_root")
    new_root.clades = [arc_tree.root, bac_tree.root]
    return Tree(root=new_root, rooted=True)


def split_and_save(out_dir, X, y, feature_names, sample_ids, site_map):
    os.makedirs(out_dir, exist_ok=True)
    idx = np.arange(len(sample_ids))
    groups = [site_map[s] for s in sample_ids]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, eval_idx = next(splitter.split(idx, y, groups=groups))

    np.save(os.path.join(out_dir, "X_train.npy"), X[train_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "X_eval.npy"), X[eval_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "Y_train.npy"), y[train_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "Y_eval.npy"), y[eval_idx].astype(np.float32))

    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(out_dir, "sample_ids_train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_ids[i] for i in train_idx))
    with open(os.path.join(out_dir, "sample_ids_eval.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_ids[i] for i in eval_idx))

    print(f"  -> {out_dir}")
    print(f"     X_train {X[train_idx].shape}  X_eval {X[eval_idx].shape}  n_features {len(feature_names)}")


def build_single_domain(domain, labels, site_map):
    print(f"=== {domain} ===")
    counts = load_feature_table(domain)  # features x samples (round-no keyed)
    sample_ids = [s for s in counts.columns if s in labels]
    counts = counts[sample_ids]
    counts = prevalence_filter(counts)
    rel = relative_abundance(counts)  # features x samples

    feature_names = list(rel.index)
    X = rel.T.values  # samples x features
    y = np.array([labels[s] for s in sample_ids])

    tree_path = os.path.join(QIIME_DIR, domain, "rooted-tree.nwk")
    tree = prune_tree(tree_path, feature_names)
    C = patristic_distance_matrix(tree, feature_names)

    out_dir = os.path.join(OUT_BASE, f"vfa_alk_{domain}")
    split_and_save(out_dir, X, y, feature_names, sample_ids, site_map)
    np.save(os.path.join(out_dir, "c.npy"), C.astype(np.float32))
    return feature_names, sample_ids


def build_merged(labels, site_map):
    print("=== merged (ARC + BAC) ===")
    arc_counts = load_feature_table("ARC")
    bac_counts = load_feature_table("BAC")

    sample_ids = [s for s in arc_counts.columns if s in bac_counts.columns and s in labels]

    arc_counts = prevalence_filter(arc_counts[sample_ids])
    bac_counts = prevalence_filter(bac_counts[sample_ids])
    arc_rel = relative_abundance(arc_counts)
    bac_rel = relative_abundance(bac_counts)

    arc_features = list(arc_rel.index)
    bac_features = list(bac_rel.index)
    feature_names = arc_features + bac_features

    X = np.concatenate([arc_rel.T.values, bac_rel.T.values], axis=1)  # samples x (arc+bac features)
    y = np.array([labels[s] for s in sample_ids])

    arc_tree = prune_tree(os.path.join(QIIME_DIR, "ARC", "rooted-tree.nwk"), arc_features)
    bac_tree = prune_tree(os.path.join(QIIME_DIR, "BAC", "rooted-tree.nwk"), bac_features)
    merged_tree = graft_domain_trees(arc_tree, bac_tree)
    C = patristic_distance_matrix(merged_tree, feature_names)

    out_dir = os.path.join(OUT_BASE, "vfa_alk_merged")
    split_and_save(out_dir, X, y, feature_names, sample_ids, site_map)
    np.save(os.path.join(out_dir, "c.npy"), C.astype(np.float32))


def main():
    labels = load_labels()
    site_map = load_site_map()
    print(f"Loaded {len(labels)} labeled samples (eff_TVFAs / eff_ALK) from metadata.csv\n")

    build_single_domain("ARC", labels, site_map)
    build_single_domain("BAC", labels, site_map)
    build_merged(labels, site_map)


if __name__ == "__main__":
    main()

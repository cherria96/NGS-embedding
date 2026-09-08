"""
Build inputs for the multi-target AD-digester stability-warning task, for
both DeepPhylo (multi-label) and Phylo-Spec (single-label, run once per flag).

Targets (from data/final/metadata.csv, already computed by
code/metadata_targetwriter.py from stability-indicator thresholds):
  warning_acid_base_balance, warning_buffer_capacity,
  warning_acid_accumulation, warning_ammonia_toxicity, warning_biogas_quality

For each domain (ARC, BAC, merged dual-domain) this produces:
  - Phylospec/multi_models/DeepPhylo/data/warnings_{domain}/
        X_train.npy, X_eval.npy, Y_train.npy (N,5), Y_eval.npy (N,5), c.npy,
        feature_names.txt, label_names.txt, sample_ids_{train,eval}.txt
  - Phylospec/example/warnings_{domain}/phylogeny.nwk
        (same pruned/grafted tree, shared across all 5 flags for that domain)
  - Phylospec/example/warnings_{domain}/{flag}/example_train.csv
    Phylospec/example/warnings_{domain}/{flag}/example_test.csv
        (SampleID, <features>, Group in {"Normal","Warning"})

Both model families use the *same* train/eval sample split per domain
(multilabel-stratified, to keep at least some positives of every rare flag
in the eval fold) so their reported metrics are comparable.
"""
import csv
import os

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = os.path.dirname(os.path.abspath(__file__))
QIIME_DIR = os.path.join(ROOT, "data", "qiimeresult")
METADATA_PATH = os.path.join(ROOT, "data", "final", "metadata.csv")
DEEPPHYLO_OUT_BASE = os.path.join(ROOT, "Phylospec", "multi_models", "DeepPhylo", "data")
PHYLOSPEC_OUT_BASE = os.path.join(ROOT, "Phylospec", "example")

LABEL_COLS = [
    "warning_acid_base_balance",
    "warning_buffer_capacity",
    "warning_acid_accumulation",
    "warning_ammonia_toxicity",
    "warning_biogas_quality",
]

PREVALENCE_THRESH = 0.15
TEST_SIZE = 0.2
RANDOM_STATE = 42
DOMAIN_CONNECT_BRANCH_LEN = 1.0


def round_no_key(sample_col):
    parts = sample_col.split("-")
    return parts[0] + "-" + parts[1]


def load_feature_table(domain):
    path = os.path.join(QIIME_DIR, domain, "table_filtered.tsv")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    df.columns = [round_no_key(c) for c in df.columns]
    return df  # features x samples, indexed by round-no key


def load_label_matrix():
    labels = {}
    with open(METADATA_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vec = [int(row[c]) for c in LABEL_COLS]
            labels[row["SampleID"]] = vec
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
    prevalence = (counts_df > 0).sum(axis=1) / counts_df.shape[1]
    return counts_df.loc[prevalence >= thresh]


def relative_abundance(counts_df):
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
    name_to_clade = {c.name: c for c in tree.get_terminals()}
    n = len(ordered_leaf_names)
    dist = np.zeros((n, n), dtype=np.float64)
    clades = [name_to_clade[name] for name in ordered_leaf_names]
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(clades[i], clades[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def graft_domain_trees(arc_tree, bac_tree, connect_len=DOMAIN_CONNECT_BRANCH_LEN):
    arc_tree.root.branch_length = connect_len
    bac_tree.root.branch_length = connect_len
    new_root = Clade(branch_length=0.0, name="domain_root")
    new_root.clades = [arc_tree.root, bac_tree.root]
    return Tree(root=new_root, rooted=True)


def grouped_multilabel_split(y_matrix, sample_ids, site_map):
    """Group by site (a site's seasonal samples aren't independent - same
    digester, same microbial lineage), then multilabel-stratify at the site
    level so eval still gets some positives of every rare flag."""
    sites = [site_map[s] for s in sample_ids]
    unique_sites = sorted(set(sites))
    site_to_indices = {site: [i for i, s in enumerate(sites) if s == site] for site in unique_sites}
    # a site counts as positive for a flag if any of its sampled seasons was flagged
    site_label = np.array([y_matrix[site_to_indices[site]].max(axis=0) for site in unique_sites])

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_site_idx, eval_site_idx = next(splitter.split(np.zeros((len(unique_sites), 1)), site_label))
    train_sites = {unique_sites[i] for i in train_site_idx}
    eval_sites = {unique_sites[i] for i in eval_site_idx}

    train_idx = np.array([i for i, s in enumerate(sites) if s in train_sites])
    eval_idx = np.array([i for i, s in enumerate(sites) if s in eval_sites])
    return train_idx, eval_idx


def write_phylospec_csvs(out_dir, X, y_matrix, sample_ids, feature_names, train_idx, eval_idx):
    os.makedirs(out_dir, exist_ok=True)
    for li, label_col in enumerate(LABEL_COLS):
        flag_name = label_col.replace("warning_", "")
        flag_dir = os.path.join(out_dir, flag_name)
        os.makedirs(flag_dir, exist_ok=True)
        labels_str = np.where(y_matrix[:, li] == 1, "Warning", "Normal")
        df = pd.DataFrame(X, columns=feature_names)
        df.insert(0, "SampleID", sample_ids)
        df["Group"] = labels_str
        df.iloc[train_idx].to_csv(os.path.join(flag_dir, "example_train.csv"), index=False)
        df.iloc[eval_idx].to_csv(os.path.join(flag_dir, "example_test.csv"), index=False)
        n_pos_train = int((y_matrix[train_idx, li] == 1).sum())
        n_pos_eval = int((y_matrix[eval_idx, li] == 1).sum())
        print(f"    {flag_name}: train warning={n_pos_train}/{len(train_idx)}  eval warning={n_pos_eval}/{len(eval_idx)}")


def write_deepphylo_npys(out_dir, X, y_matrix, feature_names, sample_ids, C, train_idx, eval_idx):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X[train_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "X_eval.npy"), X[eval_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "Y_train.npy"), y_matrix[train_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "Y_eval.npy"), y_matrix[eval_idx].astype(np.float32))
    np.save(os.path.join(out_dir, "c.npy"), C.astype(np.float32))
    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(out_dir, "label_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LABEL_COLS))
    with open(os.path.join(out_dir, "sample_ids_train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_ids[i] for i in train_idx))
    with open(os.path.join(out_dir, "sample_ids_eval.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_ids[i] for i in eval_idx))
    print(f"  -> {out_dir}  X_train {X[train_idx].shape}  X_eval {X[eval_idx].shape}")


def build_single_domain(domain, labels, site_map):
    print(f"=== {domain} ===")
    counts = load_feature_table(domain)
    sample_ids = [s for s in counts.columns if s in labels]
    counts = counts[sample_ids]
    counts = prevalence_filter(counts)
    rel = relative_abundance(counts)

    feature_names = list(rel.index)
    X = rel.T.values
    y_matrix = np.array([labels[s] for s in sample_ids])

    tree = prune_tree(os.path.join(QIIME_DIR, domain, "rooted-tree.nwk"), feature_names)
    C = patristic_distance_matrix(tree, feature_names)

    train_idx, eval_idx = grouped_multilabel_split(y_matrix, sample_ids, site_map)

    dp_out = os.path.join(DEEPPHYLO_OUT_BASE, f"warnings_{domain}")
    write_deepphylo_npys(dp_out, X, y_matrix, feature_names, sample_ids, C, train_idx, eval_idx)

    ps_out = os.path.join(PHYLOSPEC_OUT_BASE, f"warnings_{domain}")
    os.makedirs(ps_out, exist_ok=True)
    Phylo.write(tree, os.path.join(ps_out, "phylogeny.nwk"), "newick")
    write_phylospec_csvs(ps_out, X, y_matrix, sample_ids, feature_names, train_idx, eval_idx)


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

    X = np.concatenate([arc_rel.T.values, bac_rel.T.values], axis=1)
    y_matrix = np.array([labels[s] for s in sample_ids])

    arc_tree = prune_tree(os.path.join(QIIME_DIR, "ARC", "rooted-tree.nwk"), arc_features)
    bac_tree = prune_tree(os.path.join(QIIME_DIR, "BAC", "rooted-tree.nwk"), bac_features)
    merged_tree = graft_domain_trees(arc_tree, bac_tree)
    C = patristic_distance_matrix(merged_tree, feature_names)

    train_idx, eval_idx = grouped_multilabel_split(y_matrix, sample_ids, site_map)

    dp_out = os.path.join(DEEPPHYLO_OUT_BASE, "warnings_merged")
    write_deepphylo_npys(dp_out, X, y_matrix, feature_names, sample_ids, C, train_idx, eval_idx)

    ps_out = os.path.join(PHYLOSPEC_OUT_BASE, "warnings_merged")
    os.makedirs(ps_out, exist_ok=True)
    Phylo.write(merged_tree, os.path.join(ps_out, "phylogeny.nwk"), "newick")
    write_phylospec_csvs(ps_out, X, y_matrix, sample_ids, feature_names, train_idx, eval_idx)


def main():
    labels = load_label_matrix()
    site_map = load_site_map()
    print(f"Loaded {len(labels)} samples with warning-flag labels\n")
    build_single_domain("ARC", labels, site_map)
    build_single_domain("BAC", labels, site_map)
    build_merged(labels, site_map)


if __name__ == "__main__":
    main()

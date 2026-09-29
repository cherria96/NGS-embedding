"""
Build the PMCNN feature-ordering list for a domain, following the same
cophenetic-distance + hierarchical-clustering recipe as PMCNN_list.py, but
driven by a tree file + feature list instead of hardcoded paths.

Each output row is a full permutation of all leaf/feature names, ordered by
phylogenetic clustering at one of four distance thresholds -- this matches
what PMCNN.py expects from its "-list" argument (see load_data(), which
selects X[My_list[i]] for i in 0..3, then PMCNN.py's own training loop
concatenates and re-splits these four re-orderings into four conv branches).
"""
import argparse
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from Bio import Phylo

THRESHOLDS = [0.1, 0.01, 0.3, 0.2]


def cophenetic_matrix(tree_file):
    tree = Phylo.read(tree_file, "newick")
    terminals = tree.get_terminals()
    names = [t.name for t in terminals]
    n = len(terminals)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = tree.distance(terminals[i], terminals[j])
    return names, matrix


def transform_zero_one(matrix):
    return np.exp(-(matrix * matrix) / 0.5).astype(np.float32)


def cluster_orderings(distance_matrix, thresholds=THRESHOLDS):
    def my_dist(p1, p2):
        return 1.0 - distance_matrix[int(p1), int(p2)]

    X = np.arange(distance_matrix.shape[0]).reshape(-1, 1)
    linked = linkage(X, method="single", metric=my_dist)

    orderings = []
    for threshold in thresholds:
        clusters = fcluster(linked, t=threshold, criterion="distance")
        index_dict = {}
        for i, val in enumerate(clusters):
            index_dict.setdefault(val, []).append(i)
        order = [idx for indexes in index_dict.values() for idx in indexes]
        orderings.append(order)
    return orderings


def build_group_csv(tree_file, out_csv):
    names, dist = cophenetic_matrix(tree_file)
    transformed = transform_zero_one(dist)
    orderings = cluster_orderings(transformed)
    rows = [[names[idx] for idx in order] for order in orderings]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved PMCNN group list ({len(rows)} orderings x {len(names)} features) -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree", required=True, help="Path to phylogeny .nwk")
    parser.add_argument("-o", "--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    build_group_csv(args.tree, args.out)

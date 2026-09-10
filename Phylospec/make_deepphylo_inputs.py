#!/usr/bin/env python3
"""
make_deepphylo_inputs.py -- build the three .npy files DeepPhylo needs, starting
from a Phylo-Spec-style abundance CSV plus a Newick tree.

Input CSV layout (identical to what Phylo-Spec / PM-CNN / MetaDR / CNN / RF read):
    col 0            sample identifier
    cols 1 .. n-2    one column per feature (OTU / ASV / taxon)
    col n-1          label column (named "Group" by convention)

The Newick tree must contain a leaf whose name is exactly equal to each feature
column name. Features with no matching leaf are dropped (count reported on stderr).

Outputs:
    <prefix>_DeepPhylo_X.npy         float32  (n_samples, n_taxa)  abundances
    <prefix>_DeepPhylo_y.npy         int64    (n_samples,)         encoded labels
    <prefix>_DeepPhylo_embeding.npy  float64  (n_taxa, d), d = min(200, n_taxa)
    <prefix>_DeepPhylo_labelmap.csv  label <-> integer mapping

The embedding is a PCA of the patristic (tip-to-tip branch-length) distance matrix
of the tree pruned to the retained features, following DeepPhylo
(Adv. Sci. 2024, doi:10.1002/advs.202404277): phylogenetic embeddings for each OTU
are obtained by PCA-based dimensionality reduction of the phylogenetic distance
matrix, then summation-pooled per sample inside the network.

Usage:
    python make_deepphylo_inputs.py -c abundance.csv -t phylogeny.nwk -o mydata
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------- Newick parsing
class Node:
    __slots__ = ("name", "blen", "children", "parent")

    def __init__(self, parent=None):
        self.name = ""
        self.blen = 0.0
        self.children = []
        self.parent = parent


def parse_newick(text):
    """Recursive-descent Newick reader. Returns the root Node."""
    text = re.sub(r"\[[^\]]*\]", "", text)  # drop [comments]
    text = text.strip()
    pos = 0

    def read_label(node):
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] not in "(),;":
            pos += 1
        raw = text[start:pos].strip()
        if not raw:
            return
        if ":" in raw:
            lab, _, dist = raw.rpartition(":")
        else:
            lab, dist = raw, ""
        lab = lab.strip().strip("'\"")
        if lab:
            node.name = lab
        if dist.strip():
            node.blen = float(dist)

    def read_node(parent):
        nonlocal pos
        node = Node(parent)
        if pos < len(text) and text[pos] == "(":
            pos += 1  # consume '('
            while True:
                node.children.append(read_node(node))
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                break
            if pos < len(text) and text[pos] == ")":
                pos += 1  # consume ')'
        read_label(node)
        return node

    root = read_node(None)
    return root


def leaf_index(root):
    """{leaf_name: Node} for every named tip."""
    out, stack = {}, [root]
    while stack:
        n = stack.pop()
        if not n.children:
            if n.name:
                out[n.name] = n
        else:
            stack.extend(n.children)
    return out


def patristic_matrix(root, tips):
    """Exact tip-to-tip branch-length distances for the given list of tip Nodes.

    d(i, j) = depth(i) + depth(j) - 2 * depth(LCA(i, j)), where depth is the
    cumulative branch length from the root.
    """
    idx = {id(t): i for i, t in enumerate(tips)}
    n = len(tips)

    # cumulative root-to-node depth, iterative pre-order
    depth = {id(root): 0.0}
    order, stack = [], [root]
    while stack:
        nd = stack.pop()
        order.append(nd)
        for ch in nd.children:
            depth[id(ch)] = depth[id(nd)] + ch.blen
            stack.append(ch)

    lca_depth = np.zeros((n, n), dtype=np.float64)
    members = {}  # id(node) -> list of tip positions beneath it (only kept tips)
    for nd in reversed(order):  # post-order
        if not nd.children:
            members[id(nd)] = [idx[id(nd)]] if id(nd) in idx else []
            continue
        groups = [members.pop(id(ch)) for ch in nd.children]
        d = depth[id(nd)]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if groups[a] and groups[b]:
                    ia = np.array(groups[a])
                    ib = np.array(groups[b])
                    lca_depth[np.ix_(ia, ib)] = d
                    lca_depth[np.ix_(ib, ia)] = d
        merged = [i for g in groups for i in g]
        members[id(nd)] = merged

    dt = np.array([depth[id(t)] for t in tips])
    D = dt[:, None] + dt[None, :] - 2.0 * lca_depth
    np.fill_diagonal(D, 0.0)
    return D


# ------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--csv", required=True, help="abundance CSV (id | features | label)")
    ap.add_argument("-t", "--tree", required=True, help="Newick tree with leaf names == feature names")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("-d", "--dim", type=int, default=200, help="max embedding dimension (default 200)")
    ap.add_argument("--regression", action="store_true",
                    help="treat the label column as numeric (no label encoding)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ids, labels = df.columns[0], df.columns[-1]
    feats = list(df.columns[1:-1])

    root = parse_newick(open(args.tree, encoding="utf-8", errors="replace").read())
    leaves = leaf_index(root)
    kept = [f for f in feats if f in leaves]
    dropped = len(feats) - len(kept)
    if not kept:
        sys.exit("ERROR: no feature name matched a tree leaf -- check that your feature IDs "
                 "are the same strings as the Newick tip labels.")
    print(f"{len(df)} samples | {len(feats)} features | {len(kept)} matched to tree "
          f"| {dropped} dropped", file=sys.stderr)

    X = df[kept].to_numpy(dtype=np.float32)
    if args.regression:
        y = df[labels].to_numpy(dtype=np.float32)
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[labels].astype(str)).astype(np.int64)
        pd.DataFrame({"label": le.classes_, "code": range(len(le.classes_))}).to_csv(
            f"{args.out}_DeepPhylo_labelmap.csv", index=False)

    D = patristic_matrix(root, [leaves[f] for f in kept])
    dim = min(args.dim, len(kept))
    emb = PCA(n_components=dim, random_state=0).fit_transform(D)

    np.save(f"{args.out}_DeepPhylo_X.npy", X)
    np.save(f"{args.out}_DeepPhylo_y.npy", y)
    np.save(f"{args.out}_DeepPhylo_embeding.npy", emb)
    pd.Series(kept).to_csv(f"{args.out}_DeepPhylo_features.txt", index=False, header=False)
    print(f"X {X.shape} | y {y.shape} | embedding {emb.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()

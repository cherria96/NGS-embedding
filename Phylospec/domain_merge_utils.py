#!/usr/bin/env python3
"""
domain_merge_utils.py -- drop-in replacements for the three functions in
prepare_warning_targets_data.py that either scale badly or silently distort the
cross-domain geometry.

    from domain_merge_utils import (
        fast_patristic, audit_tree_for_phylospec, calibrate_connect_len,
        graft_domain_trees, rescale_for_node_weights)

Why each one exists is documented on the function.
"""
import numpy as np

# Median Bacteria<->Archaea patristic distance measured on the Greengenes 13_8
# reference tree shipped with the Phylo-Spec repo (Phylospec/database/16S/gg_13.nwk,
# 2,446 archaeal + 96,876 bacterial tips, 4,000 random cross-domain pairs).
# Within-domain medians on the same tree: Bacteria 0.735, Archaea 0.600.
REF_CROSS_DOMAIN_DISTANCE = 1.350


# ---------------------------------------------------------------- patristic
def fast_patristic(tree, ordered_leaf_names):
    """Full patristic distance matrix in one post-order pass.

    Replaces the O(n^2) loop over Bio.Phylo's tree.distance(), which re-walks the
    tree from the root for every pair: at n=3,000 features that is 4.5M traversals
    and runs for hours. Here every pair is written exactly once, vectorised at the
    node where the two tips' lineages meet, so a 3,000-tip tree takes seconds.

    Returns an (n, n) float64 array in the order of `ordered_leaf_names`.
    """
    idx = {nm: i for i, nm in enumerate(ordered_leaf_names)}
    n = len(ordered_leaf_names)
    D = np.zeros((n, n), dtype=np.float64)

    depth = {id(tree.root): 0.0}
    order, stack = [], [(tree.root, False)]
    while stack:
        node, seen = stack.pop()
        if seen:
            order.append(node)
            continue
        stack.append((node, True))
        for ch in node.clades:
            depth[id(ch)] = depth[id(node)] + (ch.branch_length or 0.0)
            stack.append((ch, False))

    tip_depth = np.zeros(n, dtype=np.float64)
    for node in order:
        if not node.clades and node.name in idx:
            tip_depth[idx[node.name]] = depth[id(node)]

    EMPTY = np.array([], dtype=int)
    tips = {}
    for node in order:
        if not node.clades:
            tips[id(node)] = np.array([idx[node.name]]) if node.name in idx else EMPTY
            continue
        blocks = [tips.pop(id(c)) for c in node.clades]
        dn2 = 2.0 * depth[id(node)]
        for a in range(len(blocks)):
            A = blocks[a]
            if not len(A):
                continue
            for b in range(a + 1, len(blocks)):
                B = blocks[b]
                if not len(B):
                    continue
                blk = tip_depth[A][:, None] + tip_depth[B][None, :] - dn2
                D[np.ix_(A, B)] = blk
                D[np.ix_(B, A)] = blk.T
        tips[id(node)] = np.concatenate(blocks) if blocks else EMPTY

    missing = [nm for nm in ordered_leaf_names if nm not in
               {c.name for c in tree.get_terminals()}]
    if missing:
        raise ValueError(f"{len(missing)} names absent from tree, e.g. {missing[:5]}")
    return D


# ------------------------------------------------------- Phylo-Spec audit
def audit_tree_for_phylospec(tree, label=""):
    """Phylo-Spec weights every node by (1 - branch_length) and MULTIPLIES the
    aggregated feature by it (src/model/data_processing.py::calculate_node_weights,
    src/model/PhyloSpec.py line 58). So a branch of exactly 1.0 zeroes that node's
    output, and anything longer flips its sign. Because feature_map[parent] is what
    the next level up consumes, a zeroed node also blanks everything above it.

    This is not hypothetical for a grafted tree: DOMAIN_CONNECT_BRANCH_LEN = 1.0
    puts both domain roots at weight exactly 0.
    """
    bl = np.array([c.branch_length for c in tree.find_clades()
                   if c.branch_length is not None], dtype=float)
    rep = {
        "label": label,
        "n_branches": int(bl.size),
        "max_branch_length": float(bl.max()) if bl.size else 0.0,
        "branches_eq_1_weight_zero": int(np.isclose(bl, 1.0).sum()),
        "branches_gt_1_weight_negative": int((bl > 1.0).sum()),
        "suggested_rescale_factor": float(1.0 / (bl.max() / 0.95)) if bl.size and bl.max() >= 1.0 else 1.0,
    }
    return rep


def rescale_for_node_weights(tree, factor):
    """Multiply every branch length by `factor`, in place.

    Phylo-Spec's 1 - branch_length weighting is NOT scale-invariant, so if any
    branch reaches 1.0 the tree has to be rescaled rather than clipped: clipping
    distorts only the long branches, rescaling preserves every relative distance.
    Apply the SAME factor to every arm of the experiment, or the arms are no
    longer comparable. DeepPhylo is unaffected in kind (a uniform scaling of C
    rescales the PCA axes but not their directions).
    """
    for c in tree.find_clades():
        if c.branch_length is not None:
            c.branch_length *= factor
    return tree


# ------------------------------------------------------------- graft + calibration
def calibrate_connect_len(arc_tree, bac_tree, target=REF_CROSS_DOMAIN_DISTANCE):
    """Choose DOMAIN_CONNECT_BRANCH_LEN instead of guessing it.

    Grafting makes every cross-domain distance
        d(arc_tip -> arc_root) + L + L + d(bac_root -> bac_tip),
    so L is a free parameter that single-handedly sets how far every archaeon sits
    from every bacterium -- the exact quantity a syntrophy-aware model is supposed
    to learn. Solving for the value that reproduces the reference cross-domain
    distance removes the guess:
        L = (target - mean_arc_tip_depth - mean_bac_tip_depth) / 2, floored at 0.

    On the Greengenes reference the two tip depths already sum to ~1.37 against a
    cross-domain median of 1.35, i.e. the true inter-domain stem is near zero; a
    separately midpoint-rooted domain tree has shallower tips, so a small positive
    L is expected here.
    """
    d_arc = float(np.mean([arc_tree.distance(arc_tree.root, t) for t in arc_tree.get_terminals()]))
    d_bac = float(np.mean([bac_tree.distance(bac_tree.root, t) for t in bac_tree.get_terminals()]))
    L = max(0.0, (target - d_arc - d_bac) / 2.0)
    return L, {"mean_arc_tip_depth": d_arc, "mean_bac_tip_depth": d_bac,
               "target_cross_domain_distance": target, "calibrated_connect_len": L,
               "resulting_cross_domain_distance": d_arc + d_bac + 2 * L}


def graft_domain_trees(arc_tree, bac_tree, connect_len, guard=True):
    """As in the original script, but refuses a value that breaks Phylo-Spec."""
    from Bio.Phylo.BaseTree import Clade, Tree
    if guard and connect_len >= 1.0:
        raise ValueError(
            f"connect_len={connect_len} gives node weight 1-{connect_len} <= 0 at both "
            "domain roots, which zeroes (or sign-flips) the whole-domain aggregates in "
            "Phylo-Spec. Use calibrate_connect_len().")
    arc_tree.root.branch_length = connect_len
    bac_tree.root.branch_length = connect_len
    new_root = Clade(branch_length=0.0, name="domain_root")
    new_root.clades = [arc_tree.root, bac_tree.root]
    return Tree(root=new_root, rooted=True)

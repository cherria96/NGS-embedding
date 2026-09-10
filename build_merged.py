#!/usr/bin/env python3
"""
build_merged.py -- task Sec 5.5: build ONE ARC+BAC domain tree, then prune it
three ways (ARC tips only, BAC tips only, both), so the three experimental arms
share a single definition of cross-domain distance instead of each inventing
its own.

Route actually used on this machine: the CALIBRATED GRAFT FALLBACK
(domain_merge_utils.calibrate_connect_len + graft_domain_trees), not the spec's
preferred SEPP fragment-insertion (`qiime fragment-insertion sepp`). QIIME2 is
not installed here and installing it (a multi-GB conda environment plus a ~1GB
SILVA SEPP reference) was explicitly declined for this session. If a SEPP tree
becomes available later (e.g. `qiime tools export --input-path insertion-tree.qza
--output-path sepp_tree && ...` to get a plain .nwk), pass it via --use-sepp-tree
and this script skips the graft entirely -- everything downstream (the three-way
prune, the audit, the rescale, the per-domain-closed abundance table) is
unaffected by which route produced the merged tree.

Also implements Sec 5.6: ARC and BAC relative abundances are computed
independently (each summing to 1 per sample) and concatenated WITHOUT a joint
renormalisation, because the two domains were amplified with different primers
and their read-count ratio reflects primer efficiency/sequencing effort, not
community composition. Do not "fix" this by closing the merged table jointly --
see the task doc Sec 5.6 for why that would launder an artifact as biology.

Imports fast_patristic, audit_tree_for_phylospec, rescale_for_node_weights,
calibrate_connect_len, graft_domain_trees from Phylospec/domain_merge_utils.py
verbatim, per the task's "do not reimplement them" instruction. prune_tree is
NOT one of those four functions, and IS reimplemented here (prune_to_subtree)
using the Sec 5.7-recommended approach of copying the retained subtree once
rather than calling Bio.Phylo's tree.prune() once per dropped tip -- BAC alone
has ~13k ASVs, so the per-tip-rewrite version would be tens of thousands of
O(n) tree walks.

Requires biopython (Bio.Phylo) -- see Phylospec/requirements.txt. Not runnable
in this repo's system python3 as of this writing (Bio is not installed here);
written and reviewed but not executed in this environment, per instruction.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from Bio import Phylo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Phylospec"))
from domain_merge_utils import (  # noqa: E402
    audit_tree_for_phylospec,
    calibrate_connect_len,
    graft_domain_trees,
    rescale_for_node_weights,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
QIIME_DIR = os.path.join(ROOT, "data", "qiimeresult")


def round_no_key(sample_col):
    parts = sample_col.split("-")
    return parts[0] + "-" + parts[1]


def load_feature_table(domain):
    path = os.path.join(QIIME_DIR, domain, "table_filtered.tsv")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    df.columns = [round_no_key(c) for c in df.columns]
    return df  # features x samples, raw counts


def prune_to_subtree(tree, keep_names):
    """Build the pruned tree by copying the retained subtree in one pass, instead
    of Bio.Phylo's tree.prune() once per dropped tip (Sec 5.7). A node with only
    one surviving child is collapsed into that child, with branch lengths summed,
    so no spurious unary nodes appear in the output tree."""
    from Bio.Phylo.BaseTree import Clade, Tree

    keep = set(keep_names)

    def copy_pruned(clade):
        if not clade.clades:
            if clade.name not in keep:
                return None
            return Clade(branch_length=clade.branch_length, name=clade.name)
        kids = [k for k in (copy_pruned(c) for c in clade.clades) if k is not None]
        if not kids:
            return None
        if len(kids) == 1:
            kid = kids[0]
            kid.branch_length = (kid.branch_length or 0.0) + (clade.branch_length or 0.0)
            return kid
        new = Clade(branch_length=clade.branch_length, name=clade.name)
        new.clades = kids
        return new

    new_root = copy_pruned(tree.root)
    if new_root is None:
        raise ValueError("none of the requested tip names were found in the tree")
    new_root.branch_length = 0.0  # the arm's own root has no meaningful parent stem
    t = Tree(root=new_root, rooted=True)
    found = {c.name for c in t.get_terminals()}
    missing = keep - found
    if missing:
        raise ValueError(f"{len(missing)} requested tips absent from tree, "
                         f"e.g. {sorted(missing)[:5]} -- report this, do not drop silently")
    return t


def audit_and_maybe_rescale(trees_by_label):
    """Audit every arm; if ANY branch across ANY arm is >= 1 (this already
    happens pre-merge -- the raw ARC rooted-tree.nwk itself has branches at
    1.357 and 1.548), rescale ALL arms by the SAME factor so the arms stay
    comparable (domain_merge_utils.rescale_for_node_weights docstring: 'Apply
    the SAME factor to every arm of the experiment, or the arms are no longer
    comparable')."""
    pre = {lab: audit_tree_for_phylospec(t, label=lab) for lab, t in trees_by_label.items()}
    worst = max(a["max_branch_length"] for a in pre.values())
    factor = 1.0
    if worst >= 1.0:
        factor = 0.95 / worst
        for t in trees_by_label.values():
            rescale_for_node_weights(t, factor)
    post = {lab: audit_tree_for_phylospec(t, label=lab) for lab, t in trees_by_label.items()}
    return pre, post, factor


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out-dir", default=os.path.join(ROOT, "data", "merged"))
    ap.add_argument("--use-sepp-tree", default=None,
                     help="path to a SEPP fragment-insertion tree already exported to "
                          "plain Newick, if one exists; skips the calibrated-graft "
                          "fallback entirely")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    arc_tree = Phylo.read(os.path.join(QIIME_DIR, "ARC", "rooted-tree.nwk"), "newick")
    bac_tree = Phylo.read(os.path.join(QIIME_DIR, "BAC", "rooted-tree.nwk"), "newick")
    arc_tips = {c.name for c in arc_tree.get_terminals()}
    bac_tips = {c.name for c in bac_tree.get_terminals()}

    report = {}
    if a.use_sepp_tree:
        merged_tree = Phylo.read(a.use_sepp_tree, "newick")
        report["route"] = "sepp_fragment_insertion"
        report["sepp_tree_path"] = a.use_sepp_tree
    else:
        connect_len, calib_report = calibrate_connect_len(arc_tree, bac_tree)
        merged_tree = graft_domain_trees(arc_tree, bac_tree, connect_len, guard=True)
        report["route"] = "calibrated_graft_fallback"
        report["calibration"] = calib_report
        report["fallback_reason"] = (
            "qiime2 / q2-fragment-insertion is not installed on this machine "
            "and a fresh conda install was declined for this session; used "
            "domain_merge_utils.calibrate_connect_len() (task Sec 5.5 fallback "
            "route) instead of SEPP fragment-insertion.")

    arm_trees = {
        "ARC": prune_to_subtree(merged_tree, arc_tips),
        "BAC": prune_to_subtree(merged_tree, bac_tips),
        "merged": prune_to_subtree(merged_tree, arc_tips | bac_tips),
    }

    pre, post, rescale_factor = audit_and_maybe_rescale(arm_trees)
    report["phylospec_audit_pre_rescale"] = pre
    report["phylospec_audit_final"] = post
    report["rescale_factor_applied"] = rescale_factor

    for lab, t in arm_trees.items():
        Phylo.write(t, os.path.join(a.out_dir, f"{lab}_tree.nwk"), "newick")

    # Sec 5.6: per-domain closure, never joint -- concatenate [arc_rel | bac_rel],
    # each independently summing to 1 per sample, and do NOT renormalise the result.
    arc_counts = load_feature_table("ARC")
    bac_counts = load_feature_table("BAC")
    common_samples = sorted(set(arc_counts.columns) & set(bac_counts.columns))
    arc_c = arc_counts[common_samples]
    bac_c = bac_counts[common_samples]
    arc_rel = arc_c.div(arc_c.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    bac_rel = bac_c.div(bac_c.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    merged_rel = pd.concat([arc_rel, bac_rel], axis=0).T  # samples x features
    merged_rel.index.name = "SampleID"
    merged_rel.to_csv(os.path.join(a.out_dir, "merged_relative_abundance.csv"))

    report["n_common_samples"] = len(common_samples)
    report["n_arc_features"] = len(arc_tips)
    report["n_bac_features"] = len(bac_tips)
    report["merged_abundance_units"] = (
        "per-domain-closed relative abundance, NOT raw reads -- each sample sums "
        "to 1.0 within ARC and 1.0 within BAC (2.0 total), by design (Sec 5.6). "
        "reduce_features_for_tree_models.py's --protect-min-abundance / "
        "--min-abundance thresholds and its 'abundance conserved under collapsing' "
        "behaviour both operate correctly on this, but 'total reads conserved' "
        "(task invariant 4) means 'total relative-abundance units conserved' for "
        "this arm specifically, not literal read counts -- note this in REPORT.md.")

    with open(os.path.join(a.out_dir, "build_merged_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

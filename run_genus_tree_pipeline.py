#!/usr/bin/env python3
"""
run_genus_tree_pipeline.py -- task Sec 6 item 3: driver that runs the reduction
pipeline for ARC, BAC, and merged, at genus level with --safe-ids, and writes
for each: <name>_table.csv, <name>_tree.nwk, <name>_mapping.csv,
<name>_report.json -- plus the Sec 3.1 per-flag / multi_output label views.

Also enforces invariant 6 (no branch length >= 1) on the FINAL reduced tree,
not just the pre-merge tree: build_cross_domain_graph.prune_tree collapses
unary chains by SUMMING their branch lengths, so a merged/filtered tree can
end up with a branch >= 1 even when every branch in the input tree (already
rescaled by build_merged.py) was < 1. Confirmed on a real run: the merged
genus-mode tree came out of reduce_features_for_tree_models.py with one branch
at 1.0239 despite build_merged.py's input tree topping out at 0.95.
`enforce_branch_length_invariant()` re-checks and rescales the OUTPUT tree
in place if needed, after every run_reduce() call.

Usage:
    python run_genus_tree_pipeline.py --mode genus
    python run_genus_tree_pipeline.py --mode asv      # sanity-check arm, Sec 6
    python run_genus_tree_pipeline.py --mode phylo    # sanity-check arm, Sec 6
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(ROOT, "Phylospec")
REDUCE_SCRIPT = os.path.join(PHYLOSPEC_DIR, "reduce_features_for_tree_models.py")
BUILD_MERGED_SCRIPT = os.path.join(ROOT, "build_merged.py")
QIIME_DIR = os.path.join(ROOT, "data", "qiimeresult")
METADATA_PATH = os.path.join(ROOT, "data", "final", "metadata.csv")
DEFAULT_OUT_DIR = os.path.join(ROOT, "output", "genus_tree")
DEFAULT_SEPP_TREE = os.path.join(QIIME_DIR, "sepp_output", "sepp_tree_export", "tree.nwk")

LABEL_COLS = [
    "warning_acid_base_balance",
    "warning_buffer_capacity",
    "warning_acid_accumulation",
    "warning_ammonia_toxicity",
    "warning_biogas_quality",
]

sys.path.insert(0, ROOT)
sys.path.insert(0, PHYLOSPEC_DIR)
from prepare_qiime_inputs import prepare_domain_inputs  # noqa: E402
from make_deepphylo_inputs import parse_newick  # noqa: E402


def run_reduce(counts_csv, tree_nwk, taxonomy_csv, out_prefix, mode):
    cmd = [sys.executable, REDUCE_SCRIPT,
           "-c", counts_csv, "-t", tree_nwk, "-x", taxonomy_csv,
           "-o", out_prefix, "--mode", mode, "--safe-ids", "--protect", "archaea"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    factor = enforce_branch_length_invariant(f"{out_prefix}_tree.nwk")
    if factor != 1.0:
        print(f"  invariant 6: rescaled {out_prefix}_tree.nwk output tree by "
             f"{factor:.6f} (a branch reached >= 1 after reduce_features_for_"
             f"tree_models.py's own prune_tree collapsed a unary chain)")
        # Keep report.json's own audit block in sync -- it was computed by
        # reduce_features_for_tree_models.py BEFORE this rescale, so left
        # alone it would understate max_branch_length and claim a negative-
        # node-weight branch that the actual, final tree file no longer has.
        report_path = f"{out_prefix}_report.json"
        report = json.load(open(report_path))
        audit = report.get("phylospec_audit", {})
        audit["max_branch_length"] = round(audit.get("max_branch_length", 0.0) * factor, 4)
        audit["branches_ge_1_negative_node_weight"] = 0
        audit["post_hoc_rescale_factor_applied"] = factor
        report["phylospec_audit"] = audit
        json.dump(report, open(report_path, "w"), indent=2)


def enforce_branch_length_invariant(tree_path, target_max=0.95):
    """Invariant 6 (task Sec 4 #6) can be violated by reduce_features_for_tree_
    models.py's OWN pruning step even when the input tree was already rescaled:
    build_cross_domain_graph.prune_tree collapses a node with a single
    surviving child by SUMMING its branch length into the child's, so two
    branches each < 1 can combine into one >= 1 after filtering removes
    everything else along that path. Checked and fixed on the actual output
    tree, not assumed safe because the input was safe. Returns the rescale
    factor applied (1.0 if none was needed)."""
    root = parse_newick(open(tree_path).read())
    nodes = []
    stack = [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children)
    worst = max((max(n.blen, 0.0) for n in nodes), default=0.0)
    if worst < 1.0:
        return 1.0
    factor = target_max / worst
    for n in nodes:
        n.blen *= factor

    def to_newick(n):
        inner = "(" + ",".join(to_newick(c) for c in n.children) + ")" if n.children else ""
        return f"{inner}{n.name}:{max(n.blen, 0.0):.8f}"

    open(tree_path, "w").write(to_newick(root) + ";\n")
    return factor


def write_label_views(table_csv, out_dir, labels):
    """Sec 3.1: a Group=Normal/Warning subfolder per flag, plus a multi_output/
    folder carrying all 5 flags as 0/1 columns -- same rows and row order as
    `table_csv` in every subfolder, so ARC/BAC/merged and Phylo-Spec/DeepPhylo
    all evaluate identical sample sets per flag (evaluate_multilabel.site_grouped_cv
    consumes multi_output/table.csv plus a site map built from this same index)."""
    X = pd.read_csv(table_csv, index_col=0)
    joined = X.join(labels, how="inner")
    if len(joined) < len(X):
        missing = set(X.index) - set(labels.index)
        print(f"WARNING: {len(missing)} rows in {table_csv} have no label match: "
              f"{sorted(missing)[:5]}")

    for col in LABEL_COLS:
        flag = col.replace("warning_", "")
        flag_dir = os.path.join(out_dir, flag)
        os.makedirs(flag_dir, exist_ok=True)
        out = joined.drop(columns=LABEL_COLS).copy()
        out["Group"] = np.where(joined[col] == 1, "Warning", "Normal")
        out.to_csv(os.path.join(flag_dir, "table.csv"))
        n_pos = int((joined[col] == 1).sum())
        print(f"    {flag}: warning={n_pos}/{len(joined)}")

    multi_dir = os.path.join(out_dir, "multi_output")
    os.makedirs(multi_dir, exist_ok=True)
    out = joined.drop(columns=LABEL_COLS).copy()
    for col in LABEL_COLS:
        out[col.replace("warning_", "")] = joined[col].astype(int)
    out.to_csv(os.path.join(multi_dir, "table.csv"))


def build_single_domain(domain, out_dir, mode, labels):
    print(f"=== {domain} ===")
    prefix = os.path.join(out_dir, f"{domain.lower()}_input")
    prepare_domain_inputs(domain, prefix)
    out_prefix = os.path.join(out_dir, domain.lower())
    run_reduce(f"{prefix}_abundance.csv",
               os.path.join(QIIME_DIR, domain, "rooted-tree.nwk"),
               f"{prefix}_taxonomy.csv", out_prefix, mode)
    write_label_views(f"{out_prefix}_table.csv", os.path.join(out_dir, domain.lower()), labels)
    return prefix


def build_merged_domain(out_dir, mode, labels, arc_prefix, bac_prefix, use_sepp_tree=None):
    print("=== merged (ARC + BAC) ===")
    merged_build_dir = os.path.join(out_dir, "merged_build")
    cmd = [sys.executable, BUILD_MERGED_SCRIPT, "-o", merged_build_dir]
    if use_sepp_tree:
        cmd += ["--use-sepp-tree", use_sepp_tree]
    subprocess.run(cmd, check=True)

    tax = pd.concat([pd.read_csv(f"{arc_prefix}_taxonomy.csv"),
                     pd.read_csv(f"{bac_prefix}_taxonomy.csv")], ignore_index=True)
    tax_path = os.path.join(out_dir, "merged_taxonomy.csv")
    tax.to_csv(tax_path, index=False)

    out_prefix = os.path.join(out_dir, "merged")
    run_reduce(os.path.join(merged_build_dir, "merged_relative_abundance.csv"),
               os.path.join(merged_build_dir, "merged_tree.nwk"),
               tax_path, out_prefix, mode)
    write_label_views(f"{out_prefix}_table.csv", os.path.join(out_dir, "merged"), labels)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["asv", "phylo", "genus"], default="genus")
    ap.add_argument("--out-dir", default=None,
                     help="default: output/genus_tree/<mode>/")
    ap.add_argument("--use-sepp-tree", default=DEFAULT_SEPP_TREE,
                     help="Sec 5.5 route 1: plain-Newick SEPP fragment-insertion tree "
                          "(qiime fragment-insertion sepp, exported) to use for the ARC+BAC "
                          "merge instead of the calibrated-graft fallback. Defaults to "
                          "data/qiimeresult/sepp_output/sepp_tree_export/tree.nwk if it "
                          "exists; pass --use-sepp-tree '' to force the graft fallback.")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(DEFAULT_OUT_DIR, a.mode)
    os.makedirs(out_dir, exist_ok=True)
    sepp_tree = a.use_sepp_tree if a.use_sepp_tree and os.path.exists(a.use_sepp_tree) else None
    print(f"merge route: {'SEPP fragment insertion (' + sepp_tree + ')' if sepp_tree else 'calibrated graft fallback'}")

    labels = pd.read_csv(METADATA_PATH, encoding="utf-8-sig",
                         index_col="SampleID")[LABEL_COLS]

    arc_prefix = build_single_domain("ARC", out_dir, a.mode, labels)
    bac_prefix = build_single_domain("BAC", out_dir, a.mode, labels)
    build_merged_domain(out_dir, a.mode, labels, arc_prefix, bac_prefix, use_sepp_tree=sepp_tree)

    summary = {"mode": a.mode, "out_dir": out_dir, "arms": ["arc", "bac", "merged"],
               "merge_route": "sepp_fragment_insertion" if sepp_tree else "calibrated_graft_fallback"}
    with open(os.path.join(out_dir, "pipeline_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

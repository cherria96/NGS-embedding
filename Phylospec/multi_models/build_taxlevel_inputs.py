#!/usr/bin/env python3
"""
build_taxlevel_inputs.py -- naive taxonomy-LABEL collapse (not a tree collapse) of
relative abundance at Phylum/Class/Order/Family/Genus rank, for the classical-ML
baseline comparison (RF/KNN/SVM/NNET/XGBoost/GLMNET). These 6 models consume a
flat feature vector and never touch a tree, so the tree-correspondence invariants
in the genus-tree task doc (Sec 4) do not apply to them, and there is nothing
wrong -- for THESE models only -- with the plain "sum ASVs sharing a taxonomy
label" operation that Sec 2 of that doc forbids for Phylo-Spec/DeepPhylo/MetaDR/
PMCNN (a label has no defensible position on a phylogenetic tree for a
non-monophyletic genus; a flat classifier never asks the question).

Reuses reduce_features_for_tree_models.load_taxonomy() for the lineage-string
parsing -- same one place that logic already lives, not reimplemented here.

Per-domain source of abundance:
  ARC / BAC : output/genus_tree/genus_autorun/{arc,bac}_input_abundance.csv --
              raw ASV counts, samples x ASVs. Collapsed by summing (never
              averaging, Sec 4 #4), then row-normalised to relative abundance.
  merged    : output/genus_tree/genus_autorun/merged_build/
              merged_relative_abundance.csv -- ALREADY per-domain-closed relative
              abundance (Sec 5.6: ARC part sums to 1, BAC part sums to 1 per
              sample, concatenated with no joint renormalisation). Collapsing by
              rank and summing preserves that closure automatically -- this
              script does NOT re-normalise the merged table, because doing so
              would launder the ARC:BAC primer-efficiency artefact Sec 5.6 warns
              about back in through the back door.

All ASVs unclassified at a given rank (per reduce_features_for_tree_models.
UNCLASSIFIED) are pooled into one 'Unclassified' column per domain per rank.
This is the standard flat level-N taxonomy-table convention (what a QIIME
level-N.csv already does) -- it is not the same situation as Sec 7's "do not
pool unassigned features into one Unclassified column" rule, which is
specifically about starving Phylo-Spec's tip-matching of a real branch; none of
these 6 models have a tip-matching step.

Writes, per level x domain: multi_models/input_for_all_models_taxlevel/<Level>/
Warnings_<Domain>/table.csv -- SampleID, <rank feature columns>, 5 FLAG_COLS --
the same shape run_site_grouped_cv_benchmark.make_csv_fit_predict already
consumes, so run_taxlevel_baseline_benchmark.py can reuse that harness verbatim.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(ROOT, "..")
REPO_ROOT = os.path.join(PHYLOSPEC_DIR, "..")
GENUS_DIR = os.path.join(REPO_ROOT, "output", "genus_tree", "genus_autorun")
METADATA_PATH = os.path.join(REPO_ROOT, "data", "final", "metadata.csv")
OUT_ROOT = os.path.join(ROOT, "input_for_all_models_taxlevel")

sys.path.insert(0, PHYLOSPEC_DIR)
from reduce_features_for_tree_models import load_taxonomy  # noqa: E402
from evaluate_multilabel import LABEL_COLS  # noqa: E402

FLAG_COLS = sorted(c.replace("warning_", "") for c in LABEL_COLS)
LEVELS = {"Phylum": 1, "Class": 2, "Order": 3, "Family": 4, "Genus": 5}
DOMAINS = ["ARC", "BAC", "merged"]


def collapse_by_rank(abundance, tax_map, rank_idx, already_relative):
    """abundance: samples x ASV DataFrame (raw counts or relative abundance).
    tax_map: {asv_id: 7-list from load_taxonomy}. Returns samples x rank-label
    relative-abundance DataFrame, '' label folded to 'Unclassified', collapsed
    by SUM (never mean, Sec 4 #4)."""
    labels = pd.Series({c: (tax_map.get(c, [""] * 7)[rank_idx] or "Unclassified")
                         for c in abundance.columns})
    grouped = abundance.T.groupby(labels).sum().T  # samples x rank labels
    if already_relative:
        return grouped
    totals = grouped.sum(axis=1).replace(0, np.nan)
    return grouped.div(totals, axis=0).fillna(0.0)


def load_domain_source(domain):
    if domain == "merged":
        ab = pd.read_csv(os.path.join(GENUS_DIR, "merged_build", "merged_relative_abundance.csv"),
                          index_col=0)
        tax = load_taxonomy(os.path.join(GENUS_DIR, "merged_taxonomy.csv"))
        return ab, tax, True
    prefix = domain.lower()
    ab = pd.read_csv(os.path.join(GENUS_DIR, f"{prefix}_input_abundance.csv"), index_col=0)
    tax = load_taxonomy(os.path.join(GENUS_DIR, f"{prefix}_input_taxonomy.csv"))
    return ab, tax, False


def main():
    labels_df = pd.read_csv(METADATA_PATH, encoding="utf-8-sig",
                             index_col="SampleID")[LABEL_COLS]
    labels_df = labels_df.rename(columns=lambda c: c.replace("warning_", ""))
    labels_df = labels_df[FLAG_COLS]  # reorder alphabetically, matching FLAG_COLS

    for domain in DOMAINS:
        print(f"=== {domain} ===")
        abundance, tax_map, already_relative = load_domain_source(domain)
        for level, rank_idx in LEVELS.items():
            collapsed = collapse_by_rank(abundance, tax_map, rank_idx, already_relative)
            joined = collapsed.join(labels_df, how="inner")
            missing = set(collapsed.index) - set(labels_df.index)
            if missing:
                print(f"  WARNING: {len(missing)} {domain} samples have no label "
                      f"match, dropped: {sorted(missing)[:5]}")
            joined = joined.reset_index().rename(columns={joined.index.name or "index": "SampleID"})

            out_dir = os.path.join(OUT_ROOT, level, f"Warnings_{domain}")
            os.makedirs(out_dir, exist_ok=True)
            joined.to_csv(os.path.join(out_dir, "table.csv"), index=False)
            n_features = collapsed.shape[1]
            print(f"  {level}: {joined.shape[0]} samples x {n_features} features "
                  f"-> {out_dir}/table.csv")


if __name__ == "__main__":
    main()

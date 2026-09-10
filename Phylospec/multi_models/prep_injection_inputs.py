#!/usr/bin/env python3
"""
prep_injection_inputs.py -- one-off prep for build_cross_domain_graph.py's
inputs, for the merged-domain arms only (ARC-only / BAC-only have no
Bacteria x Archaea pair to find -- build_cross_domain_graph.py itself refuses
to run without both domains present).

Builds, per {genustree, asvtree}:
  Warnings_merged/domain_taxonomy.csv   Feature ID (safe-id), Domain
                                        (Bacteria/Archaea), keyed by the same
                                        safe-ids as table.csv/phylogeny.nwk --
                                        joins merged_mapping.csv (safe-id ->
                                        original ASV id) with merged_taxonomy.
                                        csv (original ASV id -> Taxon lineage
                                        string) to recover domain per safe-id.
  Warnings_merged/table_for_injection.csv   SampleID, <features>, ammonia_toxicity
                                        -- build_cross_domain_graph.py's -c
                                        format wants exactly ONE trailing label
                                        column; it is never read by the
                                        correlation/fusion computation itself
                                        (only used to write a _DeepPhylo_y.npy
                                        this pipeline doesn't consume), so any
                                        single flag works as a format
                                        placeholder -- ammonia_toxicity chosen
                                        only because it is the best-powered.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
GENUS_SRC = os.path.join(ROOT, "output", "genus_tree", "genus_autorun")
ASV_SRC = os.path.join(ROOT, "output", "genus_tree", "asv")
FLAG_COLS = ["acid_accumulation", "acid_base_balance", "ammonia_toxicity",
            "biogas_quality", "buffer_capacity"]
PLACEHOLDER_LABEL = "ammonia_toxicity"

ARMS = {
    "genustree": (GENUS_SRC, os.path.join(HERE, "input_for_all_models_genus")),
    "asvtree": (ASV_SRC, os.path.join(HERE, "input_for_all_models_asvtree")),
}


def domain_of(taxon_string):
    if not isinstance(taxon_string, str):
        return "Other"
    if "Archaea" in taxon_string:
        return "Archaea"
    if "Bacteria" in taxon_string:
        return "Bacteria"
    return "Other"


def build(arm, src_dir, table_dir):
    mapping = pd.read_csv(os.path.join(src_dir, "merged_mapping.csv"))
    tax = pd.read_csv(os.path.join(src_dir, "merged_taxonomy.csv"))
    tax_lookup = dict(zip(tax["Feature ID"], tax["Taxon"]))

    first_member = mapping.drop_duplicates("reduced_feature")
    dom_df = pd.DataFrame({
        "Feature ID": first_member["reduced_feature"].values,
        "Domain": [domain_of(tax_lookup.get(f)) for f in first_member["feature"]],
    })
    n_arc = (dom_df["Domain"] == "Archaea").sum()
    n_bac = (dom_df["Domain"] == "Bacteria").sum()
    out_dir = os.path.join(table_dir, "Warnings_merged")
    dom_path = os.path.join(out_dir, "domain_taxonomy.csv")
    dom_df.to_csv(dom_path, index=False)
    print(f"[{arm}] {dom_path}: {n_arc} Archaea, {n_bac} Bacteria, "
         f"{len(dom_df) - n_arc - n_bac} Other/unmapped")

    table = pd.read_csv(os.path.join(out_dir, "table.csv"))
    feat_cols = [c for c in table.columns if c not in FLAG_COLS + ["SampleID"]]
    single = table[["SampleID"] + feat_cols + [PLACEHOLDER_LABEL]]
    single_path = os.path.join(out_dir, "table_for_injection.csv")
    single.to_csv(single_path, index=False)
    print(f"[{arm}] {single_path}: {single.shape[0]} samples x {len(feat_cols)} features")


if __name__ == "__main__":
    for arm, (src_dir, table_dir) in ARMS.items():
        build(arm, src_dir, table_dir)

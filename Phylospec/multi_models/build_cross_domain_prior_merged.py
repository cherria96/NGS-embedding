#!/usr/bin/env python3
"""
build_cross_domain_prior_merged.py -- one-time step for
run_cross_domain_injection_benchmark.py: infer the Bacteria x Archaea
association network on the merged-domain genus table (per
CROSS_DOMAIN_INJECTION_BY_MODEL.md) and write everything the three per-model
injection arms need, all keyed to the SAME feature order as
input_for_all_models_genus/Warnings_merged/{table.csv,X.npy}:

  <out>_Dp.npy           (n_feat, n_feat) patristic distance matrix
  <out>_Dfused.npy       (n_feat, n_feat) syntrophy-shortcut distance matrix
  <out>_edges.csv         retained cross-domain edges (bacterium, archaeon, rho, q)
  <out>_seriation.txt     optimal-leaf-ordering seriation of Dfused, one feature/line
  <out>_diagnostics.json

Domain (Bacteria/Archaea) per feature is derived one of two ways:
  - genus mode (default): join merged_mapping.csv (original ASV/OTU id ->
    reduced F##### id, from reduce_features_for_tree_models.py --safe-ids)
    against merged_taxonomy.csv (original id -> taxonomy string) and take the
    (unanimous, checked) domain of each genus's member ASVs -- a genus cannot
    itself straddle domains.
  - ASV mode (--arc-dir/--bac-dir given): merged's ASV-hash feature columns
    are, by construction, a subset of the disjoint union of the ARC-only and
    BAC-only tables' own columns (each domain's raw ASVs never appear in the
    other domain's table), so domain-of-feature is just ARC/BAC membership --
    no reduced-feature mapping exists at ASV level to join against.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = os.path.join(PHYLOSPEC_DIR, "..")
sys.path.insert(0, PHYLOSPEC_DIR)
from build_cross_domain_graph import (  # noqa: E402
    parse_newick, leaf_index, clr, design_matrix, residualise,
    cross_domain_network, sparsify, shortcut_distance,
)

GENUS_DIR = os.path.join(ROOT, "output", "genus_tree", "genus_autorun")
METADATA_PATH = os.path.join(ROOT, "data", "final", "metadata.csv")
COVARIATES = ["T_C", "HRT_d", "Q_total_Tpy", "Season", "substrate_avg_pH",
             "substrate_avg_COD", "substrate_avg_sCOD", "substrate_avg_Protein",
             "substrate_avg_Lipid", "substrate_avg_TC", "substrate_avg_TVFAs"]
# Reactor config + influent only, per build_cross_domain_graph.py's outcome-
# covariate guard: no eff_*/warning_* (those are downstream of the community
# -- collider bias + label leakage). substrate_avg_ALK dropped (33/138 = 24%
# missing, mean-imputed -- too sparse to condition on reliably); substrate_type
# (categorical) deliberately left out too.
FLAG_COLS = ["acid_accumulation", "acid_base_balance", "ammonia_toxicity",
            "biogas_quality", "buffer_capacity"]


def domain_of(taxon):
    t = str(taxon).lower()
    if "archaea" in t:
        return "Archaea"
    if "bacteria" in t:
        return "Bacteria"
    return "Other"


def feature_domains(feats):
    """Genus-level: join the reduced-feature mapping against taxonomy."""
    mapping = pd.read_csv(os.path.join(GENUS_DIR, "merged_mapping.csv"), dtype=str)
    tax = pd.read_csv(os.path.join(GENUS_DIR, "merged_taxonomy.csv"), dtype=str) \
        .set_index("Feature ID")["Taxon"]
    mapping["domain"] = mapping["feature"].map(tax).map(domain_of)
    mixed = mapping.groupby("reduced_feature")["domain"].nunique()
    bad = mixed[mixed > 1].index.tolist()
    if bad:
        sys.exit(f"ERROR: {len(bad)} reduced (genus) features span >1 domain, e.g. {bad[:5]} "
                 "-- domain_of() heuristic is unreliable for this taxonomy.")
    by_feat = mapping.groupby("reduced_feature")["domain"].first()
    dom = pd.Series(feats).map(by_feat)
    if dom.isna().any():
        missing = [f for f, d in zip(feats, dom) if pd.isna(d)]
        sys.exit(f"ERROR: {len(missing)} features have no domain mapping, e.g. {missing[:5]}")
    return dom.to_numpy()


def feature_domains_from_membership(feats, arc_dir, bac_dir):
    """ASV-level: domain-of-feature by which single-domain table the raw ASV
    hash id appears in (ARC-only vs BAC-only tables are disjoint by
    construction)."""
    arc_cols = set(pd.read_csv(os.path.join(arc_dir, "table.csv"), nrows=0).columns)
    bac_cols = set(pd.read_csv(os.path.join(bac_dir, "table.csv"), nrows=0).columns)
    dom, missing = [], []
    for f in feats:
        if f in arc_cols:
            dom.append("Archaea")
        elif f in bac_cols:
            dom.append("Bacteria")
        else:
            dom.append("Other")
            missing.append(f)
    if missing:
        sys.exit(f"ERROR: {len(missing)} features found in neither {arc_dir}/table.csv nor "
                 f"{bac_dir}/table.csv, e.g. {missing[:5]}")
    return np.array(dom)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_prefix", nargs="?",
                    default=os.path.join(HERE, "results", "cross_domain_injection", "merged"))
    ap.add_argument("--data-dir", default=os.path.join(HERE, "input_for_all_models_genus"),
                    help="Directory containing Warnings_merged/{table.csv,phylogeny.nwk}. "
                         "Default: the genus-level arm.")
    ap.add_argument("--arc-dir", default=None,
                    help="Warnings_ARC dir (needs --bac-dir too): switches domain-of-feature "
                         "to ASV-membership mode instead of the genus mapping/taxonomy join. "
                         "Required for ASV-level --data-dir, since there is no reduced-feature "
                         "mapping to join against at that level.")
    ap.add_argument("--bac-dir", default=None)
    ap.add_argument("--scale", type=float, default=0.25,
                    help="shortcut branch length in units of mean patristic distance -- "
                         "smaller pulls retained syntrophic pairs closer together. "
                         "scale=inf disables the shortcut (Dfused == Dp).")
    ap.add_argument("--topk", type=int, default=5,
                    help="max bacterial partners retained per archaeon.")
    ap.add_argument("--fdr", type=float, default=0.05,
                    help="BH-adjusted q-value threshold for edge retention.")
    return ap.parse_args()


def main():
    args = parse_args()
    out = args.out_prefix
    warnings_merged = os.path.join(args.data_dir, "Warnings_merged")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    table = pd.read_csv(os.path.join(warnings_merged, "table.csv"))
    feats = [c for c in table.columns if c not in FLAG_COLS + ["SampleID"]]
    tree_path = os.path.join(warnings_merged, "phylogeny.nwk")
    root = parse_newick(open(tree_path, encoding="utf-8", errors="replace").read())
    leaves = leaf_index(root)
    missing = [f for f in feats if f not in leaves]
    if missing:
        sys.exit(f"ERROR: {len(missing)} table.csv features are not tree tips, e.g. {missing[:5]}")

    if args.arc_dir and args.bac_dir:
        dom = feature_domains_from_membership(feats, args.arc_dir, args.bac_dir)
    else:
        dom = feature_domains(feats)
    bac_idx = np.where(dom == "Bacteria")[0]
    arc_idx = np.where(dom == "Archaea")[0]
    print(f"{len(feats)} features: {len(bac_idx)} bacteria, {len(arc_idx)} archaea, "
          f"{len(feats) - len(bac_idx) - len(arc_idx)} other", file=sys.stderr)

    X = table[feats].to_numpy(np.float64)
    Z = clr(X)

    meta = pd.read_csv(METADATA_PATH, encoding="utf-8-sig").set_index("SampleID")
    meta = meta.reindex(table["SampleID"].astype(str))
    Dm, dm_names = design_matrix(meta, COVARIATES)
    Zc = residualise(Z, Dm)

    rho, q = cross_domain_network(Zc[:, bac_idx], Zc[:, arc_idx], n_perm=500, seed=0)
    keep = sparsify(rho, q, fdr=args.fdr, topk=args.topk, positive_only=True)
    print(f"cross-domain pairs tested: {rho.size}, retained edges: {int(keep.sum())} "
          f"(fdr<={args.fdr}, topk={args.topk})", file=sys.stderr)

    # shortcut_distance() also returns the plain patristic matrix Dp every
    # call (it does not depend on `scale`), so the without-syntrophy arm's Dp
    # and the with-syntrophy arm's Dfused come from one call, on one code path.
    Dfused, Dp = shortcut_distance(root, feats, bac_idx, arc_idx, rho, keep, scale=args.scale)

    np.save(f"{out}_Dp.npy", Dp)
    np.save(f"{out}_Dfused.npy", Dfused)
    with open(f"{out}_features.txt", "w") as f:
        f.write("\n".join(feats))

    bi, ai = np.where(keep)
    edges = pd.DataFrame({
        "bacterium": [feats[bac_idx[b]] for b in bi],
        "archaeon": [feats[arc_idx[a]] for a in ai],
        "rho": rho[bi, ai],
        "q": q[bi, ai],
    }).sort_values("rho", ascending=False)
    edges.to_csv(f"{out}_edges.csv", index=False)

    # MetaDR with-arm: optimal-leaf-ordering seriation of Dfused.
    Z_link = linkage(squareform(Dfused, checks=False), method="average")
    Z_olo = optimal_leaf_ordering(Z_link, squareform(Dfused, checks=False))
    order = leaves_list(Z_olo)
    with open(f"{out}_seriation.txt", "w") as f:
        f.write("\n".join(feats[i] for i in order))

    diag = {
        "n_features": len(feats), "n_bacteria": int(len(bac_idx)), "n_archaea": int(len(arc_idx)),
        "n_other": int(len(feats) - len(bac_idx) - len(arc_idx)),
        "covariates": COVARIATES, "scale": args.scale, "topk": args.topk, "fdr": args.fdr,
        "cross_domain_pairs_tested": int(rho.size),
        "edges_retained": int(keep.sum()),
        "mean_rho_kept": float(rho[keep].mean()) if keep.any() else None,
        "archaea_with_no_partner": int((keep.sum(axis=0) == 0).sum()),
    }
    json.dump(diag, open(f"{out}_diagnostics.json", "w"), indent=2)
    print(json.dumps(diag, indent=2), file=sys.stderr)
    print(f"-> {out}_{{Dp,Dfused}}.npy, {out}_edges.csv, {out}_seriation.txt", file=sys.stderr)


if __name__ == "__main__":
    main()

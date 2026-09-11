#!/usr/bin/env python3
"""
build_cross_domain_prior_merged.py -- one-time step for
run_cross_domain_injection_benchmark.py: infer the Bacteria x Archaea
association network on the merged-domain genus table and write everything
the three per-model injection arms need, all keyed to the SAME feature order
as input_for_all_models_genus/Warnings_merged/{table.csv,X.npy}.

--method selects how the association network itself is estimated (same
downstream test/sparsification either way -- see parse_args()):
  conditioned  (default) -- the original analysis: CLR abundances are
               residualised against reactor operating covariates first, so a
               bacterium-archaeon pair that merely tracks the same operating
               conditions is not counted as an edge.
  cooccurrence -- a simpler, standard co-occurrence-network analysis: raw
               CLR-abundance correlation, no covariate conditioning.

Output files (same names regardless of --method -- point different --method
runs at different out_prefix values to keep both on disk):

  <out>_Dp.npy           (n_feat, n_feat) patristic distance matrix
  <out>_Dfused.npy       (n_feat, n_feat) syntrophy-shortcut distance matrix
  <out>_edges.csv         retained cross-domain edges (bacterium, archaeon, rho, q)
  <out>_seriation.txt     MetaDR with-arm column order: features placed on a 2-D
                          grid (spectral embedding + Hungarian assignment,
                          CLAUDE_CODE_TASK_cross_domain_injection.md Sec B.2 --
                          supersedes an earlier 1-D optimal-leaf-ordering
                          seriation of Dfused), one feature per line,
                          pre-compensated for transform_image()'s zigzag so the
                          written order lands each feature at its intended
                          (row, col) after MetaDR's own reshape+zigzag.
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
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

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
    ap.add_argument("--method", choices=["conditioned", "cooccurrence"], default="conditioned",
                    help="'conditioned' (default, original analysis): residualise CLR "
                         "abundances against reactor operating covariates (COVARIATES) before "
                         "testing association, so a pair sharing only response to operating "
                         "conditions (not a direct relationship) is not counted as an edge. "
                         "'cooccurrence': skip residualisation and test raw CLR-abundance "
                         "correlation directly -- the simpler, standard co-occurrence-network "
                         "analysis. Both use the same Spearman-plus-permutation test and "
                         "FDR/topk sparsification (cross_domain_network/sparsify); only the "
                         "input matrix differs. Output file names are unaffected -- point "
                         "different --method runs at different out_prefix values to keep both.")
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

    if args.method == "conditioned":
        meta = pd.read_csv(METADATA_PATH, encoding="utf-8-sig").set_index("SampleID")
        meta = meta.reindex(table["SampleID"].astype(str))
        Dm, dm_names = design_matrix(meta, COVARIATES)
        Z_tested = residualise(Z, Dm)
    else:  # cooccurrence: plain CLR-abundance correlation, no covariate conditioning
        dm_names = None
        Z_tested = Z

    rho, q = cross_domain_network(Z_tested[:, bac_idx], Z_tested[:, arc_idx], n_perm=500, seed=0)
    keep = sparsify(rho, q, fdr=args.fdr, topk=args.topk, positive_only=True)
    print(f"method={args.method}: cross-domain pairs tested: {rho.size}, retained edges: "
          f"{int(keep.sum())} (fdr<={args.fdr}, topk={args.topk})", file=sys.stderr)

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

    # MetaDR with-arm (CLAUDE_CODE_TASK_cross_domain_injection.md Sec B.2):
    # place taxa on a 2-D grid instead of a 1-D seriation. A 1-D ordering
    # gives each taxon only two neighbours, so --topk 5 cross-domain
    # partners can never all land in one conv's footprint; a direct 2-D
    # placement lets a taxon have up to 8 immediate neighbours (24 within a
    # 5x5 footprint, matching the first conv's receptive field).
    #
    # W: p x p symmetric similarity. Within-domain: patristic distance Dp
    # converted to a similarity with the SAME (scale, mean_pat) reference
    # shortcut_distance() uses for its cross-domain shortcut-edge lengths --
    # inverting that transform (length = scale*mean_pat*(1-s)) gives
    # s = 1 - Dp/(scale*mean_pat), clipped to >=0. Cross-domain: retained
    # edges get their actual estimated rho (already a similarity in (0,1]
    # via sparsify(..., positive_only=True)), overriding whatever the
    # patristic-distance proxy would have given that pair.
    mean_pat = Dp[np.triu_indices_from(Dp, 1)].mean()
    W = np.clip(1.0 - Dp / (args.scale * mean_pat), 0.0, None)
    np.fill_diagonal(W, 0.0)
    for b, a, r in zip(bi, ai, rho[bi, ai]):
        i, j = bac_idx[b], arc_idx[a]
        W[i, j] = W[j, i] = float(r)

    L = np.diag(W.sum(1)) - W
    _, vecs = eigsh(L, k=3, which="SM")
    Y = vecs[:, 1:3]
    n = int(np.ceil(len(feats) ** 0.5))
    Y = (Y - Y.min(0)) / np.ptp(Y, 0) * (n - 1)

    # Restrict assignment targets to the p grid cells closest to the grid
    # centre, leaving the (n*n - p) most peripheral cells as the padding
    # cells transform_image() zero-fills -- an edge/corner cell has the
    # fewest real neighbours in any conv footprint anyway, so this is where
    # padding costs the least real adjacency (task doc: "do not scatter them
    # through the middle").
    all_cells = np.array([(r, c) for r in range(n) for c in range(n)], float)
    centre = (n - 1) / 2.0
    central_order = np.argsort(np.abs(all_cells - centre).sum(1))  # L1 distance from centre
    cells = all_cells[central_order[:len(feats)]]

    cost = ((Y[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    assign_rows, assign_cols = linear_sum_assignment(cost)

    # transform_image() reshapes the order file row-major into img_size x
    # img_size THEN reverses every odd row (zigzag). Pre-compensate here so
    # a taxon assigned to grid (r, c) above ends up physically at (r, c)
    # AFTER that reversal, not before it -- write it into the order file at
    # column (n-1-c) instead of c on odd target rows.
    order = [None] * (n * n)
    grid_pos = {}
    for i, cell_idx in zip(assign_rows, assign_cols):
        r, c = divmod(int(cell_idx), n)
        grid_pos[feats[i]] = (r, c)
        flat_col = (n - 1 - c) if (r % 2 == 1) else c
        order[r * n + flat_col] = feats[i]
    with open(f"{out}_seriation.txt", "w") as f:
        f.write("\n".join(name for name in order if name is not None))

    # Placement-quality: fraction of retained edges whose two endpoints land
    # within one 5x5 conv footprint (Chebyshev distance <=2) of each other
    # in the FINAL (post-zigzag-compensation) grid. If this is low, the
    # injection did not happen and the with-arm is uninformative regardless
    # of score (task doc Sec B.2).
    n_edges_total = int(keep.sum())
    n_within_footprint = 0
    for b, a in zip(bi, ai):
        bac_name, arc_name = feats[bac_idx[b]], feats[arc_idx[a]]
        (rb, cb), (ra, ca) = grid_pos[bac_name], grid_pos[arc_name]
        if max(abs(rb - ra), abs(cb - ca)) <= 2:
            n_within_footprint += 1
    placement_quality = n_within_footprint / n_edges_total if n_edges_total else None
    print(f"MetaDR 2-D placement quality: {n_within_footprint}/{n_edges_total} retained edges "
          f"within one 5x5 footprint ({placement_quality if placement_quality is None else round(placement_quality, 3)})",
          file=sys.stderr)

    diag = {
        "n_features": len(feats), "n_bacteria": int(len(bac_idx)), "n_archaea": int(len(arc_idx)),
        "n_other": int(len(feats) - len(bac_idx) - len(arc_idx)),
        "method": args.method,
        "covariates": COVARIATES if args.method == "conditioned" else None,
        "scale": args.scale, "topk": args.topk, "fdr": args.fdr,
        "cross_domain_pairs_tested": int(rho.size),
        "edges_retained": int(keep.sum()),
        "mean_rho_kept": float(rho[keep].mean()) if keep.any() else None,
        "archaea_with_no_partner": int((keep.sum(axis=0) == 0).sum()),
        "metadr_grid_size": n,
        "metadr_placement_quality": placement_quality,
    }
    json.dump(diag, open(f"{out}_diagnostics.json", "w"), indent=2)
    print(json.dumps(diag, indent=2), file=sys.stderr)
    print(f"-> {out}_{{Dp,Dfused}}.npy, {out}_edges.csv, {out}_seriation.txt", file=sys.stderr)


if __name__ == "__main__":
    main()

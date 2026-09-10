#!/usr/bin/env python3
"""
build_cross_domain_graph.py -- infer a Bacteria x Archaea association network from an
AD 16S table, fuse it with the phylogeny, and emit DeepPhylo inputs in which the
syntrophic partners are no longer at maximum prior distance.

Why the extra machinery over a plain correlation network
--------------------------------------------------------
In an anaerobic digester a fatty-acid oxidiser and a methanogen co-vary mostly because
both respond to organic loading rate, temperature, ammonia and retention time. A raw
cross-domain correlation network is therefore dominated by the shared driver, not by
the interaction. This script conditions the association on the measured operating
variables (per-taxon OLS on the centred-log-ratio abundances, correlations computed on
the residuals) before testing, which is the practical stand-in for FlashWeave-style
conditional independence testing and is the step that decides whether the resulting
graph means anything.

Three further design choices, all reportable:
  * only the Bacteria x Archaea block is tested (n_bac * n_arc tests instead of
    n^2 / 2), so FDR power is spent where the hypothesis is;
  * significance comes from a permutation null that shuffles samples in the archaeal
    block only, preserving each domain's internal correlation structure and marginals;
  * fusion is done in DISTANCE space as a shortcut added to the tree, so the
    tree-only baseline is nested exactly (scale -> inf reproduces
    make_deepphylo_inputs.py bit-for-bit) and the prior stays a proper metric.

Inputs
------
  -c  abundance CSV, Phylo-Spec layout: col 0 = sample id, cols 1..n-2 = features,
      col n-1 = label
  -t  Newick tree, tip labels == feature names
  -x  taxonomy CSV: first column = feature id, plus a column naming the domain
      (Kingdom / Domain / taxonomy string containing k__Bacteria / d__Archaea)
  -m  metadata TSV (optional but strongly recommended): sample id + operating variables
  --covariates  comma-separated metadata columns to condition on.  These must be
      UPSTREAM of the community -- influent composition, OLR, HRT, temperature,
      digester type, sequencing depth.  Effluent chemistry (eff_*) is a DESCENDANT
      of the community and a *collider*: conditioning on it manufactures spurious
      cross-domain edges and deletes the mediated signal you are trying to detect.
      It is also what the warning_* labels are thresholds on, so conditioning on it
      is label leakage.  eff_* / warning_* columns are refused unless
      --allow-outcome-covariates is passed.

Outputs (prefix from -o)
------------------------
  <p>_DeepPhylo_X.npy / _y.npy / _labelmap.csv / _features.txt   as before
  <p>_DeepPhylo_embeding.npy                                     fused prior, main arm
  <p>_alpha<A>_DeepPhylo_embeding.npy                            one per swept value
  <p>_null_DeepPhylo_embeding.npy                                degree-preserving null
  <p>_edges.csv          scored cross-domain edges (r, p, q, patristic distance)
  <p>_diagnostics.json   edge counts before/after conditioning, tree overlap, etc.

Usage
-----
  python build_cross_domain_graph.py -c abundance.csv -t tree.nwk -x taxonomy.csv \
      -m metadata.tsv --covariates OLR,temperature,HRT,inf_TS,seq_depth \
      -o run1 --sweep 1.0,0.5,0.25
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_deepphylo_inputs import parse_newick, leaf_index, patristic_matrix  # noqa: E402


# --------------------------------------------------------------------- tree utils
def prune_tree(root, keep):
    """Reduce the tree to the paths leading to `keep` tips, collapsing single-child
    chains and summing their branch lengths. Returns the new root (may be the old one).
    Operates in place on a parsed tree."""
    def rec(node):
        if not node.children:
            return node if node.name in keep else None
        kids = [k for k in (rec(c) for c in node.children) if k is not None]
        if not kids:
            return None
        if len(kids) == 1:
            kids[0].blen += node.blen          # collapse the chain
            kids[0].parent = node.parent
            return kids[0]
        node.children = kids
        for k in kids:
            k.parent = node
        return node
    out = rec(root)
    if out is None:
        sys.exit("ERROR: pruning removed every tip -- feature names do not match the tree.")
    out.blen = 0.0
    return out


def tree_edges(root):
    """(n_nodes, list of (parent_idx, child_idx, branch_length), {tip_name: idx})."""
    nodes, stack = [], [root]
    idx = {}
    while stack:
        n = stack.pop()
        idx[id(n)] = len(nodes)
        nodes.append(n)
        stack.extend(n.children)
    edges = [(idx[id(n)], idx[id(c)], max(c.blen, 0.0)) for n in nodes for c in n.children]
    tips = {n.name: idx[id(n)] for n in nodes if not n.children and n.name}
    return len(nodes), edges, tips


# ---------------------------------------------------------------- transforms
def clr(X, pseudo=None):
    """Centred log-ratio on a samples x taxa matrix of (relative) abundances."""
    X = np.asarray(X, dtype=np.float64)
    if pseudo is None:
        nz = X[X > 0]
        pseudo = (nz.min() / 2.0) if nz.size else 1e-6
    L = np.log(X + pseudo)
    return L - L.mean(axis=1, keepdims=True)


def design_matrix(meta, covariates):
    """Intercept + standardised numerics + one-hot categoricals."""
    cols = [np.ones((len(meta), 1))]
    names = ["intercept"]
    for c in covariates:
        s = meta[c]
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype(float).to_numpy()
            v = np.nan_to_num(v, nan=np.nanmean(v))
            sd = v.std()
            cols.append(((v - v.mean()) / (sd if sd > 0 else 1.0))[:, None])
            names.append(c)
        else:
            d = pd.get_dummies(s.astype(str), prefix=c, drop_first=True).to_numpy(float)
            if d.size:
                cols.append(d)
                names.extend([f"{c}_lvl{i}" for i in range(d.shape[1])])
    return np.hstack(cols), names


def residualise(Z, D):
    """Least-squares residuals of every column of Z on design matrix D."""
    beta, *_ = np.linalg.lstsq(D, Z, rcond=None)
    return Z - D @ beta


def rank_standardise(M):
    """Column-wise rank transform then z-score, so R.T @ R / n == Spearman rho."""
    R = np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0, M)
    R = R - R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return R / sd


# ------------------------------------------------------- cross-domain association
def cross_domain_network(Zb, Za, n_perm=500, seed=0):
    """Spearman association for every bacteria x archaea pair plus a permutation null.

    Returns (rho[n_bac, n_arc], q[n_bac, n_arc]) with q = BH-adjusted empirical p from
    a pooled null built by shuffling samples in the archaeal block only.
    """
    rng = np.random.default_rng(seed)
    n = Zb.shape[0]
    Rb, Ra = rank_standardise(Zb), rank_standardise(Za)
    rho = (Rb.T @ Ra) / n

    null = np.empty(n_perm * rho.size, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i * rho.size:(i + 1) * rho.size] = ((Rb.T @ Ra[perm]) / n).ravel()
    null = np.sort(np.abs(null))

    # empirical two-sided p from the pooled null, then Benjamini-Hochberg
    ranks = np.searchsorted(null, np.abs(rho).ravel(), side="left")
    p = (null.size - ranks + 1) / (null.size + 1)
    order = np.argsort(p)
    m = p.size
    q_sorted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.clip(q_sorted, 0, 1)
    return rho, q.reshape(rho.shape)


def sparsify(rho, q, fdr=0.05, topk=None, positive_only=True):
    """Boolean keep-mask over the bacteria x archaea block."""
    keep = q <= fdr
    if positive_only:
        keep &= rho > 0
    if topk:
        for j in range(rho.shape[1]):                     # top-k partners per archaeon
            col = np.where(keep[:, j])[0]
            if col.size > topk:
                drop = col[np.argsort(-rho[col, j])[topk:]]
                keep[drop, j] = False
    return keep


def rewire_null(keep, seed=0):
    """Degree-preserving rewiring of the bipartite keep-mask (Curveball swaps)."""
    rng = np.random.default_rng(seed)
    A = keep.copy()
    rows = [set(np.where(A[i])[0]) for i in range(A.shape[0])]
    for _ in range(20 * int(A.sum() + 1)):
        i, j = rng.integers(0, len(rows), 2)
        if i == j:
            continue
        only_i, only_j = rows[i] - rows[j], rows[j] - rows[i]
        if not only_i or not only_j:
            continue
        a, b = rng.choice(list(only_i)), rng.choice(list(only_j))
        rows[i].discard(a); rows[i].add(b)
        rows[j].discard(b); rows[j].add(a)
    out = np.zeros_like(A)
    for i, s in enumerate(rows):
        out[i, list(s)] = True
    return out


# ------------------------------------------------------------------ prior fusion
def shortcut_distance(root, kept, bac_idx, arc_idx, weight, keep, scale):
    """Geodesic tip-to-tip distance on the tree augmented with syntrophic shortcuts.

    A retained edge (b, a) with association s in (0, 1] is added as an extra branch of
    length  scale * mean_patristic * (1 - s).  scale = inf adds nothing, so the
    baseline is nested exactly.
    """
    n_nodes, edges, tips = tree_edges(root)
    tip_pos = [tips[f] for f in kept]
    src = [e[0] for e in edges]; dst = [e[1] for e in edges]; w = [e[2] for e in edges]

    Dp = patristic_matrix(root, [leaf_index(root)[f] for f in kept])
    mean_pat = Dp[np.triu_indices_from(Dp, 1)].mean()

    if np.isfinite(scale):
        bi, ai = np.where(keep)
        s = np.clip(weight[bi, ai], 1e-6, 1.0)
        L = scale * mean_pat * (1.0 - s)
        for b, a, l in zip(bi, ai, L):
            src.append(tip_pos[bac_idx[b]]); dst.append(tip_pos[arc_idx[a]]); w.append(float(l))

    G = coo_matrix((np.array(w + w), (np.array(src + dst), np.array(dst + src))),
                   shape=(n_nodes, n_nodes)).tocsr()
    D = dijkstra(G, directed=False, indices=np.array(tip_pos))[:, tip_pos]
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D, Dp


def embed(D, dim):
    return PCA(n_components=min(dim, D.shape[0]), random_state=0).fit_transform(D)


# ------------------------------------------------------------------------ driver
def domain_of(tax_df, feature):
    row = " ".join(str(v) for v in tax_df.loc[feature].values).lower()
    if "archaea" in row:
        return "Archaea"
    if "bacteria" in row:
        return "Bacteria"
    return "Other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--csv", required=True)
    ap.add_argument("-t", "--tree", required=True)
    ap.add_argument("-x", "--taxonomy", required=True)
    ap.add_argument("-m", "--metadata", default=None)
    ap.add_argument("--covariates", default="")
    ap.add_argument("--allow-outcome-covariates", action="store_true",
                    dest="allow_outcome_covariates",
                    help="permit eff_* / warning_* covariates (label leakage + collider "
                         "bias; only for deliberate sensitivity analysis)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-d", "--dim", type=int, default=200)
    ap.add_argument("--prevalence", type=float, default=0.2,
                    help="drop taxa present in fewer than this fraction of samples")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--topk", type=int, default=5, help="max bacterial partners per archaeon")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="shortcut length in units of mean patristic distance")
    ap.add_argument("--sweep", default="", help="comma-separated extra scales, e.g. 1.0,0.5,0.1")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- load ---------------------------------------------------------------
    df = pd.read_csv(args.csv)
    idcol, labcol = df.columns[0], df.columns[-1]
    feats = list(df.columns[1:-1])

    root = parse_newick(open(args.tree, encoding="utf-8", errors="replace").read())
    leaves = leaf_index(root)
    tax = pd.read_csv(args.taxonomy, index_col=0)
    tax.index = tax.index.astype(str)

    kept = [f for f in feats if f in leaves and f in tax.index]
    A_raw = df[kept].to_numpy(float)
    prev = (A_raw > 0).mean(axis=0)
    kept = [f for f, ok in zip(kept, prev >= args.prevalence) if ok]
    if len(kept) < 4:
        sys.exit("ERROR: fewer than 4 features survive tree matching + prevalence filter.")

    dom = np.array([domain_of(tax, f) for f in kept])
    bac_idx = np.where(dom == "Bacteria")[0]
    arc_idx = np.where(dom == "Archaea")[0]
    if arc_idx.size == 0 or bac_idx.size == 0:
        sys.exit("ERROR: need both Bacteria and Archaea features; check the taxonomy file.")

    root = prune_tree(root, set(kept))
    leaves = leaf_index(root)

    X = df[kept].to_numpy(np.float32)
    Z = clr(X)

    # ---- condition on operating variables ------------------------------------
    covs = [c for c in args.covariates.split(",") if c.strip()]
    outcome_like = [c for c in covs
                    if c.strip().lower().startswith(("eff_", "warning_"))]
    if outcome_like and not args.allow_outcome_covariates:
        sys.exit(
            "ERROR: refusing to condition on outcome-side columns: "
            f"{outcome_like}\n"
            "  These are descendants of the microbial community, not confounders.\n"
            "  Conditioning on them (a) induces collider bias -- spurious bacteria x\n"
            "  archaea edges from a shared downstream constraint, (b) removes the\n"
            "  mediated covariation that a real syntrophic pair produces, and (c) leaks\n"
            "  the warning_* labels, which are thresholds on these same eff_* columns,\n"
            "  into the edge selection that later serves as a fixed prior.\n"
            "  Condition on upstream operating variables instead (OLR, HRT, temperature,\n"
            "  influent composition, sequencing depth). Pass --allow-outcome-covariates\n"
            "  only for a deliberate sensitivity analysis.")
    conditioned = False
    if args.metadata and covs:
        meta = pd.read_csv(args.metadata, sep=None, engine="python")
        meta = meta.set_index(meta.columns[0]).loc[df[idcol].astype(str).values]
        missing = [c for c in covs if c not in meta.columns]
        if missing:
            sys.exit(f"ERROR: covariates absent from metadata: {missing}")
        Dm, _ = design_matrix(meta, covs)
        Zc = residualise(Z, Dm)
        conditioned = True
    else:
        print("WARNING: no covariates given -- the network will be confounded by reactor "
              "operating conditions. Pass -m/--covariates.", file=sys.stderr)
        Zc = Z - Z.mean(axis=0, keepdims=True)

    # ---- association, raw and conditioned ------------------------------------
    rho_raw, q_raw = cross_domain_network(Z[:, bac_idx], Z[:, arc_idx], args.perms, args.seed)
    keep_raw = sparsify(rho_raw, q_raw, args.fdr, args.topk)
    if conditioned:
        rho, q = cross_domain_network(Zc[:, bac_idx], Zc[:, arc_idx], args.perms, args.seed)
    else:
        rho, q = rho_raw, q_raw
    keep = sparsify(rho, q, args.fdr, args.topk)

    # ---- fuse, embed, write --------------------------------------------------
    Dfused, Dp = shortcut_distance(root, kept, bac_idx, arc_idx, rho, keep, args.scale)
    dim = min(args.dim, len(kept))

    if args.regression:
        y = df[labcol].to_numpy(np.float32)
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[labcol].astype(str)).astype(np.int64)
        pd.DataFrame({"label": le.classes_, "code": range(len(le.classes_))}).to_csv(
            f"{args.out}_DeepPhylo_labelmap.csv", index=False)

    np.save(f"{args.out}_DeepPhylo_X.npy", X)
    np.save(f"{args.out}_DeepPhylo_y.npy", y)
    np.save(f"{args.out}_DeepPhylo_embeding.npy", embed(Dfused, dim))
    pd.Series(kept).to_csv(f"{args.out}_DeepPhylo_features.txt", index=False, header=False)

    for sc in [float(s) for s in args.sweep.split(",") if s.strip()]:
        Ds, _ = shortcut_distance(root, kept, bac_idx, arc_idx, rho, keep, sc)
        np.save(f"{args.out}_scale{sc}_DeepPhylo_embeding.npy", embed(Ds, dim))
    np.save(f"{args.out}_treeonly_DeepPhylo_embeding.npy", embed(Dp, dim))

    kn = rewire_null(keep, args.seed)
    Dn, _ = shortcut_distance(root, kept, bac_idx, arc_idx, rho, kn, args.scale)
    np.save(f"{args.out}_null_DeepPhylo_embeding.npy", embed(Dn, dim))

    bi, ai = np.where(keep)
    pd.DataFrame({
        "bacterium": [kept[bac_idx[b]] for b in bi],
        "archaeon": [kept[arc_idx[a]] for a in ai],
        "rho_conditioned": rho[bi, ai],
        "rho_raw": rho_raw[bi, ai],
        "q": q[bi, ai],
        "patristic_distance": Dp[np.ix_(bac_idx, arc_idx)][bi, ai],
    }).sort_values("rho_conditioned", ascending=False).to_csv(f"{args.out}_edges.csv", index=False)

    Dpx = Dp[np.ix_(bac_idx, arc_idx)]
    diag = {
        "n_samples": int(len(df)), "n_features_kept": len(kept),
        "n_bacteria": int(bac_idx.size), "n_archaea": int(arc_idx.size),
        "conditioned_on": covs if conditioned else [],
        "cross_domain_pairs_tested": int(rho.size),
        "edges_raw": int(keep_raw.sum()),
        "edges_conditioned": int(keep.sum()),
        "edges_lost_to_conditioning": int((keep_raw & ~keep).sum()),
        "edges_gained_by_conditioning": int((keep & ~keep_raw).sum()),
        "mean_rho_kept": float(rho[keep].mean()) if keep.any() else None,
        "spearman_weight_vs_patristic": float(
            pd.Series(rho[keep]).corr(pd.Series(Dpx[keep]), method="spearman")) if keep.sum() > 2 else None,
        "archaea_with_no_partner": int((keep.sum(axis=0) == 0).sum()),
        "shortcut_scale": args.scale,
    }
    json.dump(diag, open(f"{args.out}_diagnostics.json", "w"), indent=2)
    print(json.dumps(diag, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

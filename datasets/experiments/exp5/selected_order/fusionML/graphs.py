"""Phylogenetic and abundance graph construction (design-spec items 4, 12).

Both graphs are built from a training partition only. Outcome data never
enter either graph. Graphs are frozen once built and reused as fixed data
for the coefficient-fitting optimization (item 12/13).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocessing import site_balanced_weights
from .io_tree import DomainTree


# ---------------------------------------------------------------------------
# Phylogenetic graph
# ---------------------------------------------------------------------------

EMPTY_EDGE_COLUMNS = ["feature_j", "feature_k", "distance", "phylo_weight",
                      "v_all", "v_joint", "v_star", "joint_samples", "joint_sites",
                      "support", "eligibility_frequency", "empirical_weight"]


@dataclasses.dataclass
class PhyloGraphResult:
    domain: str
    edges: pd.DataFrame  # feature_j, feature_k, distance, phylo_weight (normalized, sums to 1 unless empty)
    covered_features: List[str]
    uncovered_features: List[str]
    tau_p: Optional[float]


def _empty_edges(cols):
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def build_phylogenetic_graph(tree: Optional[DomainTree], domain: str,
                              retained_features: List[str], k: int) -> PhyloGraphResult:
    if tree is None:
        return PhyloGraphResult(
            domain=domain, edges=_empty_edges(["feature_j", "feature_k", "distance", "phylo_weight"]),
            covered_features=[], uncovered_features=list(retained_features), tau_p=None,
        )
    covered, uncovered = tree.coverage(retained_features)
    if len(covered) < 2:
        return PhyloGraphResult(
            domain=domain, edges=_empty_edges(["feature_j", "feature_k", "distance", "phylo_weight"]),
            covered_features=covered, uncovered_features=uncovered, tau_p=None,
        )
    covered_sorted = sorted(covered)  # canonical feature-ID order
    dist = tree.patristic_distance_matrix(covered_sorted)
    p = len(covered_sorted)

    neighbor_pairs = set()
    for i in range(p):
        order = sorted(range(p), key=lambda j: (dist[i, j], covered_sorted[j]))
        order = [j for j in order if j != i][:k]
        for j in order:
            a, b = (i, j) if covered_sorted[i] < covered_sorted[j] else (j, i)
            neighbor_pairs.add((a, b))

    if not neighbor_pairs:
        return PhyloGraphResult(
            domain=domain, edges=_empty_edges(["feature_j", "feature_k", "distance", "phylo_weight"]),
            covered_features=covered, uncovered_features=uncovered, tau_p=None,
        )

    dvals = np.array([dist[a, b] for a, b in neighbor_pairs])
    strictly_pos = dvals[dvals > 0]
    tau_p = float(np.median(strictly_pos)) if strictly_pos.size > 0 else 1.0

    raw_w = np.exp(-dvals / tau_p)
    total = raw_w.sum()
    norm_w = raw_w / total if total > 0 else raw_w

    rows = []
    for (a, b), d, w in zip(neighbor_pairs, dvals, norm_w):
        rows.append({
            "feature_j": covered_sorted[a], "feature_k": covered_sorted[b],
            "distance": float(d), "phylo_weight": float(w),
        })
    edges = pd.DataFrame(rows).sort_values(["feature_j", "feature_k"]).reset_index(drop=True)
    return PhyloGraphResult(domain=domain, edges=edges, covered_features=covered,
                             uncovered_features=uncovered, tau_p=tau_p)


# ---------------------------------------------------------------------------
# MetaDR-style traversal-order graph (Phylospec/multi_models/MetaDR's
# spatial-locality idea, reused as a fusion-graph edge source instead of a
# CNN -- see search.py's metadr_order_only family)
# ---------------------------------------------------------------------------

def _levelorder_leaves_iterative(root) -> list:
    """BFS leaf order, matching MetaDR-multilabel.py's
    `tree.traverse("levelorder")` filtered to `is_leaf()`."""
    leaves = []
    queue = [root]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if node.is_terminal():
            leaves.append(node)
        else:
            queue.extend(node.clades)
    return leaves


def _postorder_leaves_iterative(root) -> list:
    """Iterative post-order leaf traversal (children left-to-right, then
    parent), matching MetaDR-multilabel.py's `tree.traverse("postorder")`
    filtered to `is_leaf()`. Standard "modified-preorder-with-reversed-
    children, then reverse the whole sequence" trick, generalized to
    n-ary clades -- avoids Python recursion depth limits on real,
    potentially deep/unbalanced ASV trees (same concern as io_tree.py and
    interactions.py's iterative traversals)."""
    stack = [root]
    visited = []
    while stack:
        node = stack.pop()
        visited.append(node)
        stack.extend(node.clades)
    visited.reverse()
    return [n for n in visited if n.is_terminal()]


def build_traversal_order_graph(tree: Optional[DomainTree], domain: str, retained_features: List[str],
                                 order: str = "postorder", zigzag: bool = True) -> PhyloGraphResult:
    """MetaDR's pseudo-image locality idea (transform_image's level-order/
    post-order leaf traversal, reshaped row-major into an img_size x
    img_size grid with odd rows reversed when zigzag=True), reused here as
    a fusion-graph edge source: taxa that land in grid-adjacent cells (4-
    connectivity) become a fusion-graph edge with uniform weight, fed into
    the SAME mix_graphs/lambda_f convex penalty every other graph family
    uses. No CNN, no non-convexity, no learned pixel binning -- purely the
    traversal order's spatial-adjacency structure. Defaults to the
    post-order+zigzag branch to match this repo's other MetaDR-derived
    mechanism's precedent (CLAUDE.md's cross-domain-injection notes: "the
    level-order branch is left untouched as an internal control")."""
    empty_cols = ["feature_j", "feature_k", "distance", "phylo_weight"]
    if tree is None or tree.raw_tree is None:
        return PhyloGraphResult(domain=domain, edges=_empty_edges(empty_cols),
                                 covered_features=[], uncovered_features=list(retained_features), tau_p=None)
    covered, uncovered = tree.coverage(retained_features)
    if len(covered) < 2:
        return PhyloGraphResult(domain=domain, edges=_empty_edges(empty_cols),
                                 covered_features=covered, uncovered_features=uncovered, tau_p=None)

    raw_to_prefixed = {f.split("::", 1)[1]: f for f in covered}
    if order == "levelorder":
        leaves = _levelorder_leaves_iterative(tree.raw_tree.root)
    elif order == "postorder":
        leaves = _postorder_leaves_iterative(tree.raw_tree.root)
    else:
        raise ValueError(f"order must be 'levelorder' or 'postorder', got {order!r}")

    taxa_order = [raw_to_prefixed[leaf.name] for leaf in leaves if leaf.name in raw_to_prefixed]
    missing = sorted(set(covered) - set(taxa_order))  # not visited by this traversal (malformed tree edge case)
    taxa_order += missing

    n = len(taxa_order)
    img_size = math.ceil(math.sqrt(n))
    grid = np.full((img_size, img_size), -1, dtype=int)
    for i in range(n):
        r, c = divmod(i, img_size)
        if zigzag and r % 2 == 1:
            c = img_size - 1 - c
        grid[r, c] = i

    edges_set = set()
    for r in range(img_size):
        for c in range(img_size):
            if grid[r, c] < 0:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < img_size and cc < img_size and grid[rr, cc] >= 0:
                    a, b = int(grid[r, c]), int(grid[rr, cc])
                    i0, j0 = (a, b) if taxa_order[a] < taxa_order[b] else (b, a)
                    edges_set.add((i0, j0))

    if not edges_set:
        return PhyloGraphResult(domain=domain, edges=_empty_edges(empty_cols),
                                 covered_features=covered, uncovered_features=uncovered, tau_p=None)

    rows = [{"feature_j": taxa_order[i0], "feature_k": taxa_order[j0], "distance": 1.0, "phylo_weight": 1.0}
            for i0, j0 in edges_set]
    edges = pd.DataFrame(rows)
    edges["phylo_weight"] = edges["phylo_weight"] / edges["phylo_weight"].sum()
    edges = edges.sort_values(["feature_j", "feature_k"]).reset_index(drop=True)
    return PhyloGraphResult(domain=domain, edges=edges, covered_features=covered,
                             uncovered_features=uncovered, tau_p=None)


# ---------------------------------------------------------------------------
# Abundance (empirical) graph
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AbundanceGraphResult:
    domain: str
    edges: pd.DataFrame
    tau_e: Optional[float]
    n_candidate_full_training: int
    n_bootstrap: int
    approximate: bool


def _weighted_mean_cov(L: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = w @ L
    Xc = L - m
    C = (Xc * w[:, None]).T @ Xc
    return m, C


def _v_all_from_cov(C: np.ndarray) -> np.ndarray:
    d = np.diag(C)
    V = d[:, None] + d[None, :] - 2 * C
    np.fill_diagonal(V, 0.0)
    return np.maximum(V, 0.0)


def _joint_eligibility_and_vjoint(L: np.ndarray, D: np.ndarray,
                                   weight_site_labels: np.ndarray, eligibility_site_labels: np.ndarray,
                                   min_joint_samples: int, min_joint_sites: int):
    """Full (p,p) joint-detection eligibility + v_joint.

    weight_site_labels drives the site-balanced weighting inside v_joint
    (bootstrap copies count separately, per item 12's "each copy receives
    equal total weight"). eligibility_site_labels drives the distinct-site
    threshold (bootstrap copies collapse to their source site, per item 12's
    "count distinct source sites... so duplicate copies do not manufacture
    independent evidence").
    """
    n, p = D.shape
    Df = D.astype(float)
    joint_count = Df.T @ Df
    np.fill_diagonal(joint_count, 0)

    sites = pd.unique(eligibility_site_labels)
    site_joint = np.zeros((len(sites), p, p), dtype=bool)
    for i, s in enumerate(sites):
        rows = eligibility_site_labels == s
        Ds = Df[rows]
        site_joint[i] = (Ds.T @ Ds) > 0
    joint_site_count = site_joint.sum(axis=0)

    eligible = (joint_count >= min_joint_samples) & (joint_site_count >= min_joint_sites)
    np.fill_diagonal(eligible, False)

    v_joint = np.full((p, p), np.nan)
    idx_pairs = np.argwhere(np.triu(eligible, k=1))
    for j, k in idx_pairs:
        mask = D[:, j] & D[:, k]
        sub_w_labels = weight_site_labels[mask]
        uniq, counts = np.unique(sub_w_labels, return_counts=True)
        cnt_map = dict(zip(uniq.tolist(), counts.tolist()))
        w = np.array([1.0 / (len(uniq) * cnt_map[s]) for s in sub_w_labels])
        r = L[mask, j] - L[mask, k]
        mu = float(np.sum(w * r))
        v = float(np.sum(w * (r - mu) ** 2))
        v_joint[j, k] = v
        v_joint[k, j] = v
    return eligible, joint_count, joint_site_count, v_joint


def _mutual_topk(v_star: np.ndarray, eligible: np.ndarray, feature_ids: List[str], k: int) -> set:
    p = len(feature_ids)
    neighbors = [set() for _ in range(p)]
    for j in range(p):
        cand = [(v_star[j, kk], feature_ids[kk], kk) for kk in range(p) if eligible[j, kk]]
        cand.sort(key=lambda t: (t[0], t[1]))
        neighbors[j] = {kk for _, _, kk in cand[:k]}
    selected = set()
    for j in range(p):
        for kk in neighbors[j]:
            if j in neighbors[kk]:
                a, b = (j, kk) if j < kk else (kk, j)
                selected.add((a, b))
    return selected


def _pairwise_stats(L: np.ndarray, D: np.ndarray, feature_ids: List[str],
                     weight_site_labels: np.ndarray, eligibility_site_labels: np.ndarray,
                     min_joint_samples: int, min_joint_sites: int, k: int):
    w = site_balanced_weights(list(weight_site_labels))
    _, C = _weighted_mean_cov(L, w)
    v_all = _v_all_from_cov(C)
    eligible, joint_count, joint_site_count, v_joint = _joint_eligibility_and_vjoint(
        L, D, weight_site_labels, eligibility_site_labels, min_joint_samples, min_joint_sites,
    )
    v_star = np.where(eligible, np.maximum(v_all, v_joint), np.inf)
    np.fill_diagonal(v_star, np.inf)
    selected = _mutual_topk(v_star, eligible, feature_ids, k)
    return dict(v_all=v_all, eligible=eligible, joint_count=joint_count,
                joint_site_count=joint_site_count, v_joint=v_joint, v_star=v_star, selected=selected)


def build_abundance_graph(log_q: pd.DataFrame, detection_matrix: pd.DataFrame,
                           site_of_sample: Dict[str, str], cfg: dict, seed: int,
                           domain: str = "?") -> AbundanceGraphResult:
    feature_ids = list(log_q.columns)
    p = len(feature_ids)
    sample_ids = list(log_q.index)
    L = log_q.to_numpy()
    D = detection_matrix.reindex(columns=feature_ids).to_numpy()
    site_labels = np.array([site_of_sample[s] for s in sample_ids])

    approximate = bool(cfg.get("approximate_bootstrap", False))
    B = int(cfg["bootstrap_sites"])
    k = int(cfg["k_abundance"])
    min_js = int(cfg["min_joint_samples"])
    min_jt = int(cfg["min_joint_sites"])

    full_stats = _pairwise_stats(L, D, feature_ids, site_labels, site_labels, min_js, min_jt, k)
    selected_full = full_stats["selected"]

    if not selected_full:
        return AbundanceGraphResult(
            domain=domain, edges=_empty_edges(EMPTY_EDGE_COLUMNS[:-1] + ["empirical_weight"]),
            tau_e=None, n_candidate_full_training=0, n_bootstrap=B, approximate=approximate,
        )

    pool_idx = list(range(p))
    if approximate:
        candidate_k = int(cfg.get("approximate_candidate_k", 20))
        pool = set()
        for j, kk in selected_full:
            pool.add(j); pool.add(kk)
        for j in range(p):
            elig_partners = [(full_stats["v_star"][j, m], feature_ids[m], m)
                              for m in range(p) if full_stats["eligible"][j, m]]
            elig_partners.sort(key=lambda t: (t[0], t[1]))
            for _, _, m in elig_partners[:candidate_k]:
                pool.add(j); pool.add(m)
        pool_idx = sorted(pool)

    pool_feature_ids = [feature_ids[i] for i in pool_idx]
    L_pool = L[:, pool_idx]
    D_pool = D[:, pool_idx]
    pos_in_pool = {orig: i for i, orig in enumerate(pool_idx)}
    selected_full_pool = {(pos_in_pool[j], pos_in_pool[k]) for j, k in selected_full
                           if j in pos_in_pool and k in pos_in_pool}

    unique_sites = sorted(pd.unique(site_labels).tolist())
    site_to_rows = {s: np.where(site_labels == s)[0] for s in unique_sites}

    rng = np.random.default_rng(seed)
    support_count = {pair: 0 for pair in selected_full_pool}
    eligible_count = {pair: 0 for pair in selected_full_pool}

    for _ in range(B):
        draw = rng.choice(unique_sites, size=len(unique_sites), replace=True)
        row_idx, slot_labels, orig_labels = [], [], []
        copy_counter: Dict[str, int] = {}
        for s in draw:
            copy_counter[s] = copy_counter.get(s, 0) + 1
            rows = site_to_rows[s]
            row_idx.extend(rows.tolist())
            slot_labels.extend([f"{s}__copy{copy_counter[s]}"] * len(rows))
            orig_labels.extend([s] * len(rows))
        row_idx = np.array(row_idx)
        Lb = L_pool[row_idx]
        Db = D_pool[row_idx]
        slot_arr = np.array(slot_labels)
        orig_arr = np.array(orig_labels)
        rep_stats = _pairwise_stats(Lb, Db, pool_feature_ids, slot_arr, orig_arr, min_js, min_jt, k)
        elig_b = rep_stats["eligible"]
        selected_b = rep_stats["selected"]
        for pair in selected_full_pool:
            j, kk = pair
            if bool(elig_b[j, kk]):
                eligible_count[pair] += 1
            if pair in selected_b:
                support_count[pair] += 1

    support_threshold = float(cfg["support_threshold"])
    retained = []
    for pair in selected_full_pool:
        j, kk = pair
        support = support_count[pair] / B
        elig_freq = eligible_count[pair] / B
        if support >= support_threshold:
            gj, gk = pool_idx[j], pool_idx[kk]
            retained.append({
                "feature_j": feature_ids[gj], "feature_k": feature_ids[gk],
                "v_all": float(full_stats["v_all"][gj, gk]),
                "v_joint": float(full_stats["v_joint"][gj, gk]),
                "v_star": float(full_stats["v_star"][gj, gk]),
                "joint_samples": int(full_stats["joint_count"][gj, gk]),
                "joint_sites": int(full_stats["joint_site_count"][gj, gk]),
                "support": support, "eligibility_frequency": elig_freq,
            })

    if not retained:
        return AbundanceGraphResult(
            domain=domain, edges=_empty_edges(EMPTY_EDGE_COLUMNS[:-1] + ["empirical_weight"]),
            tau_e=None, n_candidate_full_training=len(selected_full), n_bootstrap=B, approximate=approximate,
        )

    df = pd.DataFrame(retained)
    strictly_pos = df.loc[df["v_star"] > 0, "v_star"]
    tau_e = float(strictly_pos.median()) if len(strictly_pos) > 0 else 1.0
    raw_w = df["support"] * np.exp(-df["v_star"] / tau_e)
    total = raw_w.sum()
    df["empirical_weight"] = raw_w / total if total > 0 else raw_w
    df = df.sort_values(["feature_j", "feature_k"]).reset_index(drop=True)

    return AbundanceGraphResult(
        domain=domain, edges=df, tau_e=tau_e,
        n_candidate_full_training=len(selected_full), n_bootstrap=B, approximate=approximate,
    )


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------

def mix_graphs(domain: str, phylo: PhyloGraphResult, emp: AbundanceGraphResult, alpha: float) -> pd.DataFrame:
    """W_d = alpha*W_P,d + (1-alpha)*W_E,d, edge-union, no renormalization.

    Both component weight columns are preserved (NaN where that component
    has no edge, distinct from an edge existing with weight 0)."""
    pe = phylo.edges[["feature_j", "feature_k", "distance", "phylo_weight"]].copy() if len(phylo.edges) else \
        pd.DataFrame(columns=["feature_j", "feature_k", "distance", "phylo_weight"])
    ee = emp.edges[["feature_j", "feature_k", "v_all", "v_joint", "v_star", "support",
                     "eligibility_frequency", "empirical_weight"]].copy() if len(emp.edges) else \
        pd.DataFrame(columns=["feature_j", "feature_k", "v_all", "v_joint", "v_star", "support",
                               "eligibility_frequency", "empirical_weight"])
    merged = pd.merge(pe, ee, on=["feature_j", "feature_k"], how="outer")
    if merged.empty:
        merged["mixed_weight"] = pd.Series(dtype=float)
        return merged
    p_w = merged["phylo_weight"].astype(float).fillna(0.0)
    e_w = merged["empirical_weight"].astype(float).fillna(0.0)
    merged["mixed_weight"] = alpha * p_w + (1 - alpha) * e_w
    merged["domain"] = domain
    return merged

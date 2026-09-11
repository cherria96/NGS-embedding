"""
cross_domain_edges.py -- shared loader/validator for the bacterium<->archaeon
syntrophy edge set, used identically by both injection channels:
  - Phylo-Spec: appends one conv_order merge per edge (data_processing.py's
    inject_cross_domain_edges, task doc Sec A.2).
  - MetaDR: places taxa on a 2-D grid so each edge's endpoints land in one
    conv footprint (build_cross_domain_prior_merged.py, task doc Sec B.2).

CLAUDE_CODE_TASK_cross_domain_injection.md Sec 4.
"""
import pandas as pd


def load_edges(path, features, mode="real"):
    """Read an edge CSV, keep only edges whose BOTH endpoints are in
    `features`, return a list of (bacterium, archaeon, rho) sorted by
    descending |rho|, plus the count of edges dropped for a missing endpoint.

    Reads `rho_conditioned` if present (the task doc's own edge-CSV schema),
    else falls back to `rho` (the column name this repo's actual edge
    producer, build_cross_domain_prior_merged.py, currently writes).

    Does not raise on a dropped edge itself -- a caller restricting to a
    fold's or arm's own feature subset is expected to lose some edges. It
    only reports the count; a caller that requires zero drops must check the
    returned count itself, per task doc Sec 7 ("do not silently drop").
    """
    df = pd.read_csv(path)
    weight_col = "rho_conditioned" if "rho_conditioned" in df.columns else "rho"
    feature_set = set(features)
    edges, dropped = [], 0
    for bac, arc, rho in zip(df["bacterium"].astype(str), df["archaeon"].astype(str), df[weight_col]):
        if bac in feature_set and arc in feature_set:
            edges.append((bac, arc, float(rho)))
        else:
            dropped += 1
    if dropped:
        print(f"load_edges({mode}): dropped {dropped}/{len(df)} edge(s) with an endpoint "
              f"missing from this arm's feature set")
    edges.sort(key=lambda e: -abs(e[2]))
    return edges, dropped


def check_edges(edges, features, topk):
    """Assert: no self-pairs; no duplicate unordered pairs; every archaeon has
    at most `topk` partners; every endpoint appears in `features`.

    Raises on any violation, never warns -- a silently dropped or duplicated
    edge changes the with-arm's parameter count (one conv_order merge per
    edge, for Phylo-Spec) and breaks the parameter match against the null
    arm (task doc Sec A.3)."""
    feature_set = set(features)
    seen_pairs = set()
    per_archaeon = {}
    for bac, arc, _rho in edges:
        if bac == arc:
            raise ValueError(f"cross_domain_edges.check_edges: self-pair edge {bac!r}")
        if bac not in feature_set or arc not in feature_set:
            raise ValueError(f"cross_domain_edges.check_edges: edge endpoint not in features: "
                             f"{bac!r}, {arc!r}")
        pair = frozenset((bac, arc))
        if pair in seen_pairs:
            raise ValueError(f"cross_domain_edges.check_edges: duplicate unordered pair: "
                             f"{bac!r}, {arc!r}")
        seen_pairs.add(pair)
        per_archaeon[arc] = per_archaeon.get(arc, 0) + 1
    over = {a: n for a, n in per_archaeon.items() if n > topk}
    if over:
        raise ValueError(f"cross_domain_edges.check_edges: archaeon/a exceeding topk={topk} "
                         f"partner(s): {over}")

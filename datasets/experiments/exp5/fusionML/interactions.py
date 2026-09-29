"""ARC-BAC cross-domain predictive interaction
(ad-microbial-model-design-arc-bac-interactions.md items 8, 10, 12-15).

Within each domain, candidate "balances" are zero-sum, unit-scale log-ratio
contrasts built from every tree bipartition (not just the exploratory
13/8 block). A small number of the most stable, mutually-uncorrelated
balances per domain are selected on TRAINING DATA ONLY, standardized, and
crossed pairwise (ARC balance x BAC balance) into standardized product
features. These products enter the model as additional linear predictors
with their own coefficient vector gamma -- a genuinely new predictor set,
not an edge in either domain's fusion graph (item 12: "Store cross-domain
balance-product pairs in a separate interaction table; these are predictor
definitions, not extra fusion edges").
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .io_tree import DomainTree


class InteractionUnavailableError(ValueError):
    """Raised when an interaction-on candidate has no surviving product in
    this partition. The interaction-off candidate remains fittable."""
    pass


# ---------------------------------------------------------------------------
# Item 10.1: tree-split candidate balances
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BalanceCandidate:
    domain: str
    positive_members: Tuple[str, ...]  # sorted, domain-prefixed feature IDs
    negative_members: Tuple[str, ...]

    @property
    def r(self) -> int:
        return len(self.positive_members)

    @property
    def s(self) -> int:
        return len(self.negative_members)

    @property
    def canonical_id(self) -> str:
        return f"{self.domain}:pos=[{','.join(self.positive_members)}]|neg=[{','.join(self.negative_members)}]"


def _tip_sets_bottom_up(root) -> Dict[int, frozenset]:
    """Iterative (non-recursive) post-order traversal computing each
    clade's terminal tip-name set exactly once -- real ASV trees can be
    deep, unbalanced ladders (same concern as io_tree.py's patristic-
    distance computation)."""
    tip_sets: Dict[int, frozenset] = {}
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if node.is_terminal():
            tip_sets[id(node)] = frozenset([node.name])
            continue
        if processed:
            s = frozenset()
            for child in node.clades:
                s = s | tip_sets[id(child)]
            tip_sets[id(node)] = s
        else:
            stack.append((node, True))
            for child in node.clades:
                stack.append((child, False))
    return tip_sets


def enumerate_tree_splits(tree: Optional[DomainTree], domain: str,
                           retained_features: List[str]) -> Tuple[List[BalanceCandidate], dict]:
    """Every tree edge's descendant-tip bipartition, restricted to
    retained+mapped taxa, deduplicated (the lexicographically smaller
    sorted side is "positive"), omitting empty sides and the trivial root
    split. Polytomies contribute exactly one split each (their full
    descendant set vs. the complement) -- no invented binary resolution.
    A wholly missing tree makes this representation unavailable."""
    report = {"n_retained": len(retained_features), "n_mapped": 0, "n_unmapped": len(retained_features),
              "unmapped_features": list(retained_features), "n_candidate_splits": 0}
    if tree is None:
        return [], report

    covered, uncovered = tree.coverage(retained_features)
    universe = set(covered)
    report["n_mapped"] = len(covered)
    report["n_unmapped"] = len(uncovered)
    report["unmapped_features"] = uncovered
    if len(universe) < 2 or tree.raw_tree is None:
        return [], report

    raw_to_prefixed = {f.split("::", 1)[1]: f for f in covered}
    tip_sets = _tip_sets_bottom_up(tree.raw_tree.root)

    seen_sides = set()
    candidates: List[BalanceCandidate] = []
    for clade in tree.raw_tree.get_nonterminals():
        raw_tips = tip_sets.get(id(clade))
        if raw_tips is None:
            continue
        side_a = frozenset(raw_to_prefixed[n] for n in raw_tips if n in raw_to_prefixed)
        if not side_a or len(side_a) == len(universe):
            continue
        side_b = frozenset(universe - side_a)
        if sorted(side_a) < sorted(side_b):
            positive_set, negative_set = side_a, side_b
        else:
            positive_set, negative_set = side_b, side_a
        if positive_set in seen_sides:
            continue
        seen_sides.add(positive_set)
        candidates.append(BalanceCandidate(
            domain=domain,
            positive_members=tuple(sorted(positive_set)),
            negative_members=tuple(sorted(negative_set)),
        ))
    report["n_candidate_splits"] = len(candidates)
    return candidates, report


def compute_loading(candidate: BalanceCandidate, feature_order: List[str]) -> np.ndarray:
    """Item 8's balance loading: a_j = k/r on the positive side, -k/s on
    the negative side, k = sqrt(r*s/(r+s)), 0 elsewhere. Zero-sum,
    unit-scale by construction."""
    r, s = candidate.r, candidate.s
    k = math.sqrt(r * s / (r + s))
    idx = {f: i for i, f in enumerate(feature_order)}
    a = np.zeros(len(feature_order))
    for f in candidate.positive_members:
        a[idx[f]] = k / r
    for f in candidate.negative_members:
        a[idx[f]] = -k / s
    return a


def group_detection_eligible(candidate: BalanceCandidate, detection_df: pd.DataFrame,
                              site_of_sample: Dict[str, str], min_samples: int, min_sites: int) -> bool:
    """Item 10.2's group-detection gate (distinct from item 11's pairwise
    taxon-graph eligibility): at least one taxon on EACH side detected
    simultaneously in >=min_samples training samples spanning >=min_sites
    training sites."""
    pos_cols = [f for f in candidate.positive_members if f in detection_df.columns]
    neg_cols = [f for f in candidate.negative_members if f in detection_df.columns]
    if not pos_cols or not neg_cols:
        return False
    pos_detected = detection_df[pos_cols].any(axis=1)
    neg_detected = detection_df[neg_cols].any(axis=1)
    joint = pos_detected & neg_detected
    if int(joint.sum()) < min_samples:
        return False
    joint_sites = {site_of_sample[sid] for sid in detection_df.index[joint]}
    return len(joint_sites) >= min_sites


def weighted_mean_var(z: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    mu = float(np.sum(weights * z))
    var = float(np.sum(weights * (z - mu) ** 2))
    return mu, var


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mx, vx = weighted_mean_var(x, weights)
    my, vy = weighted_mean_var(y, weights)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = float(np.sum(weights * (x - mx) * (y - my)))
    return cov / math.sqrt(vx * vy)


# ---------------------------------------------------------------------------
# Item 10.3-10.4: selection, standardization, products
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SelectedBalance:
    candidate: BalanceCandidate
    loading: np.ndarray  # ordered per this domain's retained-feature order
    mean: float
    sd: float
    variance: float

    @property
    def canonical_id(self) -> str:
        return self.candidate.canonical_id

    @property
    def members(self) -> List[str]:
        return list(self.candidate.positive_members) + list(self.candidate.negative_members)


def select_balances(candidates: List[BalanceCandidate], log_q: pd.DataFrame, feature_order: List[str],
                     weights: np.ndarray, detection_df: pd.DataFrame, site_of_sample: Dict[str, str],
                     min_group_samples: int, min_group_sites: int, max_balances: int,
                     max_abs_correlation: float) -> Tuple[List[SelectedBalance], dict]:
    eligible = []
    for c in candidates:
        if not group_detection_eligible(c, detection_df, site_of_sample, min_group_samples, min_group_sites):
            continue
        loading = compute_loading(c, feature_order)
        z = log_q[feature_order].to_numpy() @ loading
        mu, var = weighted_mean_var(z, weights)
        if var <= 1e-12:
            continue
        eligible.append((c, loading, z, mu, var))
    eligible.sort(key=lambda t: (-t[4], t[0].canonical_id))

    selected: List[SelectedBalance] = []
    selected_loadings: List[np.ndarray] = []
    selected_scores: List[np.ndarray] = []
    for c, loading, z, mu, var in eligible:
        if len(selected) >= max_balances:
            break
        candidate_matrix = np.column_stack(selected_loadings + [loading])
        rank_before = np.linalg.matrix_rank(np.column_stack(selected_loadings), tol=1e-10) if selected_loadings else 0
        rank_after = np.linalg.matrix_rank(candidate_matrix, tol=1e-10)
        if rank_after <= rank_before:
            continue
        if any(abs(weighted_correlation(z, prev, weights)) > max_abs_correlation for prev in selected_scores):
            continue
        selected.append(SelectedBalance(candidate=c, loading=loading, mean=mu, sd=math.sqrt(var), variance=var))
        selected_loadings.append(loading)
        selected_scores.append(z)

    report = {"n_candidates": len(candidates), "n_eligible": len(eligible), "n_selected": len(selected),
              "selected_ids": [b.canonical_id for b in selected]}
    return selected, report


@dataclasses.dataclass
class ProductFeature:
    balance_a_id: str
    balance_b_id: str
    mean: float
    sd: float


@dataclasses.dataclass
class InteractionTransform:
    balances_a: List[SelectedBalance]
    balances_b: List[SelectedBalance]
    feature_order_a: List[str]
    feature_order_b: List[str]
    products: List[ProductFeature]

    @property
    def n_products(self) -> int:
        return len(self.products)

    def _standardized_balance_scores(self, log_q: pd.DataFrame, balances: List[SelectedBalance],
                                      feature_order: List[str]) -> np.ndarray:
        n = log_q.shape[0]
        if not balances:
            return np.zeros((n, 0))
        L = log_q[feature_order].to_numpy()
        out = np.zeros((n, len(balances)))
        for i, b in enumerate(balances):
            z = L @ b.loading
            out[:, i] = (z - b.mean) / b.sd
        return out

    def transform(self, log_q_a: pd.DataFrame, log_q_b: pd.DataFrame) -> np.ndarray:
        n = log_q_a.shape[0]
        if not self.products:
            return np.zeros((n, 0))
        ta = self._standardized_balance_scores(log_q_a, self.balances_a, self.feature_order_a)
        tb = self._standardized_balance_scores(log_q_b, self.balances_b, self.feature_order_b)
        ta_map = {b.canonical_id: ta[:, i] for i, b in enumerate(self.balances_a)}
        tb_map = {b.canonical_id: tb[:, i] for i, b in enumerate(self.balances_b)}
        cols = []
        for p in self.products:
            h = ta_map[p.balance_a_id] * tb_map[p.balance_b_id]
            cols.append((h - p.mean) / p.sd)
        return np.column_stack(cols)

    def required_parent_taxa(self) -> List[str]:
        """All raw taxa any active balance parent depends on (main-effect
        beta may be zero for some of these -- item 15's compression
        accounting requires disclosing them anyway)."""
        used_a = {b.canonical_id for p in self.products for b in self.balances_a if b.canonical_id == p.balance_a_id}
        used_b = {b.canonical_id for p in self.products for b in self.balances_b if b.canonical_id == p.balance_b_id}
        members = set()
        for b in self.balances_a:
            if b.canonical_id in used_a:
                members.update(b.members)
        for b in self.balances_b:
            if b.canonical_id in used_b:
                members.update(b.members)
        return sorted(members)


def fit_interaction_transform(
    log_q_a: pd.DataFrame, log_q_b: pd.DataFrame, det_a: pd.DataFrame, det_b: pd.DataFrame,
    site_of_sample: Dict[str, str], weights: np.ndarray,
    tree_a: Optional[DomainTree], tree_b: Optional[DomainTree],
    retained_a: List[str], retained_b: List[str], cfg: dict,
) -> Tuple[InteractionTransform, dict]:
    """Fits the full training-only balance+product pipeline (item 10.1-10.4)
    on one training partition. Must be called fresh inside every inner/
    outer/final training partition -- never reused across partitions."""
    cand_a, rep_split_a = enumerate_tree_splits(tree_a, "ARC", retained_a)
    cand_b, rep_split_b = enumerate_tree_splits(tree_b, "BAC", retained_b)

    sel_a, rep_sel_a = select_balances(
        cand_a, log_q_a, retained_a, weights, det_a, site_of_sample,
        cfg["min_joint_group_samples"], cfg["min_joint_group_sites"],
        cfg["max_balances_arc"], cfg["max_abs_balance_correlation"],
    )
    sel_b, rep_sel_b = select_balances(
        cand_b, log_q_b, retained_b, weights, det_b, site_of_sample,
        cfg["min_joint_group_samples"], cfg["min_joint_group_sites"],
        cfg["max_balances_bac"], cfg["max_abs_balance_correlation"],
    )

    raw_pairs = sorted(
        ((ba, bb) for ba in sel_a for bb in sel_b),
        key=lambda pair: (pair[0].canonical_id, pair[1].canonical_id),
    )

    n = log_q_a.shape[0]
    ref_cols = [np.ones(n)]
    products: List[ProductFeature] = []
    for ba, bb in raw_pairs[: cfg["max_products"]]:
        za = (log_q_a[retained_a].to_numpy() @ ba.loading - ba.mean) / ba.sd
        zb = (log_q_b[retained_b].to_numpy() @ bb.loading - bb.mean) / bb.sd
        h = za * zb
        mu, var = weighted_mean_var(h, weights)
        if var <= 1e-12:
            continue
        sd = math.sqrt(var)
        x = (h - mu) / sd
        rank_before = np.linalg.matrix_rank(np.column_stack(ref_cols), tol=1e-10)
        rank_after = np.linalg.matrix_rank(np.column_stack(ref_cols + [x]), tol=1e-10)
        if rank_after <= rank_before:
            continue
        products.append(ProductFeature(balance_a_id=ba.canonical_id, balance_b_id=bb.canonical_id, mean=mu, sd=sd))
        ref_cols.append(x)

    transform = InteractionTransform(balances_a=sel_a, balances_b=sel_b,
                                      feature_order_a=retained_a, feature_order_b=retained_b, products=products)
    report = {
        "tree_split_arc": rep_split_a, "tree_split_bac": rep_split_b,
        "balance_selection_arc": rep_sel_a, "balance_selection_bac": rep_sel_b,
        "n_candidate_products": len(raw_pairs), "n_products": len(products),
    }
    return transform, report

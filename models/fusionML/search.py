"""Model families (design-spec item 14 table) and deterministic candidate
hyperparameter generation."""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.modules.setdefault("fusionML.search", _repository_sys.modules[__name__])
_repository_sys.modules.setdefault("models.fusionML.search", _repository_sys.modules[__name__])


import dataclasses
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class Candidate:
    lambda_s: float
    lambda_r: float
    lambda_f: float
    alpha_arc: Optional[float]
    alpha_bac: Optional[float]

    def key(self):
        aa = None if self.alpha_arc is None else round(self.alpha_arc, 6)
        ab = None if self.alpha_bac is None else round(self.alpha_bac, 6)
        return (round(self.lambda_s, 8), round(self.lambda_r, 8), round(self.lambda_f, 8), aa, ab)


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    name: str
    uses_graph: bool
    fix_lambda_s_zero: bool  # ridge families
    alpha_mode: str  # "none" | "fixed" | "shared" | "independent"
    fixed_alpha: Optional[Tuple[float, float]] = None
    resolution: str = "any"  # "asv" | "genus" | "any"
    uses_interaction: bool = False  # ARC-BAC balance-product extension
    graph_source: str = "phylo_abundance"  # "phylo_abundance" | "metadr_order"
    # Which domains receive a fusion graph at all. lambda_F is a single
    # global scalar over both domains' edge sets, so "restrict fusion to one
    # domain" is expressed by giving the other domain an EMPTY edge set --
    # its coefficients are then fit completely unfused.
    fusion_domains: str = "both"  # "both" | "arc_only" | "bac_only"


PRIMARY_FAMILY = "domain_adaptive_hybrid"  # required additive reference
INTERACTION_FAMILY = "domain_adaptive_interaction"  # proposed primary extension
# The interaction family's base beta/alpha grid must be the identical
# deterministic set the additive reference uses (same seed derivation),
# so both are evaluated on the same candidate base points.
INTERACTION_BASE_FAMILY_ALIAS = {INTERACTION_FAMILY: PRIMARY_FAMILY}

FAMILY_SPECS: Dict[str, FamilySpec] = {
    "intercept_only": FamilySpec("intercept_only", uses_graph=False, fix_lambda_s_zero=True, alpha_mode="none"),
    "genus_ridge": FamilySpec("genus_ridge", uses_graph=False, fix_lambda_s_zero=True, alpha_mode="none", resolution="genus"),
    "genus_elastic_net": FamilySpec("genus_elastic_net", uses_graph=False, fix_lambda_s_zero=False, alpha_mode="none", resolution="genus"),
    "asv_ridge": FamilySpec("asv_ridge", uses_graph=False, fix_lambda_s_zero=True, alpha_mode="none", resolution="asv"),
    "asv_elastic_net": FamilySpec("asv_elastic_net", uses_graph=False, fix_lambda_s_zero=False, alpha_mode="none", resolution="asv"),
    "phylogeny_only": FamilySpec("phylogeny_only", uses_graph=True, fix_lambda_s_zero=False, alpha_mode="fixed", fixed_alpha=(1.0, 1.0)),
    "abundance_only": FamilySpec("abundance_only", uses_graph=True, fix_lambda_s_zero=False, alpha_mode="fixed", fixed_alpha=(0.0, 0.0)),
    # MetaDR's spatial-locality idea (Phylospec/multi_models/MetaDR/MetaDR-multilabel.py's
    # post-order+zigzag pseudo-image grid), reused as a fusion-graph edge source
    # feeding the SAME convex log-contrast objective -- no CNN, per user request
    # to try "(b)": keep fusionML's objective/loss, drop MetaDR's non-convex model.
    # fixed_alpha is unused (graph_source bypasses the phylo/abundance alpha mix
    # entirely) but alpha_mode="fixed" is required so generate_candidates still
    # emits lambda_f>0 candidates (alpha_mode="none" would suppress them).
    "metadr_order_only": FamilySpec("metadr_order_only", uses_graph=True, fix_lambda_s_zero=False,
                                     alpha_mode="fixed", fixed_alpha=(1.0, 1.0), graph_source="metadr_order"),
    # Domain-restricted fusion. Motivation: data/Data_investigation_result.md's
    # Mantel test found significant phylogenetic signal in ARC at every rank
    # (r=0.28-0.40, p=0.001) but NONE in BAC (r~0.002-0.014, all p>0.6) -- yet
    # BAC supplies ~90% of genus-level features, so a fusion penalty applied to
    # both domains spends most of its effect enforcing similarity along a graph
    # that carries no information for BAC. These two families isolate that:
    # fuse only the domain where the prior is defensible, and the converse as a
    # control. alpha_mode="shared" here only avoids wasting candidate budget on
    # the unfused domain's alpha, which has no effect (empty edge set).
    "fusion_arc_only": FamilySpec("fusion_arc_only", uses_graph=True, fix_lambda_s_zero=False,
                                   alpha_mode="shared", fusion_domains="arc_only"),
    "fusion_bac_only": FamilySpec("fusion_bac_only", uses_graph=True, fix_lambda_s_zero=False,
                                   alpha_mode="shared", fusion_domains="bac_only"),
    "shared_hybrid": FamilySpec("shared_hybrid", uses_graph=True, fix_lambda_s_zero=False, alpha_mode="shared"),
    "domain_adaptive_hybrid": FamilySpec("domain_adaptive_hybrid", uses_graph=True, fix_lambda_s_zero=False, alpha_mode="independent"),
    "domain_adaptive_interaction": FamilySpec("domain_adaptive_interaction", uses_graph=True, fix_lambda_s_zero=False,
                                               alpha_mode="independent", uses_interaction=True),
}


def allowed_alpha_combos(spec: FamilySpec, alpha_arc_grid: List[float], alpha_bac_grid: List[float]) -> List[Tuple[float, float]]:
    if spec.alpha_mode == "none":
        return []
    if spec.alpha_mode == "fixed":
        return [spec.fixed_alpha]
    if spec.alpha_mode == "shared":
        common = [a for a in alpha_arc_grid if a in alpha_bac_grid]
        return [(a, a) for a in (common or alpha_arc_grid)]
    if spec.alpha_mode == "independent":
        return list(itertools.product(alpha_arc_grid, alpha_bac_grid))
    raise ValueError(spec.alpha_mode)


def generate_candidates(spec: FamilySpec, search_cfg: dict, seed: int) -> List[Candidate]:
    lambda_s_grid = [0.0] if spec.fix_lambda_s_zero else list(search_cfg["lambda_s"])
    lambda_r_grid = list(search_cfg["lambda_r"])
    lambda_f_grid = list(search_cfg["lambda_f"]) if spec.uses_graph else [0.0]
    budget = int(search_cfg["candidate_budget_per_family"])
    exhaustive = bool(search_cfg.get("exhaustive", False))

    all_combos: Dict[tuple, Candidate] = {}
    if not spec.uses_graph:
        for ls in lambda_s_grid:
            for lr in lambda_r_grid:
                c = Candidate(ls, lr, 0.0, None, None)
                all_combos[c.key()] = c
    else:
        alphas = allowed_alpha_combos(spec, list(search_cfg["alpha_arc"]), list(search_cfg["alpha_bac"]))
        for ls in lambda_s_grid:
            for lr in lambda_r_grid:
                for lf in lambda_f_grid:
                    if lf == 0.0:
                        c = Candidate(ls, lr, 0.0, None, None)
                        all_combos[c.key()] = c
                    else:
                        for aa, ab in alphas:
                            c = Candidate(ls, lr, lf, aa, ab)
                            all_combos[c.key()] = c

    combos = list(all_combos.values())
    if exhaustive or len(combos) <= budget:
        return sorted(combos, key=lambda c: c.key())

    # Required coverage set: one no-fusion candidate, plus one candidate per
    # allowed alpha corner at the smallest positive lambda_f (item 14:
    # "Include no-fusion candidates, cover every allowed alpha combination").
    default_ls = lambda_s_grid[0]
    default_lr = lambda_r_grid[len(lambda_r_grid) // 2]
    required: Dict[tuple, Candidate] = {}
    no_fusion = Candidate(default_ls, default_lr, 0.0, None, None)
    required[no_fusion.key()] = no_fusion
    if spec.uses_graph:
        pos_lf = [lf for lf in lambda_f_grid if lf > 0]
        smallest_lf = min(pos_lf) if pos_lf else None
        if smallest_lf is not None:
            for aa, ab in allowed_alpha_combos(spec, list(search_cfg["alpha_arc"]), list(search_cfg["alpha_bac"])):
                c = Candidate(default_ls, default_lr, smallest_lf, aa, ab)
                required[c.key()] = c

    remaining_pool = [c for c in combos if c.key() not in required]
    remaining_budget = max(0, budget - len(required))
    rng = np.random.default_rng(seed)
    if remaining_budget > 0 and remaining_pool:
        idx = rng.choice(len(remaining_pool), size=min(remaining_budget, len(remaining_pool)), replace=False)
        sampled = [remaining_pool[i] for i in idx]
    else:
        sampled = []
    final = list(required.values()) + sampled
    return sorted(final, key=lambda c: c.key())[:budget] if len(final) > budget else sorted(final, key=lambda c: c.key())


@dataclasses.dataclass(frozen=True)
class InteractionCandidate:
    """Wraps a base beta/alpha Candidate with the interaction-specific
    settings (item 14's interaction comparison). An interaction-OFF
    candidate has gamma fixed to zero -- it is fit via the plain additive
    estimator, reusing that exact formulation, not merely "lambda_i1=0"
    (which alone would still fit a free, near-zero gamma)."""
    base: Candidate
    interaction_on: bool
    lambda_i1: float = 0.0
    lambda_i2: float = 0.0

    def key(self):
        return self.base.key() + (self.interaction_on, round(self.lambda_i1, 8), round(self.lambda_i2, 8))


def generate_interaction_candidates(base_candidates: List[Candidate], interaction_cfg: dict) -> List[InteractionCandidate]:
    """Item 14's interaction comparison: for each base beta/alpha
    candidate, one interaction-off variant plus every (lambda_i1,
    lambda_i2) on-setting. Smoke configs shrink interaction_cfg's grids
    (e.g. to one or two settings) rather than changing this function."""
    l1_grid = list(interaction_cfg.get("lambda_i_l1", [0.0]))
    l2_grid = list(interaction_cfg.get("lambda_i_l2", [0.01]))
    out: List[InteractionCandidate] = []
    for base in base_candidates:
        out.append(InteractionCandidate(base=base, interaction_on=False))
        if interaction_cfg.get("include_on_candidates", True):
            for l1 in l1_grid:
                for l2 in l2_grid:
                    if l2 <= 0:
                        continue  # item 13: lambda_i2 strictly positive for interaction-on
                    out.append(InteractionCandidate(base=base, interaction_on=True, lambda_i1=l1, lambda_i2=l2))
    return out


def one_standard_error_select(candidate_scores: List[Tuple[object, np.ndarray, tuple]]) -> Tuple[object, dict]:
    """candidate_scores: list of (candidate, per_inner_site_score_array,
    complexity_tuple). complexity_tuple is compared lexicographically
    (smaller preferred) as the tie-break among candidates within one
    inner-CV standard error of the lowest mean site-balanced score --
    item 14's default is (n_active_groups, n_active_taxa) for additive
    families, and its interaction-family override is (n_active_products,
    n_required_raw_taxa, n_active_groups). Candidates are Candidate or
    InteractionCandidate objects (either works: both expose .key()).
    Returns (selected, info); info also reports the exact minimum-error
    candidate.
    """
    means = [(c, float(np.mean(sm)), sm, cx) for c, sm, cx in candidate_scores]
    best_mean = min(m for _, m, _, _ in means)
    best_entry = min(means, key=lambda t: t[1])
    n_sites = len(best_entry[2])
    se_best = float(np.std(best_entry[2], ddof=0) / np.sqrt(n_sites)) if n_sites > 0 else 0.0
    threshold = best_mean + se_best

    permitted = [(c, m, cx) for c, m, sm, cx in means if m <= threshold]
    permitted.sort(key=lambda t: (t[2], t[1], t[0].key()))
    selected = permitted[0][0]

    min_error_candidate = min(candidate_scores, key=lambda t: (float(np.mean(t[1])), t[0].key()))[0]
    return selected, {
        "best_mean_site_score": best_mean,
        "se_best": se_best,
        "threshold": threshold,
        "n_permitted": len(permitted),
        "minimum_error_candidate": min_error_candidate,
    }

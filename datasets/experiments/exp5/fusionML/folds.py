"""Site-grouped outer/inner CV fold construction (design-spec item 14).

Sites are assigned to folds deterministically from a saved seed, balancing
sample counts across folds via greedy bin-balancing -- never inspecting
target values. All observations of a site stay together.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

import numpy as np


class TooFewSitesError(ValueError):
    pass


def _greedy_balance_assign(site_counts: Dict[str, int], k: int, seed: int) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    sites = list(site_counts.keys())
    perm = rng.permutation(len(sites))
    order = [sites[i] for i in perm]
    order.sort(key=lambda s: site_counts[s], reverse=True)  # largest-first greedy bin packing

    fold_totals = [0] * k
    assignment: Dict[str, int] = {}
    for s in order:
        fold = int(np.argmin(fold_totals))
        assignment[s] = fold
        fold_totals[fold] += site_counts[s]
    return assignment


@dataclasses.dataclass
class FoldAssignment:
    k: int
    assignment: Dict[str, int]  # site_id -> fold index
    seed: int

    def sites_in_fold(self, fold: int) -> List[str]:
        return [s for s, f in self.assignment.items() if f == fold]

    def train_sites_for_outer_fold(self, fold: int) -> List[str]:
        return [s for s, f in self.assignment.items() if f != fold]


def make_site_grouped_folds(site_of_sample: Dict[str, str], k: int, seed: int) -> FoldAssignment:
    sites = sorted(set(site_of_sample.values()))
    if len(sites) < k:
        raise TooFewSitesError(
            f"only {len(sites)} distinct sites available but {k} folds requested; "
            "reduce cv.outer_folds/cv.inner_folds explicitly rather than silently "
            "switching to row-wise CV (design-spec item 14)."
        )
    site_counts: Dict[str, int] = {}
    for sample_id, site in site_of_sample.items():
        site_counts[site] = site_counts.get(site, 0) + 1
    assignment = _greedy_balance_assign(site_counts, k, seed)
    return FoldAssignment(k=k, assignment=assignment, seed=seed)


def sample_ids_for_sites(site_of_sample: Dict[str, str], sites: List[str]) -> List[str]:
    site_set = set(sites)
    return [sid for sid, s in site_of_sample.items() if s in site_set]

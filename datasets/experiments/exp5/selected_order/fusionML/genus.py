"""Genus-resolution aggregation: sum raw ASV counts by (lineage, genus)
before zero handling (design-spec item 9). Uses the lineage-aware grouping
key from io_taxonomy so unclassified ASVs land in explicit, lineage-scoped
groups rather than one artificial bucket.

Genus tips are NOT fabricated on the tree here: this module only builds the
abundance table, never a genus-level Newick. Phylogeny-enabled families are
marked unavailable at genus resolution (see pipeline.py) rather than
assigning a genus an arbitrary representative ASV tip (item 9 forbids this
without "checking the construction", which would require the full
monophyly-aware genus-tree pipeline this repo already has elsewhere for a
different purpose -- out of scope here).
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple

import pandas as pd


@dataclasses.dataclass
class GenusAggregationReport:
    domain: str
    n_asv_features: int
    n_genus_groups: int
    n_features_missing_taxonomy: int
    features_missing_taxonomy: List[str]
    mass_fraction_missing_taxonomy: float


def aggregate_to_genus(counts: pd.DataFrame, taxonomy: pd.DataFrame, domain: str) -> Tuple[pd.DataFrame, pd.DataFrame, GenusAggregationReport]:
    mapping = taxonomy["genus_group"]
    present = [c for c in counts.columns if c in mapping.index]
    missing = [c for c in counts.columns if c not in mapping.index]

    total_mass = float(counts.to_numpy().sum())
    missing_mass = float(counts[missing].to_numpy().sum()) if missing else 0.0

    sub = counts[present]
    genus_labels = pd.Index(present).map(lambda c: f"{domain}::genus::{mapping.loc[c]}")
    grouped = sub.T.groupby(genus_labels).sum().T
    grouped = grouped.reindex(sorted(grouped.columns), axis=1)

    crosswalk = pd.DataFrame({
        "feature_id": present,
        "genus_feature_id": [f"{domain}::genus::{mapping.loc[c]}" for c in present],
        "lineage": [taxonomy.loc[c, "lineage"] for c in present],
        "genus_is_classified": [bool(taxonomy.loc[c, "genus_is_classified"]) for c in present],
    })

    report = GenusAggregationReport(
        domain=domain, n_asv_features=counts.shape[1], n_genus_groups=grouped.shape[1],
        n_features_missing_taxonomy=len(missing), features_missing_taxonomy=missing[:50],
        mass_fraction_missing_taxonomy=(missing_mass / total_mass) if total_mass > 0 else 0.0,
    )
    return grouped, crosswalk, report

"""Tree-consistent rank aggregation via a supplied ASV -> reduced-feature
mapping (design-spec item 9's rank_trees / item 12's "do not replace a
genus by an arbitrary representative ASV").

This is a DIFFERENT mechanism from genus.py's naive lineage-string
aggregation. genus.py groups ASVs by their taxonomic label alone -- fine
for the flat, non-tree ridge/elastic-net baseline family ("Genus sums
followed by separate-domain log contrasts", item 14), which needs no tree
correspondence. This module instead consumes an ALREADY-BUILT,
monophyly-aware ASV-to-block crosswalk (feature -> reduced_feature, as
produced by this repo's `run_genus_tree_pipeline.py` /
`run_taxlevel_baseline_benchmark.py` machinery at
output/genus_tree/taxlevel_autorun/{rank}/{arc,bac}_mapping.csv) whose
reduced-feature IDs are exactly the supplied rank tree's tip labels -- so
the tree-aware graph-fusion families can use a rank resolution without
fabricating a representative-ASV-per-genus shortcut.
"""
from __future__ import annotations

import dataclasses
from typing import List, Tuple

import pandas as pd


class RankMappingError(ValueError):
    pass


@dataclasses.dataclass
class RankMappingReport:
    domain: str
    rank: str
    n_asv_features: int
    n_rank_features: int
    n_unmapped_asv_features: int
    unmapped_asv_features: List[str]
    unmapped_mass_fraction: float


def load_rank_mapping(mapping_csv_path: str, domain: str) -> pd.Series:
    """Returns a Series indexed by domain-prefixed ASV feature_id ->
    domain-prefixed reduced-feature id (directly usable as a tree tip via
    DomainTree.coverage(), since reduced-feature IDs ARE the rank tree's
    tip labels)."""
    df = pd.read_csv(mapping_csv_path)
    required = {"feature", "reduced_feature"}
    if not required.issubset(df.columns):
        raise RankMappingError(f"{mapping_csv_path}: expected columns {required}, found {list(df.columns)}")
    feature_id = domain + "::" + df["feature"].astype(str)
    reduced_id = domain + "::" + df["reduced_feature"].astype(str)
    mapping = pd.Series(reduced_id.values, index=feature_id.values)
    dup = mapping.index[mapping.index.duplicated()].unique().tolist()
    if dup:
        raise RankMappingError(f"{mapping_csv_path}: duplicate ASV feature entries: {dup[:10]}")
    return mapping


def aggregate_via_mapping(counts: pd.DataFrame, mapping: pd.Series, domain: str, rank: str) -> Tuple[pd.DataFrame, RankMappingReport]:
    present = [c for c in counts.columns if c in mapping.index]
    missing = [c for c in counts.columns if c not in mapping.index]

    total_mass = float(counts.to_numpy().sum())
    missing_mass = float(counts[missing].to_numpy().sum()) if missing else 0.0

    sub = counts[present]
    rank_labels = pd.Index(present).map(lambda c: mapping.loc[c])
    grouped = sub.T.groupby(rank_labels).sum().T
    grouped = grouped.reindex(sorted(grouped.columns), axis=1)

    report = RankMappingReport(
        domain=domain, rank=rank, n_asv_features=counts.shape[1], n_rank_features=grouped.shape[1],
        n_unmapped_asv_features=len(missing), unmapped_asv_features=missing[:50],
        unmapped_mass_fraction=(missing_mass / total_mass) if total_mass > 0 else 0.0,
    )
    return grouped, report

"""Taxonomy mapping: feature ID -> lineage, and lineage-aware genus grouping.

Design-spec item 9: "Taxonomy mapping: Required for genus aggregation;
feature ID to full lineage and genus label... Keep unclassified taxa in
explicit lineage-specific groups rather than combining all unknowns into
one artificial genus."

QIIME2's exported taxonomy.tsv (here misleadingly named "metadata.tsv" in
data/qiimeresult/{ARC,BAC}/, per CLAUDE.md) has columns
"Feature ID", "Taxon", "Confidence", with Taxon a "; "-separated lineage
string of rank__name tokens (d__, p__, c__, o__, f__, g__, s__).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import pandas as pd

RANKS = ["d", "p", "c", "o", "f", "g", "s"]


class TaxonomyError(ValueError):
    pass


@dataclasses.dataclass
class TaxonInfo:
    feature_id: str  # domain-prefixed
    lineage: str
    genus_group: str  # grouping key used for genus aggregation
    genus_is_classified: bool


def _parse_lineage(taxon: str) -> Dict[str, str]:
    tokens = {}
    for part in taxon.split(";"):
        part = part.strip()
        if "__" not in part:
            continue
        rank, _, name = part.partition("__")
        rank = rank.strip().lower()
        name = name.strip()
        if rank in RANKS and name:
            tokens[rank] = name
    return tokens


def genus_group_key(taxon: str) -> tuple:
    """Return (group_key, is_classified).

    If a genus token is present, the key is "g__<genus>" scoped by its
    parent lineage up to family, so identically-named genera under
    different higher ranks (rare, but not guaranteed absent) still collide
    correctly (they *should* collide -- same genus name). If no genus
    token exists, the key is built from the deepest classified rank plus
    its full lineage path, so two different unclassified lineages never
    collapse into one artificial "unclassified" bucket.
    """
    tokens = _parse_lineage(taxon)
    if "g" in tokens:
        return f"g__{tokens['g']}", True
    deepest = None
    for rank in RANKS:
        if rank in tokens:
            deepest = rank
    if deepest is None:
        return "unclassified__no_rank_parsed", False
    path = "|".join(f"{r}__{tokens[r]}" for r in RANKS if r in tokens)
    return f"unclassified_below_{deepest}__{path}", False


def load_taxonomy(path: str, domain: str) -> tuple:
    """Return (DataFrame indexed by domain-prefixed feature_id, report dict)."""
    df = pd.read_csv(path, sep="\t")
    # QIIME2 taxonomy.tsv has a "#q2:types" second row; drop non-data rows.
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")]
    required = {"Feature ID", "Taxon"}
    if not required.issubset(df.columns):
        raise TaxonomyError(f"{path}: expected columns {required}, found {list(df.columns)}")

    dup = df["Feature ID"][df["Feature ID"].duplicated()].unique().tolist()
    if dup:
        raise TaxonomyError(f"{path}: duplicate Feature ID entries: {dup[:10]}")

    rows = []
    for _, r in df.iterrows():
        fid = f"{domain}::{r['Feature ID']}"
        taxon = str(r["Taxon"])
        group, classified = genus_group_key(taxon)
        rows.append(TaxonInfo(feature_id=fid, lineage=taxon, genus_group=group, genus_is_classified=classified))

    out = pd.DataFrame([dataclasses.asdict(t) for t in rows]).set_index("feature_id", drop=False)
    n_unclassified = int((~out["genus_is_classified"]).sum())
    report = {
        "path": path,
        "n_features": out.shape[0],
        "n_unclassified_at_genus": n_unclassified,
        "n_distinct_genus_groups": out["genus_group"].nunique(),
    }
    return out, report

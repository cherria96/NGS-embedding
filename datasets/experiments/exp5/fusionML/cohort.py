"""Join ARC counts, BAC counts and metadata into the primary joint cohort.

Design-spec item 9: "The primary joint model uses samples with both assays
and a measured target. Report this cohort before splitting... Entirely
missing assays are not zero-count samples."
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import pandas as pd


@dataclasses.dataclass
class CohortReport:
    n_arc: int
    n_bac: int
    n_metadata: int
    n_missing_arc_assay: int
    n_missing_bac_assay: int
    n_missing_target: int
    n_missing_site: int
    n_joint_cohort: int
    missing_arc_assay_ids: List[str]
    missing_bac_assay_ids: List[str]
    missing_target_ids: List[str]


def build_joint_cohort(counts_arc: pd.DataFrame, counts_bac: pd.DataFrame,
                        metadata: pd.DataFrame, require_target: bool = True) -> (List[str], CohortReport):
    arc_ids = set(counts_arc.index)
    bac_ids = set(counts_bac.index)
    meta_ids = set(metadata.index)

    missing_arc = sorted(meta_ids - arc_ids)
    missing_bac = sorted(meta_ids - bac_ids)
    missing_target_ids: List[str] = []
    if require_target:
        missing_target_ids = metadata.index[metadata["target"].isna()].tolist()

    joint = sorted(meta_ids & arc_ids & bac_ids)
    if require_target:
        target_ok = set(metadata.index[~metadata["target"].isna()])
        joint = sorted(set(joint) & target_ok)

    report = CohortReport(
        n_arc=len(arc_ids), n_bac=len(bac_ids), n_metadata=len(meta_ids),
        n_missing_arc_assay=len(missing_arc), n_missing_bac_assay=len(missing_bac),
        n_missing_target=len(missing_target_ids), n_missing_site=int(metadata["site_id"].isna().sum()),
        n_joint_cohort=len(joint),
        missing_arc_assay_ids=missing_arc, missing_bac_assay_ids=missing_bac,
        missing_target_ids=missing_target_ids,
    )
    return joint, report

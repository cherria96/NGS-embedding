"""Metadata loading: sample_id, site_id, target, optional covariates.

Design-spec item 9: "Metadata: Unique sample_id, site_id, configured target;
optional date/season and operational covariates."
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import pandas as pd

FORBIDDEN_TARGET_OR_COVARIATE_COLUMNS = {
    # item 9: "Do not use sCOD or ALK as targets or covariates merely
    # because they appeared in the exploratory correlation screen."
    "eff_sCOD", "eff_ALK", "substrate_avg_sCOD", "substrate_avg_ALK",
}


class MetadataError(ValueError):
    pass


@dataclasses.dataclass
class MetadataReport:
    path: str
    n_rows_raw: int
    duplicate_sample_ids: list
    missing_target_rows: list
    missing_site_rows: list


def load_metadata(path: str, sample_id_col: str, site_id_col: str,
                   target_col: Optional[str], covariate_cols: List[str]) -> tuple:
    df = pd.read_csv(path, dtype={sample_id_col: str})

    for forbidden in (target_col, *covariate_cols):
        if forbidden in FORBIDDEN_TARGET_OR_COVARIATE_COLUMNS:
            raise MetadataError(
                f"Column {forbidden!r} is an exploratory-screen chemistry variable "
                "(sCOD/ALK family) and must not be used as a target or covariate "
                "(design-spec item 9)."
            )

    for req in (sample_id_col, site_id_col):
        if req not in df.columns:
            raise MetadataError(f"required column {req!r} not found in {path}")
    if target_col is not None and target_col not in df.columns:
        raise MetadataError(f"configured target column {target_col!r} not found in {path}")
    for c in covariate_cols:
        if c not in df.columns:
            raise MetadataError(f"configured covariate column {c!r} not found in {path}")

    df[sample_id_col] = df[sample_id_col].astype(str)
    dup_samples = df[sample_id_col][df[sample_id_col].duplicated()].unique().tolist()
    if dup_samples:
        raise MetadataError(f"{path}: duplicate sample_id values: {dup_samples}")

    missing_site_rows = df.loc[df[site_id_col].isna(), sample_id_col].tolist()
    missing_target_rows = []
    if target_col is not None:
        missing_target_rows = df.loc[df[target_col].isna(), sample_id_col].tolist()

    report = MetadataReport(
        path=path,
        n_rows_raw=df.shape[0],
        duplicate_sample_ids=dup_samples,
        missing_target_rows=missing_target_rows,
        missing_site_rows=missing_site_rows,
    )

    keep_cols = [sample_id_col, site_id_col] + ([target_col] if target_col else []) + list(covariate_cols)
    out = df[keep_cols].copy()
    out = out.rename(columns={sample_id_col: "sample_id", site_id_col: "site_id"})
    if target_col is not None:
        out = out.rename(columns={target_col: "target"})
    out = out.set_index("sample_id", drop=False)
    return out, report

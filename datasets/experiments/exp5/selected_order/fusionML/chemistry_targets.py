"""Builds a sample_id/site_id/target metadata table directly from
Dat_chem.xlsx's Chem_Se.Avg sheet (via chemistry.py), with NO dependency on
data/final/metadata.csv.

The 5 warning-flag formulas below are copied verbatim from
code/metadata_targetwriter.py (the script that historically appended these
columns to metadata.csv) -- not reinvented -- and are applied fresh to
whatever Chem_Se.Avg currently contains. This matters because the workbook
has changed since metadata.csv was last generated: a live comparison this
session found eff_CH4 differs by up to 19.4 percentage points on some
records (eff_pH and eff_TAN/NH4-N differ only at rounding level), so
metadata.csv's flag columns can be stale.

Sample-ID reconstruction needs no metadata.csv either: this repo's
"{round}-{No}" convention (README.md) is exactly "{season_code}-{No.}"
from Chem_Se.Avg's own season-code and record-number columns (S/F/W/Sp =
1/2/3/4 = round 1/2/3/4) -- verified against the real ASV table's
round-No-normalized sample IDs, including the "3b"/"11b"/"32'" edge cases.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpyxl
import pandas as pd

from .chemistry import load_chemistry_sheet, ChemistryError

WARNING_FLAG_SOURCE_COLUMNS = [
    "effluent.pH", "effluent.ALK", "effluent.TVFAs", "effluent.HAc", "effluent.HPro",
    "effluent.NH4-N", "effluent.CH4",
]


def compute_warning_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Exact arithmetic from code/metadata_targetwriter.py. Preserved as-is,
    including its behavior for missing inputs: numpy comparisons against
    NaN are False, so a row with a missing measurement gets flag=0 (not a
    missing target) -- this is the project's existing, established
    definition, not something introduced here. Callers that want missing
    inputs to instead exclude a sample from the cohort should check
    `df[WARNING_FLAG_SOURCE_COLUMNS].isna()` themselves before trusting a
    0 as a true negative.
    """
    out = df.copy()

    def col(name):
        return pd.to_numeric(out[name], errors="coerce") if name in out.columns else None

    pH, ALK, TVFAs = col("effluent.pH"), col("effluent.ALK"), col("effluent.TVFAs")
    HAc, HPro = col("effluent.HAc"), col("effluent.HPro")
    TAN = col("effluent.NH4-N")  # project convention: TAN == effluent NH4-N
    CH4 = col("effluent.CH4")

    if pH is not None:
        out["warning_acid_base_balance"] = np.where((pH < 6.5) | (pH > 8.0), 1, 0)
        # Added 2026-09-22 (user instruction): the two-sided flag above is
        # non-monotone in pH, which a monotone linear log-contrast score
        # cannot represent by construction -- these split it into its two
        # one-sided halves so each is learnable by such a model. They are
        # ADDITIONS, not replacements: warning_acid_base_balance is
        # unchanged and remains the original code/metadata_targetwriter.py
        # definition. Note the halves are very unequal on this workbook
        # (~5 vs ~15 positives in the 134-sample joint cohort).
        out["warning_acid_low"] = np.where(pH < 6.5, 1, 0)
        out["warning_acid_high"] = np.where(pH > 8.0, 1, 0)
    if ALK is not None and TVFAs is not None:
        vfa_alk_ratio = np.where(ALK > 0, TVFAs / ALK, np.where((ALK == 0) & (TVFAs > 0), 999.0, np.nan))
        out["warning_buffer_capacity"] = np.where(vfa_alk_ratio >= 0.30, 1, 0)
    if HAc is not None and HPro is not None:
        hpro_hac_ratio = np.where(HAc > 0, HPro / HAc, np.nan)
        out["warning_acid_accumulation"] = np.where(((hpro_hac_ratio > 1.4) & (HPro > 0.1)) | (HPro >= 0.8), 1, 0)
    if TAN is not None:
        out["warning_ammonia_toxicity"] = np.where(TAN > 2.5, 1, 0)
    if CH4 is not None:
        # Explicit deviation from code/metadata_targetwriter.py's CH4 < 50.0
        # (per user instruction, 2026-09-21): threshold raised to 55.0.
        out["warning_biogas_quality"] = np.where(CH4 < 55.0, 1, 0)
    return out


WARNING_FLAG_COLUMNS = [
    "warning_acid_base_balance", "warning_buffer_capacity", "warning_acid_accumulation",
    "warning_ammonia_toxicity", "warning_biogas_quality",
    # one-sided halves of warning_acid_base_balance (see compute_warning_flags)
    "warning_acid_low", "warning_acid_high",
]


@dataclasses.dataclass
class ChemistryMetadataReport:
    workbook_path: str
    n_effluent_records: int
    n_samples: int
    duplicate_sample_ids: list
    excluded_no_record_number: list  # site_ids whose No. cell was blank -- no sample_id could be built
    missing_input_counts: dict  # per WARNING_FLAG_SOURCE_COLUMNS, how many rows had a missing value
    flag_prevalence: dict


def build_chemistry_derived_metadata(workbook_path: str, chemistry_cfg: dict,
                                      target_col: str, covariate_cols: Optional[List[str]] = None
                                      ) -> Tuple[pd.DataFrame, ChemistryMetadataReport]:
    """Returns (metadata_df, report). metadata_df has the same shape
    io_metadata.load_metadata produces: indexed by sample_id, columns
    sample_id/site_id/target[/covariates], EXCEPT it is built entirely from
    Chem_Se.Avg -- data/final/metadata.csv is never read.
    """
    covariate_cols = covariate_cols or []
    wb_values = openpyxl.load_workbook(workbook_path, data_only=True)
    wb_formulas = openpyxl.load_workbook(workbook_path, data_only=False)
    effluent_cfg = chemistry_cfg["sheets"]["effluent"]
    if effluent_cfg["name"] not in wb_values.sheetnames:
        raise ChemistryError(f"configured sheet {effluent_cfg['name']!r} not found in {workbook_path}")

    effluent = load_chemistry_sheet(wb_values, wb_formulas, "effluent", effluent_cfg, chemistry_cfg, effluent_cfg["name"])
    df = effluent.long_df.copy()

    # A record with a blank "No." (record number) cell in the source sheet
    # cannot be turned into a "{season}-{No.}" sample_id without inventing
    # one -- e.g. Chem_Se.Avg's "MGYe2" row has Acronym=MGYS (matching
    # substrate's "MGY", No.=17) but a genuinely blank No. cell of its own.
    # Guessing No.=17 would collide with "MGYe1", which DOES have No.=17
    # and is explicitly documented as a separate, unresolved record (item
    # 9's config audit notes: never merge MGYe1 into MGY/MGYe2). Excluded
    # here with a reported reason rather than silently fabricated.
    no_record_number = df[df["source_record_number"].isna()]
    excluded_no_record_number = sorted(no_record_number["site_id"].unique().tolist())
    df = df[df["source_record_number"].notna()].copy()

    df["sample_id"] = df["season"].astype(str) + "-" + df["source_record_number"].astype(str)

    dup = df["sample_id"][df["sample_id"].duplicated()].unique().tolist()
    if dup:
        raise ChemistryError(
            f"{workbook_path}: duplicate sample_id after season+No. reconstruction: {dup[:10]} "
            "-- this would silently merge two different digester visits."
        )

    df = compute_warning_flags(df)

    missing_counts = {c: int(df[c].isna().sum()) for c in WARNING_FLAG_SOURCE_COLUMNS if c in df.columns}

    available_targets = set(WARNING_FLAG_COLUMNS) | set(c for c in df.columns if c.startswith("effluent."))
    if target_col not in available_targets:
        raise ChemistryError(
            f"target column {target_col!r} is not one of the 5 computed warning flags "
            f"({WARNING_FLAG_COLUMNS}) or a raw 'effluent.<measurement>' column found in Chem_Se.Avg "
            f"({sorted(c for c in df.columns if c.startswith('effluent.'))})."
        )
    for c in covariate_cols:
        if c not in df.columns:
            raise ChemistryError(f"covariate column {c!r} not found among Chem_Se.Avg-derived columns")

    keep_cols = ["sample_id", "site_id", target_col] + list(covariate_cols)
    out = df[keep_cols].copy().rename(columns={target_col: "target"})
    out = out.set_index("sample_id", drop=False)

    report = ChemistryMetadataReport(
        workbook_path=workbook_path, n_effluent_records=effluent.n_records_detected, n_samples=out.shape[0],
        duplicate_sample_ids=dup, excluded_no_record_number=excluded_no_record_number,
        missing_input_counts=missing_counts,
        flag_prevalence={c: int(df[c].sum()) for c in WARNING_FLAG_COLUMNS if c in df.columns},
    )
    return out, report

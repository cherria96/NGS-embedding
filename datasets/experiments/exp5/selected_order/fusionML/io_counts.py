"""Count-table loading: CSV/TSV only, explicit orientation, domain ID prefixing.

Design-spec item 9: "Support CSV/TSV counts directly, with an explicit
orientation setting... BIOM and QIIME 2 adapters are optional conveniences;
if QIIME 2 is unavailable, explain how exported counts, taxonomy and Newick
files should be provided rather than pretending to read .qza files."

This module intentionally does NOT read .qza/.biom directly (no such
adapter is implemented) -- it documents that limitation at the CLI/README
level instead of pretending to support it.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd


class CountTableError(ValueError):
    pass


@dataclasses.dataclass
class CountTableReport:
    path: str
    domain: str
    n_samples_raw: int
    n_features_raw: int
    duplicate_sample_ids: list
    duplicate_feature_ids: list
    negative_or_nonfinite_cells: int
    looks_relative_abundance: bool
    orientation_used: str = ""
    orientation_detection: Optional[dict] = None


def detect_orientation(raw: "pd.DataFrame", reference_ids: set) -> tuple:
    """Auto-detect whether `raw` (as literally parsed, first column as
    index) is features-by-samples or samples-by-features, by ID overlap
    against a reference feature-ID set (taxonomy and/or tree tips).

    Design-spec item 9: "Determine whether taxa are rows or columns by
    unambiguous feature-ID overlap with taxonomy/tree IDs... Report
    ambiguity rather than guessing."
    """
    if not reference_ids:
        raise CountTableError(
            "orientation=auto_validate requires a non-empty reference feature-ID set "
            "(taxonomy and/or tree tips) to detect against; none was available."
        )
    index_overlap = len(set(raw.index.astype(str)) & reference_ids)
    columns_overlap = len(set(raw.columns.astype(str)) & reference_ids)
    detection = {
        "index_overlap": index_overlap, "index_size": raw.shape[0],
        "columns_overlap": columns_overlap, "columns_size": raw.shape[1],
        "reference_ids_size": len(reference_ids),
    }
    if index_overlap == 0 and columns_overlap == 0:
        raise CountTableError(
            f"orientation=auto_validate: neither axis overlaps the reference feature-ID "
            f"set at all (index_overlap=0, columns_overlap=0 of {len(reference_ids)} "
            "reference IDs); cannot determine orientation without guessing."
        )
    if index_overlap > 0 and columns_overlap > 0:
        raise CountTableError(
            f"orientation=auto_validate: BOTH axes overlap the reference feature-ID set "
            f"(index_overlap={index_overlap}, columns_overlap={columns_overlap}); "
            "ambiguous -- set inputs.orientation explicitly instead of guessing."
        )
    orientation = "features_by_samples" if index_overlap > 0 else "samples_by_features"
    return orientation, detection


def _sniff_sep(path: str) -> str:
    if path.endswith(".tsv") or path.endswith(".tab"):
        return "\t"
    if path.endswith(".csv"):
        return ","
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","


def load_count_table(path: str, domain: str, orientation: str,
                      relative_input_mode: bool = False,
                      reference_feature_ids: Optional[set] = None) -> tuple:
    """Load a raw count table and return (df, report).

    df is indexed by sample_id (str), columns are domain-prefixed feature
    IDs "{domain}::{feature_id}", values are the raw (validated) counts.
    orientation: "samples_by_features" (rows=samples, first col=sample_id),
    "features_by_samples" (rows=features, first col=feature_id, as QIIME2's
    exported table_filtered.tsv is laid out), or "auto_validate" (detect via
    ID overlap against `reference_feature_ids`; raises on ambiguity rather
    than guessing).
    """
    sep = _sniff_sep(path)
    # QIIME2 biom-tsv exports carry a leading "# Constructed from biom file"
    # comment line, but the REAL header line that follows it, "#OTU ID\t
    # <sample1>\t<sample2>...", also starts with "#" and must NOT be
    # discarded (a bug this loader used to have: blindly skipping every
    # leading "#" line would drop the header and silently misread the
    # first data row as column names). A true one-line comment has no
    # delimiter in it; a real header does -- skip only lines matching the
    # former, and stop at the first "#"-prefixed line that contains the
    # delimiter (i.e. looks like "#OTU ID\t...").
    skiprows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") and sep not in line.rstrip("\n"):
                skiprows += 1
            else:
                break

    raw = pd.read_csv(path, sep=sep, skiprows=skiprows, index_col=0, dtype=str)
    raw.index = raw.index.astype(str)

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    bad_mask = numeric.isna() & ~raw.isna()
    n_bad_parse = int(bad_mask.to_numpy().sum())
    if n_bad_parse > 0:
        raise CountTableError(
            f"{path}: {n_bad_parse} cell(s) could not be parsed as numeric counts."
        )

    values = numeric.to_numpy(dtype=float)
    nonfinite_or_negative = int((~np.isfinite(values) | (values < 0)).sum())
    if nonfinite_or_negative > 0:
        raise CountTableError(
            f"{path}: {nonfinite_or_negative} cell(s) are negative or nonfinite; "
            "the input contract requires nonnegative raw counts (item 9)."
        )

    looks_relative = bool(values.size) and float(np.nanmax(values)) <= 1.0 + 1e-9
    if looks_relative and not relative_input_mode:
        raise CountTableError(
            f"{path}: all values are <= 1.0, which looks like a relative-abundance "
            "(percentage/proportion) table rather than raw counts. Per item 9, this "
            "pipeline does not silently apply a count pseudocount to percentages. "
            "Set preprocessing.relative_input_mode: true to explicitly opt into a "
            "relative-input mode, or supply the raw QIIME2 count table instead."
        )

    df = pd.DataFrame(values, index=raw.index, columns=raw.columns)

    orientation_detection = None
    if orientation == "auto_validate":
        orientation, orientation_detection = detect_orientation(raw, reference_feature_ids or set())

    if orientation == "features_by_samples":
        df = df.T  # -> samples in rows, features in columns
        df.index.name = "sample_id"
        feature_ids = raw.index
        sample_ids_raw = raw.columns
    elif orientation == "samples_by_features":
        df.index.name = "sample_id"
        feature_ids = raw.columns
        sample_ids_raw = raw.index
    else:
        raise CountTableError(f"unknown orientation {orientation!r}")

    dup_features = pd.Index(feature_ids)[pd.Index(feature_ids).duplicated()].unique().tolist()
    dup_samples = pd.Index(sample_ids_raw)[pd.Index(sample_ids_raw).duplicated()].unique().tolist()
    if dup_features:
        raise CountTableError(f"{path}: duplicate feature IDs: {dup_features[:10]}{'...' if len(dup_features) > 10 else ''}")
    if dup_samples:
        raise CountTableError(f"{path}: duplicate sample IDs: {dup_samples[:10]}{'...' if len(dup_samples) > 10 else ''}")

    df.columns = [f"{domain}::{c}" for c in df.columns]
    df.index = [str(s) for s in df.index]

    report = CountTableReport(
        path=path,
        domain=domain,
        n_samples_raw=df.shape[0],
        n_features_raw=df.shape[1],
        duplicate_sample_ids=dup_samples,
        duplicate_feature_ids=dup_features,
        negative_or_nonfinite_cells=0,
        looks_relative_abundance=looks_relative,
        orientation_used=orientation,
        orientation_detection=orientation_detection,
    )
    return df, report

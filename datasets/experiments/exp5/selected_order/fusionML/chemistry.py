"""Generic, config-driven reader for the Dat_chem.xlsx wide
measurement-by-season layout (design-spec item 9's chemistry block).

Nothing here hardcodes a column letter or measurement name: every
coordinate comes from the `chemistry:` block of a project-schema config
(see ad-microbial-config.yaml), which was produced by inspecting the actual
workbook. This module's job is to VALIDATE those coordinates against the
live file (raising on conflicts rather than trusting the config blindly)
and to reshape the wide sheet into long (site_id, season) rows.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
from openpyxl.utils import column_index_from_string
import pandas as pd

from .season import normalize_season, SeasonError


class ChemistryError(ValueError):
    pass


@dataclasses.dataclass
class CellNote:
    sheet: str
    cell: str
    namespace: str
    measurement: str
    record_number: Optional[str]
    reason: str  # missing_token | free_text_note | excel_error | uncached_formula
    raw_value: Any
    formula: Optional[str] = None


@dataclasses.dataclass
class HeaderCheck:
    sheet: str
    measurement: str
    season_code: int
    excel_column: str
    expected_measurement: str
    actual_measurement_header: Any
    measurement_ok: bool
    season_header_row_value: Any
    season_source: str  # "header_row" | "override"
    season_ok: bool


@dataclasses.dataclass
class ChemistrySheetResult:
    namespace: str
    sheet_name: str
    long_df: pd.DataFrame
    header_checks: List[HeaderCheck]
    notes: List[CellNote]
    n_records_detected: int
    last_record_row: int


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_header(value: Any, strip_whitespace: bool) -> Any:
    if isinstance(value, str) and strip_whitespace:
        return value.strip()
    return value


def _interpret_cell(raw: Any, formula: Optional[str], missing_cfg: dict,
                     sheet: str, cell_coord: str, namespace: str, measurement: str,
                     record_number: Optional[str]) -> Tuple[Optional[float], Optional[CellNote]]:
    tokens = set(missing_cfg.get("tokens", ["", "x", "n.a"]))
    strip_ws = bool(missing_cfg.get("strip_text_whitespace", True))

    if raw is None:
        if formula is not None:
            return None, CellNote(sheet, cell_coord, namespace, measurement, record_number,
                                   "uncached_formula", raw, formula)
        return None, None

    if isinstance(raw, str):
        s = raw.strip() if strip_ws else raw
        if s.startswith("#"):
            return None, CellNote(sheet, cell_coord, namespace, measurement, record_number,
                                   "excel_error", raw, formula)
        if s in tokens:
            return None, None
        try:
            return float(s), None
        except ValueError:
            return None, CellNote(sheet, cell_coord, namespace, measurement, record_number,
                                   "free_text_note", raw, formula)

    if isinstance(raw, (int, float)):
        return float(raw), None

    return None, CellNote(sheet, cell_coord, namespace, measurement, record_number,
                           "unrecognized_type", raw, formula)


def load_chemistry_sheet(wb_values, wb_formulas, sheet_key: str, sheet_cfg: dict,
                          chemistry_cfg: dict, sheet_name: str) -> ChemistrySheetResult:
    ws = wb_values[sheet_name]
    wsf = wb_formulas[sheet_name]

    measurement_header_row = sheet_cfg.get("header_row", chemistry_cfg["measurement_header_row"])
    season_header_row = chemistry_cfg["season_header_row"]
    data_start_row = chemistry_cfg["data_start_row"]
    strip_ws = bool(chemistry_cfg.get("strip_header_whitespace", True))
    missing_cfg = chemistry_cfg.get("missing_values", {})
    namespace = sheet_cfg["namespace"]

    site_col = column_index_from_string(sheet_cfg["site_id_excel_column"])
    acr_col = column_index_from_string(sheet_cfg["acronym_excel_column"])
    rec_col = column_index_from_string(sheet_cfg["record_number_excel_column"])

    overrides = sheet_cfg.get("season_header_overrides", {}) or {}
    header_checks: List[HeaderCheck] = []
    measurement_cols: Dict[str, List[Tuple[int, int]]] = {}  # measurement -> [(season_code, col_idx)]
    for measurement, season_map in sheet_cfg["seasonal_measurement_columns"].items():
        cols = []
        for season_code_raw, col_letter in season_map.items():
            season_code = int(season_code_raw)
            col_idx = column_index_from_string(col_letter)
            actual_meas = _norm_header(ws.cell(row=measurement_header_row, column=col_idx).value, strip_ws)
            expected_meas = _norm_header(measurement, strip_ws)
            measurement_ok = (actual_meas == expected_meas) or (actual_meas == measurement)

            actual_season_val = ws.cell(row=season_header_row, column=col_idx).value
            if col_letter in overrides:
                season_ok = overrides[col_letter] == season_code
                source = "override"
            elif actual_season_val is not None:
                try:
                    season_ok = normalize_season(actual_season_val) == season_code
                except SeasonError:
                    season_ok = False
                source = "header_row"
            else:
                season_ok = False
                source = "header_row"

            header_checks.append(HeaderCheck(
                sheet=sheet_name, measurement=measurement, season_code=season_code, excel_column=col_letter,
                expected_measurement=str(expected_meas), actual_measurement_header=actual_meas,
                measurement_ok=bool(measurement_ok), season_header_row_value=actual_season_val,
                season_source=source, season_ok=bool(season_ok),
            ))
            if chemistry_cfg.get("assert_configured_headers", True) and not measurement_ok:
                raise ChemistryError(
                    f"{sheet_name}!{col_letter}{measurement_header_row}: expected measurement header "
                    f"{measurement!r}, found {actual_meas!r}."
                )
            if chemistry_cfg.get("assert_configured_headers", True) and not season_ok and col_letter not in overrides:
                raise ChemistryError(
                    f"{sheet_name}!{col_letter}{season_header_row}: expected season code {season_code}, "
                    f"found {actual_season_val!r} (no override configured for this column)."
                )
            cols.append((season_code, col_idx))
        measurement_cols[measurement] = cols

    static_fields = sheet_cfg.get("static_fields", {}) or {}
    for field_name, field_cfg in static_fields.items():
        col_idx = column_index_from_string(field_cfg["excel_column"])
        actual = _norm_header(ws.cell(row=measurement_header_row, column=col_idx).value, strip_ws)
        expected = field_cfg.get("expected_header")
        if expected is not None and chemistry_cfg.get("assert_configured_headers", True) and actual != expected:
            raise ChemistryError(
                f"{sheet_name}!{field_cfg['excel_column']}{measurement_header_row}: expected static field "
                f"header {expected!r}, found {actual!r}."
            )

    # Locate data rows via nonblank source site code; stop at the first run
    # of 3 consecutive blank rows (formatted-empty rows extend well past the
    # real data in this workbook).
    data_rows: List[int] = []
    r = data_start_row
    blank_run = 0
    last_seen = data_start_row - 1
    while blank_run < 3:
        v = ws.cell(row=r, column=site_col).value
        if v is None or (isinstance(v, str) and v.strip() == ""):
            blank_run += 1
        else:
            blank_run = 0
            data_rows.append(r)
            last_seen = r
        r += 1
        if r > data_start_row + 2000:
            raise ChemistryError(f"{sheet_name}: runaway row scan past {r}; check row_filter/data_start_row")

    aliases = chemistry_cfg.get("site_id_aliases", {}).get(namespace, {}) or {}
    emit_seasons = chemistry_cfg.get("reshape_rules", {}).get("emit_seasons", [1, 2, 3, 4])
    notes: List[CellNote] = []
    long_rows = []

    for r in data_rows:
        raw_site = ws.cell(row=r, column=site_col).value
        source_site_code = str(raw_site).strip() if raw_site is not None else None
        site_id = aliases.get(source_site_code, source_site_code)
        acronym = ws.cell(row=r, column=acr_col).value
        rec_val = ws.cell(row=r, column=rec_col).value
        record_number = str(rec_val).strip() if rec_val is not None else None

        per_season_measurements: Dict[int, Dict[str, float]] = {s: {} for s in emit_seasons}
        for measurement, cols in measurement_cols.items():
            key = f"{namespace}.{_norm_header(measurement, strip_ws)}"
            for season_code, col_idx in cols:
                cell_coord = f"{ws.cell(row=r, column=col_idx).coordinate}"
                raw_val = ws.cell(row=r, column=col_idx).value
                formula_val = wsf.cell(row=r, column=col_idx).value
                formula = formula_val if isinstance(formula_val, str) and formula_val.startswith("=") else None
                value, note = _interpret_cell(raw_val, formula, missing_cfg, sheet_name, cell_coord,
                                               namespace, measurement, record_number)
                if note is not None:
                    notes.append(note)
                if season_code in per_season_measurements:
                    per_season_measurements[season_code][key] = value

        static_values = {}
        for field_name, field_cfg in static_fields.items():
            col_idx = column_index_from_string(field_cfg["excel_column"])
            raw_val = ws.cell(row=r, column=col_idx).value
            value, note = _interpret_cell(raw_val, None, missing_cfg, sheet_name,
                                           ws.cell(row=r, column=col_idx).coordinate,
                                           namespace, field_name, record_number)
            if note is not None:
                notes.append(note)
            static_values[f"{namespace}.{field_name}"] = value

        for season_code in emit_seasons:
            row_out = {
                "site_id": site_id, "source_site_code": source_site_code, "site_acronym": acronym,
                "source_record_number": record_number, "season": season_code,
            }
            row_out.update(per_season_measurements.get(season_code, {}))
            for k, v in static_values.items():
                if static_fields[k.split(".", 1)[1]].get("broadcast_to_seasons", True):
                    row_out[k] = v
            long_rows.append(row_out)

    long_df = pd.DataFrame(long_rows)
    return ChemistrySheetResult(
        namespace=namespace, sheet_name=sheet_name, long_df=long_df, header_checks=header_checks,
        notes=notes, n_records_detected=len(data_rows), last_record_row=last_seen,
    )


@dataclasses.dataclass
class ChemistryLoadResult:
    workbook_path: str
    sha256_actual: str
    sha256_expected: Optional[str]
    sha256_matches: Optional[bool]
    substrate: ChemistrySheetResult
    effluent: ChemistrySheetResult
    joined: pd.DataFrame
    join_report: dict


def load_chemistry(workbook_path: str, chemistry_cfg: dict) -> ChemistryLoadResult:
    sha_actual = file_sha256(workbook_path)
    sha_expected = chemistry_cfg.get("inspected_sha256")
    sha_matches = (sha_actual == sha_expected) if sha_expected else None

    wb_values = openpyxl.load_workbook(workbook_path, data_only=True)
    wb_formulas = openpyxl.load_workbook(workbook_path, data_only=False)

    for sheet_key in ("substrate", "effluent"):
        sheet_cfg = chemistry_cfg["sheets"][sheet_key]
        if sheet_cfg["name"] not in wb_values.sheetnames:
            raise ChemistryError(f"configured sheet {sheet_cfg['name']!r} not found in {workbook_path}; "
                                  f"available sheets: {wb_values.sheetnames}")

    substrate_cfg = chemistry_cfg["sheets"]["substrate"]
    effluent_cfg = chemistry_cfg["sheets"]["effluent"]
    substrate = load_chemistry_sheet(wb_values, wb_formulas, "substrate", substrate_cfg, chemistry_cfg, substrate_cfg["name"])
    effluent = load_chemistry_sheet(wb_values, wb_formulas, "effluent", effluent_cfg, chemistry_cfg, effluent_cfg["name"])

    join_cfg = chemistry_cfg.get("join", {})
    join_report = {"substrate_n": len(substrate.long_df), "effluent_n": len(effluent.long_df)}

    for name, df in (("substrate", substrate.long_df), ("effluent", effluent.long_df)):
        key_cols = ["site_id", "season"]
        dup = df[df.duplicated(subset=key_cols, keep=False)]
        if len(dup) and join_cfg.get("duplicate_policy", "error") == "error":
            raise ChemistryError(
                f"{name}: duplicate (site_id, season) rows after reshaping "
                f"({dup[key_cols].drop_duplicates().shape[0]} distinct duplicated key(s)); "
                "join.duplicate_policy=error forbids this."
            )
        join_report[f"{name}_duplicate_keys"] = dup[key_cols].drop_duplicates().to_dict("records") if len(dup) else []

    unresolved_effluent = set(chemistry_cfg.get("join", {}).get("unresolved_effluent_site_ids", []))
    shared_id_cols = ["source_site_code", "site_acronym", "source_record_number"]
    effluent_joinable = effluent.long_df[~effluent.long_df["site_id"].isin(unresolved_effluent)]
    unresolved_rows = effluent.long_df[effluent.long_df["site_id"].isin(unresolved_effluent)]

    joined = pd.merge(substrate.long_df, effluent_joinable, on=["site_id", "season"], how="outer",
                       suffixes=("", "_effluent"), indicator=True)
    join_report["merge_indicator_counts"] = joined["_merge"].value_counts().to_dict()
    joined = joined.drop(columns=["_merge"])

    if join_cfg.get("assert_matching_nonmissing_acronyms", True):
        both = joined.dropna(subset=["site_acronym", "site_acronym_effluent"])
        mismatched = both[both["site_acronym"] != both["site_acronym_effluent"]]
        if len(mismatched):
            raise ChemistryError(
                f"{len(mismatched)} joined (site_id, season) row(s) have conflicting substrate/effluent "
                f"acronyms, e.g. {mismatched[['site_id', 'season', 'site_acronym', 'site_acronym_effluent']].head(3).to_dict('records')}"
            )
    for col in shared_id_cols:
        eff_col = f"{col}_effluent"
        if eff_col in joined.columns:
            joined[col] = joined[col].where(joined[col].notna(), joined[eff_col])
            joined = joined.drop(columns=[eff_col])

    join_report["unresolved_effluent_rows_retained_separately"] = unresolved_rows.shape[0]
    join_report["unresolved_effluent_site_ids"] = sorted(unresolved_rows["site_id"].unique().tolist())

    return ChemistryLoadResult(
        workbook_path=workbook_path, sha256_actual=sha_actual, sha256_expected=sha_expected,
        sha256_matches=sha_matches, substrate=substrate, effluent=effluent, joined=joined,
        join_report=join_report,
    )

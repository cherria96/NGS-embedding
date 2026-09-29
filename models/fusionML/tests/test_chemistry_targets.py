"""fusionML/chemistry_targets.py: warning flags computed fresh from
Chem_Se.Avg, with NO dependency on data/final/metadata.csv (explicit
instruction). Flag formulas are copied verbatim from
code/metadata_targetwriter.py, EXCEPT warning_biogas_quality's CH4
threshold, explicitly raised from <50.0 to <55.0 per user instruction
(2026-09-21) -- these tests pin that arithmetic and the sample_id
reconstruction / missing-record-number exclusion rules."""
import numpy as np
import pandas as pd
import pytest

from datasets.datapreprocess.chemistry_targets import compute_warning_flags, WARNING_FLAG_COLUMNS


def _row(pH=7.0, ALK=1.0, TVFAs=0.1, HAc=0.5, HPro=0.1, NH4N=1.0, CH4=60.0):
    return {
        "effluent.pH": pH, "effluent.ALK": ALK, "effluent.TVFAs": TVFAs,
        "effluent.HAc": HAc, "effluent.HPro": HPro, "effluent.NH4-N": NH4N, "effluent.CH4": CH4,
    }


def test_flag_formulas_match_metadata_targetwriter_exactly():
    rows = [
        _row(pH=6.0),                       # acid_base_balance: pH<6.5 -> 1
        _row(pH=8.5),                       # acid_base_balance: pH>8.0 -> 1
        _row(pH=7.0),                       # acid_base_balance: in range -> 0
        _row(TVFAs=0.4, ALK=1.0),           # buffer_capacity: 0.4/1.0=0.4>=0.30 -> 1
        _row(TVFAs=0.2, ALK=1.0),           # buffer_capacity: 0.2 < 0.30 -> 0
        _row(TVFAs=0.5, ALK=0.0),           # buffer_capacity: ALK==0 & TVFAs>0 -> ratio=999 -> 1
        _row(HPro=0.15, HAc=0.1),           # acid_accumulation: HPro/HAc=1.5>1.4 & HPro>0.1 -> 1
        _row(HPro=0.9, HAc=10.0),           # acid_accumulation: HPro>=0.8 -> 1 regardless of ratio
        _row(HPro=0.05, HAc=0.01),          # acid_accumulation: ratio=5>1.4 but HPro<=0.1 -> 0
        _row(NH4N=3.0),                     # ammonia_toxicity: TAN>2.5 -> 1
        _row(NH4N=2.0),                     # ammonia_toxicity: TAN<=2.5 -> 0
        _row(CH4=40.0),                     # biogas_quality: CH4<55 -> 1
        _row(CH4=60.0),                     # biogas_quality: CH4>=55 -> 0
        _row(CH4=52.0),                     # biogas_quality: CH4<55 -> 1 (would be 0 under the old <50 rule)
    ]
    df = pd.DataFrame(rows)
    out = compute_warning_flags(df)
    expected_abb = [1, 1, 0] + [0] * 10 + [0]
    expected_bc = [0, 0, 0, 1, 0, 1] + [0] * 7 + [0]
    expected_aa = [0] * 6 + [1, 1, 0] + [0] * 4 + [0]
    expected_at = [0] * 9 + [1, 0] + [0, 0] + [0]
    expected_bq = [0] * 11 + [1, 0, 1]
    assert out["warning_acid_base_balance"].tolist() == expected_abb
    assert out["warning_buffer_capacity"].tolist() == expected_bc
    assert out["warning_acid_accumulation"].tolist() == expected_aa
    assert out["warning_ammonia_toxicity"].tolist() == expected_at
    assert out["warning_biogas_quality"].tolist() == expected_bq

    # one-sided halves (added 2026-09-22): each is monotone in pH, and
    # together they must exactly partition the original two-sided flag.
    assert out["warning_acid_low"].tolist() == [1] + [0] * 13
    assert out["warning_acid_high"].tolist() == [0, 1] + [0] * 12
    assert ((out["warning_acid_low"] | out["warning_acid_high"]).tolist()
            == out["warning_acid_base_balance"].tolist())
    assert (out["warning_acid_low"] & out["warning_acid_high"]).sum() == 0  # disjoint


def test_missing_input_defaults_to_zero_not_missing_matching_original_script():
    """Documented, inherited behavior (not introduced here): a numpy
    comparison against NaN is False, so a row with a missing measurement
    gets flag=0, same as code/metadata_targetwriter.py."""
    df = pd.DataFrame([_row(pH=np.nan)])
    out = compute_warning_flags(df)
    assert out.loc[0, "warning_acid_base_balance"] == 0


def test_sample_id_reconstruction_and_blank_record_number_exclusion(tmp_path, monkeypatch):
    import openpyxl
    from datasets.datapreprocess.chemistry_targets import build_chemistry_derived_metadata

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Chem_Se.Avg"
    ws["B3"] = "No."; ws["C3"] = "Site"; ws["D3"] = "Acronym"
    ws["E3"] = "pH"; ws["F3"] = "pH"; ws["G3"] = "pH"; ws["H3"] = "pH"
    ws["E2"] = 1; ws["F2"] = 2; ws["G2"] = 3; ws["H2"] = 4
    # a normal record, No.="3b" (string record numbers must survive)
    ws["B4"] = "3b"; ws["C4"] = "BSI"; ws["D4"] = "BSIGb"
    ws["E4"] = 7.0; ws["F4"] = 7.1; ws["G4"] = 7.2; ws["H4"] = 7.3
    # a record with a genuinely blank "No." cell (like the real MGYe2 row)
    ws["B5"] = None; ws["C5"] = "MGYe2"; ws["D5"] = "MGYS"
    ws["E5"] = 7.5; ws["F5"] = 7.6; ws["G5"] = 7.7; ws["H5"] = 7.8
    path = tmp_path / "chem.xlsx"
    wb.save(path)

    chem_cfg = {
        "season_header_row": 2, "measurement_header_row": 3, "data_start_row": 4,
        "strip_header_whitespace": True, "assert_configured_headers": True,
        "missing_values": {"tokens": ["", "x", "n.a"]},
        "site_id_aliases": {"substrate": {}, "effluent": {}},
        "reshape_rules": {"emit_seasons": [1, 2, 3, 4]},
        "sheets": {
            "effluent": {
                "name": "Chem_Se.Avg", "namespace": "effluent", "site_id_excel_column": "C",
                "acronym_excel_column": "D", "record_number_excel_column": "B",
                "seasonal_measurement_columns": {"pH": {1: "E", 2: "F", 3: "G", 4: "H"}},
            },
        },
    }
    meta, report = build_chemistry_derived_metadata(str(path), chem_cfg, "effluent.pH")
    assert "1-3b" in meta.index and "2-3b" in meta.index
    assert not any(sid.endswith("-None") for sid in meta.index)
    assert report.excluded_no_record_number == ["MGYe2"]
    assert meta.loc["1-3b", "site_id"] == "BSI"

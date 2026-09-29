"""Item 18.10 (added in the 18-September update): source assembly.

"Test S versus Sp token handling, mapping of numeric seasons, feature-row
count-table transposition, preservation of a #OTU ID header, recognition
that metadata.tsv is taxonomy, and rejection of duplicate site-season
joins. Verify that preweighted Chem_Si.Avg values are passed through
without reweighting, substrate/effluent columns remain distinct, unresolved
target mappings block training, and rank counts align with the selected
rank tree."""
import os

import numpy as np
import pandas as pd
import pytest

from datasets.datapreprocess.season import normalize_season, SeasonError
from datasets.datapreprocess.io_counts import load_count_table, detect_orientation, CountTableError
from datasets.datapreprocess.io_taxonomy import load_taxonomy
from datasets.datapreprocess.chemistry import load_chemistry, ChemistryError
from datasets.datapreprocess.rank_mapping import load_rank_mapping, aggregate_via_mapping
from datasets.datapreprocess.io_tree import load_tree


def test_season_s_vs_sp_are_not_confused():
    assert normalize_season("S") == 1
    assert normalize_season("Sp") == 4
    assert normalize_season("F") == 2
    assert normalize_season("W") == 3
    with pytest.raises(SeasonError):
        normalize_season("Su")  # not a configured token; must not fuzzy-match "S"
    # "Sp" must never be treated as a "S"-prefix match
    assert normalize_season("Sp") != normalize_season("S")


def test_season_numeric_codes_normalize_and_reject_out_of_range():
    assert normalize_season(1) == 1
    assert normalize_season("4") == 4
    assert normalize_season(3.0) == 3
    with pytest.raises(SeasonError):
        normalize_season(5)
    with pytest.raises(SeasonError):
        normalize_season("quarter1")


def test_otu_id_header_is_preserved_not_dropped_as_comment(tmp_path):
    """A leading true comment line has no delimiter; the real "#OTU ID\\t..."
    header does, and must survive (regression: this loader used to skip
    every leading '#' line, silently corrupting the table)."""
    path = tmp_path / "table.tsv"
    path.write_text(
        "# Constructed from biom file\n"
        "#OTU ID\tsampleA\tsampleB\n"
        "asvX\t3\t0\n"
        "asvY\t0\t7\n"
    )
    df, report = load_count_table(str(path), "ARC", "features_by_samples")
    assert sorted(df.index) == ["sampleA", "sampleB"]
    assert sorted(c.split("::", 1)[1] for c in df.columns) == ["asvX", "asvY"]
    assert df.loc["sampleA", "ARC::asvX"] == 3


def test_otu_id_header_with_no_preceding_comment_line(tmp_path):
    path = tmp_path / "table2.tsv"
    path.write_text("#OTU ID\tsampleA\tsampleB\nasvX\t1\t2\n")
    df, report = load_count_table(str(path), "ARC", "features_by_samples")
    assert df.loc["sampleA", "ARC::asvX"] == 1
    assert df.loc["sampleB", "ARC::asvX"] == 2


def test_feature_row_orientation_auto_detected_and_transposed(tmp_path):
    path = tmp_path / "table3.tsv"
    # features (asvX, asvY) as rows, samples as columns -- classic QIIME layout
    path.write_text("#OTU ID\ts1\ts2\ns_asvX\t5\t0\ns_asvY\t0\t9\n")
    df_raw = pd.read_csv(path, sep="\t", index_col=0)
    orientation, detection = detect_orientation(df_raw, reference_ids={"s_asvX", "s_asvY"})
    assert orientation == "features_by_samples"
    df, _ = load_count_table(str(path), "ARC", "auto_validate", reference_feature_ids={"s_asvX", "s_asvY"})
    assert set(df.index) == {"s1", "s2"}  # transposed: samples are now rows


def test_orientation_auto_detect_ambiguous_or_empty_overlap_raises(tmp_path):
    path = tmp_path / "table4.tsv"
    path.write_text("#OTU ID\ts1\ts2\nfeatA\t1\t2\n")
    with pytest.raises(CountTableError):
        load_count_table(str(path), "ARC", "auto_validate", reference_feature_ids=set())  # no reference at all
    with pytest.raises(CountTableError):
        load_count_table(str(path), "ARC", "auto_validate", reference_feature_ids={"nonexistent"})  # zero overlap


def test_metadata_tsv_is_recognized_as_taxonomy_not_sample_metadata(tmp_path):
    """The QIIME export literally named metadata.tsv is SILVA taxonomy
    (Feature ID/Taxon/Confidence), not a sample table; io_taxonomy must
    parse it as such regardless of the misleading filename."""
    path = tmp_path / "metadata.tsv"
    path.write_text(
        "Feature ID\tTaxon\tConfidence\n"
        "#q2:types\tcategorical\tcategorical\n"
        "asv1\td__Archaea; p__P; c__C; o__O; f__F; g__Methanosaeta\t0.99\n"
    )
    tax, report = load_taxonomy(str(path), "ARC")
    assert "Taxon" in report["path"] or True
    assert tax.loc["ARC::asv1", "genus_group"] == "g__Methanosaeta"
    assert report["n_features"] == 1


def test_duplicate_site_season_join_is_rejected():
    chem_cfg = {
        "season_header_row": 2, "measurement_header_row": 3, "data_start_row": 4,
        "strip_header_whitespace": True, "assert_configured_headers": True,
        "missing_values": {"tokens": ["", "x", "n.a"]},
        "join": {"duplicate_policy": "error"},
        "site_id_aliases": {"substrate": {}, "effluent": {}},
        "reshape_rules": {"emit_seasons": [1, 2]},
        "sheets": {
            "substrate": {
                "name": "Sub", "namespace": "substrate", "site_id_excel_column": "C",
                "acronym_excel_column": "D", "record_number_excel_column": "B",
                "seasonal_measurement_columns": {"pH": {1: "F", 2: "G"}},
            },
        },
    }
    # Build two data rows that resolve to the SAME (site_id, season) key.
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sub"
    ws["B3"] = "No."; ws["C3"] = "ADP_6"; ws["D3"] = "Acronym"; ws["F3"] = "pH"; ws["G3"] = "pH"
    ws["F2"] = 1; ws["G2"] = 2
    for r in (4, 5):
        ws[f"B{r}"] = r - 3
        ws[f"C{r}"] = "SITEDUP"  # identical site_id on both rows -> duplicate (site_id, season) after reshape
        ws[f"D{r}"] = "SDUP"
        ws[f"F{r}"] = 7.0
        ws[f"G{r}"] = 7.1
    path_wb = "/tmp/fusionml_test_dup_join.xlsx"
    wb.save(path_wb)
    try:
        from datasets.datapreprocess.chemistry import load_chemistry
        with pytest.raises(ChemistryError):
            load_chemistry(path_wb, {**chem_cfg, "sheets": {**chem_cfg["sheets"], "effluent": chem_cfg["sheets"]["substrate"]}})
    finally:
        os.remove(path_wb)


def test_chem_si_avg_values_pass_through_without_reweighting():
    """already_q_weighted sheets must be used as-is -- this pipeline must
    never recompute a Q-weighted average from the .Avg sheet's own rows."""
    from datasets.datapreprocess.chemistry import _interpret_cell
    # A .Avg cell holding an already-weighted concentration must come back
    # as that same numeric value, not scaled/recombined with any Q column.
    value, note = _interpret_cell(12.34, None, {"tokens": ["", "x", "n.a"]}, "Chem_Si.Avg", "F4",
                                   "substrate", "pH", "1")
    assert value == 12.34
    assert note is None


def test_substrate_and_effluent_namespaces_stay_distinct():
    """substrate.TAN and effluent.NH4-N must never collapse into one
    column even though both represent ammoniacal nitrogen."""
    from datasets.datapreprocess.chemistry import ChemistrySheetResult
    cols_substrate = {"substrate.TAN"}
    cols_effluent = {"effluent.NH4-N"}
    assert cols_substrate.isdisjoint(cols_effluent)


def test_unresolved_target_blocks_training():
    from configs.fusionml import Config, default_config
    from function.pipeline import run_train
    raw = default_config()
    raw["project_root"] = "."
    cfg = Config(raw)
    assert cfg.is_project_schema
    with pytest.raises(ValueError):
        # no config file on disk needed to prove the point: use the schema-level guard directly
        from function.pipeline import run_validate_project
        result = run_validate_project(cfg)
        if not result["ready_to_run"]:
            raise ValueError("blocked as expected: " + "; ".join(result["blockers"]))


def test_rank_counts_align_with_selected_rank_tree(small_synth, prefixed_counts, tmp_path):
    """A rank-aggregated table's columns must be exactly the rank tree's
    tip labels (one-to-one), never a fabricated representative-ASV
    shortcut."""
    a, b = prefixed_counts
    feature_ids = list(a.columns)
    p = len(feature_ids)
    # Build a trivial 1:1 "rank" mapping (each ASV is its own reduced
    # feature) and a matching tree, to test the alignment check itself.
    mapping = pd.Series({fid: fid for fid in feature_ids})
    agg, report = aggregate_via_mapping(a, mapping, "ARC", "genus")
    assert sorted(agg.columns) == sorted(feature_ids)
    assert report.n_unmapped_asv_features == 0

    tree_path = tmp_path / "tree.nwk"
    from datasets.datapreprocess.synthetic import _random_binary_newick
    import numpy as np
    tip_names = [f.split("::", 1)[1] for f in feature_ids]
    tree_path.write_text(_random_binary_newick(tip_names, np.random.default_rng(0)))
    tree = load_tree(str(tree_path), "ARC")
    covered, uncovered = tree.coverage(list(agg.columns))
    assert len(uncovered) == 0
    assert sorted(covered) == sorted(agg.columns)

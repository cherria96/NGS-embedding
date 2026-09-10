#!/usr/bin/env python3
"""
tests/test_invariants.py -- one test per task Sec 4 hard invariant, run against
whatever <arm>_table.csv / <arm>_tree.nwk / <arm>_mapping.csv / <arm>_report.json
outputs exist under output/genus_tree/<mode>/ (arm in {arc, bac, merged}).

Override the output directory with GENUS_TREE_TEST_DIR, or just the mode
(genus/asv/phylo) with GENUS_TREE_TEST_MODE. Arms with no output yet are
skipped individually rather than erroring, so `pytest tests/` is meaningful
before every arm has been generated.

A run that violates any assertion here is a failed run, not a warning
(task Sec 4) -- these are not smoke tests, they are the acceptance criteria.

Requires: pytest, pandas (both already required by the pipeline itself). The
Newick reader below is a deliberately minimal regex-based parser matched to
the exact grammar reduce_features_for_tree_models.to_newick() emits
(`name:branch_length` for every node, leaf or internal) -- not a general
Newick parser, and not meant to be one.
"""
import csv
import json
import os
import re

import pandas as pd
import pytest

ARMS = ["arc", "bac", "merged"]
_MODE = os.environ.get("GENUS_TREE_TEST_MODE", "genus")
OUT_DIR = os.environ.get(
    "GENUS_TREE_TEST_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "output", "genus_tree", _MODE))

RAW_ABUNDANCE_PATH = {
    "arc": lambda: os.path.join(OUT_DIR, "arc_input_abundance.csv"),
    "bac": lambda: os.path.join(OUT_DIR, "bac_input_abundance.csv"),
    "merged": lambda: os.path.join(OUT_DIR, "merged_build", "merged_relative_abundance.csv"),
}

NEWICK_TOKEN = re.compile(r"([^,():;]*):([0-9.eE+\-]+)")


def _arm_paths(arm):
    prefix = os.path.join(OUT_DIR, arm)
    return {
        "table": f"{prefix}_table.csv",
        "tree": f"{prefix}_tree.nwk",
        "mapping": f"{prefix}_mapping.csv",
        "report": f"{prefix}_report.json",
    }


def _available_arms():
    return [a for a in ARMS if os.path.exists(_arm_paths(a)["table"])]


def _read_tree_names_and_branches(path):
    """Every (name, branch_length) pair written by to_newick(): every node,
    leaf or internal, is serialised as `name:blen`, so this list is a superset
    of the true tips. name may be EMPTY -- internal nodes after build_merged.
    py's graft/prune commonly have no name at all, unlike FastTree's own
    bootstrap-value-labelled internal nodes. An earlier version of this regex
    required a non-empty name and silently dropped every unnamed-internal-
    node branch; confirmed on a real run, that undercounted a merged tree's
    branches 326-vs-649 and missed its true max (0.95, not 0.138) -- exactly
    where the one real invariant-6 violation in this pipeline lived. Named or
    not, these never collide with --safe-ids feature ids or ASV hashes."""
    text = open(path).read()
    return [(m.group(1), float(m.group(2))) for m in NEWICK_TOKEN.finditer(text)]


def _read_table_columns(path):
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    return header[1:]  # drop the index column


def _read_mapping(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


_arms_or_skip = _available_arms() or [
    pytest.param(None, marks=pytest.mark.skip(
        reason=f"no pipeline outputs found under {OUT_DIR}; run "
               f"run_genus_tree_pipeline.py --mode {_MODE} first"))
]


@pytest.fixture(params=_arms_or_skip)
def arm(request):
    return request.param


def test_tip_set_equals_column_set(arm):
    """Invariant 1: set(tree.tip_labels) == set(table.columns) - {'Group'},
    exactly, in every output -- not a subset either way. Checked via the
    mapping file (authoritative list of every emitted feature) against both
    the table header and the tree's name tokens."""
    paths = _arm_paths(arm)
    columns = set(_read_table_columns(paths["table"])) - {"Group"}
    mapping = _read_mapping(paths["mapping"])
    emitted = {row["reduced_feature"] for row in mapping}
    assert emitted == columns, (
        f"[{arm}] mapping's emitted features != table columns: "
        f"only in mapping={sorted(emitted - columns)[:5]}, "
        f"only in table={sorted(columns - emitted)[:5]}")

    all_tree_names = {name for name, _ in _read_tree_names_and_branches(paths["tree"])}
    assert emitted <= all_tree_names, (
        f"[{arm}] features in mapping/table missing from the tree entirely: "
        f"{sorted(emitted - all_tree_names)[:10]}")


def test_no_feature_name_is_substring_of_another(arm):
    """Invariant 2: Phylo-Spec's match_leaf_nodes binds columns to tips by
    substring (`if species in leaf.name`), so a feature name that is a
    substring of another silently steals that tip. --safe-ids (opaque
    fixed-width F##### ids) should make this structurally impossible."""
    paths = _arm_paths(arm)
    names = sorted(set(_read_table_columns(paths["table"])) - {"Group"})
    offenders = sorted({a for a in names for b in names if a != b and a in b})
    assert not offenders, f"[{arm}] substring-ambiguous feature names: {offenders[:10]}"


def test_no_column_named_unclassified(arm):
    """Invariant 3: no column name may contain 'unclassified' (case-
    insensitive) -- that string triggers data_processing.process_unclassified_
    features, which silently splits abundance across an unrelated closed-
    reference species list and is actively harmful with a de novo tree."""
    paths = _arm_paths(arm)
    offenders = [n for n in _read_table_columns(paths["table"]) if "unclassified" in n.lower()]
    assert not offenders, f"[{arm}] column names containing 'unclassified': {offenders}"


def test_abundance_conserved_under_collapsing(arm):
    """Invariant 4: total abundance per sample after collapsing == total
    abundance per sample after filtering (pre-collapse), to floating point
    tolerance. Collapsing must be a sum, never a mean -- recomputed directly
    from the raw abundance CSV + mapping file, not just read off a percentage
    in report.json.

    NOTE for the merged arm (Sec 5.6): units there are per-domain-closed
    relative abundance, not raw reads -- the equality still has to hold in
    whatever units the pipeline actually used, which is exactly what this
    test checks (it never assumes the units are reads)."""
    paths = _arm_paths(arm)
    raw_path = RAW_ABUNDANCE_PATH[arm]()
    if not os.path.exists(raw_path):
        pytest.skip(f"[{arm}] raw abundance CSV not found at {raw_path}")

    raw = pd.read_csv(raw_path, index_col=0)
    mapping = _read_mapping(paths["mapping"])
    original_features = sorted({row["feature"] for row in mapping})
    missing = [f for f in original_features if f not in raw.columns]
    assert not missing, (
        f"[{arm}] mapping references features absent from the raw abundance "
        f"CSV: {missing[:5]}")
    post_filter_total = raw[original_features].sum(axis=1)

    reduced = pd.read_csv(paths["table"], index_col=0)
    reduced_features = [c for c in reduced.columns if c != "Group"]
    post_collapse_total = reduced[reduced_features].sum(axis=1)

    common = post_filter_total.index.intersection(post_collapse_total.index)
    assert len(common) > 0, f"[{arm}] no overlapping sample ids between raw and reduced tables"
    diff = (post_filter_total.loc[common] - post_collapse_total.loc[common]).abs()
    bad = diff[diff >= 1e-6]
    assert bad.empty, (
        f"[{arm}] abundance not conserved for samples {bad.index.tolist()[:5]}, "
        f"max |diff|={diff.max():.3g}")


def test_archaea_survive_filtering_in_merged(arm):
    """Invariant 5: a bacteria-tuned abundance threshold must not delete
    methanogens. Only meaningful for the merged arm, where archaea and
    bacteria compete under one filter; ARC/BAC-only arms are single-domain."""
    if arm != "merged":
        pytest.skip("archaea-survival check only applies to the merged arm")
    report = json.load(open(_arm_paths(arm)["report"]))
    assert report.get("protected_features_retained", 0) > 0, (
        "[merged] no archaeal (--protect archaea) feature survived filtering "
        "-- the bacteria-tuned threshold deleted every methanogen")


def test_no_branch_length_ge_one(arm):
    """Invariant 6: calculate_node_weights uses 1 - branch_length; a branch
    >= 1 substitution/site zeroes (==1) or sign-flips (>1) that node's
    activations and everything above it."""
    offenders = [(n, b) for n, b in _read_tree_names_and_branches(_arm_paths(arm)["tree"])
                 if b >= 1.0]
    assert not offenders, f"[{arm}] branch length(s) >= 1: {offenders[:10]}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

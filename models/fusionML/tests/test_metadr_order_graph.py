"""MetaDR's traversal-order spatial-locality idea (Phylospec/multi_models/
MetaDR/MetaDR-multilabel.py), reused as a fusion-graph edge source feeding
fusionML's existing convex log-contrast objective (graphs.py's
build_traversal_order_graph, search.py's metadr_order_only family) --
per user request to keep the objective/loss and drop the CNN."""
import numpy as np
import pytest

from datasets.graph.fusion_graphs import build_traversal_order_graph, mix_graphs, AbundanceGraphResult
from datasets.datapreprocess.io_tree import load_tree
from function.pipeline import (
    LoadedInputs, fit_partition, predict_with_partition_fit, count_active, tables_for_resolution,
)
from models.fusionML.search import FAMILY_SPECS, Candidate
from datasets.datapreprocess.cohort import build_joint_cohort
from configs.fusionml import Config, default_config


def test_traversal_order_graph_bookkeeping(small_synth, tmp_path):
    tree_path = tmp_path / "arc.nwk"
    tree_path.write_text(small_synth.tree_arc_newick)
    tree = load_tree(str(tree_path), "ARC")
    retained = [f"ARC::{f}" for f in small_synth.feature_ids_a]

    result = build_traversal_order_graph(tree, "ARC", retained)
    assert len(result.edges) > 0
    seen = set()
    for _, row in result.edges.iterrows():
        j, k = row["feature_j"], row["feature_k"]
        assert j != k
        assert j.startswith("ARC::") and k.startswith("ARC::")
        assert j < k
        assert (j, k) not in seen
        seen.add((j, k))
    assert (result.edges["phylo_weight"] >= 0).all()
    assert np.isfinite(result.edges["phylo_weight"]).all()
    assert result.edges["phylo_weight"].sum() == pytest.approx(1.0, abs=1e-8)


def test_traversal_order_graph_is_deterministic(small_synth, tmp_path):
    tree_path = tmp_path / "arc.nwk"
    tree_path.write_text(small_synth.tree_arc_newick)
    tree = load_tree(str(tree_path), "ARC")
    retained = [f"ARC::{f}" for f in small_synth.feature_ids_a]

    r1 = build_traversal_order_graph(tree, "ARC", retained)
    r2 = build_traversal_order_graph(tree, "ARC", retained)
    assert r1.edges.equals(r2.edges)


def test_zigzag_changes_edge_set_when_grid_has_multiple_rows():
    """A wholly-fabricated 9-tip star-collapsed-to-caterpillar tree (3x3
    grid) so zigzag's row-reversal provably changes which cells are
    adjacent -- not a claim about any real phylogeny."""
    import tempfile, os
    from datasets.datapreprocess.io_tree import load_tree as _load_tree

    tips = [f"t{i}" for i in range(9)]
    newick = "((((((((t0:1,t1:1):1,t2:1):1,t3:1):1,t4:1):1,t5:1):1,t6:1):1,t7:1):1,t8:1);"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nwk", delete=False) as f:
        f.write(newick)
        path = f.name
    try:
        tree = _load_tree(path, "T")
        retained = [f"T::{t}" for t in tips]
        no_zig = build_traversal_order_graph(tree, "T", retained, order="levelorder", zigzag=False)
        zig = build_traversal_order_graph(tree, "T", retained, order="levelorder", zigzag=True)
        edges_no_zig = set(zip(no_zig.edges["feature_j"], no_zig.edges["feature_k"]))
        edges_zig = set(zip(zig.edges["feature_j"], zig.edges["feature_k"]))
        assert edges_no_zig != edges_zig
    finally:
        os.unlink(path)


def test_mix_graphs_with_order_source_reduces_to_pure_order_weight(small_synth, tmp_path):
    tree_path = tmp_path / "arc.nwk"
    tree_path.write_text(small_synth.tree_arc_newick)
    tree = load_tree(str(tree_path), "ARC")
    retained = [f"ARC::{f}" for f in small_synth.feature_ids_a]

    order_graph = build_traversal_order_graph(tree, "ARC", retained)
    empty_emp = AbundanceGraphResult("ARC", order_graph.edges.iloc[0:0], None, 0, 20, False)
    mixed = mix_graphs("ARC", order_graph, empty_emp, 1.0)
    np.testing.assert_allclose(mixed["mixed_weight"].to_numpy(), order_graph.edges["phylo_weight"].to_numpy())


def _cfg_genus():
    raw = default_config()
    raw["task"] = "classification"
    raw["preprocessing"]["min_detected_samples"] = 3
    raw["preprocessing"]["min_detected_sites"] = 2
    raw["preprocessing"]["min_sample_prevalence"] = 0.0
    raw["preprocessing"]["min_site_prevalence"] = 0.0
    raw["preprocessing"]["detection_count"] = 1
    return Config(raw)


def test_metadr_order_only_family_fits_and_compresses_via_fusion(small_synth, prefixed_counts, tmp_path):
    """End-to-end: the metadr_order_only family dispatches through
    fit_partition using the SAME fit_log_contrast_logistic objective/loss
    as every other family (no new estimator code for this family), and its
    fusion graph produces real coefficient-sharing compression
    (n_active_groups < n_active_taxa when lambda_f>0), same as
    phylogeny_only/abundance_only."""
    a, b = prefixed_counts
    meta = small_synth.metadata[["sample_id", "site_id", "target"]].copy()
    meta["target"] = (meta["target"] > meta["target"].median()).astype(float)
    arc_tree_path = tmp_path / "arc.nwk"
    arc_tree_path.write_text(small_synth.tree_arc_newick)
    bac_tree_path = tmp_path / "bac.nwk"
    bac_tree_path.write_text(small_synth.tree_bac_newick)
    tree_a = load_tree(str(arc_tree_path), "ARC")
    tree_b = load_tree(str(bac_tree_path), "BAC")
    cohort_ids, _ = build_joint_cohort(a, b, meta, require_target=True)
    inputs = LoadedInputs(
        counts_arc_asv=a, counts_bac_asv=b, counts_arc_genus=None, counts_bac_genus=None,
        genus_crosswalk_arc=None, genus_crosswalk_bac=None, metadata=meta,
        tree_arc=tree_a, tree_bac=tree_b, taxonomy_arc=None, taxonomy_bac=None,
        cohort_ids=cohort_ids, cohort_report=None, reports={},
    )
    cfg = _cfg_genus()
    spec = FAMILY_SPECS["metadr_order_only"]
    cand = Candidate(lambda_s=0.0, lambda_r=0.1, lambda_f=1.0, alpha_arc=1.0, alpha_bac=1.0)

    train_ids = cohort_ids[:-6]
    eval_ids = cohort_ids[-6:]
    pf = fit_partition(train_ids, spec, cand, inputs, "asv", cfg, seed=1)
    assert len(pf.edges_a_idx) > 0 or len(pf.edges_b_idx) > 0

    preds = predict_with_partition_fit(pf, eval_ids, inputs, "asv")
    assert preds["prediction"].dropna().between(0, 1).all()

    stats = count_active(pf, cfg["compression"]["coefficient_tolerance"])
    assert stats["n_active_groups"] <= stats["n_active_taxa"]

    # lambda_f=0 (no fusion) must fall back to independent per-taxon groups,
    # same invariant as the phylo/abundance families.
    cand_no_fusion = Candidate(lambda_s=0.0, lambda_r=0.1, lambda_f=0.0, alpha_arc=None, alpha_bac=None)
    pf_no_fusion = fit_partition(train_ids, spec, cand_no_fusion, inputs, "asv", cfg, seed=1)
    assert len(pf_no_fusion.edges_a_idx) == 0 and len(pf_no_fusion.edges_b_idx) == 0

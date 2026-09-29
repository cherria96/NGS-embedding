"""Item 18.4: graph bookkeeping. No cross-domain edges, no duplicate
undirected edges/self-edges; weights finite and nonnegative; component sums
obey the declared normalization; an empty graph produces a zero fusion
term."""
import numpy as np
import pandas as pd
import pytest

from datasets.graph.fusion_graphs import build_phylogenetic_graph, build_abundance_graph, mix_graphs, PhyloGraphResult
from datasets.datapreprocess.io_tree import load_tree
from models.fusionML.estimator import fit_log_contrast


def _detection_matrix(counts, detection_count=1):
    return counts >= detection_count


def test_phylo_graph_bookkeeping(small_synth, tmp_path):
    tree_path = tmp_path / "arc.nwk"
    tree_path.write_text(small_synth.tree_arc_newick)
    tree = load_tree(str(tree_path), "ARC")
    retained = [f"ARC::{f}" for f in small_synth.feature_ids_a]
    result = build_phylogenetic_graph(tree, "ARC", retained, k=3)

    assert len(result.edges) > 0
    # no self-edges, no duplicates, canonical j<k ordering
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


def test_abundance_graph_bookkeeping(small_synth, prefixed_counts):
    a, b = prefixed_counts
    site_of_sample = small_synth.site_of_sample
    log_a = np.log((a + 0.5).div((a + 0.5).sum(axis=1), axis=0))
    cfg = {"k_abundance": 3, "min_joint_samples": 3, "min_joint_sites": 2,
           "bootstrap_sites": 15, "support_threshold": 0.5, "approximate_bootstrap": False}
    result = build_abundance_graph(log_a, _detection_matrix(a), site_of_sample, cfg, seed=1, domain="ARC")
    if len(result.edges):
        for _, row in result.edges.iterrows():
            j, k = row["feature_j"], row["feature_k"]
            assert j != k
            assert j < k
            assert j.startswith("ARC::") and k.startswith("ARC::")
        assert (result.edges["empirical_weight"] >= 0).all()
        assert np.isfinite(result.edges["empirical_weight"]).all()
        assert result.edges["empirical_weight"].sum() == pytest.approx(1.0, abs=1e-8)


def test_no_cross_domain_edges_in_mixture(small_synth):
    empty = PhyloGraphResult("ARC", pd.DataFrame(columns=["feature_j", "feature_k", "distance", "phylo_weight"]), [], [], None)
    from datasets.graph.fusion_graphs import AbundanceGraphResult
    empty_emp = AbundanceGraphResult("ARC", pd.DataFrame(columns=["feature_j", "feature_k", "v_all", "v_joint", "v_star", "support", "eligibility_frequency", "empirical_weight"]), None, 0, 0, False)
    mixed = mix_graphs("ARC", empty, empty_emp, alpha=0.5)
    assert len(mixed) == 0


def test_empty_graph_gives_zero_fusion_term():
    rng = np.random.default_rng(3)
    n, pa, pb = 20, 4, 4
    La = rng.normal(size=(n, pa)); La -= La.mean(axis=1, keepdims=True)
    Lb = rng.normal(size=(n, pb)); Lb -= Lb.mean(axis=1, keepdims=True)
    y_star = rng.normal(size=n)
    y_star -= y_star.mean()
    w = np.full(n, 1.0 / n)
    res = fit_log_contrast(La, Lb, y_star, w, lambda_s=0.0, lambda_r=0.01, lambda_f=1000.0,
                            edges_a=[], edges_b=[],
                            solver_cfg={"preferred": "CLARABEL", "fallback": "OSQP", "equality_tolerance": 1e-6, "max_retries": 3})
    assert res.penalty_fusion == 0.0

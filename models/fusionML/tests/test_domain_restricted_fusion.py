"""Domain-restricted fusion (search.FamilySpec.fusion_domains).

lambda_F is a single global scalar over both domains' fusion terms, so
"fuse only ARC" is expressed by giving BAC an empty edge set -- its
coefficients are then fit completely unfused, and the fusion penalty
reduces to the ARC term alone. Motivated by the Mantel result in
data/Data_investigation_result.md (phylogenetic signal present in ARC,
absent in BAC, which supplies ~90% of genus-level features)."""
import numpy as np
import pytest

from datasets.datapreprocess.cohort import build_joint_cohort
from configs.fusionml import Config, default_config
from datasets.datapreprocess.io_tree import load_tree
from function.pipeline import LoadedInputs, fit_partition, count_active
from models.fusionML.search import FAMILY_SPECS, Candidate


def _inputs(small_synth, prefixed_counts, tmp_path):
    a, b = prefixed_counts
    meta = small_synth.metadata[["sample_id", "site_id", "target"]].copy()
    meta["target"] = (meta["target"] > meta["target"].median()).astype(float)
    (tmp_path / "arc.nwk").write_text(small_synth.tree_arc_newick)
    (tmp_path / "bac.nwk").write_text(small_synth.tree_bac_newick)
    cohort_ids, _ = build_joint_cohort(a, b, meta, require_target=True)
    return LoadedInputs(
        counts_arc_asv=a, counts_bac_asv=b, counts_arc_genus=None, counts_bac_genus=None,
        genus_crosswalk_arc=None, genus_crosswalk_bac=None, metadata=meta,
        tree_arc=load_tree(str(tmp_path / "arc.nwk"), "ARC"),
        tree_bac=load_tree(str(tmp_path / "bac.nwk"), "BAC"),
        taxonomy_arc=None, taxonomy_bac=None,
        cohort_ids=cohort_ids, cohort_report=None, reports={},
    ), cohort_ids


def _cfg():
    raw = default_config()
    raw["task"] = "classification"
    raw["preprocessing"].update({"min_detected_samples": 3, "min_detected_sites": 2,
                                  "min_sample_prevalence": 0.0, "min_site_prevalence": 0.0,
                                  "detection_count": 1})
    return Config(raw)


def test_fusion_domains_restricts_edges_to_the_named_domain(small_synth, prefixed_counts, tmp_path):
    inputs, cohort_ids = _inputs(small_synth, prefixed_counts, tmp_path)
    cfg = _cfg()
    train = cohort_ids[:-6]
    cand = Candidate(lambda_s=0.0, lambda_r=0.1, lambda_f=1.0, alpha_arc=0.5, alpha_bac=0.5)

    both = fit_partition(train, FAMILY_SPECS["domain_adaptive_hybrid"], cand, inputs, "asv", cfg, seed=1)
    arc = fit_partition(train, FAMILY_SPECS["fusion_arc_only"], cand, inputs, "asv", cfg, seed=1)
    bac = fit_partition(train, FAMILY_SPECS["fusion_bac_only"], cand, inputs, "asv", cfg, seed=1)

    assert len(both.edges_a_idx) > 0 and len(both.edges_b_idx) > 0
    # the restricted families keep their own domain's edges and drop the other's
    assert len(arc.edges_a_idx) == len(both.edges_a_idx)
    assert len(arc.edges_b_idx) == 0
    assert len(bac.edges_b_idx) == len(both.edges_b_idx)
    assert len(bac.edges_a_idx) == 0

    # the unfused domain must not be compressed: with no edges, extract_groups
    # leaves every one of its features as its own singleton group.
    assert count_active(arc, 1e-6)["n_active_groups"] >= count_active(both, 1e-6)["n_active_groups"]
    assert count_active(bac, 1e-6)["n_active_groups"] >= count_active(both, 1e-6)["n_active_groups"]


def test_restricting_fusion_still_satisfies_both_zero_sum_constraints(small_synth, prefixed_counts, tmp_path):
    """The unfused domain still gets its own zero-sum constraint -- dropping
    its fusion edges must not drop its compositional constraint."""
    inputs, cohort_ids = _inputs(small_synth, prefixed_counts, tmp_path)
    cfg = _cfg()
    train = cohort_ids[:-6]
    cand = Candidate(lambda_s=0.0, lambda_r=0.1, lambda_f=1.0, alpha_arc=0.5, alpha_bac=0.5)
    pf = fit_partition(train, FAMILY_SPECS["fusion_arc_only"], cand, inputs, "asv", cfg, seed=1)
    assert abs(float(np.sum(pf.fit_result.beta_a))) <= 1e-6
    assert abs(float(np.sum(pf.fit_result.beta_b))) <= 1e-6

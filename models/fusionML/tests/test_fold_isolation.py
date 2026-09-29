"""Item 18.3: fold isolation. No site crosses train/held-out boundaries;
perturbing held-out counts/labels leaves training feature selection, graph
weights, target scaling and fitted coefficients unchanged for a fixed
training partition."""
import copy

import numpy as np
import pandas as pd
import pytest

from configs.fusionml import Config, default_config
from datasets.datapreprocess.cohort import build_joint_cohort
from function.folds import make_site_grouped_folds, sample_ids_for_sites
from function.pipeline import LoadedInputs, fit_partition
from models.fusionML.search import FAMILY_SPECS, Candidate


def _make_inputs(small_synth, prefixed_counts):
    a, b = prefixed_counts
    meta = small_synth.metadata.rename(columns={"sample_id": "sample_id"})
    meta = meta[["sample_id", "site_id", "target"]]
    cohort_ids, cohort_report = build_joint_cohort(a, b, meta, require_target=True)
    return LoadedInputs(
        counts_arc_asv=a, counts_bac_asv=b, counts_arc_genus=None, counts_bac_genus=None,
        genus_crosswalk_arc=None, genus_crosswalk_bac=None, metadata=meta,
        tree_arc=None, tree_bac=None, taxonomy_arc=None, taxonomy_bac=None,
        cohort_ids=cohort_ids, cohort_report=cohort_report, reports={},
    ), meta


def _cfg():
    raw = default_config()
    raw["preprocessing"]["min_detected_samples"] = 3
    raw["preprocessing"]["min_detected_sites"] = 2
    raw["preprocessing"]["min_sample_prevalence"] = 0.0
    raw["preprocessing"]["min_site_prevalence"] = 0.0
    return Config(raw)


def test_no_site_crosses_outer_train_test(small_synth, prefixed_counts):
    inputs, meta = _make_inputs(small_synth, prefixed_counts)
    site_map = {sid: meta.loc[sid, "site_id"] for sid in inputs.cohort_ids}
    folds = make_site_grouped_folds(site_map, k=4, seed=7)
    for o in range(folds.k):
        test_sites = set(folds.sites_in_fold(o))
        train_sites = set(folds.train_sites_for_outer_fold(o))
        assert test_sites.isdisjoint(train_sites)


def test_perturbing_heldout_data_does_not_change_training_fit(small_synth, prefixed_counts):
    inputs, meta = _make_inputs(small_synth, prefixed_counts)
    site_map = {sid: meta.loc[sid, "site_id"] for sid in inputs.cohort_ids}
    folds = make_site_grouped_folds(site_map, k=4, seed=7)
    train_sites = folds.train_sites_for_outer_fold(0)
    test_sites = folds.sites_in_fold(0)
    train_ids = sample_ids_for_sites(site_map, train_sites)
    test_ids = sample_ids_for_sites(site_map, test_sites)
    assert set(train_ids).isdisjoint(test_ids)

    cfg = _cfg()
    spec = FAMILY_SPECS["asv_ridge"]
    cand = Candidate(lambda_s=0.0, lambda_r=0.01, lambda_f=0.0, alpha_arc=None, alpha_bac=None)

    pf1 = fit_partition(train_ids, spec, cand, inputs, "asv", cfg, seed=42)

    perturbed = copy.deepcopy(inputs)
    perturbed.counts_arc_asv = perturbed.counts_arc_asv.copy()
    perturbed.counts_bac_asv = perturbed.counts_bac_asv.copy()
    perturbed.metadata = perturbed.metadata.copy()
    perturbed.counts_arc_asv.loc[test_ids, :] = perturbed.counts_arc_asv.loc[test_ids, :] * 0 + 999999
    perturbed.counts_bac_asv.loc[test_ids, :] = perturbed.counts_bac_asv.loc[test_ids, :] * 0 + 999999
    perturbed.metadata.loc[test_ids, "target"] = -99999.0

    pf2 = fit_partition(train_ids, spec, cand, perturbed, "asv", cfg, seed=42)

    assert pf1.retained_a == pf2.retained_a
    assert pf1.retained_b == pf2.retained_b
    assert pf1.target_scaler.mu_ == pytest.approx(pf2.target_scaler.mu_)
    assert pf1.target_scaler.sigma_ == pytest.approx(pf2.target_scaler.sigma_)
    np.testing.assert_allclose(pf1.fit_result.beta_a, pf2.fit_result.beta_a, atol=1e-8)
    np.testing.assert_allclose(pf1.fit_result.beta_b, pf2.fit_result.beta_b, atol=1e-8)
    assert pf1.fit_result.b0 == pytest.approx(pf2.fit_result.b0, abs=1e-8)

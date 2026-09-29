"""ARC-BAC cross-domain predictive interaction (ad-microbial-model-design-
arc-bac-interactions.md items 8, 10, 12-15). Covers acceptance-criteria
tests 12 (invariance), 14 (on/off equivalence), 15 (prediction algebra),
17 (classification gate) and 18 (synthetic demonstration, incl. the
zero-product case)."""
import math

import numpy as np
import pandas as pd
import pytest

from configs.fusionml import Config, default_config
from datasets.datapreprocess.cohort import build_joint_cohort
from datasets.datapreprocess.io_tree import load_tree
from models.fusionML.interactions import (
    BalanceCandidate, compute_loading, enumerate_tree_splits, fit_interaction_transform,
    InteractionUnavailableError,
)
from datasets.datapreprocess.preprocessing import site_balanced_weights, SingleClassTrainingError
from models.fusionML.estimator import fit_log_contrast, fit_log_contrast_logistic, fit_log_contrast_with_interaction
from function.pipeline import LoadedInputs, fit_partition, predict_with_partition_fit
from models.fusionML.search import FAMILY_SPECS, Candidate, InteractionCandidate
from function.model_io import FusionModel, save_model, load_model
from datasets.datapreprocess.synthetic import generate_synthetic_dataset, write_synthetic_dataset_to_dir
from function import pipeline as pipeline_mod

SOLVER_CFG = {"preferred": "CLARABEL", "fallback": "OSQP", "classification_fallback": "SCS",
              "equality_tolerance": 1e-6, "max_retries": 3}


# ---------------------------------------------------------------------------
# Item 12: interaction invariance
# ---------------------------------------------------------------------------

def test_balance_loading_sums_to_zero_and_score_is_rescale_invariant():
    for r, s in [(1, 1), (2, 3), (1, 4), (3, 2)]:
        cand = BalanceCandidate(domain="ARC", positive_members=tuple(f"p{i}" for i in range(r)),
                                 negative_members=tuple(f"n{i}" for i in range(s)))
        feature_order = list(cand.positive_members) + list(cand.negative_members)
        loading = compute_loading(cand, feature_order)
        assert loading.sum() == pytest.approx(0.0, abs=1e-12)

        rng = np.random.default_rng(r * 10 + s)
        tilde_c = rng.uniform(1.0, 50.0, size=len(feature_order))  # already zero-handled positive vector
        q = tilde_c / tilde_c.sum()
        score = float(np.log(q) @ loading)
        for k in (0.001, 1.0, 7.3, 1000.0):
            q_k = (k * tilde_c) / (k * tilde_c).sum()
            score_k = float(np.log(q_k) @ loading)
            assert score_k == pytest.approx(score, abs=1e-10)


def test_tree_splits_are_deduplicated_and_deterministically_oriented(small_synth, tmp_path):
    tree_path = tmp_path / "arc.nwk"
    tree_path.write_text(small_synth.tree_arc_newick)
    tree = load_tree(str(tree_path), "ARC")
    retained = [f"ARC::{f}" for f in small_synth.feature_ids_a]

    candidates, report = enumerate_tree_splits(tree, "ARC", retained)
    assert report["n_mapped"] == len(retained)
    assert len(candidates) > 0
    # deterministic orientation: positive side is always lexicographically smaller
    for c in candidates:
        assert sorted(c.positive_members) < sorted(c.negative_members)
    # no duplicate canonical splits
    ids = [c.canonical_id for c in candidates]
    assert len(ids) == len(set(ids))
    # rerunning is exactly reproducible
    candidates2, _ = enumerate_tree_splits(tree, "ARC", retained)
    assert ids == [c.canonical_id for c in candidates2]


def _log_composition(counts: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    tilde_c = counts.astype(float) + pseudocount
    return np.log(tilde_c)


def _interaction_cfg(overrides=None):
    cfg = {
        "max_balances_arc": 2, "max_balances_bac": 2, "max_products": 4,
        "min_joint_group_samples": 5, "min_joint_group_sites": 3,
        "max_abs_balance_correlation": 0.95,
    }
    cfg.update(overrides or {})
    return cfg


def test_interaction_transform_invariant_to_per_sample_domain_rescale(small_synth, prefixed_counts, tmp_path):
    """Item 12: independent sample-specific rescaling of positive ARC (or
    BAC) inputs after zero handling leaves balances, products and a fitted
    eta unchanged. Rescaling tilde_c (the already-pseudocounted positive
    vector) by a per-sample factor k_i is exactly an additive shift of
    log(tilde_c) by log(k_i), uniform across that domain's own retained
    columns -- which a zero-sum loading (or zero-sum beta) annihilates."""
    a, b = prefixed_counts
    arc_tree_path = tmp_path / "arc.nwk"
    arc_tree_path.write_text(small_synth.tree_arc_newick)
    bac_tree_path = tmp_path / "bac.nwk"
    bac_tree_path.write_text(small_synth.tree_bac_newick)
    tree_a = load_tree(str(arc_tree_path), "ARC")
    tree_b = load_tree(str(bac_tree_path), "BAC")

    retained_a = [f"ARC::{f}" for f in small_synth.feature_ids_a]
    retained_b = [f"BAC::{f}" for f in small_synth.feature_ids_b]
    log_a = _log_composition(a[retained_a])
    log_b = _log_composition(b[retained_b])
    det_a = a[retained_a] >= 1
    det_b = b[retained_b] >= 1
    site_of_sample = small_synth.site_of_sample
    weights = site_balanced_weights([site_of_sample[s] for s in a.index])

    cfg = _interaction_cfg()
    transform, report = fit_interaction_transform(
        log_a, log_b, det_a, det_b, site_of_sample, weights, tree_a, tree_b, retained_a, retained_b, cfg,
    )
    assert transform.n_products > 0, "fixture must produce >=1 product for this test to be meaningful"

    rng = np.random.default_rng(42)
    log_k = rng.normal(scale=2.0, size=len(a.index))  # arbitrary per-sample log-rescale factor
    log_a_shifted = log_a.add(pd.Series(log_k, index=log_a.index), axis=0)

    transform_shifted, report_shifted = fit_interaction_transform(
        log_a_shifted, log_b, det_a, det_b, site_of_sample, weights, tree_a, tree_b, retained_a, retained_b, cfg,
    )
    # fresh training-only refit on rescaled data selects the identical balances
    assert [x.canonical_id for x in transform_shifted.balances_a] == [x.canonical_id for x in transform.balances_a]
    for orig, shifted in zip(transform.balances_a, transform_shifted.balances_a):
        assert orig.mean == pytest.approx(shifted.mean, abs=1e-8)
        assert orig.sd == pytest.approx(shifted.sd, abs=1e-8)

    # the ORIGINAL (frozen) transform applied to rescaled inputs reproduces
    # identical product columns (this is the prediction-time invariance).
    x_i_orig = transform.transform(log_a, log_b)
    x_i_shifted = transform.transform(log_a_shifted, log_b)
    np.testing.assert_allclose(x_i_shifted, x_i_orig, atol=1e-8)

    # full eta (main effects + interaction) is unchanged for any zero-sum beta_a
    rng2 = np.random.default_rng(7)
    beta_a = rng2.normal(size=len(retained_a)); beta_a -= beta_a.mean()
    beta_b = rng2.normal(size=len(retained_b)); beta_b -= beta_b.mean()
    gamma = rng2.normal(size=transform.n_products)
    eta_orig = 0.3 + log_a.to_numpy() @ beta_a + log_b.to_numpy() @ beta_b + x_i_orig @ gamma
    eta_shifted = 0.3 + log_a_shifted.to_numpy() @ beta_a + log_b.to_numpy() @ beta_b + x_i_shifted @ gamma
    np.testing.assert_allclose(eta_shifted, eta_orig, atol=1e-8)


# ---------------------------------------------------------------------------
# Item 14: on/off equivalence
# ---------------------------------------------------------------------------

def _toy(seed=11, n=60, pa=4, pb=4, n_i=2):
    rng = np.random.default_rng(seed)
    La = rng.normal(size=(n, pa)); La -= La.mean(axis=1, keepdims=True)
    Lb = rng.normal(size=(n, pb)); Lb -= Lb.mean(axis=1, keepdims=True)
    Xi = rng.normal(size=(n, n_i))
    Xi = (Xi - Xi.mean(axis=0)) / Xi.std(axis=0)
    w = np.full(n, 1.0 / n)
    return La, Lb, Xi, w, rng


def test_on_off_equivalence_regression():
    La, Lb, Xi, w, rng = _toy()
    y = 0.5 + 1.0 * La[:, 0] - 1.0 * La[:, 1] + 0.6 * Lb[:, 0] + 0.05 * rng.normal(size=La.shape[0])
    y_star = (y - y.mean()) / y.std()

    res_off = fit_log_contrast(La, Lb, y_star, w, lambda_s=0.0, lambda_r=0.1, lambda_f=0.0,
                                edges_a=[], edges_b=[], solver_cfg=SOLVER_CFG)
    res_on_heavy = fit_log_contrast_with_interaction(
        La, Lb, Xi, y_star, w, lambda_s=0.0, lambda_r=0.1, lambda_f=0.0, lambda_i1=0.0, lambda_i2=1.0e6,
        edges_a=[], edges_b=[], task="regression", solver_cfg=SOLVER_CFG,
    )
    np.testing.assert_allclose(res_on_heavy.beta_a, res_off.beta_a, atol=5e-3)
    np.testing.assert_allclose(res_on_heavy.beta_b, res_off.beta_b, atol=5e-3)
    np.testing.assert_allclose(res_on_heavy.gamma, 0.0, atol=5e-3)
    assert res_on_heavy.beta_a.shape == res_off.beta_a.shape  # no extra columns in the taxon design


def test_on_off_equivalence_classification():
    La, Lb, Xi, w, rng = _toy(seed=23)
    eta_true = 0.2 + 1.2 * La[:, 0] - 1.2 * La[:, 1]
    prob = 1.0 / (1.0 + np.exp(-eta_true))
    y = (rng.uniform(size=La.shape[0]) < prob).astype(float)

    res_off = fit_log_contrast_logistic(La, Lb, y, w, lambda_s=0.0, lambda_r=0.1, lambda_f=0.0,
                                         edges_a=[], edges_b=[], solver_cfg=SOLVER_CFG)
    res_on_heavy = fit_log_contrast_with_interaction(
        La, Lb, Xi, y, w, lambda_s=0.0, lambda_r=0.1, lambda_f=0.0, lambda_i1=0.0, lambda_i2=1.0e6,
        edges_a=[], edges_b=[], task="classification", solver_cfg=SOLVER_CFG,
    )
    np.testing.assert_allclose(res_on_heavy.beta_a, res_off.beta_a, atol=5e-2)
    np.testing.assert_allclose(res_on_heavy.beta_b, res_off.beta_b, atol=5e-2)
    np.testing.assert_allclose(res_on_heavy.gamma, 0.0, atol=5e-2)


def test_empty_product_set_is_unavailable_as_on_candidate():
    La, Lb, _, w, rng = _toy(seed=3)
    y_star = rng.normal(size=La.shape[0])
    empty_xi = np.zeros((La.shape[0], 0))
    with pytest.raises(ValueError):
        fit_log_contrast_with_interaction(La, Lb, empty_xi, y_star, w, 0.0, 0.1, 0.0, 0.0, 1.0,
                                           [], [], "regression", SOLVER_CFG)


# ---------------------------------------------------------------------------
# Item 15/17: full prediction algebra + classification gates, via the real
# pipeline (fit_partition/predict_with_partition_fit), reusing the
# small_synth fixture with a real ARC/BAC tree pair.
# ---------------------------------------------------------------------------

def _cfg_with_interaction(task="classification"):
    raw = default_config()
    raw["task"] = task
    raw["preprocessing"]["min_detected_samples"] = 3
    raw["preprocessing"]["min_detected_sites"] = 2
    raw["preprocessing"]["min_sample_prevalence"] = 0.0
    raw["preprocessing"]["min_site_prevalence"] = 0.0
    raw["preprocessing"]["detection_count"] = 1
    raw["optional"]["cross_domain_interactions"] = True
    raw["interaction"]["enabled"] = True
    raw["interaction"]["group_detection_count"] = 1
    raw["interaction"]["min_joint_group_samples"] = 3
    raw["interaction"]["min_joint_group_sites"] = 2
    return Config(raw)


def _build_interaction_inputs(small_synth, prefixed_counts, tmp_path, task="classification"):
    a, b = prefixed_counts
    meta = small_synth.metadata[["sample_id", "site_id", "target"]].copy()
    if task == "classification":
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
    return inputs, cohort_ids


def test_full_prediction_algebra_includes_interaction_term(small_synth, prefixed_counts, tmp_path):
    inputs, cohort_ids = _build_interaction_inputs(small_synth, prefixed_counts, tmp_path, task="classification")
    cfg = _cfg_with_interaction("classification")
    spec = FAMILY_SPECS["domain_adaptive_interaction"]
    base = Candidate(lambda_s=0.0, lambda_r=0.1, lambda_f=0.0, alpha_arc=None, alpha_bac=None)
    cand = InteractionCandidate(base=base, interaction_on=True, lambda_i1=0.0, lambda_i2=0.5)

    train_ids = cohort_ids[:-6]
    eval_ids = cohort_ids[-6:]
    pf = fit_partition(train_ids, spec, cand, inputs, "asv", cfg, seed=1)
    assert pf.interaction_transform is not None

    preds = predict_with_partition_fit(pf, eval_ids, inputs, "asv")
    tf = pf.jp.transform(inputs.counts_arc_asv.loc[eval_ids], inputs.counts_bac_asv.loc[eval_ids], pf.pseudocount)
    log_a, log_b = tf["log_a"], tf["log_b"]
    x_i = pf.interaction_transform.transform(log_a, log_b)
    manual_eta = pf.fit_result.b0 + log_a.to_numpy() @ pf.fit_result.beta_a + log_b.to_numpy() @ pf.fit_result.beta_b
    if pf.fit_result.gamma.size > 0:
        manual_eta = manual_eta + x_i @ pf.fit_result.gamma
    manual_prob = pf.target_scaler.inverse(manual_eta)
    np.testing.assert_allclose(preds.loc[eval_ids, "prediction"].to_numpy(), manual_prob, atol=1e-8)


def test_single_class_training_gate_for_interaction_family(small_synth, prefixed_counts, tmp_path):
    inputs, cohort_ids = _build_interaction_inputs(small_synth, prefixed_counts, tmp_path, task="classification")
    inputs.metadata.loc[:, "target"] = 0.0  # single-class training partition
    cfg = _cfg_with_interaction("classification")
    spec = FAMILY_SPECS["domain_adaptive_interaction"]
    base = Candidate(lambda_s=0.0, lambda_r=0.1, lambda_f=0.0, alpha_arc=None, alpha_bac=None)
    cand = InteractionCandidate(base=base, interaction_on=False)

    with pytest.raises(SingleClassTrainingError):
        fit_partition(cohort_ids, spec, cand, inputs, "asv", cfg, seed=1)

    # intercept_only remains fittable with a clipped constant probability
    intercept_spec = FAMILY_SPECS["intercept_only"]
    pf = fit_partition(cohort_ids, intercept_spec, Candidate(0.0, 0.0, 0.0, None, None), inputs, "asv", cfg, seed=1)
    assert pf.fit_result.intercept_only


# ---------------------------------------------------------------------------
# Item 18: synthetic interaction demonstration (nested run), incl. the
# zero-product data-generating case.
# ---------------------------------------------------------------------------

def _synthetic_interaction_config(tmp_path, task, interaction_overrides=None):
    ds = generate_synthetic_dataset(n_sites=10, visits_per_site=4, p_arc=6, p_bac=7, seed=888,
                                     depth_arc=4000, depth_bac=6000, missing_visit_prob=0.1, task=task)
    data_dir = str(tmp_path / "data")
    paths = write_synthetic_dataset_to_dir(ds, data_dir)
    out_dir = str(tmp_path / "results")

    raw = default_config()
    raw.update({
        "seed": 777, "task": task, "resolution": "asv",
        "inputs": {**raw["inputs"], "arc_counts": paths["arc_counts"], "bac_counts": paths["bac_counts"],
                   "metadata": paths["metadata"], "arc_tree": paths["arc_tree"], "bac_tree": paths["bac_tree"],
                   "arc_taxonomy": paths["arc_taxonomy"], "bac_taxonomy": paths["bac_taxonomy"]},
        "columns": {"sample_id": "sample_id", "site_id": "site_id",
                    "target": "methane_yield_SYNTHETIC", "covariates": []},
        "output_dir": out_dir, "run_id": "synthetic_interaction_test", "smoke": True,
    })
    raw["preprocessing"].update({"detection_count": 1, "min_detected_samples": 3, "min_sample_prevalence": 0.0,
                                  "min_detected_sites": 2, "min_site_prevalence": 0.0})
    raw["cv"].update({"outer_folds": 3, "inner_folds": 2, "required_validation_coverage": 0.5,
                       "families": ["intercept_only", "domain_adaptive_hybrid", "domain_adaptive_interaction"]})
    raw["search"].update({"candidate_budget_per_family": 6, "lambda_s": [0, 0.1], "lambda_r": [0.01, 0.5],
                           "lambda_f": [0, 5], "alpha_arc": [0, 1], "alpha_bac": [0, 1]})
    raw["optional"]["cross_domain_interactions"] = True
    raw["interaction"]["enabled"] = True
    raw["interaction"]["group_detection_count"] = 1
    raw["interaction"]["min_joint_group_samples"] = 3
    raw["interaction"]["min_joint_group_sites"] = 2
    raw["interaction"]["lambda_i_l1"] = [0.0, 0.1]
    raw["interaction"]["lambda_i_l2"] = [0.1, 1.0]
    raw["interaction"].update(interaction_overrides or {})

    cfg_path = tmp_path / "synthetic_interaction_config.yaml"
    import yaml
    with open(cfg_path, "w") as f:
        yaml.safe_dump(raw, f)
    return str(cfg_path), paths


def test_synthetic_interaction_nested_run_classification(tmp_path):
    config_path, paths = _synthetic_interaction_config(tmp_path, "classification")
    result = pipeline_mod.run_train(config_path)
    results_df = result["results"]
    assert "domain_adaptive_interaction" in results_df["family"].tolist()

    final_info = result["final_model_info"]
    assert final_info["status"] == "ok"
    model = load_model(final_info["model_path"])
    assert model.task == "classification"
    assert model.classification_threshold is not None

    a = pd.read_csv(paths["arc_counts"], index_col=0)
    a.columns = [f"ARC::{c}" for c in a.columns]
    b = pd.read_csv(paths["bac_counts"], index_col=0)
    b.columns = [f"BAC::{c}" for c in b.columns]
    preds = model.predict(a, b)
    assert preds["prediction"].dropna().between(0, 1).all()

    # save/reload equality, including the interaction term if the final model has one
    reloaded = load_model(final_info["model_path"])
    preds2 = reloaded.predict(a, b)
    np.testing.assert_allclose(preds["prediction"].to_numpy(), preds2["prediction"].to_numpy())


def test_synthetic_interaction_zero_product_case_still_fits_off_candidates(tmp_path):
    """Item 18's zero-product data-generating case: an impossibly strict
    group-detection gate leaves no eligible balance/product anywhere, so
    every interaction-ON candidate is unavailable, but interaction-OFF
    candidates (and thus the family overall) remain fittable."""
    config_path, _ = _synthetic_interaction_config(
        tmp_path, "classification",
        interaction_overrides={"min_joint_group_samples": 10_000, "min_joint_group_sites": 10_000},
    )
    result = pipeline_mod.run_train(config_path)
    results_df = result["results"]
    assert "domain_adaptive_interaction" in results_df["family"].tolist()
    row = results_df[results_df["family"] == "domain_adaptive_interaction"].iloc[0]
    assert row["coverage"] > 0

    final_info = result["final_model_info"]
    if final_info["status"] == "ok" and final_info["family"] == "domain_adaptive_interaction":
        assert final_info["selected_candidate"]["interaction_on"] is False

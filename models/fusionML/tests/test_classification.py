"""Classification extension (design-spec item 7: 'substitute logistic
loss and report appropriate discrimination and threshold metrics...
Select the decision threshold using training data only')."""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from datasets.datapreprocess.synthetic import generate_synthetic_dataset, write_synthetic_dataset_to_dir
from function.evaluate import select_threshold_on_training, compute_classification_metrics
from function import pipeline


def test_threshold_selected_on_training_data_only():
    rng = np.random.default_rng(0)
    y = np.array([0.0] * 15 + [1.0] * 5)
    prob = np.clip(y * 0.7 + rng.normal(0, 0.15, size=20) + 0.15, 0, 1)
    thr = select_threshold_on_training(y, prob)
    pred = (prob >= thr).astype(float)
    from sklearn.metrics import matthews_corrcoef
    baseline_mcc = matthews_corrcoef(y, (prob >= 0.5).astype(float))
    selected_mcc = matthews_corrcoef(y, pred)
    assert selected_mcc >= baseline_mcc - 1e-9


def test_threshold_selection_degenerate_single_class_falls_back_to_half():
    y = np.zeros(10)
    prob = np.random.default_rng(1).random(10)
    assert select_threshold_on_training(y, prob) == 0.5


def test_classification_metrics_handle_single_class_pooled_auc_as_nan():
    y = np.zeros(10)
    prob = np.full(10, 0.3)
    pred_class = np.zeros(10)
    m = compute_classification_metrics(y, prob, [f"s{i}" for i in range(10)], pred_class)
    assert np.isnan(m.pooled_auc_roc)
    assert np.isnan(m.pooled_auc_pr)
    assert not np.isnan(m.pooled_brier)


def _write_classification_config(tmp_path, paths, out_dir):
    cfg = {
        "seed": 777, "task": "classification", "resolution": "asv",
        "inputs": {
            "arc_counts": paths["arc_counts"], "bac_counts": paths["bac_counts"],
            "metadata": paths["metadata"], "arc_tree": paths["arc_tree"], "bac_tree": paths["bac_tree"],
            "arc_taxonomy": paths["arc_taxonomy"], "bac_taxonomy": paths["bac_taxonomy"],
            "orientation": "samples_by_features", "exploratory_block_list": None,
        },
        "columns": {"sample_id": "sample_id", "site_id": "site_id",
                    "target": "methane_yield_SYNTHETIC", "covariates": []},
        "preprocessing": {
            "detection_count": 1, "min_detected_samples": 3, "min_sample_prevalence": 0.0,
            "min_detected_sites": 2, "min_site_prevalence": 0.0, "pseudocount": 0.5,
            "sensitivity_pseudocounts": [], "standardize_target": True,
            "standardize_microbial_columns": False, "relative_input_mode": False,
            "missing_retained_feature_means_zero": False,
        },
        "graph": {"k_phylogeny": 3, "k_abundance": 3, "min_joint_samples": 4, "min_joint_sites": 2,
                  "empirical_score": "max_all_and_joint_lrv", "bootstrap_sites": 10, "support_threshold": 0.5,
                  "approximate_bootstrap": False, "approximate_candidate_k": 20},
        "cv": {"outer_folds": 3, "inner_folds": 2, "selection_metric": "site_balanced_mse",
               "selection_rule": "one_standard_error", "required_validation_coverage": 0.5,
               "families": ["intercept_only", "asv_ridge", "abundance_only", "domain_adaptive_hybrid"]},
        "search": {"candidate_budget_per_family": 6, "lambda_s": [0, 0.1], "lambda_r": [0.01, 0.5],
                   "lambda_f": [0, 5], "alpha_arc": [0, 1], "alpha_bac": [0, 1], "exhaustive": False},
        "solver": {"preferred": "CLARABEL", "fallback": "OSQP", "equality_tolerance": 1e-6,
                   "max_retries": 3, "time_limit_seconds": 60},
        "compression": {"coefficient_tolerance": 1e-6, "keep_residual_coefficients": True},
        "optional": {"b12_comparator": False, "cross_domain_interactions": False},
        "output_dir": out_dir, "run_id": "synthetic_classification_test", "smoke": True,
    }
    cfg_path = tmp_path / "synthetic_classification_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(cfg_path)


def test_synthetic_classification_nested_run(tmp_path):
    ds = generate_synthetic_dataset(n_sites=10, visits_per_site=4, p_arc=6, p_bac=7, seed=888,
                                     depth_arc=4000, depth_bac=6000, missing_visit_prob=0.1,
                                     task="classification")
    n_pos = sum(ds.metadata["target"])
    assert 0 < n_pos < len(ds.metadata)  # sanity: not a degenerate all-one-class draw

    data_dir = str(tmp_path / "data")
    paths = write_synthetic_dataset_to_dir(ds, data_dir)
    out_dir = str(tmp_path / "results")
    config_path = _write_classification_config(tmp_path, paths, out_dir)

    result = pipeline.run_train(config_path)
    results_df = result["results"]

    assert "intercept_only" in results_df["family"].tolist()
    for col in ("pooled_auc_roc", "pooled_brier", "site_balanced_log_loss", "mcc", "f2"):
        assert col in results_df.columns
    for _, row in results_df.iterrows():
        assert row["coverage"] > 0
        assert 0 <= row["n_positive"] <= row["n"]

    oof = pd.read_csv(os.path.join(out_dir, "smoke", "oof_predictions.csv"))
    assert "predicted_class" in oof.columns
    assert oof["prediction"].dropna().between(0, 1).all()  # predictions are probabilities

    final_info = result["final_model_info"]
    assert final_info["status"] == "ok"
    from function.model_io import load_model
    model = load_model(final_info["model_path"])
    assert model.task == "classification"
    assert model.classification_threshold is not None

    a = pd.read_csv(paths["arc_counts"], index_col=0)
    a.columns = [f"ARC::{c}" for c in a.columns]
    b = pd.read_csv(paths["bac_counts"], index_col=0)
    b.columns = [f"BAC::{c}" for c in b.columns]
    preds = model.predict(a, b)
    assert preds["prediction"].dropna().between(0, 1).all()
    assert set(preds["predicted_class"].dropna().unique()) <= {0.0, 1.0}

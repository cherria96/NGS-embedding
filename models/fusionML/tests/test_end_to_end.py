"""Item 18.10: end-to-end demonstration. Generates positive compositions for
two separate domains with known zero-sum coefficients and repeated-site
variation, converts them to simulated count assays, and executes a small
nested grouped run through the real CLI pipeline (`--mode train`).

This tests IMPLEMENTATION correctness only -- it is not evidence for the
scientific hypothesis on real AD data (design-spec item 18)."""
import os

import yaml

from datasets.datapreprocess.synthetic import generate_synthetic_dataset, write_synthetic_dataset_to_dir
from function import pipeline


def _write_config(tmp_path, paths, out_dir):
    cfg = {
        "seed": 999,
        "task": "regression",
        "resolution": "asv",
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
        "output_dir": out_dir, "run_id": "synthetic_e2e_test", "smoke": True,
    }
    cfg_path = tmp_path / "synthetic_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(cfg_path)


def test_synthetic_nested_grouped_run(tmp_path):
    ds = generate_synthetic_dataset(n_sites=9, visits_per_site=4, p_arc=6, p_bac=7, seed=555,
                                     depth_arc=4000, depth_bac=6000, missing_visit_prob=0.1)
    data_dir = str(tmp_path / "data")
    paths = write_synthetic_dataset_to_dir(ds, data_dir)
    out_dir = str(tmp_path / "results")
    config_path = _write_config(tmp_path, paths, out_dir)

    result = pipeline.run_train(config_path)
    results_df = result["results"]

    assert os.path.exists(os.path.join(out_dir, "smoke", "results.csv"))
    assert os.path.exists(os.path.join(out_dir, "smoke", "oof_predictions.csv"))
    assert os.path.exists(os.path.join(out_dir, "smoke", "run_manifest.json"))
    assert len(results_df) >= 2
    assert "intercept_only" in results_df["family"].tolist()
    for _, row in results_df.iterrows():
        assert row["coverage"] > 0

    final_info = result["final_model_info"]
    assert final_info["status"] == "ok"
    assert os.path.exists(final_info["model_path"])

    from function.model_io import load_model
    model = load_model(final_info["model_path"])
    import pandas as pd
    a = pd.read_csv(paths["arc_counts"], index_col=0)
    a.columns = [f"ARC::{c}" for c in a.columns]
    b = pd.read_csv(paths["bac_counts"], index_col=0)
    b.columns = [f"BAC::{c}" for c in b.columns]
    sample_ids = a.index[:5]
    preds = model.predict(a.loc[sample_ids], b.loc[sample_ids])
    assert len(preds) == 5

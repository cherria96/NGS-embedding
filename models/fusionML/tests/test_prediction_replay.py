"""Item 18.8: prediction replay. Saved and reloaded models give matching
predictions despite input row/column reordering when IDs are preserved; new
samples/features do not alter the trained feature mapping."""
import numpy as np
import pandas as pd

from configs.fusionml import Config, default_config
from datasets.datapreprocess.cohort import build_joint_cohort
from function.pipeline import LoadedInputs, fit_partition, predict_with_partition_fit
from models.fusionML.search import FAMILY_SPECS, Candidate
from function.model_io import FusionModel, save_model, load_model


def _cfg():
    raw = default_config()
    raw["preprocessing"]["min_detected_samples"] = 3
    raw["preprocessing"]["min_detected_sites"] = 2
    raw["preprocessing"]["min_sample_prevalence"] = 0.0
    raw["preprocessing"]["min_site_prevalence"] = 0.0
    return Config(raw)


def _build_model(small_synth, prefixed_counts):
    a, b = prefixed_counts
    meta = small_synth.metadata[["sample_id", "site_id", "target"]]
    cohort_ids, _ = build_joint_cohort(a, b, meta, require_target=True)
    inputs = LoadedInputs(
        counts_arc_asv=a, counts_bac_asv=b, counts_arc_genus=None, counts_bac_genus=None,
        genus_crosswalk_arc=None, genus_crosswalk_bac=None, metadata=meta,
        tree_arc=None, tree_bac=None, taxonomy_arc=None, taxonomy_bac=None,
        cohort_ids=cohort_ids, cohort_report=None, reports={},
    )
    cfg = _cfg()
    spec = FAMILY_SPECS["asv_elastic_net"]
    cand = Candidate(lambda_s=0.01, lambda_r=0.05, lambda_f=0.0, alpha_arc=None, alpha_bac=None)
    train_ids = cohort_ids[:-6]
    eval_ids = cohort_ids[-6:]
    pf = fit_partition(train_ids, spec, cand, inputs, "asv", cfg, seed=1)
    beta_a_orig, b0_orig = pf.target_scaler.inverse_coefficients(pf.fit_result.beta_a, pf.fit_result.b0)
    beta_b_orig, _ = pf.target_scaler.inverse_coefficients(pf.fit_result.beta_b, pf.fit_result.b0)
    model = FusionModel(
        resolution="asv", domain="joint", feature_ids_a=pf.retained_a, feature_ids_b=pf.retained_b,
        preproc_a=pf.jp.arc, preproc_b=pf.jp.bac, pseudocount=pf.pseudocount, target_scaler=pf.target_scaler,
        covariate_preproc=None, b0_star=pf.fit_result.b0, beta_a_star=pf.fit_result.beta_a, beta_b_star=pf.fit_result.beta_b,
        b0_orig=b0_orig, beta_a_orig=beta_a_orig, beta_b_orig=beta_b_orig,
        graph_edges_a=pf.mixed_edges_a, graph_edges_b=pf.mixed_edges_b, alpha_arc=None, alpha_bac=None,
        lambda_s=cand.lambda_s, lambda_r=cand.lambda_r, lambda_f=cand.lambda_f, family="asv_elastic_net", metadata={},
    )
    return model, inputs, eval_ids


def test_reload_matches_original_predictions(small_synth, prefixed_counts, tmp_path):
    model, inputs, eval_ids = _build_model(small_synth, prefixed_counts)
    model_path = str(tmp_path / "model.pkl")
    save_model(model, model_path)
    reloaded = load_model(model_path)

    pred_orig = model.predict(inputs.counts_arc_asv.loc[eval_ids], inputs.counts_bac_asv.loc[eval_ids])
    pred_reload = reloaded.predict(inputs.counts_arc_asv.loc[eval_ids], inputs.counts_bac_asv.loc[eval_ids])
    np.testing.assert_allclose(pred_orig["prediction"].to_numpy(), pred_reload["prediction"].to_numpy())


def test_prediction_invariant_to_row_and_column_reordering(small_synth, prefixed_counts, tmp_path):
    model, inputs, eval_ids = _build_model(small_synth, prefixed_counts)

    a_eval = inputs.counts_arc_asv.loc[eval_ids]
    b_eval = inputs.counts_bac_asv.loc[eval_ids]
    baseline = model.predict(a_eval, b_eval).sort_index()

    shuffled_rows = list(reversed(eval_ids))
    a_shuf = a_eval.loc[shuffled_rows]
    b_shuf = b_eval.loc[shuffled_rows]
    a_shuf = a_shuf[list(reversed(a_shuf.columns))]  # shuffle columns too
    reordered = model.predict(a_shuf, b_shuf).sort_index()

    np.testing.assert_allclose(baseline["prediction"].to_numpy(), reordered["prediction"].to_numpy())


def test_unseen_feature_column_does_not_alter_trained_mapping(small_synth, prefixed_counts, tmp_path):
    model, inputs, eval_ids = _build_model(small_synth, prefixed_counts)
    a_eval = inputs.counts_arc_asv.loc[eval_ids].copy()
    b_eval = inputs.counts_bac_asv.loc[eval_ids]
    baseline = model.predict(a_eval, b_eval)

    a_with_new_feature = a_eval.copy()
    a_with_new_feature["ARC::brand_new_unseen_asv"] = 12345
    with_extra = model.predict(a_with_new_feature, b_eval)

    np.testing.assert_allclose(baseline["prediction"].to_numpy(), with_extra["prediction"].to_numpy())
    assert list(model.feature_ids_a) == list(model.feature_ids_a)  # mapping object unmodified in place


def test_sample_missing_from_one_domain_is_reported_not_misaligned(small_synth, prefixed_counts, tmp_path):
    """Regression test: a sample present in only one domain's count table
    (real ARC/BAC coverage differs on the real dataset -- 6 metadata
    samples have no ARC assay at all) must be reported as a missing assay,
    not silently misaligned against a same-length-but-different-sample-set
    array from the other domain."""
    model, inputs, eval_ids = _build_model(small_synth, prefixed_counts)
    a_eval = inputs.counts_arc_asv.loc[eval_ids]
    b_eval = inputs.counts_bac_asv.loc[eval_ids]

    baseline = model.predict(a_eval, b_eval).sort_index()

    dropped_sample = eval_ids[0]
    a_missing_one = a_eval.drop(index=dropped_sample)
    result = model.predict(a_missing_one, b_eval).sort_index()

    assert not bool(result.loc[dropped_sample, "scorable"])
    assert "missing ARC assay" in result.loc[dropped_sample, "reason"]
    assert pd.isna(result.loc[dropped_sample, "prediction"])

    remaining = [s for s in eval_ids if s != dropped_sample]
    np.testing.assert_allclose(
        baseline.loc[remaining, "prediction"].to_numpy(),
        result.loc[remaining, "prediction"].to_numpy(),
    )

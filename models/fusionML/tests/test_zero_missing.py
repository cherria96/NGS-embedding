"""Item 18.5: zero and missing-data behavior. Co-absence does not create
eligible empirical edges; missing assays and zero retained-count rows are
reported; true zeros vs. unspecified missing feature columns are handled
differently."""
import numpy as np
import pandas as pd
import pytest

from datasets.datapreprocess.cohort import build_joint_cohort
from datasets.datapreprocess.preprocessing import DomainPreprocessor, JointPreprocessor, SchemaError
from datasets.graph.fusion_graphs import _joint_eligibility_and_vjoint


def test_co_absence_does_not_create_eligible_edge():
    n = 30
    # two taxa that are co-absent in almost every sample (both zero)
    D = np.zeros((n, 4), dtype=bool)
    D[:5, 0] = True  # taxon 0 detected in 5 samples
    D[:5, 1] = True  # taxon 1 detected in the SAME 5 samples (co-detected)
    D[0, 2] = True  # taxon 2 detected once, never with taxon 3
    L = np.random.default_rng(0).normal(size=(n, 4))
    site_labels = np.array([f"S{i % 6}" for i in range(n)])
    eligible, joint_count, joint_site_count, v_joint = _joint_eligibility_and_vjoint(
        L, D, site_labels, site_labels, min_joint_samples=10, min_joint_sites=3,
    )
    # pair (0,1): only 5 joint detections (< 10) -> not eligible despite being co-detected
    assert not eligible[0, 1]
    # co-absent pair (2,3): essentially never jointly detected -> not eligible
    assert not eligible[2, 3]
    assert joint_count[2, 3] == 0


def test_missing_assay_reported_and_excluded_from_cohort():
    a = pd.DataFrame({"ARC::f1": [10, 20, 30]}, index=["s1", "s2", "s3"])
    b = pd.DataFrame({"BAC::f1": [10, 20]}, index=["s1", "s2"])  # s3 missing from BAC
    meta = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "site_id": ["A", "A", "B"], "target": [1.0, 2.0, 3.0]}).set_index("sample_id", drop=False)
    cohort_ids, report = build_joint_cohort(a, b, meta, require_target=True)
    assert "s3" in report.missing_bac_assay_ids
    assert "s3" not in cohort_ids
    assert cohort_ids == ["s1", "s2"]


def test_zero_retained_count_training_row_is_dropped_with_reason():
    n = 12
    idx = [f"s{i}" for i in range(n)]
    a = pd.DataFrame(np.random.default_rng(1).integers(5, 50, size=(n, 4)), index=idx,
                      columns=[f"ARC::f{i}" for i in range(4)])
    b = pd.DataFrame(np.random.default_rng(2).integers(5, 50, size=(n, 4)), index=idx,
                      columns=[f"BAC::f{i}" for i in range(4)])
    a.iloc[0, :] = 0  # sample s0 has zero raw counts among ARC features
    site_map = {sid: f"SITE{i % 4}" for i, sid in enumerate(idx)}
    pp_cfg = {"detection_count": 1, "min_detected_samples": 2, "min_sample_prevalence": 0.0,
              "min_detected_sites": 1, "min_site_prevalence": 0.0}
    jp, kept = JointPreprocessor.fit(a, b, site_map, pp_cfg)
    assert "s0" not in kept
    reasons = {d["sample_id"]: d["reason"] for d in jp.dropped_train_samples_}
    assert "zero ARC total among retained features" in reasons["s0"]


def test_missing_retained_column_raises_schema_error_unless_opted_in():
    idx = [f"s{i}" for i in range(10)]
    train = pd.DataFrame(np.random.default_rng(3).integers(5, 50, size=(10, 3)), index=idx,
                          columns=["ARC::f0", "ARC::f1", "ARC::f2"])
    site_map = {sid: f"SITE{i % 3}" for i, sid in enumerate(idx)}
    dp = DomainPreprocessor("ARC", detection_count=1, min_detected_samples=2, min_sample_prevalence=0.0,
                             min_detected_sites=1, min_site_prevalence=0.0)
    dp.fit(train, site_map)

    # a genuine zero in a PRESENT column is fine
    new_ok = train.copy()
    new_ok.iloc[0, 0] = 0
    log_q, pmf = dp.transform_log_composition(new_ok, pseudocount=0.5)
    assert np.isfinite(log_q.to_numpy()).all()

    # an entirely ABSENT retained column is a schema error, not an implicit zero
    new_missing = train.drop(columns=["ARC::f1"])
    with pytest.raises(SchemaError):
        dp.transform_log_composition(new_missing, pseudocount=0.5)

    # opting in explicitly makes it an explicit zero-fill instead
    dp2 = DomainPreprocessor("ARC", detection_count=1, min_detected_samples=2, min_sample_prevalence=0.0,
                              min_detected_sites=1, min_site_prevalence=0.0,
                              missing_retained_feature_means_zero=True)
    dp2.fit(train, site_map)
    log_q2, pmf2 = dp2.transform_log_composition(new_missing, pseudocount=0.5)
    assert np.isfinite(log_q2.to_numpy()).all()

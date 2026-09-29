"""Item 18.6: group algebra. Shared group terms plus residual terms
reproduce the original predictor on a small known coefficient vector; the
r-vs-s scaling of an unequal-size balance; coefficient chains that would
exceed grouping tolerance are flagged ambiguous, not silently merged."""
import numpy as np
import pandas as pd
import pytest

from models.fusionML.groups import extract_groups, verify_group_reconstruction, B12Groups, compute_b12_feature, expand_b12_coefficient


def test_group_reconstruction_exact(small_synth, prefixed_counts):
    a, b = prefixed_counts
    log_a = np.log((a + 0.5).div((a + 0.5).sum(axis=1), axis=0))
    feature_ids = list(log_a.columns)
    p = len(feature_ids)
    rng = np.random.default_rng(5)
    beta = rng.normal(size=p)
    # force features 0,1,2 to share an exact value (a true group)
    beta[0] = beta[1] = beta[2] = 0.37
    beta -= beta.mean()  # keep zero-sum for realism (not required by the algebra itself)
    edges = [(0, 1, 1.0), (1, 2, 1.0)]

    result = extract_groups("ARC", feature_ids, beta, edges, tolerance=1e-6)
    assert result.group_of_feature[feature_ids[0]] == result.group_of_feature[feature_ids[1]] == result.group_of_feature[feature_ids[2]]

    max_err = verify_group_reconstruction(log_a, beta, feature_ids, result)
    assert max_err < 1e-8


def test_coefficient_chain_exceeding_tolerance_is_ambiguous(small_synth, prefixed_counts):
    a, b = prefixed_counts
    log_a = np.log((a + 0.5).div((a + 0.5).sum(axis=1), axis=0))
    feature_ids = list(log_a.columns)[:4]
    tol = 1e-6
    # chained pairwise differences within tolerance, but cumulative range exceeds tolerance
    beta = np.array([0.0, 0.6e-6, 1.2e-6, 1.8e-6])
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    result = extract_groups("ARC", feature_ids, beta, edges, tolerance=tol)
    assert len(result.ambiguous_components) == 1
    assert set(result.ambiguous_components[0]) == set(feature_ids)
    # conservatively retained as singletons, not one large fused group
    gids = {result.group_of_feature[f] for f in feature_ids}
    assert len(gids) == 4


def test_b12_r_s_scaling_is_not_equal_magnitude():
    groups = B12Groups(group1=[f"g1_{i}" for i in range(13)], group2=[f"g2_{i}" for i in range(8)])
    gamma = 2.0
    coefs = expand_b12_coefficient(gamma, groups)
    g1_val = coefs[groups.group1[0]]
    g2_val = coefs[groups.group2[0]]
    assert g1_val != pytest.approx(-g2_val)  # r != s -> not equal-magnitude opposite coefficients
    r, s = groups.r, groups.s
    k = np.sqrt(r * s / (r + s))
    assert g1_val == pytest.approx(gamma * k / r)
    assert g2_val == pytest.approx(-gamma * k / s)


def test_b12_feature_matches_group_mean_definition():
    idx = [f"s{i}" for i in range(10)]
    cols = [f"ARC::f{i}" for i in range(6)]
    rng = np.random.default_rng(9)
    log_q = pd.DataFrame(rng.normal(size=(10, 6)), index=idx, columns=cols)
    groups = B12Groups(group1=cols[:3], group2=cols[3:])
    b12 = compute_b12_feature(log_q, groups)
    r, s = groups.r, groups.s
    k = np.sqrt(r * s / (r + s))
    expected = k * (log_q[groups.group1].mean(axis=1) - log_q[groups.group2].mean(axis=1))
    pd.testing.assert_series_equal(b12, expected, check_names=False)

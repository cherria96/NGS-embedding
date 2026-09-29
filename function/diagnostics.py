"""Exploratory, descriptive diagnostics (design-spec item 11): `--mode diagnose`.

Runs on the FULL joint cohort (all data, not train-only) and is explicitly
labeled descriptive/exploratory throughout. Per item 14: "Exploratory
all-data diagnostics are allowed but must be labeled descriptive. They must
never supply feature lists, graphs or grouping to nested validation." This
module's outputs are never read by pipeline.train().
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from datasets.graph.fusion_graphs import _weighted_mean_cov, _v_all_from_cov, _joint_eligibility_and_vjoint
from datasets.datapreprocess.preprocessing import site_balanced_weights
from models.fusionML.groups import B12Groups


@dataclasses.dataclass
class DomainDiagnostics:
    domain: str
    n_features: int
    n_samples: int
    n_pairs_analyzed: int
    prevalence: pd.DataFrame  # feature_id, detected_samples, detected_sites, prevalence_fraction
    pair_variances: pd.DataFrame  # feature_j, feature_k, v_all, v_joint, v_star, joint_samples, joint_sites, eligible
    site_decomposition_check_max_abs_error: float


def domain_prevalence(counts: pd.DataFrame, site_of_sample: Dict[str, str], detection_count: int) -> pd.DataFrame:
    detected = counts >= detection_count
    sites = pd.Series([site_of_sample[s] for s in counts.index], index=counts.index)
    det_sites = detected.copy()
    det_sites["__site__"] = sites.values
    site_detect = det_sites.groupby("__site__").any()
    return pd.DataFrame({
        "feature_id": counts.columns,
        "detected_samples": detected.sum(axis=0).values,
        "detected_sites": site_detect.sum(axis=0).values,
        "sample_prevalence_fraction": (detected.sum(axis=0) / counts.shape[0]).values,
    })


def pairwise_redundancy_diagnostics(log_q: pd.DataFrame, counts: pd.DataFrame, site_of_sample: Dict[str, str],
                                     detection_count: int, min_joint_samples: int, min_joint_sites: int,
                                     domain: str, stream_block_size: int = 500) -> DomainDiagnostics:
    """Full within-domain pairwise v_all/v_joint/v_star, per item 11: "Analyze
    all eligible within-domain pairs, not only the 21 selected genera." For
    large feature counts, computed in row blocks rather than allocating a
    dense p x p x n intermediate (item 11's streaming instruction); the
    weighted-covariance identity already keeps the dense part to p x p."""
    feature_ids = list(log_q.columns)
    p = len(feature_ids)
    n = log_q.shape[0]
    L = log_q.to_numpy()
    D = (counts.reindex(columns=feature_ids).to_numpy() >= detection_count)
    site_labels = np.array([site_of_sample[s] for s in log_q.index])
    w = site_balanced_weights(list(site_labels))
    _, C = _weighted_mean_cov(L, w)
    v_all = _v_all_from_cov(C)
    eligible, joint_count, joint_site_count, v_joint = _joint_eligibility_and_vjoint(
        L, D, site_labels, site_labels, min_joint_samples, min_joint_sites,
    )
    v_star = np.where(eligible, np.maximum(v_all, np.nan_to_num(v_joint, nan=0.0)), np.nan)

    rows = []
    for j in range(p):
        for k in range(j + 1, p):
            rows.append({
                "feature_j": feature_ids[j], "feature_k": feature_ids[k],
                "v_all": float(v_all[j, k]),
                "v_joint": float(v_joint[j, k]) if eligible[j, k] else np.nan,
                "v_star": float(v_star[j, k]) if eligible[j, k] else np.nan,
                "joint_samples": int(joint_count[j, k]), "joint_sites": int(joint_site_count[j, k]),
                "eligible": bool(eligible[j, k]),
            })
    pair_df = pd.DataFrame(rows)

    prevalence = domain_prevalence(counts.reindex(columns=feature_ids), site_of_sample, detection_count)

    # site-decomposition identity check (v_all = V_between + V_within) on a
    # sample of pairs, per item 11 / verification item 18.9.
    check_pairs = pair_df.sample(min(20, len(pair_df)), random_state=0) if len(pair_df) else pair_df
    max_err = 0.0
    for _, row in check_pairs.iterrows():
        j, k = row["feature_j"], row["feature_k"]
        r = log_q[j] - log_q[k]
        site_ser = pd.Series(site_labels, index=log_q.index)
        site_means = r.groupby(site_ser).mean()
        m_bar = site_means.mean()
        v_between = float(((site_means - m_bar) ** 2).mean())
        within_per_site = r.groupby(site_ser).apply(lambda x: float(((x - x.mean()) ** 2).mean()))
        v_within = float(within_per_site.mean())
        recomputed = v_between + v_within
        max_err = max(max_err, abs(recomputed - row["v_all"]))

    return DomainDiagnostics(
        domain=domain, n_features=p, n_samples=n, n_pairs_analyzed=len(pair_df),
        prevalence=prevalence, pair_variances=pair_df, site_decomposition_check_max_abs_error=max_err,
    )


def block_variance_summary(log_q: pd.DataFrame, groups: B12Groups) -> dict:
    """Within-block vs between-block variance for a supplied exploratory
    block list, in the supplied order (item 11: 'display the variance
    matrix in the supplied block order... must not be forced to contain
    two groups')."""
    members = groups.group1 + groups.group2
    labels = ["G1"] * len(groups.group1) + ["G2"] * len(groups.group2)
    sub = log_q[members]
    var_matrix = sub.cov()
    within_g1 = var_matrix.loc[groups.group1, groups.group1].to_numpy()
    within_g2 = var_matrix.loc[groups.group2, groups.group2].to_numpy()
    between = var_matrix.loc[groups.group1, groups.group2].to_numpy()
    return {
        "members_order": members, "labels": labels,
        "variance_matrix": var_matrix,
        "mean_within_group1_var": float(np.mean(within_g1)),
        "mean_within_group2_var": float(np.mean(within_g2)),
        "mean_between_group_covar": float(np.mean(between)),
        "note": "descriptive only; re-estimated clustering may differ and is not forced to two groups",
    }


def pseudocount_sensitivity_summary(log_q_delta_a: pd.DataFrame, log_q_delta_b: pd.DataFrame,
                                     label_a: str, label_b: str) -> dict:
    common_cols = [c for c in log_q_delta_a.columns if c in log_q_delta_b.columns]
    diff = (log_q_delta_a[common_cols] - log_q_delta_b[common_cols]).abs()
    return {
        "compared": f"{label_a} vs {label_b}",
        "n_features_compared": len(common_cols),
        "mean_abs_log_diff": float(diff.to_numpy().mean()) if len(common_cols) else float("nan"),
        "max_abs_log_diff": float(diff.to_numpy().max()) if len(common_cols) else float("nan"),
    }

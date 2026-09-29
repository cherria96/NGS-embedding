"""Group extraction algebra and the optional b12 balance comparator.

Design-spec item 15. Step 6's algebra: for a group G with equal fitted
coefficients beta_j = b_G, sum_{j in G} beta_j log q_ij = |G| b_G
log(GM_{j in G} q_ij) -- a lossless algebraic reparameterization, not a
second model.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.modules.setdefault("fusionML.groups", _repository_sys.modules[__name__])
_repository_sys.modules.setdefault("models.fusionML.groups", _repository_sys.modules[__name__])


import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclasses.dataclass
class GroupExtractionResult:
    domain: str
    feature_ids: List[str]
    beta: np.ndarray
    group_of_feature: Dict[str, int]
    group_members: Dict[int, List[str]]
    group_mean_beta: Dict[int, float]
    group_theta: Dict[int, float]  # |G| * beta_G
    residuals: Dict[str, float]  # beta_j - beta_group(j)
    ambiguous_components: List[List[str]]
    tolerance: float


def extract_groups(domain: str, feature_ids: List[str], beta: np.ndarray,
                    edges: List[Tuple[int, int, float]], tolerance: float = 1e-6) -> GroupExtractionResult:
    """Connect positive-weight edges whose fitted coefficients differ by
    <= tolerance. A candidate connected component is only accepted as one
    group if its FULL coefficient range is also <= tolerance; otherwise it
    is reported ambiguous and every member is conservatively kept as its
    own singleton (item 15: "conservatively retain singleton features
    rather than chaining many small differences into one large
    difference")."""
    p = len(feature_ids)
    adj = {i: set() for i in range(p)}
    for j, k, wgt in edges:
        if wgt > 0 and abs(beta[j] - beta[k]) <= tolerance:
            adj[j].add(k)
            adj[k].add(j)

    visited = [False] * p
    components: List[List[int]] = []
    for i in range(p):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(sorted(comp))

    group_of_feature: Dict[str, int] = {}
    group_members: Dict[int, List[str]] = {}
    group_mean_beta: Dict[int, float] = {}
    group_theta: Dict[int, float] = {}
    residuals: Dict[str, float] = {}
    ambiguous_components: List[List[str]] = []
    gid = 0

    for comp in components:
        vals = beta[comp]
        coef_range = float(vals.max() - vals.min())
        if len(comp) > 1 and coef_range > tolerance:
            ambiguous_components.append([feature_ids[i] for i in comp])
            for i in comp:
                group_members[gid] = [feature_ids[i]]
                group_of_feature[feature_ids[i]] = gid
                group_mean_beta[gid] = float(beta[i])
                group_theta[gid] = float(beta[i])
                residuals[feature_ids[i]] = 0.0
                gid += 1
            continue
        members = [feature_ids[i] for i in comp]
        mean_beta = float(vals.mean())
        group_members[gid] = members
        group_mean_beta[gid] = mean_beta
        group_theta[gid] = len(comp) * mean_beta
        for i in comp:
            group_of_feature[feature_ids[i]] = gid
            residuals[feature_ids[i]] = float(beta[i] - mean_beta)
        gid += 1

    return GroupExtractionResult(
        domain=domain, feature_ids=feature_ids, beta=beta,
        group_of_feature=group_of_feature, group_members=group_members,
        group_mean_beta=group_mean_beta, group_theta=group_theta,
        residuals=residuals, ambiguous_components=ambiguous_components, tolerance=tolerance,
    )


def group_summary_feature(log_q: pd.DataFrame, members: List[str]) -> pd.Series:
    """m_iG = mean_{j in G} log q_ij."""
    return log_q[members].mean(axis=1)


def verify_group_reconstruction(log_q: pd.DataFrame, beta: np.ndarray, feature_ids: List[str],
                                 result: GroupExtractionResult, atol: float = 1e-8) -> float:
    """Max abs difference between the original predictor sum(beta*logq) and
    the group-summary + residual reconstruction. Should be ~0 (item 15:
    "Group summaries plus these residual terms reconstruct the fitted
    predictor exactly up to numerical precision")."""
    original = log_q[feature_ids].to_numpy() @ beta
    recon = np.zeros(log_q.shape[0])
    for gid, members in result.group_members.items():
        m_ig = group_summary_feature(log_q, members).to_numpy()
        theta = result.group_theta[gid]
        recon += theta * m_ig
        for fid in members:
            resid = result.residuals[fid]
            if resid != 0.0:
                recon += resid * (log_q[fid].to_numpy() - m_ig)
    return float(np.max(np.abs(original - recon)))


# ---------------------------------------------------------------------------
# Optional b12 balance comparator (item 15). Disabled unless
# optional.b12_comparator=true AND an exploratory_block_list is supplied
# with explicit genus IDs -- never guessed memberships.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class B12Groups:
    group1: List[str]
    group2: List[str]

    @property
    def r(self) -> int:
        return len(self.group1)

    @property
    def s(self) -> int:
        return len(self.group2)


def load_b12_groups(path: str) -> B12Groups:
    """exploratory_block_list format: two columns, `group` in {1,2} and
    `feature_id` (domain-prefixed), e.g. a CSV with columns group,feature_id."""
    df = pd.read_csv(path)
    if not {"group", "feature_id"}.issubset(df.columns):
        raise ValueError(f"{path}: expected columns 'group','feature_id'")
    g1 = df.loc[df["group"] == 1, "feature_id"].astype(str).tolist()
    g2 = df.loc[df["group"] == 2, "feature_id"].astype(str).tolist()
    if not g1 or not g2:
        raise ValueError(f"{path}: both group 1 and group 2 must be non-empty")
    return B12Groups(group1=g1, group2=g2)


def compute_b12_feature(log_q: pd.DataFrame, groups: B12Groups) -> pd.Series:
    """b12 = sqrt(r*s/(r+s)) * (mean(log q_G1) - mean(log q_G2))."""
    r, s = groups.r, groups.s
    k = np.sqrt(r * s / (r + s))
    m1 = log_q[groups.group1].mean(axis=1)
    m2 = log_q[groups.group2].mean(axis=1)
    return k * (m1 - m2)


def expand_b12_coefficient(gamma: float, groups: B12Groups) -> Dict[str, float]:
    """gamma*b12 expands to per-taxon coefficients gamma*k/r on group 1 and
    -gamma*k/s on group 2 (item 15)."""
    r, s = groups.r, groups.s
    k = np.sqrt(r * s / (r + s))
    out = {fid: gamma * k / r for fid in groups.group1}
    out.update({fid: -gamma * k / s for fid in groups.group2})
    return out

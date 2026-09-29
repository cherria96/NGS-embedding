"""Training-partition-only preprocessing: filtering, zero handling, log
composition, target/covariate scaling. Design-spec item 10.

Every stat here (retained feature set, pseudocount mass, site weights,
target mean/std, covariate imputers) is fit on a training partition and
then frozen/applied to held-out data -- never refit on validation or test
rows. Callers are responsible for passing only the correct sample subset
into `.fit(...)`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class InsufficientFeaturesError(ValueError):
    pass


class SchemaError(ValueError):
    pass


class SingleClassTrainingError(ValueError):
    """Item 10's classification-preprocessing rule: a training partition
    with only one class cannot fit the unpenalized logistic intercept at a
    finite optimum. Non-intercept families/candidates must report this
    partition unavailable rather than silently substituting a clipped
    constant fit -- only the dedicated intercept_only family may report
    that explicitly-labeled constant baseline."""
    pass


def site_balanced_weights(site_of_sample: List[str]) -> np.ndarray:
    """w_i = 1/(S * n_s), sum to 1. Design-spec item 11."""
    sites = pd.Series(site_of_sample)
    counts = sites.value_counts()
    S = counts.shape[0]
    w = sites.map(lambda s: 1.0 / (S * counts[s])).to_numpy(dtype=float)
    return w


@dataclasses.dataclass
class DomainFilterThresholds:
    detection_count: int
    min_detected_samples: int
    min_detected_sites: int
    n_train_samples: int
    n_train_sites: int


class DomainPreprocessor:
    """Fits retained-feature selection for one domain (ARC or BAC)."""

    def __init__(self, domain: str, detection_count: int,
                 min_detected_samples: int, min_sample_prevalence: float,
                 min_detected_sites: int, min_site_prevalence: float,
                 missing_retained_feature_means_zero: bool = False):
        self.domain = domain
        self.detection_count = detection_count
        self.min_sample_prevalence = min_sample_prevalence
        self.min_detected_samples_floor = min_detected_samples
        self.min_site_prevalence = min_site_prevalence
        self.min_detected_sites_floor = min_detected_sites
        self.missing_retained_feature_means_zero = missing_retained_feature_means_zero
        self.retained_features_: Optional[List[str]] = None
        self.thresholds_: Optional[DomainFilterThresholds] = None
        self.excluded_features_: Optional[pd.DataFrame] = None

    def fit(self, train_counts: pd.DataFrame, train_site_of_sample: Dict[str, str]) -> "DomainPreprocessor":
        n_train = train_counts.shape[0]
        sites = pd.Series(train_site_of_sample).reindex(train_counts.index)
        S_train = sites.nunique()

        min_detected_samples = max(self.min_detected_samples_floor, math.ceil(self.min_sample_prevalence * n_train))
        min_detected_sites = max(self.min_detected_sites_floor, math.ceil(self.min_site_prevalence * S_train))

        detected = train_counts >= self.detection_count
        detected_sample_counts = detected.sum(axis=0)

        # distinct-site detection count per feature
        det_sites = detected.copy()
        det_sites["__site__"] = sites.values
        site_detect = det_sites.groupby("__site__").any()
        detected_site_counts = site_detect.sum(axis=0)

        keep_mask = (detected_sample_counts >= min_detected_samples) & (detected_site_counts >= min_detected_sites)
        retained = sorted(train_counts.columns[keep_mask].tolist())
        excluded = pd.DataFrame({
            "feature_id": train_counts.columns,
            "detected_samples": detected_sample_counts.values,
            "detected_sites": detected_site_counts.values,
            "retained": keep_mask.values,
        })

        self.retained_features_ = retained
        self.excluded_features_ = excluded
        self.thresholds_ = DomainFilterThresholds(
            detection_count=self.detection_count,
            min_detected_samples=min_detected_samples,
            min_detected_sites=min_detected_sites,
            n_train_samples=n_train,
            n_train_sites=S_train,
        )
        if len(retained) < 2:
            raise InsufficientFeaturesError(
                f"domain {self.domain}: only {len(retained)} taxa retained after filtering "
                f"(detection_count={self.detection_count}, min_detected_samples="
                f"{min_detected_samples}, min_detected_sites={min_detected_sites}). "
                "The primary joint fit requires >= 2 retained taxa per domain (item 10). "
                "A domain-specific baseline can be requested explicitly as a separate model."
            )
        return self

    def raw_totals(self, counts: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        cols = self._present_retained_columns(counts)
        return counts[cols].sum(axis=1)

    def new_feature_mass_fraction(self, counts: pd.DataFrame) -> pd.Series:
        """Fraction of each sample's total abundance in columns present in
        `counts` but NOT part of the frozen retained set (informational)."""
        self._check_fitted()
        all_total = counts.sum(axis=1)
        retained_present = self._present_retained_columns(counts)
        retained_total = counts[retained_present].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(all_total > 0, 1.0 - (retained_total / all_total), 0.0)
        return pd.Series(frac, index=counts.index)

    def _present_retained_columns(self, counts: pd.DataFrame) -> List[str]:
        missing = [f for f in self.retained_features_ if f not in counts.columns]
        if missing and not self.missing_retained_feature_means_zero:
            raise SchemaError(
                f"domain {self.domain}: {len(missing)} retained feature column(s) are absent "
                f"from the supplied table (e.g. {missing[:5]}). The input contract does not "
                "establish that an omitted column means zero reads; set "
                "preprocessing.missing_retained_feature_means_zero=true to opt into that "
                "assumption explicitly."
            )
        return [f for f in self.retained_features_ if f in counts.columns]

    def transform_log_composition(self, counts: pd.DataFrame, pseudocount: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (log_q DataFrame over the frozen retained features, pseudocount_mass_fraction).

        Missing retained columns are filled with zero only when
        `missing_retained_feature_means_zero` was explicitly set.
        """
        self._check_fitted()
        present = self._present_retained_columns(counts)
        sub = counts.reindex(columns=self.retained_features_)
        if self.missing_retained_feature_means_zero:
            sub = sub.fillna(0.0)
        raw_total = sub[present].sum(axis=1) if not self.missing_retained_feature_means_zero else sub.sum(axis=1)
        p = len(self.retained_features_)
        delta = pseudocount
        tilde = sub.fillna(0.0) + delta
        closed_total = tilde.sum(axis=1)
        q = tilde.div(closed_total, axis=0)
        log_q = np.log(q)
        pseudocount_mass_fraction = (p * delta) / (raw_total + p * delta)
        return log_q, pseudocount_mass_fraction

    def _check_fitted(self):
        if self.retained_features_ is None:
            raise RuntimeError("DomainPreprocessor.fit() must be called before use")


@dataclasses.dataclass
class ScorabilityReport:
    dropped_train_samples: List[dict]
    unscorable_eval_samples: List[dict]


class JointPreprocessor:
    """Coordinates ARC + BAC DomainPreprocessors and cross-domain scorability."""

    def __init__(self, arc: DomainPreprocessor, bac: DomainPreprocessor):
        self.arc = arc
        self.bac = bac
        self.dropped_train_samples_: List[dict] = []

    @classmethod
    def fit(cls, train_counts_arc: pd.DataFrame, train_counts_bac: pd.DataFrame,
            train_site_of_sample: Dict[str, str], preprocessing_cfg: dict) -> Tuple["JointPreprocessor", List[str]]:
        common = "missing_retained_feature_means_zero"
        mrz = bool(preprocessing_cfg.get(common, False))
        arc = DomainPreprocessor(
            "ARC", preprocessing_cfg["detection_count"], preprocessing_cfg["min_detected_samples"],
            preprocessing_cfg["min_sample_prevalence"], preprocessing_cfg["min_detected_sites"],
            preprocessing_cfg["min_site_prevalence"], missing_retained_feature_means_zero=mrz,
        ).fit(train_counts_arc, train_site_of_sample)
        bac = DomainPreprocessor(
            "BAC", preprocessing_cfg["detection_count"], preprocessing_cfg["min_detected_samples"],
            preprocessing_cfg["min_sample_prevalence"], preprocessing_cfg["min_detected_sites"],
            preprocessing_cfg["min_site_prevalence"], missing_retained_feature_means_zero=mrz,
        ).fit(train_counts_bac, train_site_of_sample)

        jp = cls(arc, bac)
        totals_a = arc.raw_totals(train_counts_arc)
        totals_b = bac.raw_totals(train_counts_bac)
        scorable = (totals_a > 0) & (totals_b > 0)
        dropped = []
        for sid in train_counts_arc.index:
            if not bool(scorable.get(sid, False)):
                reason = []
                if not (totals_a.get(sid, 0) > 0):
                    reason.append("zero ARC total among retained features")
                if not (totals_b.get(sid, 0) > 0):
                    reason.append("zero BAC total among retained features")
                dropped.append({
                    "sample_id": sid,
                    "site_id": train_site_of_sample.get(sid),
                    "reason": "; ".join(reason),
                })
        jp.dropped_train_samples_ = dropped
        kept_ids = [s for s in train_counts_arc.index if bool(scorable.get(s, False))]
        return jp, kept_ids

    def transform(self, counts_arc: pd.DataFrame, counts_bac: pd.DataFrame,
                  pseudocount: float) -> dict:
        totals_a = self.arc.raw_totals(counts_arc)
        totals_b = self.bac.raw_totals(counts_bac)
        scorable = (totals_a > 0) & (totals_b > 0)
        reasons = {}
        for sid in counts_arc.index:
            if not bool(scorable.get(sid, False)):
                r = []
                if not (totals_a.get(sid, 0) > 0):
                    r.append("zero ARC total among retained features")
                if not (totals_b.get(sid, 0) > 0):
                    r.append("zero BAC total among retained features")
                reasons[sid] = "; ".join(r)

        log_a, pmf_a = self.arc.transform_log_composition(counts_arc, pseudocount)
        log_b, pmf_b = self.bac.transform_log_composition(counts_bac, pseudocount)
        return {
            "log_a": log_a, "log_b": log_b,
            "pseudocount_mass_fraction_a": pmf_a, "pseudocount_mass_fraction_b": pmf_b,
            "scorable": scorable, "unscorable_reasons": reasons,
        }


class TargetScaler:
    def __init__(self):
        self.mu_: Optional[float] = None
        self.sigma_: Optional[float] = None
        self.is_constant_: Optional[bool] = None

    def fit(self, y_train: np.ndarray, site_of_sample_train: List[str]) -> "TargetScaler":
        w = site_balanced_weights(site_of_sample_train)
        mu = float(np.sum(w * y_train))
        var = float(np.sum(w * (y_train - mu) ** 2))
        self.mu_ = mu
        self.sigma_ = math.sqrt(var) if var > 0 else 0.0
        self.is_constant_ = self.sigma_ == 0.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.is_constant_:
            return np.zeros_like(y, dtype=float)
        return (y - self.mu_) / self.sigma_

    def inverse(self, y_star: np.ndarray) -> np.ndarray:
        if self.is_constant_:
            return np.full_like(y_star, fill_value=self.mu_, dtype=float)
        return y_star * self.sigma_ + self.mu_

    def inverse_coefficients(self, beta_star: np.ndarray, b0_star: float) -> Tuple[np.ndarray, float]:
        return self.sigma_ * beta_star, self.mu_ + self.sigma_ * b0_star


class ClassificationLinkHandler:
    """Same fit/transform/inverse/inverse_coefficients interface as
    TargetScaler, so fit_partition/predict_with_partition_fit can stay
    task-agnostic, but for item 7's logistic-loss extension: labels are
    used raw (no standardization -- that's a regression-only concern),
    and "inverse" applies the logit link (eta -> probability) rather than
    undoing a linear rescale."""

    def __init__(self):
        self.is_constant_: Optional[bool] = None
        self.prevalence_: Optional[float] = None

    def fit(self, y_train: np.ndarray, site_of_sample_train: List[str]) -> "ClassificationLinkHandler":
        w = site_balanced_weights(site_of_sample_train)
        self.prevalence_ = float(np.sum(w * y_train))
        self.is_constant_ = len(np.unique(y_train)) < 2
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(eta, dtype=float)))

    def inverse_coefficients(self, beta_star: np.ndarray, b0_star: float) -> Tuple[np.ndarray, float]:
        return beta_star, b0_star  # logit-scale coefficients need no unscaling


class CovariatePreprocessor:
    """Numeric: site-balanced-weighted median impute + standardize (train-only).
    Categorical (object dtype): one-hot on train-seen categories; unseen
    categories at transform time map to an all-zero row and are reported.
    """

    def __init__(self, covariate_cols: List[str]):
        self.covariate_cols = covariate_cols
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.impute_values_: Dict[str, float] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.categories_: Dict[str, List[str]] = {}

    def fit(self, train_df: pd.DataFrame, site_of_sample_train: List[str]) -> "CovariatePreprocessor":
        w = site_balanced_weights(site_of_sample_train)
        for c in self.covariate_cols:
            col = train_df[c]
            if pd.api.types.is_numeric_dtype(col):
                self.numeric_cols_.append(c)
                filled = col.fillna(col.median())
                self.impute_values_[c] = float(col.median())
                mu = float(np.sum(w * filled.to_numpy()))
                var = float(np.sum(w * (filled.to_numpy() - mu) ** 2))
                self.means_[c] = mu
                self.stds_[c] = math.sqrt(var) if var > 0 else 1.0
            else:
                self.categorical_cols_.append(c)
                self.categories_[c] = sorted(col.dropna().astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        out = {}
        unseen_report = []
        for c in self.numeric_cols_:
            filled = df[c].fillna(self.impute_values_[c])
            out[c] = (filled - self.means_[c]) / self.stds_[c]
        for c in self.categorical_cols_:
            cats = self.categories_[c]
            for cat in cats:
                out[f"{c}={cat}"] = (df[c].astype(str) == cat).astype(float)
            unseen = ~df[c].astype(str).isin(cats)
            if unseen.any():
                unseen_report.extend(df.index[unseen].tolist())
        return pd.DataFrame(out, index=df.index), unseen_report

    def feature_names(self) -> List[str]:
        names = list(self.numeric_cols_)
        for c in self.categorical_cols_:
            names.extend(f"{c}={cat}" for cat in self.categories_[c])
        return names

"""Save/load a trained FusionModel bundle. `predict` mode loads this and
NEVER refits any part of it (design-spec item 17)."""
from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.fusionML import __version__ as FUSIONML_VERSION
from datasets.datapreprocess.preprocessing import DomainPreprocessor, CovariatePreprocessor, TargetScaler
from models.fusionML.interactions import InteractionTransform


@dataclasses.dataclass
class FusionModel:
    resolution: str
    domain: str  # "joint" for the primary model
    feature_ids_a: List[str]
    feature_ids_b: List[str]
    preproc_a: DomainPreprocessor
    preproc_b: DomainPreprocessor
    pseudocount: float
    target_scaler: TargetScaler
    covariate_preproc: Optional[CovariatePreprocessor]
    b0_star: float
    beta_a_star: np.ndarray
    beta_b_star: np.ndarray
    b0_orig: float
    beta_a_orig: np.ndarray
    beta_b_orig: np.ndarray
    graph_edges_a: pd.DataFrame
    graph_edges_b: pd.DataFrame
    alpha_arc: Optional[float]
    alpha_bac: Optional[float]
    lambda_s: float
    lambda_r: float
    lambda_f: float
    family: str
    metadata: dict
    task: str = "regression"
    classification_threshold: Optional[float] = None
    interaction_transform: Optional[InteractionTransform] = None
    gamma_star: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    gamma_orig: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))

    def predict(self, counts_a: pd.DataFrame, counts_b: pd.DataFrame) -> pd.DataFrame:
        """counts_a/counts_b: raw counts indexed by sample_id, domain-prefixed
        columns. Rows are matched by sample_id, never by position, and a
        sample present in only one domain's table is an entirely missing
        assay (item 9), not a zero-count row. New samples never alter the
        trained feature mapping; rows that fail scorability get a missing
        prediction + explicit reason."""
        all_ids = sorted(set(counts_a.index) | set(counts_b.index))
        present_a = pd.Series(True, index=counts_a.index).reindex(all_ids, fill_value=False)
        present_b = pd.Series(True, index=counts_b.index).reindex(all_ids, fill_value=False)

        log_a, _ = self.preproc_a.transform_log_composition(counts_a, self.pseudocount)
        log_b, _ = self.preproc_b.transform_log_composition(counts_b, self.pseudocount)
        totals_a = self.preproc_a.raw_totals(counts_a).reindex(all_ids)
        totals_b = self.preproc_b.raw_totals(counts_b).reindex(all_ids)
        la = log_a.reindex(all_ids).fillna(0.0)
        lb = log_b.reindex(all_ids).fillna(0.0)

        scorable = present_a & present_b & (totals_a > 0) & (totals_b > 0)
        reasons = []
        for sid in all_ids:
            if not present_a[sid] and not present_b[sid]:
                reasons.append("missing ARC and BAC assays")
            elif not present_a[sid]:
                reasons.append("missing ARC assay")
            elif not present_b[sid]:
                reasons.append("missing BAC assay")
            elif not (totals_a[sid] > 0) or not (totals_b[sid] > 0):
                reasons.append("zero total among retained features in ARC and/or BAC")
            else:
                reasons.append("")

        eta = self.b0_orig + la.to_numpy() @ self.beta_a_orig + lb.to_numpy() @ self.beta_b_orig
        if self.interaction_transform is not None and self.gamma_orig.size > 0:
            x_i = self.interaction_transform.transform(la, lb)
            eta = eta + x_i @ self.gamma_orig
        yhat = 1.0 / (1.0 + np.exp(-eta)) if self.task == "classification" else eta
        out = pd.DataFrame({
            "sample_id": all_ids,
            "prediction": np.where(scorable.to_numpy(), yhat, np.nan),
            "scorable": scorable.to_numpy(),
            "reason": reasons,
        })
        if self.task == "classification" and self.classification_threshold is not None:
            out["predicted_class"] = np.where(scorable.to_numpy(), (yhat >= self.classification_threshold).astype(float), np.nan)
        return out.set_index("sample_id", drop=False)


def save_model(model: FusionModel, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    sidecar = {
        "fusionml_version": FUSIONML_VERSION,
        "python_version": sys.version,
        "saved_at_unix": time.time(),
        "resolution": model.resolution,
        "family": model.family,
        "n_features_a": len(model.feature_ids_a),
        "n_features_b": len(model.feature_ids_b),
        "alpha_arc": model.alpha_arc,
        "alpha_bac": model.alpha_bac,
        "lambda_s": model.lambda_s,
        "lambda_r": model.lambda_r,
        "lambda_f": model.lambda_f,
        "task": model.task,
        "classification_threshold": model.classification_threshold,
        "has_interaction": model.interaction_transform is not None,
        "n_interaction_products": model.interaction_transform.n_products if model.interaction_transform is not None else 0,
    }
    with open(path + ".json", "w") as f:
        json.dump(sidecar, f, indent=2, default=str)


def load_model(path: str) -> FusionModel:
    with open(path, "rb") as f:
        return pickle.load(f)

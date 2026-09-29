"""Single-domain (ARC-only / BAC-only) ablation of the item-14 table's
models 2-5 (genus/ASV ridge & elastic-net, phylogeny-only, abundance-only).

Not part of the core two-domain family comparison in pipeline.py -- a
separately-requested ablation answering "how much predictive power comes
from one domain alone?" Reuses the exact same site-grouped nested-CV
protocol, evaluation conventions (site-balanced pooled OOF metrics,
inner-OOF-F2 classification threshold, item 10's single-class training
gate) and the SAME joint cohort (inputs.cohort_ids, built requiring both
assays present) as the two-domain families in pipeline.py, so results are
directly comparable fold-for-fold against models 2-5's joint numbers.

No family here gets an "intercept_only" exception for a single-class
training partition -- that baseline already exists once, in the joint
comparison; every single-domain candidate raises SingleClassTrainingError
on a single-class partition, same as every non-intercept_only joint family.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.modules.setdefault("fusionML.single_domain", _repository_sys.modules[__name__])
_repository_sys.modules.setdefault("models.fusionML.single_domain", _repository_sys.modules[__name__])


import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.fusionml import Config
from datasets.datapreprocess.preprocessing import (
    DomainPreprocessor, TargetScaler, ClassificationLinkHandler,
    InsufficientFeaturesError, SingleClassTrainingError, site_balanced_weights,
)
from datasets.graph.fusion_graphs import build_phylogenetic_graph, build_abundance_graph, mix_graphs, PhyloGraphResult, AbundanceGraphResult
from models.fusionML.estimator import fit_log_contrast_single_domain, SolverFailure, FitResult
from models.fusionML.groups import extract_groups
from function.folds import make_site_grouped_folds, sample_ids_for_sites, TooFewSitesError
from models.fusionML.search import one_standard_error_select
from function.evaluate import (
    compute_metrics, compute_classification_metrics, coverage as coverage_frac,
    per_sample_log_loss, select_threshold_via_inner_oof_f2,
)
from function.pipeline import LoadedInputs, tables_for_resolution, derive_seed


@dataclasses.dataclass(frozen=True)
class SingleDomainSpec:
    name: str
    resolution: str  # "genus" | "asv"
    fix_lambda_s_zero: bool  # ridge (True) vs elastic-net (False)
    graph_type: Optional[str]  # None | "phylogeny" | "abundance"


SINGLE_DOMAIN_SPECS: Dict[str, SingleDomainSpec] = {
    "ridge": SingleDomainSpec("ridge", "genus", True, None),
    "elastic_net": SingleDomainSpec("elastic_net", "genus", False, None),
    "asv_ridge": SingleDomainSpec("asv_ridge", "asv", True, None),
    "asv_elastic_net": SingleDomainSpec("asv_elastic_net", "asv", False, None),
    "phylogeny_only": SingleDomainSpec("phylogeny_only", "genus", False, "phylogeny"),
    "abundance_only": SingleDomainSpec("abundance_only", "genus", False, "abundance"),
}


@dataclasses.dataclass(frozen=True)
class SingleDomainCandidate:
    lambda_s: float
    lambda_r: float
    lambda_f: float = 0.0

    def key(self):
        return (round(self.lambda_s, 8), round(self.lambda_r, 8), round(self.lambda_f, 8))


def generate_single_domain_candidates(spec: SingleDomainSpec, search_cfg: dict, seed: int) -> List[SingleDomainCandidate]:
    lambda_s_grid = [0.0] if spec.fix_lambda_s_zero else list(search_cfg["lambda_s"])
    lambda_r_grid = list(search_cfg["lambda_r"])
    lambda_f_grid = list(search_cfg["lambda_f"]) if spec.graph_type else [0.0]
    budget = int(search_cfg["candidate_budget_per_family"])

    all_combos: Dict[tuple, SingleDomainCandidate] = {}
    for ls in lambda_s_grid:
        for lr in lambda_r_grid:
            for lf in lambda_f_grid:
                c = SingleDomainCandidate(ls, lr, lf)
                all_combos[c.key()] = c
    combos = list(all_combos.values())
    if len(combos) <= budget:
        return sorted(combos, key=lambda c: c.key())

    default_ls = lambda_s_grid[0]
    default_lr = lambda_r_grid[len(lambda_r_grid) // 2]
    required = {SingleDomainCandidate(default_ls, default_lr, 0.0).key(): SingleDomainCandidate(default_ls, default_lr, 0.0)}
    pool = [c for c in combos if c.key() not in required]
    rng = np.random.default_rng(seed)
    remaining = max(0, budget - len(required))
    sampled = []
    if remaining > 0 and pool:
        idx = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
        sampled = [pool[i] for i in idx]
    final = list(required.values()) + sampled
    return sorted(final, key=lambda c: c.key())[:budget]


@dataclasses.dataclass
class SingleDomainPartitionFit:
    dp: DomainPreprocessor
    target_scaler: object
    retained: List[str]
    pseudocount: float
    fit_result: FitResult
    edges_idx: List[Tuple[int, int, float]]
    kept_train_ids: List[str]
    classification_threshold: Optional[float] = None


def _domain_tree(inputs: LoadedInputs, domain: str, resolution: str):
    if resolution == "asv":
        return inputs.tree_arc if domain == "ARC" else inputs.tree_bac
    if inputs.rank_used == resolution:
        return inputs.rank_tree_arc if domain == "ARC" else inputs.rank_tree_bac
    return None


def fit_single_domain_partition(train_ids: List[str], domain: str, spec: SingleDomainSpec,
                                 candidate: SingleDomainCandidate, inputs: LoadedInputs,
                                 cfg: Config, seed: int) -> SingleDomainPartitionFit:
    counts_a_all, counts_b_all = tables_for_resolution(inputs, spec.resolution)
    counts_all = counts_a_all if domain == "ARC" else counts_b_all
    site_of_sample_train = {sid: inputs.metadata.loc[sid, "site_id"] for sid in train_ids}

    preprocessing_cfg = cfg["preprocessing"]
    task = cfg["task"]
    dp = DomainPreprocessor(
        domain, preprocessing_cfg["detection_count"], preprocessing_cfg["min_detected_samples"],
        preprocessing_cfg["min_sample_prevalence"], preprocessing_cfg["min_detected_sites"],
        preprocessing_cfg["min_site_prevalence"],
        missing_retained_feature_means_zero=bool(preprocessing_cfg.get("missing_retained_feature_means_zero", False)),
    ).fit(counts_all.loc[train_ids], site_of_sample_train)

    pseudocount = preprocessing_cfg["pseudocount"]
    totals = dp.raw_totals(counts_all.loc[train_ids])
    kept_train_ids = [s for s in train_ids if float(totals.get(s, 0)) > 0]
    site_of_sample_kept = {sid: site_of_sample_train[sid] for sid in kept_train_ids}

    y_train = inputs.metadata.loc[kept_train_ids, "target"].to_numpy(dtype=float)
    if task == "classification":
        if not set(np.unique(y_train).tolist()) <= {0.0, 1.0}:
            raise ValueError(f"task=classification requires a {{0,1}} target column, got values {np.unique(y_train)}")
        target_scaler = ClassificationLinkHandler().fit(y_train, [site_of_sample_kept[s] for s in kept_train_ids])
    else:
        target_scaler = TargetScaler().fit(y_train, [site_of_sample_kept[s] for s in kept_train_ids])

    if target_scaler.is_constant_:
        raise SingleClassTrainingError(
            f"single-domain family={spec.name} domain={domain}: training partition has a single class; "
            "not eligible for a non-intercept fit under item 10's classification-evaluation gate "
            "(no single-domain intercept_only baseline exists -- use the joint intercept_only row)."
        )

    log_q, _ = dp.transform_log_composition(counts_all.loc[kept_train_ids], pseudocount)
    retained = dp.retained_features_
    weights = site_balanced_weights([site_of_sample_kept[s] for s in kept_train_ids])
    y_star = target_scaler.transform(y_train)

    empty_edges = pd.DataFrame(columns=["feature_j", "feature_k", "mixed_weight"])
    edges_idx: List[Tuple[int, int, float]] = []
    if spec.graph_type and candidate.lambda_f > 0:
        graph_cfg = cfg["graph"]
        tree = _domain_tree(inputs, domain, spec.resolution)
        det = counts_all.loc[kept_train_ids, retained] >= preprocessing_cfg["detection_count"]
        if spec.graph_type == "phylogeny":
            phylo = build_phylogenetic_graph(tree, domain, retained, graph_cfg["k_phylogeny"]) if tree is not None \
                else PhyloGraphResult(domain, empty_edges, [], retained, None)
            emp = AbundanceGraphResult(domain, empty_edges, None, 0, graph_cfg["bootstrap_sites"], False)
            alpha = 1.0
        else:
            phylo = PhyloGraphResult(domain, empty_edges, [], retained, None)
            emp = build_abundance_graph(log_q, det, site_of_sample_kept, graph_cfg,
                                         seed=derive_seed(seed, "graph", domain), domain=domain)
            alpha = 0.0
        mixed = mix_graphs(domain, phylo, emp, alpha)
        pos = {f: i for i, f in enumerate(retained)}
        for _, row in mixed.iterrows():
            j, k, w = row["feature_j"], row["feature_k"], row["mixed_weight"]
            if pd.isna(w) or w <= 0 or j not in pos or k not in pos:
                continue
            edges_idx.append((pos[j], pos[k], float(w)))

    fit_fn = fit_log_contrast_single_domain
    fr = fit_fn(log_q.to_numpy(), y_star, weights, candidate.lambda_s, candidate.lambda_r,
                candidate.lambda_f, edges_idx, task, cfg["solver"])

    return SingleDomainPartitionFit(dp, target_scaler, retained, pseudocount, fr, edges_idx, kept_train_ids)


def predict_single_domain_partition(pf: SingleDomainPartitionFit, eval_ids: List[str],
                                     inputs: LoadedInputs, resolution: str, domain: str) -> pd.DataFrame:
    counts_a_all, counts_b_all = tables_for_resolution(inputs, resolution)
    counts_all = counts_a_all if domain == "ARC" else counts_b_all
    log_q, _ = pf.dp.transform_log_composition(counts_all.loc[eval_ids], pf.pseudocount)
    totals = pf.dp.raw_totals(counts_all.loc[eval_ids])
    scorable = (totals > 0).reindex(eval_ids).to_numpy()

    eta = pf.fit_result.b0 + log_q.to_numpy() @ pf.fit_result.beta_a
    yhat = pf.target_scaler.inverse(eta)
    out = pd.DataFrame({
        "sample_id": eval_ids,
        "prediction": np.where(scorable, yhat, np.nan),
        "scorable": scorable,
        "reason": np.where(scorable, "", f"zero {domain} total among retained features"),
    }).set_index("sample_id", drop=False)
    if pf.classification_threshold is not None:
        out["predicted_class"] = np.where(scorable, (yhat >= pf.classification_threshold).astype(float), np.nan)
    return out


def count_active_single_domain(pf: SingleDomainPartitionFit, tolerance: float) -> dict:
    active_mask = np.abs(pf.fit_result.beta_a) > tolerance
    g = extract_groups("single", pf.retained, pf.fit_result.beta_a, pf.edges_idx, tolerance)
    n_active_groups = sum(1 for v in g.group_mean_beta.values() if abs(v) > tolerance)
    return {"n_active_taxa": int(active_mask.sum()), "n_active_groups": n_active_groups}


def run_single_domain_inner_search(domain: str, spec: SingleDomainSpec, outer_train_ids: List[str],
                                    inputs: LoadedInputs, cfg: Config, outer_fold_idx) -> Tuple[SingleDomainCandidate, pd.DataFrame, dict]:
    site_map = {sid: inputs.metadata.loc[sid, "site_id"] for sid in outer_train_ids}
    inner_k = cfg["cv"]["inner_folds"]
    inner_seed = derive_seed(cfg["seed"], "inner_folds", "single", domain, spec.name, outer_fold_idx)
    inner_folds = make_site_grouped_folds(site_map, inner_k, inner_seed)

    cand_seed = derive_seed(cfg["seed"], "candidates", "single", domain, spec.name, outer_fold_idx)
    candidates = generate_single_domain_candidates(spec, cfg["search"], cand_seed)
    tolerance = cfg["compression"]["coefficient_tolerance"]
    required_coverage = cfg["cv"]["required_validation_coverage"]
    task = cfg["task"]
    fail_exceptions = (InsufficientFeaturesError, SolverFailure, TooFewSitesError, SingleClassTrainingError, ValueError)

    rows = []
    candidate_site_score = []
    candidate_oof: Dict[tuple, dict] = {}
    for cand in candidates:
        pooled_true, pooled_pred, pooled_site = [], [], []
        active_taxa_list, active_group_list = [], []
        n_scorable, n_total = 0, 0
        failed_reason = None
        for inner_fold_idx in range(inner_folds.k):
            inner_val_sites = inner_folds.sites_in_fold(inner_fold_idx)
            inner_train_sites = inner_folds.train_sites_for_outer_fold(inner_fold_idx)
            inner_train_ids = sample_ids_for_sites(site_map, inner_train_sites)
            inner_val_ids = sample_ids_for_sites(site_map, inner_val_sites)
            try:
                pf = fit_single_domain_partition(
                    inner_train_ids, domain, spec, cand, inputs, cfg,
                    seed=derive_seed(cfg["seed"], "single", domain, spec.name, outer_fold_idx, inner_fold_idx, cand.key()),
                )
            except fail_exceptions as e:
                failed_reason = f"inner_fold {inner_fold_idx}: {e}"
                break
            preds = predict_single_domain_partition(pf, inner_val_ids, inputs, spec.resolution, domain)
            n_total += len(inner_val_ids)
            scorable_ids = preds.index[preds["scorable"]].tolist()
            n_scorable += len(scorable_ids)
            if scorable_ids:
                pooled_true.extend(inputs.metadata.loc[scorable_ids, "target"].tolist())
                pooled_pred.extend(preds.loc[scorable_ids, "prediction"].tolist())
                pooled_site.extend(inputs.metadata.loc[scorable_ids, "site_id"].tolist())
            stats = count_active_single_domain(pf, tolerance)
            active_taxa_list.append(stats["n_active_taxa"])
            active_group_list.append(stats["n_active_groups"])

        cov = coverage_frac(n_scorable, n_total) if n_total > 0 else 0.0
        valid = failed_reason is None and n_total > 0 and cov >= required_coverage and len(pooled_true) > 0
        mean_site_score = None
        if valid:
            if task == "classification":
                ll = per_sample_log_loss(np.array(pooled_true), np.array(pooled_pred))
                per_site_ll = pd.DataFrame({"site_id": pooled_site, "log_loss": ll}).groupby("site_id")["log_loss"].mean()
                mean_site_score = float(per_site_ll.mean())
                site_score_array = per_site_ll.to_numpy()
            else:
                m = compute_metrics(np.array(pooled_true), np.array(pooled_pred), pooled_site)
                mean_site_score = m.site_balanced_mse
                site_score_array = m.per_site["mse"].to_numpy()
            complexity = (float(np.mean(active_group_list)), float(np.mean(active_taxa_list)))
            candidate_site_score.append((cand, site_score_array, complexity))
            candidate_oof[cand.key()] = {
                "prediction": np.array(pooled_pred), "target": np.array(pooled_true), "site_id": list(pooled_site),
            }
        rows.append({
            "lambda_s": cand.lambda_s, "lambda_r": cand.lambda_r, "lambda_f": cand.lambda_f,
            "coverage": cov, "valid": valid, "failed_reason": failed_reason, "mean_site_score": mean_site_score,
            "n_active_taxa": float(np.mean(active_taxa_list)) if active_taxa_list else None,
            "n_active_groups": float(np.mean(active_group_list)) if active_group_list else None,
        })

    search_df = pd.DataFrame(rows)
    if not candidate_site_score:
        raise RuntimeError(
            f"single-domain family={spec.name} domain={domain}, outer fold {outer_fold_idx}: no candidate met "
            f"the required validation coverage ({required_coverage}) with a successful constrained solve."
        )
    selected, info = one_standard_error_select(candidate_site_score)
    info["selected_inner_oof"] = candidate_oof[selected.key()]
    return selected, search_df, info


def run_single_domain_train(cfg: Config, inputs: LoadedInputs, domain: str, model_name: str) -> dict:
    """Mirrors pipeline.run_train's per-family outer loop for exactly one
    (domain, model) pair, using the SAME cohort/outer folds inputs.cohort_ids
    would give a joint family (same cfg["seed"] -> identical fold structure
    to the two-domain comparison, for a fold-for-fold-comparable ablation)."""
    spec = SINGLE_DOMAIN_SPECS[model_name]
    cohort = inputs.cohort_ids
    site_map_full = {sid: inputs.metadata.loc[sid, "site_id"] for sid in cohort}
    outer_folds = make_site_grouped_folds(site_map_full, cfg["cv"]["outer_folds"], cfg["seed"])

    family_oof = []
    for o in range(outer_folds.k):
        train_sites = outer_folds.train_sites_for_outer_fold(o)
        test_sites = outer_folds.sites_in_fold(o)
        train_ids = [s for s in sample_ids_for_sites(site_map_full, train_sites) if s in cohort]
        test_ids = [s for s in sample_ids_for_sites(site_map_full, test_sites) if s in cohort]

        try:
            selected, _, info = run_single_domain_inner_search(domain, spec, train_ids, inputs, cfg, o)
        except RuntimeError as e:
            for sid in test_ids:
                family_oof.append({"sample_id": sid, "site_id": site_map_full[sid],
                                    "target": inputs.metadata.loc[sid, "target"], "prediction": np.nan,
                                    "outer_fold": o, "scorable": False, "reason": f"inner search failed: {e}"})
            continue

        try:
            pf = fit_single_domain_partition(train_ids, domain, spec, selected, inputs, cfg,
                                              seed=derive_seed(cfg["seed"], "single_refit", domain, model_name, o))
        except (InsufficientFeaturesError, SolverFailure) as e:
            for sid in test_ids:
                family_oof.append({"sample_id": sid, "site_id": site_map_full[sid],
                                    "target": inputs.metadata.loc[sid, "target"], "prediction": np.nan,
                                    "outer_fold": o, "scorable": False, "reason": f"outer-training refit failed: {e}"})
            continue

        if cfg["task"] == "classification":
            inner_oof = info["selected_inner_oof"]
            pf.classification_threshold = select_threshold_via_inner_oof_f2(
                inner_oof["target"], inner_oof["prediction"], inner_oof["site_id"],
                grid=cfg.get("classification", "threshold_grid"),
            )
        preds = predict_single_domain_partition(pf, test_ids, inputs, spec.resolution, domain)
        for sid in test_ids:
            row = {"sample_id": sid, "site_id": site_map_full[sid], "target": inputs.metadata.loc[sid, "target"],
                   "prediction": preds.loc[sid, "prediction"], "outer_fold": o,
                   "scorable": bool(preds.loc[sid, "scorable"]), "reason": preds.loc[sid, "reason"]}
            if "predicted_class" in preds.columns:
                row["predicted_class"] = preds.loc[sid, "predicted_class"]
            family_oof.append(row)

    oof_df = pd.DataFrame(family_oof)
    if not len(oof_df):
        return {"status": "failed", "domain": domain, "model": model_name, "reason": "no scorable outer-fold predictions"}
    scorable_df = oof_df[oof_df["scorable"]]
    if not len(scorable_df):
        return {"status": "failed", "domain": domain, "model": model_name, "reason": "zero coverage"}

    if cfg["task"] == "classification":
        cm = compute_classification_metrics(
            scorable_df["target"].to_numpy(), scorable_df["prediction"].to_numpy(),
            scorable_df["site_id"].tolist(), scorable_df["predicted_class"].to_numpy(),
        )
        metrics = {
            "n": cm.n, "n_sites": cm.n_sites, "n_positive": cm.n_positive,
            "coverage": coverage_frac(len(scorable_df), len(oof_df)),
            "pooled_auc_roc": cm.pooled_auc_roc, "pooled_auc_pr": cm.pooled_auc_pr, "pooled_brier": cm.pooled_brier,
            "site_balanced_log_loss": cm.site_balanced_log_loss,
            "precision": cm.precision, "recall": cm.recall, "f1": cm.f1, "f2": cm.f2,
            "mcc": cm.mcc, "accuracy": cm.accuracy,
        }
    else:
        m = compute_metrics(scorable_df["target"].to_numpy(), scorable_df["prediction"].to_numpy(), scorable_df["site_id"].tolist())
        metrics = {
            "n": m.n, "n_sites": m.n_sites, "coverage": coverage_frac(len(scorable_df), len(oof_df)),
            "pooled_rmse": m.pooled_rmse, "pooled_mae": m.pooled_mae, "pooled_r2": m.pooled_r2,
            "site_balanced_mse": m.site_balanced_mse, "site_balanced_rmse": m.site_balanced_rmse,
            "site_balanced_mae": m.site_balanced_mae,
        }
    return {"status": "ok", "domain": domain, "model": model_name, "oof": oof_df, **metrics}

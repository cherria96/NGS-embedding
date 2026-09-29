"""Top-level orchestration for validate / diagnose / train / predict.

Wires together io_*, cohort, genus, preprocessing, graphs, estimator,
groups, folds, search, evaluate, artifacts, figures, model_io.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.fusionml import Config, validate_config_schema
from datasets.datapreprocess.io_counts import load_count_table, CountTableError
from datasets.datapreprocess.io_metadata import load_metadata, MetadataError
from datasets.datapreprocess.chemistry_targets import build_chemistry_derived_metadata
from datasets.datapreprocess.io_taxonomy import load_taxonomy, TaxonomyError
from datasets.datapreprocess.io_tree import load_tree, TreeError, DomainTree
from datasets.datapreprocess.cohort import build_joint_cohort, CohortReport
from datasets.datapreprocess.genus import aggregate_to_genus, GenusAggregationReport
from datasets.datapreprocess.rank_mapping import load_rank_mapping, aggregate_via_mapping
from datasets.datapreprocess.preprocessing import (
    JointPreprocessor, DomainPreprocessor, TargetScaler, ClassificationLinkHandler, InsufficientFeaturesError,
    SchemaError, SingleClassTrainingError, site_balanced_weights,
)
from datasets.graph.fusion_graphs import (
    build_phylogenetic_graph, build_abundance_graph, build_traversal_order_graph, mix_graphs,
    PhyloGraphResult, AbundanceGraphResult,
)
from models.fusionML.estimator import (
    fit_log_contrast, fit_log_contrast_logistic, fit_log_contrast_with_interaction,
    intercept_only_result, intercept_only_result_logistic,
    SolverFailure, FitResult, solver_environment_info,
)
from models.fusionML.groups import extract_groups
from function.folds import make_site_grouped_folds, sample_ids_for_sites, TooFewSitesError, FoldAssignment
from models.fusionML.search import (
    FAMILY_SPECS, FamilySpec, Candidate, InteractionCandidate, generate_candidates,
    generate_interaction_candidates, one_standard_error_select, PRIMARY_FAMILY, INTERACTION_FAMILY,
    INTERACTION_BASE_FAMILY_ALIAS,
)
from models.fusionML.interactions import fit_interaction_transform, InteractionTransform, InteractionUnavailableError
from function.evaluate import (
    compute_metrics, coverage as coverage_frac, bootstrap_paired_site_differences,
    compute_classification_metrics, bootstrap_paired_site_log_loss_differences,
    per_sample_log_loss, select_threshold_via_inner_oof_f2,
)
from function.artifacts import ensure_dir, write_json, write_df, hash_file, build_run_manifest, dependency_versions
from function.figures import (
    plot_oof_predicted_vs_observed, plot_residuals_by_site, plot_compression_vs_error, plot_graph_weight_summary,
    plot_roc_curve,
)
from function.model_io import FusionModel, save_model, load_model


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def derive_seed(base_seed: int, *parts) -> int:
    h = hashlib.sha256(f"{base_seed}::{'::'.join(str(p) for p in parts)}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LoadedInputs:
    counts_arc_asv: Optional[pd.DataFrame]
    counts_bac_asv: Optional[pd.DataFrame]
    counts_arc_genus: Optional[pd.DataFrame]
    counts_bac_genus: Optional[pd.DataFrame]
    genus_crosswalk_arc: Optional[pd.DataFrame]
    genus_crosswalk_bac: Optional[pd.DataFrame]
    metadata: pd.DataFrame
    tree_arc: Optional[DomainTree]
    tree_bac: Optional[DomainTree]
    taxonomy_arc: Optional[pd.DataFrame]
    taxonomy_bac: Optional[pd.DataFrame]
    cohort_ids: List[str]
    cohort_report: CohortReport
    reports: dict
    rank_used: Optional[str] = None  # which rank (genus/family/order/class) rank_counts_* hold, if any
    rank_counts_arc: Optional[pd.DataFrame] = None
    rank_counts_bac: Optional[pd.DataFrame] = None
    rank_tree_arc: Optional[DomainTree] = None
    rank_tree_bac: Optional[DomainTree] = None
    rank_crosswalk_arc: Optional[pd.DataFrame] = None
    rank_crosswalk_bac: Optional[pd.DataFrame] = None


def load_all_inputs(cfg: Config, require_target: bool = True) -> LoadedInputs:
    orientation = cfg.get("inputs", "orientation")
    relative_mode = bool(cfg.get("preprocessing", "relative_input_mode"))
    reports = {}

    arc_path = cfg.resolved_input_path("arc_counts")
    bac_path = cfg.resolved_input_path("bac_counts")
    meta_path = cfg.resolved_input_path("metadata")
    chem_cfg = cfg.get("chemistry")
    if arc_path is None or bac_path is None:
        raise ValueError("inputs.arc_counts and inputs.bac_counts are required")
    if meta_path is None and chem_cfg is None:
        raise ValueError("either inputs.metadata or a chemistry: block is required to supply sample_id/site_id/target")

    counts_arc, rep_a = load_count_table(arc_path, "ARC", orientation, relative_mode)
    counts_bac, rep_b = load_count_table(bac_path, "BAC", orientation, relative_mode)
    reports["arc_counts"] = rep_a
    reports["bac_counts"] = rep_b

    columns_cfg = cfg["columns"]
    if meta_path is not None:
        metadata, meta_report = load_metadata(
            meta_path, columns_cfg["sample_id"], columns_cfg["site_id"],
            columns_cfg["target"] if require_target else None, columns_cfg.get("covariates", []),
        )
    else:
        # design-spec item 9's chemistry-derived assembly path -- data/final/metadata.csv
        # is never read; sample_id/site_id/target come straight from Dat_chem.xlsx's
        # Chem_Se.Avg sheet (chemistry_targets.py), per explicit user instruction.
        workbook_path = cfg.resolved_path(chem_cfg["workbook"])
        metadata, meta_report = build_chemistry_derived_metadata(
            workbook_path, chem_cfg, columns_cfg["target"], columns_cfg.get("covariates", []),
        )
    reports["metadata"] = meta_report

    taxonomy_arc = taxonomy_bac = None
    tax_a_path = cfg.resolved_input_path("arc_taxonomy")
    tax_b_path = cfg.resolved_input_path("bac_taxonomy")
    if tax_a_path and tax_b_path:
        taxonomy_arc, reports["arc_taxonomy"] = load_taxonomy(tax_a_path, "ARC")
        taxonomy_bac, reports["bac_taxonomy"] = load_taxonomy(tax_b_path, "BAC")

    tree_arc = tree_bac = None
    tree_a_path = cfg.resolved_input_path("arc_tree")
    tree_b_path = cfg.resolved_input_path("bac_tree")
    if tree_a_path:
        tree_arc = load_tree(tree_a_path, "ARC")
    if tree_b_path:
        tree_bac = load_tree(tree_b_path, "BAC")

    counts_arc_genus = counts_bac_genus = None
    gcw_a = gcw_b = None
    if taxonomy_arc is not None and taxonomy_bac is not None:
        counts_arc_genus, gcw_a, reports["arc_genus_agg"] = aggregate_to_genus(counts_arc, taxonomy_arc, "ARC")
        counts_bac_genus, gcw_b, reports["bac_genus_agg"] = aggregate_to_genus(counts_bac, taxonomy_bac, "BAC")

    cohort_ids, cohort_report = build_joint_cohort(counts_arc, counts_bac, metadata, require_target=require_target)

    rank_used = None
    rank_counts_arc = rank_counts_bac = None
    rank_tree_arc = rank_tree_bac = None
    rank_cw_a = rank_cw_b = None
    resolution = cfg["resolution"]
    rank_trees_cfg = cfg.get("inputs", "rank_trees") or {}
    if resolution in ("genus", "family", "order", "class") and resolution in rank_trees_cfg:
        rank_used, rank_counts_arc, rank_counts_bac, rank_tree_arc, rank_tree_bac, rank_cw_a, rank_cw_b, rank_reports = \
            load_rank_resolution(cfg, resolution, counts_arc, counts_bac)
        reports.update(rank_reports)

    return LoadedInputs(
        counts_arc_asv=counts_arc, counts_bac_asv=counts_bac,
        counts_arc_genus=counts_arc_genus, counts_bac_genus=counts_bac_genus,
        genus_crosswalk_arc=gcw_a, genus_crosswalk_bac=gcw_b,
        metadata=metadata, tree_arc=tree_arc, tree_bac=tree_bac,
        taxonomy_arc=taxonomy_arc, taxonomy_bac=taxonomy_bac,
        cohort_ids=cohort_ids, cohort_report=cohort_report, reports=reports,
        rank_used=rank_used, rank_counts_arc=rank_counts_arc, rank_counts_bac=rank_counts_bac,
        rank_tree_arc=rank_tree_arc, rank_tree_bac=rank_tree_bac,
        rank_crosswalk_arc=rank_cw_a, rank_crosswalk_bac=rank_cw_b,
    )


def load_rank_resolution(cfg: Config, rank: str, counts_arc: pd.DataFrame, counts_bac: pd.DataFrame):
    """Loads a rank_trees[{rank}] tree + its co-located ASV->reduced-feature
    mapping.csv (or an explicit rank_tree_mapping.mapping_file override) and
    aggregates raw ASV counts to that tree-consistent resolution. See
    rank_mapping.py for why this differs from genus.py's naive aggregation."""
    rank_trees_cfg = cfg.get("inputs", "rank_trees")[rank]
    mapping_override = (cfg.get("parsing", "rank_tree_mapping", default={}) or {}).get("mapping_file")
    reports = {}
    results = {}
    for domain, counts in (("ARC", counts_arc), ("BAC", counts_bac)):
        tree_key = "arc" if domain == "ARC" else "bac"
        tree_path = cfg.resolved_path(rank_trees_cfg[tree_key])
        tree = load_tree(tree_path, domain)
        mapping_path = mapping_override or (tree_path.rsplit("_tree.nwk", 1)[0] + "_mapping.csv")
        mapping = load_rank_mapping(mapping_path, domain)
        agg, rep = aggregate_via_mapping(counts, mapping, domain, rank)
        covered, uncovered = tree.coverage(list(agg.columns))
        if cfg.get("parsing", "rank_tree_mapping", default={}).get("require_unambiguous_taxon_mapping", True) and uncovered:
            raise ValueError(
                f"{domain} rank={rank}: {len(uncovered)} aggregated feature(s) not found as tree tips "
                f"(e.g. {uncovered[:5]}); rank_tree_mapping.require_unambiguous_taxon_mapping=true forbids this."
            )
        results[domain] = (agg, tree, rep, mapping_path)
        reports[f"{domain.lower()}_rank_{rank}_agg"] = dataclasses.asdict(rep)
        reports[f"{domain.lower()}_rank_{rank}_tree_coverage"] = {"covered": len(covered), "uncovered": len(uncovered)}

    return (rank, results["ARC"][0], results["BAC"][0], results["ARC"][1], results["BAC"][1],
            None, None, reports)


def tables_for_resolution(inputs: LoadedInputs, resolution: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if resolution == "asv":
        return inputs.counts_arc_asv, inputs.counts_bac_asv
    if inputs.rank_used == resolution and inputs.rank_counts_arc is not None:
        return inputs.rank_counts_arc, inputs.rank_counts_bac
    if resolution == "genus":
        if inputs.counts_arc_genus is None:
            raise ValueError("resolution=genus requires arc_taxonomy and bac_taxonomy inputs (or inputs.rank_trees.genus)")
        return inputs.counts_arc_genus, inputs.counts_bac_genus
    raise ValueError(
        f"resolution={resolution!r} requires inputs.rank_trees.{resolution} (tree + co-located mapping.csv) "
        "to be configured -- there is no naive/tree-free aggregation available at this rank (item 9: "
        "'do not replace a genus by an arbitrary representative ASV')."
    )


# ---------------------------------------------------------------------------
# Family applicability
# ---------------------------------------------------------------------------

def applicable_families(cfg: Config, inputs: LoadedInputs) -> List[str]:
    resolution = cfg["resolution"]
    requested = cfg.get("cv", "families")
    tree_available_at_resolution = (resolution == "asv" and inputs.tree_arc is not None and inputs.tree_bac is not None) or \
        (inputs.rank_used == resolution and inputs.rank_tree_arc is not None and inputs.rank_tree_bac is not None)
    out = []
    for name, spec in FAMILY_SPECS.items():
        if spec.uses_interaction and not cfg.get("interaction", "enabled"):
            continue
        if spec.resolution == "genus" and inputs.counts_arc_genus is None:
            continue
        if spec.resolution == "asv" and inputs.counts_arc_asv is None:
            continue
        if spec.uses_graph:
            is_abundance_only = spec.alpha_mode == "fixed" and spec.fixed_alpha == (0.0, 0.0)
            if not is_abundance_only and not tree_available_at_resolution:
                continue
            fam_res = spec.resolution if spec.resolution in ("genus", "asv") else resolution
            if fam_res == "genus" and inputs.counts_arc_genus is None and inputs.rank_used != "genus":
                continue
            if fam_res == "asv" and inputs.counts_arc_asv is None:
                continue
            if fam_res in ("family", "order", "class") and inputs.rank_used != fam_res:
                continue
        out.append(name)
    if requested:
        out = [f for f in out if f in requested]
    return out


def family_resolution(spec: FamilySpec, cfg_resolution: str) -> str:
    return spec.resolution if spec.resolution in ("genus", "asv") else cfg_resolution


# ---------------------------------------------------------------------------
# Single-partition fit / predict
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PartitionFit:
    jp: JointPreprocessor
    target_scaler: TargetScaler
    retained_a: List[str]
    retained_b: List[str]
    pseudocount: float
    fit_result: FitResult
    mixed_edges_a: pd.DataFrame
    mixed_edges_b: pd.DataFrame
    edges_a_idx: List[Tuple[int, int, float]]
    edges_b_idx: List[Tuple[int, int, float]]
    dropped_train_samples: list
    kept_train_ids: List[str]
    classification_threshold: Optional[float] = None
    interaction_transform: Optional[InteractionTransform] = None
    interaction_report: Optional[dict] = None
    inner_oof_probs: Optional[np.ndarray] = None  # this candidate's pooled inner-OOF (prob, y, site) -- set by run_inner_search
    inner_oof_y: Optional[np.ndarray] = None
    inner_oof_sites: Optional[List[str]] = None


def _mixed_edges_to_index_pairs(mixed: pd.DataFrame, feature_order: List[str]) -> List[Tuple[int, int, float]]:
    if mixed is None or len(mixed) == 0 or "mixed_weight" not in mixed.columns:
        return []
    pos = {f: i for i, f in enumerate(feature_order)}
    out = []
    for _, row in mixed.iterrows():
        j, k, w = row["feature_j"], row["feature_k"], row["mixed_weight"]
        if pd.isna(w) or w <= 0:
            continue
        if j not in pos or k not in pos:
            continue
        out.append((pos[j], pos[k], float(w)))
    return out


def fit_partition(train_ids: List[str], spec: FamilySpec, candidate, inputs: LoadedInputs,
                   resolution: str, cfg: Config, seed: int) -> PartitionFit:
    is_interaction_cand = isinstance(candidate, InteractionCandidate)
    base = candidate.base if is_interaction_cand else candidate

    counts_a_all, counts_b_all = tables_for_resolution(inputs, resolution)
    site_of_sample_train = {sid: inputs.metadata.loc[sid, "site_id"] for sid in train_ids}

    preprocessing_cfg = cfg["preprocessing"]
    task = cfg["task"]
    jp, kept_train_ids = JointPreprocessor.fit(
        counts_a_all.loc[train_ids], counts_b_all.loc[train_ids], site_of_sample_train, preprocessing_cfg,
    )
    site_of_sample_kept = {sid: site_of_sample_train[sid] for sid in kept_train_ids}
    y_train = inputs.metadata.loc[kept_train_ids, "target"].to_numpy(dtype=float)
    if task == "classification":
        if not set(np.unique(y_train).tolist()) <= {0.0, 1.0}:
            raise ValueError(f"task=classification requires a {{0,1}} target column, got values {np.unique(y_train)}")
        target_scaler = ClassificationLinkHandler().fit(y_train, [site_of_sample_kept[s] for s in kept_train_ids])
    else:
        target_scaler = TargetScaler().fit(y_train, [site_of_sample_kept[s] for s in kept_train_ids])

    pseudocount = preprocessing_cfg["pseudocount"]
    tf = jp.transform(counts_a_all.loc[kept_train_ids], counts_b_all.loc[kept_train_ids], pseudocount)
    log_a, log_b = tf["log_a"], tf["log_b"]
    retained_a, retained_b = jp.arc.retained_features_, jp.bac.retained_features_

    weights = site_balanced_weights([site_of_sample_kept[s] for s in kept_train_ids])
    y_star = target_scaler.transform(y_train)

    empty_edges = pd.DataFrame(columns=["feature_j", "feature_k", "mixed_weight"])

    if target_scaler.is_constant_:
        # item 10's classification-preprocessing rule: a single-class training
        # partition cannot fit the unpenalized logistic intercept at a finite
        # optimum. Only the dedicated intercept_only family may report the
        # resulting clipped constant as an explicitly-labeled baseline; every
        # other family/candidate must report this partition unavailable.
        if task == "classification" and spec.name != "intercept_only":
            raise SingleClassTrainingError(
                f"family={spec.name}: training partition has a single class; not eligible for a "
                "non-intercept fit under item 10's classification-evaluation gate."
            )
        fr = (intercept_only_result_logistic(y_star, weights, len(retained_a), len(retained_b)) if task == "classification"
              else intercept_only_result(y_star, weights, len(retained_a), len(retained_b)))
        return PartitionFit(jp, target_scaler, retained_a, retained_b, pseudocount, fr,
                             empty_edges, empty_edges, [], [], jp.dropped_train_samples_, kept_train_ids)

    # Resolved once, unconditionally: used by the fusion graphs (only when
    # lambda_f>0) AND by the interaction transform (regardless of lambda_f
    # -- items 8/10 make interactions independent of the fusion graph).
    if resolution == "asv":
        tree_a, tree_b = inputs.tree_arc, inputs.tree_bac
    elif inputs.rank_used == resolution:
        tree_a, tree_b = inputs.rank_tree_arc, inputs.rank_tree_bac
    else:
        tree_a = tree_b = None
    det_a = counts_a_all.loc[kept_train_ids, retained_a] >= preprocessing_cfg["detection_count"]
    det_b = counts_b_all.loc[kept_train_ids, retained_b] >= preprocessing_cfg["detection_count"]

    mixed_a, mixed_b = empty_edges, empty_edges
    if base.lambda_f > 0 and spec.uses_graph and spec.graph_source == "metadr_order":
        # MetaDR's traversal-order spatial-locality graph (search.py's
        # metadr_order_only family) -- alpha is unused here (there's only
        # one graph component, not a phylo/abundance mix); mix_graphs is
        # still reused with alpha=1.0 against an empty abundance graph so
        # the rest of the pipeline (edge indexing, penalty construction)
        # needs no separate code path.
        order_a = build_traversal_order_graph(tree_a, "ARC", retained_a)
        order_b = build_traversal_order_graph(tree_b, "BAC", retained_b)
        empty_emp_a = AbundanceGraphResult("ARC", empty_edges, None, 0, cfg["graph"]["bootstrap_sites"], False)
        empty_emp_b = AbundanceGraphResult("BAC", empty_edges, None, 0, cfg["graph"]["bootstrap_sites"], False)
        mixed_a = mix_graphs("ARC", order_a, empty_emp_a, 1.0)
        mixed_b = mix_graphs("BAC", order_b, empty_emp_b, 1.0)
    elif base.lambda_f > 0 and spec.uses_graph:
        graph_cfg = cfg["graph"]

        # A domain excluded by spec.fusion_domains gets NO graph built at all
        # (not merely a zero weight) -- its coefficients stay unfused and the
        # expensive bootstrap abundance graph is skipped for it entirely.
        arc_fused = spec.fusion_domains in ("both", "arc_only")
        bac_fused = spec.fusion_domains in ("both", "bac_only")

        need_phylo_a = arc_fused and base.alpha_arc is not None and base.alpha_arc > 0 and tree_a is not None
        need_emp_a = arc_fused and base.alpha_arc is not None and base.alpha_arc < 1
        need_phylo_b = bac_fused and base.alpha_bac is not None and base.alpha_bac > 0 and tree_b is not None
        need_emp_b = bac_fused and base.alpha_bac is not None and base.alpha_bac < 1

        phylo_a = build_phylogenetic_graph(tree_a if need_phylo_a else None, "ARC", retained_a, graph_cfg["k_phylogeny"]) \
            if need_phylo_a else PhyloGraphResult("ARC", empty_edges, [], retained_a, None)
        phylo_b = build_phylogenetic_graph(tree_b if need_phylo_b else None, "BAC", retained_b, graph_cfg["k_phylogeny"]) \
            if need_phylo_b else PhyloGraphResult("BAC", empty_edges, [], retained_b, None)
        emp_a = build_abundance_graph(log_a, det_a, site_of_sample_kept, graph_cfg, seed=derive_seed(seed, "graphA"), domain="ARC") \
            if need_emp_a else AbundanceGraphResult("ARC", empty_edges, None, 0, graph_cfg["bootstrap_sites"], False)
        emp_b = build_abundance_graph(log_b, det_b, site_of_sample_kept, graph_cfg, seed=derive_seed(seed, "graphB"), domain="BAC") \
            if need_emp_b else AbundanceGraphResult("BAC", empty_edges, None, 0, graph_cfg["bootstrap_sites"], False)

        mixed_a = mix_graphs("ARC", phylo_a, emp_a, base.alpha_arc) if arc_fused else empty_edges
        mixed_b = mix_graphs("BAC", phylo_b, emp_b, base.alpha_bac) if bac_fused else empty_edges

    edges_a_idx = _mixed_edges_to_index_pairs(mixed_a, retained_a)
    edges_b_idx = _mixed_edges_to_index_pairs(mixed_b, retained_b)

    interaction_transform, interaction_report = None, None
    if spec.uses_interaction:
        interaction_transform, interaction_report = fit_interaction_transform(
            log_a, log_b, det_a, det_b, site_of_sample_kept, weights,
            tree_a, tree_b, retained_a, retained_b, cfg["interaction"],
        )

    if is_interaction_cand and candidate.interaction_on:
        x_i_train = interaction_transform.transform(log_a, log_b)
        if x_i_train.shape[1] == 0:
            raise InteractionUnavailableError(
                f"family={spec.name}: no interaction product survived in this training partition "
                "(interaction-on candidate unavailable; interaction-off remains eligible)."
            )
        fr = fit_log_contrast_with_interaction(
            log_a.to_numpy(), log_b.to_numpy(), x_i_train, y_star, weights,
            base.lambda_s, base.lambda_r, base.lambda_f, candidate.lambda_i1, candidate.lambda_i2,
            edges_a_idx, edges_b_idx, task, cfg["solver"],
        )
    else:
        # Interaction-off (or a non-interaction family): reuse the plain
        # additive estimator exactly, per "reuse exact matching additive
        # candidate fits" -- fr.gamma defaults to an empty array.
        fit_fn = fit_log_contrast_logistic if task == "classification" else fit_log_contrast
        fr = fit_fn(
            log_a.to_numpy(), log_b.to_numpy(), y_star, weights,
            base.lambda_s, base.lambda_r, base.lambda_f, edges_a_idx, edges_b_idx, cfg["solver"],
        )

    return PartitionFit(jp, target_scaler, retained_a, retained_b, pseudocount, fr,
                         mixed_a, mixed_b, edges_a_idx, edges_b_idx, jp.dropped_train_samples_, kept_train_ids,
                         interaction_transform=interaction_transform, interaction_report=interaction_report)


def predict_with_partition_fit(pf: PartitionFit, eval_ids: List[str], inputs: LoadedInputs,
                                resolution: str) -> pd.DataFrame:
    counts_a_all, counts_b_all = tables_for_resolution(inputs, resolution)
    tf = pf.jp.transform(counts_a_all.loc[eval_ids], counts_b_all.loc[eval_ids], pf.pseudocount)
    log_a, log_b, scorable, reasons = tf["log_a"], tf["log_b"], tf["scorable"], tf["unscorable_reasons"]

    eta = pf.fit_result.b0 + log_a.to_numpy() @ pf.fit_result.beta_a + log_b.to_numpy() @ pf.fit_result.beta_b
    if pf.interaction_transform is not None and pf.fit_result.gamma.size > 0:
        x_i_eval = pf.interaction_transform.transform(log_a, log_b)
        eta = eta + x_i_eval @ pf.fit_result.gamma
    yhat_orig = pf.target_scaler.inverse(eta)
    scorable_arr = scorable.reindex(eval_ids).to_numpy()
    out = pd.DataFrame({
        "sample_id": eval_ids,
        "prediction": np.where(scorable_arr, yhat_orig, np.nan),
        "scorable": scorable_arr,
        "reason": [reasons.get(sid, "") for sid in eval_ids],
    }).set_index("sample_id", drop=False)
    if pf.classification_threshold is not None:
        out["predicted_class"] = np.where(scorable_arr, (yhat_orig >= pf.classification_threshold).astype(float), np.nan)
    return out


def count_active(pf: PartitionFit, tolerance: float) -> dict:
    """Item 15's compression accounting, additive part always; item 15's
    interaction-accounting addendum (active products, distinct parent
    balances, required raw taxa = active main effects UNION active
    products' balance parents) when this partition used the interaction
    transform."""
    active_a_mask = np.abs(pf.fit_result.beta_a) > tolerance
    active_b_mask = np.abs(pf.fit_result.beta_b) > tolerance
    ga = extract_groups("ARC", pf.retained_a, pf.fit_result.beta_a, pf.edges_a_idx, tolerance)
    gb = extract_groups("BAC", pf.retained_b, pf.fit_result.beta_b, pf.edges_b_idx, tolerance)
    ng_a = sum(1 for v in ga.group_mean_beta.values() if abs(v) > tolerance)
    ng_b = sum(1 for v in gb.group_mean_beta.values() if abs(v) > tolerance)

    result = {
        "n_active_taxa": int(active_a_mask.sum() + active_b_mask.sum()),
        "n_active_groups": ng_a + ng_b,
        "n_active_products": 0, "n_distinct_parent_balances": 0, "n_required_taxa": None,
    }
    if pf.interaction_transform is not None:
        gamma = pf.fit_result.gamma
        active_products = [p for p, g in zip(pf.interaction_transform.products, gamma) if abs(g) > tolerance]
        used_a = {p.balance_a_id for p in active_products}
        used_b = {p.balance_b_id for p in active_products}
        parent_taxa = set()
        for b in pf.interaction_transform.balances_a:
            if b.canonical_id in used_a:
                parent_taxa.update(b.members)
        for b in pf.interaction_transform.balances_b:
            if b.canonical_id in used_b:
                parent_taxa.update(b.members)
        main_effect_taxa = {f for f, active in zip(pf.retained_a, active_a_mask) if active}
        main_effect_taxa |= {f for f, active in zip(pf.retained_b, active_b_mask) if active}
        result.update({
            "n_active_products": len(active_products),
            "n_distinct_parent_balances": len(used_a) + len(used_b),
            "n_required_taxa": len(main_effect_taxa | parent_taxa),
        })
    return result


def interaction_artifacts_from_pf(pf: PartitionFit, tolerance: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """item 17's "Additional required interaction artifacts":
    balance_definitions.csv (one row per selected balance, both domains)
    and interaction_coefficients.csv (one row per fitted product, parent
    balance IDs + gamma in both fitted (standardized) and original target
    units, plus an activity flag at the compression tolerance)."""
    it = pf.interaction_transform
    balance_rows = []
    for balances in (it.balances_a, it.balances_b):
        for b in balances:
            r, s = b.candidate.r, b.candidate.s
            k = math.sqrt(r * s / (r + s))
            balance_rows.append({
                "domain": b.candidate.domain, "canonical_id": b.canonical_id,
                "positive_members": ";".join(b.candidate.positive_members),
                "negative_members": ";".join(b.candidate.negative_members),
                "r": r, "s": s, "loading_pos_weight": k / r, "loading_neg_weight": -k / s,
                "mean": b.mean, "sd": b.sd, "variance": b.variance,
            })
    balance_df = pd.DataFrame(balance_rows)

    gamma_star = pf.fit_result.gamma
    gamma_orig = np.zeros(0)
    if gamma_star.size > 0:
        gamma_orig, _ = pf.target_scaler.inverse_coefficients(gamma_star, 0.0)
    coef_rows = []
    for i, p in enumerate(it.products):
        g_star = float(gamma_star[i]) if i < len(gamma_star) else 0.0
        g_orig = float(gamma_orig[i]) if i < len(gamma_orig) else 0.0
        coef_rows.append({
            "product_index": i, "balance_a_id": p.balance_a_id, "balance_b_id": p.balance_b_id,
            "gamma_star": g_star, "gamma_orig": g_orig, "active": abs(g_star) > tolerance,
        })
    coef_df = pd.DataFrame(coef_rows)
    return balance_df, coef_df


# ---------------------------------------------------------------------------
# Inner search
# ---------------------------------------------------------------------------

def run_inner_search(family_name: str, spec: FamilySpec, outer_train_ids: List[str], inputs: LoadedInputs,
                      resolution: str, cfg: Config, outer_fold_idx) -> Tuple[object, pd.DataFrame, dict]:
    site_map = {sid: inputs.metadata.loc[sid, "site_id"] for sid in outer_train_ids}
    inner_k = cfg["cv"]["inner_folds"]
    inner_seed = derive_seed(cfg["seed"], "inner_folds", family_name, outer_fold_idx)
    inner_folds = make_site_grouped_folds(site_map, inner_k, inner_seed)

    # The interaction family's base beta/alpha grid must be the identical
    # deterministic set domain_adaptive_hybrid uses (same seed derivation
    # and same underlying spec fields), so the two are compared on the same
    # candidate base points (arc-bac-interactions.md item 14).
    base_family_for_seed = INTERACTION_BASE_FAMILY_ALIAS.get(family_name, family_name)
    cand_seed = derive_seed(cfg["seed"], "candidates", base_family_for_seed, outer_fold_idx)
    base_candidates = generate_candidates(FAMILY_SPECS[base_family_for_seed], cfg["search"], cand_seed)
    candidates = generate_interaction_candidates(base_candidates, cfg["interaction"]) if spec.uses_interaction else base_candidates

    tolerance = cfg["compression"]["coefficient_tolerance"]
    required_coverage = cfg["cv"]["required_validation_coverage"]
    task = cfg["task"]
    fail_exceptions = (InsufficientFeaturesError, SolverFailure, TooFewSitesError,
                        SingleClassTrainingError, InteractionUnavailableError)

    rows = []
    candidate_site_score = []
    candidate_oof: Dict[tuple, dict] = {}
    for cand in candidates:
        pooled_true, pooled_pred, pooled_site = [], [], []
        active_taxa_list, active_group_list = [], []
        active_product_list, required_taxa_list = [], []
        n_scorable, n_total = 0, 0
        failed_reason = None
        for inner_fold_idx in range(inner_folds.k):
            inner_val_sites = inner_folds.sites_in_fold(inner_fold_idx)
            inner_train_sites = inner_folds.train_sites_for_outer_fold(inner_fold_idx)
            inner_train_ids = sample_ids_for_sites(site_map, inner_train_sites)
            inner_val_ids = sample_ids_for_sites(site_map, inner_val_sites)
            try:
                pf = fit_partition(inner_train_ids, spec, cand, inputs, resolution, cfg,
                                    seed=derive_seed(cfg["seed"], family_name, outer_fold_idx, inner_fold_idx, cand.key()))
            except fail_exceptions as e:
                failed_reason = f"inner_fold {inner_fold_idx}: {e}"
                break
            preds = predict_with_partition_fit(pf, inner_val_ids, inputs, resolution)
            n_total += len(inner_val_ids)
            scorable_ids = preds.index[preds["scorable"]].tolist()
            n_scorable += len(scorable_ids)
            if scorable_ids:
                pooled_true.extend(inputs.metadata.loc[scorable_ids, "target"].tolist())
                pooled_pred.extend(preds.loc[scorable_ids, "prediction"].tolist())
                pooled_site.extend(inputs.metadata.loc[scorable_ids, "site_id"].tolist())
            stats = count_active(pf, tolerance)
            active_taxa_list.append(stats["n_active_taxa"])
            active_group_list.append(stats["n_active_groups"])
            if spec.uses_interaction:
                active_product_list.append(stats["n_active_products"])
                required_taxa_list.append(stats["n_required_taxa"])

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
            # item 14's complexity tie-break: additive families use
            # (n_active_groups, n_active_taxa); the interaction family uses
            # (n_active_products, n_required_raw_taxa, n_active_groups).
            if spec.uses_interaction:
                complexity = (float(np.mean(active_product_list)), float(np.mean(required_taxa_list)),
                              float(np.mean(active_group_list)))
            else:
                complexity = (float(np.mean(active_group_list)), float(np.mean(active_taxa_list)))
            candidate_site_score.append((cand, site_score_array, complexity))
            candidate_oof[cand.key()] = {
                "prediction": np.array(pooled_pred), "target": np.array(pooled_true), "site_id": list(pooled_site),
            }
        base = cand.base if spec.uses_interaction else cand
        row = {
            "lambda_s": base.lambda_s, "lambda_r": base.lambda_r, "lambda_f": base.lambda_f,
            "alpha_arc": base.alpha_arc, "alpha_bac": base.alpha_bac,
            "coverage": cov, "valid": valid, "failed_reason": failed_reason,
            "mean_site_score": mean_site_score,
            "n_active_taxa": float(np.mean(active_taxa_list)) if active_taxa_list else None,
            "n_active_groups": float(np.mean(active_group_list)) if active_group_list else None,
        }
        if spec.uses_interaction:
            row.update({
                "interaction_on": cand.interaction_on, "lambda_i1": cand.lambda_i1, "lambda_i2": cand.lambda_i2,
                "n_active_products": float(np.mean(active_product_list)) if active_product_list else None,
                "n_required_taxa": float(np.mean(required_taxa_list)) if required_taxa_list else None,
            })
        rows.append(row)

    search_df = pd.DataFrame(rows)
    if not candidate_site_score:
        raise RuntimeError(
            f"family {family_name}, outer fold {outer_fold_idx}: no candidate met the required validation "
            f"coverage ({required_coverage}) with a successful constrained solve across all inner folds."
        )
    selected, info = one_standard_error_select(candidate_site_score)
    info["selected_inner_oof"] = candidate_oof[selected.key()]
    return selected, search_df, info


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_validate(config_path: str) -> dict:
    cfg = Config.from_yaml(config_path)
    if cfg.is_project_schema:
        return run_validate_project(cfg)

    issues = validate_config_schema(cfg)
    report = {"config_issues": [str(i) for i in issues], "input_checks": []}

    for key in ("arc_counts", "bac_counts", "metadata", "arc_tree", "bac_tree", "arc_taxonomy", "bac_taxonomy"):
        path = cfg.resolved_input_path(key)
        if path is None:
            report["input_checks"].append({"input": key, "status": "not configured"})
            continue
        if not os.path.exists(path):
            report["input_checks"].append({"input": key, "status": "MISSING", "path": path})
        else:
            report["input_checks"].append({"input": key, "status": "found", "path": path})

    chem_cfg = cfg.get("chemistry")
    if chem_cfg:
        wb_path = cfg.resolved_path(chem_cfg.get("workbook"))
        status = "not configured" if wb_path is None else ("found" if os.path.exists(wb_path) else "MISSING")
        report["input_checks"].append({"input": "chemistry.workbook", "status": status, "path": wb_path})

    has_errors = any(i.level == "error" for i in issues)
    has_missing = any(c["status"] == "MISSING" for c in report["input_checks"])
    if has_errors or has_missing:
        report["ready_to_run"] = False
        return report

    try:
        inputs = load_all_inputs(cfg, require_target=(cfg.get("columns", "target") is not None))
        report["cohort_report"] = dataclasses.asdict(inputs.cohort_report)
        report["loader_reports"] = {
            k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v) for k, v in inputs.reports.items()
        }
        report["ready_to_run"] = True
    except (CountTableError, MetadataError, TaxonomyError, TreeError, ValueError) as e:
        report["ready_to_run"] = False
        report["load_error"] = str(e)
    return report


def run_validate_project(cfg: Config) -> dict:
    """--mode validate for the "project config" schema (ad-microbial-config.yaml):
    inspects microbial files (with orientation auto-detection) and the
    chemistry workbook, and reports a resolved schema plus explicit
    blockers -- it never guesses a target or sample crosswalk (item 16:
    "validate command should inspect files, output a proposed/resolved
    schema and list remaining blockers; train must not guess them")."""
    from datasets.datapreprocess.chemistry import load_chemistry, ChemistryError

    report: dict = {"schema": "project", "blockers": [], "warnings": [], "input_checks": []}

    microbial_keys = ["arc_counts", "bac_counts", "arc_taxonomy", "bac_taxonomy", "arc_tree", "bac_tree"]
    for key in microbial_keys:
        path = cfg.resolved_input_path(key)
        status = "not configured" if path is None else ("found" if os.path.exists(path) else "MISSING")
        report["input_checks"].append({"input": key, "status": status, "path": path})

    rank_trees_cfg = cfg.get("inputs", "rank_trees") or {}
    for rank, paths in rank_trees_cfg.items():
        for side in ("arc", "bac"):
            p = cfg.resolved_path(paths.get(side))
            status = "not configured" if p is None else ("found" if os.path.exists(p) else "MISSING")
            report["input_checks"].append({"input": f"rank_trees.{rank}.{side}", "status": status, "path": p})
            if status == "found":
                mapping_path = p.rsplit("_tree.nwk", 1)[0] + "_mapping.csv"
                report["input_checks"].append({
                    "input": f"rank_trees.{rank}.{side}_mapping (derived)", "status":
                        "found" if os.path.exists(mapping_path) else "MISSING", "path": mapping_path,
                })

    if any(c["status"] == "MISSING" for c in report["input_checks"]):
        report["blockers"].append("one or more configured input files are missing on disk; see input_checks")

    # --- microbial: orientation auto-detect + taxonomy/tree load -----------
    try:
        tax_a, tax_a_rep = load_taxonomy(cfg.resolved_input_path("arc_taxonomy"), "ARC")
        tax_b, tax_b_rep = load_taxonomy(cfg.resolved_input_path("bac_taxonomy"), "BAC")
        report["taxonomy"] = {"arc": tax_a_rep, "bac": tax_b_rep}

        orientation = cfg.get("inputs", "orientation")
        relative_mode = bool(cfg.get("preprocessing", "relative_input_mode", default=False))
        ref_a = set(f.split("::", 1)[1] for f in tax_a.index)
        ref_b = set(f.split("::", 1)[1] for f in tax_b.index)
        counts_a, rep_a = load_count_table(cfg.resolved_input_path("arc_counts"), "ARC", orientation, relative_mode, ref_a)
        counts_b, rep_b = load_count_table(cfg.resolved_input_path("bac_counts"), "BAC", orientation, relative_mode, ref_b)
        report["counts"] = {
            "arc": {"shape": list(counts_a.shape), "orientation_used": rep_a.orientation_used,
                    "orientation_detection": rep_a.orientation_detection},
            "bac": {"shape": list(counts_b.shape), "orientation_used": rep_b.orientation_used,
                    "orientation_detection": rep_b.orientation_detection},
        }

        tree_a = load_tree(cfg.resolved_input_path("arc_tree"), "ARC")
        tree_b = load_tree(cfg.resolved_input_path("bac_tree"), "BAC")
        cov_a, uncov_a = tree_a.coverage(list(counts_a.columns))
        cov_b, uncov_b = tree_b.coverage(list(counts_b.columns))
        report["asv_tree_coverage"] = {
            "arc": {"covered": len(cov_a), "uncovered": len(uncov_a)},
            "bac": {"covered": len(cov_b), "uncovered": len(uncov_b)},
        }

        resolution = cfg["resolution"]
        if resolution in ("family", "order", "class") or (resolution == "genus" and resolution in rank_trees_cfg):
            _, _, _, _, _, _, _, rank_reports = load_rank_resolution(cfg, resolution, counts_a, counts_b)
            report["rank_aggregation"] = rank_reports
    except (CountTableError, TaxonomyError, TreeError, ValueError) as e:
        report["microbial_load_error"] = str(e)
        report["blockers"].append(f"microbial input loading failed: {e}")

    # --- chemistry -----------------------------------------------------
    chem_cfg = cfg.get("chemistry")
    if chem_cfg:
        try:
            workbook_path = cfg.resolved_path(chem_cfg["workbook"])
            chem_result = load_chemistry(workbook_path, chem_cfg)
            bad_headers = [h for h in chem_result.substrate.header_checks + chem_result.effluent.header_checks
                           if not h.measurement_ok or not h.season_ok]
            pending_overrides = [
                h for h in chem_result.substrate.header_checks + chem_result.effluent.header_checks
                if h.season_source == "override"
            ]
            report["chemistry"] = {
                "workbook_path": workbook_path,
                "sha256_actual": chem_result.sha256_actual,
                "sha256_expected": chem_result.sha256_expected,
                "sha256_matches": chem_result.sha256_matches,
                "substrate_records": chem_result.substrate.n_records_detected,
                "effluent_records": chem_result.effluent.n_records_detected,
                "header_checks_total": len(chem_result.substrate.header_checks) + len(chem_result.effluent.header_checks),
                "header_checks_failed": len(bad_headers),
                "season_header_overrides_used_pending_confirmation": len(pending_overrides),
                "join_report": chem_result.join_report,
                "n_notes_substrate": len(chem_result.substrate.notes),
                "n_notes_effluent": len(chem_result.effluent.notes),
                "sample_notes": [dataclasses.asdict(n) for n in (chem_result.substrate.notes + chem_result.effluent.notes)[:10]],
            }
            if chem_result.sha256_matches is False:
                report["warnings"].append(
                    "chemistry.workbook sha256 no longer matches chemistry.inspected_sha256 -- the file changed "
                    "since it was inspected. Every configured header was still re-validated live against the "
                    "current file and passed, but re-confirm before treating this as authoritative."
                )
            if bad_headers:
                report["blockers"].append(f"{len(bad_headers)} configured chemistry header coordinate(s) no longer match the workbook")
            if pending_overrides:
                report["warnings"].append(
                    f"{len(pending_overrides)} chemistry cells use season_header_overrides "
                    "(explicit_assumption_pending_confirmation, per the config's own audit notes) rather than a "
                    "season code read directly from the sheet."
                )
        except (ChemistryError, KeyError, FileNotFoundError) as e:
            report["chemistry_load_error"] = str(e)
            report["blockers"].append(f"chemistry loading failed: {e}")

    # --- target / crosswalk resolution status ---------------------------
    target_cfg = cfg.get("target") or {}
    sample_metadata_cfg = cfg.get("sample_metadata") or {}
    if not target_cfg.get("source_column"):
        report["blockers"].append(
            "target.source_column is not set -- no scientific target is configured; train must not guess one "
            "(item 9). Resolve target.source/source_column/units (or an approved formula) before training."
        )
    if not cfg.get("inputs", "metadata") and not sample_metadata_cfg.get("crosswalk_path"):
        report["blockers"].append(
            "inputs.metadata is null and sample_metadata.crosswalk_path is null -- no microbial "
            "sample<->site/season crosswalk is configured; train must not guess sample/site identity from "
            "chemistry record numbers or acronyms (item 9)."
        )

    report["ready_to_run"] = len(report["blockers"]) == 0
    return report


def run_diagnose(config_path: str | Config) -> dict:
    from function.diagnostics import pairwise_redundancy_diagnostics, block_variance_summary, pseudocount_sensitivity_summary
    from models.fusionML.groups import load_b12_groups

    cfg = config_path if isinstance(config_path, Config) else Config.from_yaml(config_path)
    if cfg.is_project_schema:
        raise NotImplementedError(
            "diagnose from a project-schema config is not wired in (same blocker as train: chemistry-derived "
            "target/crosswalk assembly isn't implemented). Run --mode validate first."
        )
    inputs = load_all_inputs(cfg, require_target=True)
    out_dir = ensure_dir(os.path.join(cfg.resolved_output_dir(), "diagnostics"))

    resolution = cfg["resolution"]
    counts_a, counts_b = tables_for_resolution(inputs, resolution)
    cohort = inputs.cohort_ids
    site_map = {sid: inputs.metadata.loc[sid, "site_id"] for sid in cohort}
    pp = cfg["preprocessing"]
    graph_cfg = cfg["graph"]

    jp, kept_ids = JointPreprocessor.fit(counts_a.loc[cohort], counts_b.loc[cohort], site_map, pp)
    tf = jp.transform(counts_a.loc[kept_ids], counts_b.loc[kept_ids], pp["pseudocount"])
    site_map_kept = {sid: site_map[sid] for sid in kept_ids}

    diag_a = pairwise_redundancy_diagnostics(
        tf["log_a"], counts_a.loc[kept_ids, jp.arc.retained_features_], site_map_kept,
        pp["detection_count"], graph_cfg["min_joint_samples"], graph_cfg["min_joint_sites"], "ARC",
    )
    diag_b = pairwise_redundancy_diagnostics(
        tf["log_b"], counts_b.loc[kept_ids, jp.bac.retained_features_], site_map_kept,
        pp["detection_count"], graph_cfg["min_joint_samples"], graph_cfg["min_joint_sites"], "BAC",
    )
    write_df(diag_a.prevalence, os.path.join(out_dir, f"prevalence_ARC_{resolution}.csv"))
    write_df(diag_b.prevalence, os.path.join(out_dir, f"prevalence_BAC_{resolution}.csv"))
    write_df(diag_a.pair_variances, os.path.join(out_dir, f"pair_variances_ARC_{resolution}.csv"))
    write_df(diag_b.pair_variances, os.path.join(out_dir, f"pair_variances_BAC_{resolution}.csv"))

    summary = {
        "resolution_used": resolution,
        "note": "descriptive/exploratory only; not used by pipeline.train() feature/graph selection",
        "ARC": {"n_features": diag_a.n_features, "n_samples": diag_a.n_samples,
                "n_pairs_analyzed": diag_a.n_pairs_analyzed,
                "site_decomposition_check_max_abs_error": diag_a.site_decomposition_check_max_abs_error},
        "BAC": {"n_features": diag_b.n_features, "n_samples": diag_b.n_samples,
                "n_pairs_analyzed": diag_b.n_pairs_analyzed,
                "site_decomposition_check_max_abs_error": diag_b.site_decomposition_check_max_abs_error},
    }

    block_list_path = cfg.resolved_input_path("exploratory_block_list")
    if block_list_path and os.path.exists(block_list_path):
        groups = load_b12_groups(block_list_path)
        bv_a = block_variance_summary(tf["log_a"], groups) if set(groups.group1 + groups.group2) <= set(jp.arc.retained_features_) else None
        bv_b = block_variance_summary(tf["log_b"], groups) if set(groups.group1 + groups.group2) <= set(jp.bac.retained_features_) else None
        if bv_a:
            write_df(bv_a["variance_matrix"].reset_index(), os.path.join(out_dir, "block_variance_ARC.csv"))
            summary["block_variance_ARC"] = {k: v for k, v in bv_a.items() if k != "variance_matrix"}
        if bv_b:
            write_df(bv_b["variance_matrix"].reset_index(), os.path.join(out_dir, "block_variance_BAC.csv"))
            summary["block_variance_BAC"] = {k: v for k, v in bv_b.items() if k != "variance_matrix"}

    sens_deltas = cfg["preprocessing"]["sensitivity_pseudocounts"]
    if sens_deltas:
        sens_summary = {}
        for delta in sens_deltas:
            tf_alt = jp.transform(counts_a.loc[kept_ids], counts_b.loc[kept_ids], delta)
            s_a = pseudocount_sensitivity_summary(tf["log_a"], tf_alt["log_a"], f"delta={pp['pseudocount']}", f"delta={delta}")
            s_b = pseudocount_sensitivity_summary(tf["log_b"], tf_alt["log_b"], f"delta={pp['pseudocount']}", f"delta={delta}")
            sens_summary[str(delta)] = {"ARC": s_a, "BAC": s_b}
        summary["pseudocount_sensitivity"] = sens_summary

    write_json(summary, os.path.join(out_dir, "diagnostics_summary.json"))
    return summary


def _print(msg: str):
    print(msg, flush=True)


def run_train(config_path: str) -> dict:
    cfg = Config.from_yaml(config_path)
    if cfg.is_project_schema:
        validation = run_validate_project(cfg)
        if not validation["ready_to_run"]:
            raise ValueError(
                "project-schema config is not ready to train; run --mode validate for the full report. "
                "Blockers:\n" + "\n".join(f"- {b}" for b in validation["blockers"])
            )
        raise NotImplementedError(
            "training directly from a project-schema config (chemistry-derived target/crosswalk assembly) "
            "is not wired into run_train; once target/crosswalk are resolved, assemble a metadata CSV "
            "(sample_id, site_id, target, ...) from the validated chemistry join and use the item-16-style "
            "config schema (see configs/real_research_template.yaml) to train."
        )

    issues = validate_config_schema(cfg)
    if any(i.level == "error" for i in issues):
        raise ValueError("config has errors; run --mode validate first:\n" + "\n".join(str(i) for i in issues))

    run_id = cfg.get("run_id") or f"train_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir = ensure_dir(cfg.resolved_output_dir())
    _print(f"[fusionML] output_dir={out_dir} smoke={cfg.get('smoke')} resolution={cfg['resolution']}")

    inputs = load_all_inputs(cfg, require_target=True)
    cohort = inputs.cohort_ids
    _print(f"[fusionML] joint cohort: {len(cohort)} samples "
           f"(ARC n={inputs.cohort_report.n_arc}, BAC n={inputs.cohort_report.n_bac}, "
           f"missing_arc_assay={inputs.cohort_report.n_missing_arc_assay}, "
           f"missing_bac_assay={inputs.cohort_report.n_missing_bac_assay}, "
           f"missing_target={inputs.cohort_report.n_missing_target})")

    families = applicable_families(cfg, inputs)
    _print(f"[fusionML] applicable model families: {families}")

    est_fits = 0
    for fam in families:
        spec = FAMILY_SPECS[fam]
        if fam == "intercept_only":
            est_fits += cfg["cv"]["outer_folds"]
            continue
        n_cand = len(generate_candidates(spec, cfg["search"], seed=0))
        est_fits += cfg["cv"]["outer_folds"] * (cfg["cv"]["inner_folds"] * n_cand + 1)
    _print(f"[fusionML] estimated total constrained-fit count (rough): {est_fits}")

    site_map_full = {sid: inputs.metadata.loc[sid, "site_id"] for sid in cohort}
    outer_folds = make_site_grouped_folds(site_map_full, cfg["cv"]["outer_folds"], cfg["seed"])

    fold_manifest_rows = []
    for o in range(outer_folds.k):
        for s in outer_folds.sites_in_fold(o):
            fold_manifest_rows.append({"stage": "outer", "fold": o, "site_id": s, "role": "test"})
        for s in outer_folds.train_sites_for_outer_fold(o):
            fold_manifest_rows.append({"stage": "outer", "fold": o, "site_id": s, "role": "train"})
    write_df(pd.DataFrame(fold_manifest_rows), os.path.join(out_dir, "fold_manifest.csv"))

    all_oof_rows = []
    all_search_rows = []
    all_edge_rows = []
    all_coef_rows = []
    all_balance_rows = []
    all_interaction_coef_rows = []
    results_rows = []

    for family_name in families:
        spec = FAMILY_SPECS[family_name]
        fam_res = family_resolution(spec, cfg["resolution"])
        _print(f"[fusionML] === family: {family_name} (resolution={fam_res}) ===")
        family_oof = []

        for o in range(outer_folds.k):
            train_sites = outer_folds.train_sites_for_outer_fold(o)
            test_sites = outer_folds.sites_in_fold(o)
            train_ids = [s for s in sample_ids_for_sites(site_map_full, train_sites) if s in cohort]
            test_ids = [s for s in sample_ids_for_sites(site_map_full, test_sites) if s in cohort]

            if family_name == "intercept_only":
                y_tr = inputs.metadata.loc[train_ids, "target"].to_numpy(dtype=float)
                w_tr = site_balanced_weights([site_map_full[s] for s in train_ids])
                mu = float(np.sum(w_tr * y_tr))  # for classification this is already a probability (site-balanced prevalence)
                for sid in test_ids:
                    row = {"sample_id": sid, "site_id": site_map_full[sid],
                           "target": inputs.metadata.loc[sid, "target"], "prediction": mu,
                           "outer_fold": o, "family": family_name, "scorable": True, "reason": ""}
                    if cfg["task"] == "classification":
                        row["predicted_class"] = 1.0 if mu >= 0.5 else 0.0
                    family_oof.append(row)
                continue

            try:
                selected, search_df, info = run_inner_search(family_name, spec, train_ids, inputs, fam_res, cfg, o)
            except RuntimeError as e:
                _print(f"[fusionML] WARNING family={family_name} outer_fold={o}: {e}")
                for sid in test_ids:
                    family_oof.append({"sample_id": sid, "site_id": site_map_full[sid],
                                        "target": inputs.metadata.loc[sid, "target"], "prediction": np.nan,
                                        "outer_fold": o, "family": family_name, "scorable": False,
                                        "reason": f"inner search failed for this outer fold: {e}"})
                continue
            search_df["family"] = family_name
            search_df["outer_fold"] = o
            all_search_rows.append(search_df)

            try:
                pf = fit_partition(train_ids, spec, selected, inputs, fam_res, cfg,
                                    seed=derive_seed(cfg["seed"], "refit", family_name, o))
            except (InsufficientFeaturesError, SolverFailure) as e:
                _print(f"[fusionML] WARNING family={family_name} outer_fold={o}: outer-training refit failed: {e}")
                for sid in test_ids:
                    family_oof.append({"sample_id": sid, "site_id": site_map_full[sid],
                                        "target": inputs.metadata.loc[sid, "target"], "prediction": np.nan,
                                        "outer_fold": o, "family": family_name, "scorable": False,
                                        "reason": f"outer-training refit failed: {e}"})
                continue
            if cfg["task"] == "classification":
                # item 14's classification threshold rule: freeze the
                # threshold from the SELECTED candidate's pooled inner-CV
                # OOF probabilities, before this one-time outer prediction.
                inner_oof = info["selected_inner_oof"]
                pf.classification_threshold = select_threshold_via_inner_oof_f2(
                    inner_oof["target"], inner_oof["prediction"], inner_oof["site_id"],
                    grid=cfg.get("classification", "threshold_grid"),
                )
            preds = predict_with_partition_fit(pf, test_ids, inputs, fam_res)
            for sid in test_ids:
                row = {
                    "sample_id": sid, "site_id": site_map_full[sid],
                    "target": inputs.metadata.loc[sid, "target"], "prediction": preds.loc[sid, "prediction"],
                    "outer_fold": o, "family": family_name,
                    "scorable": bool(preds.loc[sid, "scorable"]), "reason": preds.loc[sid, "reason"],
                }
                if "predicted_class" in preds.columns:
                    row["predicted_class"] = preds.loc[sid, "predicted_class"]
                family_oof.append(row)

            if len(pf.mixed_edges_a):
                ea = pf.mixed_edges_a.copy(); ea["domain"] = "ARC"; ea["family"] = family_name; ea["outer_fold"] = o
                all_edge_rows.append(ea)
            if len(pf.mixed_edges_b):
                eb = pf.mixed_edges_b.copy(); eb["domain"] = "BAC"; eb["family"] = family_name; eb["outer_fold"] = o
                all_edge_rows.append(eb)

            if pf.interaction_transform is not None:
                bal_df, coef_df = interaction_artifacts_from_pf(pf, cfg["compression"]["coefficient_tolerance"])
                bal_df["family"] = family_name; bal_df["outer_fold"] = o
                coef_df["family"] = family_name; coef_df["outer_fold"] = o
                all_balance_rows.append(bal_df)
                all_interaction_coef_rows.append(coef_df)

            tol = cfg["compression"]["coefficient_tolerance"]
            for dname, feats, beta, edges in (("ARC", pf.retained_a, pf.fit_result.beta_a, pf.edges_a_idx),
                                               ("BAC", pf.retained_b, pf.fit_result.beta_b, pf.edges_b_idx)):
                gres = extract_groups(dname, feats, beta, edges, tol)
                for fid, gid in gres.group_of_feature.items():
                    all_coef_rows.append({
                        "family": family_name, "outer_fold": o, "domain": dname, "feature_id": fid,
                        "beta_standardized": float(beta[feats.index(fid)]),
                        "group_id": gid, "group_mean_beta": gres.group_mean_beta[gid],
                        "residual": gres.residuals[fid], "active": abs(beta[feats.index(fid)]) > tol,
                    })

        if not family_oof:
            _print(f"[fusionML] family {family_name}: no scorable outer-fold predictions; skipping from results.")
            continue

        oof_df = pd.DataFrame(family_oof)
        all_oof_rows.append(oof_df)
        scorable_df = oof_df[oof_df["scorable"]]
        if cfg["task"] == "classification":
            cm = compute_classification_metrics(
                scorable_df["target"].to_numpy(), scorable_df["prediction"].to_numpy(),
                scorable_df["site_id"].tolist(), scorable_df["predicted_class"].to_numpy(),
            )
            results_rows.append({
                "family": family_name, "n": cm.n, "n_sites": cm.n_sites, "n_positive": cm.n_positive,
                "coverage": coverage_frac(len(scorable_df), len(oof_df)),
                "pooled_auc_roc": cm.pooled_auc_roc, "pooled_auc_pr": cm.pooled_auc_pr, "pooled_brier": cm.pooled_brier,
                "site_balanced_log_loss": cm.site_balanced_log_loss,
                "precision": cm.precision, "recall": cm.recall, "f1": cm.f1, "f2": cm.f2,
                "mcc": cm.mcc, "accuracy": cm.accuracy,
            })
            _print(f"[fusionML] {family_name}: auc_roc={cm.pooled_auc_roc:.4f} mcc={cm.mcc:.4f} f2={cm.f2:.4f} "
                   f"n_positive={cm.n_positive}/{cm.n} coverage={coverage_frac(len(scorable_df), len(oof_df)):.2f}")
        else:
            m = compute_metrics(scorable_df["target"].to_numpy(), scorable_df["prediction"].to_numpy(), scorable_df["site_id"].tolist())
            results_rows.append({
                "family": family_name, "n": m.n, "n_sites": m.n_sites,
                "coverage": coverage_frac(len(scorable_df), len(oof_df)),
                "pooled_rmse": m.pooled_rmse, "pooled_mae": m.pooled_mae, "pooled_r2": m.pooled_r2,
                "site_balanced_mse": m.site_balanced_mse, "site_balanced_rmse": m.site_balanced_rmse,
                "site_balanced_mae": m.site_balanced_mae,
            })
            _print(f"[fusionML] {family_name}: site_balanced_rmse={m.site_balanced_rmse:.4f} pooled_r2={m.pooled_r2:.4f} coverage={coverage_frac(len(scorable_df), len(oof_df)):.2f}")

    results_df = pd.DataFrame(results_rows)
    write_df(results_df, os.path.join(out_dir, "results.csv"))
    if all_oof_rows:
        oof_all = pd.concat(all_oof_rows, ignore_index=True)
        write_df(oof_all, os.path.join(out_dir, "oof_predictions.csv"))
        primary_or_fallback = oof_all[oof_all["family"] == PRIMARY_FAMILY] if PRIMARY_FAMILY in oof_all["family"].values else oof_all
        if cfg["task"] == "classification":
            plot_roc_curve(primary_or_fallback, os.path.join(out_dir, "figures", "roc_curve.png"), synthetic=cfg.get("smoke", False))
        else:
            plot_oof_predicted_vs_observed(primary_or_fallback, os.path.join(out_dir, "figures", "oof_predicted_vs_observed.png"),
                                            synthetic=cfg.get("smoke", False))
            plot_residuals_by_site(primary_or_fallback, os.path.join(out_dir, "figures", "residuals_by_site.png"),
                                    synthetic=cfg.get("smoke", False))
    if all_search_rows:
        search_all = pd.concat(all_search_rows, ignore_index=True)
        write_df(search_all, os.path.join(out_dir, "search_results.csv"))
        valid_search = search_all[search_all["valid"]]
        if len(valid_search):
            score_label = "Inner site-balanced log-loss" if cfg["task"] == "classification" else "Inner site-balanced MSE"
            plot_compression_vs_error(valid_search, os.path.join(out_dir, "figures", "compression_vs_error.png"),
                                       synthetic=cfg.get("smoke", False), score_label=score_label)
    if all_edge_rows:
        edges_all = pd.concat(all_edge_rows, ignore_index=True)
        write_df(edges_all, os.path.join(out_dir, "graph_edges.csv"))
    if all_coef_rows:
        write_df(pd.DataFrame(all_coef_rows), os.path.join(out_dir, "coefficients_and_groups.csv"))
    if all_balance_rows:
        # Per-outer-fold balances/products (one row set per refit); the
        # final deployable model's own balance_definitions.csv/
        # interaction_coefficients.csv (written by train_final_deployable_model
        # below) are a separate, singular artifact matching final_model.pkl.
        write_df(pd.concat(all_balance_rows, ignore_index=True), os.path.join(out_dir, "balance_definitions_outer_folds.csv"))
    if all_interaction_coef_rows:
        write_df(pd.concat(all_interaction_coef_rows, ignore_index=True), os.path.join(out_dir, "interaction_coefficients_outer_folds.csv"))

    # Bootstrap paired comparison: primary family vs. intercept-only baseline.
    paired = {}
    if all_oof_rows:
        oof_all = pd.concat(all_oof_rows, ignore_index=True)
        if PRIMARY_FAMILY in oof_all["family"].values and "intercept_only" in oof_all["family"].values:
            a = oof_all[(oof_all["family"] == PRIMARY_FAMILY) & (oof_all["scorable"])]
            b = oof_all[(oof_all["family"] == "intercept_only") & (oof_all["scorable"])]
            paired_fn = bootstrap_paired_site_log_loss_differences if cfg["task"] == "classification" else bootstrap_paired_site_differences
            paired["primary_vs_intercept_only"] = paired_fn(a, b, B=1000, seed=cfg["seed"])
    write_json(paired, os.path.join(out_dir, "paired_comparisons.json"))

    final_model_info = train_final_deployable_model(cfg, inputs, families, out_dir)

    manifest = build_run_manifest(
        cfg.to_dict(), input_hashes={k: hash_file(cfg.resolved_input_path(k)) for k in
                                      ("arc_counts", "bac_counts", "metadata", "arc_tree", "bac_tree", "arc_taxonomy", "bac_taxonomy")},
        mode="train", run_id=run_id,
        extra={"cohort_report": dataclasses.asdict(inputs.cohort_report), "solver_environment": solver_environment_info(),
               "families_run": families, "final_model": final_model_info},
    )
    write_json(manifest, os.path.join(out_dir, "run_manifest.json"))
    _print(f"[fusionML] train complete. results -> {os.path.join(out_dir, 'results.csv')}")
    return {"results": results_df, "manifest": manifest, "final_model_info": final_model_info}


def train_final_deployable_model(cfg: Config, inputs: LoadedInputs, families: List[str], out_dir: str) -> dict:
    """item 14: 'The final deployable model is selected through a fresh
    grouped inner-style search on all eligible data and then refitted on
    all of it.' Uses the primary family if available, else the best
    available graph-based family, else the best ridge/EN baseline."""
    candidates_order = [INTERACTION_FAMILY, PRIMARY_FAMILY, "shared_hybrid", "phylogeny_only", "abundance_only",
                         "asv_elastic_net", "genus_elastic_net", "asv_ridge", "genus_ridge"]
    chosen = next((f for f in candidates_order if f in families), None)
    if chosen is None:
        return {"status": "skipped", "reason": "no eligible family for a final deployable model"}

    spec = FAMILY_SPECS[chosen]
    fam_res = family_resolution(spec, cfg["resolution"])
    cohort = inputs.cohort_ids
    site_map = {sid: inputs.metadata.loc[sid, "site_id"] for sid in cohort}
    try:
        selected, search_df, info = run_inner_search(chosen, spec, cohort, inputs, fam_res, cfg, outer_fold_idx="final")
    except RuntimeError as e:
        return {"status": "failed", "family": chosen, "reason": str(e)}

    pf = fit_partition(cohort, spec, selected, inputs, fam_res, cfg, seed=derive_seed(cfg["seed"], "final_refit", chosen))
    if cfg["task"] == "classification":
        # item 14: "For the final model use its full-data grouped OOF
        # training predictions to select the deployment threshold."
        inner_oof = info["selected_inner_oof"]
        pf.classification_threshold = select_threshold_via_inner_oof_f2(
            inner_oof["target"], inner_oof["prediction"], inner_oof["site_id"],
            grid=cfg.get("classification", "threshold_grid"),
        )
    target_scaler = pf.target_scaler
    beta_a_orig, b0_orig = target_scaler.inverse_coefficients(pf.fit_result.beta_a, pf.fit_result.b0)
    beta_b_orig, _ = target_scaler.inverse_coefficients(pf.fit_result.beta_b, pf.fit_result.b0)
    gamma_orig = np.zeros(0)
    if pf.fit_result.gamma.size > 0:
        gamma_orig, _ = target_scaler.inverse_coefficients(pf.fit_result.gamma, 0.0)

    base = selected.base if isinstance(selected, InteractionCandidate) else selected
    model = FusionModel(
        resolution=fam_res, domain="joint", feature_ids_a=pf.retained_a, feature_ids_b=pf.retained_b,
        preproc_a=pf.jp.arc, preproc_b=pf.jp.bac, pseudocount=pf.pseudocount, target_scaler=target_scaler,
        covariate_preproc=None, b0_star=pf.fit_result.b0, beta_a_star=pf.fit_result.beta_a, beta_b_star=pf.fit_result.beta_b,
        b0_orig=b0_orig, beta_a_orig=beta_a_orig, beta_b_orig=beta_b_orig,
        graph_edges_a=pf.mixed_edges_a, graph_edges_b=pf.mixed_edges_b,
        alpha_arc=base.alpha_arc, alpha_bac=base.alpha_bac,
        lambda_s=base.lambda_s, lambda_r=base.lambda_r, lambda_f=base.lambda_f,
        family=chosen, metadata={"n_train": len(pf.kept_train_ids), "note": "trained on all eligible joint-cohort data"},
        task=cfg["task"], classification_threshold=pf.classification_threshold,
        interaction_transform=pf.interaction_transform, gamma_star=pf.fit_result.gamma, gamma_orig=gamma_orig,
    )
    model_path = os.path.join(out_dir, "final_model.pkl")
    save_model(model, model_path)

    result = {"status": "ok", "family": chosen, "model_path": model_path,
              "selected_candidate": dataclasses.asdict(selected), "n_train": len(pf.kept_train_ids)}
    if pf.interaction_transform is not None:
        balance_df, coef_df = interaction_artifacts_from_pf(pf, cfg["compression"]["coefficient_tolerance"])
        write_df(balance_df, os.path.join(out_dir, "balance_definitions.csv"))
        write_df(coef_df, os.path.join(out_dir, "interaction_coefficients.csv"))
        result["n_interaction_products"] = pf.interaction_transform.n_products
    return result


def run_predict(config_path: str, model_path: str, output_path: str) -> pd.DataFrame:
    """Loads counts at whatever raw (ASV) resolution the config's inputs
    provide, then applies the SAME resolution-dependent table preparation
    (e.g. genus aggregation) the saved model was trained with, before
    calling its frozen predict(). Never refits any preprocessing."""
    cfg = Config.from_yaml(config_path)
    model = load_model(model_path)
    orientation = cfg.get("inputs", "orientation")
    relative_mode = bool(cfg.get("preprocessing", "relative_input_mode"))
    counts_a_asv, _ = load_count_table(cfg.resolved_input_path("arc_counts"), "ARC", orientation, relative_mode)
    counts_b_asv, _ = load_count_table(cfg.resolved_input_path("bac_counts"), "BAC", orientation, relative_mode)

    if model.resolution == "genus":
        tax_a_path = cfg.resolved_input_path("arc_taxonomy")
        tax_b_path = cfg.resolved_input_path("bac_taxonomy")
        if not tax_a_path or not tax_b_path:
            raise ValueError(
                "the saved model was trained at genus resolution, but this config has no "
                "arc_taxonomy/bac_taxonomy configured to re-aggregate new ASV counts to genus."
            )
        taxonomy_a, _ = load_taxonomy(tax_a_path, "ARC")
        taxonomy_b, _ = load_taxonomy(tax_b_path, "BAC")
        counts_a, _, _ = aggregate_to_genus(counts_a_asv, taxonomy_a, "ARC")
        counts_b, _, _ = aggregate_to_genus(counts_b_asv, taxonomy_b, "BAC")
    elif model.resolution == "asv":
        counts_a, counts_b = counts_a_asv, counts_b_asv
    else:
        raise ValueError(f"predict does not know how to prepare resolution={model.resolution!r} inputs")

    preds = model.predict(counts_a, counts_b)
    write_df(preds, output_path)
    return preds

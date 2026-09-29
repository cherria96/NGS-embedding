"""Config schema and loader for the fusionML pipeline.

Mirrors the YAML schema in design-spec item 16 exactly for all keys listed
there. A few keys are added beyond that schema (documented inline with
"EXTENSION:") because the spec explicitly allows adding "resource limits,
solver-specific options, model-family selection and run identifiers as
needed" (item 16) and because item 9 requires rejecting relative-abundance
tables unless an explicit alternative mode is configured.
"""
from __future__ import annotations

import copy
import dataclasses
import os
from typing import Any, Optional

import yaml

REQUIRED_TOP_KEYS = [
    "seed", "task", "resolution", "inputs", "columns", "preprocessing",
    "graph", "cv", "search", "solver", "compression", "optional",
    "interaction", "classification", "output_dir",
]


def _default_inputs() -> dict:
    return {
        "arc_counts": None,
        "bac_counts": None,
        "metadata": None,
        "arc_tree": None,
        "bac_tree": None,
        "arc_taxonomy": None,
        "bac_taxonomy": None,
        "orientation": "samples_by_features",
        # EXTENSION: item 9's input contract lists an "Optional exploratory
        # block list" input that the item-16 YAML schema omits a key for.
        "exploratory_block_list": None,
        # EXTENSION (project schema, item 9's later update): supplied
        # rank-consistent trees for resolution in {genus,family,order,class},
        # each with a co-located ASV->reduced-feature mapping.csv.
        "rank_trees": {},
    }


def _default_columns() -> dict:
    return {
        "sample_id": "sample_id",
        "site_id": "site_id",
        "target": None,
        "covariates": [],
    }


def _default_preprocessing() -> dict:
    return {
        "detection_count": 5,
        "min_detected_samples": 5,
        "min_sample_prevalence": 0.10,
        "min_detected_sites": 3,
        "min_site_prevalence": 0.10,
        "pseudocount": 0.5,
        "sensitivity_pseudocounts": [1.0],
        "standardize_target": True,
        "standardize_microbial_columns": False,
        # EXTENSION: item 9 forbids silently treating percentage tables as
        # counts; this flag is the "explicit alternative preprocessing mode"
        # it asks for. False => relative-looking tables raise a schema error.
        "relative_input_mode": False,
        # EXTENSION: item 10 says a missing retained column may be filled
        # with zero "only if the input contract explicitly establishes"
        # that; false => a schema error instead of a silent zero-fill.
        "missing_retained_feature_means_zero": False,
    }


def _default_graph() -> dict:
    return {
        "k_phylogeny": 5,
        "k_abundance": 5,
        "min_joint_samples": 10,
        "min_joint_sites": 5,
        "empirical_score": "max_all_and_joint_lrv",
        "bootstrap_sites": 500,
        "support_threshold": 0.80,
        "approximate_bootstrap": False,
        "approximate_candidate_k": 20,
    }


def _default_cv() -> dict:
    return {
        "outer_folds": 6,
        "inner_folds": 4,
        "selection_metric": "site_balanced_mse",
        "selection_rule": "one_standard_error",
        "required_validation_coverage": 1.0,
        # EXTENSION: whitelist of design-spec item-14-table model family
        # names to run; null means "every family applicable to the
        # configured resolution and available inputs" (the default, full
        # comparison). Used by smoke configs to bound runtime.
        "families": None,
    }


def _default_search() -> dict:
    return {
        "candidate_budget_per_family": 48,
        "lambda_s": [0, 0.001, 0.01, 0.1],
        "lambda_r": [0.001, 0.01, 0.1, 1],
        "lambda_f": [0, 0.1, 1, 10, 100, 1000],
        "alpha_arc": [0, 0.5, 1],
        "alpha_bac": [0, 0.5, 1],
        "exhaustive": False,
    }


def _default_solver() -> dict:
    return {
        "preferred": "CLARABEL",
        "fallback": "OSQP",  # regression only -- OSQP cannot solve cp.logistic()'s exponential cone.
        "classification_fallback": "SCS",
        "equality_tolerance": 1.0e-6,
        # EXTENSION: retry/runtime bookkeeping knobs (item 13's "documented
        # solver/tolerance policy").
        "max_retries": 3,
        "time_limit_seconds": 120,
    }


def _default_compression() -> dict:
    return {
        "coefficient_tolerance": 1.0e-6,
        "keep_residual_coefficients": True,
    }


def _default_optional() -> dict:
    return {
        "b12_comparator": False,
        "cross_domain_interactions": False,
    }


def _default_interaction() -> dict:
    """arc-bac-interactions.md item 16's `interaction:` block. Disabled
    (enabled=False) unless a config explicitly turns it (and the matching
    optional.cross_domain_interactions flag) on -- see the contradiction
    check in validate_config_schema."""
    return {
        "enabled": False,
        "family": "domain_adaptive_interaction",
        "retain_additive_reference": True,
        "balance_source": "tree_splits",  # tree_splits | external_balances (external not yet implemented)
        "external_balance_file": None,
        "max_balances_arc": 2,
        "max_balances_bac": 2,
        "max_products": 4,
        "balance_selection": "site_weighted_variance_greedy",
        "sign_orientation": "lexicographic_positive_side",
        # EXTENSION: item 10.2's "raw threshold" for the group-detection gate
        # is preprocessing.detection_count itself (validated equal below);
        # kept here too so the interaction block stays self-documenting.
        "group_detection_count": 5,
        "min_joint_group_samples": 10,
        "min_joint_group_sites": 5,
        "max_abs_balance_correlation": 0.95,
        "loading_rank_absolute_tolerance": 1.0e-10,
        "product_rank_relative_tolerance": 1.0e-10,
        "min_variance": 1.0e-12,
        "center_scale_balances": True,
        "center_scale_products": True,
        "scaling_weights": "equal_site",
        "require_cross_domain_correlation": False,
        "enforce_strong_heredity": False,
        "fit_jointly_with_main_effects": True,
        "include_off_candidate": True,
        "lambda_i_l1": [0, 0.01, 0.1],
        "lambda_i_l2": [0.01, 0.1, 1],
        "search": "full_cross_with_shared_base_candidates",
        "no_eligible_products": "mark_on_candidate_unavailable",
    }


def _default_classification() -> dict:
    """item 16's `classification:` block (item 14's threshold-selection
    override + item 7's evaluation conventions)."""
    return {
        "selection_metric": "site_balanced_log_loss",
        "standardize_target": False,
        "class_weighting": "none",
        "single_class_training": "fail_primary_fit",
        "reference_threshold": 0.5,
        "threshold_selection": "inner_oof_site_weighted_f2",
        "threshold_grid": [round(0.05 * i, 2) for i in range(1, 20)],
        "threshold_tie_break": "larger_threshold",
        "probability_clip_for_metrics": 1.0e-12,
    }


def default_config() -> dict:
    return {
        "seed": 20260917,
        "task": "regression",
        "resolution": "asv",
        "inputs": _default_inputs(),
        "columns": _default_columns(),
        "preprocessing": _default_preprocessing(),
        "graph": _default_graph(),
        "cv": _default_cv(),
        "search": _default_search(),
        "solver": _default_solver(),
        "compression": _default_compression(),
        "optional": _default_optional(),
        "interaction": _default_interaction(),
        "classification": _default_classification(),
        "output_dir": "results/ad_microbiome",
        # EXTENSION: a run identifier used to namespace artifacts; if left
        # null the pipeline derives one from mode + timestamp.
        "run_id": None,
        # EXTENSION: smoke mode toggle per item 16 ("Default smoke mode may
        # reduce bootstrap repetitions to 20, use three outer/two inner
        # folds, limit candidates and use a small clearly labeled synthetic
        # dataset"). When true, artifacts are written under a smoke_/
        # subdirectory and a warning banner is attached to every report.
        "smoke": False,
    }


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_raw_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(default_config(), user_cfg)


@dataclasses.dataclass
class ConfigIssue:
    level: str  # "error" or "warning"
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.field}: {self.message}"


class Config:
    """Thin typed wrapper around the merged config dict with path resolution."""

    def __init__(self, raw: dict, source_path: Optional[str] = None):
        self.raw = raw
        self.source_path = source_path

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        raw = load_raw_config(path)
        return cls(raw, source_path=path)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys, default=None):
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def is_project_schema(self) -> bool:
        """True for the "project config" schema (ad-microbial-config.yaml):
        it carries project_root and resolves relative paths against it
        (invocation cwd if project_root itself is relative), per item 16 --
        DIFFERENT from the item-16-example schema's config-file-relative
        resolution, which older configs (e.g. configs/real_smoke.yaml) use."""
        return "project_root" in self.raw

    def _base_dir(self) -> str:
        if self.is_project_schema:
            root = self.raw["project_root"]
            return root if os.path.isabs(root) else os.path.normpath(os.path.join(os.getcwd(), root))
        if self.source_path is not None:
            return os.path.dirname(os.path.abspath(self.source_path))
        return os.getcwd()

    def resolved_path(self, value: Optional[str]) -> Optional[str]:
        """Resolve an arbitrary relative path against this config's base
        directory (project_root, or the config file's own directory for the
        older schema)."""
        if value is None:
            return None
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(self._base_dir(), value))

    def resolved_input_path(self, key: str) -> Optional[str]:
        return self.resolved_path(self.get("inputs", key))

    def resolved_output_dir(self) -> str:
        out = self["output_dir"]
        base = out
        if self.get("smoke"):
            base = os.path.join(out, "smoke")
        return self.resolved_path(base) if not os.path.isabs(base) else base

    def to_dict(self) -> dict:
        return copy.deepcopy(self.raw)


def validate_config_schema(cfg: Config) -> list:
    """Structural validation only (keys/types/ranges). Does not touch data files."""
    issues = []
    raw = cfg.raw
    for key in REQUIRED_TOP_KEYS:
        if key not in raw:
            issues.append(ConfigIssue("error", key, "missing required top-level key"))

    task = raw.get("task")
    if task not in ("regression", "classification"):
        issues.append(ConfigIssue(
            "error", "task",
            f"task={task!r} is not implemented; must be 'regression' (the required initial "
            "task per design-spec item 9) or 'classification' (item 7's explicit logistic-loss "
            "extension: 'The same feature construction can use logistic loss for an "
            "operational-state classification task')."
        ))

    resolution = raw.get("resolution")
    valid_resolutions = ("asv", "genus", "family", "order", "class")
    if resolution not in valid_resolutions:
        issues.append(ConfigIssue("error", "resolution", f"resolution must be one of {valid_resolutions}, got {resolution!r}"))
    if resolution in ("family", "order", "class") and resolution not in (raw.get("inputs", {}).get("rank_trees") or {}):
        issues.append(ConfigIssue(
            "error", "inputs.rank_trees",
            f"resolution={resolution!r} requires inputs.rank_trees.{resolution}.{{arc,bac}} tree paths "
            "(there is no naive/tree-free aggregation at this rank).",
        ))

    orientation = raw.get("inputs", {}).get("orientation")
    if orientation not in ("samples_by_features", "features_by_samples"):
        issues.append(ConfigIssue(
            "error", "inputs.orientation",
            f"orientation must be 'samples_by_features' or 'features_by_samples', got {orientation!r}",
        ))

    columns = raw.get("columns", {})
    if not columns.get("sample_id"):
        issues.append(ConfigIssue("error", "columns.sample_id", "sample_id column name is required"))
    if not columns.get("site_id"):
        issues.append(ConfigIssue("error", "columns.site_id", "site_id column name is required"))
    if not columns.get("target"):
        issues.append(ConfigIssue(
            "error", "columns.target",
            "target column is not configured. The design spec requires one explicitly "
            "configured, user-approved continuous target column per run (item 9); "
            "it must not be invented by the pipeline.",
        ))

    pp = raw.get("preprocessing", {})
    if pp.get("pseudocount", 0) <= 0:
        issues.append(ConfigIssue("error", "preprocessing.pseudocount", "pseudocount must be > 0"))

    solver = raw.get("solver", {})
    if solver.get("equality_tolerance", 0) <= 0:
        issues.append(ConfigIssue("error", "solver.equality_tolerance", "equality_tolerance must be > 0"))

    search = raw.get("search", {})
    lr = search.get("lambda_r", [])
    if not lr or min(lr) <= 0:
        issues.append(ConfigIssue(
            "error", "search.lambda_r",
            "lambda_r grid must be nonempty with strictly positive values "
            "(item 13: 'Keep lambda_R strictly positive in the default search').",
        ))

    cv = raw.get("cv", {})
    if cv.get("outer_folds", 0) < 2:
        issues.append(ConfigIssue("error", "cv.outer_folds", "outer_folds must be >= 2"))
    if cv.get("inner_folds", 0) < 2:
        issues.append(ConfigIssue("error", "cv.inner_folds", "inner_folds must be >= 2"))

    optional = raw.get("optional", {})
    interaction = raw.get("interaction", {})
    cdi = bool(optional.get("cross_domain_interactions"))
    interaction_enabled = bool(interaction.get("enabled"))
    if cdi != interaction_enabled:
        issues.append(ConfigIssue(
            "error", "optional.cross_domain_interactions",
            f"optional.cross_domain_interactions={cdi} but interaction.enabled={interaction_enabled}; "
            "these two flags must agree (arc-bac-interactions.md item 16's config-migration note: "
            "'Reject contradictory interaction enable flags instead of guessing').",
        ))
    if interaction_enabled:
        pp = raw.get("preprocessing", {})
        if interaction.get("group_detection_count") != pp.get("detection_count"):
            issues.append(ConfigIssue(
                "error", "interaction.group_detection_count",
                f"interaction.group_detection_count={interaction.get('group_detection_count')!r} must equal "
                f"preprocessing.detection_count={pp.get('detection_count')!r} -- item 10.2's group-detection "
                "gate uses 'the raw threshold', i.e. the same detection count already used to build the "
                "detection tables; the pipeline does not compute a second, independently-thresholded table.",
            ))
        if interaction.get("balance_source") == "external_balances":
            issues.append(ConfigIssue(
                "error", "interaction.balance_source",
                "balance_source=external_balances (item 10.6) is not implemented; only tree_splits is wired "
                "into interactions.py's fit_interaction_transform.",
            ))
        elif interaction.get("balance_source") != "tree_splits":
            issues.append(ConfigIssue(
                "error", "interaction.balance_source",
                f"balance_source must be 'tree_splits', got {interaction.get('balance_source')!r}",
            ))
        for l2 in interaction.get("lambda_i_l2", []):
            if l2 <= 0:
                issues.append(ConfigIssue(
                    "error", "interaction.lambda_i_l2",
                    "lambda_i_l2 entries must be strictly positive (item 13: interaction-on candidates "
                    "require lambda_I,2 > 0; an on-candidate with lambda_i2<=0 is not a valid setting).",
                ))
                break
    if optional.get("b12_comparator") and not cfg.get("inputs", "exploratory_block_list"):
        issues.append(ConfigIssue(
            "error", "optional.b12_comparator",
            "b12_comparator requires inputs.exploratory_block_list (explicit genus IDs for "
            "the two groups; item 9/15 forbid guessed memberships).",
        ))

    return issues

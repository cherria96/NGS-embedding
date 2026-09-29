"""Generate Experiment 1 tables, figures, and report using copied outputs only.

No fusionML imports, raw sequencing data, external network, or model refitting.
"""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))


import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fusionmlv2-exp1-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fusionmlv2-exp1-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, roc_auc_score

EXPERIMENT = Path(__file__).resolve().parents[1]
TARGETS = ["warning_ammonia_toxicity", "warning_acid_base_balance"]
TITLES = dict(zip(TARGETS, ["Ammonia toxicity", "Acid–base balance"]))
GROUP_MODELS = ["genus_ridge", "genus_elastic_net", "phylogeny_only", "abundance_only",
                "shared_hybrid", "domain_adaptive_hybrid", "domain_adaptive_interaction",
                "adaptive_expanded"]
BASELINES = ["elasticnet_RA_matched", "xgboost_RA_matched", "elasticnet_RA_flat", "xgboost_RA_flat"]
MODELS = GROUP_MODELS + BASELINES
LABELS = dict(zip(MODELS, ["Log-contrast ridge", "Log-contrast elastic net", "Phylogeny fusion",
    "Abundance fusion", "Shared hybrid", "Domain-adaptive hybrid", "ARC–BAC interaction",
    "Expanded adaptive grid", "Elastic net: matched RA", "XGBoost: matched RA",
    "Elastic net: flat genus RA", "XGBoost: flat genus RA"]))
METRICS = ["auc", "ap", "logloss", "brier", "mcc", "f2"]
PRED_KEYS = ["target", "repeat", "model", "selection"]
FIT_KEYS = PRED_KEYS + ["outer_fold"]
COLORS = {"phylogeny": "#2563a6", "abundance": "#d16b19", "shared": "#24816c"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_table(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / "tables" / name, index=False, float_format="%.12g")


def md(df: pd.DataFrame) -> str:
    def cell(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def metrics(d: pd.DataFrame) -> dict:
    y, p = d.y.to_numpy(), d.p.to_numpy()
    clipped = np.clip(p, 1e-6, 1 - 1e-6)  # Match the saved fusionML metric definition.
    loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    decision = p >= d.threshold.to_numpy()
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "logloss": pd.Series(loss).groupby(d.site_id.to_numpy()).mean().mean(),
            "brier": np.mean((y - p) ** 2), "mcc": matthews_corrcoef(y, decision),
            "f2": fbeta_score(y, decision, beta=2, zero_division=0),
            "n_scored": len(d), "n_sites": d.site_id.nunique(),
            "n_outer_folds": d.outer_fold.nunique()}


def verify_snapshot(data: Path) -> dict:
    manifest = json.loads((data / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        assert sha(data / entry["copy"]) == entry["sha256"], entry["copy"]
    old = json.loads((data / "investigation_20260921/learning_qc/manifest.json").read_text())
    historical_hash_checks = {}
    for name, expected in old["hashes"].items():
        if name.startswith("fusionML/") and (data / name.removeprefix("fusionML/")).exists():
            historical_hash_checks[name] = sha(data / name.removeprefix("fusionML/")) == expected
    return {"copied_files_verified": len(manifest["files"]),
            "matches_original_run_code_config_hashes": historical_hash_checks,
            "raw_workbook_not_copied_or_revalidated": True}


def qc_tables(base: Path, out: Path, checks: dict, bootstrap: int, seed: int):
    source = base / "learning_qc"
    rows = pd.read_csv(source / "rows.csv")
    pred = pd.read_csv(source / "predictions.csv")
    saved = pd.read_csv(source / "pooled_metrics.csv")
    splits = pd.read_csv(source / "splits.csv")
    rows, pred, saved, splits = [x[x.fraction.eq(1)].copy() for x in (rows, pred, saved, splits)]
    assert not rows.duplicated(FIT_KEYS).any()
    assert not pred.duplicated(PRED_KEYS + ["sample_id"]).any()
    assert rows.status.eq("ok").all() and rows.coverage.eq(1).all()
    assert rows.retained_taxa.gt(0).all()
    assert rows.n_active_groups.between(0, rows.retained_taxa).all()
    assert pred.p.between(0, 1).all() and pred.y.isin([0, 1]).all()
    assert pred.groupby("sample_id").site_id.nunique().eq(1).all()
    assert pred.groupby(["target", "sample_id"]).y.nunique().eq(1).all()
    assert set(pred.target) == set(TARGETS)
    assert set(pred.model) == set(MODELS)
    reference_ids = set(pred.sample_id)
    fold_site_checks = 0
    for key, d in splits.groupby(["target", "repeat", "outer_fold"]):
        train, test = d[d.role.eq("train")], d[d.role.eq("test")]
        assert set(train.site_id).isdisjoint(test.site_id), key
        assert set(train.sample_id).isdisjoint(test.sample_id), key
        subset = pred[(pred.target == key[0]) & (pred.repeat == key[1]) & (pred.outer_fold == key[2])]
        for _, pp in subset.groupby(["model", "selection"]):
            assert set(pp.sample_id) == set(test.sample_id), key
        fold_site_checks += 1
    scored = []
    for key, d in pred.groupby(PRED_KEYS):
        assert set(d.sample_id) == reference_ids
        assert d.outer_fold.nunique() == 3 and d.site_id.nunique() == 35
        scored.append({**dict(zip(PRED_KEYS, key)), **metrics(d)})
    scored = pd.DataFrame(scored)
    comparison = scored.merge(saved, on=PRED_KEYS, suffixes=("_new", "_saved"), validate="one_to_one")
    errors = {m: float((comparison[m + "_new"] - comparison[m + "_saved"]).abs().max()) for m in METRICS}
    assert max(errors.values()) < 1e-10, errors
    checks.update(qc_full_size_fit_rows=len(rows), qc_full_size_prediction_rows=len(pred),
                  qc_metric_max_absolute_differences=errors, outer_partitions_without_site_overlap=fold_site_checks,
                  samples=len(reference_ids), qc_site_groups=35)

    summaries = []
    for key, fits in rows.groupby(["target", "model", "selection"]):
        target, model, selection = key
        m = scored[(scored.target == target) & (scored.model == model) & (scored.selection == selection)]
        assert len(fits) == 6 and len(m) == 2
        grouped = model in GROUP_MODELS
        record = dict(target=target, model=model, selection_tag=selection,
            selection_rule="minimum inner log-loss" if model in BASELINES or selection == "min_loss" else "one-standard-error, fewer groups",
            representation="flat genus RA" if model.endswith("flat") else "matched retained RA" if model in BASELINES else "log-contrast",
            complexity_kind="active coefficient groups" if grouped else "nonzero coefficients" if model.startswith("elasticnet") else "taxa with nonzero tree importance",
            n_fits=len(fits), n_repeats=len(m), n_samples=len(reference_ids), n_sites=35,
            retained_features_mean=fits.retained_taxa.mean(), active_taxa_mean=fits.n_active_taxa.mean(),
            active_groups_mean=fits.n_active_groups.mean() if grouped else np.nan,
            active_groups_min=fits.n_active_groups.min() if grouped else np.nan,
            active_groups_max=fits.n_active_groups.max() if grouped else np.nan,
            structural_compression_pct=100 * (1 - fits.n_active_groups / fits.retained_taxa).mean() if grouped else np.nan,
            active_products_mean=fits.n_active_products.mean(),
            train_site_logloss_mean=fits.train_logloss.mean())
        for metric in METRICS:
            record[metric] = m[metric].mean()
            record[metric + "_repeat_min"] = m[metric].min()
            record[metric + "_repeat_max"] = m[metric].max()
        summaries.append(record)
    all_summary = pd.DataFrame(summaries)
    primary = all_summary[all_summary.selection_tag.eq("one_se")].copy()
    sensitivity = all_summary[all_summary.model.isin(GROUP_MODELS)].pivot(
        index=["target", "model"], columns="selection_tag", values=["active_groups_mean", "logloss", "auc"])
    sensitivity.columns = [f"{metric}_{rule}" for metric, rule in sensitivity.columns]
    sensitivity = sensitivity.reset_index()
    sensitivity["delta_groups_min_loss_minus_one_se"] = sensitivity.active_groups_mean_min_loss - sensitivity.active_groups_mean_one_se
    sensitivity["delta_logloss_min_loss_minus_one_se"] = sensitivity.logloss_min_loss - sensitivity.logloss_one_se
    save_table(sensitivity, out, "qc_selection_sensitivity.csv")
    # Do not treat ML source n_active_groups as actual microbial groups.
    assert primary[primary.model.isin(BASELINES)].active_groups_mean.isna().all()
    primary["observed_nondominated_logcontrast"] = pd.NA
    for target in TARGETS:
        d = primary[(primary.target == target) & primary.model.isin(GROUP_MODELS)]
        for idx, row in d.iterrows():
            smaller = d.active_groups_mean <= row.active_groups_mean + 1e-12
            better = d.logloss <= row.logloss + 1e-12
            strict = (d.active_groups_mean < row.active_groups_mean - 1e-12) | (d.logloss < row.logloss - 1e-12)
            primary.loc[idx, "observed_nondominated_logcontrast"] = not (smaller & better & strict).any()

    existing = pd.read_csv(base / "qc_groups_comparison.csv")
    matched = primary.merge(existing, on=["target", "model"], suffixes=("_new", "_saved"), validate="one_to_one")
    assert max((matched[m + "_new"] - matched[m + "_saved"]).abs().max() for m in ["auc", "ap", "logloss", "mcc", "f2"]) <= 5.0001e-5

    # Pair models by site, averaging loss across repeats before resampling sites.
    pp = pred[pred.selection.eq("one_se")].copy()
    clipped = np.clip(pp.p.to_numpy(), 1e-6, 1 - 1e-6)
    pp["loss"] = -(pp.y.to_numpy() * np.log(clipped) + (1 - pp.y.to_numpy()) * np.log1p(-clipped))
    site_losses = pp.groupby(["target", "model", "site_id"], as_index=False).loss.mean()
    boot_rows = []
    rng = np.random.default_rng(seed)
    for target in TARGETS:
        matrix = site_losses[site_losses.target.eq(target)].pivot(index="site_id", columns="model", values="loss")
        assert matrix.notna().all().all()
        indices = rng.integers(0, len(matrix), size=(bootstrap, len(matrix)))
        for model in MODELS:
            delta = (matrix[model] - matrix.genus_ridge).to_numpy()
            samples = delta[indices].mean(axis=1)
            boot_rows.append(dict(target=target, model=model, reference="genus_ridge", n_sites=len(delta),
                delta_logloss_vs_ridge=delta.mean(), ci_low=np.quantile(samples, .025), ci_high=np.quantile(samples, .975),
                bootstrap_resamples=bootstrap, seed=seed, uncertainty="paired site bootstrap of fixed OOF losses; no refitting"))
    paired = pd.DataFrame(boot_rows)
    primary = primary.merge(paired[["target", "model", "delta_logloss_vs_ridge", "ci_low", "ci_high"]], on=["target", "model"], validate="one_to_one")
    checks["paired_loss_delta_matches_pooled_repeat_mean"] = True
    for target in TARGETS:
        d = primary[primary.target.eq(target)].set_index("model")
        assert np.allclose(d.logloss - d.loc["genus_ridge", "logloss"], d.delta_logloss_vs_ridge, atol=1e-12)
    for frame, name in [(rows, "qc_full_size_fits.csv"), (scored, "qc_repeat_metrics.csv"),
                        (primary, "qc_primary_comparison.csv"), (all_summary, "qc_all_selection_rules.csv"),
                        (site_losses, "qc_fixed_oof_site_losses.csv"), (paired, "qc_paired_loss_vs_ridge.csv")]:
        save_table(frame, out, name)
    return primary, all_summary


def path_tables(base: Path, out: Path, checks: dict):
    raw = pd.read_csv(base / "regularization_paths.csv")
    assert len(raw) == 162 and not raw.duplicated(["target", "fold", "graph", "lambda_f"]).any()
    assert raw.groupby(["target", "graph", "lambda_f"]).size().eq(3).all()
    assert (raw.n_active_groups <= raw.retained_taxa).all()
    assert raw.n_active_products.eq(0).all()
    # Each graph's lambda=0 entry should be the identical unfused predictor.
    zero = raw[raw.lambda_f.eq(0)]
    assert zero.groupby(["target", "fold"]).test_logloss.agg(lambda x: x.max() - x.min()).max() < 1e-12
    raw["structural_compression_pct"] = 100 * (1 - raw.n_active_groups / raw.retained_taxa)
    reference = zero[zero.graph.eq("phylogeny")][["target", "fold", "test_logloss"]].rename(columns={"test_logloss": "unfused_test_logloss"})
    raw = raw.merge(reference, on=["target", "fold"], validate="many_to_one")
    raw["delta_logloss_vs_unfused"] = raw.test_logloss - raw.unfused_test_logloss
    summary = raw.groupby(["target", "graph", "lambda_f"], as_index=False).agg(
        n_folds=("fold", "size"), retained_taxa_mean=("retained_taxa", "mean"),
        active_groups_mean=("n_active_groups", "mean"), active_groups_min=("n_active_groups", "min"),
        active_groups_max=("n_active_groups", "max"), compression_pct=("structural_compression_pct", "mean"),
        test_logloss_mean=("test_logloss", "mean"), test_logloss_sd=("test_logloss", "std"),
        test_auc_fold_mean=("test_auc", "mean"), train_logloss_mean=("train_logloss", "mean"),
        delta_logloss_vs_unfused=("delta_logloss_vs_unfused", "mean"))
    checks["regularization_rows_verified"] = len(raw)
    save_table(raw, out, "regularization_fold_results.csv")
    save_table(summary, out, "regularization_summary.csv")
    return summary


def historical_table(base: Path, out: Path):
    compression = pd.read_csv(base / "saved_compression.csv")
    scores = pd.read_csv(base / "saved_results.csv")
    keys = ["run", "target", "resolution", "timestamp", "family"]
    compression = compression[compression.target.isin(TARGETS)].copy()
    compression["structural_compression_pct"] = 100 * (1 - compression.active_groups / compression.retained_taxa)
    grouped = compression.groupby(keys, as_index=False).agg(
        n_outer_folds=("outer_fold", "nunique"), retained_taxa_mean=("retained_taxa", "mean"),
        active_taxa_mean=("active_taxa", "mean"), active_groups_mean=("active_groups", "mean"),
        all_groups_mean=("all_groups", "mean"), compression_pct=("structural_compression_pct", "mean"),
        maximum_residual=("maximum_residual", "max"))
    result = grouped.merge(scores[keys + ["n", "n_sites", "coverage", "pooled_auc_roc", "pooled_auc_pr", "site_balanced_log_loss"]],
                           on=keys, validate="one_to_one", how="left", indicator=True)
    assert result._merge.eq("both").all()
    result = result.drop(columns="_merge")
    result["use"] = "historical appendix; keep run-specific protocols separate"
    save_table(result, out, "historical_saved_runs.csv")
    return result


def save_figure(fig, out: Path, name: str):
    fig.savefig(out / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out / "figures" / (name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figures(paths, primary, all_summary, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for i, target in enumerate(TARGETS):
        for graph, color in COLORS.items():
            d = paths[(paths.target == target) & (paths.graph == graph)].sort_values("lambda_f")
            for j, metric in enumerate(["test_logloss_mean", "test_auc_fold_mean"]):
                axes[i, j].plot(d.active_groups_mean, d[metric], "o-", color=color, label=graph.title(), ms=5)
            for _, row in d[(d.graph == "phylogeny") & d.lambda_f.isin([0, 1, 100])].iterrows():
                axes[i, 0].annotate(f"λ={row.lambda_f:g}", (row.active_groups_mean, row.test_logloss_mean),
                                    xytext=(3, 7), textcoords="offset points", fontsize=8, color=color)
        for ax in axes[i]:
            ax.set_title(TITLES[target]); ax.set_xlabel("Mean active coefficient groups (3 folds)")
            ax.grid(alpha=.18); ax.margins(x=.09, y=.17)
        axes[i, 0].set_ylabel("Mean held-out site log-loss ↓")
        axes[i, 1].set_ylabel("Mean fold AUROC ↑ (not pooled)")
    axes[0, 0].legend(title="Graph", fontsize=9)
    fig.suptitle("Compression–prediction paths: original 37-site protocol\nFixed ridge=0.1, sparsity=0; connected points follow increasing fusion strength", y=1.01)
    fig.tight_layout()
    save_figure(fig, out, "01_regularization_tradeoff")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(GROUP_MODELS))
    for i, target in enumerate(TARGETS):
        d = primary[primary.target.eq(target)].set_index("model").loc[GROUP_MODELS]
        axes[i, 0].barh(y, d.retained_features_mean, color="#dce5ec", label="Retained taxa")
        axes[i, 0].barh(y, d.active_groups_mean, color="#39799b", label="Active groups")
        axes[i, 0].set_yticks(y, [LABELS[m] for m in GROUP_MODELS]); axes[i, 0].invert_yaxis()
        for yy, val in zip(y, d.active_groups_mean):
            axes[i, 0].text(val + 1, yy, f"{val:.1f}", va="center", fontsize=9)
        axes[i, 0].set_xlabel("Mean counts across 6 outer fits")
        axes[i, 0].set_xlim(0, d.retained_features_mean.max() * 1.2)
        axes[i, 1].hlines(y, d.ci_low, d.ci_high, color="#607d8b", lw=2)
        axes[i, 1].scatter(d.delta_logloss_vs_ridge, y, color="#c66120", zorder=3)
        axes[i, 1].axvline(0, color="#333333", ls="--", lw=1)
        axes[i, 1].set_yticks(y, [LABELS[m] for m in GROUP_MODELS]); axes[i, 1].invert_yaxis()
        axes[i, 1].set_xlabel("Held-out log-loss difference vs ridge (lower is better)")
        for ax in axes[i]:
            ax.set_title(TITLES[target]); ax.grid(axis="x", alpha=.18)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("QC-corrected 35-group evaluation: compression and predictive cost\nIntervals: 95% paired site bootstrap of fixed OOF losses; models are not refitted", y=1.01)
    fig.tight_layout()
    save_figure(fig, out, "02_qc_compression_and_loss")

    # Same-protocol descriptive tradeoff; external ML has different complexity semantics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = plt.get_cmap("tab10")
    offsets = [(5, 6), (5, 6), (8, 7), (5, 7), (-16, -12), (5, -14), (5, 7), (5, 7)]
    for ax, target in zip(axes, TARGETS):
        d = primary[primary.target.eq(target)].set_index("model")
        for k, model in enumerate(GROUP_MODELS):
            r = d.loc[model]
            ax.scatter(r.active_groups_mean, r.logloss, s=65, color=palette(k), label=f"{k}: {LABELS[model]}")
            offset = (-16, -12) if target == TARGETS[1] and model == "adaptive_expanded" else offsets[k]
            ax.annotate(str(k), (r.active_groups_mean, r.logloss), xytext=offset, textcoords="offset points", fontsize=9)
        ax.set_title(TITLES[target]); ax.set_xlabel("Mean active coefficient groups")
        ax.set_ylabel("Mean repeat-pooled site log-loss ↓"); ax.grid(alpha=.18); ax.margins(.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(.5, -.1))
    fig.suptitle("Observed tradeoff among saved log-contrast models (QC-corrected CV)\nPoints are descriptive comparisons, not test-selected deployment settings", y=1.02)
    fig.tight_layout()
    save_figure(fig, out, "03_qc_tradeoff")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, TARGETS):
        for k, model in enumerate(GROUP_MODELS):
            d = all_summary[(all_summary.target == target) & (all_summary.model == model)].set_index("selection_tag")
            a, b = d.loc["one_se"], d.loc["min_loss"]
            ax.plot([a.active_groups_mean, b.active_groups_mean], [a.logloss, b.logloss], "-", color=palette(k), alpha=.7)
            ax.scatter(a.active_groups_mean, a.logloss, marker="o", color=palette(k), s=40)
            ax.scatter(b.active_groups_mean, b.logloss, marker="x", color=palette(k), s=65)
        ax.set_title(TITLES[target]); ax.set_xlabel("Mean active coefficient groups")
        ax.set_ylabel("Mean repeat-pooled site log-loss ↓"); ax.grid(alpha=.18)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(.5, -.1))
    fig.suptitle("Selection-rule sensitivity on the same saved QC candidates\nCircle: one-standard-error; cross: minimum inner loss. No outer-test selection.", y=1.02)
    fig.tight_layout()
    save_figure(fig, out, "04_qc_selection_sensitivity")


def report(paths, primary, all_summary, historical, out, checks, asv=None, domain_fusion=None):
    table_cols = ["model", "retained_features_mean", "active_groups_mean", "structural_compression_pct", "logloss", "auc", "ap", "delta_logloss_vs_ridge"]
    names = {"model": "Model", "retained_features_mean": "Retained taxa", "active_groups_mean": "Active groups",
             "structural_compression_pct": "Reduction (%)", "logloss": "Log-loss", "auc": "AUROC", "ap": "AP",
             "delta_logloss_vs_ridge": "Δ loss vs ridge"}
    sections = []
    for target in TARGETS:
        d = primary[(primary.target == target) & primary.model.isin(GROUP_MODELS)].set_index("model").loc[GROUP_MODELS].reset_index()
        sections.append(f"### {TITLES[target]}\n\n" + md(d[table_cols].rename(columns=names)))
    example = paths[(paths.target == TARGETS[0]) & paths.graph.eq("phylogeny") & paths.lambda_f.isin([0, 1, 100])]
    example = example[["lambda_f", "active_groups_mean", "test_logloss_mean", "delta_logloss_vs_unfused"]].rename(columns={
        "lambda_f": "Fusion strength", "active_groups_mean": "Active groups", "test_logloss_mean": "Fold-mean log-loss", "delta_logloss_vs_unfused": "Δ loss vs unfused"})
    findings = []
    for target in TARGETS:
        d = primary[primary.target.eq(target)].set_index("model")
        ridge, shared, adaptive, abundance = [d.loc[m] for m in ["genus_ridge", "shared_hybrid", "domain_adaptive_hybrid", "abundance_only"]]
        findings.append(f"- **{TITLES[target]}:** shared hybrid uses {shared.active_groups_mean:.1f} active groups from {shared.retained_features_mean:.1f} retained taxa ({shared.structural_compression_pct:.1f}% mean within-fold reduction), with log-loss {shared.logloss:.3f} versus ridge {ridge.logloss:.3f}. Adaptive hybrid uses {adaptive.active_groups_mean:.1f} groups and has log-loss {adaptive.logloss:.3f}; abundance-only uses {abundance.active_groups_mean:.1f} groups and has log-loss {abundance.logloss:.3f}.")
    baselines = primary[primary.model.isin(BASELINES)][["target", "model", "retained_features_mean", "active_taxa_mean", "logloss", "auc", "ap"]].copy()
    baselines["target"] = baselines.target.map(TITLES)
    selection_examples = all_summary[all_summary.model.isin(["shared_hybrid", "adaptive_expanded"])][
        ["target", "model", "selection_tag", "active_groups_mean", "logloss", "auc"]].copy()
    selection_examples["target"] = selection_examples.target.map(TITLES)
    mismatches = [name for name, matches in checks["matches_original_run_code_config_hashes"].items() if not matches]
    provenance_note = ("**Training provenance limitation:** the copied current context files " + ", ".join(f"`{name}`" for name in mismatches)
        + " do not match the code/config hashes recorded in the original run manifest. They are context snapshots, not certified copies of the training versions. This does not prevent exact reanalysis of saved predictions, but exact reproduction of the original fits is not established. The workbook hash is preserved in the copied manifest; the workbook itself was not copied or revalidated.") if mismatches else "The copied code/config hashes match those recorded in the original run manifest. The raw workbook was not copied or revalidated."
    content = f"""# Experiment 1 — prediction–compression tradeoff

This experiment reuses saved fusionML outputs. No models were trained, no hyperparameters were chosen from test results, and no raw sequencing or chemistry inputs were changed. These are results for the existing fusionML models, not newly fitted fusionMLv2 models.

## Main findings

{chr(10).join(findings)}

Compression is present, but the hybrid models selected by the primary **one-SE rule** do not improve probability prediction over the tuned ridge reference in this QC-corrected comparison. Less aggressive compression under the saved minimum-loss rule gives different tradeoffs, including near-ridge loss for some acid–base models; those results are shown separately below. No noninferiority margin was prespecified, and fixed-prediction intervals do not include model-retraining uncertainty. See group counts and discrimination alongside loss rather than declaring a universal winner.

## Two evidence sets, kept separate

| Evidence | Protocol | Use |
| --- | --- | --- |
| Regularization paths | Original 37 site labels, 3 outer folds; two targets; phylogeny, abundance and 50:50 shared graphs; 9 fusion strengths; ridge=0.1 and sparsity=0 | Descriptive effect of stronger fusion; no inner tuning of this path |
| QC-corrected model comparison | 134 samples, 35 groups; BSI/BSIb and DGY/DGYb kept together; 3 outer folds × 2 repeats; 2 inner folds; common splits and partition graph seeds | Primary comparison of saved selected models at full outer-training size |

Only fraction=1.0 is used from the saved learning experiment. This means all available **outer-training sites**, not training on all 134 samples. The approximately 23 training groups per outer fit remain disjoint from test groups. No new learning curves are generated.

## Primary QC-corrected results

Counts are means over six fitted models. Metrics are computed from pooled outer predictions separately in each repeat, then averaged over two repeats. Log-loss gives each site equal weight; AUROC, AP, Brier, MCC and F2 retain the source sample-level definitions. Group reduction is averaged within folds relative to retained taxa, not relative to raw ASVs. Fusion uses the saved one-SE selection rule; the ordinary ML baselines minimize inner loss even though their source selection tag is `one_se`.

{(chr(10) * 2).join(sections)}

![QC compression and predictive loss](figures/02_qc_compression_and_loss.png)

The error bars are 95% percentile intervals from paired resampling of 35 sites using fixed OOF losses, averaging repeat losses before resampling. They describe variability across the observed sites conditional on the saved predictions. They omit refitting uncertainty, are not adjusted for multiple comparisons, and are not a formal noninferiority test. CV folds and repeats are not treated as independent observations.

![QC tradeoff](figures/03_qc_tradeoff.png)

`observed_nondominated_logcontrast` in the primary CSV identifies points not beaten simultaneously in mean group count and mean loss by another saved log-contrast model. It is descriptive, excludes the external ML baselines, and must not be used to choose a deployment setting from outer-test results. No matched-size optimization or interpolation was performed.

## Descriptive regularization paths

{md(example)}

![Regularization tradeoff](figures/01_regularization_tradeoff.png)

On this ammonia phylogenetic path, moderate fusion reduces group count while improving loss relative to the fixed unfused model; stronger fusion loses predictive information. The unfused model here has ridge=0.1 and is **not** the tuned QC ridge reference. These path values are means of three fold scores, not pooled CV metrics. Do not pool or rank these points together with the QC-corrected results.

## Uncompressed ML references

{md(baselines.rename(columns={"target": "Target", "model": "Model", "retained_features_mean": "Input taxa", "active_taxa_mean": "Selected/used taxa", "logloss": "Log-loss", "auc": "AUROC", "ap": "AP"}))}

Matched RA uses the same retained taxa as fusion but ordinary relative abundance. Flat RA uses taxonomic genus sums with a different, larger input feature universe. Elastic net here is scikit-learn logistic elastic net, not an R glmnet run. Its active count means nonzero coefficients; XGBoost's count means taxa with nonzero feature importance. Neither count is a fused-group representation, and neither pipeline was trained on compressed group features (Experiment 4 remains unperformed). Their source `n_active_groups` values are deliberately not presented as microbial groups.

## Selection sensitivity and historical appendix

![Selection sensitivity](figures/04_qc_selection_sensitivity.png)

The supplementary selection comparison uses already saved one-SE and minimum-inner-loss candidates under the **same QC protocol**. It explains the impact of favoring compression; it does not endorse a selection rule after observing its test performance. The expanded adaptive grid remains a separate exploratory family.

{md(selection_examples.rename(columns={"target": "Target", "model": "Model", "selection_tag": "Selection", "active_groups_mean": "Active groups", "logloss": "Log-loss", "auc": "AUROC"}))}

For example, minimum-loss selection for shared hybrid gives acid–base log-loss about 0.439 with 57.5 groups, compared with ridge loss 0.443 using 94 groups. This is an observed tradeoff in the same cohort, not a confirmed superiority or noninferiority result. The complete rule comparison is in `tables/qc_selection_sensitivity.csv`.

`tables/historical_saved_runs.csv` contains {len(historical)} matched historical run/family records for the two targets, including any saved ranks. Scores and group counts are joined on run, target, resolution, timestamp and family; unrelated runs are never pooled. That appendix retains original run protocols and is not a harmonized rank comparison. `metadr_order_only`, where present, is a graph-penalty comparator, not the MetaDR neural network.

## Meaning and limits of compression

- Active coefficient groups describe the fitted log-contrast model's structure. They are not a measured effective degrees of freedom, a certified exported K-column representation, or a reduction in sequencing requirements.
- Zero-coefficient groups are omitted from the active count. Small taxon residuals can remain; the historical appendix reports their maximum absolute value where saved. Historical genus QC row exports do not include residual coefficients, so exact group-only reconstruction cannot be verified for those fits. The new ASV exports include residuals and verify group-plus-residual reconstruction.
- ARC–BAC product counts are reported separately. All primary one-SE full-size QC interaction fits in these outputs have zero active product terms at the original tolerance; some ASV minimum-loss fits retain interaction terms.
- Existing search budgets, prevalence filtering and the one-SE preference for compact models affect these results. This reanalysis does not correct the original fitting choices.
- Missing Experiment 1 comparisons include trac/tree-guided alternatives, log-ratio PCA, and uniformly evaluated compression budgets. The results establish an existing tradeoff, not superiority over those untested methods.
- No group-stability, biological-validation or new downstream-representation claim is made by Experiment 1.

## Reproduction and validation

From the repository root, after the input snapshot exists:

```bash
/home/best/anaconda3/envs/torch/bin/python fusionMLv2/exp1/scripts/generate_results.py
```

The generator reads only `../data/`; original fusionML outputs and raw datasets are not required. `../scripts/copy_inputs.py` refreshes the snapshot explicitly. See `../README.md` for the complete commands and metric definitions.

{provenance_note}

Validation verified {checks['copied_files_verified']} copied-file hashes, {checks['outer_partitions_without_site_overlap']} outer partitions with disjoint train/test sites, full prediction coverage, and all six recomputed metrics against saved repeat-level values (maximum absolute discrepancy {max(checks['qc_metric_max_absolute_differences'].values()):.3g}). Paths contain all 162 expected evaluations. Checks and software versions are recorded in `validation.json`; output hashes are recorded in `artifact_manifest.json`.
"""
    if asv is not None:
        content = content.replace(
            "This experiment reuses saved fusionML outputs. No models were trained, no hyperparameters were chosen from test results, and no raw sequencing or chemistry inputs were changed. These are results for the existing fusionML models, not newly fitted fusionMLv2 models.",
            "This report combines reused genus-level fusionML outputs with a newly completed ASV-level fusionML run using the original ARC and BAC rooted trees. The report generator itself performs no fitting. Both resolutions use the existing fusionML estimator; these are not a new fusionMLv2 algorithm. No hyperparameters were selected using outer-test results.")
        content = content.replace("## Main findings\n", "## New ASV findings\n\n" + asv["findings"] + "\n\n## Reused genus-level findings\n", 1)
        content = content.replace("## Two evidence sets, kept separate", "## Evaluation protocols and resolution")
        content = content.replace("| QC-corrected model comparison |", "| Reused genus-level QC comparison |")
        marker = "Only fraction=1.0 is used"
        content = content.replace(marker, "The new ASV experiment also uses the 35-group QC protocol and the same cohort, labels and outer split assignments. Its full configuration and input hashes are recorded in a separate run manifest.\n\n" + marker, 1)
        content = content.replace("## Primary QC-corrected results", "## Primary QC-corrected results — genus-level inputs")
        content = content.replace("## Descriptive regularization paths", asv["section"] + "## Descriptive regularization paths — reused genus-level trees", 1)
    if domain_fusion is not None:
        content = content.replace("## Descriptive regularization paths", domain_fusion["section"] + "## Descriptive regularization paths", 1)
        content = content.replace(
            "The report generator itself performs no fitting.",
            "A third run adds the ARC-only/BAC-only/both-domain fusion arms for three warning flags. The report generator itself performs no fitting.", 1)
        content = content.replace(
            "| Reused genus-level QC comparison |",
            "| Domain-restricted fusion, order/family/genus/ASV | Same 35-group QC protocol, splits and seeds at every resolution; three flags; fusion edges given to ARC only, BAC only, or both domains; minimum-inner-loss selection ranked by MCC | Where the fusion penalty should act and at what resolution, not which domain to assay |\n| Reused genus-level QC comparison |", 1)
        content = content.replace("## Main findings\n", "## Domain-restricted fusion findings\n\n" + domain_fusion["findings"] + "\n\n## Main findings\n", 1)
        content = content.replace("## New ASV findings\n", "## Domain-restricted fusion findings (three flags, best resolution per arm, minimum-inner-loss)\n\n" + domain_fusion["findings"] + "\n\n## New ASV findings\n", 1)
    (out / "REPORT.md").write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT / "data")
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("Use at least 100 bootstrap resamples")
    out = args.results_dir
    for folder in [out, out / "tables", out / "figures"]:
        folder.mkdir(parents=True, exist_ok=True)
    checks = verify_snapshot(args.data_dir)
    base = args.data_dir / "investigation_20260921"
    primary, all_summary = qc_tables(base, out, checks, args.bootstrap_resamples, args.seed)
    paths = path_tables(base, out, checks)
    historical = historical_table(base, out)
    from asv_results import build as build_asv_results
    from domain_restricted_results import build as build_domain_fusion_results
    asv = build_asv_results(args.data_dir, out, primary, sys.modules[__name__])
    if asv is not None:
        checks["asv"] = asv["validation"]
    domain_fusion = build_domain_fusion_results(args.data_dir, out, sys.modules[__name__])
    if domain_fusion is not None:
        checks["domain_fusion"] = domain_fusion["validation"]
    checks["asv_results_module_sha256"] = sha(Path(__file__).with_name("asv_results.py"))
    checks["domain_restricted_results_module_sha256"] = sha(Path(__file__).with_name("domain_restricted_results.py"))
    checks["primary_active_interaction_products"] = float(primary.active_products_mean.max())
    assert checks["primary_active_interaction_products"] == 0
    checks["versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__}
    checks["generator_sha256"] = sha(Path(__file__))
    checks["source_manifest_sha256"] = sha(args.data_dir / "source_manifest.json")
    checks["bootstrap"] = {"seed": args.seed, "resamples": args.bootstrap_resamples, "refit": False}
    (out / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    figures(paths, primary, all_summary, out)
    report(paths, primary, all_summary, historical, out, checks, asv, domain_fusion)
    outputs = [p for p in out.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    (out / "artifact_manifest.json").write_text(json.dumps({"files": [{"path": str(p.relative_to(out)), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(outputs)]}, indent=2) + "\n")
    print(f"Generated {len(outputs)} artifacts in {out}")
    print(primary[["target", "model", "active_groups_mean", "logloss", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

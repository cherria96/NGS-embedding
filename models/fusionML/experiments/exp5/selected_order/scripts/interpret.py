"""Biological interpretation of the two selected order-level fusion models.

    warning_acid_high        (effluent pH > 8.0)  x fusion_arc_only x order
    warning_ammonia_toxicity (effluent NH4-N > 2.5) x shared_hybrid x order

No models are fitted. Every input is a hash-verified local snapshot written by
copy_inputs.py. Reads nothing outside this experiment's data directory.
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
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fusionmlv2-exp5-selected-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fusionmlv2-exp5-selected-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_evidence import BACKGROUND, CONTEXT, EVIDENCE

# --- the two selected configurations ----------------------------------------
SELECTED = [
    dict(key="ammonia", flag="warning_ammonia_toxicity", model="shared_hybrid", arm="both",
         title="Ammonia toxicity", definition="effluent NH4-N (TAN) > 2.5",
         arm_label="Both domains fused"),
    dict(key="acid_high", flag="warning_acid_high", model="fusion_arc_only", arm="fusion_arc_only",
         title="Acid high", definition="effluent pH > 8.0", arm_label="ARC-only fusion"),
]
RESOLUTIONS = ["order", "family", "genus", "asv"]
MODELS = ["fusion_arc_only", "fusion_bac_only", "shared_hybrid",
          "domain_adaptive_hybrid", "domain_adaptive_interaction"]
# Experiment 1 reports the domain-restricted arms under the minimum-inner-loss
# rule; one_se is carried through every table as the sensitivity arm.
RULE = "min_loss"
PRED_KEYS = ["target", "repeat", "model", "selection"]
DETECTION_COUNT = 5   # fusionML config: a feature counts as detected at >=5 reads
PSEUDOCOUNT = 0.5     # fusionML config: log-composition pseudocount
ZERO = 1e-8           # coefficients below this magnitude are read as "no contribution"

# Repo house palette. Domain is the only categorical encoding in these figures;
# coefficient sign is carried by position about zero, never by hue.
ARC, BAC = "#2563a6", "#d16b19"
INK, MUTED, GRID, BACKDROP = "#1f2933", "#52606d", "#cbd2d9", "#eef2f5"
DOMAIN_COLOR = {"ARC": ARC, "BAC": BAC}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float) and not np.isfinite(v):
            return "NA"
        if pd.isna(v):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return str(v).replace("|", "\\|")
    head = ["| " + " | ".join(str(c) for c in df.columns) + " |",
            "| " + " | ".join("---" for _ in df.columns) + " |"]
    return "\n".join(head + ["| " + " | ".join(cell(v) for v in row) + " |"
                             for row in df.itertuples(index=False, name=None)])


def metrics(d: pd.DataFrame) -> dict:
    """Experiment 1's definitions, unchanged: pooled within a repeat, saved
    training-selected threshold, sample-level AUROC/AP, site-balanced log loss."""
    y, p = d.y.to_numpy(), d.p.to_numpy()
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    decision = p >= d.threshold.to_numpy()
    single_class = y.sum() in (0, len(y))
    return {"auc": np.nan if single_class else roc_auc_score(y, p),
            "ap": np.nan if single_class else average_precision_score(y, p),
            "logloss": pd.Series(loss).groupby(d.site_id.to_numpy()).mean().mean(),
            "brier": float(np.mean((y - p) ** 2)),
            "mcc": matthews_corrcoef(y, decision),
            "f2": fbeta_score(y, decision, beta=2, zero_division=0),
            "n_scored": len(d), "n_sites": d.site_id.nunique(), "n_outer_folds": d.outer_fold.nunique()}


# --- inputs -----------------------------------------------------------------
def verify_snapshot(data: Path) -> dict:
    manifest = json.loads((data / "source_manifest.json").read_text())
    bad = [e["copy"] for e in manifest["files"] if sha(data / e["copy"]) != e["sha256"]]
    if bad:
        raise SystemExit(f"snapshot hash mismatch: {bad[:5]}")
    return {"files_verified": len(manifest["files"]), "selected": manifest["selected"]}


def read_run(data: Path, key: str, resolution: str, name: str) -> pd.DataFrame:
    return pd.read_csv(data / "runs" / f"{key}__{resolution}" / name)


def parse_lineage(taxon: str) -> dict:
    ranks = {}
    for part in str(taxon).split(";"):
        part = part.strip()
        if "__" in part:
            rank, value = part.split("__", 1)
            if value:
                ranks[rank.strip()] = value.strip()
    return ranks


def block_catalog(data: Path) -> pd.DataFrame:
    """feature_id -> order-block label, ASV count, genus composition, phylum.

    The rank tree splits non-monophyletic orders into blocks (``__b1``,
    ``__b2``...), so the block, not the order name, is the model's feature.
    """
    rows = []
    for domain in ["ARC", "BAC"]:
        mapping = pd.read_csv(data / "reference/order" / f"{domain.lower()}_mapping.csv")
        taxonomy = pd.read_csv(data / "reference/taxonomy" / f"{domain}_metadata.tsv", sep="\t")
        taxonomy = taxonomy[~taxonomy["Feature ID"].str.startswith("#")].set_index("Feature ID")
        for reduced, group in mapping.groupby("reduced_feature"):
            label = group.label.iloc[0]
            lineages = [parse_lineage(taxonomy.loc[asv, "Taxon"]) for asv in group.feature]
            genera = [lin.get("g") or ("f__" + lin["f"] if "f" in lin else "unclassified") for lin in lineages]
            phyla = [lin.get("p", "unassigned") for lin in lineages]
            classified = not label.strip("0123456789abcdef") == ""
            rows.append(dict(
                feature_id=f"{domain}::{reduced}", domain=domain, block=label, n_asv=len(group),
                classified=classified,
                phylum="; ".join(sorted({p for p in phyla})) if len(set(phyla)) <= 3 else "mixed",
                genus_composition="; ".join(f"{g} ({n})" if n > 1 else g for g, n in Counter(genera).most_common()),
                dominant_genus=Counter(genera).most_common(1)[0][0],
                dominant_genus_fraction=Counter(genera).most_common(1)[0][1] / len(group),
                min_confidence=float(min(float(taxonomy.loc[a, "Confidence"]) for a in group.feature))))
    return pd.DataFrame(rows)


def chemistry(data: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """pH, TAN and the two flags for the cohort, from the snapshot workbook.

    `site_id` comes from the runs' own splits (35 QC-adjusted site groups),
    not from the workbook's raw acronyms (37), so every site-level number here
    matches the grouping the models were cross-validated under. The recomputed
    flags are asserted equal to the labels the models were trained on.
    """
    sys.path.insert(0, str(data))
    from fusionML.config import Config
    from fusionML.chemistry_targets import build_chemistry_derived_metadata
    cfg = Config.from_yaml(str(data / "fusionML/configs/real_smoke_classification_phylo.yaml"))
    frame = {}
    for column in ["effluent.pH", "effluent.NH4-N", "warning_acid_high", "warning_ammonia_toxicity"]:
        table, _ = build_chemistry_derived_metadata(
            str(data / "reference/chemistry/Dat_chem.xlsx"), cfg.raw["chemistry"], column, [])
        frame[column] = table["target"]
    chem = pd.DataFrame(frame).loc[cohort.index]
    chem["site_id"] = cohort.site_id
    for selection in SELECTED:
        assert chem[selection["flag"]].to_numpy().astype(int).tolist() == \
            cohort[selection["key"]].to_numpy().astype(int).tolist(), \
            f"recomputed {selection['flag']} differs from the trained labels"
    return chem


# --- analysis ---------------------------------------------------------------
def ranking(data: Path) -> pd.DataFrame:
    """Where the two selected configurations sit among all 20 flag x resolution
    x model combinations. Recomputed from saved probabilities, not copied."""
    rows = []
    for selection in SELECTED:
        for resolution in RESOLUTIONS:
            pred = read_run(data, selection["key"], resolution, "predictions.csv")
            pred = pred[pred.target.eq(selection["flag"])]
            for keys, d in pred.groupby(PRED_KEYS):
                rows.append({**dict(zip(PRED_KEYS, keys)), "resolution": resolution, **metrics(d)})
    per_repeat = pd.DataFrame(rows)
    group_keys = ["target", "resolution", "model", "selection"]
    pooled = per_repeat.groupby(group_keys, as_index=False)[
        ["auc", "ap", "logloss", "brier", "mcc", "f2", "n_scored"]].mean()
    pooled["n_repeats"] = per_repeat.groupby(group_keys).repeat.nunique().to_numpy()
    # Rank within (flag, selection rule): 4 resolutions x 5 models = 20 candidates.
    for _, block in pooled.groupby(["target", "selection"]):
        pooled.loc[block.index, "rank_by_mcc"] = block.mcc.rank(ascending=False, method="min")
        pooled.loc[block.index, "rank_by_f2"] = block.f2.rank(ascending=False, method="min")
        pooled.loc[block.index, "n_candidates"] = len(block)
    pooled["selected"] = [any(s["flag"] == t and s["model"] == m and r == "order" and sel == RULE
                              for s in SELECTED)
                          for t, r, m, sel in zip(pooled.target, pooled.resolution, pooled.model, pooled.selection)]
    return pooled.sort_values(["target", "selection", "mcc"], ascending=[True, True, False])


def fit_inventory(data: Path) -> pd.DataFrame:
    rows = []
    for selection in SELECTED:
        fits = read_run(data, selection["key"], "order", "rows.csv")
        counts = read_run(data, selection["key"], "order", "domain_counts.csv")
        fits = fits[fits.target.eq(selection["flag"]) & fits.model.eq(selection["model"])]
        counts = counts[counts.target.eq(selection["flag"]) & counts.model.eq(selection["model"])]
        wide = counts.pivot_table(index=["repeat", "outer_fold", "selection"], columns="domain",
                                  values=["retained_features", "active_groups", "n_edges"]).reset_index()
        wide.columns = [c[0] if not c[1] else f"{c[1].lower()}_{c[0]}" for c in wide.columns]
        merged = fits.merge(wide, on=["repeat", "outer_fold", "selection"], validate="one_to_one")
        merged.insert(0, "config", selection["key"])
        rows.append(merged)
    keep = ["config", "target", "model", "selection", "repeat", "outer_fold", "status", "candidate", "threshold",
            "n_train", "n_train_sites", "n_train_positive", "n_test_positive", "retained_taxa",
            "arc_retained_features", "bac_retained_features", "n_active_groups", "arc_active_groups",
            "bac_active_groups", "arc_n_edges", "bac_n_edges", "n_active_taxa", "inner_loss", "inner_se",
            "equality_residual", "group_reconstruction_error", "test_auc", "test_ap", "test_mcc", "test_f2"]
    return pd.concat(rows, ignore_index=True)[keep]


def coefficients(data: Path, catalog: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for selection in SELECTED:
        coeff = read_run(data, selection["key"], "order", "coefficients_and_groups.csv")
        coeff = coeff[coeff.target.eq(selection["flag"]) & coeff.model.eq(selection["model"])].copy()
        coeff.insert(0, "config", selection["key"])
        frames.append(coeff)
    coeff = pd.concat(frames, ignore_index=True)
    return coeff.merge(catalog[["feature_id", "block", "n_asv", "genus_composition", "phylum",
                                "dominant_genus", "classified"]], on="feature_id", validate="many_to_one")


def taxon_summary(coeff: pd.DataFrame, abundance: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, feature): direction, cross-fold agreement, size."""
    rule = coeff[coeff.selection.eq(RULE)].copy()
    fit_keys = ["config", "repeat", "outer_fold"]
    n_fits = rule.groupby("config")[fit_keys].apply(lambda d: d.drop_duplicates().shape[0])
    # How many features share each feature's fused group, per fit.
    rule["group_size"] = rule.groupby(fit_keys + ["domain", "group_id"]).feature_id.transform("size")
    rows = []
    for (config, domain, feature), d in rule.groupby(["config", "domain", "feature_id"]):
        first = d.iloc[0]
        rows.append(dict(
            config=config, domain=domain, feature_id=feature, block=first.block, n_asv=int(first.n_asv),
            phylum=first.phylum, genus_composition=first.genus_composition,
            fits_retained=len(d), fits_planned=int(n_fits[config]),
            mean_beta=d.beta.mean(), median_beta=d.beta.median(), min_beta=d.beta.min(), max_beta=d.beta.max(),
            n_positive_fits=int((d.beta > ZERO).sum()), n_negative_fits=int((d.beta < -ZERO).sum()),
            n_zero_fits=int((d.beta.abs() <= ZERO).sum()), n_active_fits=int(d.active.sum()),
            mean_group_size=float(d.group_size.mean()), n_fits_fused=int((d.group_size > 1).sum())))
    summary = pd.DataFrame(rows)
    summary["sign_agreement"] = summary[["n_positive_fits", "n_negative_fits"]].max(axis=1) / summary.fits_retained
    summary["direction"] = np.where(summary.mean_beta > ZERO, "numerator (higher -> higher risk)",
                                    np.where(summary.mean_beta < -ZERO, "denominator (higher -> lower risk)", "none"))
    merged = summary.merge(abundance, on=["domain", "feature_id"], how="left", validate="many_to_one")
    return merged.sort_values(["config", "mean_beta"], ascending=[True, False])


def group_catalog(coeff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, d in coeff.groupby(["config", "selection", "repeat", "outer_fold", "domain", "group_id"]):
        config, selection, repeat, fold, domain, group = keys
        members = sorted(d.block)
        rows.append(dict(config=config, selection=selection, repeat=repeat, outer_fold=fold, domain=domain,
                         group_id=group, n_members=len(d), n_asv=int(d.n_asv.sum()),
                         group_mean_beta=float(d.group_mean_beta.iloc[0]),
                         theta=float(len(d) * d.group_mean_beta.iloc[0]),
                         max_residual=float(d.residual.abs().max()), active=bool(d.active.any()),
                         n_classified_members=int(d.classified.sum()),
                         phyla="; ".join(sorted({p for p in d.phylum})),
                         members="; ".join(members)))
    return pd.DataFrame(rows).sort_values(["config", "selection", "repeat", "outer_fold", "domain",
                                           "group_mean_beta"])


def recurrent_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Multi-member fused sets that recur across outer fits, exact membership."""
    rule = groups[groups.selection.eq(RULE) & groups.n_members.gt(1)]
    rows = []
    for (config, domain, members), d in rule.groupby(["config", "domain", "members"]):
        rows.append(dict(config=config, domain=domain, n_members=int(d.n_members.iloc[0]),
                         n_fits=len(d), fits="; ".join(f"r{r}f{f}" for r, f in zip(d.repeat, d.outer_fold)),
                         mean_group_beta=d.group_mean_beta.mean(), mean_theta=d.theta.mean(),
                         n_positive=int((d.group_mean_beta > ZERO).sum()),
                         n_negative=int((d.group_mean_beta < -ZERO).sum()), members=members))
    out = pd.DataFrame(rows)
    return out[out.n_fits > 1].sort_values(["config", "n_fits", "mean_theta"], ascending=[True, False, False])


def domain_contrasts(coeff: pd.DataFrame) -> pd.DataFrame:
    """Per fit and domain, the weighted log-ratio the zero-sum constraint defines."""
    rows = []
    rule = coeff[coeff.selection.eq(RULE)]
    for keys, d in rule.groupby(["config", "repeat", "outer_fold", "domain"]):
        config, repeat, fold, domain = keys
        pos, neg = d[d.beta > ZERO].sort_values("beta", ascending=False), d[d.beta < -ZERO].sort_values("beta")
        rows.append(dict(config=config, repeat=repeat, outer_fold=fold, domain=domain,
                         positive_mass=float(pos.beta.sum()), negative_mass=float(neg.beta.sum()),
                         zero_sum_residual=float(d.beta.sum()),
                         n_numerator=len(pos), n_denominator=len(neg),
                         top_numerator="; ".join(f"{b} ({v:+.4f})" for b, v in zip(pos.block.head(6), pos.beta.head(6))),
                         top_denominator="; ".join(f"{b} ({v:+.4f})" for b, v in zip(neg.block.head(6), neg.beta.head(6)))))
    return pd.DataFrame(rows)


def abundance_context(data: Path, catalog: pd.DataFrame, chem: pd.DataFrame) -> pd.DataFrame:
    """Descriptive size and direction of each block in the cohort, independent
    of any model: within-domain relative abundance, detection prevalence, and
    the site-level AUROC of its log relative abundance against each flag."""
    rows = []
    for domain in ["ARC", "BAC"]:
        table = pd.read_csv(data / "reference/order" / f"{domain.lower()}_table.csv", index_col=0)
        table = table.loc[table.index.intersection(chem.index)]
        relative = table.div(table.sum(axis=1), axis=0)
        log_ratio = np.log10((table + PSEUDOCOUNT).div(table.sum(axis=1) + PSEUDOCOUNT * table.shape[1], axis=0))
        site = chem.loc[table.index, "site_id"]
        site_mean = log_ratio.groupby(site.to_numpy()).mean()
        for feature in table.columns:
            entry = dict(domain=domain, feature_id=f"{domain}::{feature}",
                         mean_relative_abundance=float(relative[feature].mean()),
                         max_relative_abundance=float(relative[feature].max()),
                         detection_prevalence=float((table[feature] >= DETECTION_COUNT).mean()))
            for selection in SELECTED:
                labels = chem.loc[table.index, selection["flag"]].groupby(site.to_numpy()).max()
                x, y = site_mean[feature].to_numpy(), labels.to_numpy()
                entry[f"site_auc_{selection['key']}"] = (
                    float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else np.nan)
            rows.append(entry)
    return pd.DataFrame(rows)


def label_overlap(chem: pd.DataFrame) -> tuple:
    acid, ammonia = chem.warning_acid_high, chem.warning_ammonia_toxicity
    table = pd.crosstab(acid, ammonia)
    both = chem.dropna(subset=["effluent.pH", "effluent.NH4-N"])
    sites = {key: set(chem.loc[chem[flag].eq(1), "site_id"]) for key, flag in
             [("acid_high", "warning_acid_high"), ("ammonia", "warning_ammonia_toxicity")]}
    summary = dict(
        n_samples=int(len(chem)),
        n_sites=int(chem.site_id.nunique()),
        acid_high_positives=int(acid.sum()), ammonia_positives=int(ammonia.sum()),
        both_positive=int(((acid == 1) & (ammonia == 1)).sum()),
        acid_high_only=int(((acid == 1) & (ammonia == 0)).sum()),
        ammonia_only=int(((acid == 0) & (ammonia == 1)).sum()),
        acid_high_sites=len(sites["acid_high"]), ammonia_sites=len(sites["ammonia"]),
        shared_sites=len(sites["acid_high"] & sites["ammonia"]),
        ph_tan_pearson=float(both["effluent.pH"].corr(both["effluent.NH4-N"])),
        ph_tan_spearman=float(both["effluent.pH"].corr(both["effluent.NH4-N"], method="spearman")),
        n_with_both_measurements=int(len(both)),
        max_ph_in_cohort=float(chem["effluent.pH"].max()),
        phi=float(np.corrcoef(acid.to_numpy(), ammonia.to_numpy())[0, 1]))
    return table, summary


def cross_flag(taxa: pd.DataFrame) -> pd.DataFrame:
    wide = taxa.pivot_table(index=["domain", "feature_id", "block"], columns="config",
                            values=["mean_beta", "fits_retained", "sign_agreement"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in wide.columns]
    wide = wide.dropna(subset=["mean_beta_acid_high", "mean_beta_ammonia"])
    wide["same_sign"] = np.sign(wide.mean_beta_acid_high) == np.sign(wide.mean_beta_ammonia)
    return wide.sort_values("mean_beta_ammonia", ascending=False)


# --- figures ----------------------------------------------------------------
def savefig(fig, out: Path, name: str) -> list:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ["png", "pdf"]:
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def style() -> None:
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.titlesize": 11.5, "figure.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": INK, "axes.edgecolor": GRID})


def short(label: str, width: int = 30) -> str:
    """Keep the block suffix, which is the part that distinguishes blocks."""
    if len(label) <= width:
        return label
    stem, _, suffix = label.partition("__")
    return f"{stem[:width - len(suffix) - 4]}...__{suffix}" if suffix else label[:width - 3] + "..."


def figure_coefficients(taxa: pd.DataFrame, out: Path, top: int = 18) -> list:
    """Per flag: the blocks with the largest mean coefficient, with the range
    across outer fits. Position carries sign; hue carries domain only."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4))
    for ax, selection in zip(axes, SELECTED):
        d = taxa[taxa.config.eq(selection["key"]) & taxa.fits_retained.ge(4)].copy()
        d = d.reindex(d.mean_beta.abs().sort_values(ascending=False).index).head(top)
        d = d.sort_values("mean_beta")
        y = np.arange(len(d))
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        ax.hlines(y, d.min_beta, d.max_beta, color=[DOMAIN_COLOR[x] for x in d.domain], lw=2, alpha=.34, zorder=2)
        ax.hlines(y, 0, d.mean_beta, color=[DOMAIN_COLOR[x] for x in d.domain], lw=2.6, zorder=3)
        ax.scatter(d.mean_beta, y, s=54, c=[DOMAIN_COLOR[x] for x in d.domain],
                   edgecolor="white", linewidth=1.6, zorder=4)
        labels = [short(b) + ("" if n == 1 else f"  ({n} ASVs)") for b, n in zip(d.block, d.n_asv)]
        ax.set_yticks(y, labels)
        ax.set_ylim(-0.8, len(d) - 0.2)
        span = max(d.max_beta.max(), 0) - min(d.min_beta.min(), 0)
        tally_x = d.max_beta.max() + span * 0.04
        ax.set_xlim(min(d.min_beta.min(), 0) - span * 0.05, tally_x + span * 0.10)
        for i, (agreement, retained) in enumerate(zip(d.sign_agreement, d.fits_retained)):
            ax.text(tally_x, i, f"{int(round(agreement * retained))}/{retained}",
                    va="center", ha="left", fontsize=8, color=MUTED)
        ax.set_title(f"{selection['title']}  ({selection['definition']})\n{selection['arm_label']}, order level",
                     pad=9, loc="left")
        ax.grid(axis="x", alpha=.2, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ARC, markeredgecolor="white", markersize=8,
                          label="Archaea (ARC assay)"),
               plt.Line2D([], [], marker="o", ls="", color=BAC, markeredgecolor="white", markersize=8,
                          label="Bacteria (BAC assay)")]
    fig.tight_layout(w_pad=4, rect=(0, 0.10, 1, 1))
    fig.supxlabel("Mean log-contrast coefficient over 6 outer fits (pale bar = range across fits)",
                  y=0.072, fontsize=9.5, color=MUTED)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.030))
    fig.text(0.5, 0.005, "Right-hand counts: outer fits agreeing on the sign / outer fits where the block "
                         "survived training-partition prevalence filtering.",
             ha="center", fontsize=8.5, color=MUTED)
    return savefig(fig, out, "01_taxon_coefficients")


def figure_archaea(coeff: pd.DataFrame, out: Path) -> list:
    """Every archaeal block, every outer fit. The question is whether the
    methanogen orders separate or move together."""
    style()
    rule = coeff[coeff.selection.eq(RULE) & coeff.domain.eq("ARC") & coeff.classified]
    order = ["Methanosarciniales__b5", "Methanosarciniales__b2", "Methanosarciniales__b3",
             "Methanosarciniales__b4", "Methanosarciniales__b1", "Methanomicrobiales",
             "Methanomassiliicoccales", "Methanofastidiosales", "Methanomethyliales",
             "Bathyarchaeia", "Woesearchaeales__b1", "Woesearchaeales__b2"]
    order = [b for b in order if b in set(rule.block)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, sharex=True)
    for ax, selection in zip(axes, SELECTED):
        d = rule[rule.config.eq(selection["key"])]
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        for i, block in enumerate(order):
            values = d[d.block.eq(block)].beta
            ax.scatter(values, np.full(len(values), i), s=42, color=DOMAIN_COLOR["ARC"],
                       alpha=.72, edgecolor="white", linewidth=1.2, zorder=3)
            if len(values):
                ax.scatter([values.mean()], [i], s=120, marker="|", color=INK, zorder=4)
        ax.set_yticks(range(len(order)), order)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_title(f"{selection['title']} — {selection['arm_label']}", pad=8, loc="left")
        ax.set_xlabel("Per-fit coefficient (dot); mean (tick)")
        ax.grid(axis="x", alpha=.2, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Archaeal order blocks across the six outer fits", x=0.012, ha="left", y=1.04, fontsize=12.5)
    fig.tight_layout()
    return savefig(fig, out, "02_archaeal_blocks")


def figure_cross_flag(shared: pd.DataFrame, out: Path) -> list:
    style()
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    lim = max(shared[["mean_beta_acid_high", "mean_beta_ammonia"]].abs().max()) * 1.18
    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    for domain, d in shared.groupby("domain"):
        ax.scatter(d.mean_beta_ammonia, d.mean_beta_acid_high, s=48, color=DOMAIN_COLOR[domain],
                   edgecolor="white", linewidth=1.3, zorder=4,
                   label=f"{'Archaea' if domain == 'ARC' else 'Bacteria'} ({len(d)} blocks)")
    # Label the extremes, laddered down the right margin so nothing collides.
    labelled = shared.reindex(
        (shared.mean_beta_acid_high.abs() + shared.mean_beta_ammonia.abs()).sort_values(ascending=False).index).head(8)
    labelled = labelled.sort_values("mean_beta_acid_high", ascending=False)
    step = lim * 0.105
    cursor = {True: lim * 0.92, False: lim * 0.92}  # one ladder per margin
    for _, row in labelled.iterrows():
        side = row.mean_beta_ammonia >= 0  # label in the margin the point is nearest
        y = row.mean_beta_acid_high
        cursor[side] = min(cursor[side], y) if cursor[side] - step > y else cursor[side] - step
        ax.annotate(short(row.block, 28), xy=(row.mean_beta_ammonia, y),
                    xytext=(lim * 0.99 if side else -lim * 0.99, cursor[side]),
                    ha="right" if side else "left", va="center", fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=.9, shrinkA=2, shrinkB=4))
    r = shared.mean_beta_ammonia.corr(shared.mean_beta_acid_high)
    substantial = shared[(shared.mean_beta_acid_high.abs() > 0.015) & (shared.mean_beta_ammonia.abs() > 0.015)]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Mean coefficient — ammonia toxicity (shared hybrid)")
    ax.set_ylabel("Mean coefficient — acid high (ARC-only fusion)")
    ax.set_title(f"Two flags, two models, one correlated contrast\n"
                 f"Pearson r = {r:.2f} over {len(shared)} shared blocks; "
                 f"{int(substantial.same_sign.sum())}/{len(substantial)} of the blocks\n"
                 f"that are substantial in both (|mean beta| > 0.015) agree in sign",
                 loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=.18, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID)
    fig.tight_layout()
    return savefig(fig, out, "03_cross_flag_agreement")


# --- report -----------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT / "data")
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results")
    args = parser.parse_args()
    data, out = args.data_dir, args.results_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)

    checks = {"snapshot": verify_snapshot(data)}
    catalog = block_catalog(data)

    # Cohort identity, QC site grouping and trained labels all come from the runs.
    cohort = None
    for selection in SELECTED:
        pred = read_run(data, selection["key"], "order", "predictions.csv")
        pred = pred[pred.target.eq(selection["flag"])].drop_duplicates("sample_id")
        part = pred.set_index("sample_id")[["site_id", "y"]].rename(columns={"y": selection["key"]}).sort_index()
        cohort = part if cohort is None else cohort.join(part[[selection["key"]]], validate="one_to_one")
        assert cohort.site_id.equals(part.site_id), "the two runs disagree on site assignment"
    chem = chemistry(data, cohort)
    checks["cohort"] = {"n_samples": len(cohort), "n_sites": int(chem.site_id.nunique()),
                        "labels_match_trained": True}

    rank = ranking(data)
    fits = fit_inventory(data)
    coeff = coefficients(data, catalog)
    abundance = abundance_context(data, catalog, chem)
    taxa = taxon_summary(coeff, abundance)
    groups = group_catalog(coeff)
    recurrent = recurrent_groups(groups)
    contrasts = domain_contrasts(coeff)
    shared = cross_flag(taxa)
    overlap_table, overlap = label_overlap(chem)

    # --- integrity checks ---------------------------------------------------
    rule_fits = fits[fits.selection.eq(RULE)]
    checks["fits"] = {"n_fits": len(rule_fits), "all_ok": bool(rule_fits.status.eq("ok").all()),
                      "max_equality_residual": float(rule_fits.equality_residual.max()),
                      "max_group_reconstruction_error": float(rule_fits.group_reconstruction_error.max())}
    zero_sum = coeff[coeff.selection.eq(RULE)].groupby(
        ["config", "repeat", "outer_fold", "domain"]).beta.sum().abs().max()
    checks["zero_sum_max_abs_residual"] = float(zero_sum)
    checks["arc_only_restriction_holds"] = bool(
        fits[fits.config.eq("acid_high") & fits.selection.eq(RULE)].bac_n_edges.eq(0).all())
    checks["shared_hybrid_fuses_both_domains"] = bool(
        fits[fits.config.eq("ammonia") & fits.selection.eq(RULE)][["arc_n_edges", "bac_n_edges"]].sum().gt(0).all())

    reproduced = []
    for selection in SELECTED:
        pred = read_run(data, selection["key"], "order", "predictions.csv")
        pred = pred[pred.target.eq(selection["flag"]) & pred.model.eq(selection["model"])
                    & pred.selection.eq(RULE)]
        for (repeat, fold), d in pred.groupby(["repeat", "outer_fold"]):
            saved = fits[(fits.config == selection["key"]) & (fits.selection == RULE)
                         & (fits.repeat == repeat) & (fits.outer_fold == fold)].iloc[0]
            got = metrics(d)
            reproduced.append(dict(config=selection["key"], repeat=repeat, outer_fold=fold,
                                   mcc_difference=abs(got["mcc"] - saved.test_mcc),
                                   f2_difference=abs(got["f2"] - saved.test_f2)))
    reproduced = pd.DataFrame(reproduced)
    checks["metric_reproduction"] = {"n_fits": len(reproduced),
                                     "max_mcc_difference": float(reproduced.mcc_difference.max()),
                                     "max_f2_difference": float(reproduced.f2_difference.max())}

    splits_isolated = 0
    for selection in SELECTED:
        splits = read_run(data, selection["key"], "order", "splits.csv")
        splits = splits[splits.target.eq(selection["flag"])]
        for _, d in splits.groupby(["repeat", "outer_fold"]):
            test = set(d[d.role.eq("test")].site_id)
            assert test.isdisjoint(d[d.role.eq("train")].site_id)
            splits_isolated += 1
    checks["site_isolated_splits"] = splits_isolated

    # --- tables -------------------------------------------------------------
    evidence = pd.DataFrame(EVIDENCE, columns=["domain", "block", "role", "basis", "source", "note"])
    written = {}
    for name, frame in [("selection_ranking", rank), ("fit_inventory", fits),
                        ("taxon_coefficients", taxa), ("group_catalog", groups),
                        ("recurrent_groups", recurrent), ("domain_contrasts", contrasts),
                        ("cross_flag_agreement", shared), ("functional_annotation", evidence),
                        ("unassigned_lineages", pd.DataFrame(CONTEXT, columns=["domain", "lineage", "source", "note"])),
                        ("background_literature", pd.DataFrame(BACKGROUND, columns=["claim", "source", "statement"])),
                        ("abundance_context", abundance),
                        ("metric_reproduction", reproduced),
                        ("block_catalog", catalog)]:
        path = out / "tables" / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.12g")
        written[name] = path

    figures = (figure_coefficients(taxa, out / "figures")
               + figure_archaea(coeff, out / "figures")
               + figure_cross_flag(shared, out / "figures"))

    report = build_report(rank, fits, taxa, groups, recurrent, shared, overlap, overlap_table, evidence, chem)
    (out / "REPORT.md").write_text(report)

    manifest = {"tables": {k: sha(v) for k, v in written.items()},
                "figures": {p.name: sha(p) for p in figures},
                "report": sha(out / "REPORT.md")}
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    checks["scripts"] = {p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    checks["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                          "matplotlib": matplotlib.__version__}
    (out / "validation.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))


def build_report(rank, fits, taxa, groups, recurrent, shared, overlap, overlap_table, evidence, chem) -> str:
    from report_text import compose
    return compose(rank, fits, taxa, groups, recurrent, shared, overlap, overlap_table, evidence, chem,
                   SELECTED=SELECTED, RULE=RULE, md=md, ZERO=ZERO)


if __name__ == "__main__":
    main()

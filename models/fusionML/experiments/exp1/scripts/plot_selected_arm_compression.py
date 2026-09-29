"""Compression figure for one selected configuration per flag and fusion arm.

A focused companion to figure 07's upper row: instead of every resolution, it
draws only the chosen (resolution, model) per arm, for the flags listed in
SELECTION. Reads the run directories directly and recomputes the means, so the
drawn numbers cannot drift from the fits.

    python fusionMLv2/exp1/scripts/plot_selected_arm_compression.py

Writes results/figures/08_selected_arm_compression.{png,pdf}. This figure is
NOT produced by generate_results.py; run this script to refresh it.
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
import os
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = _REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fusionmlv2-exp1-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Run directories per flag. acid_high was fit separately from the other flags.
RUN_DIR = {
    "warning_acid_high": ROOT / "results/classification result/fusionML/real_results_domain_fusion_acidhigh_{resolution}_exp1_20260922",
    "__default__": ROOT / "results/classification result/fusionML/real_results_domain_fusion_{resolution}_exp1_20260922",
}
TITLES = {"warning_acid_high": "Acid high (pH > 8.0)", "warning_ammonia_toxicity": "Ammonia toxicity",
          "warning_acid_base_balance": "Acid–base balance", "warning_biogas_quality": "Biogas quality"}
RESOLUTION_LABELS = {"order": "Order", "family": "Family", "genus": "Genus", "asv": "ASV"}
ARM_LABELS = {"fusion_arc_only": "ARC-only fusion", "fusion_bac_only": "BAC-only fusion", "both": "Both domains fused"}
RULE = "min_loss"

# The configuration drawn for each flag and arm: (resolution, model), each the
# highest-MCC combination for that arm under the min_loss rule.
SELECTION = {
    "warning_acid_high": [
        ("fusion_arc_only", "order", "fusion_arc_only"),
        ("fusion_bac_only", "genus", "fusion_bac_only"),
        ("both", "asv", "domain_adaptive_hybrid"),
    ],
    "warning_ammonia_toxicity": [
        ("fusion_arc_only", "order", "fusion_arc_only"),
        ("fusion_bac_only", "order", "fusion_bac_only"),
        ("both", "order", "shared_hybrid"),
    ],
}

# Repo-standard marks: neutral backdrop for the pre-fusion feature count, then
# the two domains stacked. Validated as a categorical pair (normal-vision
# deltaE 30.8; worst CVD deltaE 23.2, both above the 15 / 8 floors).
RETAINED, ARC, BAC = "#dce5ec", "#2563a6", "#d16b19"
INK, MUTED = "#1f2933", "#52606d"


def measure(flag, resolution, model):
    """Mean counts over the available outer fits for one configuration."""
    base = Path(str(RUN_DIR.get(flag, RUN_DIR["__default__"])).format(resolution=resolution))
    rows = pd.read_csv(base / "rows.csv")
    counts = pd.read_csv(base / "domain_counts.csv")
    keys = ["target", "repeat", "outer_fold", "model", "selection"]
    ok = rows[rows.status.eq("ok") & rows.target.eq(flag) & rows.model.eq(model) & rows.selection.eq(RULE)]
    assert len(ok), f"no fits for {flag}/{resolution}/{model}"
    wide = counts.pivot_table(index=keys, columns="domain", values="active_groups").reset_index()
    d = ok.merge(wide, on=keys, validate="one_to_one")
    assert d.ARC.add(d.BAC).eq(d.n_active_groups).all()
    return dict(n_fits=len(d), retained=d.retained_taxa.mean(), groups=d.n_active_groups.mean(),
                arc=d.ARC.mean(), bac=d.BAC.mean(),
                reduction=100 * (1 - d.n_active_groups / d.retained_taxa).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results")
    args = parser.parse_args()
    out = args.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    data = {flag: [dict(arm=arm, resolution=res, model=model, **measure(flag, res, model))
                   for arm, res, model in picks] for flag, picks in SELECTION.items()}
    for flag, rows in data.items():
        for r in rows:
            print(f"{TITLES[flag]:<22s} {ARM_LABELS[r['arm']]:<20s} {RESOLUTION_LABELS[r['resolution']]:<7s} "
                  f"{r['groups']:5.1f}/{r['retained']:5.1f}  ARC {r['arc']:4.1f} / BAC {r['bac']:4.1f}  "
                  f"{r['reduction']:5.1f}%  ({r['n_fits']} fits)")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.titlesize": 12, "figure.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED})
    flags = list(SELECTION)
    fig, axes = plt.subplots(1, len(flags), figsize=(6.6 * len(flags), 3.6), squeeze=False)
    axes = axes[0]
    xmax = max(r["retained"] for rows in data.values() for r in rows) * 1.22
    gap = xmax * 0.004  # ~2px surface gap between the two stacked segments
    for ax, flag in zip(axes, flags):
        rows = data[flag]
        y = np.arange(len(rows))
        ax.barh(y, [r["retained"] for r in rows], height=.62, color=RETAINED, label="Retained features", zorder=1)
        ax.barh(y, [r["arc"] for r in rows], height=.62, color=ARC, label="Active ARC groups", zorder=2)
        ax.barh(y, [r["bac"] for r in rows], height=.62, left=[r["arc"] + gap for r in rows],
                color=BAC, label="Active BAC groups", zorder=2)
        for i, r in enumerate(rows):
            # Count carries the emphasis; the reduction is the derived quantity.
            ax.text(r["groups"] + gap + xmax * .014, i, f"{r['groups']:.1f}",
                    va="center", fontsize=10, color=INK, fontweight="medium")
            ax.text(r["groups"] + gap + xmax * .105, i, f"({r['reduction']:.0f}% fewer)",
                    va="center", fontsize=9, color=MUTED)
        ax.set_yticks(y, [f"{ARM_LABELS[r['arm']]}\n{RESOLUTION_LABELS[r['resolution']]}" for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(TITLES[flag], pad=10)
        ax.set_xlabel("Mean active coefficient groups over 6 outer fits")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.16, zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(.5, -.10))
    fig.suptitle("Compression at each arm's best configuration\n"
                 "Bar = active groups split by domain; pale backdrop = features retained after prevalence filtering; "
                 "label = groups and reduction vs retained", y=1.10, fontsize=12)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"08_selected_arm_compression.{suffix}", dpi=180, bbox_inches="tight",
                    **({"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}))
    plt.close(fig)
    print(f"\nWrote {out}/08_selected_arm_compression.png and .pdf")


if __name__ == "__main__":
    main()

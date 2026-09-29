"""DeepPhylo on fusionML's retained ASV set, one single-label model per flag.

Ablation for the parameter-count comparison: how much of DeepPhylo's
performance survives when it is fed the same ~78 ASVs fusionML keeps, and
trained one flag at a time (n_labels=1) instead of all six jointly.

Two things change versus run_site_grouped_cv_benchmark.py's ASV arm:

  1. Features. fusionML's prevalence filter (>=`detection_count` reads in
     >=40% of TRAINING samples and >=40% of TRAINING sites, per domain --
     fusionML/preprocessing.py:DomainPreprocessor.fit) is recomputed inside
     every fold from raw counts and applied to X and to the patristic matrix.
     Refitting per fold is what keeps this honest: a filter fit on all 134
     samples would leak test-fold prevalence into the feature set.
     Each domain's retained columns are then renormalised to sum to 1, which
     is what building a 78-ASV table from scratch would have produced.
  2. Labels. One model per flag, so fc_pred is 32->1 rather than 32->6.
     All six flags still go into evaluate_cv's label matrix, because
     site_grouped_cv stratifies on it -- handing it a narrower matrix would
     change the folds and break comparability with the as-built ASV run.
     So each fold trains six independent single-label models and their
     columns are concatenated back into one (n, 6) score array.

Everything else -- splits, pooling, MCC threshold tuning, repeats -- is
evaluate_multilabel.evaluate_cv(), unchanged, as the project requires.

    python Phylospec/multi_models/run_deepphylo_fusionml_features.py
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
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYLOSPEC_DIR = os.path.join(HERE, "..")
ROOT = str(_REPO_ROOT)
sys.path.insert(0, os.path.abspath(ROOT))
from Phylospec.evaluate_multilabel import evaluate_cv  # noqa: E402

CSV_DATA_DIR = os.path.join(HERE, "input_for_all_models_asv")
METADATA_PATH = os.path.join(ROOT, "datasets/raw", "final", "metadata.csv")
# The round/No-renamed cache fusionML itself reads (inputs.arc_counts/bac_counts),
# whose column labels already match the SampleID convention used here.
COUNTS = {d: os.path.join(ROOT, "models/fusionML", "real_data_cache", f"{d}_table_filtered_roundno.tsv")
          for d in ("ARC", "BAC")}
DP_SCRIPT = os.path.join(HERE, "DeepPhylo", "deepphylo_classification_multi_label_bce.py")
OUT_DIR = os.path.join(HERE, "results", "deepphylo_fusionml_features")
PY = sys.executable

# fusionML's preprocessing block (fusionML/configs/real_smoke_classification_phylo.yaml)
DETECTION_COUNT = 5
MIN_SAMPLE_PREVALENCE, MIN_DETECTED_SAMPLES = 0.40, 5
MIN_SITE_PREVALENCE, MIN_DETECTED_SITES = 0.40, 3
# Benchmark protocol, matching run_site_grouped_cv_benchmark.py exactly.
N_SPLITS, N_REPEATS, SEED, POS_WEIGHT_CAP = 5, 5, 42, 10.0
NN_EPOCHS, NN_PATIENCE = 60, 10
DOMAIN = "merged"


def load_inputs():
    csv_dir = os.path.join(CSV_DATA_DIR, f"Warnings_{DOMAIN}")
    table = pd.read_csv(os.path.join(csv_dir, "table.csv"))
    X = np.load(os.path.join(csv_dir, "X.npy"))
    C = np.load(os.path.join(csv_dir, "c.npy"))
    flags = [n.replace("warning_", "") for n in
             open(os.path.join(csv_dir, "label_names.txt")).read().split()]
    features = [c for c in table.columns if c != "SampleID" and c not in flags]
    assert X.shape == (len(table), len(features)) and C.shape == (len(features), len(features))

    meta = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    sites = table["SampleID"].astype(str).map(dict(zip(meta["SampleID"].astype(str), meta["Site"]))).values
    assert not pd.isna(sites).any()

    # Raw counts for the prevalence filter, aligned to X's rows and columns.
    raw, domain_of = {}, {}
    for domain, path in COUNTS.items():
        counts = pd.read_csv(path, sep="\t", index_col=0).T
        counts.index = counts.index.astype(str)
        mine = [f for f in features if f in counts.columns]
        for f in mine:
            domain_of[f] = domain
        raw[domain] = counts.reindex(index=table["SampleID"].astype(str), columns=mine).fillna(0).to_numpy(float)
    assert len(domain_of) == len(features), f"{len(features) - len(domain_of)} features in neither count table"
    index_of = {d: [i for i, f in enumerate(features) if domain_of[f] == d] for d in COUNTS}
    y = table[flags].to_numpy(int)
    return dict(table=table, X=X, C=C, y=y, flags=flags, features=features,
                sites=sites, raw=raw, index_of=index_of)


def retained_mask(data, train_idx):
    """fusionML's training-only prevalence filter, per domain."""
    sites = data["sites"][train_idx]
    n, n_sites = len(train_idx), len(set(sites))
    need_samples = max(MIN_DETECTED_SAMPLES, math.ceil(MIN_SAMPLE_PREVALENCE * n))
    need_sites = max(MIN_DETECTED_SITES, math.ceil(MIN_SITE_PREVALENCE * n_sites))
    mask = np.zeros(len(data["features"]), bool)
    for domain, cols in data["index_of"].items():
        detected = data["raw"][domain][train_idx] >= DETECTION_COUNT
        by_site = pd.DataFrame(detected).groupby(sites).any().to_numpy()
        keep = (detected.sum(0) >= need_samples) & (by_site.sum(0) >= need_sites)
        mask[np.asarray(cols)[keep]] = True
    return mask


def subset(data, mask):
    """Columns kept, each domain renormalised to sum to 1 within the subset."""
    X = data["X"][:, mask].copy()
    for cols in data["index_of"].values():
        local = np.flatnonzero(mask[np.asarray(cols)])
        if not len(local):
            continue
        block = np.searchsorted(np.flatnonzero(mask), np.asarray(cols)[local])
        total = X[:, block].sum(axis=1, keepdims=True)
        X[:, block] = np.divide(X[:, block], total, out=np.zeros_like(X[:, block]), where=total > 0)
    return X, data["C"][np.ix_(mask, mask)]


def make_fit_predict(data, scratch_root, trace):
    """One independent single-label DeepPhylo per flag, columns concatenated."""

    def fit_predict(tr, va, te, pos_weight):
        mask = retained_mask(data, tr)
        trace.append(dict(n_features=int(mask.sum()),
                          **{f"n_{d.lower()}": int(mask[np.asarray(c)].sum()) for d, c in data["index_of"].items()}))
        X, C = subset(data, mask)
        val_cols, test_cols = [], []
        for col, flag in enumerate(data["flags"]):
            y = data["y"][:, [col]]
            with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
                for name, idx in [("train", tr), ("val", va), ("eval", te)]:
                    np.save(os.path.join(tmp, f"X_{name}.npy"), X[idx])
                    np.save(os.path.join(tmp, f"Y_{name}.npy"), y[idx])
                np.save(os.path.join(tmp, "c.npy"), C)
                with open(os.path.join(tmp, "label_names.txt"), "w") as handle:
                    handle.write(flag + "\n")
                scores = os.path.join(tmp, "scores")
                subprocess.run([PY, DP_SCRIPT, "--data_dir", tmp, "--val_dir", tmp, "--domain", DOMAIN,
                                "--pos-weight-cap", str(POS_WEIGHT_CAP), "--scores-out-dir", scores,
                                "--epochs", str(NN_EPOCHS), "--patience", str(NN_PATIENCE)],
                               check=True, cwd=HERE)
                val_cols.append(np.load(os.path.join(scores, "val_scores.npy")).reshape(len(va)))
                test_cols.append(np.load(os.path.join(scores, "test_scores.npy")).reshape(len(te)))
        return np.column_stack(val_cols), np.column_stack(test_cols)
    return fit_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags", nargs="+", default=["acid_high", "ammonia_toxicity"])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scratch_root = os.path.join(args.out_dir, "_scratch")
    os.makedirs(scratch_root, exist_ok=True)

    data = load_inputs()
    print(f"merged ASV inputs: {data['X'].shape[0]} samples x {len(data['features'])} features "
          f"({len(data['index_of']['ARC'])} ARC / {len(data['index_of']['BAC'])} BAC), "
          f"{len(set(data['sites']))} sites")
    trace = []
    print(f"\nTraining {len(data['flags'])} independent single-label models per fold "
          f"({N_SPLITS} splits x {N_REPEATS} repeats): {data['flags']}")
    result = evaluate_cv(data["y"], data["sites"], make_fit_predict(data, scratch_root, trace),
                         label_names=data["flags"], n_splits=N_SPLITS, n_repeats=N_REPEATS,
                         seed=SEED, pos_weight_cap=POS_WEIGHT_CAP)
    per_label = pd.DataFrame(result["per_label"])
    per_label.insert(0, "model", "DeepPhylo_singlelabel_fusionML_features")
    per_label.insert(0, "domain", DOMAIN)
    per_label.to_csv(os.path.join(args.out_dir, f"results_{DOMAIN}_DeepPhylo_singlelabel.csv"), index=False)
    feats = pd.DataFrame(trace)
    feats.to_csv(os.path.join(args.out_dir, f"features_{DOMAIN}.csv"), index=False)
    print(f"\nretained features per fold: mean {feats.n_features.mean():.1f} "
          f"({feats.n_features.min()}-{feats.n_features.max()}); "
          f"ARC {feats.n_arc.mean():.1f}, BAC {feats.n_bac.mean():.1f}, {len(feats)} folds")
    summary = {}
    for flag in args.flags:
        row = per_label[per_label.label.eq(flag)].iloc[0]
        summary[flag] = {k: float(row[k]) for k in
                         ["n_pos", "mcc", "mcc_lo", "mcc_hi", "f2", "f2_lo", "f2_hi", "auroc", "auprc", "auprc_lift"]}
        print(f"  {flag:18s} MCC {row.mcc:.3f} [{row.mcc_lo:.3f}, {row.mcc_hi:.3f}]   "
              f"F2 {row.f2:.3f} [{row.f2_lo:.3f}, {row.f2_hi:.3f}]   AUROC {row.auroc:.3f}")
    summary["_features_per_fold_mean"] = float(feats.n_features.mean())
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(dict(protocol=dict(n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED,
                                     epochs=NN_EPOCHS, patience=NN_PATIENCE, n_labels=1,
                                     feature_filter="fusionML DomainPreprocessor, refit per training fold"),
                       flags=summary), handle, indent=2)
    shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()

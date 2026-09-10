"""
Run the 5-fold cross-validated version of the multi-label warning benchmark:
CNN, MetaDR, PMCNN, RF and DeepPhylo, each trained with BCEWithLogitsLoss +
per-flag pos_weight (the ablation showed this matters), across ARC/BAC/merged.

Folds come from build_cv_folds.py (must be run first). This script runs each
model on each fold, then aggregates per-domain mean +/- std across folds for
Accuracy / MCC / ROC-AUC / AUPR.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.join(HERE, "cv_folds")
STATIC_DATA_DIR = os.path.join(HERE, "input_for_all_models")  # for domain-level phylogeny.nwk / PMCNN_list.csv
RESULTS_DIR = os.path.join(HERE, "results", "cv")
DOMAINS = ["ARC", "BAC", "merged"]
MODELS = ["CNN", "MetaDR", "PMCNN", "RF", "DeepPhylo"]
METRICS = ["accuracy", "mcc", "roc_auc", "aupr"]
N_SPLITS = 5

PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for domain in DOMAINS:
        static_dir = os.path.join(STATIC_DATA_DIR, f"Warnings_{domain}")
        tree = os.path.join(static_dir, "phylogeny.nwk")
        pmcnn_list = os.path.join(static_dir, "PMCNN_list.csv")

        for fold in range(N_SPLITS):
            fold_csv_dir = os.path.join(CV_DIR, f"Warnings_{domain}", f"fold_{fold}")
            fold_dp_dir = os.path.join(CV_DIR, f"DeepPhylo_warnings_{domain}", f"fold_{fold}")
            train, test = os.path.join(fold_csv_dir, "train.csv"), os.path.join(fold_csv_dir, "test.csv")

            run([PY, os.path.join(HERE, "RF", "RF-multilabel.py"),
                 "--train", train, "--test", test, "--domain", domain,
                 "--out", os.path.join(RESULTS_DIR, f"RF_{domain}_fold{fold}.json")])

            run([PY, os.path.join(HERE, "CNN", "CNN-multilabel.py"),
                 "--train", train, "--test", test, "--domain", domain,
                 "--out", os.path.join(RESULTS_DIR, f"CNN_{domain}_fold{fold}.json")])

            run([PY, os.path.join(HERE, "PMCNN", "PMCNN-multilabel.py"),
                 "--train", train, "--test", test, "--list", pmcnn_list, "--domain", domain,
                 "--out", os.path.join(RESULTS_DIR, f"PMCNN_{domain}_fold{fold}.json")])

            run([PY, os.path.join(HERE, "MetaDR", "MetaDR-multilabel.py"),
                 "--train", train, "--test", test, "-t", tree, "--domain", domain,
                 "--out", os.path.join(RESULTS_DIR, f"MetaDR_{domain}_fold{fold}.json")])

            run([PY, os.path.join(HERE, "DeepPhylo", "deepphylo_classification_multi_label_bce.py"),
                 "--data_dir", fold_dp_dir, "--domain", domain,
                 "--out", os.path.join(RESULTS_DIR, f"DeepPhylo_{domain}_fold{fold}.json")])


def build_tables():
    tables = {}
    for domain in DOMAINS:
        rows = []
        for model in MODELS:
            fold_macros = []
            for fold in range(N_SPLITS):
                path = os.path.join(RESULTS_DIR, f"{model}_{domain}_fold{fold}.json")
                fold_macros.append(json.load(open(path))["macro"])
            row = {"model": model}
            for metric in METRICS:
                vals = [fm[metric] for fm in fold_macros]
                row[f"{metric}_mean"] = float(np.nanmean(vals))
                row[f"{metric}_std"] = float(np.nanstd(vals))
            rows.append(row)
        tables[domain] = rows

    csv_path = os.path.join(RESULTS_DIR, "cv_summary.csv")
    with open(csv_path, "w") as f:
        header = "domain,model," + ",".join(f"{m}_mean,{m}_std" for m in METRICS)
        f.write(header + "\n")
        for domain, rows in tables.items():
            for row in rows:
                vals = ",".join(f"{row[f'{m}_mean']:.4f},{row[f'{m}_std']:.4f}" for m in METRICS)
                f.write(f"{domain},{row['model']},{vals}\n")
    print(f"Wrote {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "cv_summary.md")
    with open(md_path, "w") as f:
        for domain, rows in tables.items():
            f.write(f"## {domain} (5-fold CV, mean +/- std)\n\n")
            f.write("| Model | Accuracy | MCC | ROC-AUC | AUPR |\n")
            f.write("|---|---|---|---|---|\n")
            for row in rows:
                cells = " | ".join(f"{row[f'{m}_mean']:.4f} +/- {row[f'{m}_std']:.4f}" for m in METRICS)
                f.write(f"| {row['model']} | {cells} |\n")
            f.write("\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    run_all()
    build_tables()

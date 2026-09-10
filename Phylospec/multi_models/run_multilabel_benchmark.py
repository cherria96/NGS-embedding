"""
Run CNN, MetaDR, PMCNN, RF and DeepPhylo on multi-label (5-flag) warning
prediction across the ARC, BAC and merged domains, then build per-domain
comparison tables (rows = model, columns = accuracy / MCC / ROC-AUC / AUPR,
macro-averaged over the 5 flags).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
DATA_DIR = os.path.join(HERE, "input_for_all_models")
DEEPPHYLO_DATA_DIR = os.path.join(HERE, "DeepPhylo", "data")
DOMAINS = ["ARC", "BAC", "merged"]
METRICS = ["accuracy", "mcc", "roc_auc", "aupr"]

PY = sys.executable


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for domain in DOMAINS:
        d = os.path.join(DATA_DIR, f"Warnings_{domain}")
        train, test = os.path.join(d, "train.csv"), os.path.join(d, "test.csv")
        tree = os.path.join(d, "phylogeny.nwk")
        pmcnn_list = os.path.join(d, "PMCNN_list.csv")

        run([PY, os.path.join(HERE, "RF", "RF-multilabel.py"),
             "--train", train, "--test", test, "--domain", domain,
             "--out", os.path.join(RESULTS_DIR, f"RF_{domain}.json")])

        run([PY, os.path.join(HERE, "CNN", "CNN-multilabel.py"),
             "--train", train, "--test", test, "--domain", domain,
             "--out", os.path.join(RESULTS_DIR, f"CNN_{domain}.json")])

        run([PY, os.path.join(HERE, "PMCNN", "PMCNN-multilabel.py"),
             "--train", train, "--test", test, "--list", pmcnn_list, "--domain", domain,
             "--out", os.path.join(RESULTS_DIR, f"PMCNN_{domain}.json")])

        run([PY, os.path.join(HERE, "MetaDR", "MetaDR-multilabel.py"),
             "--train", train, "--test", test, "-t", tree, "--domain", domain,
             "--out", os.path.join(RESULTS_DIR, f"MetaDR_{domain}.json")])

        deepphylo_dir = os.path.join(DEEPPHYLO_DATA_DIR, f"warnings_{domain}")
        run([PY, os.path.join(HERE, "DeepPhylo", "deepphylo_classification_multi_label_bce.py"),
             "--data_dir", deepphylo_dir, "--domain", domain,
             "--out", os.path.join(RESULTS_DIR, f"DeepPhylo_{domain}.json")])


def build_tables():
    models = ["CNN", "MetaDR", "PMCNN", "RF", "DeepPhylo"]
    tables = {}
    for domain in DOMAINS:
        rows = []
        for model in models:
            path = os.path.join(RESULTS_DIR, f"{model}_{domain}.json")
            with open(path) as f:
                payload = json.load(f)
            row = {"model": model, **payload["macro"]}
            rows.append(row)
        tables[domain] = rows

    # CSV per domain
    for domain, rows in tables.items():
        csv_path = os.path.join(RESULTS_DIR, f"summary_{domain}.csv")
        with open(csv_path, "w") as f:
            f.write("model," + ",".join(METRICS) + "\n")
            for row in rows:
                f.write(row["model"] + "," + ",".join(f"{row[m]:.4f}" for m in METRICS) + "\n")
        print(f"Wrote {csv_path}")

    # Markdown report
    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w") as f:
        for domain, rows in tables.items():
            f.write(f"## {domain}\n\n")
            f.write("| Model | Accuracy | MCC | ROC-AUC | AUPR |\n")
            f.write("|---|---|---|---|---|\n")
            for row in rows:
                f.write(f"| {row['model']} | " + " | ".join(f"{row[m]:.4f}" for m in METRICS) + " |\n")
            f.write("\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    run_all()
    build_tables()

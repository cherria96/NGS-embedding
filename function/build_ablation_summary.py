"""Build a weighted-vs-unweighted (pos_weight ablation) comparison table
across all three domains from results/*.json and results/ablation_no_posweight/*.json."""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.path.insert(0, str(_REPO_ROOT / "models/baselines/PhylogenicDL/multi_models"))

import json
import os

HERE = str(_REPO_ROOT / "models/baselines/PhylogenicDL/multi_models")
RESULTS_DIR = os.path.join(HERE, "results")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation_no_posweight")
DOMAINS = ["ARC", "BAC", "merged"]
MODELS = ["CNN", "PMCNN", "MetaDR", "DeepPhylo"]
METRICS = ["accuracy", "mcc", "roc_auc", "aupr"]


def main():
    md_path = os.path.join(ABLATION_DIR, "ablation_summary.md")
    csv_path = os.path.join(ABLATION_DIR, "ablation_summary.csv")

    with open(md_path, "w") as md, open(csv_path, "w") as csv:
        csv.write("domain,model,loss," + ",".join(METRICS) + "\n")
        for domain in DOMAINS:
            md.write(f"## {domain}\n\n")
            md.write("| Model | Loss | Accuracy | MCC | ROC-AUC | AUPR |\n")
            md.write("|---|---|---|---|---|---|\n")
            for model in MODELS:
                weighted = json.load(open(os.path.join(RESULTS_DIR, f"{model}_{domain}.json")))["macro"]
                unweighted = json.load(open(os.path.join(ABLATION_DIR, f"{model}_{domain}.json")))["macro"]
                for label, row in (("weighted (pos_weight)", weighted), ("unweighted", unweighted)):
                    md.write(f"| {model} | {label} | " + " | ".join(f"{row[m]:.4f}" for m in METRICS) + " |\n")
                    csv.write(f"{domain},{model},{label}," + ",".join(f"{row[m]:.4f}" for m in METRICS) + "\n")
            md.write("\n")

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

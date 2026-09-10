"""Build a weighted-vs-unweighted (pos_weight ablation) comparison table
across all three domains from results/*.json and results/ablation_no_posweight/*.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
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

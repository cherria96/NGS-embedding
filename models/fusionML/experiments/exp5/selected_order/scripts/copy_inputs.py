"""Snapshot every input the selected-order interpretation reads, with hashes.

No models are fitted here and nothing under fusionML/ is modified. The two
selected configurations live in Experiment 1's domain-restricted fusion runs:

    warning_acid_high        x fusion_arc_only x order
    warning_ammonia_toxicity x shared_hybrid   x order

The other three resolutions are copied at prediction/row level only, because
the interpretation has to show *where* the two selected configurations rank
among all flag x resolution x model combinations rather than assert it.
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
from pathlib import Path
import shutil

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = _REPO_ROOT

# acid_high was fit in its own run directory; the other flags share one.
RUN_TEMPLATE = {
    "acid_high": 'results/classification result/fusionML/real_results_domain_fusion_acidhigh_{resolution}_exp1_20260922',
    "ammonia": 'results/classification result/fusionML/real_results_domain_fusion_{resolution}_exp1_20260922',
}
RESOLUTIONS = ["order", "family", "genus", "asv"]
# Full export set for the two selected (order) runs; ranking-only for the rest.
FULL_FILES = ["rows.csv", "predictions.csv", "coefficients_and_groups.csv", "domain_counts.csv",
              "filter_audit.csv", "interactions.csv", "splits.csv", "search.csv", "failures.csv",
              "run_manifest.json"]
RANKING_FILES = ["rows.csv", "predictions.csv"]

REFERENCE = {
    "reference/order/arc_mapping.csv": 'datasets/processed/genus_tree/taxlevel_autorun/order/arc_mapping.csv',
    "reference/order/bac_mapping.csv": 'datasets/processed/genus_tree/taxlevel_autorun/order/bac_mapping.csv',
    "reference/order/arc_table.csv": 'datasets/processed/genus_tree/taxlevel_autorun/order/arc_table.csv',
    "reference/order/bac_table.csv": 'datasets/processed/genus_tree/taxlevel_autorun/order/bac_table.csv',
    "reference/order/arc_tree.nwk": 'datasets/processed/genus_tree/taxlevel_autorun/order/arc_tree.nwk',
    "reference/order/bac_tree.nwk": 'datasets/processed/genus_tree/taxlevel_autorun/order/bac_tree.nwk',
    "reference/taxonomy/ARC_metadata.tsv": 'datasets/raw/qiimeresult/ARC/metadata.tsv',
    "reference/taxonomy/BAC_metadata.tsv": 'datasets/raw/qiimeresult/BAC/metadata.tsv',
    "reference/chemistry/Dat_chem.xlsx": 'datasets/raw/final/Dat_chem.xlsx',
    "fusionML/configs/real_smoke_classification_phylo.yaml": 'configs/fusionML/real_smoke_classification_phylo.yaml',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source: Path, destination: Path, data: Path, entries: list) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    entries.append({"copy": str(destination.relative_to(data)), "source": str(source.relative_to(ROOT)),
                    "sha256": sha(destination), "bytes": destination.stat().st_size})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT / "data")
    args = parser.parse_args()
    data = args.data_dir

    entries: list = []
    for flag, template in RUN_TEMPLATE.items():
        for resolution in RESOLUTIONS:
            base = ROOT / template.format(resolution=resolution)
            if not base.is_dir():
                raise SystemExit(f"missing run directory {base}")
            names = FULL_FILES if resolution == "order" else RANKING_FILES
            for name in names:
                copy(base / name, data / "runs" / f"{flag}__{resolution}" / name, data, entries)

    for destination, source in REFERENCE.items():
        copy(ROOT / source, data / destination, data, entries)

    # fusionML source is snapshotted so label/chemistry definitions are pinned;
    # the interpretation never calls the estimator.
    for module in sorted((ROOT / "models/fusionML").glob("*.py")):
        copy(module, data / "fusionML" / module.name, data, entries)

    from function.snapshots import snapshot_current_sources
    entries.extend(snapshot_current_sources(ROOT, data))

    manifest = {"experiment": "fusionMLv2/exp5/selected_order", "root": str(ROOT),
                "selected": [{"flag": "warning_acid_high", "model": "fusion_arc_only", "resolution": "order",
                              "run": "acid_high__order"},
                             {"flag": "warning_ammonia_toxicity", "model": "shared_hybrid", "resolution": "order",
                              "run": "ammonia__order"}],
                "files": entries}
    (data / "source_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"copied {len(entries)} files into {data}")


if __name__ == "__main__":
    main()

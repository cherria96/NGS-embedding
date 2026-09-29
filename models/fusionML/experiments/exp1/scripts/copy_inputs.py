"""Snapshot existing fusionML outputs; no fitting and no source-file changes."""
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
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
DOMAIN_FUSION_RESOLUTIONS = ["order", "family", "genus", "asv"]
FILES = [
    "investigation_20260921/regularization_paths.csv",
    "investigation_20260921/saved_compression.csv",
    "investigation_20260921/saved_results.csv",
    "investigation_20260921/qc_groups_comparison.csv",
    "investigation_20260921/learning_qc/rows.csv",
    "investigation_20260921/learning_qc/predictions.csv",
    "investigation_20260921/learning_qc/pooled_metrics.csv",
    "investigation_20260921/learning_qc/splits.csv",
    "investigation_20260921/learning_qc/manifest.json",
    "investigation_20260921/REPORT.md",
    "investigation_20260921/mechanism_probes.py",
    "investigation_20260921/learning_curves.py",
    "configs/real_smoke_classification_phylo.yaml",
    "evaluate.py",
    "groups.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=_REPO_ROOT / "models/fusionML")
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT / "data")
    parser.add_argument("--include-asv", action="store_true", help="Also copy the completed ASV-tree run")
    parser.add_argument("--include-domain-fusion", action="store_true",
                        help="Also copy the completed ARC-only/BAC-only/both fusion run")
    args = parser.parse_args()
    files = list(FILES)
    if args.include_asv:
        run = args.source_root / "real_results_asv_exp1_20260922"
        asv_manifest = json.loads((run / "run_manifest.json").read_text())
        assert asv_manifest.get("completed_tasks") == 12, "ASV run is not complete"
        for name, expected in asv_manifest["output_hashes"].items():
            assert digest(run / name) == expected, name
        files += [f"real_results_asv_exp1_20260922/{name}" for name in sorted(asv_manifest["output_hashes"])]
        files.append("real_results_asv_exp1_20260922/run_manifest.json")
    if args.include_domain_fusion:
        copied_resolutions = []
        for resolution in DOMAIN_FUSION_RESOLUTIONS:
            relative = f"real_results_domain_fusion_{resolution}_exp1_20260922"
            run = args.source_root / relative
            if not (run / "run_manifest.json").is_file():
                continue
            fusion_manifest = json.loads((run / "run_manifest.json").read_text())
            assert fusion_manifest.get("completed_tasks") == 18, f"{relative} is not complete"
            assert fusion_manifest.get("resolution") == resolution, relative
            for name, expected in fusion_manifest["output_hashes"].items():
                assert digest(run / name) == expected, f"{relative}/{name}"
            files += [f"{relative}/{name}" for name in sorted(fusion_manifest["output_hashes"])]
            files.append(f"{relative}/run_manifest.json")
            copied_resolutions.append(resolution)
        if not copied_resolutions:
            raise FileNotFoundError("No completed domain-restricted fusion run found")
        print("Domain-restricted fusion resolutions:", ", ".join(copied_resolutions))
    missing = [p for p in files if not (args.source_root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    records = []
    for relative in files:
        source, destination = args.source_root / relative, args.data_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sha = digest(source)
        assert digest(destination) == sha, relative
        role = "current code/config context, not certified original training version" if source.suffix in {".py", ".yaml"} else "saved output or protocol document"
        records.append({"source": f"fusionML/{relative}", "copy": relative,
                        "sha256": sha, "bytes": destination.stat().st_size, "role": role})
    from function.snapshots import snapshot_current_sources
    records.extend(snapshot_current_sources(_REPO_ROOT, args.data_dir, prefix="source_context"))
    manifest = {"copied_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Experiment 1 reanalysis of existing outputs; no model training",
                "files": records}
    (args.data_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Copied and SHA-256 verified {len(records)} files into {args.data_dir}")


if __name__ == "__main__":
    main()

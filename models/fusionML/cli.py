"""fusionML CLI: validate / diagnose / train / predict (design-spec item 17)."""
from __future__ import annotations


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))
_repository_sys.modules.setdefault("fusionML.cli", _repository_sys.modules[__name__])
_repository_sys.modules.setdefault("models.fusionML.cli", _repository_sys.modules[__name__])


import argparse
import json
import sys

from function import pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fusionML", description=(
        "Domain-constrained log-contrast fusion regression for AD microbiome data "
        "(docs/superpowers/specs/ad-microbial-model-design.md)."
    ))
    p.add_argument("--config", required=True, help="path to a YAML config (see configs/*.yaml)")
    p.add_argument("--mode", required=True, choices=["validate", "diagnose", "train", "predict"])
    p.add_argument("--model-path", default=None, help="predict mode: path to a saved final_model.pkl")
    p.add_argument("--output", default=None, help="predict mode: output CSV path for predictions")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "validate":
        report = pipeline.run_validate(args.config)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("ready_to_run") else 1

    if args.mode == "diagnose":
        pipeline.run_diagnose(args.config)
        return 0

    if args.mode == "train":
        pipeline.run_train(args.config)
        return 0

    if args.mode == "predict":
        if not args.model_path or not args.output:
            parser.error("--mode predict requires --model-path and --output")
        preds = pipeline.run_predict(args.config, args.model_path, args.output)
        print(f"wrote {len(preds)} predictions -> {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

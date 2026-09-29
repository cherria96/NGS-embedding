#!/usr/bin/env python3
"""Run the existing microbiome workflows from one repository entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = ('validate', 'preprocess', 'graph', 'investigate', 'train', 'predict', 'interpret', 'ablate', 'all')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, default='validate')
    parser.add_argument('--config', type=Path, help='fusionML YAML configuration')
    parser.add_argument('--model', choices=('fusionML', 'TaxlevelML', 'PhylogenicDL', 'all'), default='fusionML')
    parser.add_argument('--model-path', type=Path, help='saved fusionML model for predict/interpret')
    parser.add_argument('--output', type=Path, help='prediction CSV, or output directory for descriptive stages')
    parser.add_argument('--baseline-args', default='', help='quoted arguments forwarded to the baseline benchmark')
    parser.add_argument('--dry-run', action='store_true', help='print the stages and baseline commands without executing')
    return parser


def baseline_commands(model, extra):
    scripts = {
        'TaxlevelML': ROOT / 'models/baselines/TaxlevelML/run_taxlevel_baseline_benchmark.py',
        'PhylogenicDL': ROOT / 'models/baselines/PhylogenicDL/multi_models/run_site_grouped_cv_benchmark.py',
    }
    names = list(scripts) if model == 'all' else [model]
    return [[sys.executable, str(scripts[name]), *shlex.split(extra)] for name in names if name in scripts]


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    stages = ['validate', 'preprocess', 'graph', 'investigate', 'train', 'interpret', 'ablate'] if args.stage == 'all' else [args.stage]
    if args.model not in ('fusionML', 'all') and args.stage != 'train':
        parser.error('standalone baseline workflows support --stage train; use --model all for the full workflow')
    if args.model in ('fusionML', 'all') and not args.config:
        parser.error('--config is required for fusionML stages')
    if args.stage == 'predict' and (not args.model_path or not args.output):
        parser.error('--stage predict requires --model-path and --output')
    if args.stage == 'all' and args.output:
        parser.error('--output is stage-specific; configure training output_dir in the YAML for --stage all')
    if args.dry_run:
        print(json.dumps({'stages': stages, 'model': args.model,
                          'config': str(args.config) if args.config else None,
                          'baseline_commands': baseline_commands(args.model, args.baseline_args) if 'train' in stages else []}, indent=2))
        return 0

    for stage in stages:
        print(f'[{stage}]', flush=True)
        if stage == 'train':
            for command in baseline_commands(args.model, args.baseline_args):
                subprocess.run(command, cwd=ROOT, check=True)
        if args.model not in ('fusionML', 'all'):
            continue
        from function import pipeline
        from function.workflow import run_descriptive, run_interpretation, run_ablation_summary
        config = str(args.config.resolve())
        if stage == 'validate':
            result = pipeline.run_validate(config)
            print(json.dumps(result, indent=2, default=str))
            if not result.get('ready_to_run'):
                return 1
        elif stage in ('preprocess', 'graph'):
            run_descriptive(config, stage, args.output)
        elif stage == 'investigate':
            run_descriptive(config, stage, args.output)
        elif stage == 'train':
            result = pipeline.run_train(config)
            info = result.get('final_model_info', {})
            if info.get('status') != 'ok':
                print(json.dumps(info, indent=2, default=str), file=sys.stderr)
                return 1
            args.model_path = Path(info['model_path'])
        elif stage == 'predict':
            pipeline.run_predict(config, str(args.model_path.resolve()), str(args.output.resolve()))
        elif stage == 'interpret':
            run_interpretation(config, args.model_path, args.output)
        elif stage == 'ablate':
            run_ablation_summary(config, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

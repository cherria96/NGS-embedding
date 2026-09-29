"""Snapshot the inputs and existing model exports for biological interpretation."""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path
import hashlib
import json
import shutil

ROOT = _REPO_ROOT
DATA = Path(__file__).resolve().parents[1] / 'data'


def main():
    paths = [ROOT / 'configs/fusionML/real_smoke_classification_phylo.yaml',
             ROOT / 'datasets/raw/final/Dat_chem.xlsx']
    for domain in ['ARC', 'BAC']:
        paths.append(ROOT / f'datasets/fusionML/real_data_cache/{domain}_table_filtered_roundno.tsv')
        paths.extend(ROOT / f'datasets/raw/qiimeresult/{domain}/{name}' for name in ['metadata.tsv', 'rooted-tree.nwk'])
        paths.extend(ROOT / f'datasets/processed/genus_tree/taxlevel_autorun/genus/{domain.lower()}_{name}'
                     for name in ['mapping.csv', 'tree.nwk'])
    # Copy canonical input paths too: refreshed configs use the new layout.
    paths.extend(path.resolve() for path in list(paths))
    for suffix in ['interaction', 'acid_base_balance_interaction', 'biogas_quality_interaction']:
        source = ROOT / f'results/classification result/fusionML/real_results_classification_phylo_{suffix}/smoke'
        paths.extend(source / name for name in ['coefficients_and_groups.csv', 'graph_edges.csv',
            'interaction_coefficients_outer_folds.csv', 'balance_definitions_outer_folds.csv',
            'results.csv', 'oof_predictions.csv', 'search_results.csv', 'fold_manifest.csv', 'run_manifest.json'])
    source = ROOT / 'results/classification result/fusionML/real_results_asv_exp1_20260922'
    paths.extend(source.glob('*.csv'))
    paths.append(source / 'run_manifest.json')
    # Capture the exact source used by the new fits, independently of later repository edits.
    paths.extend((ROOT / 'models/fusionML').glob('*.py'))
    paths.append(ROOT / 'function/investigation_20260921/learning_curves.py')
    records = []
    for src in sorted(set(paths)):
        relative = src.relative_to(ROOT)
        dst = DATA / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest
        records.append(dict(path=str(relative), source=str(relative), sha256=digest, bytes=src.stat().st_size))
    from function.snapshots import snapshot_current_sources
    records.extend(snapshot_current_sources(ROOT, DATA))
    (DATA / 'source_manifest.json').write_text(json.dumps({'files': records}, indent=2) + '\n')
    print(f'Copied and verified {len(records)} input files into {DATA}')


if __name__ == '__main__':
    main()

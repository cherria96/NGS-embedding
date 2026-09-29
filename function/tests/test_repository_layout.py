"""Regression checks for relocation, compatibility, and the root workflow."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize('old,new', [
    ('config', 'configs.fusionml'),
    ('graphs', 'datasets.graph.fusion_graphs'),
    ('preprocessing', 'datasets.datapreprocess.preprocessing'),
    ('pipeline', 'function.pipeline'),
    ('model_io', 'function.model_io'),
    ('estimator', 'models.fusionML.estimator'),
    ('search', 'models.fusionML.search'),
])
def test_saved_model_compatibility_uses_same_module(old, new):
    # A fresh interpreter catches import-order bugs hidden by pytest collection.
    subprocess.run([sys.executable, '-c',
                    f'import importlib; import models.fusionML; a=importlib.import_module("fusionML.{old}"); '
                    f'b=importlib.import_module("{new}"); assert a is b'], cwd=ROOT, check=True)


def test_config_alias_has_identical_paths():
    from configs.fusionml import Config
    canonical = Config.from_yaml(str(ROOT / 'configs/fusionML/synthetic_demo.yaml'))
    legacy = Config.from_yaml(str(ROOT / 'models/fusionML/configs/synthetic_demo.yaml'))
    for key in ['arc_counts', 'bac_counts', 'metadata', 'arc_tree', 'bac_tree']:
        assert canonical.resolved_input_path(key) == legacy.resolved_input_path(key)
        assert Path(canonical.resolved_input_path(key)).is_file()
    assert canonical.resolved_output_dir() == legacy.resolved_output_dir()


def test_full_root_workflow_on_synthetic_data(tmp_path, monkeypatch):
    import main
    from datasets.datapreprocess.synthetic import generate_synthetic_dataset, write_synthetic_dataset_to_dir
    from function import workflow
    from configs.fusionml import Config

    paths = write_synthetic_dataset_to_dir(generate_synthetic_dataset(
        n_sites=8, visits_per_site=3, p_arc=4, p_bac=5, seed=123,
        depth_arc=4000, depth_bac=6000, missing_visit_prob=0), str(tmp_path / 'inputs'))
    raw = {
        'seed': 123, 'task': 'regression', 'resolution': 'asv', 'inputs': paths,
        'columns': {'sample_id': 'sample_id', 'site_id': 'site_id', 'target': 'methane_yield_SYNTHETIC'},
        'preprocessing': {'detection_count': 1, 'min_detected_samples': 2, 'min_detected_sites': 2,
                          'min_sample_prevalence': 0, 'min_site_prevalence': 0},
        'graph': {'bootstrap_sites': 3, 'min_joint_samples': 2, 'min_joint_sites': 2, 'support_threshold': 0.1},
        'cv': {'outer_folds': 2, 'inner_folds': 2, 'families': ['intercept_only', 'domain_adaptive_hybrid']},
        'search': {'candidate_budget_per_family': 1, 'lambda_s': [0], 'lambda_r': [0.1],
                   'lambda_f': [0.1], 'alpha_arc': [0.5], 'alpha_bac': [0.5]},
        'output_dir': str(tmp_path / 'training'),
    }
    config = tmp_path / 'workflow.yaml'
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    assert main.main(['--stage', 'all', '--config', str(config)]) == 0
    assert (tmp_path / 'datasets/processed/fusionML/workflow/preprocess/metadata.csv').is_file()
    graph_dir = tmp_path / 'datasets/processed/fusionML/workflow/graph'
    assert (graph_dir / 'ARC_abundance_edges.csv').is_file()
    assert json.loads((graph_dir / 'summary.json').read_text())['resolution'] == 'asv'
    for category in ['data investigation', 'model interpretation', 'ablation analysis']:
        assert (tmp_path / 'results' / category / 'workflow/report.md').is_file()
    run = Path(Config.from_yaml(str(config)).resolved_output_dir())
    assert (run / 'final_model.pkl').is_file()
    output = tmp_path / 'prediction.csv'
    assert main.main(['--stage', 'predict', '--config', str(config), '--model-path',
                      str(run / 'final_model.pkl'), '--output', str(output)]) == 0
    predicted = pd.read_csv(output)
    assert predicted['scorable'].all()
    assert predicted['prediction'].notna().all()


def test_failed_validation_stops_workflow(monkeypatch, tmp_path):
    import main
    from function import pipeline, workflow
    monkeypatch.setattr(pipeline, 'run_validate', lambda _: {'ready_to_run': False})
    def unexpected(*args, **kwargs):
        pytest.fail('A downstream stage ran after validation failed')
    monkeypatch.setattr(workflow, 'run_descriptive', unexpected)
    assert main.main(['--stage', 'all', '--config', str(tmp_path / 'missing.yaml')]) == 1


def test_refreshed_source_snapshot_is_self_contained(tmp_path):
    import shutil
    from function.snapshots import snapshot_current_sources
    snapshot_current_sources(ROOT, tmp_path)
    code = (
        'import sys, pathlib; root=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(root)); '
        'from function import pipeline; from fusionML.config import Config; '
        'from fusionML.search import Candidate; import inspect; '
        'assert pathlib.Path(pipeline.__file__).is_relative_to(root); '
        'assert pathlib.Path(inspect.getfile(Config)).is_relative_to(root); '
        'assert pathlib.Path(inspect.getfile(Candidate)).is_relative_to(root)'
    )
    subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path)], cwd=tmp_path, check=True)


def test_taxlevel_input_builder_help_is_read_only():
    import hashlib
    inputs = ROOT / 'datasets/baselines/input_for_all_models_taxlevel'
    before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
              for p in inputs.glob('*/Warnings_*/table.csv')}
    result = subprocess.run([sys.executable, str(ROOT / 'datasets/datapreprocess/build_taxlevel_inputs.py'), '--help'],
                            cwd='/tmp', check=True, capture_output=True, text=True)
    assert 'usage:' in result.stdout
    assert before == {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in before}

"""Stage adapters around the existing fusionML implementation.

Full-cohort preprocessing and graphs are descriptive exports only. Training
continues to fit preprocessing and graph construction within each CV partition.
"""
from __future__ import annotations

import json
from pathlib import Path

from configs.fusionml import Config
from datasets.datapreprocess.preprocessing import JointPreprocessor
from datasets.graph.fusion_graphs import build_abundance_graph, build_phylogenetic_graph
from function import pipeline
from function.model_io import load_model
from models.fusionML.groups import extract_groups
from models.fusionML.search import FAMILY_SPECS

ROOT = Path(__file__).resolve().parents[1]


def _destination(config_path, category, output=None):
    path = Path(output) if output else ROOT / 'results' / category / Path(config_path).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_descriptive(config_path, stage, output=None):
    cfg = Config.from_yaml(config_path)
    if stage == 'investigate':
        out = _destination(config_path, 'data investigation', output)
        cfg.raw['output_dir'] = str(out.resolve())
        cfg.raw['smoke'] = False
        result = pipeline.run_diagnose(cfg)
        (out / 'report.md').write_text(
            '# Data investigation\n\nDescriptive full-cohort diagnostics; these exports are not used in model selection.\n\n'
            'See [diagnostics](diagnostics/) for prevalence, log-ratio variance, and pseudocount sensitivity.\n'
        )
        return result
    if stage not in ('preprocess', 'graph'):
        raise ValueError(f'Unknown descriptive stage: {stage}')
    inputs = pipeline.load_all_inputs(cfg)
    counts_a, counts_b = pipeline.tables_for_resolution(inputs, cfg['resolution'])
    cohort = inputs.cohort_ids
    sites = inputs.metadata.loc[cohort, 'site_id'].to_dict()
    preprocessor, kept = JointPreprocessor.fit(counts_a.loc[cohort], counts_b.loc[cohort], sites, cfg['preprocessing'])
    transformed = preprocessor.transform(counts_a.loc[kept], counts_b.loc[kept], cfg['preprocessing']['pseudocount'])
    out = Path(output) if output else ROOT / 'datasets/processed/fusionML' / Path(config_path).stem / stage
    out.mkdir(parents=True, exist_ok=True)
    summary = {'scope': 'descriptive full cohort; never reused by cross-validation',
               'config': str(Path(config_path).resolve()), 'resolution': cfg['resolution'],
               'n_cohort': len(cohort), 'n_retained_samples': len(kept)}
    if stage == 'preprocess':
        inputs.metadata.loc[kept].to_csv(out / 'metadata.csv')
        for domain, frame in [('ARC', transformed['log_a']), ('BAC', transformed['log_b'])]:
            frame.to_csv(out / f'{domain}_log_composition.csv')
    else:
        trees = (inputs.tree_arc, inputs.tree_bac) if cfg['resolution'] == 'asv' else (
            (inputs.rank_tree_arc, inputs.rank_tree_bac) if inputs.rank_used == cfg['resolution'] else (None, None))
        for index, (domain, counts, logs, tree) in enumerate(zip(
                ['ARC', 'BAC'], [counts_a, counts_b], [transformed['log_a'], transformed['log_b']], trees)):
            features = list(logs.columns)
            phylo = build_phylogenetic_graph(tree, domain, features, cfg['graph']['k_phylogeny'])
            detection = counts.loc[kept, features] >= cfg['preprocessing']['detection_count']
            abundance = build_abundance_graph(logs, detection, sites, cfg['graph'], cfg['seed'] + index, domain)
            phylo.edges.to_csv(out / f'{domain}_phylogeny_edges.csv', index=False)
            abundance.edges.to_csv(out / f'{domain}_abundance_edges.csv', index=False)
            summary[domain] = {'features': len(features), 'phylogeny_edges': len(phylo.edges),
                               'abundance_edges': len(abundance.edges), 'tree_available': tree is not None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(out)
    return summary


def run_interpretation(config_path, model_path=None, output=None):
    import pandas as pd

    cfg = Config.from_yaml(config_path)
    path = Path(model_path) if model_path else Path(cfg.resolved_output_dir()) / 'final_model.pkl'
    model = load_model(str(path))
    out = _destination(config_path, 'model interpretation', output)
    summaries = []
    for domain, features, beta, original_beta, edges in [
        ('ARC', model.feature_ids_a, model.beta_a_star, model.beta_a_orig, model.graph_edges_a),
        ('BAC', model.feature_ids_b, model.beta_b_star, model.beta_b_orig, model.graph_edges_b),
    ]:
        indexed = pipeline._mixed_edges_to_index_pairs(edges, features)
        groups = extract_groups(domain, features, beta, indexed, cfg['compression']['coefficient_tolerance'])
        rows = [{'feature_id': feature, 'group': groups.group_of_feature[feature],
                 'coefficient_fit_scale': float(beta[i]), 'coefficient_original_scale': float(original_beta[i]),
                 'residual_fit_scale': groups.residuals[feature]}
                for i, feature in enumerate(features)]
        pd.DataFrame(rows).to_csv(out / f'{domain}_coefficients.csv', index=False)
        pd.DataFrame([{'group': group, 'size': len(members), 'members': ';'.join(members),
                       'group_coefficient_fit_scale': groups.group_theta[group]}
                      for group, members in groups.group_members.items()]).to_csv(out / f'{domain}_groups.csv', index=False)
        summaries.append(f'- {domain}: {len(features)} features in {len(groups.group_members)} coefficient groups.')
    if model.gamma_orig.size:
        pd.DataFrame({'product_index': range(len(model.gamma_orig)),
                      'coefficient_original_scale': model.gamma_orig}).to_csv(out / 'interaction_coefficients.csv', index=False)
    (out / 'report.md').write_text(
        f'# Model interpretation\n\nModel: `{path.resolve()}`\n\nFamily: `{model.family}`; task: `{model.task}`.\n\n'
        + '\n'.join(summaries) + '\n\nCSV files retain group membership, coefficients, and residuals. '
        'Groups describe fitted predictive effects; biological function requires separate evidence.\n'
    )
    print(out)
    return out


def run_ablation_summary(config_path, output=None):
    import pandas as pd

    cfg = Config.from_yaml(config_path)
    source = Path(cfg.resolved_output_dir()) / 'results.csv'
    results = pd.read_csv(source)
    out = _destination(config_path, 'ablation analysis', output)
    details = []
    for name in results['family']:
        spec = FAMILY_SPECS[name]
        details.append({'family': name, 'uses_graph': spec.uses_graph, 'graph_source': spec.graph_source,
                        'alpha_mode': spec.alpha_mode, 'fusion_domains': spec.fusion_domains,
                        'uses_interaction': spec.uses_interaction})
    summary = results.merge(pd.DataFrame(details), on='family', validate='one_to_one')
    summary.to_csv(out / 'family_comparison.csv', index=False)
    note = ('Only one family was evaluated; no between-family ablation is available.' if len(results) < 2 else
            'This table compares the configured families from the same run. Interpret each contrast using its graph, domain, and interaction settings.')
    (out / 'report.md').write_text(
        f'# Model ablation analysis\n\nSource: `{source.resolve()}`\n\n{note}\n\n'
        'See [family_comparison.csv](family_comparison.csv). No additional models were fitted by this reporting stage. '
        'Set `cv.families` in the config and run training to evaluate additional controls.\n'
    )
    print(out)
    return summary

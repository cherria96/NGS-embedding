"""Read-only audit of saved fusionML and external benchmark artifacts."""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path
import json
import sys

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from configs.fusionml import Config
from function.pipeline import load_all_inputs, derive_seed
from models.fusionML.search import FAMILY_SPECS, generate_candidates
from datasets.datapreprocess.chemistry_targets import build_chemistry_derived_metadata
from function.folds import make_site_grouped_folds, sample_ids_for_sites

OUT = _REPO_ROOT / "results/data investigation/fusionML"


def main():
    results, compression, grids, graph_rows = [], [], [], []
    for path in sorted((ROOT / 'models/fusionML').glob('real_results*/smoke/results.csv')):
        manifest = json.loads((path.parent / 'run_manifest.json').read_text())
        cfg = manifest['config']
        tags = dict(run=path.parent.parent.name, target=cfg['columns']['target'],
                    resolution=cfg['resolution'], timestamp=manifest['timestamp_iso'])
        results.extend([{**tags, **r} for r in pd.read_csv(path).to_dict('records')])
        coef_path = path.parent / 'coefficients_and_groups.csv'
        if coef_path.exists() and coef_path.stat().st_size > 1:
            coef = pd.read_csv(coef_path)
            for (family, fold), d in coef.groupby(['family', 'outer_fold']):
                active_groups = d.loc[d.group_mean_beta.abs() > 1e-6, ['domain', 'group_id']].drop_duplicates()
                compression.append({**tags, 'family': family, 'outer_fold': fold,
                                    'retained_taxa': len(d), 'active_taxa': int((d.beta_standardized.abs() > 1e-6).sum()),
                                    'all_groups': len(d[['domain','group_id']].drop_duplicates()),
                                    'active_groups': len(active_groups),
                                    'maximum_residual': d.residual.abs().max()})
        edge_path = path.parent / 'graph_edges.csv'
        if edge_path.exists() and edge_path.stat().st_size > 1:
            ed = pd.read_csv(edge_path)
            for (family, fold, domain), d in ed.groupby(['family','outer_fold','domain']):
                graph_rows.append({**tags, 'family': family, 'outer_fold': fold, 'domain': domain,
                                   'union_edges': len(d), 'positive_edges': int((d.mixed_weight > 0).sum()),
                                   'phylo_edges': int(d.phylo_weight.notna().sum()),
                                   'abundance_edges': int(d.empirical_weight.notna().sum())})
        sr = pd.read_csv(path.parent / 'search_results.csv')
        for (family, fold), d in sr.groupby(['family','outer_fold']):
            base = d[['lambda_s','lambda_r','lambda_f','alpha_arc','alpha_bac']].drop_duplicates()
            grids.append({**tags, 'family': family, 'outer_fold': fold, 'candidates': len(d),
                          'base_candidates': len(base), 'invalid': int((~d.valid).sum()),
                          'lambda_s_values': sorted(base.lambda_s.unique().tolist()),
                          'lambda_r_values': sorted(base.lambda_r.unique().tolist()),
                          'lambda_f_values': sorted(base.lambda_f.unique().tolist()),
                          'no_fusion_candidates': int((base.lambda_f == 0).sum()),
                          'best_inner_score': d.loc[d.valid, 'mean_site_score' if 'mean_site_score' in d else 'mean_site_mse'].min()})
    for name, rows in [('saved_results',results), ('saved_compression',compression),
                       ('saved_grids',grids), ('saved_graphs',graph_rows)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)

    cfg = Config.from_yaml(str(ROOT / 'configs/fusionML/real_smoke_classification_phylo.yaml'))
    inp = load_all_inputs(cfg)
    old = pd.read_csv(ROOT / 'datasets/raw/final/metadata.csv', encoding='utf-8-sig').set_index('SampleID')
    cohort = inp.cohort_ids
    label_rows, changed = [], []
    for flag in ['ammonia_toxicity','acid_base_balance','biogas_quality','acid_accumulation','buffer_capacity']:
        target = 'warning_' + flag
        fresh, _ = build_chemistry_derived_metadata(cfg.resolved_path(cfg['chemistry']['workbook']), cfg['chemistry'], target, [])
        fresh = fresh.loc[cohort]
        ids = fresh.index.intersection(old.index)
        diff = fresh.loc[ids,'target'].astype(float) != old.loc[ids,target].astype(float)
        pos = fresh[fresh.target == 1]
        label_rows.append(dict(target=target,n=len(fresh),sites=fresh.site_id.nunique(),
                               positives=int(fresh.target.sum()),positive_sites=pos.site_id.nunique(),
                               old_positive_matched=int(old.loc[ids,target].sum()),changed_labels=int(diff.sum())))
        for sid in ids[diff]:
            changed.append(dict(sample_id=sid,target=target,old=old.loc[sid,target],current=fresh.loc[sid,'target']))
    pd.DataFrame(label_rows).to_csv(OUT/'label_audit.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'changed_labels.csv',index=False)

    bench = []
    for folder in ['site_grouped_cv_taxlevel','site_grouped_cv','holdout']:
        for path in (ROOT/'models/baselines/PhylogenicDL/multi_models/results'/folder).glob('results_*.csv'):
            if any(x in path.name for x in ['GLMNET','XGBoost','PhyloSpec','MetaDR']):
                bench.extend([dict(source=str(path.relative_to(ROOT)), protocol=folder, **r) for r in pd.read_csv(path).to_dict('records')])
    pd.DataFrame(bench).to_csv(OUT/'legacy_benchmarks.csv',index=False)
    ml_files=list((ROOT/'models/baselines/TaxlevelML').glob('results_*_macro/per_warning_metrics.csv'))
    if ml_files:
        pd.concat([pd.read_csv(p).assign(rank=p.parent.name) for p in ml_files],ignore_index=True).to_csv(
            OUT/'legacy_ml_metrics.csv',index=False)
    site_map=inp.metadata.loc[cohort,'site_id'].to_dict()
    qc_rows=[]
    for repeat in [0,1]:
        outer=make_site_grouped_folds(site_map,3,cfg['seed']+repeat)
        for fold in range(3):
            te=set(outer.sites_in_fold(fold));tr=set(outer.train_sites_for_outer_fold(fold))
            for a,b in [('BSI','BSIb'),('DGY','DGYb')]:
                qc_rows.append(dict(stage='outer',repeat=repeat,fold=fold,family='',pair=a+'/'+b,
                                    crosses=(a in te and b in tr) or (b in te and a in tr)))
            if repeat: continue
            ids=sample_ids_for_sites(site_map,list(tr));inner_map={s:site_map[s] for s in ids}
            for family in ['genus_ridge','phylogeny_only','domain_adaptive_hybrid','domain_adaptive_interaction']:
                inner=make_site_grouped_folds(inner_map,2,derive_seed(cfg['seed'],'inner_folds',family,fold))
                for a,b in [('BSI','BSIb'),('DGY','DGYb')]:
                    qc_rows.append(dict(stage='inner',repeat=repeat,fold=fold,family=family,pair=a+'/'+b,
                                        crosses=a in inner.assignment and b in inner.assignment and inner.assignment[a]!=inner.assignment[b]))
    pd.DataFrame(qc_rows).to_csv(OUT/'qc_pair_partition_audit.csv',index=False)
    print(pd.DataFrame(label_rows).to_string(index=False))
    print('Saved artifact audit to', OUT)


if __name__ == '__main__':
    main()

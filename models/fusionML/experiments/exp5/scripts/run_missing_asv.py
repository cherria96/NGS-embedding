"""Fill ASV model/flag gaps using the frozen fusionML implementation and inputs.

Existing Experiment 1 fits are reused separately by generate_results.py.
No outer-test labels are used for candidate or biological-feature selection.
"""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path
import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fusionmlv2-exp2-mpl')
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
sys.path.insert(0, str(DATA))
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from fusionML import pipeline as pipe
from fusionML.config import Config
from fusionML.search import FAMILY_SPECS, generate_candidates, generate_interaction_candidates
from fusionML.folds import make_site_grouped_folds, sample_ids_for_sites
from fusionML.evaluate import per_sample_log_loss, select_threshold_via_inner_oof_f2
from fusionML.groups import extract_groups, verify_group_reconstruction
from fusionML.investigation_20260921.learning_curves import Workbench

TARGETS = ['warning_ammonia_toxicity', 'warning_acid_base_balance', 'warning_biogas_quality']
MODELS = ['fusion_arc_only', 'fusion_bac_only', 'shared_hybrid', 'domain_adaptive_hybrid', 'domain_adaptive_interaction']
OUT = BASE / 'results/model_fits/new_asv'
KEYS = ['rows', 'coefficients_and_groups', 'graph_edges', 'balances', 'interactions', 'predictions', 'splits', 'search', 'failures']
INPUTS = {}


def config(target, resolution='asv'):
    cfg = Config.from_yaml(str(DATA / 'fusionML/configs/real_smoke_classification_phylo.yaml'))
    cfg.raw['columns']['target'] = target
    cfg.raw['resolution'] = resolution
    if resolution == 'asv':
        cfg.raw['inputs']['rank_trees'] = {}
    return cfg


def initialize():
    for target in TARGETS:
        cfg = config(target)
        inp = pipe.load_all_inputs(cfg)
        inp.metadata['site_id'] = inp.metadata.site_id.replace({'BSIb': 'BSI', 'DGYb': 'DGY'})
        INPUTS[target] = cfg, inp


class CachedFits(Workbench):
    def fit_model(self, train, model, candidate):
        key = (tuple(train), model, candidate.key())
        if key not in self.fit_cache:
            self.fit_cache[key] = pipe.fit_partition(train, FAMILY_SPECS[model], candidate,
                self.inp, 'asv', self.cfg, pipe.derive_seed(self.cfg['seed'], 'matched_graph', *sorted(train)))
        return self.fit_cache[key]


def export(pf, inp, tags, result):
    transformed = pf.jp.transform(inp.counts_arc_asv.loc[pf.kept_train_ids], inp.counts_bac_asv.loc[pf.kept_train_ids], pf.pseudocount)
    reconstruction = 0.
    for domain, features, beta, graph, edges, logs in [
        ('ARC', pf.retained_a, pf.fit_result.beta_a, pf.mixed_edges_a, pf.edges_a_idx, transformed['log_a']),
        ('BAC', pf.retained_b, pf.fit_result.beta_b, pf.mixed_edges_b, pf.edges_b_idx, transformed['log_b'])]:
        group = extract_groups(domain, features, beta, edges, 1e-6)
        reconstruction = max(reconstruction, verify_group_reconstruction(logs, beta, features, group))
        for feature, b in zip(features, beta):
            gid = group.group_of_feature[feature]
            result['coefficients_and_groups'].append(dict(tags, domain=domain, feature_id=feature,
                beta=float(b), group_id=gid, group_mean_beta=group.group_mean_beta[gid],
                residual=group.residuals[feature], active=bool(abs(b)>1e-6)))
        result['graph_edges'].extend(dict(row, **tags, domain=domain) for row in graph.to_dict('records'))
    if pf.interaction_transform is not None:
        balances, interactions = pipe.interaction_artifacts_from_pf(pf, 1e-6)
        result['balances'].extend(dict(row, **tags) for row in balances.to_dict('records'))
        result['interactions'].extend(dict(row, **tags) for row in interactions.to_dict('records'))
    return reconstruction


def run_task(task):
    target, repeat, fold = task
    cfg, inp = INPUTS[target]
    sites = inp.metadata.loc[inp.cohort_ids, 'site_id'].to_dict()
    outer = make_site_grouped_folds(sites, 3, cfg['seed'] + repeat)
    full = sorted(outer.train_sites_for_outer_fold(fold))
    rng = np.random.default_rng(pipe.derive_seed(cfg['seed'], 'learning_sites', repeat, fold))
    train = sample_ids_for_sites(sites, list(rng.permutation(full)))
    test = sample_ids_for_sites(sites, outer.sites_in_fold(fold))
    imap = {s: sites[s] for s in train}
    inner = make_site_grouped_folds(imap, 2, pipe.derive_seed(cfg['seed'], 'matched_inner', repeat, fold, 1.0))
    splits = [(sample_ids_for_sites(imap, inner.train_sites_for_outer_fold(k)), sample_ids_for_sites(imap, inner.sites_in_fold(k))) for k in range(2)]
    tags = dict(target=target, repeat=repeat, outer_fold=fold, resolution='asv', selection='one_se', source='new_asv')
    result = {key: [] for key in KEYS}
    for role, ids in [('train', train), ('test', test)] + [(f'inner{k}_{role}', ids) for k, split in enumerate(splits) for role, ids in zip(['train','validation'], split)]:
        result['splits'].extend(dict(tags, role=role, sample_id=s, site_id=sites[s]) for s in ids)
    assert set(sites[s] for s in train).isdisjoint(sites[s] for s in test)
    wb = CachedFits(cfg, inp, None)
    try:
        for model in (MODELS if target == TARGETS[2] else MODELS[:2]):
            mtags = dict(tags, model=model)
            base = 'domain_adaptive_hybrid' if model == 'domain_adaptive_interaction' else model
            candidates = generate_candidates(FAMILY_SPECS[base], cfg['search'], pipe.derive_seed(cfg['seed'], 'candidates', base, fold if repeat == 0 else f'{repeat}:{fold}'))
            if model == 'domain_adaptive_interaction':
                candidates = generate_interaction_candidates(candidates, cfg['interaction'])
            scores = []
            for candidate in candidates:
                yy, pp, ss, complexity = [], [], [], []
                try:
                    for tr, va in splits:
                        pf = wb.fit_model(tr, model, candidate)
                        pred = pipe.predict_with_partition_fit(pf, va, inp, 'asv')
                        good = pred.index[pred.scorable]
                        if len(good) != len(va):
                            raise ValueError('Incomplete inner prediction coverage')
                        yy.extend(inp.metadata.loc[good, 'target']); pp.extend(pred.loc[good,'prediction']); ss.extend(inp.metadata.loc[good,'site_id'])
                        stats = pipe.count_active(pf, 1e-6)
                        complexity.append((stats['n_active_products'], stats['n_required_taxa'], stats['n_active_groups']) if model.endswith('interaction') else (stats['n_active_groups'], stats['n_active_taxa']))
                    losses = pd.Series(per_sample_log_loss(np.asarray(yy), np.asarray(pp))).groupby(np.asarray(ss)).mean().to_numpy()
                    scores.append(dict(candidate=candidate, loss=float(losses.mean()), losses=losses, complexity=tuple(np.mean(complexity, axis=0)), y=np.asarray(yy), p=np.asarray(pp), sites=ss))
                except Exception as exc:
                    result['failures'].append(dict(mtags, candidate=str(candidate), error=f'{type(exc).__name__}: {exc}'))
            if not scores:
                result['rows'].append(dict(mtags, status='unavailable', reason='No valid candidate across all inner folds'))
                continue
            best = min(scores, key=lambda x:x['loss'])
            se = float(np.std(best['losses']) / np.sqrt(len(best['losses'])))
            chosen = min([s for s in scores if s['loss'] <= best['loss'] + se], key=lambda s:(s['complexity'],s['loss'],s['candidate'].key()))
            result['search'].extend(dict(mtags,candidate=str(s['candidate']),loss=s['loss'],selected_one_se=s is chosen) for s in scores)
            pf = wb.fit_model(train, model, chosen['candidate'])
            pred = pipe.predict_with_partition_fit(pf, test, inp, 'asv')
            threshold = select_threshold_via_inner_oof_f2(chosen['y'], chosen['p'], chosen['sites'])
            reconstruction = export(pf, inp, mtags, result)
            if model == 'fusion_arc_only':
                assert not pf.edges_b_idx
            if model == 'fusion_bac_only':
                assert not pf.edges_a_idx
            assert reconstruction < 1e-8
            result['rows'].append(dict(mtags, status='ok', candidate=str(chosen['candidate']), threshold=threshold,
                n_train=len(train), n_train_sites=len(full), n_train_positive=int(inp.metadata.loc[train,'target'].sum()),
                retained_taxa=len(pf.retained_a)+len(pf.retained_b), **pipe.count_active(pf,1e-6),
                group_reconstruction_error=reconstruction, inner_loss=chosen['loss'],inner_se=se))
            result['predictions'].extend(dict(mtags,sample_id=s,site_id=sites[s],y=float(inp.metadata.loc[s,'target']),p=float(pred.loc[s,'prediction']),threshold=threshold) for s in pred.index[pred.scorable])
    finally:
        wb.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / 'parts'; checkpoint.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + (DATA/'source_manifest.json').read_bytes()).hexdigest()
    lock = OUT/'fingerprint.txt'
    if lock.exists():
        assert lock.read_text().strip() == fingerprint, 'Source changed; use a separate output directory'
    lock.write_text(fingerprint+'\n')
    tasks = [(target, repeat, fold) for target in TARGETS for repeat in range(2) for fold in range(3)]
    def part(task):
        return checkpoint / ('__'.join(map(str, task))+'.json')
    pending = [task for task in tasks if not part(task).exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        futures = {pool.submit(run_task, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            tmp = part(task).with_suffix('.tmp')
            tmp.write_text(json.dumps(result, default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
            tmp.replace(part(task))
            print('Completed',task,flush=True)
    combined = {key:[] for key in KEYS}
    for task in tasks:
        for key, rows in json.loads(part(task).read_text()).items():
            combined[key].extend(rows)
    for key, rows in combined.items():
        pd.DataFrame(rows).to_csv(OUT/f'{key}.csv',index=False)
    (OUT/'run_manifest.json').write_text(json.dumps(dict(fingerprint=fingerprint, completed_tasks=len(tasks), targets=TARGETS, models=MODELS, resolution='asv', inner_folds=2, outer_folds=3, repeats=2, selection='one_se', source_manifest_sha256=hashlib.sha256((DATA/'source_manifest.json').read_bytes()).hexdigest(), output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}),indent=2)+'\n')
    print('Completed all missing ASV fits',flush=True)


if __name__ == '__main__':
    main()

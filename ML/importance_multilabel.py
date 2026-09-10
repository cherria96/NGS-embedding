"""Training-OOF permutation importance for the frozen CV-selected model."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multilabel_models import WarningModel, TARGETS, NUMS, CATS
from train_multilabel import metrics, save_json


def permute_column(frame, column, order):
    result = frame.copy()
    result[column] = frame[column].to_numpy()[order]
    return result


def summarize(drops):
    rows = []
    for feature, part in drops.groupby('Feature', sort=False):
        row = {'Feature': feature, 'Type': 'Categorical' if feature in CATS else 'Physicochemical'}
        for metric in ['Micro-AUPR', 'Multilabel MCC']:
            values = part[metric + ' drop']
            row[metric + ' mean drop'] = values.mean()
            row[metric + ' SD'] = values.std(ddof=1)
            row[metric + ' permutation p025'] = values.quantile(.025)
            row[metric + ' permutation p975'] = values.quantile(.975)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('Micro-AUPR mean drop', ascending=False)
    table.insert(0, 'Rank', range(1, len(table)+1))
    positive = table['Micro-AUPR mean drop'].clip(lower=0)
    table['Share of positive metadata importance (%)'] = 100*positive/positive.sum() if positive.sum() else 0.
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent/'results_multilabel')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260908)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('At least two repeats are required')
    root = args.results_dir
    out = root/'physicochemical_importance'
    out.mkdir(exist_ok=True)
    # Hash original result files without evaluating their contents; confirm no modification.
    original_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in root.iterdir() if p.is_file()}
    config = json.loads((root/'run_config.json').read_text())
    selected = json.loads((root/'cv_selected_model.json').read_text())['Model']
    best = json.loads((root/f'{selected}_best_params.json').read_text())
    thresholds = np.array([best['thresholds'][t] for t in TARGETS])
    assignment = pd.read_csv(root/'split_and_cv_assignment.csv', dtype={'SampleID': str})
    train_assignment = assignment.loc[assignment.split.eq('train')].set_index('SampleID')
    train_ids = set(train_assignment.index)
    # Restrict every downstream operation to training rows; never load test predictions.
    chunks = []
    for chunk in pd.read_csv(root/'cleaned_merged.csv', dtype={'SampleID': str, 'No': str}, chunksize=32):
        chunks.append(chunk.loc[chunk.SampleID.isin(train_ids)])
    data = pd.concat(chunks).set_index('SampleID').loc[train_assignment.index]
    features = NUMS+CATS
    X = data[CATS+NUMS+[c for c in data if '::' in c]]
    y = data[TARGETS].to_numpy(int)
    groups = data.SiteGroup.to_numpy()
    assert set(data.index) == train_ids
    fold_ids = train_assignment.cv_validation_fold.to_numpy(int)
    assert train_assignment.groupby('SiteGroup').cv_validation_fold.nunique().max() == 1
    baseline = np.zeros(y.shape)
    shuffled = np.zeros((len(features), args.repeats, len(y), len(TARGETS)))
    for fold in sorted(set(fold_ids)):
        fit, valid = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        assert not set(groups[fit]) & set(groups[valid])
        model = WarningModel(selected, best['params'], config['seed']).fit(X.iloc[fit], y[fit], groups[fit])
        frame = X.iloc[valid]
        baseline[valid] = model.predict_proba(frame)
        for repeat in range(args.repeats):
            # Same row order for every feature in a repeat supports paired comparisons.
            order = np.random.default_rng(np.random.SeedSequence([args.seed, int(fold), repeat])).permutation(len(valid))
            for j, feature in enumerate(features):
                shuffled[j, repeat, valid] = model.predict_proba(permute_column(frame, feature, order))
        print(f'Completed fold {fold}: {len(fit)} fit / {len(valid)} validation samples', flush=True)
    saved = pd.read_csv(root/f'{selected}_CV_OOF_predictions.csv', index_col='SampleID').loc[data.index]
    np.testing.assert_allclose(baseline, saved[[t+'_probability' for t in TARGETS]].to_numpy(), atol=1e-12, rtol=1e-10)
    reference = metrics(y, baseline, thresholds)
    rows = []
    for j, feature in enumerate(features):
        for repeat in range(args.repeats):
            perturbed = metrics(y, shuffled[j, repeat], thresholds)
            rows.append({'Feature': feature, 'Repeat': repeat+1,
                         **{metric+' drop': reference[metric]-perturbed[metric] for metric in reference}})
    drops = pd.DataFrame(rows)
    drops.to_csv(out/'permutation_repeats.csv',index=False)
    table = summarize(drops)
    table.to_csv(out/'metadata_importance.csv',index=False)
    continuous = table[table.Type.eq('Physicochemical')].copy()
    continuous['Rank'] = range(1,len(continuous)+1)
    continuous.to_csv(out/'physicochemical_importance.csv',index=False)
    table[table.Type.eq('Categorical')].to_csv(out/'categorical_importance.csv',index=False)
    train_assignment.to_csv(out/'training_cv_assignment.csv')
    pd.DataFrame(baseline,index=data.index,columns=TARGETS).to_csv(out/'baseline_oof_probabilities.csv')
    np.savez_compressed(out/'permuted_oof_probabilities.npz', probabilities=shuffled,
                        features=np.array(features),sample_ids=np.array(data.index,dtype=str))
    for metric, name in [('Micro-AUPR','micro_aupr_importance'),('Multilabel MCC','multilabel_mcc_importance')]:
        ordered = table.sort_values(metric+' mean drop')
        fig, ax = plt.subplots(figsize=(10,7))
        colors = ['#cf8530' if t=='Categorical' else '#267b91' for t in ordered.Type]
        ax.barh(ordered.Feature,ordered[metric+' mean drop'],xerr=ordered[metric+' SD'],color=colors,capsize=3)
        ax.axvline(0,color='black',linewidth=.8)
        ax.set(xlabel=f'Decrease in pooled {metric} after permutation (mean ± SD)',
               title=f'{selected}: training CV permutation importance\nBlue: physicochemical; orange: categorical')
        fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=180);plt.close(fig)
    method = {'model': selected, 'model_params': best['params'], 'model_seed': config['seed'],
              'permutation_seed': args.seed, 'repeats': args.repeats, 'training_samples': len(y),
              'thresholds': dict(zip(TARGETS,thresholds)), 'baseline_OOF': reference,
              'ranking': 'Descending mean decrease in pooled Micro-AUPR; MCC drop also reported.',
              'permutation': 'Shuffle one raw metadata column among validation samples within each fixed fold; refit nothing after permutation. Pool all five folds before scoring. Use same random row permutation across features per repeat.',
              'uncertainty': 'SD and quantiles describe permutation randomness only, not confidence intervals over independent sites.',
              'isolation': 'Existing training folds only; fixed selected hyperparameters and thresholds; no test scoring or method tuning; original files unchanged.',
              'original_artifact_sha256': original_hashes}
    save_json(out/'method.json',method)
    lines = ['# Physicochemical contribution to the joint five-label task',
             f'Analyzed **{selected}**, the model already selected by training CV. This is model-specific predictive importance, not causation.',
             f'Used {len(y)} training samples, the existing five grouped folds, and {args.repeats} permutations per feature. No test predictions were computed.',
             '## Ranked physicochemical features',
             '| Rank | Feature | Mean Micro-AUPR drop | Mean multilabel MCC drop |',
             '|---:|---|---:|---:|']
    for r in continuous.to_dict('records'):
        lines.append(f"| {r['Rank']} | {r['Feature']} | {r['Micro-AUPR mean drop']:.5f} | {r['Multilabel MCC mean drop']:.5f} |")
    lines += ['## Interpretation',
              'Largest Micro-AUPR contributions: '+', '.join(continuous.Feature.head(3))+'.',
              'Smallest Micro-AUPR contributions: '+', '.join(continuous.Feature.tail(3))+'.',
              'The MCC ranking differs: its largest contributions are '+', '.join(continuous.sort_values('Multilabel MCC mean drop',ascending=False).Feature.head(3))+'. A low Micro-AUPR rank is not evidence of low importance under MCC; probability ranking and thresholded decisions measure different aspects of prediction.',
              'A positive drop means shuffling worsened overall prediction; near zero means little detected reliance; a negative drop means shuffling improved the score. Negative importance does not establish a protective effect.',
              'Importance is the absolute performance drop in metric units. The additional positive-importance percentage in the CSV is normalized only across these 16 metadata variables; it is not explained variance or a fraction of all model information, and excludes microbial features from its denominator.',
              '## Categorical predictors']
    for r in table[table.Type.eq('Categorical')].to_dict('records'):
        lines.append(f"- {r['Feature']}: Micro-AUPR drop {r['Micro-AUPR mean drop']:.5f}; MCC drop {r['Multilabel MCC mean drop']:.5f}.")
    lines += ['## Method and limits',
              'For each original fold, refit the selected settings on the other four folds, including the training-only abundance union, imputation, scaling and encoding. Baseline OOF probabilities were verified against the original saved OOF predictions. Shuffle one original metadata column in validation rows, including its missingness, before the saved fold transformation. Each categorical variable is shuffled as one column, keeping its encoded levels together. Pool all validation predictions before calculating either metric.',
              'Use the existing saved thresholds without reoptimization. Micro-AUPR is trapezoidal PR area on flattened sample-label probabilities. Multilabel MCC is binary MCC on flattened sample-label predictions. All other predictors remain unchanged.',
              'The ranked plot uses mean Micro-AUPR decrease; a second plot reports MCC decrease. Error bars are permutation SD, not confidence intervals. Repeated seasons and site correlation make sample-level permutations dependent; shuffling can create unrealistic combinations and does not preserve site trajectories. These are descriptive model reliance estimates, not independent-site inferential effects.',
              'Correlated chemistry variables and microbiome predictors can substitute for each other, masking importance or distributing it across variables. Training CV was already used to select settings and thresholds, so this is post-selection interpretation rather than unbiased external validation.',
              'Permutation importance does not indicate whether higher or lower values increase warning probability. No direction or causal effect is inferred, and no features are removed.',
              'The two categorical variables are shown separately from the 14 continuous physicochemical predictors. Values such as operating temperature may originally be reported as ranges; this analysis uses the established parsed values.',
              '## Reproduction',
              '`python ML/importance_multilabel.py --seed 20260908 --repeats 30` using the original modeling environment. Configuration, baseline predictions, per-repeat drops, permuted OOF probabilities, ranked CSVs and both plots are saved here.']
    (out/'report.md').write_text('\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n')
    for name,digest in original_hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
    print(continuous[['Rank','Feature','Micro-AUPR mean drop','Multilabel MCC mean drop']].to_string(index=False),flush=True)
    print('Verified original analysis files unchanged.',flush=True)

if __name__=='__main__': main()

"""Enrich results from frozen CSV artifacts, without fitting or predicting."""
from pathlib import Path
import json
import pandas as pd

def enrich(out):
    out=Path(out)
    path=out/'report.md';text=path.read_text().split('\n## Overall interpretation\n')[0]
    # Keep adjacent Markdown table rows contiguous.
    while '|\n\n|' in text:text=text.replace('|\n\n|','|\n|')
    per=pd.read_csv(out/'per_warning_metrics.csv');test=per[per.scope=='test']
    table=pd.read_csv(out/'test_performance.csv').set_index('Model')
    selected=json.loads((out/'cv_selected_model.json').read_text())['Model']
    taxa=pd.read_csv(out/'selected_taxa.csv')
    text+='\n## Overall interpretation\n\n'
    text+=f'Microbiome composition alone shows uneven predictive performance across the five warnings. The training-only joint three-metric ranking selected **{selected}**; it remains the preferred model under the prespecified rule. '
    leaders={metric:table[metric].idxmax() for metric in table.columns}
    text+='Held-out metric leaders: '+', '.join(f'{metric}: {name} ({table.loc[name,metric]:.4f})' for metric,name in leaders.items())+'. '
    text+=('One model leads all three test metrics. ' if len(set(leaders.values()))==1 else 'No model wins all three test metrics. ')
    text+='Test results describe trade-offs and do not change the CV-selected model.\n\n'
    for t,frame in test.groupby('target',sort=False):
        text+=f'{t}: {int(frame.positives.iloc[0])} test positives; MCC ranges from {frame.MCC.min():.3f} to {frame.MCC.max():.3f}. '
        best=frame.loc[frame.MCC.idxmax()]
        text+=f'The highest MCC is from {best.Model}.\n\n'
    text+='Ammonia toxicity is comparatively consistent in this run; the other warnings vary much more across models. Inspect both the per-warning ranking areas and thresholded MCC: ranking one rare positive highly need not give a reliable threshold. In particular, acid accumulation has very few examples, so even perfect test ranking is not evidence of robust generalization. The macro results do not establish dependable performance on all five warnings.\n\n'
    text+='Rare-label ranking can differ substantially between training OOF and this small holdout. No model or threshold was changed in response to test results.\n\n'
    text+='## Feature counts and label distribution\n\n'
    text+=f'The final training-derived schema has {len(taxa)} features: '+', '.join(f'{n} {d}' for d,n in taxa.groupby('domain').size().items())+'. Descriptive union counts by domain: '+str(pd.read_csv(out/'taxa_audit.csv').query('samples_above_0_1_percent > 0').groupby('domain').size().to_dict())+'.\n\n'
    d=pd.read_csv(out/'label_distribution.csv');d=d[d.scope.isin(['train','test'])]
    text+='| Warning | Train positives / samples | Test positives / samples |\n|---|---:|---:|\n'
    for t in d.target.unique():
        a=d[(d.target==t)&(d.scope=='train')].iloc[0];b=d[(d.target==t)&(d.scope=='test')].iloc[0]
        text+=f'| {t} | {a.positive}/{a.samples} ({a.positive_percent:.2f}%) | {b.positive}/{b.samples} ({b.positive_percent:.2f}%) |\n'
    warnings=pd.read_csv(out/'fit_warnings.csv')
    text+='\n## Convergence diagnostics\n\n'
    text+=str(warnings.groupby(['Model','category']).size().to_dict())+'\n\n'
    final=warnings[warnings.fold.astype(str)=='final']
    text+=f'{len(final)} warnings occurred during final fits. Iteration-limit warnings mean those estimators may not have converged; no post-test retuning was performed. Full messages are preserved in fit_warnings.csv.\n'
    text+='\n## Requirement audit\n\n'
    text+='Implemented: five-label binary relevance; Family-only genomic predictors; strict sample-wise >0.1% mask; no additional ML feature selection; grouped approximately 80:20 holdout; five grouped training folds; fold-local preprocessing/balancing and tuning; training-only thresholds; arithmetic five-label macro metrics with individual scores; six model objects; final predictions; three figures; complete feature contributions and reproducibility artifacts.\n\n'
    text+='Explicit qualifications: the mutually incompatible all-sample-union and no-test-feature-use rules are reconciled with training-only modeling unions plus a descriptive full union; GLMNET is a sklearn elastic-net analogue; some validation-fold AUPR/ROC-AUC values are mathematically undefined; the existing holdout is reused at the project level; pre-warning sampling chronology is unavailable. The analysis therefore does not claim literal satisfaction of conflicting rules or a new external validation.\n\n'
    text+='Run test_family_macro.py for five protocol unit tests and verify_family_macro.py for independent verification of all 12 saved prediction-table metrics, source masks, final/fold unions and site separation, without new test predictions. See verification.json for the completed verification status.\n'
    path.write_text(text)

if __name__=='__main__':enrich(Path(__file__).resolve().parent/'results_family_macro')

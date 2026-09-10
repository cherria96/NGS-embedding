"""Post-fit training permutation diagnostics and descriptive resolution comparison.
No model fitting, threshold selection or test predictions are performed.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib
from phylum_macro_models import TARGETS
from train_phylum_macro import metrics, METRICS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results_phylum_macro');args=parser.parse_args();out=args.output_dir
    config=json.loads((out/'run_config.json').read_text());seed=config['seed']
    assignment=pd.read_csv(out/'split_and_cv_assignment.csv').set_index('SampleID')
    merged=pd.read_csv(out/'cleaned_merged.csv',index_col='SampleID')
    train=assignment.index[assignment.split=='train'];data=merged.loc[train]
    X=data[[c for c in data if c.startswith(('BAC_P::','ARC_P::'))]];y=data[TARGETS].to_numpy(int)
    rows=[]
    for name in ['RandomForest','KNN','SVM','NNET','XGBoost','GLMNET']:
        model=joblib.load(out/f'{name}.joblib');baseline=metrics(y,model.predict_proba(X),model.thresholds)
        for c in model.selector.columns_:
            values=[]
            for repeat in range(3):
                shuffled=X.copy();shuffled[c]=np.random.default_rng(seed+repeat).permutation(shuffled[c].to_numpy())
                score=metrics(y,model.predict_proba(shuffled),model.thresholds)
                values.append([baseline[k]-score[k] for k in METRICS])
            v=np.array(values)
            rows.append({'Model':name,'feature':c,'taxon':c.split('::')[1],'domain':'Bacteria' if c.startswith('BAC') else 'Archaea','taxonomic_level':'Phylum','scope':'training_resubstitution','repeats':3,
                **{k+' decrease mean':v[:,j].mean() for j,k in enumerate(METRICS)},**{k+' decrease SD':v[:,j].std(ddof=1) for j,k in enumerate(METRICS)}})
        print('Training permutation diagnostics:',name,flush=True)
    importance=pd.DataFrame(rows);importance.to_csv(out/'permutation_importance_training.csv',index=False)
    per=pd.read_csv(out/'per_warning_metrics.csv');thresholds=pd.read_csv(out/'thresholds.csv')
    warning=per[per.scope=='test'].merge(thresholds,on=['Model','target']);warning['negatives']=warning.samples-warning.positives
    warning.to_csv(out/'test_warning_summary.csv',index=False)
    all_results=[];chosen=[]
    for level in ['family','genus','order','class','phylum']:
        p=out if level=='phylum' else out.parent/f'results_{level}_macro'
        reference=pd.read_csv(p/'split_and_cv_assignment.csv').set_index('SampleID')
        pd.testing.assert_frame_equal(assignment.sort_index(),reference.sort_index())
        cfg=json.loads((p/'run_config.json').read_text())
        assert cfg['input_sha256']==config['input_sha256']
        table=pd.read_csv(p/'test_performance.csv');table['Resolution']=level.title();all_results.append(table)
        model=json.loads((p/'cv_selected_model.json').read_text())['Model'];model='Random Forest' if model=='RandomForest' else model
        chosen.append(table[table.Model==model])
    all_results=pd.concat(all_results);all_results.to_csv(out/'resolution_comparison_all_models.csv',index=False)
    chosen=pd.concat(chosen);chosen.to_csv(out/'resolution_comparison_cv_selected.csv',index=False)
    lines=['## Supplementary interpretation and resolution comparison',
        'All five resolutions have identical source hashes, sample/site partitions and CV folds. This comparison uses frozen results only; no models or thresholds are changed. The models below were selected independently by training CV. The repeatedly inspected holdout is not external validation, and resolution selection based on it would introduce selection bias.',
        '| Resolution | CV-selected model | Test Macro MCC | Test Macro AUPR | Test Macro ROC-AUC |','|---|---|---:|---:|---:|']
    for r in chosen.to_dict('records'):lines.append(f'| {r["Resolution"]} | {r["Model"]} | {r["Macro MCC"]:.4f} | {r["Macro AUPR"]:.4f} | {r["Macro ROC-AUC"]:.4f} |')
    lines+=['','Among these CV-selected models, the held-out leaders differ by metric: '+', '.join(f'{k}: {chosen.loc[chosen[k].idxmax(),"Resolution"]}' for k in [])]
    # Reset the index because each source table retains its own row indices.
    chosen=chosen.reset_index(drop=True)
    lines[-1]='Among these CV-selected models, the held-out leaders are '+', '.join(f'{k}: {chosen.loc[chosen[k].idxmax(),"Resolution"]}' for k in METRICS)+'. These descriptive rankings do not establish superiority of a taxonomic resolution.'
    lines+=['','### Training permutation importance for all six models','',
        'Each feature was shuffled independently three times across the 109 training rows (seeds 42, 43, 44 in the default run). Importance is baseline minus shuffled macro score at frozen thresholds. No test rows were predicted or permuted. This training-resubstitution diagnostic may overstate generalizable importance; permutations can create unrealistic compositional profiles and disrupt site dependence. Repetition SD is not a confidence interval. These are predictive contributions, not statistical-significance tests or causal biological effects. No feature is removed. Complete mean/SD values for all three metrics are exported.', '']
    for (name,domain),frame in importance.groupby(['Model','domain']):
        top=frame.sort_values('Macro AUPR decrease mean',ascending=False).head(3)
        lines.append(f'**{name}, {domain} (Phylum)**: '+', '.join(f'{r["taxon"]} ({r["Macro AUPR decrease mean"]:.4f})' for r in top.to_dict('records')))
        lines.append('')
    path=out/'report.md';base=path.read_text().split('\n## Supplementary interpretation and resolution comparison')[0]
    path.write_text(base+'\n'+'\n'.join(lines)+'\n')
    config['supplement']={'permutation_scope':'training resubstitution only','repeats':3,'seeds':[seed+i for i in range(3)],'cross_resolution_assignments_verified':True,'test_prediction_calls':0}
    config['code_sha256']['supplement_phylum_macro.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':main()

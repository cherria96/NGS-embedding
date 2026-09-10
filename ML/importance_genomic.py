"""Training-CV permutation importance for genomic-only joint prediction."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from genomic_models import WarningModel, SampleAbundanceTaxa, TARGETS, probability, margin
from train_genomic import metrics, save_json


def transformed_probability(model,z):
    if model.calibrators is not None:
        return np.column_stack([probability(c,margin(m,z).reshape(-1,1)) for m,c in zip(model.models,model.calibrators)])
    return np.column_stack([probability(m,z) for m in model.models])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results_genomic')
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20260908)
    args=parser.parse_args()
    root=args.results_dir;out=root/'genomic_importance';out.mkdir(exist_ok=True)
    config=json.loads((root/'run_config.json').read_text())
    name=json.loads((root/'cv_selected_model.json').read_text())['Model']
    best=json.loads((root/f'{name}_best_params.json').read_text())
    thresholds=np.array([best['thresholds'][t] for t in TARGETS])
    assignment=pd.read_csv(root/'split_and_cv_assignment.csv',index_col='SampleID')
    assignment=assignment.loc[assignment.split.eq('train')]
    pieces=[]
    for chunk in pd.read_csv(root/'cleaned_merged.csv',chunksize=32):
        pieces.append(chunk.loc[chunk.SampleID.isin(assignment.index)])
    data=pd.concat(pieces).set_index('SampleID').loc[assignment.index]
    X=data[[c for c in data if '::' in c]];y=data[TARGETS].to_numpy(int)
    features=SampleAbundanceTaxa().fit(X).columns_
    groups=data.SiteGroup.to_numpy();folds=assignment.cv_validation_fold.to_numpy(int)
    baseline=np.zeros(y.shape)
    permuted=np.zeros((len(features),args.repeats,len(y),5))
    coverage={f:0 for f in features}
    for fold in range(1,6):
        fit,valid=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not set(groups[fit])&set(groups[valid])
        model=WarningModel(name,best['params'],config['seed']).fit(X.iloc[fit],y[fit],groups[fit])
        frame=X.iloc[valid]
        z=model.preprocessor.transform(model.selector.transform(frame))
        baseline[valid]=transformed_probability(model,z)
        columns=list(model.preprocessor.get_feature_names_out())
        orders=np.array([np.random.default_rng(np.random.SeedSequence([args.seed,fold,r])).permutation(len(valid)) for r in range(args.repeats)])
        for j,feature in enumerate(features):
            if feature not in columns:
                permuted[j,:,valid,:]=baseline[valid,None,:]
                continue
            coverage[feature]+=1
            column=columns.index(feature)
            batch=np.tile(z,(args.repeats,1))
            batch[:,column]=z[orders.ravel(),column]
            # Independent per-column masking/scaling commutes with a row permutation.
            if j==0:
                raw=frame.copy();raw[feature]=frame[feature].to_numpy()[orders[0]]
                np.testing.assert_allclose(batch[:len(valid)],model.preprocessor.transform(model.selector.transform(raw)))
            scores=transformed_probability(model,batch).reshape(args.repeats,len(valid),5)
            for r in range(args.repeats): permuted[j,r,valid]=scores[r]
        print(f'{name}: importance fold {fold}/5 finished',flush=True)
    old=pd.read_csv(root/f'{name}_CV_OOF_predictions.csv',index_col='SampleID').loc[data.index]
    np.testing.assert_allclose(baseline,old[[t+'_probability' for t in TARGETS]].to_numpy(),atol=1e-12,rtol=1e-10)
    ref=metrics(y,baseline,thresholds)
    repeats=[]
    for j,f in enumerate(features):
        for r in range(args.repeats):
            score=metrics(y,permuted[j,r],thresholds)
            repeats.append({'Feature':f,'Repeat':r+1,**{m+' drop':ref[m]-score[m] for m in ref}})
    drops=pd.DataFrame(repeats);drops.to_csv(out/'permutation_repeats.csv',index=False)
    rows=[]
    for f,p in drops.groupby('Feature',sort=False):
        block,taxon=f.split('::',1)
        row={'Feature':f,'Block':block,'Taxon':taxon,'CV folds containing feature':coverage[f]}
        for metric in ref:
            v=p[metric+' drop'];row[metric+' mean drop']=v.mean();row[metric+' SD']=v.std(ddof=1)
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('Micro-AUPR mean drop',ascending=False)
    table.insert(0,'Rank',range(1,len(table)+1));table.to_csv(out/'ranked_taxa.csv',index=False)
    np.savez_compressed(out/'permuted_oof_probabilities.npz',probabilities=permuted,features=np.array(features),sample_ids=np.array(data.index,dtype=str))
    for metric,filename in [('Micro-AUPR','micro_aupr_importance'),('Multilabel MCC','multilabel_mcc_importance')]:
        top=table.sort_values(metric+' mean drop',ascending=False).head(25).iloc[::-1]
        fig,ax=plt.subplots(figsize=(12,10))
        palette={'BAC_P':'#267b91','BAC_O':'#499a6c','ARC_O':'#ce8435','ARC_G':'#9268ab'}
        ax.barh(top.Feature,top[metric+' mean drop'],xerr=top[metric+' SD'],color=[palette[b] for b in top.Block],capsize=2)
        ax.axvline(0,color='black',lw=.7)
        ax.set(xlabel=f'Pooled {metric} decrease after permutation (mean ± SD)',title=f'{name}: genomic-only CV importance, top 25 taxa')
        fig.tight_layout();fig.savefig(out/f'{filename}.png',dpi=180);plt.close(fig)
    save_json(out/'method.json',{'model':name,'model_params':best['params'],'seed':args.seed,'repeats':args.repeats,
        'baseline_OOF':ref,'thresholds':dict(zip(TARGETS,thresholds)),
        'method':'Shuffle one taxon within validation samples of each existing fold, pool all five folds before both metrics. Columnwise mask/scaling permits equivalent batched transformed-column permutations. No renormalization. Missing fold-union features produce zero drop in that fold.',
        'interpretation':'Fixed chosen settings; training-only post-selection descriptive importance. SD measures permutation randomness, not independent-site confidence. Correlated taxa, hierarchical ranks and compositional constraints limit interpretation. Permutations can create unrealistic profiles. No causation, direction, or feature removal.'})
    lines=['# Genomic-only taxa importance',f'CV-selected model: **{name}**. All {len(features)} training-union features tested with {args.repeats} repetitions on the existing training folds. Test predictions are not used.',
        'Mean pooled Micro-AUPR decrease determines rank. The complete CSV also reports pooled MCC decrease and permutation SD.',
        '| Rank | Taxon feature | Micro-AUPR drop | MCC drop |','|---:|---|---:|---:|']
    for r in table.head(25).to_dict('records'):
        lines.append(f"| {r['Rank']} | {r['Feature']} | {r['Micro-AUPR mean drop']:.5f} | {r['Multilabel MCC mean drop']:.5f} |")
    lines += ['','Largest Micro-AUPR contributions: '+', '.join(table.Feature.head(5))+'.',
        'Largest MCC contributions: '+', '.join(table.sort_values('Multilabel MCC mean drop',ascending=False).Feature.head(5))+'.',
        'Negative importance means permutation improved the score, not that a taxon protects against warnings. Small drops indicate little detected individual reliance, not biological irrelevance.',
        'Importance is predictive association in this model, not causation or direction of effect. Related ranks and correlated/compositional taxa can share information. Shuffling does not preserve site trajectories or compositional/hierarchical consistency. Error bars are permutation SD, not confidence intervals across sites.',
        'Settings and thresholds were selected using the same training CV previously, so this is post-selection interpretation. The original splits and final models are unchanged. No taxa are removed. Displaying the top 25 is only a visualization choice; all taxa are in ranked_taxa.csv.',
        'Reproduce: `python ML/importance_genomic.py --seed 20260908 --repeats 30`.']
    (out/'report.md').write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:])+'\n')
    print(table.head(10).to_string(index=False),flush=True)

if __name__=='__main__': main()

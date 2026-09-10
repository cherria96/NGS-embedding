#!/usr/bin/env python3
"""Training-CV permutation interpretation, per binary label and model."""
from pathlib import Path
import argparse
import json
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from warning_models import WarningModel, SampleTopTaxa, TARGETS, CATS, NUMS, probability, margin

COLORS = {'Metadata: operation/context':'#557c9e','Metadata: substrate':'#db9251','Bacteria':'#469b80','Archaea':'#936db4'}

def category(feature):
    if feature.startswith('BAC_'): return 'Bacteria'
    if feature.startswith('ARC_'): return 'Archaea'
    if feature.startswith('substrate'): return 'Metadata: substrate'
    return 'Metadata: operation/context'

def ap_many(y, probabilities):
    """Average precision for many permutations, including tied probabilities."""
    if y.sum() == 0: return np.full(len(probabilities), np.nan)
    order = np.argsort(-probabilities, axis=1, kind='stable')
    p = np.take_along_axis(probabilities, order, axis=1)
    truth = np.broadcast_to(y, probabilities.shape)
    truth = np.take_along_axis(truth, order, axis=1)
    n = len(y); positions = np.broadcast_to(np.arange(n), p.shape)
    end = np.concatenate([p[:,:-1] != p[:,1:], np.ones((len(p),1),bool)],axis=1)
    ends = np.minimum.accumulate(np.where(end,positions,n)[:,::-1],axis=1)[:,::-1]
    precision = np.cumsum(truth,axis=1)/(positions+1)
    return (truth*np.take_along_axis(precision,ends,axis=1)).sum(axis=1)/y.sum()

def encoded_groups(model):
    prep = model.preprocessor
    numeric_columns = prep.transformers_[0][2]
    result = {c:[i] for i,c in enumerate(numeric_columns)}
    start = len(numeric_columns)
    encoder = prep.named_transformers_['categorical'].named_steps['encode']
    for c, cats in zip(CATS,encoder.categories_):
        result[c] = list(range(start,start+len(cats))); start += len(cats)
    return result

def predict_encoded(model,z):
    if model.calibrators is None:
        return np.column_stack([probability(m,z) for m in model.models])
    return np.column_stack([probability(c,margin(m,z).reshape(-1,1)) for m,c in zip(model.models,model.calibrators)])

def plot_rank(table, name, target, out):
    top=table.head(20).iloc[::-1]
    fig,ax=plt.subplots(figsize=(11,8))
    ax.barh(range(len(top)),top.importance_mean,xerr=top.fold_sd.fillna(0),color=[COLORS[c] for c in top.category],error_kw={'elinewidth':.8,'capsize':2})
    ax.set_yticks(range(len(top)),top.feature,fontsize=8)
    ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('Mean held-out CV AUPR decrease (error bars: SD across evaluable folds)')
    ax.set_title(f'{name} · {target.removeprefix("warning_").replace("_"," ")}\nTop 20 input features · training-CV interpretation')
    ax.legend(handles=[Patch(color=color,label=(label+' (source assay)' if label in ('Bacteria','Archaea') else label)) for label,color in COLORS.items()],loc='lower right',fontsize=7)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.01,'Positive: shuffling worsened predictions. Negative: shuffling improved them. No causal direction is implied.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1))
    for ext in ['png','pdf']: fig.savefig(out/f'{target}_top20.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results_warnings')
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--models',nargs='+',default=['NNET','RandomForest','KNN','SVM','XGBoost','GLMNET'])
    args=parser.parse_args()
    if args.repeats<2: parser.error('Use at least two repeats')
    root=args.results_dir; out=root/'feature_importance'; out.mkdir(exist_ok=True)
    config=json.loads((root/'run_config.json').read_text()); seed=config['seed']
    assignment=pd.read_csv(root/'split_and_cv_assignment.csv',index_col='SampleID')
    train_ids=assignment.index[assignment.split=='train']
    # Restrict rows before doing any interpretation; no test observations enter fit or score.
    raw=pd.read_csv(root/'cleaned_merged.csv',index_col='SampleID',float_precision='round_trip').loc[train_ids]
    membership=assignment.loc[train_ids]
    X=raw[CATS+NUMS+[c for c in raw if c.startswith(('BAC_','ARC_'))]]
    y=raw[TARGETS].to_numpy(int)
    universe=SampleTopTaxa().fit(X).columns_
    assert not set(universe)&set(TARGETS)
    assert not any(c.startswith('eff_') for c in universe)
    selected=json.loads((root/'cv_selected_model.json').read_text())['Model']
    rows=[]; stability=[]; all_rankings=[]; baseline_rows=[]; fit_warnings=[]
    for name in args.models:
        directory=out/name; directory.mkdir(exist_ok=True)
        params=json.loads((root/f'{name}_best_params.json').read_text())['params']
        for fold in sorted(membership.cv_validation_fold.unique()):
            valid=np.flatnonzero(membership.cv_validation_fold.values==fold)
            fit=np.flatnonzero(membership.cv_validation_fold.values!=fold)
            assert not set(membership.SiteGroup.iloc[fit])&set(membership.SiteGroup.iloc[valid])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model=WarningModel(name,params,seed).fit(X.iloc[fit],y[fit],membership.SiteGroup.iloc[fit].values)
            fit_warnings.extend({'model':name,'fold':int(fold),'category':w.category.__name__,'message':str(w.message)} for w in caught)
            selected_X=model.selector.transform(X.iloc[valid]); z=model.preprocessor.transform(selected_X)
            baseline=predict_encoded(model,z)
            saved=pd.read_csv(root/f'{name}_CV_OOF_predictions.csv',index_col='SampleID').loc[X.index[valid]]
            np.testing.assert_allclose(baseline,saved[[t+'_probability' for t in TARGETS]].values,atol=1e-9,rtol=1e-7)
            base_ap=np.array([ap_many(y[valid,j],baseline[:,j][None,:])[0] for j in range(5)])
            for j,t in enumerate(TARGETS):
                baseline_rows.append({'model':name,'fold':int(fold),'target':t,'samples':len(valid),'positives':int(y[valid,j].sum()),'baseline_aupr':base_ap[j]})
            mapping=encoded_groups(model)
            # Same permutations for each feature/model within a fold improve comparability.
            rng=np.random.default_rng(seed+int(fold)*1000)
            permutations=np.array([rng.permutation(len(valid)) for _ in range(args.repeats)])
            for feature in universe:
                deltas=np.zeros((args.repeats,5)); deltas[:,np.isnan(base_ap)]=np.nan
                cols=mapping.get(feature)
                if cols is not None and np.any(z[:,cols]!=z[0,cols]):
                    zz=np.tile(z,(args.repeats,1))
                    zz[:,cols]=z[permutations.reshape(-1)][:,cols]
                    shuffled=predict_encoded(model,zz).reshape(args.repeats,len(valid),5)
                    for j in range(5): deltas[:,j]=base_ap[j]-ap_many(y[valid,j],shuffled[:,:,j])
                for j,t in enumerate(TARGETS):
                    for repeat,value in enumerate(deltas[:,j]):
                        rows.append({'model':name,'fold':int(fold),'target':t,'feature':feature,'category':category(feature),'selected_in_fold':cols is not None,'repeat':repeat+1,'importance':value})
            print(f'{name}: fold {int(fold)} complete ({len(mapping)} inputs, {args.repeats} permutations each)',flush=True)
        data=pd.DataFrame([r for r in rows if r['model']==name])
        data.to_csv(directory/'permutation_repeats.csv.gz',index=False,compression='gzip')
        fm=data.groupby(['target','feature','category','fold'],sort=False).agg(importance=('importance','mean'),permutation_sd=('importance','std'),selected_in_fold=('selected_in_fold','first')).reset_index()
        fm['fold_rank']=fm.groupby(['target','fold']).importance.rank(ascending=False,method='average')
        fm['positive_top10']=(fm.importance>0)&(fm.fold_rank<=10)
        fm['positive_top20']=(fm.importance>0)&(fm.fold_rank<=20)
        fm.to_csv(directory/'fold_importance.csv',index=False)
        for target in TARGETS:
            f=fm[fm.target==target]
            table=f.groupby(['feature','category'],sort=False).agg(importance_mean=('importance','mean'),fold_sd=('importance','std'),evaluable_folds=('importance','count'),selected_folds=('selected_in_fold','sum'),positive_folds=('importance',lambda a:int((a>0).sum())),top10_folds=('positive_top10','sum'),top20_folds=('positive_top20','sum'),mean_fold_rank=('fold_rank','mean'),mean_permutation_sd=('permutation_sd','mean')).reset_index()
            table['top10_frequency']=table.top10_folds/table.evaluable_folds.replace(0,np.nan)
            table['positive_frequency']=table.positive_folds/table.evaluable_folds.replace(0,np.nan)
            table=table.sort_values(['importance_mean','feature'],ascending=[False,True]).reset_index(drop=True)
            table.insert(0,'rank',table.importance_mean.rank(ascending=False,method='min'))
            table.insert(0,'target',target);table.insert(0,'model',name)
            for count,suffix in [(None,'full'),(10,'top10'),(20,'top20')]:
                (table if count is None else table.head(count)).to_csv(directory/f'{target}_{suffix}.csv',index=False)
            plot_rank(table,name,target,directory)
            all_rankings.append(table)
            for a,b in itertools.combinations(sorted(f.fold.unique()),2):
                aa=f[f.fold==a].set_index('feature'); bb=f[f.fold==b].set_index('feature')
                keep=aa.importance.notna()&bb.importance.notna()
                corr=spearmanr(aa.loc[keep,'importance'],bb.loc[keep,'importance']).statistic if keep.sum()>1 and aa.loc[keep,'importance'].nunique()>1 and bb.loc[keep,'importance'].nunique()>1 else np.nan
                sa=set(aa.index[aa.positive_top10]);sb=set(bb.index[bb.positive_top10])
                valid_pair=bool(keep.any())
                stability.append({'model':name,'target':target,'fold_a':int(a),'fold_b':int(b),'spearman':corr,'positive_top10_jaccard':len(sa&sb)/len(sa|sb) if valid_pair and sa|sb else np.nan,'comparable_features':int(keep.sum())})
    rankings=pd.concat(all_rankings,ignore_index=True); rankings.to_csv(out/'all_models_full_rankings.csv',index=False)
    stability=pd.DataFrame(stability);stability.to_csv(out/'fold_pair_stability.csv',index=False)
    ss=stability.groupby(['model','target']).agg(mean_spearman=('spearman','mean'),mean_top10_jaccard=('positive_top10_jaccard','mean'),evaluable_fold_pairs=('spearman','count')).reset_index()
    ss.to_csv(out/'fold_stability_summary.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(out/'baseline_fold_scores.csv',index=False)
    pd.DataFrame(fit_warnings).to_csv(out/'fit_warnings.csv',index=False)
    save_report(out,rankings,ss,selected,args,train_ids)


def save_report(out,rankings,stability,selected,args,ids):
    category_leaders=rankings.sort_values('importance_mean',ascending=False).groupby(['model','target','category'],sort=False).head(3)
    category_leaders.to_csv(out/'category_leaders.csv',index=False)
    primary_stability=stability[stability.model==selected]
    summary=['# Per-label feature importance\n',f'Primary model: **{selected}**, chosen by the original training CV Macro AUPR. All six models are compared using their saved best hyperparameters.\n',
    'No warning-model permutation importance or SHAP outputs existed before this analysis. Earlier importance outputs in `ML/results_selected/` belong to the regression experiment.\n',
    f'Method: {args.repeats} repeated permutations per input feature in each of five held-out training CV folds. Importance is baseline average precision minus permuted average precision, independently for each binary label. Rankings average the evaluable fold means with equal fold weights.\n',
    'No independent test rows are fitted, scored, or permuted. This is post-hoc CV interpretation, not a feature-selection run. Best hyperparameters were previously selected on these folds, so the interpretation is conditional on that tuning and not nested-CV validation of a newly selected subset.\n',
    'Taxon masks/unions and preprocessing are fitted on each training fold and frozen before permutation. We permute the resulting classifier input feature, not the raw abundance followed by reranking. One-hot columns for a categorical variable move together. This answers which supplied inputs the classifier relies on. All training-union input features receive a ranking; a taxon absent from a fold model has zero effect in that fold.\n',
    'Permutations shuffle rows within the held-out fold; training remains grouped by site. They do not preserve within-site trajectories or abundance compositional constraints. Correlated metadata, phyla/genera, and taxa can dilute or redistribute importance; importance is not causal and does not indicate whether a feature raises or lowers biological risk.\n',
    'Folds without positive labels have undefined AP and are excluded for that label. Feature eligibility and evaluable-fold counts are reported separately. Negative values mean shuffling improved AP. Error bars are fold SD, not confidence intervals; repeat-level variation is also saved. Zero effects and tied ranks are retained, not interpreted as evidence of biological irrelevance. Positive-only top-10 stability excludes zero ties.\n',
    'Taxonomic caution: BAC/ARC categories identify source assays, not verified domains. The ARC workbook OTUs sheet labels Holozoa and Rhinosporideacae as Eukaryota; they must not be interpreted as archaeal biomarkers. Their small acid-accumulation importance is based on only two evaluable folds.\n',
    f'For {selected}, mean pairwise rank correlations range from {primary_stability.mean_spearman.min():.3f} to {primary_stability.mean_spearman.max():.3f}; positive top-10 Jaccard overlaps range from {primary_stability.mean_top10_jaccard.min():.3f} to {primary_stability.mean_top10_jaccard.max():.3f}. Read these values before treating the leading features as reproducible. Category-specific leaders for operation/context, substrate, BAC and ARC inputs are saved in category_leaders.csv.\n',
    '## Most influential features for the CV-selected model\n']
    for target in TARGETS:
        t=rankings[(rankings.model==selected)&(rankings.target==target)]
        s=stability[(stability.model==selected)&(stability.target==target)].iloc[0]
        summary += [f'### {target}\n',f'Evaluable folds: {int(t.evaluable_folds.max())}; mean pairwise rank correlation: {s.mean_spearman:.3f}; positive top-10 Jaccard overlap: {s.mean_top10_jaccard:.3f}.\n','| Feature | Category | Mean AUPR decrease | Positive folds | Top-10 folds |','| --- | --- | --- | --- | --- |']
        for _,r in t.head(10).iterrows(): summary.append(f'| {r.feature} | {r.category} | {r.importance_mean:.4f} | {r.positive_folds}/{r.evaluable_folds} | {r.top10_folds}/{r.evaluable_folds} |')
        summary.append('')
        # Cross-model comparison on the primary model top 20 feature set.
        names=t.head(20).feature.tolist()
        matrix=rankings[rankings.target==target].pivot(index='feature',columns='model',values='importance_mean').reindex(index=names,columns=args.models)
        fig,ax=plt.subplots(figsize=(10,9)); bound=max(float(np.nanmax(np.abs(matrix.values))),.001)
        im=ax.imshow(matrix.values,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
        ax.set(xticks=range(len(args.models)),xticklabels=args.models,yticks=range(len(names)),yticklabels=names,title=f'{target}\nCV AUPR decrease · {selected} top 20 across models')
        ax.tick_params(axis='y',labelsize=8);ax.tick_params(axis='x',rotation=40)
        for i in range(len(names)):
            for j in range(len(args.models)):
                ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if abs(matrix.iloc[i,j])>.7*bound else 'black')
        fig.colorbar(im,ax=ax,label='Mean held-out CV AUPR decrease');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'{target}_model_comparison.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    summary += ['## Reading fold consistency\n','Rank correlation near zero and low top-10 overlap indicate unstable rankings. Compare positive-fold frequencies and taxon eligibility before claiming consistency. Small validation samples and very rare labels make fine-grained rankings exploratory. Full pairwise values and summaries are in the stability CSV files.\n',
    '## Files\n','Each model directory contains per-label full/top10/top20 CSVs and top-20 PNG/PDF plots, fold-level importance, and compressed repeat-level CSV data. Root files contain all-model rankings, cross-model figures, baseline scores and fold stability. `index.html` provides a local gallery.']
    (out/'README.md').write_text('\n'.join(summary)+'\n')
    (out/'analysis_config.json').write_text(json.dumps({'seed':json.loads((out.parent/'run_config.json').read_text())['seed'],'repeats':args.repeats,'models':args.models,'primary_model':selected,'scope':'training CV interpretation only','test_used':False,'training_sample_ids':ids.tolist(),'score':'per-label average precision decrease','permutation_unit':'held-out rows; categorical encoded blocks together','taxon_selection':'fold training only; frozen for permutation','aggregation':'equal mean of evaluable folds'},indent=2)+'\n')
    options=''.join(f'<option {"selected" if m==selected else ""}>{m}</option>' for m in args.models)
    labels=''.join(f'<option>{t}</option>' for t in TARGETS)
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Warning feature importance</title><style>body{font:16px system-ui;background:#f4f6f9;color:#213449;max-width:1250px;margin:auto;padding:28px}img{max-width:100%;background:white}select{padding:10px;margin:10px;font:inherit}p{line-height:1.6}a{color:#176395}</style><h1>Which inputs contribute to each warning prediction?</h1><p>Training-CV permutation interpretation. No independent test data used. Importance measures loss of AUPR, separately for each label. NNET is the original CV-selected model. Colors distinguish metadata and BAC/ARC source assays. Some ARC entries are annotated Eukaryota, including Holozoa and Rhinosporideacae; see the report.</p><select id="model">MODELS</select><select id="target">TARGETS</select><p id="links"></p><img id="plot" alt="Top 20 importance with fold standard deviation"><h2>Compare models</h2><img id="comparison" alt="Cross model feature importance"><p>Low fold consistency means the ranking is exploratory. Correlated features and rare warnings limit interpretation. <a href="README.md">Read findings and fold stability</a> · <a href="fold_stability_summary.csv">Download consistency results</a></p><script>function update(){let m=document.getElementById('model').value,t=document.getElementById('target').value;document.getElementById('plot').src=m+'/'+t+'_top20.png';document.getElementById('comparison').src=t+'_model_comparison.png';document.getElementById('links').innerHTML=['top10','top20','full'].map(s=>'<a href="'+m+'/'+t+'_'+s+'.csv">'+s+' CSV</a>').join(' · ')+' · <a href="'+m+'/'+t+'_top20.pdf">PDF plot</a>';}document.querySelectorAll('select').forEach(e=>e.onchange=update);update();</script></html>'''.replace('MODELS',options).replace('TARGETS',labels)
    (out/'index.html').write_text(doc)
    print('Saved feature interpretation:',out,flush=True)

if __name__=='__main__': main()

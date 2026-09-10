#!/usr/bin/env python3
"""Reproducible Order-only binary-relevance analysis of five joint warnings."""
from pathlib import Path
import argparse, hashlib, json, platform, warnings
import joblib
import numpy as np
import pandas as pd
import sklearn, xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, roc_curve, roc_auc_score, auc
from order_models import TARGETS, WarningModel
from train_genomic import SPACES, partition, label_report, save_json, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['Multilabel MCC', 'Micro-AUPR', 'Micro-ROC-AUC']

def metrics(y, p, threshold=.5):
    precision, recall, _ = precision_recall_curve(y.ravel(), p.ravel())
    return dict(zip(METRICS, [float(matthews_corrcoef(y.ravel(), (p >= threshold).ravel())),
                             float(auc(recall, precision)), float(roc_auc_score(y.ravel(), p.ravel()))]))

def rank_candidates(rows):
    """Predeclared equal average rank over all three metrics; stable tie order."""
    frame = pd.DataFrame(rows)
    ranks = frame[METRICS].rank(ascending=False, method='average').mean(axis=1)
    return int(np.argmin(ranks)), ranks.tolist()

def leaders(frame):
    return {metric: frame.loc[frame[metric] == frame[metric].max(), 'Model'].tolist() for metric in METRICS}

def inspect(data_dir, out):
    metadata = pd.read_csv(data_dir/'metadata.csv', dtype={'SampleID':str,'Site':str,'No':str})
    identity = ['SampleID','Site','Season','Round','No']
    if metadata[identity].isna().any().any() or metadata.SampleID.duplicated().any() or metadata.duplicated(['Site','SampleID']).any():
        metadata.to_csv(out/'invalid_metadata.csv',index=False)
        raise ValueError('Invalid or duplicate metadata identity; see invalid_metadata.csv')
    if not metadata[TARGETS].isin([0,1]).all().all():
        raise ValueError('Targets must be complete binary labels')
    expected = metadata.Round.astype(str)+'-'+metadata.No
    assert (expected == metadata.SampleID).all()
    assert (metadata.Round.map({1:'Summer',2:'Fall',3:'Winter',4:'Spring'}) == metadata.Season).all()
    m = metadata[identity+TARGETS].set_index('SampleID').copy()
    m['SiteGroup'] = m.Site.replace({'BSIb':'BSI','DGYb':'DGY'})
    m.reset_index().groupby(['SiteGroup','Site','Season']).agg(samples=('SampleID','size'),sample_ids=('SampleID',lambda s:';'.join(s))).to_csv(out/'site_season_inventory.csv')
    frames, mapping, structures = {}, [], []
    for domain in ['BAC','ARC']:
        path = data_dir/f'Dat_{domain}.xlsx'
        f = pd.read_excel(path,sheet_name='O(%)',index_col=0)
        if f.index.name != 'Order' or f.index.has_duplicates or f.index.isna().any():
            raise ValueError('Invalid Order taxon labels')
        ids = ['-'.join(str(c).split('-')[:2]) for c in f.columns]
        if len(ids) != len(set(ids)) or any(len(s.split('-')) != 2 for s in ids):
            raise ValueError('Invalid or duplicate normalized genomic sample IDs')
        mapping.extend({'domain':domain,'workbook_sample':c,'SampleID':sid} for c,sid in zip(f.columns,ids))
        structures.append({'file':path.name,'sheet':'O(%)','taxa':len(f),'samples':len(ids)})
        f.columns = ids
        f = f.T.astype(float)
        if not np.isfinite(f.to_numpy()).all() or (f.to_numpy()<0).any():
            raise ValueError('Missing/nonfinite/negative abundance; missing profiles are not biological zeros')
        f.columns = [domain+'_O::'+str(c) for c in f.columns]
        frames[domain] = f
    all_ids = sorted(set(m.index).union(*(set(f.index) for f in frames.values())))
    audit = pd.DataFrame(index=pd.Index(all_ids,name='SampleID'))
    audit['metadata'] = audit.index.isin(m.index)
    for domain,f in frames.items(): audit[domain] = audit.index.isin(f.index)
    audit['included'] = audit.all(axis=1)
    audit['reason'] = audit.apply(lambda r: 'included' if r.included else 'missing '+','.join(k for k in ['metadata','BAC','ARC'] if not r[k]),axis=1)
    audit.to_csv(out/'sample_audit.csv')
    pd.DataFrame(mapping).to_csv(out/'sample_id_mapping.csv',index=False)
    save_json(out/'input_structure.json',{'metadata_shape':metadata.shape,'genomic':structures})
    m = m.loc[m.index.isin(audit.index[audit.included])]
    raw = pd.concat([f.loc[m.index] for f in frames.values()],axis=1)
    masked = raw.where(raw > .1,0)
    counts = (raw > .1).sum()
    X = masked.loc[:,counts > 0]
    taxa = pd.DataFrame([{'domain':c.split('_')[0],'rank':'Order','taxon':c.split('::')[1], 'feature':c,
                         'samples_above_0_1_percent':int(counts[c]),'included':bool(counts[c]>0)} for c in raw.columns])
    taxa.to_csv(out/'taxa_audit.csv',index=False)
    taxa[taxa.included].to_csv(out/'selected_taxa.csv',index=False)
    for domain in frames:
        taxa[taxa.included & (taxa.domain==domain)].to_csv(out/f'selected_{domain}_orders.csv',index=False)
    taxa.groupby('domain').agg(source_orders=('taxon','size'),selected_orders=('included','sum')).to_csv(out/'feature_counts.csv')
    X.to_csv(out/'final_feature_matrix.csv')
    pd.concat([m,X],axis=1).to_csv(out/'cleaned_merged.csv')
    excluded = [c for c in metadata.columns if c not in TARGETS]
    save_json(out/'leakage_check.json',{'passed':all(c.startswith(('BAC_O::','ARC_O::')) for c in X.columns),
        'metadata_columns_excluded_from_predictors':excluded,'target_columns_excluded_from_predictors':TARGETS,
        'predictors':list(X.columns),'direct_definition_columns_excluded':['eff_pH','eff_TVFAs','eff_ALK','eff_HPro','eff_HAc','eff_TAN','eff_CH4']})
    return m,X,m[TARGETS].to_numpy(int),audit

def write_predictions(out, name, scope, m, y, p, threshold):
    frame = m[['Site','SiteGroup','Season']].copy()
    for j,t in enumerate(TARGETS):
        frame[t+'_actual']=y[:,j]
        frame[t+'_probability']=p[:,j]
        frame[t+'_predicted']=(p[:,j]>=threshold).astype(int)
    frame.to_csv(out/f'{name}_{scope}_predictions.csv')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=20260909)
    parser.add_argument('--n-iter',type=int,default=8)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/final')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'ML/results_order')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.n_iter < 2: parser.error('At least two candidates required')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    m,X,y,audit=inspect(args.data_dir,out)
    groups=m.SiteGroup.to_numpy()
    train,test=partition(groups,y,[.8,.2],args.seed)
    Xtr,ytr,gtr=X.iloc[train],y[train],groups[train]
    valid_folds=partition(gtr,ytr,[.2]*5,args.seed+1)
    cv=[(np.setdiff1d(np.arange(len(train)),valid),valid) for valid in valid_folds]
    assignment=m[['Site','SiteGroup','Season']].copy()
    assignment['split']='test'; assignment['cv_validation_fold']=0
    assignment.loc[Xtr.index,'split']='train'
    label_rows=label_report(y,'matched')+label_report(ytr,'train')+label_report(y[test],'test')
    assert not set(gtr)&set(groups[test])
    for fold,(fit,valid) in enumerate(cv,1):
        assert not set(gtr[fit])&set(gtr[valid])
        assignment.loc[Xtr.iloc[valid].index,'cv_validation_fold']=fold
        label_rows+=label_report(ytr[fit],f'cv_training_{fold}')+label_report(ytr[valid],f'cv_validation_{fold}')
    assignment.to_csv(out/'split_and_cv_assignment.csv')
    pd.DataFrame(label_rows).to_csv(out/'label_distribution.csv',index=False)
    config={'seed':args.seed,'n_iter':args.n_iter,'targets':TARGETS,'search_spaces':SPACES,
      'selection':'Equal average descending rank of all three pooled OOF metrics across candidates; stable candidate order breaks exact rank ties. MCC uses 0.5 during hyperparameter selection. Across final models use all three thresholded OOF metrics; report leaders/trade-offs.',
      'threshold':'One shared threshold across labels, maximizing flattened training OOF MCC over 0.01..0.99, ties closest to 0.5; fixed before test prediction.',
      'filter':'O(%) only; each sample value <=0.1 percent becomes zero; global union across matched samples under the explicitly predefined rule. No further selection or renormalization. Same fixed columns in every model/fold.',
      'split':'5000 seeded candidate group allocations balancing label counts, positive-site counts and sample sizes; labels used for stratification only. QC BSIb->BSI, DGYb->DGY.',
      'preprocessing':'Fold-local StandardScaler for KNN/SVM/NNET/GLMNET; identity transform for RF/XGBoost. No missing abundance cells: no imputation. Incomplete ARC profiles excluded with audit.',
      'imbalance':{'RandomForest':'balanced class weights','SVM':'balanced weights; 3-fold inner site-grouped sigmoid calibration','GLMNET':'sklearn elastic-net logistic regression with saga and balanced weights (not R glmnet)','XGBoost':'training negative/positive scale_pos_weight','KNN':'training-only per-label random oversampling','NNET':'training-only per-label random oversampling'},
      'single_class':'Constant classifier when a training label contains only one class.',
      'metric_definitions':{'Multilabel MCC':'Binary MCC after flattening all sample-label pairs.','Micro-AUPR':'Trapezoidal area under PR curve of flattened true labels and probabilities, not average precision.','Micro-ROC-AUC':'ROC AUC of flattened true labels and probabilities.'},
      'train_sites':sorted(set(gtr)),'test_sites':sorted(set(groups[test])), 'train_samples':len(train),'test_samples':len(test),
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'matplotlib':matplotlib.__version__},
      'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.data_dir/'metadata.csv',args.data_dir/'Dat_BAC.xlsx',args.data_dir/'Dat_ARC.xlsx']}}
    save_json(out/'run_config.json',config)
    print(f'Matched {len(m)} samples, {m.Site.nunique()} site labels, {len(set(groups))} grouped sites; {X.shape[1]} Order features. Train {len(train)}, test {len(test)}.',flush=True)
    if args.audit_only:return
    tuning,cv_rows,fitted,fit_warnings=[],[],{},[]
    for name,space in SPACES.items():
        candidates=list(ParameterSampler(space,n_iter=args.n_iter,random_state=args.seed))
        scores,outputs=[],[]
        for candidate,params in enumerate(candidates):
            oof=np.zeros(ytr.shape)
            for fold,(fit,valid) in enumerate(cv,1):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model=WarningModel(name,params,args.seed).fit(Xtr.iloc[fit],ytr[fit],gtr[fit])
                    oof[valid]=model.predict_proba(Xtr.iloc[valid])
                fit_warnings.extend({'Model':name,'candidate':candidate,'fold':fold,'category':w.category.__name__,'message':str(w.message)} for w in caught)
                tuning.append({'Model':name,'candidate':candidate,'scope':f'fold_{fold}','params':json.dumps(params),**metrics(ytr[valid],oof[valid])})
            score=metrics(ytr,oof);scores.append(score);outputs.append(oof)
            tuning.append({'Model':name,'candidate':candidate,'scope':'pooled_OOF','params':json.dumps(params),**score})
            pd.DataFrame(tuning).to_csv(out/'hyperparameter_tuning.csv',index=False)
            print(f'{name} {candidate+1}/{len(candidates)}: {score}',flush=True)
        best,ranks=rank_candidates(scores)
        params,oof=candidates[best],outputs[best]
        threshold=choose_threshold(ytr,oof)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            final=WarningModel(name,params,args.seed).fit(Xtr,ytr,gtr)
        fit_warnings.extend({'Model':name,'candidate':best,'fold':'final','category':w.category.__name__,'message':str(w.message)} for w in caught)
        final.thresholds=np.full(5,threshold)
        joblib.dump(final,out/f'{name}.joblib')
        joblib.dump(final.preprocessor,out/f'{name}_preprocessing.joblib')
        save_json(out/f'{name}_best_params.json',{'params':params,'candidate':best,'candidate_mean_ranks':ranks,'thresholds':dict(zip(TARGETS,final.thresholds)),'CV_at_0_5':scores[best]})
        write_predictions(out,name,'CV_OOF',m.loc[Xtr.index],ytr,oof,threshold)
        cv_rows.append({'Model':name,**metrics(ytr,oof,threshold)})
        fitted[name]=final
        pd.DataFrame(fit_warnings,columns=['Model','candidate','fold','category','message']).to_csv(out/'fit_warnings.csv',index=False)
    cv_table=pd.DataFrame(cv_rows);cv_table.to_csv(out/'cv_performance.csv',index=False)
    best,ranks=rank_candidates(cv_rows)
    selection={'Model':cv_rows[best]['Model'],'mean_ranks':dict(zip(fitted,ranks)),'metric_leaders':leaders(cv_table),'rule':config['selection']}
    save_json(out/'cv_selected_model.json',selection)
    pd.DataFrame([{'Model':n,**dict(zip(TARGETS,f.thresholds))} for n,f in fitted.items()]).to_csv(out/'thresholds.csv',index=False)
    # All fitting, thresholds, serialized models and CV selection precede holdout prediction.
    test_rows=[];pr_rows=[];roc_rows=[]
    for name,final in fitted.items():
        p=final.predict_proba(X.iloc[test])
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        test_rows.append({'Model':name,**metrics(y[test],p,final.thresholds)})
        write_predictions(out,name,'test',m.iloc[test],y[test],p,float(final.thresholds[0]))
        precision,recall,_=precision_recall_curve(y[test].ravel(),p.ravel())
        fpr,tpr,_=roc_curve(y[test].ravel(),p.ravel())
        pr_rows.extend({'Model':name,'Recall':r,'Precision':v} for r,v in zip(recall,precision))
        roc_rows.extend({'Model':name,'FPR':r,'TPR':v} for r,v in zip(fpr,tpr))
    test_table=pd.DataFrame(test_rows);test_table.to_csv(out/'test_performance.csv',index=False)
    for metric,filename in zip(METRICS,['multilabel_mcc','micro_aupr','micro_roc_auc']):
        fig,ax=plt.subplots(figsize=(8,5));bars=ax.bar(test_table.Model,test_table[metric],color='#267d96')
        ax.bar_label(bars,fmt='%.3f',padding=3);ax.set(ylabel=metric,title=f'Order-only microbiome: held-out {metric}')
        ax.set_ylim(min(0,test_table[metric].min()-.08),max(.1,test_table[metric].max()+.12))
        fig.tight_layout();fig.savefig(out/f'{filename}.png',dpi=180);fig.savefig(out/f'{filename}.svg');plt.close(fig)
    for rows,xcol,ycol,filename,metric in [(pr_rows,'Recall','Precision','micro_precision_recall','Micro-AUPR'),(roc_rows,'FPR','TPR','micro_roc','Micro-ROC-AUC')]:
        frame=pd.DataFrame(rows);frame.to_csv(out/f'{filename}_coordinates.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,6))
        for name in fitted:
            points=frame[frame.Model==name];value=test_table.set_index('Model').loc[name,metric]
            ax.plot(points[xcol],points[ycol],label=f'{name}: {value:.3f}')
        ax.set(xlabel=xcol,ylabel=ycol,xlim=(0,1),ylim=(0,1.02),title='Held-out pooled five-label '+metric)
        ax.legend();fig.tight_layout();fig.savefig(out/f'{filename}.png',dpi=180);plt.close(fig)
    def table(frame):
        return '| Model | '+' | '.join(METRICS)+' |\n|---|---:|---:|---:|\n'+'\n'.join('| '+r['Model']+' | '+' | '.join(f'{r[k]:.4f}' for k in METRICS)+' |' for r in frame.to_dict('records'))
    report=['# Order-only joint five-label warning prediction',
      f"CV-selected compromise model: **{selection['Model']}**, using equal average ranks across all three metrics. CV leaders: {selection['metric_leaders']}. Differing leaders indicate a trade-off, not a unanimous best model.",
      '## Final held-out performance',table(test_table),f'Descriptive test metric leaders: {leaders(test_table)}. These test comparisons do not change the CV selection.',
      '## Training OOF performance',table(cv_table),
      'Hyperparameter selection and threshold optimization reuse these OOF predictions; these are optimistic tuning estimates, not nested unbiased validation estimates.',
      '## Data and integration',f'{len(m)} matched samples; {m.Site.nunique()} original site labels grouped into {len(set(groups))} sites. QC replicates BSIb→BSI and DGYb→DGY remain with their parent sites. The supplied data do not contain exactly 34 complete four-season sites.',
      'Excluded records: '+audit.loc[~audit.included,'reason'].to_string(),
      'No missing microbial cells were found. Missing entire ARC profiles are excluded explicitly, not filled with biological zeros. Target values are retained as supplied; zeros can reflect missing source chemistry, as documented in the root README.',
      f'Fixed Order features: {X.shape[1]} total; BAC {sum(c.startswith("BAC") for c in X.columns)}, ARC {sum(c.startswith("ARC") for c in X.columns)}. Full lists and sample-specific counts are in selected_BAC_orders.csv and selected_ARC_orders.csv.',
      'Union is taken over matched samples according to the prespecified >0.1% rule, the only permitted use of all-sample feature presence. Test-only taxa, if any, remain fixed zero training columns. This is a schema decision explicitly requested in the protocol; no model-driven selection follows. Rank-sheet annotations, including unidentified and off-domain annotations, are retained as supplied.',
      f'Training: {len(train)} samples / {len(set(gtr))} sites. Test: {len(test)} samples / {len(set(groups[test]))} sites.',
      'Training sites: '+', '.join(config['train_sites']), 'Test sites: '+', '.join(config['test_sites']),
      'Sample/site/season coverage and each CV assignment are recorded in site_season_inventory.csv and split_and_cv_assignment.csv. Label distributions, including folds without rare positives, are in label_distribution.csv.',
      '## Methods',config['filter'],config['preprocessing'],str(config['imbalance']),config['single_class'],config['selection'],config['threshold'],str(config['metric_definitions']),
      'No metadata enters predictors. See leakage_check.json for every excluded metadata column. Grouped SVM calibration, oversampling and scaling use training portions only. All six final fits, thresholds and CV selection are saved before test prediction.',
      '## Interpretation',
      f"The CV-selected {selection['Model']} achieved held-out MCC {test_table.set_index('Model').loc[selection['Model'], 'Multilabel MCC']:.4f}, Micro-AUPR {test_table.set_index('Model').loc[selection['Model'], 'Micro-AUPR']:.4f}, and Micro-ROC-AUC {test_table.set_index('Model').loc[selection['Model'], 'Micro-ROC-AUC']:.4f}. This shows pooled predictive signal in this split, with imperfect binary warning prediction.",
      'These results quantify pooled sample-label discrimination and binary association for five simultaneous outputs. They do not measure exact five-label-vector correctness, and common labels contribute more to pooled metrics. Sparse positives and few independent held-out sites limit precision; five-fold balancing cannot place a rare label in every fold when too few positive sites exist.',
      'A fresh seeded split is generated without reading previous results or assignments. However, the same source cohort has been analyzed previously: this is an internally isolated run, not a previously unseen external validation cohort. No causal claims or deployment reliability follow from this comparison.',
      f'Recorded fitting warnings: {len(fit_warnings)}; see fit_warnings.csv. GLMNET is an elastic-net logistic-regression implementation using sklearn, not the R glmnet package.',
      '## Reproduce',f'`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl /home/best/anaconda3/envs/tsf-ad/bin/python ML/train_order.py --seed {args.seed} --n-iter {args.n_iter}`',
      'Inputs, hashes, package versions and complete search spaces are in run_config.json. Saved models require ML on PYTHONPATH. Predict using percentage-unit Order columns named exactly as final_feature_matrix.csv; its SampleID index is not a predictor.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
    print(test_table.to_string(index=False),flush=True)

if __name__=='__main__':main()

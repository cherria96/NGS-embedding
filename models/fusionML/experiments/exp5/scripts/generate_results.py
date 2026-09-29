"""Biological interpretation of saved, site-held-out fusionML fits.

No models are trained here. All model and annotation inputs are local snapshots.
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
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/fusionmlv2-exp2-cache')
BASE = Path(__file__).resolve().parents[1]
DATA = BASE/'data'
sys.path.insert(0, str(DATA))
import argparse
from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from fusionML import pipeline as pipe
from fusionML.io_taxonomy import _parse_lineage
from fusionML.io_tree import load_tree
from fusionML.evaluate import per_sample_log_loss
from run_missing_asv import config, TARGETS, MODELS, OUT as NEW
from functional_evidence import EVIDENCE, CONTEXT

FIT = ['resolution','target','model','repeat','outer_fold']
DOM = FIT+['domain']
LABELS = dict(zip(MODELS,['ARC fusion','BAC fusion','Shared hybrid','Domain separate','Domain interaction']))
SHORT = {t:t.replace('warning_','').replace('_',' ').capitalize() for t in TARGETS}
TOL = 1e-6
EVIDENCE_MAP = {r[0]:r for r in EVIDENCE}


def read(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown(df):
    df = df.copy()
    for col in df:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x:f'{x:.3f}' if np.isfinite(x) else 'NA')
    def row(values):
        return '| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in values) + ' |'
    return '\n'.join([row(df.columns), row(['---'] * len(df.columns))] + [row(values) for values in df.fillna('NA').itertuples(index=False, name=None)])


def standard(df, target=None, resolution='asv', source='exp1_asv'):
    if df.empty:
        return df
    df=df.rename(columns={'family':'model','beta_standardized':'beta'})
    if target is not None:
        df['target']=target
        df['repeat']=0
    df['resolution']=resolution; df['source']=source
    if 'selection' in df:
        df=df[df.selection.eq('one_se')].copy()
    return df[df.model.isin(MODELS)].copy()


def collect():
    allc=[]; allp=[]; alli=[]; allb=[]; alle=[]; inventories=[]; splitrows=[]
    for target,suffix in zip(TARGETS,['interaction','acid_base_balance_interaction','biogas_quality_interaction']):
        folder=DATA/f'fusionML/real_results_classification_phylo_{suffix}/smoke'
        c=standard(read(folder/'coefficients_and_groups.csv'),target,'genus','historical_genus');allc.append(c)
        p=read(folder/'oof_predictions.csv').rename(columns={'target':'y','prediction':'p'})
        p=standard(p,target,'genus','historical_genus');p=p[p.scorable];allp.append(p)
        alli.append(standard(read(folder/'interaction_coefficients_outer_folds.csv'),target,'genus','historical_genus'))
        allb.append(standard(read(folder/'balance_definitions_outer_folds.csv'),target,'genus','historical_genus'))
        alle.append(standard(read(folder/'graph_edges.csv'),target,'genus','historical_genus'))
        inv=c[FIT+['source']].drop_duplicates();inv['status']='ok';inventories.append(inv)
    folder=DATA/'fusionML/real_results_asv_exp1_20260922'
    for filename, dest in [('coefficients_and_groups',allc),('predictions',allp),('interactions',alli),('balances',allb),('graph_edges',alle),('rows',inventories)]:
        dest.append(standard(read(folder/f'{filename}.csv')))
    for filename, dest in [('coefficients_and_groups',allc),('predictions',allp),('interactions',alli),('balances',allb),('graph_edges',alle),('rows',inventories)]:
        dest.append(standard(read(NEW/f'{filename}.csv'),source='new_asv'))
    # Both saved and new ASV fits use these same whole-site splits.
    s=read(NEW/'splits.csv'); splitrows.append(s)
    return [pd.concat([d for d in frames if not d.empty],ignore_index=True) for frames in [allc,allp,alli,allb,alle,inventories]],pd.concat(splitrows,ignore_index=True)


def taxonomy_catalog(coeff):
    lookup={}; members=[]
    for domain in ['ARC','BAC']:
        raw=read_tax= pd.read_csv(DATA/f'data/qiimeresult/{domain}/metadata.tsv',sep='\t')
        raw=raw[~raw['Feature ID'].str.startswith('#')].set_index('Feature ID')
        mapping=pd.read_csv(DATA/f'output/genus_tree/taxlevel_autorun/genus/{domain.lower()}_mapping.csv')
        blocks=mapping.groupby('reduced_feature').feature.apply(list).to_dict()
        for resolution in ['asv','genus']:
            features=sorted(coeff[(coeff.domain==domain)&(coeff.resolution==resolution)].feature_id.unique())
            for feature in features:
                rawid=feature.split('::',1)[1]
                ids=[rawid] if resolution=='asv' else blocks[rawid]
                lineages=[];genera=[];domain_labels=[];confidences=[]
                for asv in ids:
                    row=raw.loc[asv]; lineage=str(row.Taxon); ranks=_parse_lineage(lineage)
                    genus=ranks.get('g','')
                    valid=bool(genus and genus.lower() not in ['uncultured','unclassified','unknown','metagenome'] and genus not in [ranks.get(k) for k in ['p','c','o','f']])
                    label=genus if valid else next((f'{k}__{ranks[k]} (unresolved genus)' for k in ['f','o','c','p','d'] if k in ranks),'Unassigned')
                    members.append(dict(resolution=resolution,domain=domain,feature_id=feature,asv_id=domain+'::'+asv,lineage=lineage,genus=genus or 'unassigned',genus_resolved=valid,taxon_label=label,taxonomic_domain=ranks.get('d','unassigned'),confidence=float(row.Confidence)))
                    lineages.append(ranks);genera.append(label);domain_labels.append(ranks.get('d','unassigned'));confidences.append(float(row.Confidence))
                common={k:lineages[0][k] for k in ['d','p','c','o','f','g','s'] if k in lineages[0] and all(x.get(k)==lineages[0][k] for x in lineages)}
                expected='Archaea' if domain=='ARC' else 'Bacteria'
                common_genus=common.get('g','')
                resolved_genus=bool(common_genus and common_genus.lower() not in ['uncultured','unclassified','unknown','metagenome'] and common_genus not in [common.get(k) for k in ['p','c','o','f']])
                ev=EVIDENCE_MAP.get(common_genus) if set(domain_labels)=={expected} else None
                lookup[(resolution,feature)] = dict(resolution=resolution,domain=domain,feature_id=feature,n_asvs=len(ids),
                    taxon_label='; '.join(f'{k} ({v})' if v>1 else k for k,v in Counter(genera).most_common()),
                    genus=common_genus or 'mixed/unresolved',common_lineage='; '.join(f'{k}__{v}' for k,v in common.items()),
                    genus_resolved=resolved_genus,
                    confidence_min=min(confidences),domain_mismatch_asvs=sum(d not in [expected,'unassigned'] for d in domain_labels),
                    unassigned_domain_asvs=domain_labels.count('unassigned'),role=ev[1] if ev else 'unassigned',
                    role_evidence=ev[2] if ev else '',role_source=ev[3] if ev else '')
    return lookup,pd.DataFrame(members)


def analyze(coeff,preds,interactions,balances,inventory,splits,lookup):
    groups=[]; contrasts=[]; validations=[]; cohorts=[]
    cached={}; trees={}; distcache={}
    for resolution in ['asv','genus']:
        for domain in ['ARC','BAC']:
            path=DATA/(f'data/qiimeresult/{domain}/rooted-tree.nwk' if resolution=='asv' else f'output/genus_tree/taxlevel_autorun/genus/{domain.lower()}_tree.nwk')
            trees[(resolution,domain)]=load_tree(str(path),domain)
        for target in TARGETS:
            cfg=config(target,resolution);inp=pipe.load_all_inputs(cfg)
            if resolution=='asv':
                inp.metadata['site_id']=inp.metadata.site_id.replace({'BSIb':'BSI','DGYb':'DGY'})
            cached[(resolution,target)]=(cfg,inp)
            meta=inp.metadata.loc[inp.cohort_ids]
            cohorts.append(dict(resolution=resolution,target=target,n_samples=len(meta),n_sites=meta.site_id.nunique(),n_positive=int(meta.target.sum()),n_positive_sites=meta.loc[meta.target.eq(1),'site_id'].nunique()))
    for key,fit in coeff.groupby(FIT,sort=True):
        tags=dict(zip(FIT,key));resolution,target,model,repeat,fold=key
        cfg,inp=cached[(resolution,target)]
        p=preds
        for col,val in tags.items():p=p[p[col].eq(val)]
        assert not p.sample_id.duplicated().any()
        test=p.sample_id.tolist();testsites=set(p.site_id)
        assert np.array_equal(inp.metadata.loc[test,'target'].to_numpy(),p.y.to_numpy())
        assert inp.metadata.loc[test,'site_id'].tolist()==p.site_id.tolist()
        train=[s for s in inp.cohort_ids if inp.metadata.loc[s,'site_id'] not in testsites]
        assert set(inp.metadata.loc[train,'site_id']).isdisjoint(testsites)
        if resolution=='asv':
            sp=splits[(splits.target==target)&(splits.repeat==repeat)&(splits.outer_fold==fold)]
            assert set(train)==set(sp[sp.role=='train'].sample_id)
            assert set(test)==set(sp[sp.role=='test'].sample_id)
        counts_a,counts_b=pipe.tables_for_resolution(inp,resolution)
        for domain,dc in fit.groupby('domain'):
            counts=counts_a if domain=='ARC' else counts_b
            features=dc.feature_id.tolist()
            q=counts.loc[train+test,features]+cfg['preprocessing']['pseudocount']
            logs=np.log(q.div(q.sum(axis=1),axis=0))
            b=dc.set_index('feature_id').beta
            reconstructed=np.zeros(len(logs))
            sig=(resolution,domain,tuple(sorted(features)))
            if sig not in distcache:
                tree=trees[(resolution,domain)];assert not tree.coverage(sorted(features))[1]
                distcache[sig]=pd.DataFrame(tree.patristic_distance_matrix(sorted(features)),index=sorted(features),columns=sorted(features))
            distance=distcache[sig]
            background=distance.to_numpy()[np.triu_indices(len(features),1)].mean()
            weights=inp.metadata.loc[train,'site_id'].map(inp.metadata.loc[train,'site_id'].value_counts()).rdiv(1).to_numpy()
            weights=weights/weights.sum()
            for gid,g in dc.groupby('group_id'):
                ids=g.feature_id.tolist();size=len(ids);mean=float(g.group_mean_beta.iloc[0]);theta=size*mean
                assert np.max(abs(g.beta-g.group_mean_beta-g.residual))<1e-10
                assert g.beta.max()-g.beta.min()<=TOL+1e-10
                score=theta*logs[ids].mean(axis=1)
                reconstructed+=score.to_numpy()+logs[ids].to_numpy()@g.set_index('feature_id').loc[ids,'residual'].to_numpy()
                vals=score.loc[train].to_numpy();mu=np.sum(weights*vals);sd=np.sqrt(np.sum(weights*(vals-mu)**2))
                infos=[lookup[(resolution,f)] for f in ids]
                genus_counts=Counter(x['genus'] for x in infos if x['genus_resolved'])
                roles=Counter(x['role'] for x in infos)
                ds=distance.loc[ids,ids].to_numpy()
                pdistance=ds[np.triu_indices(size,1)].mean() if size>1 else np.nan
                signature=hashlib.sha256(';'.join(sorted(ids)).encode()).hexdigest()[:12]
                groups.append(dict(tags,domain=domain,group_id=int(gid),signature=signature,active=abs(mean)>TOL,n_features=size,
                    n_asvs=sum(x['n_asvs'] for x in infos),mean_beta=mean,theta=theta,association='higher flag score' if mean>TOL else 'lower flag score' if mean< -TOL else 'inactive',
                    train_score_sd=sd,members=';'.join(ids),taxa='; '.join(x['taxon_label'] for x in infos),
                    dominant_genus=genus_counts.most_common(1)[0][0] if genus_counts else 'unresolved',
                    genus_purity=genus_counts.most_common(1)[0][1]/size if genus_counts else np.nan,
                    resolved_genus_fraction=sum(genus_counts.values())/size,
                    roles='; '.join(f'{r}: {n}' for r,n in sorted(roles.items())),role_coverage=1-roles.get('unassigned',0)/size,
                    annotated_role_count=len([r for r in roles if r!='unassigned']),
                    domain_mismatch_asvs=sum(x['domain_mismatch_asvs'] for x in infos),unassigned_domain_asvs=sum(x['unassigned_domain_asvs'] for x in infos),
                    mean_patristic_distance=pdistance,retained_background_distance=background,
                    distance_ratio=pdistance/background if background>0 else np.nan))
            error=float(np.max(abs(reconstructed-logs[features].to_numpy()@b.loc[features].to_numpy())))
            assert error<1e-8
            assert abs(b.sum())<1e-6
            validations.append(dict(tags,domain=domain,n_train=len(train),n_test=len(test),equality_residual=abs(b.sum()),group_plus_residual_error=error))
            # Exact positive-vs-negative weighted geometric-mean balance per domain.
            positive=b[b>0];negative=-b[b<0];mass=float(positive.sum())
            contrasts.append(dict(tags,domain=domain,positive_mass=mass,negative_mass=float(negative.sum()),
                positive_members=json.dumps(positive.to_dict(),sort_keys=True),negative_members=json.dumps(negative.to_dict(),sort_keys=True),
                positive_taxa='; '.join(lookup[(resolution,f)]['taxon_label'] for f in positive.index),
                negative_taxa='; '.join(lookup[(resolution,f)]['taxon_label'] for f in negative.index)))
    return pd.DataFrame(groups),pd.DataFrame(contrasts),pd.DataFrame(validations),pd.DataFrame(cohorts)


def interpretation_summaries(coeff,groups,inventory,lookup):
    annotated=coeff.merge(pd.DataFrame(lookup.values()).drop(columns='domain'),on=['resolution','feature_id'],validate='many_to_one')
    feature_rows=[]
    for key,d in annotated.groupby(['resolution','target','model','domain','feature_id']):
        resolution,target,model,domain,feature=key
        inv=inventory[(inventory.resolution==resolution)&(inventory.target==target)&(inventory.model==model)]
        active=d[d.beta.abs()>TOL]
        feature_rows.append(dict(zip(['resolution','target','model','domain','feature_id'],key),taxon_label=d.taxon_label.iloc[0],genus=d.genus.iloc[0],role=d.role.iloc[0],
            planned_fits=len(inv),successful_fits=int(inv.status.eq('ok').sum()),retained_fits=len(d),active_fits=len(active),
            positive_fits=int((d.beta>TOL).sum()),negative_fits=int((d.beta< -TOL).sum()),
            sign_agreement=max((active.beta>0).sum(),(active.beta<0).sum())/len(active) if len(active) else np.nan,
            mean_beta_when_retained=d.beta.mean(),mean_abs_beta_when_retained=d.beta.abs().mean()))
    summaries=[]
    for key,g in groups.groupby(['resolution','target','model','domain']):
        active=g[g.active];multi=active[active.n_features>1]
        fitmean=active.groupby(['repeat','outer_fold']).size().reindex(g.groupby(['repeat','outer_fold']).size().index,fill_value=0).mean()
        summaries.append(dict(zip(['resolution','target','model','domain'],key),successful_fits=g[['repeat','outer_fold']].drop_duplicates().shape[0],
            retained_features_mean=g.groupby(['repeat','outer_fold']).n_features.sum().mean(),active_groups_mean=fitmean,
            multifeature_active_groups=len(multi),median_multifeature_size=multi.n_features.median(),
            multifeature_genus_purity_mean=multi.genus_purity.mean(),multifeature_distance_ratio_mean=multi.distance_ratio.mean(),
            active_feature_role_coverage=np.average(active.role_coverage,weights=active.n_features) if len(active) else np.nan))
    return annotated,pd.DataFrame(feature_rows),pd.DataFrame(summaries)


def interaction_tables(interactions,balances,inventory,lookup):
    rows=[];summaries=[]
    for key,inv in inventory[inventory.model.eq('domain_adaptive_interaction')].groupby(FIT):
        tags=dict(zip(FIT,key)); d=interactions;b=balances
        for col,val in tags.items(): d=d[d[col].eq(val)];b=b[b[col].eq(val)]
        active=d[d.gamma_star.abs()>TOL]
        summaries.append(dict(tags,status=inv.status.iloc[0],fitted_products=len(d),active_products=len(active)))
        for _,row in active.iterrows():
            out=dict(tags,product_index=row.product_index,gamma=row.gamma_star)
            for domain,col in [('ARC','balance_a_id'),('BAC','balance_b_id')]:
                definition=b[b.canonical_id.eq(row[col])].iloc[0]
                out[domain+'_balance_id']=row[col]
                for side in ['positive','negative']:
                    ids=definition[side+'_members'].split(';')
                    out[domain+'_'+side+'_taxa']='; '.join(lookup[(tags['resolution'],f)]['taxon_label'] for f in ids)
                    out[domain+'_'+side+'_members']=';'.join(ids)
                out[domain+'_training_mean']=definition['mean'];out[domain+'_training_sd']=definition.sd
            rows.append(out)
    return pd.DataFrame(rows,columns=list(rows[0]) if rows else FIT+['product_index','gamma','ARC_positive_taxa','ARC_negative_taxa','BAC_positive_taxa','BAC_negative_taxa']),pd.DataFrame(summaries)


def performance_context(preds,inventory,cohorts):
    result=[]
    for key,inv in inventory.groupby(['resolution','target','model']):
        res,target,model=key
        d=preds[(preds.resolution==res)&(preds.target==target)&(preds.model==model)]
        expected=int(cohorts[(cohorts.resolution==res)&(cohorts.target==target)].n_samples.iloc[0])
        for repeat in sorted(inv.repeat.unique()):
            p=d[d.repeat==repeat]
            assert not p.sample_id.duplicated().any()
            full=len(p)==expected
            result.append(dict(resolution=res,target=target,model=model,repeat=repeat,n_predictions=len(p),coverage=len(p)/expected,complete_oof=full,
                auc=roc_auc_score(p.y,p.p) if full and p.y.nunique()==2 else np.nan,
                average_precision=average_precision_score(p.y,p.p) if full and p.y.nunique()==2 else np.nan,
                site_balanced_logloss=pd.Series(per_sample_log_loss(p.y.to_numpy(),p.p.to_numpy()),index=p.site_id).groupby(level=0).mean().mean() if full else np.nan))
    return pd.DataFrame(result)


def figures(out,annotated,groups,summary,interaction_summary):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for resolution in ['asv','genus']:
        fig,axes=plt.subplots(3,2,figsize=(14,15),layout='constrained')
        for ri,target in enumerate(TARGETS):
            for ci,domain in enumerate(['ARC','BAC']):
                d=annotated[(annotated.resolution==resolution)&(annotated.target==target)&(annotated.domain==domain)]
                d=d.copy()
                d['display_taxon']=[r.genus if r.genus_resolved else r.taxon_label + (f' [{r.feature_id[-6:]}]' if r.taxon_label=='Unassigned' else '') for r in d.itertuples()]
                # Within-fit SUM of feature coefficients by lineage label preserves log-contrast mass;
                # it is a descriptive projection, not refitting or merging same-genus ASVs.
                perfit=d.groupby(['model','repeat','outer_fold','display_taxon']).beta.sum()
                available=d[['model','repeat','outer_fold']].drop_duplicates()
                columns=sorted(d.display_taxon.unique()); grid=available.assign(_k=1).merge(pd.DataFrame({'display_taxon':columns,'_k':1}),on='_k').drop(columns='_k')
                grid=grid.merge(perfit.rename('beta').reset_index(),how='left').fillna({'beta':0.})
                tab=grid.groupby(['display_taxon','model']).beta.mean().unstack().reindex(columns=MODELS)
                top=tab.abs().max(axis=1).nlargest(14).index; tab=tab.loc[top]
                limit=max(float(tab.abs().max().max()),.01)
                ax=axes[ri,ci];im=ax.imshow(tab,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
                ax.set_yticks(range(len(tab)),[x.replace('Candidatus_','Ca. ')[:40] for x in tab.index]);ax.set_xticks(range(len(MODELS)),[LABELS[m] for m in MODELS],rotation=35,ha='right')
                ax.set_title(f'{SHORT[target]} — {domain}')
                fig.colorbar(im,ax=ax,shrink=.7,label='Mean summed log-contrast coefficient')
        fig.suptitle(f'{resolution.upper()}: descriptive taxonomic projections across available fits\nRed = higher flag score in the fitted relative-composition contrast; blue = lower')
        savefig(fig,out,f'01_taxon_coefficients_{resolution}')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ri,resolution in enumerate(['asv','genus']):
        for ci,target in enumerate(TARGETS):
            ax=axes[ri,ci];s=summary[(summary.resolution==resolution)&(summary.target==target)]
            for j,domain in enumerate(['ARC','BAC']):
                vals=s[s.domain==domain].set_index('model').reindex(MODELS).active_groups_mean
                ax.bar(np.arange(5)+(j-.5)*.36,vals,.36,label=domain)
            ax.set_xticks(range(5),[LABELS[m] for m in MODELS],rotation=35,ha='right');ax.set_ylabel('Mean active groups');ax.set_title(f'{resolution.upper()} — {SHORT[target]}');ax.legend()
    savefig(fig,out,'02_active_groups_by_domain')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,resolution in zip(axes,['asv','genus']):
        g=groups[(groups.resolution==resolution)&groups.active&(groups.n_features>1)]
        for target in TARGETS:
            d=g[g.target==target];ax.scatter(d.n_features,d.genus_purity,s=25,alpha=.4,label=SHORT[target])
        ax.set(xlabel='Features per active fused group',ylabel='Dominant-genus fraction (features)',title=resolution.upper(),ylim=(-.03,1.03));ax.legend(fontsize=8)
    savefig(fig,out,'03_group_taxonomic_coherence')


def savefig(fig,out,name):
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/f'{name}.{ext}',dpi=160,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'fusionMLv2 exp2'})
    plt.close(fig)


def reports(out,groups,features,summary,cohorts,inventory,metrics,interaction_summary,checks):
    links=[]
    for resolution in ['asv','genus']:
        for target in TARGETS:
            slug=f'{resolution}_{target}';links.append(f'[{resolution.upper()} — {SHORT[target]}](details/{slug}.md)')
            body=[f'# {SHORT[target]}: {resolution.upper()} biological interpretation',
                'Groups are specific to a fitted training partition. The examples below are the three largest training-score standard deviations per model/domain, restricted to active groups. This describes fitted score variation, not independent or causal importance. The table may include more than one fold-specific version of a taxon group. See the complete member tables for all groups.',
                'A positive coefficient contributes toward flag=1 through a within-domain log-composition contrast. It does not mean that increasing absolute abundance causes failure. Group IDs are local; identical IDs across fits do not identify the same group.']
            for model in MODELS:
                body += [f'## {LABELS[model]} (`{model}`)']
                inv=inventory[(inventory.resolution==resolution)&(inventory.target==target)&(inventory.model==model)]
                body += [f"Available fits: {inv.status.eq('ok').sum()}/{len(inv)}. ARC-only/BAC-only refers to the domain receiving fusion; both domains remain predictors."]
                for domain in ['ARC','BAC']:
                    g=groups[(groups.resolution==resolution)&(groups.target==target)&(groups.model==model)&(groups.domain==domain)&groups.active].nlargest(3,'train_score_sd').copy()
                    g['fit']=g.repeat.astype(str)+':'+g.outer_fold.astype(str)+':G'+g.group_id.astype(str)
                    g['taxa']=g.taxa.map(lambda s:s if len(s)<260 else s[:257]+'…')
                    body += [f'### {domain}',markdown(g[['fit','n_features','theta','train_score_sd','taxa','roles','domain_mismatch_asvs']])]
                f=features[(features.resolution==resolution)&(features.target==target)&(features.model==model)&(features.active_fits>0)].nlargest(8,'mean_abs_beta_when_retained')
                body += ['Largest mean absolute coefficients among retained features (interpret retention and sign consistency together):',markdown(f[['domain','feature_id','taxon_label','active_fits','successful_fits','positive_fits','negative_fits','role']])]
            body += ['## Interaction interpretation',markdown(interaction_summary[(interaction_summary.resolution==resolution)&(interaction_summary.target==target)][['repeat','outer_fold','status','fitted_products','active_products']]),
                'No active product means no fitted interaction to interpret. Even an active centered-balance product would be a statistical interaction, not proof of syntrophy or direct metabolic exchange.']
            (out/'details'/f'{slug}.md').write_text('\n\n'.join(body)+'\n')
    usable=inventory.status.eq('ok').sum();unavailable=len(inventory)-usable
    summary_text=[
        '# Experiment 5 — Biological interpretation (stored as exp2)',
        f'Analyzed **{usable} fitted outer models** from **{len(inventory)} planned fits**, across five model families, three flags, and two resolutions. **{unavailable} ASV biogas fits are unavailable** because an inner training partition contains only one class. No replacement models or fabricated coefficients are supplied.',
        '## Main findings',
        '1. Fusion yields explicit groups and coefficient directions that can be traced back to ASVs and taxonomy. This supports interpretation of a fitted representation, but a shared coefficient is not evidence of a shared biological function.',
        f"2. The selected interaction models contain **{int(interaction_summary.active_products.sum())} active ARC–BAC products** across available fits. Inspect the interaction table before making any cross-domain biological claim.",
        f"3. Retained features include **{checks['unique_domain_mismatch_asvs']} ASVs assigned to a taxonomic domain different from their assay label**, and **{checks['unique_unassigned_domain_asvs']} without a domain assignment**. These remain in the original models and are explicitly flagged in the interpretation; the models were not silently changed.",
        '4. Curated functional hypotheses are limited to named genera supported by primary literature. Broad Bathyarchaeia, unidentified bacteria and W5 are not automatically assigned a function. Mixed-role groups should be described as predictive aggregates, not functional guilds.',
        '## Concrete observations from the ASV shared hybrid',
        'These examples describe the shared-hybrid fits only; the six detailed reports provide the corresponding results for all five models. They were examined after fitting and are exploratory observations, not prespecified biomarker tests.',
        '**Ammonia:** the Methanomassiliicoccus ASV `ARC::3b768ad6b26d688837b3511e5d2316d2` has a negative coefficient in five of six fits and a positive coefficient in one. Methanosaeta ASVs change sign across fits and differ from one another. The data therefore do not support a single stable genus-wide direction for Methanosaeta, even though its metabolic role is plausible background for an ammonia hypothesis.',
        '**Acid–base balance:** the Proteiniphilum ASV `BAC::14a311b4802d5606cbf72b2dc465ae56` has a positive coefficient in all six fits. This is an interpretable candidate association with the two-sided pH warning, not evidence that Proteiniphilum causes acidification. Low and high pH are combined in this target.',
        '**Biogas quality:** the Candidatus Methanofastidiosum ASV `ARC::abb98aa85e9528ef54e3078ff40ac726` has a positive coefficient in all four available fits. Two planned fits are unavailable; the cohort contains 13 positive samples across only six sites. This is a tentative association with low methane percentage, not evidence that this methanogen reduces methane production.',
        'Shared-hybrid groups and annotation coverage (averaged across available fits; role coverage is the fraction of active feature occurrences covered by this deliberately limited literature curation, not the fraction with a known biological function):',
        markdown(summary[summary.model.eq('shared_hybrid')][['resolution','target','domain','active_groups_mean','multifeature_genus_purity_mean','active_feature_role_coverage']]),
        '## Cohort and provenance',markdown(cohorts),
        'The three flags are ammonia toxicity, acid–base balance and biogas quality. Labels are computed from the copied chemistry workbook by the copied fusionML implementation: ammonia uses the project NH4-N/TAN convention (>2.5), acid–base uses pH <6.5 or >8.0, and biogas quality uses CH4 <55%. These are operational warning labels, not direct measurements of microbial activity or experimentally diagnosed inhibition. The two-sided pH flag combines different physiological conditions.',
        'Genus results reuse the existing three-fold, 37-site smoke runs, including all five requested families. ASV results reuse the three hybrid families from Experiment 1 for ammonia and acid–base (36 fits), and add the missing domain-restricted and biogas fits with the same two-repeat, three-outer/two-inner-fold, 35-QC-site protocol. Reused and new outputs are identified in `model_fit_inventory.csv`. The historical genus and ASV results have different protocols; this experiment does not estimate a causal resolution effect or compare them as matched performance benchmarks.',
        'ASV candidate generation and one-standard-error selection match Experiment 1. The interaction family includes an off candidate. The missing ASV runs use the copied fusionML estimator, with model-specific fit caches so domain-restricted graphs are respected. Historical input taxonomy hashes are verified; historical rank mappings were not hashed by the original run manifest, so their current copied versions provide contextual mapping rather than certified historical provenance.',
        '## Model definitions',
        '| Model | Interpretation |\n|---|---|\n| ARC fusion | Both domains predict; only ARC receives fusion edges. |\n| BAC fusion | Both domains predict; only BAC receives fusion edges. |\n| Shared hybrid | Both domains fused with a common phylogeny/proportionality mixture. |\n| Domain separate | Separate ARC and BAC mixture weights in one joint additive model. |\n| Domain interaction | Domain-adaptive main effects plus optional products of centered, standardized ARC/BAC balances. |',
        '## Group interpretation and algebra',
        'For a group G with common coefficient b, its contribution is |G| × b × log(GM(q_G)). The exported `theta` is |G| × b, and small per-taxon residuals are retained in the reconstruction check. This is a geometric-mean log-composition representation; it is not the summed relative abundance of the group. Separate zero-sum constraints make each domain score a relative contrast. `domain_contrasts.csv` gives positive/negative coefficient masses and members for an equivalent weighted geometric-mean ratio, up to the recorded numerical equality residual.',
        'The figures project coefficients onto taxonomy for readable labels. Within each fit, coefficients for features sharing a resolved genus or explicit broader label are summed, then averaged across available fits; absent retained features contribute zero to that projection. Unassigned features retain distinct identifiers, and unrelated uncultured lineages are not merged into one genus. This display can hide opposing ASV coefficients within a genus. Exact feature-level signs and denominators are preserved in `feature_associations.csv`. Sign agreement describes overlapping CV fits; it is not a confidence interval or independent replication.',
        'Taxonomic purity is the largest resolved-genus feature count divided by all features in a group; unknown labels never count as a shared genus, and all-unresolved groups have NA purity. The mean therefore describes groups with at least one resolved genus. Counts refer to model features, not read abundance. ASV membership counts within genus blocks are also unweighted. Patristic-distance ratios compare each multi-feature group with all retained features in the same domain/fit. They are descriptive, not enrichment p-values; phylogenetic compactness is partly imposed by the model and is not independent biological validation.',
        '## Results by flag and resolution', '\n\n'.join(links),
        '![ASV taxon coefficients](figures/01_taxon_coefficients_asv.png)',
        '![Genus taxon coefficients](figures/01_taxon_coefficients_genus.png)',
        '![Active groups](figures/02_active_groups_by_domain.png)',
        '![Group taxonomic coherence](figures/03_group_taxonomic_coherence.png)',
        '## Biological hypotheses supported by external evidence',
        'Methanosaeta and Methanoculleus provide a plausible acetoclastic-versus-hydrogenotrophic axis to examine for ammonia, but the direction must be read from each fitted contrast. In an ammonium perturbation study, active methanogens shifted between these pathways; that study does not establish the response of every ASV here. [Primary ammonium study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/).',
        'Syntrophomonas-associated groups are candidates for studying fatty-acid turnover; stable-isotope-informed digester metagenomics supports this role in particular populations. This does not establish that a fitted group exchanges hydrogen with its apparent ARC counterpart. [Primary isotope study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687939/).',
        'Proteiniphilum-associated groups warrant investigation of protein degradation and fermentation. Cultured representatives were proteolytic anaerobes from a wastewater reactor; this is a genus-level hypothesis, not evidence of nitrogen flux in these samples. [Primary isolation study](https://pubmed.ncbi.nlm.nih.gov/16280479/).',
        'Bathyarchaeia labels do not identify a uniform methane-producing guild: particular genomes support methane metabolism, while others suggest different substrate use and acetogenic potential. These groups remain functionally unassigned. [Methane-metabolism genomes](https://pubmed.ncbi.nlm.nih.gov/26494757/), [distinct substrate preferences](https://pubmed.ncbi.nlm.nih.gov/26626228/).',
        'All curated genus-to-role entries, evidence levels and primary links are in `functional_evidence.csv`. No gene enrichment, predicted metagenome, functional redundancy test or causal validation has been performed.',
        '## Predictive context and limitations',
        'Prediction metrics are context for how much confidence to place in associations. `prediction_context.csv` reports metrics only for complete out-of-fold repeats; ASV biogas has partial coverage, so its pooled performance is left NA. Low event counts, site/feedstock confounding, relative abundance, assay-domain mismatches and preprocessing thresholds constrain biological conclusions. No significance claims or formal group-stability experiment are made here.',
        'The acid–base flag should be examined separately for low versus high pH in a follow-up before attributing a single biological mechanism. For biogas, more positive independent sites are needed before interpreting an apparent biomarker as reproducible.',
        'Next biological validation should link specific members to independently measured genes/transcripts or targeted activity assays, measure candidate taxa absolutely where feasible, and test the proposed pathway with substrate or isotope experiments. Taxon names alone cannot establish activity.',
        '## Validation and outputs',
        f"All {checks['copied_files_verified']} source hashes were verified. Every fitted feature maps to copied taxonomy and the supplied resolution-specific tree. Held-out sites are disjoint from training sites. Group-plus-residual reconstruction has maximum error {checks['max_group_reconstruction_error']:.3g}; domain coefficient sums have maximum residual {checks['max_equality_residual']:.3g}. See `validation.json` and `artifact_manifest.json`.",
        'Tables: [group catalog](tables/group_catalog.csv), [exact ASV members and taxonomy](tables/feature_asv_taxonomy.csv), [feature associations](tables/feature_associations.csv), [domain contrasts](tables/domain_contrasts.csv), [interaction status](tables/interaction_summary.csv), [active interaction explanations](tables/active_interactions_annotated.csv), [fit inventory](tables/model_fit_inventory.csv).',
    ]
    (out/'REPORT.md').write_text('\n\n'.join(summary_text)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,default=BASE/'results');args=parser.parse_args();out=args.results_dir
    for sub in ['tables','figures','details']: (out/sub).mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'source_manifest.json').read_text())
    for f in manifest['files']:assert sha(DATA/f['path'])==f['sha256'],f['path']
    newmanifest=json.loads((NEW/'run_manifest.json').read_text());assert newmanifest['completed_tasks']==18
    for name,digest in newmanifest['output_hashes'].items():assert sha(NEW/name)==digest
    prior=DATA/'fusionML/real_results_asv_exp1_20260922'
    prior_manifest=json.loads((prior/'run_manifest.json').read_text())
    for name,digest in prior_manifest['output_hashes'].items():assert sha(prior/name)==digest
    for suffix in ['interaction','acid_base_balance_interaction','biogas_quality_interaction']:
        old=json.loads((DATA/f'fusionML/real_results_classification_phylo_{suffix}/smoke/run_manifest.json').read_text())
        for domain in ['ARC','BAC']:
            assert sha(DATA/f'data/qiimeresult/{domain}/metadata.tsv')==old['input_hashes'][domain.lower()+'_taxonomy']
    (coeff,preds,interactions,balances,edges,inventory),splits=collect()
    assert len(inventory)==135 and not inventory.duplicated(FIT).any()
    assert not coeff.duplicated(FIT+['domain','feature_id']).any()
    for key,sp in splits.groupby(['target','repeat','outer_fold']):
        for trainrole,testrole in [('train','test'),('inner0_train','inner0_validation'),('inner1_train','inner1_validation')]:
            assert set(sp[sp.role==trainrole].site_id).isdisjoint(sp[sp.role==testrole].site_id)
    for model,forbidden in [('fusion_arc_only','BAC'),('fusion_bac_only','ARC')]:
        assert not len(edges[(edges.model==model)&(edges.domain==forbidden)])
    lookup,members=taxonomy_catalog(coeff)
    groups,contrasts,fitchecks,cohorts=analyze(coeff,preds,interactions,balances,inventory,splits,lookup)
    annotated,features,summary=interpretation_summaries(coeff,groups,inventory,lookup)
    active_interactions,isummary=interaction_tables(interactions,balances,inventory,lookup)
    metrics=performance_context(preds,inventory,cohorts)
    mismatch=members[((members.domain=='ARC')&(~members.taxonomic_domain.isin(['Archaea','unassigned'])))|((members.domain=='BAC')&(~members.taxonomic_domain.isin(['Bacteria','unassigned'])))]
    checks=dict(copied_files_verified=len(manifest['files']),planned_fits=len(inventory),successful_fits=int(inventory.status.eq('ok').sum()),unavailable_fits=int(inventory.status.ne('ok').sum()),coefficient_rows=len(coeff),groups=len(groups),active_groups=int(groups.active.sum()),
        max_group_reconstruction_error=fitchecks.group_plus_residual_error.max(),max_equality_residual=fitchecks.equality_residual.max(),
        unique_domain_mismatch_asvs=mismatch.asv_id.nunique(),unique_unassigned_domain_asvs=members[members.taxonomic_domain=='unassigned'].asv_id.nunique(),
        active_interaction_products=int(isummary.active_products.sum()),domain_restricted_graphs_verified=True,train_test_site_isolation_verified=True,
        historical_taxonomy_hashes_verified=True,historical_rank_mapping_hashes_available=False,
        software_versions={'python':sys.version.split()[0], 'numpy':np.__version__, 'pandas':pd.__version__, 'matplotlib':matplotlib.__version__},
        script_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    tables=dict(model_fit_inventory=inventory,coefficients_annotated=annotated,feature_asv_taxonomy=members,group_catalog=groups,feature_associations=features,model_domain_summary=summary,domain_contrasts=contrasts,
        interaction_summary=isummary,active_interactions_annotated=active_interactions,prediction_context=metrics,cohort_summary=cohorts,fit_validation=fitchecks,
        functional_evidence=pd.DataFrame(EVIDENCE,columns=['genus','role_hypothesis','evidence_level','primary_source','scope']),taxonomy_domain_mismatches=mismatch)
    for name,df in tables.items():df.to_csv(out/'tables'/f'{name}.csv',index=False)
    figures(out,annotated,groups,summary,isummary)
    reports(out,groups,features,summary,cohorts,inventory,metrics,isummary,checks)
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    # The separately checkpointed fits retain their own output hashes.
    artifacts=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for sub in ['tables','figures','details'] for p in sorted((out/sub).glob('*'))]
    artifacts += [dict(path=name,sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in ['REPORT.md','validation.json']]
    (out/'artifact_manifest.json').write_text(json.dumps(dict(files=artifacts),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

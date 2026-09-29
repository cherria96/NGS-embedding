"""Build the investigation report and compact comparison figures from CSVs."""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = _REPO_ROOT / "results/data investigation/fusionML"


def md(df):
    def cell(value):
        if isinstance(value,(float,np.floating)): return f'{value:.3f}'
        return str(value).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(str,df.columns))+' |',
           '| '+' | '.join(['---']*len(df.columns))+' |']
    lines.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    return '\n'.join(lines)


def main():
    original=pd.read_csv(OUT/'learning/pooled_metrics.csv')
    qc=pd.read_csv(OUT/'learning_qc/pooled_metrics.csv')
    rows=pd.read_csv(OUT/'learning_qc/rows.csv')
    pred=pd.read_csv(OUT/'learning_qc/predictions.csv')
    filt=pd.read_csv(OUT/'learning_filter20/pooled_metrics.csv')
    assert len(list((OUT/'learning_qc/parts').glob('*.json')))==36
    assert (qc[qc.fraction==1].n_outer_folds==3).all()
    model_order=['genus_ridge','genus_elastic_net','phylogeny_only','abundance_only','shared_hybrid',
                 'domain_adaptive_hybrid','domain_adaptive_interaction','elasticnet_RA_matched','xgboost_RA_matched',
                 'elasticnet_RA_flat','xgboost_RA_flat']
    tables={}
    for name,data in [('production_groups',original),('qc_groups',qc)]:
        full=data[(data.fraction==1)&(data.selection=='one_se')]
        table=full.groupby(['model','target'])[['auc','ap','logloss','mcc','f2']].mean().round(4)
        table.to_csv(OUT/f'{name}_comparison.csv')
        auc=table.auc.unstack().reindex(model_order)
        auc=auc.rename(columns={'warning_ammonia_toxicity':'Ammonia AUROC','warning_acid_base_balance':'Acid–base AUROC'})
        tables[name]=auc.reset_index()
    ablation=original[(original.fraction==1)&original.model.isin(['phylogeny_only','shared_hybrid','domain_adaptive_hybrid','adaptive_expanded'])]
    ablation=ablation.groupby(['target','model','selection'])[['auc','logloss']].mean().reset_index()
    ablation.to_csv(OUT/'selection_ablation.csv',index=False)
    filtering=[]
    for label,data in [('40%',qc),('20%',filt)]:
        full=data[(data.fraction==1)&(data.selection=='one_se')&data.model.isin(['genus_ridge','domain_adaptive_hybrid'])]
        x=full.groupby(['target','model'])[['auc','logloss']].mean().reset_index();x['prevalence_filter']=label;filtering.append(x)
    filtering=pd.concat(filtering);filtering.to_csv(OUT/'filter_sensitivity.csv',index=False)

    # Paired site bootstrap of fixed OOF losses, averaging repeat losses per
    # sample before site averaging. These intervals do not retrain models.
    from function.evaluate import per_sample_log_loss
    p=pred[(pred.fraction==1)&(pred.selection=='one_se')].copy()
    p['loss']=per_sample_log_loss(p.y.to_numpy(),p.p.to_numpy())
    site_loss=p.groupby(['target','model','site_id'])['loss'].mean()
    paired=[];rng=np.random.default_rng(42)
    for target in p.target.unique():
        ref=site_loss.loc[(target,'genus_ridge')]
        for model in model_order:
            test=site_loss.loc[(target,model)]
            delta=(test-ref).dropna().to_numpy()
            boot=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
            paired.append(dict(target=target,model=model,delta_logloss_vs_ridge=delta.mean(),
                               low=np.quantile(boot,.025),high=np.quantile(boot,.975),n_sites=len(delta)))
    pd.DataFrame(paired).to_csv(OUT/'paired_fixed_oof_bootstrap.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette={'genus_ridge':'#2878a7','domain_adaptive_hybrid':'#d87927','xgboost_RA_matched':'#39805a','xgboost_RA_flat':'#975b9a'}
    labels={'genus_ridge':'Log-contrast ridge','domain_adaptive_hybrid':'Adaptive fusion',
            'xgboost_RA_matched':'XGBoost, matched taxa','xgboost_RA_flat':'XGBoost, flat genus'}
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for i,target in enumerate(['warning_ammonia_toxicity','warning_acid_base_balance']):
        good=rows[(rows.target==target)&(rows.selection=='one_se')&(rows.status=='ok')]
        for model in ['genus_ridge','domain_adaptive_hybrid']:
            g=good[good.model==model].groupby('fraction').agg(n=('n_train_sites','mean'),train=('train_logloss','mean'),test=('test_logloss','mean'),fits=('outer_fold','size'))
            axs[i,0].plot(g.n,g.test,'o-',color=palette[model],label=labels[model]+': test')
            axs[i,0].plot(g.n,g.train,'o--',color=palette[model],label=labels[model]+': train')
            for _,r in g.iterrows():
                if r.fits<6: axs[i,0].annotate(f'{int(r.fits)}/6 fits',(r.n,r.test),xytext=(2,6),textcoords='offset points',fontsize=8)
        for model in palette:
            m=qc[(qc.target==target)&(qc.selection=='one_se')&(qc.model==model)&(qc.n_outer_folds==3)]
            g=m.groupby('fraction').agg(auc=('auc','mean'),repeats=('repeat','size'))
            sizes=good[good.model==model].groupby('fraction').n_train_sites.mean()
            axs[i,1].plot(sizes.reindex(g.index),g.auc,'o-',color=palette[model],label=labels[model])
            for fraction,r in g.iterrows():
                if r.repeats<2: axs[i,1].annotate('1 repeat',(sizes.loc[fraction],r.auc),xytext=(2,6),textcoords='offset points',fontsize=8)
        for ax in axs[i]:
            ax.set_title(target.replace('warning_','').replace('_',' '));ax.set_xlabel('Mean training site groups');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axs[i,0].set_ylabel('Equal-site log-loss (lower is better)');axs[i,1].set_ylabel('Pooled outer-CV AUROC (higher is better)')
    fig.suptitle('Learning curves: 134 samples, 35 groups with QC pairs kept together\n3 outer folds × 2 repeats; 2 inner folds; training-site subsets at 50%, 75%, 100%')
    fig.tight_layout();fig.savefig(OUT/'learning_curve_overview.png',dpi=180);fig.savefig(OUT/'learning_curve_overview.pdf');plt.close(fig)

    curves=rows[(rows.status=='ok')&(rows.selection=='one_se')&rows.model.isin(['genus_ridge','domain_adaptive_hybrid'])]
    curves=curves.groupby(['target','fraction','model']).agg(fits=('outer_fold','size'),training_sites=('n_train_sites','mean'),
                                                            train_loss=('train_logloss','mean'),test_loss=('test_logloss','mean')).reset_index()
    labels_audit=pd.read_csv(OUT/'label_audit.csv')
    compression=pd.read_csv(OUT/'saved_compression.csv')
    compression=compression[compression.run=='real_results_classification_phylo_interaction'].groupby('family')[['retained_taxa','active_taxa','active_groups']].mean().reset_index()
    text=f'''**Why fusionML is not consistently improving prediction — investigation, 21 September 2026**

Fusion is reducing the number of coefficient groups, but that is not a guarantee of higher held-out accuracy. The evidence points to a combination of a restrictive search budget, selection that deliberately favors compression, a small number of independent positive sites, representation/filtering effects, and graph relationships that do not necessarily match outcome effects. The premise that every ordinary ML baseline beats fusion is also too broad: matched comparisons do not support that for every model or target.

This investigation added analysis scripts and outputs only. It did not change production fitting, replace saved models, or alter data. The saved runs have changed since the earlier code-structure discussion: the current interaction grid has all nine interaction-penalty combinations and additional rank runs, which were audited here.

**What “dimension reduction” means here**

The graph penalty is λF Σ w(j,k)|βj−βk|. When a connected group has equal coefficients, its contribution can be expressed using its mean log abundance (equivalently, its log geometric mean). This is different from summing counts into one ordinary taxonomic relative-abundance column. The implementation still starts with the retained taxa, solves for their coefficients, and saves the taxon-level predictor. Group extraction is primarily an interpretation/compression step; small residual coefficients are preserved. Group count is not a measured statistical effective degrees of freedom.

In the saved genus ammonia run, mean counts across its three outer fits are:

{md(compression)}

Shared hybrid therefore compresses about 96 retained taxa into 11 active groups. Its saved AUROC is 0.870 versus 0.909 for ridge. The failure to improve is not a failure to perform grouping. An incorrect equality assumption can discard outcome information while reducing variance. This tradeoff is also documented in [the original trac study](https://www.nature.com/articles/s41598-021-93645-3), where less aggregated alternatives sometimes outperform more parsimonious models.

The [regularization-path figure](compression_regularization.png) provides a direct check. At fixed ridge=0.1 and sparsity=0, increasing phylogenetic fusion in the original ammonia folds reduces active groups from 96.3 to 41.0 at λF=1 and 4.3 at λF=100; held-out log-loss moves from 0.361 to 0.341 and then 0.426. Moderate fusion can help, and still more compression can hurt. These paths are descriptive; outer-test values were not used to choose a deployment model.

**Concrete implementation and selection reasons**

1. The adaptive smoke search effectively does not tune the main penalties. [Candidate generation](../search.py) reserves one no-fusion candidate plus all nine αARC×αBAC combinations. That consumes the entire budget of 10. Every adaptive candidate uses λS=0 and λR=0.1; positive λF is always 1. The YAML lists other values, but they never get searched by this family. Its no-fusion fallback is only one ridge setting, whereas the ridge family tests three. A more general family is therefore not getting a comparable search over its simpler submodels.

2. The one-standard-error rule explicitly trades validation loss for a small group count. It takes the lowest-complexity model within one standard error of the best inner score. This can prefer a poorly calibrated, aggressively fused model, or a sparse model with weak ridge shrinkage. A low count of nonzero groups is not a reliable proxy for low variance when coefficient magnitudes and correlated predictors differ. I tested the original rule and minimum-inner-loss selection using exactly the same candidates, inner predictions, and outer tests. The following uses the production site labels so it isolates selection within that experiment:

{md(ablation)}

The expanded grid evaluates 168 additive candidates, including all six no-fusion penalty settings and λF=0.1, 1, 10 at every mixture. Expansion alone does not consistently improve the result. Minimum-loss selection helps several families, but no universal winning fusion setting emerges. These are exploratory comparisons on the same cohort, not a newly validated model-selection rule.

3. Interaction terms are present as candidates but mostly absent from fitted predictions. All 27 saved outer fits across the nine genus/family/order interaction runs have zero active product coefficients at the configured 1e-6 tolerance. The saved final genus ammonia and acid–base interaction models also have no active product coefficients, despite the selected candidate being labeled interaction-on. The search selects at most two balances per domain by training variance, giving at most four products. This does not test arbitrary taxon interactions, and large-variance balances need not be outcome-relevant. Different scores for the additive and interaction families can arise from different tuning splits, graph seeds, and selected additive parameters even when gamma is zero.

4. The original [inner search](../pipeline.py) includes the family name in its split seed, and both family and candidate in the graph bootstrap seed. Even the additive reference and its interaction-off counterpart are evaluated on different inner splits and graph realizations. This adds comparison noise. The new experiment uses identical inner folds and partition graph seeds across families and reuses identical fits. In that setting, interaction and additive predictions often coincide.

5. The graph is a prior about coefficient equality, not proof of shared biology or predictive effect. Phylogenetic proximity and low log-ratio variance are unsupervised relationships. A training-only ridge-coefficient probe found that bacterial phylogenetic edges connect coefficients with absolute differences 1.04–1.18 times a within-domain permutation reference in the three original ammonia folds. Abundance edges were more aligned, at 0.56–0.82 times that reference. These noisy fitted-coefficient diagnostics are exploratory, but they do not support assuming that every phylogenetic neighbor should have the same outcome effect. See [graph_prior_probe.csv](graph_prior_probe.csv).

6. Filtering and representation matter before fusion begins. The smoke configuration requires 40% sample and site prevalence. At mapped genus resolution, 389 bacterial features become 67–82 per outer training fold, retaining only 65–72% of bacterial abundance on average. For individual samples retained mass can be as low as 1.7%; the pseudocount fraction reaches 47.9%. Archaea retain roughly 96% of mass. Logs, closure after filtering, and this unequal loss of mass change the predictor substantially relative to raw taxonomic abundance. [Filter audit](filter_audit.csv). A 20% prevalence sensitivity using the same QC-grouped splits gives:

{md(filtering)}

This changes both taxa and fitted graph/regularization behavior; it is not a causal isolation of pseudocounts. It does not establish that a gentler filter always improves prediction. It does show that the 40% smoke override should be treated as a tunable scientific assumption, not a harmless runtime setting.

7. The final saved family is chosen by a fixed preference list in [train_final_deployable_model](../pipeline.py), starting with interaction when available. It is not the best-performing family from the comparison table. Thus “final_model” can be a worse-performing family by design. Selection among families should itself be nested if a deployable winner is desired.

**Learning-curve experiment and matched comparison**

The experiments use the same 134 joint-cohort samples and freshly computed chemistry labels for every model. There are 37 literal site labels and 35 groups after keeping the documented BSI/BSIb and DGY/DGYb QC pairs together. The table below uses those 35 groups. For each target: two repeats of three outer folds, two inner folds, and nested random subsets containing 50%, 75%, or 100% of outer-training groups. Whole groups are sampled without looking at outcomes. Features, graphs, interaction definitions, scaling, hyperparameters and F2 thresholds are learned inside the corresponding training partition. No outer labels determine a tuning choice.

Each repeat's outer predictions are pooled before computing AUROC/AP/MCC/F2; table entries are the mean over two repeats. The largest training fits contain only about 23 groups/89 samples, not all 134 samples. Fusion models use their current candidate pools and one-SE rule. Ordinary ML candidates minimize the same inner equal-site log-loss, use equal-site training weights, and use the same inner-OOF F2 threshold rule. The experiments do not use class balancing; this differs from the legacy benchmark and is intentional to keep the predictive loss comparable.

“RA matched” means the same mapped genus taxa retained by fusion preprocessing, but modeled as relative abundance. “RA flat” means ordinary genus-label sum aggregation from the same raw ASV counts, including an Unclassified group, with only training-constant columns removed. Elastic net is sklearn logistic elastic net (the same algorithm class used by the repository's GLMNET analogue), not R glmnet. XGBoost tests depths 1/3 and learning rates 0.05/0.1 with 200 trees. These are controlled, bounded comparisons, not exhaustive tuning of the baselines.

{md(tables['qc_groups'])}

Full probability loss, AP, MCC and F2 are in [qc_groups_comparison.csv](qc_groups_comparison.csv). The companion results using literal production site labels are in [production_groups_comparison.csv](production_groups_comparison.csv). This distinction changes fold composition as well as grouping, so differences between the two runs cannot be attributed solely to leakage removal.

![Learning-curve overview](learning_curve_overview.png)

The log-loss panels compare resubstitution training scores with genuinely held-out groups. The AUROC panels use only repeats with all three outer folds available; annotations identify where only one repeat is usable. The full family-by-family [ammonia curves](learning_qc/warning_ammonia_toxicity_learning_curves.png) and [acid–base curves](learning_qc/warning_acid_base_balance_learning_curves.png) show all successful fits and their variability.

{md(curves)}

Large train–test gaps are consistent with variance/overfitting and site-to-site differences, even after group compression. Some held-out losses improve with more sites, but these short, noisy curves cannot establish an asymptote or predict how many additional sites are needed. Small subsets sometimes produce single-class inner training data. Those fits are recorded as unavailable, not replaced by favorable constant predictions. Scores at a fraction with fewer than six successful fits are conditional on the successful subsets, so their apparent slope is not a clean paired sample-size effect. See [availability](learning_qc/availability.csv) and the saved split manifest. The production-label experiment also has a fully available 75%→100% comparison for both targets, supporting the continuing train–test-gap conclusion.

These are sample-size learning curves, not epoch curves. FusionML solves a convex optimization problem with fixed features; it has no neural-network training epochs. The relevant question is what happens when it sees more independent sites. [Official learning-curve documentation](https://scikit-learn.org/stable/modules/learning_curve.html) describes this distinction and the train–validation-gap interpretation.

**Why the historical benchmark numbers cannot be ranked directly**

The label audit on exactly the same 134 IDs is:

{md(labels_audit)}

The historical metadata gives 6 biogas positives versus 13 from the current workbook, and changes five buffer-capacity labels and one ammonia label. The source formulas also treat missing trigger measurements as zero/unflagged in several cases; no missingness policy was changed for these experiments. [Changed labels](changed_labels.csv).

The ML rank reports often quote a fixed 29-sample test containing only four ammonia-positive and five acid–base-positive samples; fusionML quotes pooled outer-CV predictions on 134 samples. Scores such as AUROC 0.98 on that small holdout cannot be compared as if they were the same evaluation. ML's `CV_OOF_optimized` scores are tuning scores reused for candidate/threshold selection, explicitly acknowledged as optimistic in those reports. ML also uses percentage thresholds, different source abundance tables, class balancing, and trapezoidal PR area rather than fusionML's average precision. [Legacy ML metrics](legacy_ml_metrics.csv).

The later PhyloSpec benchmark is already site-grouped and stratified; it should not be dismissed as ordinary random-sample CV. However, [evaluate_multilabel.evaluate_cv](../../Phylospec/evaluate_multilabel.py) selects one threshold from validation predictions pooled across all outer folds and applies it to all pooled test predictions. A sample held out in one fold can be a validation sample in another fold. Reconstructing repeat 0 found 15–19 test sample IDs per fold among the pooled threshold-tuning labels (83/134 distinct IDs overall). That violates outer-fold isolation of the operating point and can bias MCC/F2; it does not directly contaminate the threshold-free AUROC/AP scores. [Overlap audit](legacy_threshold_overlap.csv). This issue was documented, not silently patched.

FusionML's grouping also needs attention: its current seed happens to keep both QC pairs together in the saved outer folds, but several family-specific inner folds split them. In a second outer seed BSI/BSIb split as well. The 35-group experiment fixes this for its own runs without changing production data. [QC audit](qc_pair_partition_audit.csv).

Finally, fusionML's `metadr_order_only` is a traversal-order graph penalty inside the same log-contrast model. It is not the actual MetaDR CNN. The repository's MetaDR and PhyloSpec implementations learn nonlinear convolutional functions, whereas the main fusion predictor remains linear in log compositions; zero surviving interaction terms leave it additive. I audited their existing protocols/results but did not retrain those neural models. The new learning curves therefore establish comparisons against ridge/elastic net/XGBoost, not a definitive reranking of MetaDR/PhyloSpec.

**What to change first**

1. Freeze a common, versioned cohort/label table and biological site-group mapping. Use the same outer and inner folds for all families; choose each fold's threshold only from that fold's training data. Report AP and ROC alongside probability loss, operating-point metrics, coverage and positive-site support.
2. Give every fusion family the complete no-fusion baseline grid, reserve enough budget for λR/λS/λF, and share graph bootstrap realizations across candidates. The current adaptive model has almost no effective search over its main penalties.
3. Treat compression as a separate reported objective. Compare minimum-inner-loss selection against the existing one-SE compression rule in a prespecified evaluation; do not adopt a rule because it won these same outer tests. Track coefficient magnitude/calibration as well as group count.
4. Tune prevalence/pseudocount choices inside training only, inspect low retained-mass samples, and compare ordinary rank abundance against log-contrast features. Do not attribute all representation differences to “using phylogeny.”
5. Treat the interaction extension as unproven here. More candidates or more balances alone may increase variance. Check reliable positive-site support, selected balance stability and active gamma counts before increasing interaction complexity.
6. If the goal is best prediction, compare a properly tuned simpler model and select the family inside CV. If the goal is an interpretable smaller representation, quantify the accuracy–compression tradeoff explicitly. Obtaining more independent sites is better supported by the curves than collecting only more seasons of existing sites, but the gain is not yet quantified.

**Verification and artifacts**

Thirteen existing constraint, solver, fold-isolation, group-algebra and scale-invariance tests passed. A direct uncached production fit and the cached investigation fit produced identical held-out probabilities (maximum absolute difference 0) with zero-sum residual 1.27e-14. See [cache_validation.json](cache_validation.json). Solver attempts use the existing production success checks; single-class failures remain failures.

The two learning-curve runs comprise 72 target/repeat/fold/size tasks; the filtering sensitivity adds 12 tasks. Each saved part contains selected candidates, inner scores, outer predictions, group counts, failures and split IDs. Two repeats and small positive-site counts do not support a definitive significance claim. [Paired bootstrap intervals](paired_fixed_oof_bootstrap.csv) resample fixed OOF site losses and omit model-retraining uncertainty; they are descriptive.

Reproduction from the repository root, using the existing `torch` conda Python:

```bash
export MPLCONFIGDIR=/tmp/fusionml-investigation-mpl
export XDG_CACHE_HOME=/tmp/fusionml-investigation-cache
export OPENBLAS_NUM_THREADS=1
python fusionML/investigation_20260921/audit.py
python fusionML/investigation_20260921/learning_curves.py --workers 3 --repeats 2
python fusionML/investigation_20260921/learning_curves.py --group-qc --output fusionML/investigation_20260921/learning_qc --workers 3 --repeats 2
python fusionML/investigation_20260921/learning_curves.py --group-qc --min-prevalence 0.2 --models genus_ridge domain_adaptive_hybrid --fractions 1 --output fusionML/investigation_20260921/learning_filter20 --workers 2 --repeats 2
python fusionML/investigation_20260921/mechanism_probes.py
python -m fusionML.investigation_20260921.summarize_investigation
python -m pytest -q fusionML/tests/test_constraints.py fusionML/tests/test_solver.py fusionML/tests/test_fold_isolation.py fusionML/tests/test_group_algebra.py fusionML/tests/test_scale_invariance.py
```

Completed parts are reused by the learning script. Use a new output directory when changing experiment settings. Manifests record source/workbook hashes and settings. External context used Exa to review 11 search results across aggregation and learning-curve topics, plus the two primary pages cited above; the conclusions here rest on the local audits and experiments.
'''
    (OUT/'REPORT.md').write_text(text)
    print('Wrote',OUT/'REPORT.md')


if __name__=='__main__': main()

**Why fusionML is not consistently improving prediction — investigation, 21 September 2026**

Fusion is reducing the number of coefficient groups, but that is not a guarantee of higher held-out accuracy. The evidence points to a combination of a restrictive search budget, selection that deliberately favors compression, a small number of independent positive sites, representation/filtering effects, and graph relationships that do not necessarily match outcome effects. The premise that every ordinary ML baseline beats fusion is also too broad: matched comparisons do not support that for every model or target.

This investigation added analysis scripts and outputs only. It did not change production fitting, replace saved models, or alter data. The saved runs have changed since the earlier code-structure discussion: the current interaction grid has all nine interaction-penalty combinations and additional rank runs, which were audited here.

**What “dimension reduction” means here**

The graph penalty is λF Σ w(j,k)|βj−βk|. When a connected group has equal coefficients, its contribution can be expressed using its mean log abundance (equivalently, its log geometric mean). This is different from summing counts into one ordinary taxonomic relative-abundance column. The implementation still starts with the retained taxa, solves for their coefficients, and saves the taxon-level predictor. Group extraction is primarily an interpretation/compression step; small residual coefficients are preserved. Group count is not a measured statistical effective degrees of freedom.

In the saved genus ammonia run, mean counts across its three outer fits are:

| family | retained_taxa | active_taxa | active_groups |
| --- | --- | --- | --- |
| abundance_only | 96.333 | 58.333 | 37.000 |
| domain_adaptive_hybrid | 96.333 | 96.333 | 41.667 |
| domain_adaptive_interaction | 96.333 | 96.333 | 47.667 |
| genus_elastic_net | 96.333 | 59.000 | 59.000 |
| genus_ridge | 96.333 | 96.333 | 96.333 |
| metadr_order_only | 96.333 | 72.667 | 3.667 |
| phylogeny_only | 96.333 | 68.000 | 23.333 |
| shared_hybrid | 96.333 | 94.667 | 11.000 |

Shared hybrid therefore compresses about 96 retained taxa into 11 active groups. Its saved AUROC is 0.870 versus 0.909 for ridge. The failure to improve is not a failure to perform grouping. An incorrect equality assumption can discard outcome information while reducing variance. This tradeoff is also documented in [the original trac study](https://www.nature.com/articles/s41598-021-93645-3), where less aggregated alternatives sometimes outperform more parsimonious models.

The [regularization-path figure](compression_regularization.png) provides a direct check. At fixed ridge=0.1 and sparsity=0, increasing phylogenetic fusion in the original ammonia folds reduces active groups from 96.3 to 41.0 at λF=1 and 4.3 at λF=100; held-out log-loss moves from 0.361 to 0.341 and then 0.426. Moderate fusion can help, and still more compression can hurt. These paths are descriptive; outer-test values were not used to choose a deployment model.

**Concrete implementation and selection reasons**

1. The adaptive smoke search effectively does not tune the main penalties. [Candidate generation](../search.py) reserves one no-fusion candidate plus all nine αARC×αBAC combinations. That consumes the entire budget of 10. Every adaptive candidate uses λS=0 and λR=0.1; positive λF is always 1. The YAML lists other values, but they never get searched by this family. Its no-fusion fallback is only one ridge setting, whereas the ridge family tests three. A more general family is therefore not getting a comparable search over its simpler submodels.

2. The one-standard-error rule explicitly trades validation loss for a small group count. It takes the lowest-complexity model within one standard error of the best inner score. This can prefer a poorly calibrated, aggressively fused model, or a sparse model with weak ridge shrinkage. A low count of nonzero groups is not a reliable proxy for low variance when coefficient magnitudes and correlated predictors differ. I tested the original rule and minimum-inner-loss selection using exactly the same candidates, inner predictions, and outer tests. The following uses the production site labels so it isolates selection within that experiment:

| target | model | selection | auc | logloss |
| --- | --- | --- | --- | --- |
| warning_acid_base_balance | adaptive_expanded | min_loss | 0.640 | 0.425 |
| warning_acid_base_balance | adaptive_expanded | one_se | 0.595 | 0.505 |
| warning_acid_base_balance | domain_adaptive_hybrid | min_loss | 0.670 | 0.521 |
| warning_acid_base_balance | domain_adaptive_hybrid | one_se | 0.673 | 0.525 |
| warning_acid_base_balance | phylogeny_only | min_loss | 0.667 | 0.426 |
| warning_acid_base_balance | phylogeny_only | one_se | 0.653 | 0.425 |
| warning_acid_base_balance | shared_hybrid | min_loss | 0.668 | 0.426 |
| warning_acid_base_balance | shared_hybrid | one_se | 0.619 | 0.467 |
| warning_ammonia_toxicity | adaptive_expanded | min_loss | 0.900 | 0.318 |
| warning_ammonia_toxicity | adaptive_expanded | one_se | 0.860 | 0.358 |
| warning_ammonia_toxicity | domain_adaptive_hybrid | min_loss | 0.907 | 0.343 |
| warning_ammonia_toxicity | domain_adaptive_hybrid | one_se | 0.894 | 0.330 |
| warning_ammonia_toxicity | phylogeny_only | min_loss | 0.905 | 0.290 |
| warning_ammonia_toxicity | phylogeny_only | one_se | 0.863 | 0.395 |
| warning_ammonia_toxicity | shared_hybrid | min_loss | 0.910 | 0.296 |
| warning_ammonia_toxicity | shared_hybrid | one_se | 0.882 | 0.325 |

The expanded grid evaluates 168 additive candidates, including all six no-fusion penalty settings and λF=0.1, 1, 10 at every mixture. Expansion alone does not consistently improve the result. Minimum-loss selection helps several families, but no universal winning fusion setting emerges. These are exploratory comparisons on the same cohort, not a newly validated model-selection rule.

3. Interaction terms are present as candidates but mostly absent from fitted predictions. All 27 saved outer fits across the nine genus/family/order interaction runs have zero active product coefficients at the configured 1e-6 tolerance. The saved final genus ammonia and acid–base interaction models also have no active product coefficients, despite the selected candidate being labeled interaction-on. The search selects at most two balances per domain by training variance, giving at most four products. This does not test arbitrary taxon interactions, and large-variance balances need not be outcome-relevant. Different scores for the additive and interaction families can arise from different tuning splits, graph seeds, and selected additive parameters even when gamma is zero.

4. The original [inner search](../pipeline.py) includes the family name in its split seed, and both family and candidate in the graph bootstrap seed. Even the additive reference and its interaction-off counterpart are evaluated on different inner splits and graph realizations. This adds comparison noise. The new experiment uses identical inner folds and partition graph seeds across families and reuses identical fits. In that setting, interaction and additive predictions often coincide.

5. The graph is a prior about coefficient equality, not proof of shared biology or predictive effect. Phylogenetic proximity and low log-ratio variance are unsupervised relationships. A training-only ridge-coefficient probe found that bacterial phylogenetic edges connect coefficients with absolute differences 1.04–1.18 times a within-domain permutation reference in the three original ammonia folds. Abundance edges were more aligned, at 0.56–0.82 times that reference. These noisy fitted-coefficient diagnostics are exploratory, but they do not support assuming that every phylogenetic neighbor should have the same outcome effect. See [graph_prior_probe.csv](graph_prior_probe.csv).

6. Filtering and representation matter before fusion begins. The smoke configuration requires 40% sample and site prevalence. At mapped genus resolution, 389 bacterial features become 67–82 per outer training fold, retaining only 65–72% of bacterial abundance on average. For individual samples retained mass can be as low as 1.7%; the pseudocount fraction reaches 47.9%. Archaea retain roughly 96% of mass. Logs, closure after filtering, and this unequal loss of mass change the predictor substantially relative to raw taxonomic abundance. [Filter audit](filter_audit.csv). A 20% prevalence sensitivity using the same QC-grouped splits gives:

| target | model | auc | logloss | prevalence_filter |
| --- | --- | --- | --- | --- |
| warning_acid_base_balance | domain_adaptive_hybrid | 0.652 | 0.537 | 40% |
| warning_acid_base_balance | genus_ridge | 0.677 | 0.443 | 40% |
| warning_ammonia_toxicity | domain_adaptive_hybrid | 0.881 | 0.360 | 40% |
| warning_ammonia_toxicity | genus_ridge | 0.911 | 0.297 | 40% |
| warning_acid_base_balance | domain_adaptive_hybrid | 0.693 | 0.580 | 20% |
| warning_acid_base_balance | genus_ridge | 0.732 | 0.429 | 20% |
| warning_ammonia_toxicity | domain_adaptive_hybrid | 0.894 | 0.363 | 20% |
| warning_ammonia_toxicity | genus_ridge | 0.904 | 0.304 | 20% |

This changes both taxa and fitted graph/regularization behavior; it is not a causal isolation of pseudocounts. It does not establish that a gentler filter always improves prediction. It does show that the 40% smoke override should be treated as a tunable scientific assumption, not a harmless runtime setting.

7. The final saved family is chosen by a fixed preference list in [train_final_deployable_model](../pipeline.py), starting with interaction when available. It is not the best-performing family from the comparison table. Thus “final_model” can be a worse-performing family by design. Selection among families should itself be nested if a deployable winner is desired.

**Learning-curve experiment and matched comparison**

The experiments use the same 134 joint-cohort samples and freshly computed chemistry labels for every model. There are 37 literal site labels and 35 groups after keeping the documented BSI/BSIb and DGY/DGYb QC pairs together. The table below uses those 35 groups. For each target: two repeats of three outer folds, two inner folds, and nested random subsets containing 50%, 75%, or 100% of outer-training groups. Whole groups are sampled without looking at outcomes. Features, graphs, interaction definitions, scaling, hyperparameters and F2 thresholds are learned inside the corresponding training partition. No outer labels determine a tuning choice.

Each repeat's outer predictions are pooled before computing AUROC/AP/MCC/F2; table entries are the mean over two repeats. The largest training fits contain only about 23 groups/89 samples, not all 134 samples. Fusion models use their current candidate pools and one-SE rule. Ordinary ML candidates minimize the same inner equal-site log-loss, use equal-site training weights, and use the same inner-OOF F2 threshold rule. The experiments do not use class balancing; this differs from the legacy benchmark and is intentional to keep the predictive loss comparable.

“RA matched” means the same mapped genus taxa retained by fusion preprocessing, but modeled as relative abundance. “RA flat” means ordinary genus-label sum aggregation from the same raw ASV counts, including an Unclassified group, with only training-constant columns removed. Elastic net is sklearn logistic elastic net (the same algorithm class used by the repository's GLMNET analogue), not R glmnet. XGBoost tests depths 1/3 and learning rates 0.05/0.1 with 200 trees. These are controlled, bounded comparisons, not exhaustive tuning of the baselines.

| model | Acid–base AUROC | Ammonia AUROC |
| --- | --- | --- |
| genus_ridge | 0.676 | 0.911 |
| genus_elastic_net | 0.678 | 0.900 |
| phylogeny_only | 0.585 | 0.862 |
| abundance_only | 0.693 | 0.912 |
| shared_hybrid | 0.609 | 0.872 |
| domain_adaptive_hybrid | 0.652 | 0.881 |
| domain_adaptive_interaction | 0.651 | 0.878 |
| elasticnet_RA_matched | 0.660 | 0.890 |
| xgboost_RA_matched | 0.688 | 0.912 |
| elasticnet_RA_flat | 0.624 | 0.777 |
| xgboost_RA_flat | 0.708 | 0.854 |

Full probability loss, AP, MCC and F2 are in [qc_groups_comparison.csv](qc_groups_comparison.csv). The companion results using literal production site labels are in [production_groups_comparison.csv](production_groups_comparison.csv). This distinction changes fold composition as well as grouping, so differences between the two runs cannot be attributed solely to leakage removal.

![Learning-curve overview](learning_curve_overview.png)

The log-loss panels compare resubstitution training scores with genuinely held-out groups. The AUROC panels use only repeats with all three outer folds available; annotations identify where only one repeat is usable. The full family-by-family [ammonia curves](learning_qc/warning_ammonia_toxicity_learning_curves.png) and [acid–base curves](learning_qc/warning_acid_base_balance_learning_curves.png) show all successful fits and their variability.

| target | fraction | model | fits | training_sites | train_loss | test_loss |
| --- | --- | --- | --- | --- | --- | --- |
| warning_acid_base_balance | 0.500 | domain_adaptive_hybrid | 6 | 12.000 | 0.078 | 0.724 |
| warning_acid_base_balance | 0.500 | genus_ridge | 6 | 12.000 | 0.162 | 0.569 |
| warning_acid_base_balance | 0.750 | domain_adaptive_hybrid | 5 | 18.000 | 0.110 | 0.677 |
| warning_acid_base_balance | 0.750 | genus_ridge | 5 | 18.000 | 0.184 | 0.543 |
| warning_acid_base_balance | 1.000 | domain_adaptive_hybrid | 6 | 23.333 | 0.160 | 0.534 |
| warning_acid_base_balance | 1.000 | genus_ridge | 6 | 23.333 | 0.261 | 0.440 |
| warning_ammonia_toxicity | 0.500 | domain_adaptive_hybrid | 5 | 12.000 | 0.066 | 0.355 |
| warning_ammonia_toxicity | 0.500 | genus_ridge | 5 | 12.000 | 0.089 | 0.281 |
| warning_ammonia_toxicity | 0.750 | domain_adaptive_hybrid | 6 | 18.000 | 0.085 | 0.384 |
| warning_ammonia_toxicity | 0.750 | genus_ridge | 6 | 18.000 | 0.139 | 0.299 |
| warning_ammonia_toxicity | 1.000 | domain_adaptive_hybrid | 6 | 23.333 | 0.125 | 0.360 |
| warning_ammonia_toxicity | 1.000 | genus_ridge | 6 | 23.333 | 0.171 | 0.297 |

Large train–test gaps are consistent with variance/overfitting and site-to-site differences, even after group compression. Some held-out losses improve with more sites, but these short, noisy curves cannot establish an asymptote or predict how many additional sites are needed. Small subsets sometimes produce single-class inner training data. Those fits are recorded as unavailable, not replaced by favorable constant predictions. Scores at a fraction with fewer than six successful fits are conditional on the successful subsets, so their apparent slope is not a clean paired sample-size effect. See [availability](learning_qc/availability.csv) and the saved split manifest. The production-label experiment also has a fully available 75%→100% comparison for both targets, supporting the continuing train–test-gap conclusion.

These are sample-size learning curves, not epoch curves. FusionML solves a convex optimization problem with fixed features; it has no neural-network training epochs. The relevant question is what happens when it sees more independent sites. [Official learning-curve documentation](https://scikit-learn.org/stable/modules/learning_curve.html) describes this distinction and the train–validation-gap interpretation.

**Why the historical benchmark numbers cannot be ranked directly**

The label audit on exactly the same 134 IDs is:

| target | n | sites | positives | positive_sites | old_positive_matched | changed_labels |
| --- | --- | --- | --- | --- | --- | --- |
| warning_ammonia_toxicity | 134 | 37 | 24 | 10 | 25 | 1 |
| warning_acid_base_balance | 134 | 37 | 20 | 12 | 20 | 0 |
| warning_biogas_quality | 134 | 37 | 13 | 6 | 6 | 7 |
| warning_acid_accumulation | 134 | 37 | 4 | 3 | 4 | 0 |
| warning_buffer_capacity | 134 | 37 | 5 | 3 | 8 | 5 |

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

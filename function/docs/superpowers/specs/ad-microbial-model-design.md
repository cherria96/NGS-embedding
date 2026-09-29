**Adaptive microbial feature compression for anaerobic digestion prediction**

Proposed design for 140 samples from 36 sites, with at most four observations per site. This note interprets the supplied summary statistics; no raw-data reanalysis or model fitting has been performed. Methane yield is used as an example continuous target. The same feature construction can use logistic loss for an operational-state classification task.

The recommended starting point is a small supervised model that learns which microbial features can share a representation. Phylogenetic distance and abundance proportionality supply candidate relationships; prediction on held-out sites determines how much aggregation is useful. Archaea and bacteria should have separately adjustable grouping rules.

1. **Refine the premise before building the model.**

Three relationships must be distinguished: shared ancestry, functional similarity, and statistical redundancy of measured abundances. None implies the others universally. Functionally interchangeable organisms may replace one another and therefore need not co-vary positively. Conversely, organisms with complementary functions can co-occur because they depend on one another. The proposed model estimates redundancy for a prediction task; it cannot establish biological functional redundancy from 16S abundances alone.

Phylogenetic conservation depends on the trait and the evolutionary scale. Martiny and colleagues found different phylogenetic depths for different microbial traits, including relatively deep clustering of complex traits such as methanogenesis. That supports treating phylogeny as an adjustable prior, rather than fixing a common aggregation rank for all organisms. [Martiny et al., 2013](https://www.nature.com/articles/ismej2012160)

| Supplied observation | Defensible implication | Design consequence |
| --- | --- | --- |
| ARC Mantel correlations are positive, strongest at Class/Order | Archaeal abundance similarity is associated with the reported phylogenetic distances | Test stronger phylogenetic grouping for ARC, but allow it to be weak or absent |
| BAC Mantel correlations are near zero | No global relationship was detected by this particular abundance-distance analysis | Permit grouping across bacterial clades; retain phylogeny as an optional prior |
| BAC genera form two opposing abundance blocks | A low-dimensional compositional contrast may capture part of their variation | Test a contrast between the blocks, after rebuilding and checking the blocks in training data |
| Strongest selected associations concern sCOD and ALK | These variables motivate ecological follow-up | Test whether associated features predict the actual AD target; do not assume transfer to methane yield |

The results do not demonstrate that all archaeal functions are phylogenetically conserved, that bacterial functions are independent of ancestry, or that the opposing bacterial blocks are functional guilds. The ARC-versus-BAC difference also needs an uncertainty estimate for the difference itself; comparing significant and nonsignificant p-values is insufficient. Identical Class/Order results may reflect identical partitions and should not automatically count as independent evidence.

2. **Check whether the apparent redundancy survives compositional and site-aware analysis.**

Relative abundances share a denominator. Their correlations, including negative correlations, can reflect closure and changes in other organisms. Spearman correlation does not remove that problem. Examine proportionality and log-ratio variation before turning the observed blocks into model structure. [Lovell et al., 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075)

For taxa j and k, a useful training-data quantity is:

\[
v_{jk}=\operatorname{Var}\left[\log(q_j/q_k)\right],
\]

where q denotes a positive composition after documented zero handling. A small value means the ratio is comparatively stable: the two measurements supply little independent log-ratio variation in this dataset. This does not prove interchangeable metabolism. Co-absence and poorly measured low counts can also create misleading similarity, so require adequate joint detection and check zero-handling sensitivity.

Recompute relationships by resampling whole sites, retaining each selected site's observations together. Inspect pooled and within-site behavior, and sensitivity to major conditions such as temperature and feedstock. With only four visits, the within-site analysis is exploratory. Use site bootstrap replicates to assess stability of abundance similarities and the ARC-minus-BAC Mantel effect. This is distinct from the taxon-label permutations used by a conventional Mantel test.

The top-30 screen was selected from 9,724 comparisons and pairs had different sample sizes. Apply multiple-testing control and report paired sample counts for biological association claims. Avoid restricting model inputs to those selected genera: the screen was optimized for chemistry associations, not necessarily the chosen prediction target.

3. **Keep the ASV information available and use appropriate phylogenetic inputs.**

Start with the QIIME 2 feature count table, representative sequences, taxonomy, a sequence-derived phylogenetic tree with branch lengths, and sample metadata containing site and sampling date. QIIME 2's phylogeny workflow aligns and masks sequences before inferring a tree. [QIIME 2 phylogenetic tree tutorial](https://docs.qiime2.org/jupyterbooks/cancer-microbiome-intervention-tutorial/030-tutorial-downstream/030-phylogeny.html)

Use ASVs as the finest candidate resolution, allowing different regions of the tree to end at different aggregation depths. A genus-level model is a useful baseline. Taxonomic rank edges are not sequence branch lengths, and a genus may contain multiple ASVs or non-monophyletic assignments; do not assign a genus an arbitrary representative tip without checking the construction.

Define modest count/prevalence filtering and zero handling using training data only. Record how much abundance and how many taxa are excluded. Keep reproducibly detected low-abundance taxa when feasible. Avoid presenting recovery of excluded rare organisms as information preservation.

ARC and BAC were measured independently using two separate primer assays, as confirmed on 17 September 2026. Treat them as two separately closed compositions and match their rows by biological sample ID. Their sequencing read totals do not identify the true archaeal-to-bacterial cell or biomass ratio. Use separate within-domain log contrasts in the same prediction model. An independently calibrated domain-abundance measurement could be added later, with its units and interpretation stated explicitly. Separate measurements do not imply statistical or ecological independence between the domains.

4. **Construct two candidate relationship graphs within each domain.**

The phylogenetic graph favors nearby ASVs. One proposed weight is:

\[
W^{P}_{jk}=\exp(-d_{jk}/\tau_P),
\]

where d is the patristic distance: the sum of branch lengths along the path between the two tips. Use a sparse set of phylogenetic neighbors or local clade relationships, rather than connecting every pair. Short branches increase the preference to combine features; they never compel aggregation.

The abundance graph favors taxa whose log ratios vary little:

\[
W^{E}_{jk}=\exp(-v_{jk}/\tau_E).
\]

Retain relationships supported across site bootstrap samples and adequate joint detection. This graph can connect taxa in different clades. Call these abundance or proportionality groups until independent functional evidence supports a biological interpretation. Opposing blocks should not be positively fused using absolute correlation; their opposition can be retained as a contrast.

Normalize the total edge weight of each nonempty graph so that BAC's larger number of features does not automatically impose a larger graph penalty. For domain d, use:

\[
W^{(d)}=\alpha_d W^{P,(d)}+(1-\alpha_d)W^{E,(d)},\qquad 0\leq\alpha_d\leq1.
\]

Allow separate alpha values for ARC and BAC. The observations motivate the expectation alpha_ARC > alpha_BAC, but that is a hypothesis to test. Include zero phylogenetic influence and zero total fusion among the permitted configurations. Avoid converting the observed Mantel r directly into alpha: the quantities have different meanings.

Pairwise graph fusion uses phylogeny as a prior but does not guarantee that every inferred group is a complete monophyletic clade. If strict hierarchy preservation is required, use a tree-guided aggregation penalty or constrain ARC candidate groups to complete clades. The abundance component deliberately permits relationships that cross clade boundaries.

5. **Learn the compression through a supervised fusion model.**

For sample i, let c_i^A and c_i^B denote its separate archaeal and bacterial count vectors. Apply the chosen zero-handling procedure separately to obtain positive vectors tilde c. Define each domain's composition using its own total:

\[
q_{ij}^{A}=\frac{\tilde c_{ij}^{A}}{\sum_{r=1}^{p_A}\tilde c_{ir}^{A}},\qquad
q_{ik}^{B}=\frac{\tilde c_{ik}^{B}}{\sum_{r=1}^{p_B}\tilde c_{ir}^{B}}.
\]

Thus sum_j q_ij^A = 1 and sum_k q_ik^B = 1 independently. Here p_A and p_B are the numbers of retained taxa in each domain. Define l_i^A = log(q_i^A) and l_i^B = log(q_i^B), with elementwise logarithms. These are ordinary log compositions; they are not CLR vectors.

The additive prediction equation remains appropriate for the two separate assays:

\[
\hat y_i=b_0+(l_i^A)^\top\beta_A+(l_i^B)^\top\beta_B.
\]

Impose a separate zero-sum constraint for each domain:

\[
\mathbf 1^\top\beta_A=0,\qquad\mathbf 1^\top\beta_B=0.
\]

In expanded form, this is:

\[
\hat y_i=b_0+\sum_{j=1}^{p_A}\beta_{A,j}\log q_{ij}^{A}
+\sum_{k=1}^{p_B}\beta_{B,k}\log q_{ik}^{B},\qquad
\sum_j\beta_{A,j}=0,\quad\sum_k\beta_{B,k}=0.
\]

Each domain contributes only through its own log ratios. A single constraint sum_j beta_A,j + sum_k beta_B,k = 0 is weaker and does not ensure this within-domain interpretation. For example, it would permit one positive archaeal coefficient and one negative bacterial coefficient, which would compare fractions with different biological denominators.

The cancellation of each domain's denominator can be seen directly. If N_i^d is the total of its positive count vector, then:

\[
\sum_j\beta_{d,j}\log q_{ij}^{d}
=\sum_j\beta_{d,j}\log\tilde c_{ij}^{d}
-\log N_i^d\sum_j\beta_{d,j}
=\sum_j\beta_{d,j}\log\tilde c_{ij}^{d}.
\]

Multiplying the positive count vector of either domain by an arbitrary sample-specific positive factor therefore leaves its contribution unchanged. This algebra concerns the positive inputs after zero handling; it does not imply that changing sequencing depth produces identical observed reads or identical zero replacements. Separate normalization itself already removes count totals from q; the coefficient constraints additionally ensure that predictions can be expressed entirely through within-domain log contrasts. These constraints do not correct taxon-specific amplification bias.

A minimal example with two taxa per domain is:

\[
\hat y_i=b_0+a\log\frac{q_{i1}^{A}}{q_{i2}^{A}}
+b\log\frac{q_{i1}^{B}}{q_{i2}^{B}}.
\]

Both domains inform the same outcome without requiring their absolute abundance scales to be comparable. A genus at 20% in the archaeal table and a genus at 20% in the bacterial table are fractions of different communities; they are not thereby equally abundant in the digester. This adapts the established framework of log-contrast regression with separate constraints for compositional groups. [Shi, Zhang and Li, 2016](https://arxiv.org/abs/1603.00974)

For implementation, separately computed CLR or ILR features can be concatenated as predictors. Separate ILR bases have p_A - 1 and p_B - 1 coordinates and automatically encode the appropriate log contrasts. Separate CLR blocks each have a linear dependency; explicit coefficient constraints remain useful for interpretable taxon-level fusion. Computing features separately does not require fitting two independent prediction models.

An optional, prespecified set of operational covariates can enter through an additional linear term, provided those covariates are available at the intended prediction time.

A concrete training objective is:

\[
\min_{b_0,\beta_A,\beta_B}
\mathcal L_{\mathrm{site}}(y,\hat y)
+\lambda_S\|\beta\|_1+\lambda_R\|\beta\|_2^2
+\lambda_F\sum_{d\in\{A,B\}}\sum_{(j,k)\in E_d}W^{(d)}_{jk}|\beta_{d,j}-\beta_{d,k}|.
\]

Minimize this objective subject to both domain-specific zero-sum constraints above. Here beta concatenates both domains, and the fusion edges remain within each domain. The first penalty removes unhelpful features, ridge stabilizes remaining correlated effects, and the absolute-difference penalty promotes exact equality of connected coefficients. Site-balanced squared-error loss can be defined as:

\[
\mathcal L_{\mathrm{site}}=\frac1{36}\sum_{s=1}^{36}\frac1{n_s}\sum_{i\in s}(y_i-\hat y_i)^2,
\]

with the site count adjusted inside each training fold. Equal site weighting is an explicit choice to target performance at a typical facility; also report sample-weighted errors for transparency.

Fix graphs before optimizing coefficients in each training partition. With nonnegative penalties, fixed graphs, and squared-error loss, this coefficient-fitting problem is convex. Graph selection and hyperparameter selection remain part of the larger validation procedure. Begin with a short prespecified set of structural configurations and a small regularization search to limit model-selection overfitting.

Fusion provides actual feature compression. If beta_j equals b_G for every member of group G, then:

\[
\sum_{j\in G}\beta_j\log q_{ij}
=|G|b_G\log\left(\operatorname{GM}_{j\in G}q_{ij}\right).
\]

The group can therefore be represented by one log-geometric-mean feature. The fitted predictor is unchanged by this algebraic extraction. Across groups, the transformed coefficients retain the appropriate zero-sum constraint. This is geometric aggregation; it is not the same operation as summing sequencing counts.

The general foundation is established: tree-guided variable fusion and trac already learn prediction-relevant microbial aggregation. The proposal here combines phylogenetic and empirical relationships with independently adjustable domain weights. A claim that this is the first adaptive tree aggregation method would be incorrect. [Wang and Zhao, 2017](https://doi.org/10.1080/01621459.2016.1270213), [Bien et al., 2021](https://www.nature.com/articles/s41598-021-93645-3)

Apply fusion on coefficients expressed in the original log-feature scale. Naively standardizing every taxon separately changes what equal coefficients mean. If predictors are rescaled computationally, carry the inverse scaling into the compositional constraints and fusion penalty.

6. **Allow useful within-group differences to survive.**

For two candidate taxa, define a shared coordinate and a contrast:

\[
m=\frac{\log q_1+\log q_2}{2},\qquad c=\log(q_1/q_2).
\]

Then their contribution is:

\[
\beta_1\log q_1+\beta_2\log q_2=(\beta_1+\beta_2)m+\frac{\beta_1-\beta_2}{2}c.
\]

Equal coefficients remove the contrast from the predictor while retaining the shared term. Unequal coefficients retain the contrast. Thus, sufficiently useful differences can resist the fusion penalty, even for phylogenetically close organisms. Validation, rather than training fit alone, must determine the penalty strength.

For the observed bacterial blocks, an interpretable candidate is the balance:

\[
b_{12}=\sqrt{\frac{rs}{r+s}}\log\frac{\operatorname{GM}(q_{G_1})}{\operatorname{GM}(q_{G_2})}.
\]

If the supplied groups were retained, r=13 and s=8. However, rebuilding them within each training fold may change both sizes. A single balance would preserve the contrast between the groups but discard contrasts within them and between these organisms and the remaining community. It must be treated as a candidate feature, not a sufficient representation of the entire bacterial community. [PhILR: Silverman et al., 2017](https://elifesciences.org/articles/21887)

This design targets retention of predictive information. It cannot guarantee that every biological detail survives. Full abundance reconstruction is a different objective and may emphasize abundant, highly variable taxa that are unhelpful for predicting methane yield.

Clarification from the subsequent discussion: step 6 explains the behavior of the step-5 fit; it is not a second mandatory fitting stage. The b12 balance is optional. The primary fusion model uses individual log abundances and learns grouping through beta. A separate balance-based comparison may use b12 with its own coefficient. Do not add b12 alongside all the original log-abundance columns from which it was constructed, because that introduces an exactly redundant predictor. Details appear in item 15.

7. **Keep the first experiment small and evaluate generalization to sites.**

Use the same outer grouped folds for all models, keeping all observations from each site together. A workable starting design is six outer folds of approximately six sites and four inner grouped folds. Stratify or balance broad operating conditions where feasible without splitting a site. Any repeats should use a prespecified set of splits, rather than continuing until a favorable result appears.

All filtering, zero-handling parameter estimation, scaling, abundance graphs, group discovery, supervised selection, and tuning belong inside the training partitions, including the inner partitions. A sequence-based external reference phylogeny may be fixed independently; learned pruning, grouping, and abundance weights must not use held-out samples. Freeze the feature mapping before predicting each outer test fold.

Because the supplied full-dataset observations already informed the architecture, these results will constitute internal validation after exploratory analysis. A separate facility cohort is the strongest subsequent confirmation.

| Comparison | Purpose |
| --- | --- |
| Intercept and operational/chemistry-only baselines, where relevant | Establish the available nonmicrobial prediction signal |
| Genus-level compositional ridge or elastic-net model | Test whether adaptive ASV structure improves on a strong simple representation |
| ASV-level compositional ridge or elastic-net model | Measure the value of compression relative to regularization alone |
| Phylogenetic fusion only | Test the original tree-based hypothesis |
| Abundance fusion only | Test compression without ancestry |
| Hybrid graph with shared ARC/BAC weights | Test whether two sources help without domain adaptation |
| Hybrid graph with separate ARC/BAC weights | Test the proposed domain-specific extension |

Useful focused controls are branch-length versus topology-only weights, equal feature budgets, and shuffled phylogenetic tip labels. A small number of matched control runs can test whether the actual ancestry contributes beyond generic regularization. PhILR plus a regularized predictor and trac are strong additional literature baselines when implementation resources permit. PhILR gives p−1 coordinates for p taxa; it is not substantial dimension reduction by itself. Orthogonal coordinate axes also do not guarantee uncorrelated observed balance columns. [Silverman et al., 2017](https://elifesciences.org/articles/21887)

For continuous targets, report pooled out-of-fold R², RMSE and MAE, plus site-balanced errors. Do not calculate and average per-site R² when each site has only a few observations. Bootstrap sites for paired error differences between models, while recognizing that intervals from fixed out-of-fold predictions do not capture all model-training uncertainty. Report the usable sample and site count separately for each target.

For classification, substitute logistic loss and report appropriate discrimination and threshold metrics, including F2 if missed instability events are the priority. Select the decision threshold using training data only.

Plot out-of-fold prediction error against the number of retained group/contrast features over the regularization path. Prefer the smallest representation within a prespecified tolerance, such as one inner-CV standard error of the best error. Report group stability and the frequency with which fine ASV distinctions remain. Also compare a model that discards within-group distinctions with one that permits them; this directly tests whether adaptive detail retention matters.

Compression does not ensure all retained features are uncorrelated. Ridge controls the effects of remaining multicollinearity. If exact sample decorrelation is a separate requirement, train-only PCA of the retained coordinates can provide it, but its predictive value and interpretability cost must be evaluated separately.

8. **Add cross-domain coupling only as a targeted follow-up.**

The first model combines separate ARC and BAC log-contrast summaries additively. A later model can test a small prespecified number of products z_i^A z_i^B, where each z is a valid within-domain log-contrast or balance score, with strong regularization and grouped validation. Such terms ask whether the predictive contribution of a bacterial pattern depends on an archaeal pattern. They are candidates for interaction effects, not proof of syntrophy. Products of valid within-domain balances remain invariant to each domain's arbitrary scale; a raw log-geometric-mean group feature should not be assumed to have that property on its own. Ratios of read counts across the two primer assays cannot be interpreted as biological abundance ratios.

With at most four visits per site, this initial study is primarily a prediction of sampled state or performance. A claim of future forecasting requires explicitly time-aligned future targets and a validation design that respects prediction time.

The working hypothesis is: **Allowing phylogenetic grouping and abundance-based grouping to contribute differently in archaea and bacteria will retain AD-performance information in fewer features than fixed-rank aggregation.** Its support would come from an improved error-versus-dimension curve on unseen sites, reproducible selected groups, and evidence that branch lengths add value where they are used. Direct functional measurements would be needed to connect the learned compression to biological functional redundancy.

9. **Implementation scope and input contract.**

Required initial task: continuous-outcome regression, such as methane yield, with one explicitly configured target column per run. The target must be supplied or calculated from an explicitly documented user-approved formula and units. Classification, multiple simultaneous targets, nonlinear interactions, and neural encoders are extensions, not implicit substitutes for this regression. Do not use sCOD or ALK as targets or covariates merely because they appeared in the exploratory correlation screen.

Required inputs:

| Input | Required contents and behavior |
| --- | --- |
| ARC count table | Nonnegative raw counts, with biological sample IDs and archaeal feature IDs |
| BAC count table | Nonnegative raw counts, with biological sample IDs and bacterial feature IDs |
| Metadata | Unique sample_id, site_id, configured target; optional date/season and operational covariates |
| ARC and BAC trees | Separate sequence-derived Newick trees with branch lengths for phylogeny-enabled ASV models |
| Taxonomy mapping | Required for genus aggregation; feature ID to full lineage and genus label |
| Optional exploratory block list | Explicit genus IDs assigned to the earlier 13/8 groups; no guessed memberships |

Support CSV/TSV counts directly, with an explicit orientation setting. Prefer samples in rows, taxa in columns, and sample_id as the first column. BIOM and QIIME 2 adapters are optional conveniences; if QIIME 2 is unavailable, explain how exported counts, taxonomy and Newick files should be provided rather than pretending to read .qza files. Do not silently treat percentages as counts and apply a count pseudocount to them. Reject or require an explicit alternative preprocessing mode for relative-only tables.

Prefix internal IDs by domain, e.g. ARC::ASV001 and BAC::ASV001, so independently generated IDs cannot collide. Match rows by sample ID, not row order. Report duplicate IDs, missing metadata, nonfinite or negative counts, missing targets, missing assays, and sample exclusions. Do not average technical replicates or treat them as independent biological samples without an explicit mapping and rule.

The primary joint model uses samples with both assays and a measured target. Report this cohort before splitting. Entirely missing assays are not zero-count samples. A prediction request can omit targets but must supply the required assay features and sample IDs.

Make resolution configurable. ASV is the intended primary resolution when its inputs are available. For a genus baseline, sum raw counts by full lineage plus genus within each domain before zero handling. Keep unclassified taxa in explicit lineage-specific groups rather than combining all unknowns into one artificial genus. A taxonomic genus table alone does not supply branch lengths. For a genus-only input, run genus regularized and abundance-only models; mark phylogenetic variants unavailable until a justified tree mapping is supplied. Do not fabricate genus tips or a joint ARC/BAC phylogeny.

10. **Preprocessing fitted separately in every training partition.**

Use the following explicit default sequence independently for ARC and BAC:

- Define observed detection as raw count >= 5. This is a configurable working threshold, not a biological detection limit.
- Retain a taxon if detected in at least max(5, ceil(0.10*n_train)) training samples and max(3, ceil(0.10*S_train)) distinct training sites. Record actual thresholds and sensitivity to stricter filtering. The original 21 genera are not a supervised feature filter.
- Freeze the retained feature IDs and their order. Do not reselect taxa using validation or test data. Ignore new test-only features for prediction and report their abundance mass. A missing retained column may be filled with zero only if the input contract explicitly establishes that omitted features mean zero reads; otherwise report a schema error.
- Before smoothing, a sample must have positive total counts among retained features in both domains. A training sample that fails is excluded with its site and reason recorded; recompute training sample weights and verify that enough sites remain. Do not repeatedly loosen filters using test data. An unscorable validation/test row gets a missing prediction and an explicit reason; report coverage, and invalidate model selection if its prespecified coverage requirement fails. The default requires complete validation coverage. Never let pseudocounts turn a completely unsupported sample into an apparently observed uniform community.
- Add delta = 0.5 to every retained count, including nonzero counts, then close within each domain and take natural logarithms. This simple default is chosen for reproducible implementation. Record the pseudocount mass fraction p*delta/(N+p*delta) per sample/domain. It can be substantial for sparse ASV data. Run delta = 1.0 as a separate sensitivity analysis using the same folds; do not choose the better delta by outer-test error. A documented multiplicative-replacement mode may be added later.
- Do not z-score each microbial log column before taxon-level fusion. The penalty and zero-sum constraints refer to coefficients on the original log-composition scale. Intercept fitting is allowed. Do not concatenate counts and close both domains together.

If either domain has fewer than two retained taxa, fail that primary fit with an informative message rather than silently dropping its constraint. A domain-specific baseline can be requested explicitly as a separate model.

For optimization, standardize y using the training-site-weighted mean and population standard deviation. If y has zero variance, use an intercept-only fit and report that the training target is constant. Save the target transformation and invert it for every prediction and metric. With y_star = (y - mu_y)/sigma_y, coefficients in original target units are sigma_y*beta and the intercept is mu_y + sigma_y*b0. The coefficients still obey both zero-sum constraints.

Optional operational covariates are disabled by default. If enabled, specify their names and prediction-time availability; fit imputation, scaling and categorical encoding on training data only. Exclude site_id as an ordinary predictor for unseen-site prediction. Do not add target-derived variables. Penalize covariate coefficients through a separately documented ridge term, not microbial fusion or microbial zero-sum constraints. Report a matched covariate-only baseline.

11. **Compositional and site-aware redundancy diagnostics.**

Run exploratory diagnostics first at genus resolution if available. For 179 BAC and 25 ARC genera, this means up to 15,931 and 300 pairs respectively before filtering. Analyze all eligible within-domain pairs, not only the 21 selected genera. Use the selected 21 only as an annotated exploratory subset. The implementation must compute from supplied IDs; the names and memberships were not supplied in this specification.

For training site s with n_s observations, use sample weight w_i = 1/(S*n_s). These weights sum to one, giving each site equal total weight. For an eligible pair j,k, compute r_i = log(q_ij/q_ik), mu = sum_i w_i*r_i, and the site-balanced population log-ratio variance v_all = sum_i w_i*(r_i-mu)^2. This is a descriptive weighted variance, not an unbiased independent-sample variance estimator. All thresholds and plots must identify this convention.

Also compute v_joint using only samples where both taxa met the raw-count detection threshold. Rebuild weights on that subset so each represented site has equal weight. An empirical-graph pair is eligible only if both taxa were jointly detected in at least 10 training samples and 5 distinct training sites by default. Do not infer a stable relationship from shared zeros or assign zero variance to an empty or one-observation subset. Ineligible pairs receive no abundance edge but their taxa can remain in the model and in the phylogenetic graph.

For graph construction, use the conservative score v_star = max(v_all, v_joint). This is an explicit refinement of the initial variance-only proposal: it avoids labeling a pair stable solely because many smoothed co-absences flatten its full-cohort ratio. Export both components, not just the maximum. Repeat diagnostic estimates under the configured zero-handling sensitivity analysis.

For promising pairs, calculate site means m_s of r, the mean m_bar of those site means, V_between = mean_s (m_s-m_bar)^2, and V_within = mean_s mean_i_in_s (r_i-m_s)^2. With the equal-site population weighting above, v_all = V_between + V_within. Sites with one observation contribute zero observed within-site variance, which is not evidence of true temporal stability; report the number of sites with at least two visits. Plot per-site means and individual visits. Inspect major operating conditions descriptively where supplied; do not infer causal interactions or robust temporal dynamics from four visits.

For the original two blocks, display the variance matrix in the supplied block order and summarize within-block versus between-block variance. Re-estimated clustering may be exploratory, but must not be forced to contain two groups. Do not present a p-value from the same data-selected partition as independent confirmation. An observed opposition between blocks does not imply competition, and proportionality does not establish functional redundancy.

Output prevalence, co-detection, pair variances, site decomposition, zero sensitivity and selected pair plots. Full pair matrices can be saved at genus level. For large ASV sets, stream computations in blocks and save the candidate edge list plus summaries rather than allocating every full matrix and bootstrap replicate simultaneously. Never silently keep only the top-30 chemistry hits to make computation cheaper.

12. **Exact graph construction and bootstrap stability.**

Graphs contain only within-domain edges. Represent each undirected edge once, with a deterministic ordering j < k; remove self-edges and duplicates. Coefficients remain unconstrained in sign. All affinity weights are nonnegative.

Phylogenetic graph:

- Compute patristic distances from the supplied domain tree. Reject negative or nonfinite branch lengths, duplicate tip IDs and malformed mappings. Preserve path lengths if pruning or contracting the tree. Features absent from a partially matched tree can remain as isolated phylogenetic nodes; export their IDs and coverage. A wholly missing tree makes phylogeny-enabled variants unavailable, not an excuse to fabricate edges.
- For each feature, choose its k=5 nearest mapped neighbors, or all available if fewer. Break ties by canonical feature ID. Use the union of directed neighbor choices to form an undirected edge set.
- Set tau_P to the median strictly positive distance among selected edges in that training partition; use 1 if all selected distances are zero. Set raw edge weight exp(-d/tau_P). Zero-distance pairs have weight 1.
- Divide edge weights by their sum over unique undirected edges. An empty graph remains empty.

Abundance graph:

- Rank eligible partners by v_star. Break ties by feature ID. Select a pair only when each taxon is among the other's k=5 lowest-variance eligible partners. This defines the full-training mutual-neighbor candidate set. It is a relative-neighborhood rule; there is no claim that every selected pair is biologically redundant.
- Bootstrap whole training sites B=500 times with replacement. Carry every observation of each selected site into a replicate; a site drawn twice is two bootstrap copies and must not be collapsed back to one. Each copy receives equal total weight. Keep the original training feature universe fixed within this graph-construction bootstrap; recompute co-detection eligibility and pair scores from each replicate. Count distinct source sites for the minimum co-detection-site criterion, so duplicate copies do not manufacture independent evidence.
- Reapply the mutual-neighbor rule in each replicate. A pair's support is the number of replicates in which it is eligible and selected divided by B. An ineligible replicate counts as not selected. Keep full-training candidate edges with support >= 0.80. Export support and eligibility frequency separately; this support is not a p-value.
- Set tau_E to the median strictly positive full-training v_star among retained edges, using 1 if all are zero. Set raw weight = support * exp(-v_star/tau_E), then normalize its unique-edge sum to one. If no edges survive, return an empty graph and report it; do not fill it using low-confidence edges silently.
- Genus graphs should use the exact full pair search. For large ASV inputs, an optional approximate mode may bootstrap within a full-training candidate pool of 20 nearest eligible partners per taxon. It must be explicitly enabled and labeled: support is then conditional on that pool, not the exact all-pair selection frequency. Log the difference, memory limits and runtime. Never silently change modes. Cache graph artifacts by training sample IDs, data hashes and preprocessing/graph settings.

For domain d, combine the two separately normalized graphs as W_d = alpha_d*W_P,d + (1-alpha_d)*W_E,d, merging their edge unions. Do not renormalize the mixed graph again: alpha must keep its stated meaning, including when one constituent graph is empty. Store each edge's phylogenetic, empirical and mixed weights. In the objective, include only positive mixed-weight edges. If both components are empty, the fusion penalty for that domain is zero.

Taxon relationships W are fixed within a coefficient-fitting run. Training-data abundance relationships, stability filtering and graph bandwidths are rebuilt within every inner-training and outer-training partition. A supplied sequence reference tree can remain fixed; outcome data never enter either graph. alpha is selected by validation rather than read off the Mantel r values.

13. **Estimator, penalties and numerical implementation.**

Implement a constrained estimator using CVXPY or another explicitly verified convex solver. The required standardized-target objective is exactly:

\[
\sum_i w_i\left(y_i^*-b_0-\ell_i^{A\top}\beta_A-\ell_i^{B\top}\beta_B\right)^2
+\lambda_S(\|\beta_A\|_1+\|\beta_B\|_1)
+\lambda_R(\|\beta_A\|_2^2+\|\beta_B\|_2^2)
+\lambda_F\sum_{d\in\{A,B\}}\sum_{(j,k)\in E_d}W^{(d)}_{jk}|\beta_{d,j}-\beta_{d,k}|,
\qquad \mathbf1^\top\beta_A=\mathbf1^\top\beta_B=0.
\]

Do not introduce an undocumented factor of n, 1/2 or 1/p into one term. The intercept is unpenalized. Use a sparse edge-incidence matrix D_d with +1 and -1 in each edge row; the fusion term is lambda_F*sum(edge_weight*abs(D_d @ beta_d)). Sum each undirected edge once. A squared Laplacian penalty is not an equivalent replacement: it smooths coefficients without the same exact-fusion behavior.

Beta and the intercept are optimized during fitting. Lambda_S, lambda_R and lambda_F are nonnegative hyperparameters fixed during that fit. Select them through inner cross-validation, not through unconstrained minimization of the same training objective, which would favor zero penalties. W's components are fixed; alpha determines their validation-selected mixture. Keep lambda_R strictly positive in the default search for numerical stabilization. Lambda_S and lambda_F may be zero.

Use installed supported solvers, preferring CLARABEL and then OSQP where suitable; expose solver choice and its supported tolerances in config. Check solver status, finite coefficients, objective components and the two equality residuals. Aim for an absolute equality residual <= 1e-6 in standardized-target coefficients. A failed or inaccurate solve must be logged and retried with a documented solver/tolerance policy; reject candidates that still fail. Never substitute an unconstrained fit or convert failure to a favorable score. Record solver versions, tolerances, status, iterations and runtime. Warm starts and graph reuse are allowed within the same training partition.

This implementation is motivated by convex modeling capabilities documented in [CVXPY constraints](https://www.cvxpy.org/tutorial/constraints/index.html) and [solver documentation](https://www.cvxpy.org/tutorial/solvers/index.html). Check APIs against the environment and record installed dependency versions rather than claiming untested version compatibility.

14. **Hyperparameter selection, baselines and evaluation.**

Default outer validation: six site-disjoint folds. Default inner validation: four site-disjoint folds within each outer training set. Assign sites deterministically using a saved random seed and balance sample counts without inspecting target values; save explicit split manifests. All repeated observations of a site stay together. If there are too few sites after exclusions, fail with an explanation or use an explicitly configured smaller fold count. Do not silently switch to row-wise CV.

Suggested candidate pools after target standardization:

| Parameter | Candidate values |
| --- | --- |
| lambda_S | 0, 0.001, 0.01, 0.1 |
| lambda_R | 0.001, 0.01, 0.1, 1 |
| lambda_F | 0, 0.1, 1, 10, 100, 1000 |
| alpha_ARC, alpha_BAC | 0, 0.5, 1, subject to model-family restrictions |

These are starting ranges. Report winners on the search boundary; expand ranges only using a prespecified rule based on inner results or through a clearly labeled new analysis, never to improve an already-inspected outer-test result. Generate at most 48 deterministic candidate configurations per model family by default. Include no-fusion candidates, cover every allowed alpha combination for fusion-enabled families, and sample remaining combinations without replacement using the saved seed. Deduplicate alpha settings when lambda_F=0. Allow exhaustive search as an explicit option. Cache preprocessing and graph components once per partition, not once per lambda combination.

Required model comparisons, with the same outer splits and preprocessing conventions wherever applicable:

| Family | Definition |
| --- | --- |
| Intercept only | Training-site-weighted mean target |
| Genus ridge and genus elastic net | Genus sums followed by separate-domain log contrasts; lambda_F=0; ridge fixes lambda_S=0 |
| ASV ridge and ASV elastic net | ASV inputs; same respective penalty restrictions |
| Phylogeny only | alpha_ARC=alpha_BAC=1 |
| Abundance only | alpha_ARC=alpha_BAC=0 |
| Shared hybrid | alpha_ARC=alpha_BAC, selected among 0, 0.5, 1 |
| Domain-adaptive hybrid | alpha_ARC and alpha_BAC selected independently |

The domain-adaptive hybrid is the prespecified primary model. Missing input-dependent baselines are reported unavailable, not fabricated. Keep computational budgets and candidate definitions explicit; do not claim the comparison isolates one mechanism if other preprocessing or search choices differ.

For each inner candidate, concatenate predictions for all inner-held-out sites in original target units. Compute one MSE per site and average those site MSEs. Use this site-balanced MSE for selection. The default one-standard-error heuristic permits candidates within SE_best = sd(site_MSE_best)/sqrt(number_of_inner_validation_sites) of the lowest mean. Among permitted candidates choose the one with the smallest mean number of active fused groups in its inner-training fits, then fewer active taxa, then lower inner error, then deterministic configuration order. This SE is a model-selection heuristic; shared training sets mean it is not a formal independent-site confidence interval. Report the exact minimum-error candidate as well.

Refit all preprocessing, both graphs and the selected estimator on the outer training set, then predict the untouched outer sites once. Report pooled OOF RMSE, MAE and R² in original units, site-balanced MSE/RMSE and MAE, coverage, and per-site errors. Do not average per-site R² with only one to four observations. If comparing models with unscorable rows, give coverage for each and clearly label any common-row sensitivity comparison; do not silently compare different subsets.

Bootstrap sites for paired differences in stored OOF errors, using the same sampled sites for both models. Label intervals conditional on the fitted OOF models; they omit part of retraining uncertainty. Repeated folds are not independent samples for an ordinary t-test. Report all prespecified models rather than treating the post hoc best outer-CV result as an unbiased estimate of a newly selected winner.

The final deployable model is selected through a fresh grouped inner-style search on all eligible data and then refitted on all of it. Its training predictions are not an estimate of generalization. Store final model artifacts separately from outer-fold models.

Exploratory all-data diagnostics are allowed but must be labeled descriptive. They must never supply feature lists, graphs or grouping to nested validation. The full-data exploratory findings already influenced the scientific design, so a new external facility cohort remains the strongest confirmation. The general rationale for grouped validation is described in [scikit-learn cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html).

15. **What compression means and how b12 is handled.**

Step 6 is an algebraic explanation of the fitted beta values. No separate neural representation learner, reconstruction loss, or mandatory second residual model is required. For two taxa, a shared log-geometric-mean term has coefficient beta_1+beta_2 and their log-ratio contrast has coefficient (beta_1-beta_2)/2. Fusion removes that contrast only when their coefficients become equal. Useful differences can remain as unequal weights.

Extract groups separately for each domain. Numerical grouping requires a configurable absolute tolerance, default 1e-6 in standardized-target coefficient units. Connect positive-W edges whose fitted coefficient differences fall within tolerance. Candidate connected components must also have total coefficient range <= tolerance; otherwise label them ambiguous and conservatively retain singleton features rather than chaining many small differences into one large difference. With lambda_F=0, do not claim graph-induced fusion: report singleton features, apart from zero-coefficient counts. Use the same tolerance to report active coefficients, while preserving the actual fitted values for prediction.

For a group G, export member IDs, beta values, coefficient range, group mean beta_G, group summary m_iG = mean_j_in_G(log q_ij), and group coefficient theta_G = |G|*beta_G. Then sum_G theta_G = 0 separately within each domain, provided all taxa are represented. Save within-group coefficient residuals beta_j-beta_G. Group summaries plus these residual terms reconstruct the fitted predictor exactly up to numerical precision. If residuals are dropped for a compact predictor, report the resulting maximum prediction difference and only claim equivalence within the specified tolerance. Do not snap coefficients or drop near-zero features silently in the serialized primary model.

Report both active taxon count and active group count per domain, plus residual sizes. These are representation-size summaries, not formal statistical degrees of freedom. When K nonzero group features have a domain-specific zero-sum group-coefficient constraint, their allowable coefficient dimension is at most K-1; do not equate that with the one scalar prediction score. Do not claim dimension reduction simply because any regression outputs one number.

The optional b12 comparator explicitly computes k*(mean(log q_G1)-mean(log q_G2)), k=sqrt(r*s/(r+s)), and fits its coefficient gamma. Expanding gamma*b12 gives coefficient gamma*k/r for each G1 taxon and -gamma*k/s for each G2 taxon. Thus r=13 and s=8 do not imply equal-magnitude positive and negative taxon coefficients. The geometric-mean ratio compares group means, not summed community mass. Zero handling is the same as for the primary model.

The b12 feature is not appended to all its source log features in the primary model. A separate optional balance-only or nonredundant balance-basis model may use it. Full-dataset 13/8 memberships are exploratory; do not benchmark their fixed feature as if its discovery were independent of the outer tests. A confirmatory comparator needs externally specified groups or a fully specified group-discovery rule repeated within training folds. Disable this comparator by default until those inputs/rules exist. Never invent members or force every fold into two groups.

Show compression versus prediction error using inner validation paths, or a prespecified outer evaluation of fixed settings that is clearly labeled. Use outer errors to evaluate, not tune, the compression choice. A useful result is fewer retained features at comparable or lower held-out error. Neither coefficient fusion nor low ratio variance proves biological functional redundancy.

16. **Configuration and computational behavior.**

Provide a YAML config with at least the following keys. Input paths below are explicit placeholders, not claims that files exist. CLI validation must identify them before any real-data run.

```yaml
seed: 20260917
task: regression
resolution: asv
inputs:
  arc_counts: null
  bac_counts: null
  metadata: null
  arc_tree: null
  bac_tree: null
  arc_taxonomy: null
  bac_taxonomy: null
  orientation: samples_by_features
columns:
  sample_id: sample_id
  site_id: site_id
  target: null
  covariates: []
preprocessing:
  detection_count: 5
  min_detected_samples: 5
  min_sample_prevalence: 0.10
  min_detected_sites: 3
  min_site_prevalence: 0.10
  pseudocount: 0.5
  sensitivity_pseudocounts: [1.0]
  standardize_target: true
  standardize_microbial_columns: false
graph:
  k_phylogeny: 5
  k_abundance: 5
  min_joint_samples: 10
  min_joint_sites: 5
  empirical_score: max_all_and_joint_lrv
  bootstrap_sites: 500
  support_threshold: 0.80
  approximate_bootstrap: false
  approximate_candidate_k: 20
cv:
  outer_folds: 6
  inner_folds: 4
  selection_metric: site_balanced_mse
  selection_rule: one_standard_error
  required_validation_coverage: 1.0
search:
  candidate_budget_per_family: 48
  lambda_s: [0, 0.001, 0.01, 0.1]
  lambda_r: [0.001, 0.01, 0.1, 1]
  lambda_f: [0, 0.1, 1, 10, 100, 1000]
  alpha_arc: [0, 0.5, 1]
  alpha_bac: [0, 0.5, 1]
solver:
  preferred: CLARABEL
  fallback: OSQP
  equality_tolerance: 1.0e-6
compression:
  coefficient_tolerance: 1.0e-6
  keep_residual_coefficients: true
optional:
  b12_comparator: false
  cross_domain_interactions: false
output_dir: results/ad_microbiome
```

Add explicit resource limits, solver-specific options, model-family selection and run identifiers as needed. Default smoke mode may reduce bootstrap repetitions to 20, use three outer/two inner folds, limit candidates and use a small clearly labeled synthetic dataset. It must not overwrite research outputs or be labeled full scientific validation. Print estimated fit counts and graph workload before a research run. Make long runs resumable using versioned fold artifacts and cache keys that include all relevant settings and input hashes. Parallelize independent numerical work only as supported by the code environment, without mixing train/test caches or nondeterministic random states.

17. **Required scripts, artifacts and explanation.**

Deliver one usable entry point, for example python run_model.py --config config.yaml --mode train, backed by small modules for loading, preprocessing, graph construction, the estimator, grouped evaluation and reporting. Support --mode validate, diagnose, train and predict; predict loads a saved training transformation and model and must never refit them. A README must explain installation, expected input formats, how to select the target, real versus synthetic runs, expected runtime drivers and how to inspect results.

Save at least:

| Artifact | Minimum contents |
| --- | --- |
| Resolved config and run manifest | Input hashes, dependency versions, seed, timestamps, mode, target units, solver settings |
| Cohort/feature audit | Included/excluded IDs, reasons, retained counts and mass, pseudocount mass, tree coverage |
| Fold manifests | Training/validation/test sample and site IDs for every stage |
| Graph edge tables | Domain, endpoints, distances/variances, co-detection, bootstrap support, each component weight, mixed weight |
| Search results | Candidate settings, solver status, inner site errors, coverage, feature counts, selected/minimum-error candidates |
| OOF predictions | sample_id, site_id, target, prediction, outer fold, model family, scorable flag/reason |
| Evaluation | Sample-weighted and site-balanced metrics, per-site errors, paired comparisons with their uncertainty limitations |
| Coefficients and groups | Taxon labels, original/standardized target coefficient units, groups, residual contrasts, constraints and tolerances |
| Saved model | Ordered feature IDs, preprocessing, target scaling, beta/intercept, graphs, group map and version metadata |
| Figures | OOF predicted-versus-observed, residuals by site, compression/error plot, annotated genus diagnostics, graph/group summaries |

Use machine-readable CSV/JSON for results and a concise Markdown run report with links to figures. Save exact prediction artifacts needed to reproduce plots. Synthetic figures and values must be marked synthetic. Do not describe attention weights, causal syntrophy, recovered functions, or proven forecasting when the fitted model does not establish them.

18. **Scientific verification and acceptance criteria.**

The generated code must include meaningful checks of these concrete risks:

1. Separate constraints: both beta sums meet tolerance; a vector satisfying only the combined sum is not accepted as satisfying both.
2. Mathematical scale invariance: after zero handling, multiplying either domain's positive vector by a sample-specific positive factor leaves its log-contrast score unchanged. Do not incorrectly require the same result after adding a fixed count pseudocount afresh to arbitrarily rescaled raw counts.
3. Fold isolation: no site crosses training/held-out boundaries. Perturb held-out counts and labels and verify that training feature selection, graph weights, target scaling and fitted coefficients remain unchanged for a fixed training partition.
4. Graph bookkeeping: no cross-domain edges, duplicate undirected edges or self-edges; edge weights are finite/nonnegative, component sums obey the declared normalization, and an empty graph produces a zero fusion term.
5. Zero and missing-data behavior: co-absence does not create eligible empirical edges; missing assays and zero retained-count rows are reported; true zeros and unspecified missing feature columns are handled differently.
6. Group algebra: on a small known coefficient vector, shared group terms plus residual terms reproduce the original predictor; verify the r-versus-s scaling of a balance with unequal group sizes. Test against coefficient chains that would exceed grouping tolerance.
7. Solver behavior: verify objective components and equality residuals on small synthetic problems; lambda_F=0 agrees with the corresponding constrained no-fusion formulation. A symmetric toy example can demonstrate fusion, but do not require perfect recovery of true groups or higher accuracy in every random simulation.
8. Prediction replay: saved and reloaded models give matching predictions despite input row/column reordering when IDs are preserved; new samples do not alter the trained feature mapping.
9. Weighted decomposition: verify that V_between + V_within equals the declared equal-site population variance for a toy repeated-site dataset.
10. End-to-end demonstration: generate positive compositions for two separate domains, introduce known zero-sum coefficients and repeated-site variation, convert to separate simulated count assays, and execute a small nested grouped run. State explicitly that this tests implementation, not the scientific hypothesis in real AD data.

Acceptance requires runnable required commands, successful core checks, traceable data exclusions, complete OOF predictions for the declared evaluation cohort or a clearly failed coverage gate, and no hidden substitution of the statistical model. Tests of biological superiority await the real dataset. The delivered code should make uncertainty and implementation choices inspectable so that subsequent research decisions can be made from actual results.

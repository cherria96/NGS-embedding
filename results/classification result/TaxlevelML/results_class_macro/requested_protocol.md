# Multi-Label Classification Analysis for Anaerobic Digestion Warning Prediction — Class Level

Perform a **new, independent multi-label classification analysis** to predict five anaerobic digestion warning indicators simultaneously from microbiome/genomic data.

**IMPORTANT:** This is a new analysis at the **Class taxonomic level**. Do NOT reuse, modify, or inherit the previous Family-level, Genus-level, or Order-level feature matrices, feature lists, preprocessing results, model results, hyperparameters, thresholds, or conclusions. Reconstruct the microbiome features and perform the entire analysis specifically for the Class-level dataset.

---

# 1. Objective

Predict the following five warning indicators simultaneously:

* `warning_acid_base_balance`
* `warning_buffer_capacity`
* `warning_acid_accumulation`
* `warning_ammonia_toxicity`
* `warning_biogas_quality`

This is a **multi-label classification problem**, not a multiclass classification problem.

Each sample can have any combination of the five warning labels.

The final analysis must evaluate how well the models predict **all five warning indicators as a multi-label outcome**, using:

1. **Macro-average MCC**
2. **Macro-average AUPR**
3. **Macro-average ROC-AUC**

as the primary evaluation metrics.

---

# 2. Input Data

Use:

* `metadata.csv`
* `Dat_ARC.xlsx`
* `Dat_BAC.xlsx`

First inspect all files carefully and determine:

* sample identifiers
* site identifiers
* seasonal information
* taxonomic information
* abundance values
* warning labels
* possible duplicate samples
* missing values
* variables that may cause target leakage

Do not assume column names without inspecting the actual files.

---

# 3. Predictor Data: GENOMIC DATA ONLY

Use **genomic/microbiome data only** as predictor variables.

Do NOT use:

* physicochemical variables
* substrate variables
* operational variables
* process-performance variables
* warning-definition variables
* any metadata variables that could leak the targets

Examples of leakage-prone variables include:

* `eff_pH`
* `eff_TVFAs`
* `eff_ALK`
* `eff_HPro`
* `eff_HAc`
* `eff_TAN`
* `eff_CH4`

However, these are only examples. Inspect the complete metadata and exclude **any variable that directly or indirectly defines or reveals the warning labels**.

`Site ID`, `Sample ID`, and `Season` may be retained temporarily for integration, grouping, reproducibility, and tracking, but they must **never be used as model predictors**.

The final predictor matrix must contain **microbiome/genomic features only**.

---

# 4. Taxonomic Feature Definition — CLASS LEVEL

Use the following taxonomic resolution:

### Bacteria

**Class level**, using:

> relative abundance > 0.1%

### Archaea

**Class level**, using:

> relative abundance > 0.1%

This is a **Class-level analysis**.

Do NOT use:

* Phylum-level features
* Order-level features
* Family-level features
* Genus-level features
* Species-level features

Do NOT use a global mean-abundance or median-abundance threshold.

Do NOT select:

* Top 10 taxa
* Top 16 taxa
* Top N taxa
* any other fixed number of taxa

The only taxonomic predictors allowed are:

* bacterial Class features
* archaeal Class features

constructed according to the sample-specific `>0.1%` rule below.

---

# 5. Sample-Specific >0.1% Feature Construction

Apply the `>0.1%` abundance threshold **separately for every sample**.

For every sample:

1. Identify all bacterial **Class** taxa with relative abundance > 0.1%.
2. Identify all archaeal **Class** taxa with relative abundance > 0.1%.

Construct the final feature set as the **union of all bacterial and archaeal Class taxa that exceed 0.1% in at least one sample**.

After constructing this union feature list:

* if a Class has relative abundance > 0.1% in a given sample, use its actual relative abundance value;
* if a Class is present but has relative abundance ≤ 0.1%, represent it as `0`;
* if a Class is absent, represent it as `0`.

This produces a fixed microbiome feature matrix containing bacterial Class and archaeal Class columns across all samples.

The `>0.1%` filtering rule is a **predefined taxonomic feature-construction rule**.

It is NOT an additional machine-learning feature-selection procedure.

Do NOT perform additional feature selection unless explicitly requested.

Do NOT perform:

* global mean-abundance filtering
* global median-abundance filtering
* Top N selection
* variance filtering
* univariate feature selection
* recursive feature elimination
* LASSO-based feature selection
* tree-based feature selection
* test-set-based feature selection

---

# 6. Data Integration

Integrate:

* `metadata.csv`
* `Dat_ARC.xlsx`
* `Dat_BAC.xlsx`

using the appropriate sample/site identifiers after inspecting the actual structure.

Verify:

* sample matching
* site matching
* duplicate records
* missing microbiome data
* missing warning labels
* inconsistent identifiers
* number of unique sites
* number of samples per site
* seasonal coverage

The expected design is:

* 34 unique sites
* 4 seasonal samples per site:

  * Spring
  * Summer
  * Fall
  * Winter

Do NOT assume these numbers are correct.

Verify them from the actual data and report the verified numbers.

---

# 7. Site-Level Grouping and Leakage Prevention

`Site ID` is the grouping variable.

Samples from the same site are correlated.

Therefore:

* all samples from one site must belong entirely to either training or testing;
* no site may appear in both train and test;
* during cross-validation, all samples from the same site must remain in the same fold.

This site-level grouping requirement is mandatory.

---

# 8. Train/Test Split

Create an approximately **80:20 train/test split at the site level**.

Use a fixed random seed.

The split should preserve the distribution of the five warning labels as reasonably as possible while strictly maintaining site separation.

Report:

* number of training sites
* number of test sites
* number of training samples
* number of test samples
* site IDs assigned to training
* site IDs assigned to testing
* prevalence/distribution of each warning label in training and testing

Do not use test-set information for:

* feature construction
* preprocessing
* hyperparameter tuning
* resampling
* threshold optimization
* model selection

---

# 9. Cross-Validation

Perform **5-fold site-grouped cross-validation using the training set only**.

The same site must never occur in multiple folds.

Prefer a group-stratified strategy when necessary to preserve the distribution of the five warning labels while maintaining site-level grouping.

Report:

* fold assignments
* sites in each fold
* sample counts in each fold
* warning-label distributions in each fold

All preprocessing, resampling, threshold selection, and model fitting must occur inside the appropriate training portion of each fold.

---

# 10. Preprocessing

Use model-appropriate preprocessing inside the cross-validation pipeline.

Examples:

* scaling/standardization where required
* missing-value imputation where required
* model-specific preprocessing

All preprocessing parameters must be learned only from the training portion of each fold.

For final training:

* fit preprocessing using the complete training set only;
* apply the resulting preprocessing to the untouched test set.

Never fit preprocessing using the entire dataset before splitting.

---

# 11. Machine Learning Models

Evaluate the following six models:

1. Random Forest
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Neural Network (NNET)
5. XGBoost
6. GLMNET

Use an appropriate multi-label strategy.

The primary strategy must be:

> **Binary Relevance / One-vs-Rest**

Model the five warning labels as five separate binary prediction tasks within the overall multi-label framework.

A classifier-chain strategy may optionally be explored if computationally feasible, but the primary analysis must remain Binary Relevance / One-vs-Rest.

Do NOT convert the problem into multiclass classification.

---

# 12. Class Imbalance

First inspect the prevalence of each warning label.

Because warning labels may be imbalanced:

* use class weighting where supported;
* consider balanced sampling where appropriate;
* use oversampling/undersampling only inside the training portion of each CV fold;
* never resample validation or test data.

For any balancing method, document:

* method
* parameters
* where it was applied
* random seed

---

# 13. Hyperparameter Tuning

Perform hyperparameter tuning using **5-fold site-grouped CV on the training data only**.

For each model:

* define an appropriate hyperparameter search space;
* use a fixed random seed;
* record the search space;
* record the number of combinations/iterations evaluated;
* identify the best hyperparameters;
* record CV performance.

Never use the test set for hyperparameter tuning.

---

# 14. PRIMARY EVALUATION METRICS

The three primary metrics are:

1. **Macro-average MCC**
2. **Macro-average AUPR**
3. **Macro-average ROC-AUC**

Do NOT use:

* Micro-AUPR as the primary metric
* Micro-ROC-AUC as the primary metric
* flattened multi-label MCC as the primary metric

The Macro metrics must be calculated as the arithmetic mean of the five warning-specific metrics.

---

# 15. Macro-average MCC

For each warning label:

1. obtain the binary true labels;
2. obtain binary predictions using the final predefined threshold;
3. calculate binary MCC.

Calculate:

* `MCC_warning_acid_base_balance`
* `MCC_warning_buffer_capacity`
* `MCC_warning_acid_accumulation`
* `MCC_warning_ammonia_toxicity`
* `MCC_warning_biogas_quality`

Then calculate:

$$
Macro\text{-}MCC =
\frac{MCC_1+MCC_2+MCC_3+MCC_4+MCC_5}{5}
$$

Each warning contributes equally.

Do NOT flatten all five labels into one vector.

---

# 16. Macro-average AUPR

For each warning label:

1. obtain binary true labels;
2. obtain predicted positive-class probabilities or decision scores;
3. calculate the Precision-Recall curve;
4. calculate AUPR.

Then:

$$
Macro\text{-}AUPR =
\frac{AUPR_1+AUPR_2+AUPR_3+AUPR_4+AUPR_5}{5}
$$

Each warning contributes equally.

Do NOT calculate Micro-AUPR as the primary metric.

---

# 17. Macro-average ROC-AUC

For each warning label:

1. obtain binary true labels;
2. obtain predicted positive-class probabilities or decision scores;
3. calculate ROC curve;
4. calculate ROC-AUC.

Then:

$$
Macro\text{-}ROC\text{-}AUC =
\frac{ROC\text{-}AUC_1+ROC\text{-}AUC_2+ROC\text{-}AUC_3+ROC\text{-}AUC_4+ROC\text{-}AUC_5}{5}
$$

Each warning contributes equally.

Do NOT calculate Micro-ROC-AUC as the primary metric.

---

# 18. Threshold Optimization

Threshold optimization is required for metrics based on binary predictions, especially MCC.

For each warning label:

* determine the threshold using training/CV data only;
* thresholds may be optimized separately for each warning;
* do not optimize thresholds using the test set;
* once selected, fix the thresholds before final test evaluation.

Use an appropriate documented procedure, such as:

> Select the threshold that maximizes MCC using out-of-fold predictions from the training/CV process.

Report the final threshold for each warning.

AUPR and ROC-AUC must be calculated from continuous predicted probabilities/decision scores and therefore must not depend on the final classification threshold.

---

# 19. Model Selection

Compare the six models using ONLY:

* Macro MCC
* Macro AUPR
* Macro ROC-AUC

Do not select models based on:

* accuracy
* macro-F1
* micro-F1
* balanced accuracy
* precision alone
* recall alone
* single-warning performance
* flattened MCC
* Micro-AUPR
* Micro-ROC-AUC

If different models perform best on different Macro metrics, explicitly report the trade-off.

---

# 20. Per-Warning Metrics

Retain the five individual warning-level metrics for verification.

For every model, retain:

### MCC

* Acid-base balance
* Buffer capacity
* Acid accumulation
* Ammonia toxicity
* Biogas quality

### AUPR

* Acid-base balance
* Buffer capacity
* Acid accumulation
* Ammonia toxicity
* Biogas quality

### ROC-AUC

* Acid-base balance
* Buffer capacity
* Acid accumulation
* Ammonia toxicity
* Biogas quality

Verify:

> Macro metric = arithmetic mean of the five corresponding warning-specific metrics.

---

# 21. Final Test Evaluation

After hyperparameter tuning:

1. retrain each candidate model using the complete training set;
2. apply the selected preprocessing pipeline;
3. apply the selected class-imbalance strategy using training data only;
4. apply the fixed warning-specific thresholds;
5. evaluate once on the untouched test set.

Do not perform additional tuning using the test set.

Calculate:

* Macro MCC
* Macro AUPR
* Macro ROC-AUC

for every model.

The final test set must remain untouched until this final evaluation stage.

---

# 22. Final Comparison Table

Create exactly:

| Model         | Macro MCC | Macro AUPR | Macro ROC-AUC |
| ------------- | --------: | ---------: | ------------: |
| Random Forest |       ... |        ... |           ... |
| KNN           |       ... |        ... |           ... |
| SVM           |       ... |        ... |           ... |
| NNET          |       ... |        ... |           ... |
| XGBoost       |       ... |        ... |           ... |
| GLMNET        |       ... |        ... |           ... |

Use the final untouched test-set results.

Clearly identify the best model for each metric.

---

# 23. Visualizations

Create:

### Figure 1

Bar chart comparing six models using:

**Macro MCC**

### Figure 2

Bar chart comparing six models using:

**Macro AUPR**

### Figure 3

Bar chart comparing six models using:

**Macro ROC-AUC**

Each figure must clearly display:

* model names
* metric values

Optional supplementary plots may include:

* five-warning Precision-Recall curves
* five-warning ROC curves

However, the three Macro metric bar charts must remain the primary model-comparison figures.

---

# 24. Feature Contribution / Interpretation

Examine microbiome feature contribution using appropriate methods, such as:

* Random Forest feature importance
* XGBoost feature importance
* permutation importance
* SHAP
* another appropriate model-specific or model-agnostic interpretation method

Interpret feature importance as **predictive association**, not causation.

Do NOT perform additional machine-learning feature selection.

Report which:

* bacterial **Class**
* archaeal **Class**

features contribute most strongly to prediction where reliable interpretation is possible.

Clearly identify:

* taxon name
* domain: Bacteria or Archaea
* taxonomic level: Class

---

# 25. Leakage Audit

Perform an explicit leakage audit before model fitting.

Check whether any variable:

* directly defines a warning label;
* is derived from a target;
* is measured after the warning state;
* is an effluent variable used to define a warning;
* is a transformed or duplicated target representation;
* contains Site ID or other grouping information capable of leaking outcomes.

Remove leakage-prone variables.

Confirm that the final predictor matrix contains **only bacterial Class and archaeal Class microbiome features**.

---

# 26. Required Outputs

Save/export:

1. Cleaned merged dataset
2. Final microbiome-only predictor matrix
3. Selected bacterial **Class** taxa satisfying sample-specific >0.1% rule
4. Selected archaeal **Class** taxa satisfying sample-specific >0.1% rule
5. Final microbiome feature list
6. Taxonomic domain for every feature:

   * Bacteria
   * Archaea
7. Taxonomic level:

   * Class
8. Number of samples in which each taxon exceeded 0.1%
9. Site-level train/test assignment
10. Site-grouped CV fold assignment
11. Preprocessing pipeline
12. Missing-data handling procedure
13. Class-imbalance strategy
14. Hyperparameter search spaces
15. Best hyperparameters
16. Cross-validation results
17. Threshold-selection method
18. Final threshold for each warning label
19. Final test-set predicted probabilities
20. Final test-set binary predictions
21. Per-warning MCC values
22. Per-warning AUPR values
23. Per-warning ROC-AUC values
24. Macro MCC
25. Macro AUPR
26. Macro ROC-AUC
27. Final model comparison table
28. Three final comparison figures
29. Final trained model objects
30. Reproducibility information

---

# 27. Reproducibility

Use and document a fixed random seed.

Save/report:

* random seed
* train/test site assignments
* CV fold assignments
* final microbiome feature list
* sample-specific `>0.1%` Class feature-construction rule
* preprocessing steps
* missing-value handling
* class-imbalance procedure
* hyperparameter search spaces
* best hyperparameters
* threshold optimization method
* final thresholds
* exact definitions of Macro MCC, Macro AUPR, and Macro ROC-AUC
* final model results

The analysis must be fully reproducible from the provided files and documented code.

---

# 28. Final Interpretation

The final interpretation must answer:

> **How well can microbiome composition alone, represented at the Class level, predict the five anaerobic digestion warning indicators simultaneously?**

Discuss:

* which model performs best overall;
* how the six models compare using Macro MCC, Macro AUPR, and Macro ROC-AUC;
* whether performance is consistent across the five warning labels;
* whether some warning labels are substantially more difficult to predict;
* which bacterial and archaeal Class features contribute most strongly to prediction.

Because Macro averaging gives equal weight to each warning, explicitly discuss whether prediction performance is consistent across all five warning types.

Do not make causal claims from feature importance.

---

# 29. Critical Requirements

The final analysis MUST satisfy all of the following:

* **Multi-label classification**
* Predict all five warning labels simultaneously within one overall multi-label analysis
* **Genomic/microbiome data only**
* No physicochemical/process/operational predictor variables
* Bacteria at **Class level**
* Archaea at **Class level**
* **Relative abundance > 0.1%**
* Threshold applied **sample by sample**
* Feature set constructed from the union of taxa exceeding 0.1% in at least one sample
* No Top 10/16/N selection
* No additional ML feature selection
* Site-level train/test split
* Approximately 80:20 train/test split
* No site leakage
* 5-fold site-grouped CV on training only
* Model-dependent preprocessing inside CV
* Class-imbalance handling only within training folds
* Hyperparameter tuning on training data only
* Threshold optimization on training/CV data only
* Final test set used exactly once for final evaluation
* **Macro-average MCC**
* **Macro-average AUPR**
* **Macro-average ROC-AUC**
* Macro metrics calculated as arithmetic means of the five warning-specific metrics
* Do NOT use Micro-AUPR or Micro-ROC-AUC as primary metrics
* Do NOT use flattened multi-label MCC as the primary MCC metric
* Final model comparison based only on the three Macro metrics
* Preserve all five warning labels as separate binary labels
* Do NOT convert the problem into multiclass classification
* Explicit leakage audit
* Full reproducibility

Before completing the analysis, verify that every requirement above has been implemented exactly.

---

# 30. IMPORTANT: TAXONOMIC LEVEL OVERRIDE

This analysis is specifically a **Class-level analysis**.

The following specification overrides any previous Family-level, Genus-level, Order-level, or Phylum-level configuration:

**Bacteria: Class levels (relative abundance > 0.1%)**

**Archaea: Class levels (relative abundance > 0.1%)**

Do NOT use:

* Family-level features
* Genus-level features
* Order-level features
* Phylum-level features
* Species-level features

The only allowed microbiome predictors are **bacterial Class and archaeal Class features** constructed using the sample-specific `>0.1%` relative-abundance rule.

Run this as a **new independent Class-level analysis** rather than modifying the previous Family-level, Genus-level, or Order-level analysis.

# Multi-Label Classification Analysis for Anaerobic Digestion Warning Prediction

## 1. Objective

Perform a **multi-label classification analysis** to predict five anaerobic digestion warning indicators simultaneously from microbiome/genomic data.

The five target labels are:

* `warning_acid_base_balance`
* `warning_buffer_capacity`
* `warning_acid_accumulation`
* `warning_ammonia_toxicity`
* `warning_biogas_quality`

This is a **multi-label classification problem**, not a multiclass classification problem.

Each sample can have any combination of the five warning labels.

The final analysis must evaluate how well the models predict **all five warning indicators as a multi-label outcome**, while reporting **Macro-average MCC, Macro-average AUPR, and Macro-average ROC-AUC** as the primary evaluation metrics.

---

# 2. Input Data

Use the following files:

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
* any variables that may cause target leakage

Do not assume column names without checking the actual files.

---

# 3. Predictor Data: GENOMIC DATA ONLY

Use **genomic/microbiome data only** as predictor variables.

Do NOT use physicochemical, substrate, operational, process-performance, or other metadata variables as predictors.

In particular, exclude variables such as:

* pH
* alkalinity
* TVFAs
* volatile fatty acids
* TAN
* methane/biogas measurements
* effluent physicochemical measurements
* substrate/process operating conditions
* any other variables that directly or indirectly define the warning labels

Examples of leakage-prone variables include:

* `eff_pH`
* `eff_TVFAs`
* `eff_ALK`
* `eff_HPro`
* `eff_HAc`
* `eff_TAN`
* `eff_CH4`

However, do NOT limit the leakage check to these examples. Inspect the complete metadata and exclude any predictor that could leak information from the target definition.

`Site ID`, `Sample ID`, and `Season` may be retained temporarily for integration, grouping, reproducibility, and tracking, but they must **never be used as model predictor variables**.

The final predictor matrix must contain **microbiome/genomic features only**.

---

# 4. Taxonomic Feature Definition

Use the following taxonomic resolution:

### Bacteria

**Phylum level**, using the rule:

> relative abundance > 0.1%

### Archaea

**Phylum level**, using the rule:

> relative abundance > 0.1%

Do NOT use Class, Order, Family, Genus, or Species level features in this analysis.

Do NOT use a global abundance threshold based on the mean or median abundance across all samples.

Do NOT select the top 10, top 16, top N, or any other fixed number of taxa.

The final microbiome features must therefore consist of:

* bacterial Phylum features
* archaeal Phylum features

using the sample-specific `> 0.1%` relative-abundance criterion described below.

---

# 5. Sample-Specific >0.1% Feature Construction

Apply the `> 0.1%` abundance threshold **separately for each sample**.

For every sample:

1. Identify all bacterial **Phylum** taxa with relative abundance > 0.1%.
2. Identify all archaeal **Phylum** taxa with relative abundance > 0.1%.

Construct the final feature set as the **union of all bacterial and archaeal Phylum taxa that satisfy the >0.1% criterion in at least one sample**.

After the union feature list is constructed:

* if a Phylum has relative abundance > 0.1% in a given sample, use its actual relative abundance value;
* if a Phylum is present but has relative abundance ≤ 0.1%, represent it as `0`;
* if a Phylum is absent, represent it as `0`.

The resulting feature matrix must therefore contain a fixed set of bacterial Phylum and archaeal Phylum columns across all samples.

The >0.1% feature filtering rule is a predefined taxonomic feature-construction rule and is **not an additional machine-learning feature-selection procedure**.

Do not perform another feature-selection algorithm unless explicitly requested later.

Do not perform:

* global mean-abundance filtering
* global median-abundance filtering
* Top N taxa selection
* variance filtering
* univariate feature selection
* recursive feature elimination
* LASSO-based feature selection
* tree-based feature selection
* feature selection based on the test set

unless explicitly requested later.

---

# 6. Data Integration

Integrate `metadata.csv`, `Dat_ARC.xlsx`, and `Dat_BAC.xlsx` using the appropriate sample/site identifiers after inspecting the actual structure of the files.

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

* **34 unique sites**
* **4 seasonal samples per site**

  * Spring
  * Summer
  * Fall
  * Winter

Do not assume this structure is correct without verifying it from the data.

Report the verified numbers.

---

# 7. Grouping and Data Leakage Prevention

`Site ID` is the grouping variable.

Samples from the same site are correlated and must not be split between training and testing data.

Therefore:

* all samples from one site must belong entirely to either the training set or the test set;
* a site must never appear in both train and test;
* during cross-validation, all samples from the same site must remain in the same fold.

This is required to prevent site-level information leakage.

---

# 8. Train/Test Split

Create an approximately **80:20 train/test split at the site level**.

Use a fixed random seed for reproducibility.

The site-level split should preserve the five-label distribution as reasonably as possible while strictly maintaining site separation.

Report:

* number of training sites
* number of test sites
* number of training samples
* number of test samples
* site IDs assigned to train
* site IDs assigned to test
* prevalence/distribution of each of the five warning labels in train and test

Do not use any information from the test set when constructing features, preprocessing data, tuning hyperparameters, selecting thresholds, or selecting the final model.

---

# 9. Cross-Validation

Perform **5-fold site-grouped cross-validation on the training set only**.

The same site must never occur in multiple folds.

Prefer a group-stratified cross-validation strategy if necessary to preserve the distribution of the five warning labels while maintaining site-level grouping.

Clearly report:

* fold assignments
* sites in each fold
* sample counts in each fold
* warning-label distributions in each fold

All preprocessing, resampling, threshold selection, and model fitting must occur within the training portion of each fold.

---

# 10. Preprocessing

Use model-appropriate preprocessing inside the cross-validation pipeline.

Examples:

* scaling/standardization where required
* missing-value imputation where required
* other model-specific preprocessing

Any preprocessing parameters must be learned **only from the training portion of each fold** and then applied to the corresponding validation portion.

For the final model:

* fit preprocessing using the complete training set only;
* apply the fixed preprocessing to the untouched test set.

Do not fit preprocessing using the entire dataset before train/test splitting.

---

# 11. Machine Learning Models

Evaluate the following six models:

1. Random Forest
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Neural Network (NNET)
5. XGBoost
6. GLMNET

For the multi-label problem, use an appropriate multi-label strategy.

The baseline strategy should be:

> **Binary Relevance / One-vs-Rest**

This means that the five warning labels are modeled as five binary prediction tasks within the overall multi-label framework.

A classifier-chain strategy may also be explored if computationally feasible, but the primary analysis should remain clearly defined as a multi-label prediction task.

Do not convert the task into multiclass classification.

---

# 12. Class Imbalance

First inspect the prevalence of each warning label.

Because some warning labels may be highly imbalanced:

* use class weighting where supported;
* consider balanced sampling where appropriate;
* use oversampling/undersampling only inside the training portion of each cross-validation fold;
* never perform resampling using validation or test data.

For any oversampling or balancing strategy, document:

* method
* parameters
* where it was applied
* random seed

Never resample the validation or test set.

---

# 13. Hyperparameter Tuning

Perform hyperparameter tuning using **5-fold site-grouped cross-validation on the training set only**.

For each model:

* define an appropriate hyperparameter search space;
* use a fixed random seed;
* record the search space;
* record the number of combinations/iterations evaluated;
* identify the best hyperparameters;
* record cross-validation performance.

Hyperparameter tuning must not use the test set.

---

# 14. PRIMARY EVALUATION METRICS

The primary evaluation metrics must be:

1. **Macro-average MCC**
2. **Macro-average AUPR**
3. **Macro-average ROC-AUC**

Do NOT use Micro-AUPR or Micro-ROC-AUC as the primary metrics.

Do NOT use a single flattened multi-label MCC as the primary MCC metric.

The three primary metrics must be calculated using the following procedure.

---

# 15. Macro-average MCC

For each of the five warning labels:

1. obtain the binary true labels;
2. obtain the binary predictions using the predefined/final classification threshold;
3. calculate the binary Matthews Correlation Coefficient (MCC) separately for that warning label.

This produces:

* `MCC_warning_acid_base_balance`
* `MCC_warning_buffer_capacity`
* `MCC_warning_acid_accumulation`
* `MCC_warning_ammonia_toxicity`
* `MCC_warning_biogas_quality`

Then calculate:

$$
Macro\text{-}MCC =
\frac{
MCC_1 + MCC_2 + MCC_3 + MCC_4 + MCC_5
}{5}
$$

Each warning contributes equally to the Macro-average.

Do not flatten all five labels into one binary vector to calculate MCC.

The flattened multi-label MCC must NOT be reported as the primary MCC metric.

---

# 16. Macro-average AUPR

For each of the five warning labels:

1. obtain the true binary labels;
2. obtain the predicted probability/decision score for the positive class;
3. calculate the Precision-Recall curve;
4. calculate the Area Under the Precision-Recall curve (AUPR).

This produces one AUPR value for each warning.

Then calculate:

$$
Macro\text{-}AUPR =
\frac{
AUPR_1 + AUPR_2 + AUPR_3 + AUPR_4 + AUPR_5
}{5}
$$

Each warning contributes equally.

Do NOT flatten all sample-label pairs and calculate Micro-AUPR as the primary metric.

Micro-AUPR may be omitted entirely from the final comparison.

---

# 17. Macro-average ROC-AUC

For each of the five warning labels:

1. obtain the true binary labels;
2. obtain the predicted probability/decision score for the positive class;
3. calculate the ROC curve;
4. calculate ROC-AUC.

This produces one ROC-AUC value for each warning.

Then calculate:

$$
Macro\text{-}ROC\text{-}AUC =
\frac{
ROC\text{-}AUC_1 + ROC\text{-}AUC_2 + ROC\text{-}AUC_3 + ROC\text{-}AUC_4 + ROC\text{-}AUC_5
}{5}
$$

Each warning contributes equally.

Do NOT flatten all sample-label pairs and calculate Micro-ROC-AUC as the primary metric.

---

# 18. Threshold Optimization

Threshold optimization is required only for metrics that depend on binary class predictions, especially MCC.

For each warning label:

* determine the classification threshold using training/CV data only;
* do not determine thresholds using the test set;
* thresholds may be optimized separately for each warning label;
* once thresholds are selected, fix them before evaluating the test set.

A suitable threshold-selection procedure should be defined and documented.

For example:

> Select the threshold that maximizes MCC using out-of-fold training/CV predictions.

Do not optimize the thresholds on the final test set.

AUPR and ROC-AUC must be calculated from predicted probabilities/decision scores and therefore should not be dependent on the final classification threshold.

Report the final threshold for each warning label.

---

# 19. Model Selection

Compare the six models using ONLY:

* Macro-average MCC
* Macro-average AUPR
* Macro-average ROC-AUC

Do not select a model based on:

* accuracy
* macro-F1
* micro-F1
* balanced accuracy
* precision alone
* recall alone
* a single warning-specific metric
* a flattened multi-label MCC
* Micro-AUPR
* Micro-ROC-AUC

Do not arbitrarily choose a model using only one of the three primary metrics.

If different models perform best on different metrics, explicitly report the trade-off.

---

# 20. Per-Warning Metrics

Although the **primary model comparison must use Macro-average metrics**, also calculate and retain the five individual warning-level metric values internally so that the Macro averages can be verified.

For each model, retain:

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

Then verify that each reported Macro-average value is exactly the arithmetic mean of the five corresponding warning-level values.

The final model-comparison table should emphasize the Macro-average metrics rather than presenting five separate model-comparison tables.

---

# 21. Final Test Evaluation

After selecting the best hyperparameters for each model:

1. retrain each candidate model using the **complete training set**;
2. apply the selected preprocessing pipeline;
3. apply the selected class-imbalance strategy using training data only;
4. apply the fixed threshold for each warning label;
5. evaluate once on the untouched test set.

Do not perform any additional tuning using the test set.

For the final evaluation, calculate for each model:

* Macro-average MCC
* Macro-average AUPR
* Macro-average ROC-AUC

The final test set must remain completely untouched until this stage.

---

# 22. Final Comparison Table

Create one final model comparison table with exactly the following structure:

| Model         | Macro MCC | Macro AUPR | Macro ROC-AUC |
| ------------- | --------: | ---------: | ------------: |
| Random Forest |       ... |        ... |           ... |
| KNN           |       ... |        ... |           ... |
| SVM           |       ... |        ... |           ... |
| NNET          |       ... |        ... |           ... |
| XGBoost       |       ... |        ... |           ... |
| GLMNET        |       ... |        ... |           ... |

Use the final untouched test-set results.

Clearly indicate the best-performing model for each metric.

Do not replace the Macro metrics with Micro metrics.

---

# 23. Visualizations

Create the following final visualizations.

### Figure 1

Bar chart comparing the six models using:

**Macro MCC**

### Figure 2

Bar chart comparing the six models using:

**Macro AUPR**

Optionally, also provide the five-warning Precision-Recall curves for each model as supplementary diagnostics, but the primary comparison must remain the Macro AUPR.

### Figure 3

Bar chart comparing the six models using:

**Macro ROC-AUC**

Optionally, also provide the five-warning ROC curves for each model as supplementary diagnostics, but the primary comparison must remain the Macro ROC-AUC.

The final figures must clearly show model names and metric values.

Do not use individual-warning plots as the main model-comparison figures.

---

# 24. Feature Contribution / Interpretation

Because the objective is also to understand whether microbiome composition contributes to warning prediction, examine model feature importance or feature contribution where technically appropriate.

Use model-appropriate interpretation methods, such as:

* Random Forest feature importance
* XGBoost feature importance
* permutation importance
* SHAP or another appropriate model-agnostic/model-specific interpretation method

Interpret these as **predictive associations**, not causal relationships.

Do not perform a new feature-selection procedure unless explicitly requested.

The microbiome feature contribution analysis must remain separate from the predefined sample-specific `>0.1%` taxonomic feature-construction step.

Report which **bacterial Phylum** and **archaeal Phylum** features contribute most strongly to prediction where reliable interpretation is possible.

Clearly distinguish the taxonomic domain and taxonomic level of each feature.

---

# 25. Leakage Audit

Perform an explicit leakage audit before model fitting.

Check whether any predictor:

* directly defines one of the warning labels;
* is derived from a target variable;
* is measured after the warning state;
* is an effluent variable used to define a warning;
* is a transformed/duplicated representation of the target;
* contains site-level identifiers or other grouping information that could leak the outcome.

Remove any leakage-prone predictors.

Confirm that the final model matrix consists only of valid microbiome/genomic features.

The final predictor matrix must contain only:

* bacterial Phylum features
* archaeal Phylum features

No physicochemical, operational, substrate, process-performance, warning-definition, Site ID, Sample ID, or Season variables may be used as model predictors.

---

# 26. Required Outputs

Save/export the following outputs:

1. Cleaned merged dataset
2. Final microbiome-only predictor matrix
3. Selected bacterial **Phylum** taxa satisfying the sample-specific >0.1% rule
4. Selected archaeal **Phylum** taxa satisfying the sample-specific >0.1% rule
5. Final microbiome feature list
6. Taxonomic domain for each feature:

   * Bacteria
   * Archaea
7. Taxonomic level for each feature:

   * Phylum
8. Number of samples in which each Phylum exceeded 0.1%
9. Site-level train/test assignment
10. Site-grouped cross-validation fold assignment
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
28. The three final comparison figures
29. Final trained model objects
30. Reproducibility information

---

# 27. Reproducibility

Use and document a fixed random seed.

Save or report:

* random seed
* train/test site assignments
* CV fold assignments
* final microbiome feature list
* sample-specific >0.1% Phylum feature-construction rule
* preprocessing steps
* missing-value handling
* class-imbalance procedure
* hyperparameter search space
* best hyperparameters
* threshold optimization method
* final thresholds
* exact definitions of Macro MCC, Macro AUPR, and Macro ROC-AUC
* final model results

The entire analysis should be reproducible from the provided files and documented code.

---

# 28. Final Interpretation

The final interpretation must answer:

> **How well can microbiome composition alone, represented at the Phylum level, predict the five anaerobic digestion warning indicators simultaneously?**

Discuss:

* which model performs best overall;
* how the six models compare using Macro MCC, Macro AUPR, and Macro ROC-AUC;
* whether performance is consistent across the five warning labels;
* whether some warning labels are substantially more difficult to predict than others;
* which bacterial and archaeal **Phylum** features contribute most strongly to prediction.

Because Macro averaging gives equal weight to each warning, explicitly discuss whether the model performs consistently across all five warning types rather than relying only on performance for the most prevalent warnings.

Do not make causal claims from feature importance.

---

# 29. Critical Requirements

The final analysis MUST satisfy all of the following:

* **Multi-label classification**
* Predict all five warning labels within one overall multi-label analysis
* **Genomic/microbiome data only**
* No physicochemical/process/operational predictor variables
* Bacteria at **Phylum level**
* Archaea at **Phylum level**
* **Relative abundance > 0.1%**
* Threshold applied **sample by sample**
* Feature set constructed from the union of taxa exceeding 0.1% in at least one sample
* No Top 10/16/N selection
* No additional ML feature selection
* Site-level train/test split
* Approximately 80:20 train/test split
* No site leakage
* 5-fold site-grouped CV on training only
* Model-dependent preprocessing inside the CV pipeline
* Class-imbalance handling only within training folds
* Hyperparameter tuning on training data only
* Threshold optimization on training/CV data only
* Final test set used exactly once for final evaluation
* **Macro-average MCC**
* **Macro-average AUPR**
* **Macro-average ROC-AUC**
* Macro metrics calculated as the arithmetic mean of the five warning-specific metrics
* Do NOT use Micro-AUPR or Micro-ROC-AUC as primary evaluation metrics
* Do NOT use flattened multi-label MCC as the primary MCC metric
* Final model comparison based only on the three Macro metrics
* Preserve all five warning labels as separate binary labels within the multi-label framework
* Do NOT convert the problem into multiclass classification
* Explicit leakage audit
* Full reproducibility

Before completing the analysis, verify that every requirement above has been implemented exactly.

---

# 30. Independent Phylum-Level Analysis — IMPORTANT

Run this as a **new, independent Phylum-level analysis**.

Do NOT reuse, modify, or overwrite the previous Family-level or Genus-level analysis results.

The previous analyses may have used:

* Family-level features
* Genus-level features
* Family-level feature matrices
* Genus-level feature matrices
* Family-level or Genus-level feature lists
* previous model results
* previous hyperparameters
* previous preprocessing objects

These must NOT be treated as the Phylum-level analysis results.

Reconstruct the microbiome feature matrix specifically at the **Phylum level** from the original input files.

All of the following must be performed specifically for this Phylum-level analysis:

* Phylum-level feature construction
* sample-specific `>0.1%` filtering
* feature-union construction
* preprocessing
* class-imbalance handling
* model fitting
* hyperparameter tuning
* threshold optimization
* cross-validation
* final model evaluation
* feature contribution analysis

Do not simply rename Family- or Genus-level features as Phylum-level features.

---

# 31. Final Taxonomic Specification — HIGHEST PRIORITY

For this analysis, the taxonomic feature resolution is **Phylum level for both domains**:

**Bacteria: Phylum levels (relative abundance > 0.1%)**

**Archaea: Phylum levels (relative abundance > 0.1%)**

This Phylum-level specification overrides any previous Family-, Genus-, Order-, or Class-level configuration.

Do not use Family-level features in this analysis.

Do not use Genus-level features in this analysis.

Do not use Order-level features in this analysis.

Do not use Class-level features in this analysis.

Do not use Species-level features in this analysis.

The **only microbiome taxonomic predictors allowed** are bacterial Phylum and archaeal Phylum features constructed using the **sample-specific relative abundance >0.1% rule**.

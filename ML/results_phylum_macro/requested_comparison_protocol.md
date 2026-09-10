# Multi-Label Classification Analysis for Anaerobic Digestion Warning Prediction — Phylum Level

## 1. Objective

Perform a **new, independent multi-label classification analysis** to predict five anaerobic digestion warning indicators simultaneously using **microbiome/genomic data only**.

The five target labels are:

* `warning_acid_base_balance`
* `warning_buffer_capacity`
* `warning_acid_accumulation`
* `warning_ammonia_toxicity`
* `warning_biogas_quality`

### Critical taxonomic-level specification

This analysis is specifically a **Phylum-level microbiome analysis**.

The following specification **overrides any previous Family-level, Genus-level, Order-level, or Class-level configuration**:

* **Bacteria: Phylum levels (relative abundance > 0.1%)**
* **Archaea: Phylum levels (relative abundance > 0.1%)**

Do **NOT** use:

* Family-level features
* Genus-level features
* Order-level features
* Class-level features
* Species-level features
* Any other taxonomic resolution

Run this as a **new independent Phylum-level analysis**, rather than modifying, extending, or reusing the previous Family-, Genus-, Order-, or Class-level analysis results.

---

## 2. Input Data

Use the following files:

* `metadata.csv`
* `Dat_ARC.xlsx`
* `Dat_BAC.xlsx`

First inspect and verify the actual structure of all files, including:

* sample identifiers
* site identifiers
* seasonal information
* bacterial abundance data
* archaeal abundance data
* taxonomic annotations
* warning labels
* physicochemical variables

Do not assume column names or file structure without verification.

Expected study design to verify from the data:

* 34 unique sites
* 4 seasonal samples per site:

  * Spring
  * Summer
  * Fall
  * Winter

Do not fabricate or assume these values if the actual data differ. Report the verified structure.

---

# 3. Target Variables

Predict the following five warning labels simultaneously:

1. `warning_acid_base_balance`
2. `warning_buffer_capacity`
3. `warning_acid_accumulation`
4. `warning_ammonia_toxicity`
5. `warning_biogas_quality`

This is a **multi-label classification problem**, not a multiclass classification problem.

Each sample can have any combination of the five warning labels.

---

# 4. Predictor Variables: Microbiome/Genomic Data Only

Use **only microbiome/genomic information** as predictor variables.

### Include

* Bacterial Phylum-level relative abundances
* Archaeal Phylum-level relative abundances

### Exclude

Do NOT use:

* physicochemical variables
* substrate characteristics
* operational variables
* process-performance variables
* environmental variables
* warning-definition variables
* variables directly derived from the warning labels
* any other metadata that may cause target leakage

The following may be retained for data integration, tracking, grouping, and reproducibility but must **never be used as model predictors**:

* Site ID
* Sample ID
* Season

Perform an explicit leakage audit before modeling.

---

# 5. Phylum-Level Feature Construction

This step is critical.

## 5.1 Bacteria

Aggregate the bacterial abundance data to the **Phylum level**.

For each sample:

1. Calculate or use the bacterial relative abundance of each Phylum.
2. Identify bacterial Phyla whose relative abundance is **> 0.1% in that sample**.
3. Collect the union of all bacterial Phyla that exceed 0.1% in at least one sample.

## 5.2 Archaea

Aggregate the archaeal abundance data to the **Phylum level**.

For each sample:

1. Calculate or use the archaeal relative abundance of each Phylum.
2. Identify archaeal Phyla whose relative abundance is **> 0.1% in that sample**.
3. Collect the union of all archaeal Phyla that exceed 0.1% in at least one sample.

## 5.3 Final Feature Matrix

Construct the final microbiome feature matrix using the union of qualifying bacterial and archaeal Phyla.

For each sample and each selected Phylum:

* If relative abundance > 0.1%: retain the actual relative abundance.
* If the Phylum is present but relative abundance ≤ 0.1%: set the value to 0.
* If the Phylum is absent: set the value to 0.

Therefore, the final feature matrix must contain:

* bacterial Phylum-level features
* archaeal Phylum-level features

Clearly identify the domain of every feature.

### Important

The **>0.1% rule is a predefined microbiome feature-construction rule**, not machine-learning feature selection.

Do NOT perform any additional feature selection such as:

* global mean abundance filtering
* median abundance filtering
* Top-N taxa selection
* variance filtering
* univariate statistical filtering
* correlation-based feature selection
* RFE
* LASSO-based feature selection
* tree-based feature selection
* test-set-based feature selection

Report:

* total number of bacterial Phylum features
* total number of archaeal Phylum features
* total number of Phylum features
* the list of selected Phyla
* the number of samples in which each Phylum exceeded 0.1%

---

# 6. Comparison Across Taxonomic Resolution

This Phylum-level analysis is intended to be compared with previous analyses performed at:

* Family level
* Genus level
* Order level
* Class level

For a fair comparison:

* Keep the **same site-level train/test site assignment** used in the previous taxonomic-resolution analyses, if such an assignment has already been established.
* Keep the **same site-grouped CV fold assignment**, if already established.
* Do NOT reuse previous feature matrices.
* Reconstruct the feature matrix specifically at the Phylum level.
* Perform preprocessing independently for the Phylum-level feature matrix.
* Perform hyperparameter tuning independently.
* Perform threshold optimization independently.
* Fit the Phylum-level models independently.

If previous train/test or CV assignments are not available, create them using the rules below and save them for reproducibility.

Do not allow any sample from the same site to appear in both training and test sets.

---

# 7. Train/Test Split

Perform a **site-level approximately 80:20 train/test split**.

The unit of splitting is the **site**, not the individual sample.

All samples belonging to the same site must remain in the same partition.

Use a fixed random seed.

Prefer a group-aware stratified splitting strategy that maintains, as much as possible, the distribution of the five warning labels between training and test sets.

Save and report:

* training site IDs
* test site IDs
* number of sites in each partition
* number of samples in each partition
* warning-label prevalence in training and test sets

The test set must remain completely untouched until final evaluation.

---

# 8. Cross-Validation

Use **5-fold site-grouped cross-validation** on the training set only.

The same site must never appear in both the training and validation portions of a CV fold.

Prefer group-stratified CV where feasible so that the five warning-label distributions are reasonably preserved across folds.

All of the following must occur **inside each training fold only**:

* preprocessing
* missing-value handling
* scaling/normalization where required
* class-imbalance handling
* oversampling/undersampling
* model fitting
* hyperparameter tuning
* threshold optimization

Never perform these operations using validation or test data.

Save the CV fold assignments.

---

# 9. Models

Evaluate the following six models:

1. Random Forest
2. KNN
3. SVM
4. NNET
5. XGBoost
6. GLMNET

Use **Binary Relevance / One-vs-Rest** as the primary multi-label strategy.

That means five separate binary classifiers are trained for each model, one for each warning label.

The final prediction consists of the five predicted warning labels.

If computationally feasible, classifier chains may be explored as a supplementary analysis, but the primary analysis must remain **Binary Relevance**.

---

# 10. Class Imbalance

Because the five warning labels may be imbalanced, evaluate an appropriate imbalance-handling strategy.

Possible approaches include:

* class weights
* balanced sampling
* oversampling
* undersampling

If oversampling or undersampling is used, it must be performed **only within the training portion of each CV fold**.

Never oversample, undersample, or otherwise rebalance the validation or test data.

If using neural-network training with binary cross-entropy, use an appropriate weighted binary loss such as:

`BCEWithLogitsLoss`

when applicable.

Clearly document the imbalance-handling strategy used for each model.

---

# 11. Preprocessing

Apply model-appropriate preprocessing.

For models requiring scaling, such as:

* KNN
* SVM
* NNET

perform scaling using parameters estimated from the training portion of each CV fold only.

Do not calculate scaling parameters using the entire dataset.

For tree-based models such as:

* Random Forest
* XGBoost

scaling is generally unnecessary.

GLMNET should use appropriate preprocessing and regularization.

Document all preprocessing steps.

---

# 12. Hyperparameter Tuning

Perform hyperparameter tuning using **5-fold site-grouped CV on the training set only**.

Do not use the test set for:

* hyperparameter selection
* model selection
* threshold selection
* feature selection
* preprocessing decisions

Use a reproducible search strategy.

For each model, report:

* hyperparameter search space
* number of combinations/iterations evaluated
* best hyperparameters
* corresponding CV performance

The exact search strategy should be computationally reasonable and clearly documented.

---

# 13. Primary Evaluation Metrics

The primary evaluation metrics are:

1. **Macro-average MCC**
2. **Macro-average AUPR**
3. **Macro-average ROC-AUC**

Calculate each metric separately for all five warning labels.

Then calculate the macro-average as the arithmetic mean of the five warning-specific values.

For example:

`Macro MCC = mean(MCC_1, MCC_2, MCC_3, MCC_4, MCC_5)`

`Macro AUPR = mean(AUPR_1, AUPR_2, AUPR_3, AUPR_4, AUPR_5)`

`Macro ROC-AUC = mean(ROC-AUC_1, ROC-AUC_2, ROC-AUC_3, ROC-AUC_4, ROC-AUC_5)`

Do not use:

* Micro-AUPR as the primary metric
* Micro-ROC-AUC as the primary metric
* flattened multi-label MCC as the primary metric

Retain all five warning-specific metrics internally so that the macro-average can be verified.

---

# 14. MCC Threshold Optimization

MCC requires binary predictions.

Do NOT automatically assume that 0.5 is the optimal threshold.

Optimize the classification threshold for each warning label using **training/CV data only**.

A preferred approach is:

1. Generate out-of-fold predicted probabilities/scores from the training data.
2. Evaluate candidate thresholds for each warning.
3. Select the threshold that maximizes MCC.
4. Fix that threshold before evaluating the held-out test set.

Threshold optimization must never use test-set labels.

Report the final threshold for each warning label for each model.

---

# 15. AUPR and ROC-AUC

Calculate AUPR and ROC-AUC using continuous predicted probabilities or decision scores.

Do not convert probabilities to binary predictions before calculating AUPR or ROC-AUC.

AUPR and ROC-AUC should therefore be calculated independently of the final MCC threshold.

---

# 16. Final Test Evaluation

After:

* feature construction
* preprocessing decisions
* class-imbalance strategy selection
* hyperparameter tuning
* model selection procedure
* threshold optimization

has been completed using the training data and CV only, evaluate the final models on the held-out test set.

The test set must be evaluated **exactly once** for final performance reporting.

Report for every model:

### Warning-specific results

For each of the five warning labels:

* MCC
* AUPR
* ROC-AUC
* optimized threshold
* number of positive samples
* number of negative samples

### Macro results

* Macro MCC
* Macro AUPR
* Macro ROC-AUC

---

# 17. Final Model Comparison

The primary model comparison must use only the three macro-level metrics.

Create exactly the following table:

| Model         | Macro MCC | Macro AUPR | Macro ROC-AUC |
| ------------- | --------: | ---------: | ------------: |
| Random Forest |       ... |        ... |           ... |
| KNN           |       ... |        ... |           ... |
| SVM           |       ... |        ... |           ... |
| NNET          |       ... |        ... |           ... |
| XGBoost       |       ... |        ... |           ... |
| GLMNET        |       ... |        ... |           ... |

Clearly identify the best-performing model for each metric.

Do not select a model based on a single warning label alone.

---

# 18. Visualization

Create the following three primary figures:

### Figure 1

Bar chart comparing the six models by:

**Macro MCC**

### Figure 2

Bar chart comparing the six models by:

**Macro AUPR**

### Figure 3

Bar chart comparing the six models by:

**Macro ROC-AUC**

The x-axis should contain the six models.

The y-axis should contain the corresponding metric.

Optional supplementary visualizations may include:

* warning-specific PR curves
* warning-specific ROC curves
* per-warning performance comparisons
* feature importance plots

But the three macro-metric bar charts are the primary model-comparison figures.

---

# 19. Feature Contribution / Interpretation

After final model fitting, investigate which Phylum-level features contribute most strongly to model predictions.

Use appropriate model-specific interpretation methods, such as:

### Random Forest

* feature importance
* permutation importance

### XGBoost

* feature importance
* SHAP values where feasible

### SVM / KNN / NNET / GLMNET

Use an appropriate model-agnostic method such as:

* permutation importance
* SHAP where computationally appropriate

Report important bacterial and archaeal Phylum-level features.

Clearly distinguish:

* predictive importance
* statistical association
* biological interpretation

Do **not** interpret feature importance as evidence of causality.

---

# 20. Required Leakage Audit

Perform an explicit leakage audit before final modeling.

Verify that:

* no test-site samples are present in training
* no site appears in both train and test
* no test data were used for preprocessing
* no test data were used for hyperparameter tuning
* no test data were used for threshold optimization
* no test data were used for feature selection
* no warning-definition variables were used as predictors
* no physicochemical variables were included
* Site ID was not used as a predictor
* Sample ID was not used as a predictor
* Season was not used as a predictor
* no target-derived variables were included as predictors
* oversampling/undersampling was not applied to validation/test data

Report the leakage audit results explicitly.

---

# 21. Reproducibility

Use a fixed random seed throughout the analysis.

Save and report:

* random seed
* verified dataset structure
* merged dataset
* microbiome-only predictor matrix
* Phylum-level feature list
* bacterial Phylum feature list
* archaeal Phylum feature list
* number of samples exceeding 0.1% for each Phylum
* train/test site assignments
* CV fold assignments
* preprocessing procedures
* missing-data handling
* class-imbalance strategy
* hyperparameter search spaces
* best hyperparameters
* threshold optimization procedure
* final thresholds
* test-set probabilities
* test-set predictions
* warning-specific metrics
* macro metrics
* model comparison table
* feature importance results
* all primary figures
* final fitted models

Make the complete analysis reproducible from the saved outputs.

---

# 22. Final Interpretation

Provide a concise scientific interpretation addressing:

1. How well can **Phylum-level microbiome composition alone** predict the five anaerobic digestion warning indicators simultaneously?
2. Which model performs best according to:

   * Macro MCC
   * Macro AUPR
   * Macro ROC-AUC
3. Which warning indicators are easier or more difficult to predict?
4. Are model performances consistent across the five warning types?
5. Which bacterial and archaeal Phylum-level features are most predictive?
6. How does the Phylum-level performance compare with the previously analyzed:

   * Family level
   * Genus level
   * Order level
   * Class level

When comparing taxonomic resolutions, use the same evaluation framework and, where available, the same site-level train/test and CV assignments.

Do not make causal claims from predictive models or feature importance.

The final conclusion should clearly state the predictive value and limitations of **Phylum-level microbiome composition alone** for anaerobic digestion warning prediction.

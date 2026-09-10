# Physicochemical contribution to the joint five-label task

Analyzed **RandomForest**, the model already selected by training CV. This is model-specific predictive importance, not causation.

Used 109 training samples, the existing five grouped folds, and 30 permutations per feature. No test predictions were computed.

## Ranked physicochemical features
| Rank | Feature | Mean Micro-AUPR drop | Mean multilabel MCC drop |
|---:|---|---:|---:|
| 1 | substrate_avg_TVFAs | 0.00935 | -0.00129 |
| 2 | substrate_avg_sCOD | 0.00833 | 0.01939 |
| 3 | substrate_avg_Lipid | 0.00590 | 0.00801 |
| 4 | Q_total_Tpy | 0.00334 | -0.01293 |
| 5 | substrate_avg_Protein | 0.00251 | 0.00158 |
| 6 | substrate_avg_VS | 0.00213 | 0.01239 |
| 7 | substrate_avg_HAc | 0.00213 | 0.00079 |
| 8 | substrate_avg_TC | 0.00183 | 0.00177 |
| 9 | substrate_avg_COD | 0.00181 | -0.00077 |
| 10 | T_C | 0.00112 | 0.00684 |
| 11 | substrate_avg_pH | -0.00008 | 0.00795 |
| 12 | HRT_d | -0.00167 | 0.00321 |
| 13 | substrate_avg_HPro | -0.00179 | 0.02135 |
| 14 | substrate_avg_ALK | -0.00180 | 0.00000 |
## Interpretation
Largest Micro-AUPR contributions: substrate_avg_TVFAs, substrate_avg_sCOD, substrate_avg_Lipid.
Smallest Micro-AUPR contributions: HRT_d, substrate_avg_HPro, substrate_avg_ALK.
The MCC ranking differs: its largest contributions are substrate_avg_HPro, substrate_avg_sCOD, and substrate_avg_VS. A low Micro-AUPR rank is not evidence of low importance under MCC; probability ranking and thresholded decisions measure different aspects of prediction.

A positive drop means shuffling worsened overall prediction; near zero means little detected reliance; a negative drop means shuffling improved the score. Negative importance does not establish a protective effect.
Importance is the absolute performance drop in metric units. The additional positive-importance percentage in the CSV is normalized only across these 16 metadata variables; it is not explained variance or a fraction of all model information, and excludes microbial features from its denominator.
## Categorical predictors
- Season: Micro-AUPR drop 0.00264; MCC drop 0.00901.
- substrate_type: Micro-AUPR drop 0.00114; MCC drop 0.00336.
## Method and limits
For each original fold, refit the selected settings on the other four folds, including the training-only abundance union, imputation, scaling and encoding. Baseline OOF probabilities were verified against the original saved OOF predictions. Shuffle one original metadata column in validation rows, including its missingness, before the saved fold transformation. Each categorical variable is shuffled as one column, keeping its encoded levels together. Pool all validation predictions before calculating either metric.
Use the existing saved thresholds without reoptimization. Micro-AUPR is trapezoidal PR area on flattened sample-label probabilities. Multilabel MCC is binary MCC on flattened sample-label predictions. All other predictors remain unchanged.
The ranked plot uses mean Micro-AUPR decrease; a second plot reports MCC decrease. Error bars are permutation SD, not confidence intervals. Repeated seasons and site correlation make sample-level permutations dependent; shuffling can create unrealistic combinations and does not preserve site trajectories. These are descriptive model reliance estimates, not independent-site inferential effects.
Correlated chemistry variables and microbiome predictors can substitute for each other, masking importance or distributing it across variables. Training CV was already used to select settings and thresholds, so this is post-selection interpretation rather than unbiased external validation.
Permutation importance does not indicate whether higher or lower values increase warning probability. No direction or causal effect is inferred, and no features are removed.
The two categorical variables are shown separately from the 14 continuous physicochemical predictors. Values such as operating temperature may originally be reported as ranges; this analysis uses the established parsed values.
## Reproduction
`python ML/importance_multilabel.py --seed 20260908 --repeats 30` using the original modeling environment. Configuration, baseline predictions, per-repeat drops, permuted OOF probabilities, ranked CSVs and both plots are saved here.

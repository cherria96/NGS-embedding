"""Build the report from saved outputs, without refitting or predicting."""
from pathlib import Path
import json
import html
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=['Multilabel MCC','Micro-AUPR','Micro-ROC-AUC']
def table(df):
    return df.to_markdown(index=False,floatfmt='.4f')

def build(out):
    out=Path(out)
    cv=pd.read_csv(out/'cv_performance.csv'); test=pd.read_csv(out/'test_performance.csv')
    taxa=pd.read_csv(out/'selected_taxa.csv'); assignment=pd.read_csv(out/'split_and_cv_assignment.csv')
    labels=pd.read_csv(out/'label_distribution.csv')
    selected=json.loads((out/'cv_selected_model.json').read_text())['Model']
    config=json.loads((out/'run_config.json').read_text())
    config['Micro-ROC-AUC']='ROC-AUC on flattened sample-label truth and probability vectors.'
    config['feature_union_scope']=f'{len(assignment)} matched samples; fixed predefined threshold rule only; includes {(taxa.training_samples_above_0_1_percent == 0).sum()} columns with no training values above threshold.'
    config['grouping']='Site; QC replicate aliases BSIb→BSI and DGYb→DGY grouped conservatively. GHGa and GHGb retain distinct Site codes.'
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    counts=taxa.groupby('block').size().rename('final_microbiome_features').reset_index()
    counts['source_taxa']=counts.block.map({'BAC_C':150,'ARC_C':19})
    counts.to_csv(out/'feature_counts.csv',index=False)
    warnings=pd.read_csv(out/'fit_warnings.csv')
    leaders={metric:test.loc[test[metric].idxmax(),'Model'] for metric in METRICS}
    cvleaders={metric:cv.loc[cv[metric].idxmax(),'Model'] for metric in METRICS}
    text=['# Class-only multi-label warning prediction',
      f'Five warnings are predicted jointly using binary relevance and microbiome predictors only. CV selected **{selected}** using the equal-weight mean rank across the three metrics.',
      '## Final test performance',table(test),
      'Test metric leaders: '+str(leaders)+'. These are descriptive results; they do not change the CV-selected model.',
      'A unanimous winner exists only when all three metrics favor the same model. Otherwise the rankings reflect a trade-off, not a single best model on every criterion.',
      '## Training cross-validation',table(cv),'CV metric leaders: '+str(cvleaders)+'.',
      'These are pooled out-of-fold tuning estimates, not nested unbiased estimates. Hyperparameter selection and shared-threshold optimization use these same predictions, so CV scores, especially MCC, are optimistic.',
      '## Data integration and identity audit',
      'metadata.csv has 140 rows and 135 columns. BAC C(%) has 150 annotation rows and 140 samples; ARC C(%) has 19 annotation rows and 138 samples. Workbook sample suffixes are removed by retaining the first two hyphen-separated components; b and apostrophe markers remain intact. All normalized workbook IDs match metadata. No duplicate or missing SampleID/Site IDs were found; unique SampleID also guarantees unique Site/SampleID pairs.',
      "Samples 1-32' (JJY, Summer) and 4-24 (GHGa, Spring) lack ARC profiles and are explicitly excluded, not filled with zero. See sample_audit.csv and sample_id_mapping.csv. The joint cohort contains 138 samples, 37 Site codes, and 35 conservative groups after joining BSIb with BSI and DGYb with DGY. The supplied files do not support an assumption of exactly 34 complete four-season sites.",
      'ADP and YCG lack Spring metadata; JJY lacks Summer ARC and GHGa lacks Spring ARC. BSIb and DGYb are Summer-only QC records grouped with their parent sites. season_audit.csv lists every retained site/season/sample. GHGa/GHGb retain distinct metadata Site codes; facility-level identity would require source confirmation before treating this as facility-external validation.',
      '## Split and label prevalence',
      'Seed 42 creates an approximately 80:20 site-grouped split; seed 43 assigns five training CV folds. A 5,000-candidate random allocation minimizes label/sample imbalance while preserving groups. Labels are used only for initial stratification, not to select a split based on model performance. All six models use identical assignments.',
      table(assignment.groupby('split').agg(samples=('SampleID','size'),site_groups=('SiteGroup','nunique')).reset_index()),
      'Training groups: '+', '.join(sorted(assignment.loc[assignment.split=='train','SiteGroup'].unique())),
      'Test groups: '+', '.join(sorted(assignment.loc[assignment.split=='test','SiteGroup'].unique())),
      table(labels[labels.scope.isin(['train','test'])][['scope','target','samples','positive']]),
      'Rare positives cannot populate every validation fold. This limits stability despite grouped stratification. split_and_cv_assignment.csv lists sample IDs, Site, SiteGroup, Season, split, and validation fold.',
      '## Feature construction',
      f'Use only BAC C(%) and ARC C(%), in percentage units. Retain a value only when strictly >0.1 in that sample; otherwise set it to zero. Do not renormalize. The union across the 138 matched samples defines {int((taxa.block == "BAC_C").sum())} BAC-source and {int((taxa.block == "ARC_C").sum())} ARC-source columns, {len(taxa)} total, retained in every fold. This is the explicitly allowed predefined feature-space rule; no global means, Top-N, or learned feature selection are used. {int((taxa.training_samples_above_0_1_percent == 0).sum())} columns are zero throughout the training set and are retained under that rule.',
      'Domain prefixes denote workbook origin. Supplied off-domain/unidentified annotations remain included under the requested abundance-only criterion. Workbook origin is not proof that every retained annotation belongs to that biological domain; unresolved annotations are preserved as supplied. No unrequested taxonomy cleanup was applied.',
      'Complete selected annotations and sample counts:',table(taxa[['block','taxon','samples_above_0_1_percent','training_samples_above_0_1_percent']]),
      '## Pipeline, tuning and thresholds',
      'Binary relevance fits five classifiers internally; all evaluation pools their outputs. Eight seeded random hyperparameter candidates per model are tested in five grouped folds: 48 configurations and 240 outer-fold multi-label fits. Search spaces and versions are in run_config.json; all candidate/fold scores are in hyperparameter_tuning.csv. Best settings are in each model’s best_params.json.',
      'KNN, SVM, NNET and GLMNET use fold-local standardization. RF and XGBoost are unscaled. All abundance values are finite and nonnegative; no missing microbiome cells required imputation. Fold-local median imputation remains in the saved pipeline as a safeguard. Missing whole ARC profiles are excluded as above.',
      'RF/SVM/GLMNET use balanced class weights; XGBoost uses the training-label negative/positive ratio. KNN and NNET use per-label random oversampling only on fold training rows. Single-class training labels use constant predictions. SVM probabilities use three-fold site-grouped inner out-of-fold sigmoid calibration with preprocessing refitted in each inner fold.',
      'GLMNET is implemented as elastic-net logistic regression using sklearn saga, not the R glmnet package. NNET uses sklearn MLP. Model objects and preprocessing objects are saved as joblib files.',
      'Candidate and model selection use equal-weight mean rank over all three pooled CV metrics; deterministic ties retain search/model order. Candidate MCC uses a shared threshold optimized on its OOF predictions. The final shared threshold maximizes flattened MCC over 0.01–0.99, with ties closest to 0.5. A shared threshold limits extra fitting with very few positives; each of the five labels receives the same fixed threshold for a given model.',
      table(pd.read_csv(out/'thresholds.csv')),
      f'{len(warnings)} fitting warnings were recorded; see fit_warnings.csv. '+('Warnings include convergence limits; convergence is not guaranteed for those fits.' if len(warnings) else 'No fitting warnings were recorded.'),
      '## Metric definitions and isolation',
      'Multilabel MCC = binary MCC calculated after flattening all sample-label pairs across the five warning labels. Micro-AUPR = trapezoidal area under the precision–recall curve of flattened labels and probabilities, not average precision. Micro-ROC-AUC = ROC-AUC of flattened labels and probabilities. Only MCC uses thresholded predictions. These pooled metrics do not measure exact five-label-vector correctness.',
      'Every metadata column is excluded from predictors, including all effluent warning-definition measurements, substrate chemistry, operation, Season, Site and identifiers. The full excluded list and genomic whitelist are in leakage_check.json. Identifiers and Season appear only in integration/audit/prediction outputs. Feature matrix SampleID is an index, not a predictor.',
      'The test set was not used for fitting, preprocessing fitting, tuning, thresholds, or model selection in this run. Models and thresholds were saved before test evaluation. The fixed sample-specific union is the sole permitted feature-space exception. This deterministic split coincides with a previously evaluated repository holdout: it is not historically untouched or external validation.',
      '## Interpretation',
      'The results describe pooled prediction across five warning conditions from microbiome composition alone. The small test cohort (29 samples from seven groups), rare positives, prior use of this holdout, and imperfect supplied taxonomy limit generalization claims. Warning zeros may include missing source chemistry, as documented in the root README. No causal claims or deployment reliability are established.',
      '## Reproduce',
      'Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ngs-mpl /home/best/anaconda3/envs/tsf-ad/bin/python ML/train_class.py --seed 42 --n-iter 8` from the repository root. Input SHA256 hashes and package versions are saved in run_config.json.',
      'All required artifacts are alongside this report: cleaned_merged.csv, final_feature_matrix.csv, selected_taxa.csv, selected_BAC_C.txt, selected_ARC_C.txt, assignments, tuning scores, CV scores, thresholds, model/preprocessing objects, and combined actual/probability/binary prediction CSVs. Figures: multilabel_mcc.png, micro_aupr.png, micro_roc_auc.png, micro_precision_recall.png.'
    ]
    (out/'report.md').write_text('\n\n'.join(text)+'\n')
    cards = ''.join(f'<figure><img src="{name}.png" alt="{title}"><figcaption>{title}</figcaption></figure>'
                    for name,title in [('multilabel_mcc','Multilabel MCC'),('micro_aupr','Micro-AUPR'),('micro_roc_auc','Micro-ROC-AUC')])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Class-level microbiome warning prediction</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#19323c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%;max-width:800px}}figure{{margin:30px 0}}a{{color:#156779}}</style>
<h1>Class-level microbiome warning prediction</h1>
<p>129 microbiome predictors · 138 matched samples · Five labels evaluated together</p>
<p>CV-selected model: <strong>{html.escape(selected)}</strong>. Test rankings are descriptive and do not change that selection.</p>
<h2>Held-out test results</h2>{test.to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)}
<p>Test metric leaders: {html.escape(str(leaders))}.</p>
<p>The test has 29 samples from seven groups. This dataset and holdout were used in previous analyses; this run isolates test data from fitting and selection. Rare positives and tuning optimism limit conclusions.</p>
{cards}<h2>Methods and downloads</h2>
<p><a href="report.md">Complete report and taxa lists</a> · <a href="test_performance.csv">Test results</a> · <a href="cv_performance.csv">CV results</a> · <a href="final_feature_matrix.csv">Feature matrix</a> · <a href="selected_taxa.csv">Selected taxa and sample counts</a> · <a href="split_and_cv_assignment.csv">Site and fold assignments</a></p>
<p>Multilabel MCC is binary MCC across flattened sample-label pairs. Micro-AUPR is trapezoidal PR area; Micro-ROC-AUC uses flattened probabilities. No metadata predictors or additional feature selection are used.</p></html>'''
    (out/'index.html').write_text(page)
if __name__=='__main__': build(Path(__file__).resolve().parent/'results_class')

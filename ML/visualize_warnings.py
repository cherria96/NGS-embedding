#!/usr/bin/env python3
"""Create exportable overview figures and a local interactive results gallery."""
from pathlib import Path
import argparse
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

LABELS = ['acid_base_balance', 'buffer_capacity', 'acid_accumulation', 'ammonia_toxicity', 'biogas_quality']
NAMES = ['Acid–base balance', 'Buffer capacity', 'Acid accumulation', 'Ammonia toxicity', 'Biogas quality']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent/'results_warnings')
    args = parser.parse_args()
    root = args.results_dir
    out = root/'visualizations'; out.mkdir(exist_ok=True)
    cv = pd.read_csv(root/'cv_performance.csv').set_index('Model')
    test = pd.read_csv(root/'test_performance.csv').set_index('Model')
    models = cv.index.tolist()  # preserve CV ranking, not test-based selection
    test = test.loc[models]
    support = pd.read_csv(root/'label_distribution.csv')
    selected = json.loads((root/'cv_selected_model.json').read_text())['Model']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
    def save(fig, name):
        fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    x = np.arange(len(models))
    for ax, metric, title in zip(axes, ['Macro_AUPR', 'Macro_MCC', 'Micro_AUPR'], ['Macro AUPR (average precision)', 'Macro MCC at threshold 0.5', 'Micro AUPR (average precision)']):
        for offset, frame, color, legend in [(-.19, cv, '#4d718c', 'Training CV · pooled OOF'), (.19, test, '#dc8952', 'Independent test')]:
            bars = ax.bar(x+offset, frame.loc[models, metric], .36, color=color, label=legend)
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.set(xticks=x, xticklabels=models, title=title, ylim=(-.08, .72))
        ax.tick_params(axis='x', rotation=45); ax.axhline(0, color='#666', lw=.7)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Six-model comparison · five independent warning labels', fontsize=17, y=1.02)
    fig.text(.5, -.06, f'CV-selected model: {selected}  |  109 training / 29 test samples  |  Model order follows CV Macro AUPR\nCV scores were used for tuning; rare-label test scores have substantial uncertainty.', ha='center', fontsize=10)
    fig.tight_layout(); save(fig, 'model_comparison')

    counts = support[support.scope=='test'].set_index('target')
    ticks = [f'{name}\n{int(counts.loc["warning_"+label,"positive"])} positive / {int(counts.loc["warning_"+label,"samples"])} test' for label,name in zip(LABELS,NAMES)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    for ax, metric, cmap, low, title in [(axes[0], 'AUPR', 'YlGnBu', 0, 'Test AUPR · higher is better'), (axes[1], 'MCC', 'RdBu', -1, 'Test MCC · threshold 0.5')]:
        vals = test[[metric+'_warning_'+s for s in LABELS]].values
        im = ax.imshow(vals, vmin=low, vmax=1, cmap=cmap, aspect='auto')
        ax.set(xticks=range(5), xticklabels=ticks, yticks=range(6), yticklabels=models, title=title)
        ax.tick_params(axis='x', labelsize=8)
        for i in range(6):
            for j in range(5):
                v = vals[i,j]
                ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if (v>.65 or v<-.65) else '#151515')
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Which warning conditions does each model predict?', fontsize=17, y=1.02)
    fig.text(.5,-.02,'Buffer capacity and acid accumulation each have only one positive test sample. AUPR = 1 does not imply reliable detection at threshold 0.5.',ha='center')
    fig.tight_layout(); save(fig,'per_label_heatmaps')

    fig, ax = plt.subplots(figsize=(10,4.8))
    for offset, scope, color in [(-.18,'train','#4d718c'),(.18,'test','#dc8952')]:
        f = support[support.scope==scope].set_index('target').loc[['warning_'+s for s in LABELS]]
        bars=ax.bar(np.arange(5)+offset,f.positive_percent,.34,label=f'{scope.title()} (n={int(f.samples.iloc[0])})',color=color)
        ax.bar_label(bars, labels=[f'{int(n)} ({p:.1f}%)' for n,p in zip(f.positive,f.positive_percent)],fontsize=9,padding=3)
    ax.set(xticks=range(5),xticklabels=NAMES,ylabel='Positive samples (%)',title='Label support determines how much confidence to place in each score',ylim=(0,27))
    ax.legend(); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'label_support')

    predictions={model:pd.read_csv(root/f'{model}_test_predictions.csv',index_col='SampleID') for model in models}
    fig, axes=plt.subplots(2,3,figsize=(14,8))
    colors=plt.get_cmap('tab10').colors
    for j,(label,name) in enumerate(zip(LABELS,NAMES)):
        ax=axes.flat[j]; target='warning_'+label
        for k,model in enumerate(models):
            f=predictions[model]; y=f[target+'_actual']; p=f[target+'_probability']
            precision,recall,_=precision_recall_curve(y,p)
            ax.step(recall,precision,where='post',color=colors[k],label=f'{model} · AP {average_precision_score(y,p):.3f}')
        ax.axhline(y.mean(),ls='--',color='grey',label=f'Prevalence {y.mean():.3f}')
        ax.set(title=f'{name} · {int(y.sum())} positive',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.04));ax.legend(fontsize=7)
    axes.flat[5].axis('off')
    axes.flat[5].text(0,.85,'Independent test set\n\nCurves use predicted probabilities.\nAUPR means average precision.\n\nOne-positive curves are unstable;\nread them alongside label support.\n\nNo test-based threshold tuning.',va='top',fontsize=12)
    fig.suptitle('Precision–recall comparison across all six models',fontsize=17)
    fig.tight_layout();save(fig,'precision_recall_comparison')

    options=''.join(f'<option value="{m}" '+('selected' if m==selected else '')+f'>{m}</option>' for m in models)
    label_options=''.join(f'<option value="warning_{s}">{n}</option>' for s,n in zip(LABELS,NAMES))
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AD warning model results</title>
<style>body{font:16px system-ui;background:#f3f6f9;color:#172c40;margin:0}main{max-width:1400px;margin:auto;padding:28px}h1{margin-bottom:8px}p{line-height:1.6}.card{background:white;border-radius:12px;padding:22px;margin:22px 0;box-shadow:0 2px 8px #1231}img{width:100%;height:auto}select{font:inherit;padding:8px;margin:8px 20px 8px 4px}a{color:#1465a2}.note{border-left:4px solid #df9254;padding:10px 16px;background:#fff4e9}.scroll{overflow:auto;max-height:950px}#all{max-width:1100px}label{display:inline-block}</style>
<main><p><a href="../feature_importance/index.html">Explore per-label feature importance</a></p><h1>Anaerobic digestion warning predictions</h1><p>Six binary-relevance models · five warning labels · site-grouped validation.</p>
<p class="note">NNET was selected using training CV AUPR. Its test AUPR is 0.509, but its MCC is 0.004 at threshold 0.5. KNN has the highest test MCC (0.338). Buffer capacity and acid accumulation each have just one positive test sample. Test rankings are descriptive.</p>
<div class="card"><h2>Overall comparison</h2><img src="model_comparison.png" alt="CV and test model comparison"><a href="model_comparison.pdf">Download PDF</a></div>
<div class="card"><h2>Performance by warning</h2><img src="per_label_heatmaps.png" alt="Per label AUPR and MCC heatmaps"><a href="per_label_heatmaps.pdf">Download PDF</a></div>
<div class="card"><h2>How many positive samples?</h2><img src="label_support.png" alt="Training and test label counts"></div>
<div class="card"><h2>Precision–recall curves</h2><img src="precision_recall_comparison.png" alt="Six model precision recall comparison"><a href="precision_recall_comparison.pdf">Download PDF</a></div>
<div class="card"><h2>Explore a model</h2><label>Model <select id="model">OPTIONS</select></label><label>Evaluation <select id="scope"><option value="test">Independent test</option><option value="CV_OOF">Training CV (out-of-fold)</option></select></label><label>Warning <select id="label">LABELS</select></label><p id="caption"></p><img id="diagnostic" alt="Confusion matrix, precision recall, ROC and probability distributions"><p><a id="csv">Download actual labels, predicted labels and probabilities (CSV)</a></p><h3>All samples and warnings</h3><div class="scroll"><img id="all" alt="Actual labels, predicted labels and predicted probabilities for each sample"></div></div>
<p>Thresholds are fixed at 0.5. CV results were used for hyperparameter selection and are not unbiased nested-CV estimates. Known QC replicates are grouped with their parent sites; the analysis uses 35 grouping units. <a href="../performance_summary.md">Full results</a> · <a href="../split_and_cv_assignment.csv">Site assignments</a></p></main>
<script>function update(){const m=document.getElementById('model').value,s=document.getElementById('scope').value,l=document.getElementById('label').value;document.getElementById('diagnostic').src='../plots/'+m+'/'+s+'/'+l+'.png';document.getElementById('all').src='../plots/'+m+'/'+s+'/all_labels.png';document.getElementById('csv').href='../'+m+'_'+s+'_predictions.csv';document.getElementById('caption').textContent=m+' — '+(s==='test'?'29 independent test samples':'109 training out-of-fold predictions')+' — classification threshold 0.5';}document.querySelectorAll('select').forEach(e=>e.addEventListener('change',update));update();</script></html>'''.replace('OPTIONS',options).replace('LABELS',label_options)
    (out/'index.html').write_text(document)
    print(f'Created 4 PNG/PDF figures and interactive gallery: {out / "index.html"}')

if __name__=='__main__':
    main()

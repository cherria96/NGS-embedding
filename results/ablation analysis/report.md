# Ablation analysis

- [Existing neural-model positive-class-weight ablation](pos_weight/ablation_summary.md)
- [Experiment 1: graph mixture, domain restriction, resolution, and compression comparisons](../classification%20result/fusionMLv2/exp1/REPORT.md)

`function/build_ablation_summary.py` regenerates the historical weighted-versus-unweighted comparison from its existing JSON results. `python main.py --stage ablate --config <config.yaml>` annotates that fusionML run's family metrics with graph/domain/interaction settings. Additional controls must first be included in `cv.families` and trained. This reorganization adds no new ablation findings.

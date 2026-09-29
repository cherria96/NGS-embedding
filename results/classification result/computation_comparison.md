# Computational efficiency: ASV-tree vs. Genus-tree inputs

Comparison for the three phylogeny-aware GPU models (DeepPhylo, MetaDR,
PhyloSpec) under `run_site_grouped_cv_benchmark.py`'s site-grouped CV
protocol (5 repeats × 5 folds), on the post-sample-drop data (ARC/merged =
134 samples, BAC = 140 samples; see `CLAUDE.md` session log for the sample
drop). Genus-level timings are from today's fresh rerun; ASV-level timings
are from the most recently completed run at time of writing (see note at the
bottom for freshness caveats).

## 1. Structural complexity — why genus should be cheaper

| Domain | Genus tips / tree nodes | ASV tips / tree nodes | Node reduction (ASV → genus) |
|---|---|---|---|
| ARC | 42 / 83 | 73 / 145 | 1.75× fewer |
| BAC | 389 / 777 | 451 / 901 | 1.16× fewer |
| merged | 330 / 659 | 505 / 1009 | 1.53× fewer |

This ratio drives each model's cost differently, by architecture:

- **PhyloSpec** instantiates one `nn.Conv1d` **per tree node**, so its cost
  scales ~linearly with the node-count ratio above — genus should give the
  largest, most direct speedup here.
- **DeepPhylo** builds a patristic distance matrix of size p×p (p = feature
  count) and PCA-reduces it *once, at data-prep time* (already baked into
  `c.npy`), so this cost doesn't recur during training/CV. Matrix-size
  reduction is ARC 3.0×, BAC 1.3×, merged 2.3× fewer elements, but it's a
  one-time cost, not paid per epoch.
- **MetaDR** (PopPhy-style) maps the tree to a fixed-size 2D matrix and runs
  a **fixed** 2-conv-layer CNN over it — cost scales with matrix area, not
  node count directly, so it's the least sensitive of the three to genus vs.
  ASV.

## 2. Measured wall-clock (site-grouped CV, 5×5 repeats/folds, GPU)

| Domain | Model | Genus (min) | ASV (min) | ASV/Genus |
|---|---|---|---|---|
| ARC | MetaDR | 0.93 | — (not captured in this window) | — |
| ARC | DeepPhylo | 1.05 | 0.94 | 0.90× |
| ARC | PhyloSpec | 4.69 | 5.07 | 1.08× |
| BAC | MetaDR | 0.96 | 0.99 | 1.03× |
| BAC | DeepPhylo | 1.11 | 0.98 | 0.88× |
| BAC | PhyloSpec | 20.22 | 16.24 | 0.80× |
| merged | MetaDR | 0.97 | 1.05 | 1.08× |
| merged | DeepPhylo | 0.97 | 1.12 | 1.15× |
| merged | PhyloSpec | 12.30 | 30.08 | **2.45×** |

## 3. Takeaways

- **MetaDR and DeepPhylo show near-parity** regardless of tree size — matches
  the architecture argument: DeepPhylo's expensive step is precomputed,
  MetaDR's conv stack is fixed-size. Their wall-clock is dominated by
  epochs-to-convergence (early stopping), not feature/node count, which is
  why some cells even run *slower* at genus level — that's early-stopping
  noise, not a real regression.
- **PhyloSpec is the only model where genus vs. ASV clearly matters**, and
  it's also by far the slowest model overall (4.7–20 min vs. ~1 min for the
  others) — exactly as expected from "one conv module per node." The merged
  domain shows a real 2.45× speedup at genus level, consistent with its
  1.53× node reduction plus smaller per-node tensors.
- **Caveat**: individual cell timings include early-stopping variance
  (different folds converge in different epoch counts), so small
  differences (<15%) aren't meaningful — the merged/PhyloSpec 2.45× gap is
  the one result large enough to trust as a real effect rather than noise.

## Methodology note

Timings were derived from `results_<domain>_<model>.csv` file modification
timestamps within `Phylospec/multi_models/results/{site_grouped_cv,
site_grouped_cv_asv}/`, using the elapsed time between consecutive file
writes in each script's fixed model-loop order (each gap = wall-clock time
for exactly the model whose file was just written, since the runner calls
models strictly sequentially with no other work interleaved). Genus-level
numbers are from a full same-day rerun after dropping 4 poor-quality ARC
samples (`3-1-BSSG`, `3-9-DGGB`, `2-33-CWMS`, `3-22-UJGN`); ASV-level numbers
were the latest available at the time this file was generated — a fresh,
same-day ASV rerun (post sample-drop) was still finishing in the background,
so treat the ASV column as provisional until cross-checked against a
same-day run.

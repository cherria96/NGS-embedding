# Task — inject the bacteria × archaea edge set into Phylo-Spec and MetaDR

## 0. What this is and why

The benchmark's phylogeny places Bacteria and Archaea on opposite sides of the root.
Every model in the benchmark inherits that geometry as a prior, so none of them can
express a syntrophic partnership between a bacterium and a methanogen — the two
organisms are maximally distant before any data is seen. `build_cross_domain_graph.py`
estimates which bacterium–archaeon pairs actually covary after conditioning on
upstream operating variables, and writes that as an edge set.

Your job is to get that edge set **into Phylo-Spec and into MetaDR**, and to build the
control arms that make the resulting comparison mean something. This is not a
"does it improve accuracy" task. It is a "build the three arms, score them
identically, and report what the contrast supports" task.

**Read this whole file before writing code.** Several of the most important
instructions are corrections to what the bundled code appears to do.

---

## 1. Inputs that already exist

| File | What it is |
|---|---|
| `build_cross_domain_graph.py` | estimates the edge set; already patched to refuse outcome-side covariates |
| `{out}_edges.csv` | the retained edges: `bacterium, archaeon, rho_conditioned, rho_raw, q, patristic_distance` |
| `{out}_DeepPhylo_X.npy` | the **raw** abundance matrix actually fed to models (not CLR, not residualised) |
| `{out}_DeepPhylo_features.txt` | the kept feature names, in column order of `X` |
| `{out}_diagnostics.json` | includes `spearman_weight_vs_patristic`, `edges_gained_by_conditioning` |
| `evaluate_multilabel.py` | `site_grouped_cv`, `tune_thresholds_mcc`, `score_multilabel`, `pos_weight_from_labels` |
| `src_dump/Phylospec_src_model_PhyloSpec.py` | the Phylo-Spec model |
| `src_dump/Phylospec_src_model_data_processing.py` | `get_conv_order`, `calculate_node_weights`, `match_leaf_nodes`, `tree_p` |
| `src_dump/Phylospec_multi_models_MetaDR_MetaDR.py` | MetaDR |

The genus-level tree and the per-domain merge are covered by
`CLAUDE_CODE_TASK_genus_tree.md`. Assume that tree exists and that its tip labels
match `{out}_DeepPhylo_features.txt` exactly. **If they do not match, stop and report
the mismatch count — do not fuzzy-match your way past it.**

---

## 2. Prerequisite — emit the null edge list (do this first)

`build_cross_domain_graph.py` computes a degree-preserving rewired mask `kn` via
`rewire_null(keep, args.seed)` but only ever saves its *embedding*. Both injections
below need the rewired edge list itself.

Add, immediately after the existing `kn = rewire_null(...)` line:

```python
bin_, ain_ = np.where(kn)
pd.DataFrame({
    "bacterium": [kept[bac_idx[b]] for b in bin_],
    "archaeon":  [kept[arc_idx[a]] for a in ain_],
    "rho_conditioned": rho[bin_, ain_],      # read at the rewired position
}).to_csv(f"{args.out}_edges_null.csv", index=False)
```

**Then fix the weight column.** Reading `rho` at rewired positions is wrong: the null
arm would get systematically weaker edges than the real arm, and the contrast would
partly measure edge strength rather than edge identity. Replace the weight column with
a permutation of the *retained* `rho` values:

```python
rng = np.random.default_rng(args.seed)
w_real = rho[np.where(keep)]                 # the retained strengths
w_null = rng.permutation(w_real)[:len(bin_)] # same multiset, wrong pairs
```

Assert before writing: `kn.sum() == keep.sum()`, `kn.sum(0)` equals `keep.sum(0)` as a
multiset (degree preservation per archaeon), and the sorted `w_null` equals the sorted
`w_real`. Write the assertions into the script, not just into a scratch check.

---

## 3. Prerequisite — the edge set must be built inside the fold

The edge set is the object under test, and as the pipeline stands it is estimated once
on all 140 samples and then shared by every fold. Refactor `main()` so the block from
`clr(...)` through `sparsify(...)` is a function:

```python
def discover_edges(X_rows, meta_rows, covariates, fdr, topk, perms, seed):
    """Returns (keep_mask, rho, q) estimated from the given rows only."""
```

Call it once per training fold. The frozen full-data edge set stays available as a
diagnostic arm, but **the headline numbers must come from per-fold discovery.**

Report, as a first-class result and not a footnote:
- the mean pairwise Jaccard overlap between the K per-fold edge sets,
- the number of edges retained in all K folds ("core edges"),
- the mean and SD of edge count across folds.

If the core edge set is a small fraction of the per-fold sets, the network is not
stable enough to carry a conclusion and that is the finding — say so, do not bury it.

---

## 4. Shared helper — `cross_domain_edges.py`

Write one module used by both models.

```python
def load_edges(path, features, mode="real"):
    """Read an edge CSV, keep only edges whose BOTH endpoints are in `features`,
    return a list of (bacterium, archaeon, rho) sorted by descending |rho|.
    Report and return the count of edges dropped for a missing endpoint."""

def check_edges(edges, features, topk):
    """Assert: no self-pairs; no duplicate unordered pairs; every archaeon has at
    most `topk` partners; every endpoint appears in `features`."""
```

Both functions must **raise** on violation, not warn. A silently dropped edge changes
the with-arm's parameter count and breaks the match with the null arm.

---

## PART A — Phylo-Spec

### A.1 What the model actually reads

Phylo-Spec never re-reads the tree inside `forward`. It reads two precomputed objects:

- `conv_order` — a Python list of `(children, parent)` tuples from `get_conv_order`,
- `node_weights` — a dict `name -> 1 - branch_length` from `calculate_node_weights`.

And `forward` walks that list with a plain loop:

```python
for children, parent in conv_order:
    child_feats = [feature_map[child] for child in children]
    combined_feat = torch.cat(child_feats, dim=2).float()
    for res_block in self.res_blocks:
        combined_feat = res_block(combined_feat)
    combined_feat = combined_feat * node_weights[parent]
    feature_map[parent] = combined_feat
    all_features.append(feature_map[parent])
```

**`feature_map` is never popped.** A node stays available after its parent has consumed
it, so a second merge may consume it again. The schedule is therefore a DAG, not a
tree, and appending a cross-domain merge does not create a cycle in anything the code
traverses. **No modified Newick file is needed or wanted — do not write one.**

### A.2 The injection

Between `get_conv_order` / `calculate_node_weights` and the call to
`calculate_fc1_input_dim`:

```python
species_to_leaf = {v: k for k, v in leaf_to_species.items()}   # conv_order uses TREE names
for i, (bac, arc, rho) in enumerate(edges):
    lb, la = species_to_leaf.get(bac), species_to_leaf.get(arc)
    if lb is None or la is None:
        raise KeyError(f"edge endpoint absent from leaf_to_species: {bac}, {arc}")
    name = f"syn_{i}"
    assert name not in node_weights
    conv_order.append(([lb, la], name))       # APPEND AT THE END
    node_weights[name] = float(abs(rho))
```

Three hard requirements:

1. **Append at the end.** The loop has no topological sort; it trusts the order it is
   given. Appending last guarantees both children already exist in `feature_map`
   (leaves are populated in Step 1, internal nodes by the preceding merges).
2. **Add the `node_weights` entry.** `node_weights[parent]` is an unguarded dict
   lookup and will raise `KeyError` otherwise.
3. **Call `calculate_fc1_input_dim` *after* appending**, so `fc1` is sized for the
   extended schedule. It already derives the dimension by running the auxiliary model
   on the actual `conv_order`, so no arithmetic on your part is needed.

Use `abs(rho)` and let the sign be learned. `node_weights` for real nodes is
`1 - branch_length`, so values sit near 1 for short branches; `|rho| in [0,1]` is on
the same scale and in the same direction (stronger coupling → stronger pass-through).
Print the min/max of `node_weights` over both real and synthetic nodes; if any real
weight is negative (`branch_length > 1`), report it and stop — that is a tree problem,
not an injection problem.

### A.3 The parameter-count confound — this is the part people get wrong

`all_features` collects every leaf, every unmatched column, **and every merge output**,
concatenates along `dim=2`, and max-pools by 2 before `fc1`. Each pair merge appends a
length-2 block, so `fc1_input_dim` grows by one pooled unit per edge. At ~480 taxa with
mean leaf depth ~9 this takes `fc1` from roughly 38,400 to 40,000 inputs for 100
edges — about 205k extra weights.

**The without-network arm must therefore append the same number of merges**, drawn from
`{out}_edges_null.csv`. Otherwise the with-arm is simply the larger model and the
contrast measures capacity, not syntrophy.

Assert this explicitly before training:

```python
assert fc1_dim["plus_network"] == fc1_dim["plus_null"], "arms are not parameter-matched"
assert len(edges_real) == len(edges_null)
```

### A.4 Arms

| arm | `conv_order` | purpose |
|---|---|---|
| `tree_only` | unmodified | baseline geometry |
| `plus_network` | + N real merges, weight `abs(rho)` | the hypothesis |
| `plus_null` | + N rewired merges, weight = permuted `rho` | parameter- and strength-matched control |

Optional fourth arm if time allows: `plus_network_unweighted`, all synthetic weights
set to 1.0. It separates "the edge exists" from "the edge is this strong". Report it
separately; do not substitute it for `plus_null`.

### A.5 Blockers in the bundled code — fix before any arm is run

1. **Label indexing.** `forward` uses `data.columns.get_loc(species) - 1` and iterates
   `data.columns[1:-1]`, which assumes column 0 is the sample id and exactly one
   trailing label column. You have five binary targets. Minimal fix: keep `data` in the
   `[sample_id | features | one dummy column]` shape so the indexing is untouched, and
   pass the 140 × 5 label matrix separately to the loss. Do **not** widen `data` and
   patch the slices by hand in three places.
2. **`out_feature`.** `PhyloSpec(..., out_feature=1)` is the default. Set
   `out_feature=5` and use `BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(y))`.
   There is no softmax in `forward`, so logits go straight to the loss — correct as is.
3. **Lazily created leaf layers.** `self.conv1x1_layers[layer_name] = nn.Conv1d(...)`
   runs **inside `forward`**. Any parameter created there after
   `optim.Adam(model.parameters())` is constructed will never be optimised. Run one
   dummy forward pass on a single batch, then build the optimiser. Verify with
   `assert len(list(model.parameters())) == len(optimizer.param_groups[0]['params'])`.
4. **Substring leaf matching.** `match_leaf_nodes` does `if species in leaf_name` with a
   `break` on first hit — a feature name that is a substring of another tip's name
   mis-maps silently. Use the `--safe-ids` feature identifiers from the reduction step,
   and assert `len(leaf_to_species) == len(set(leaf_to_species.values()))` and that it
   covers every column of `data`.
5. **`tree_p` vs `match_leaf_nodes` disagree.** `tree_p` prunes with exact
   `search_nodes(name=feature)` while `match_leaf_nodes` matches by substring. Make both
   exact. Report any feature `tree_p` prints as "not found in tree" — do not let that
   print scroll past.

### A.6 Structural limitation to state in the report

`self.res_blocks` is a single shared `ModuleList` applied at every merge. A cross-domain
merge therefore reuses the same convolution weights as a within-domain merge; the model
can weight syntrophy differently but cannot learn a *different operator* for it. Say this
in the report. It bounds what a positive result means.

---

## PART B — MetaDR

### B.1 What the model actually reads

The tree's entire contribution is the column order:

```python
level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
taxa_level  = [i for i in level_order if i in X.columns]
```

No branch length is read anywhere in the file. `transform_image` then pads the row to a
square, reshapes to `img_size × img_size`, optionally zigzags odd rows, log-transforms,
and quantile-bins to 10 levels. **So the injection is a column permutation and nothing
else.** Do not modify the tree, the architecture, or the transform.

### B.2 Do the placement in 2-D, not as a 1-D seriation

A 1-D ordering gives each taxon two horizontal neighbours, so `--topk 5` cannot be
expressed. The first conv is 5×5, so its footprint reaches offsets 0, ±1, ±2 within a
row and ±`img_size`, ±2·`img_size` across rows — 25 cells. Place taxa on the grid
directly:

```python
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment

# W: p x p symmetric similarity. Within-domain: convert patristic distance to
# similarity with the same transform used in shortcut_distance. Cross-domain:
# the retained edges, weighted by rho. Zero elsewhere.
L = np.diag(W.sum(1)) - W
_, vecs = eigsh(L, k=3, which="SM")
Y = vecs[:, 1:3]
n = int(np.ceil(len(taxa) ** 0.5))
Y = (Y - Y.min(0)) / np.ptp(Y, 0) * (n - 1)
cells = np.array([(r, c) for r in range(n) for c in range(n)], float)
cost = ((Y[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
rows, cols = linear_sum_assignment(cost)
order = [None] * (n * n)
for i, g in zip(rows, cols):
    order[g] = taxa[i]
```

Padding cells (`n*n - p` of them) stay `None` and become zeros. Assign them to the
**lowest-degree** taxa's neighbourhood, i.e. let the assignment place them last — do not
scatter them through the middle of the grid.

Report a placement quality number: the fraction of retained edges whose two endpoints
land within one 5×5 footprint of each other. If that fraction is low, the injection did
not happen and the arm is uninformative regardless of its score.

### B.3 Which branch to inject into

MetaDR trains two models per fold and averages their probabilities — level-order without
zigzag, post-order with zigzag. Put the placement in the **post-order/zigzag branch**,
because zigzag is what makes 1-D adjacency map to 2-D locality across row wraps, and keep
the level-order branch unchanged as an internal tree control. This makes the with/without
contrast tighter than swapping both branches.

### B.4 Arms

| arm | zigzag branch order | level-order branch |
|---|---|---|
| `tree_only` | post-order traversal | unchanged |
| `plus_network` | 2-D placement on the fused similarity | unchanged |
| `plus_null` | same placement algorithm, run on the rewired graph | unchanged |

The null arm uses the identical algorithm on `{out}_edges_null.csv`, so image statistics,
padding count, and bin occupancy are matched.

### B.5 What cannot be carried — state this in the report

A permutation has no weights. **ρ magnitude and ρ sign are both discarded**; only
adjacency survives, and degree is capped at roughly 8 partners inside the footprint. This
is the weakest of the injection channels. A null result here must be reported as
uninformative about the hypothesis, not as evidence against it.

### B.6 Blockers in the bundled code — fix before any arm is run

1. **The script cannot run.** `parser.add_argument('-c', '--abundance', ...)` sets the
   attribute `abundance`, but the code reads `args.c` and `args.t`. `AttributeError` on
   first use. Whatever numbers exist for MetaDR were not produced by this file.
2. **`epochs = 1`, full-batch.** One gradient step per model. Set a real budget with
   mini-batches and early stopping on an inner validation split drawn from the training
   folds only.
3. **`nn.Linear(1800, 500)` is hard-coded.** It admits only `img_size` 22–25, i.e.
   **442–625 features**. Compute `img_size` from your feature count and either confirm it
   lands in range or replace the layer with a dimension derived from a dummy forward
   pass. Do not silently reshape the data to fit.
4. **Quantile bins are fitted on all samples.** `np.quantile` runs over the full flattened
   array — every sample, padding zeros included — before the CV split. Fit the bins on
   training rows only and apply them to held-out rows.
5. **In-place binning aliases.** The loop assigns `new_X[mask] = color_vals[i]` in
   ascending order, and `color_vals` (0.1 … 1.0) overlaps the range of the log-transformed
   values being binned, so a pixel already assigned can be re-captured by a later bin.
   Write to a separate output array.
6. **The top bin is half-open.** `(new_X >= low) & (new_X < high)` leaves the maximum
   pixel unbinned, retaining its raw log value. Make the last bin closed.
7. **`StratifiedKFold` on a single `LabelEncoder` column.** Replace with `site_grouped_cv`
   and the five-target head, as below.

---

## 5. Evaluation — identical across every arm of both models

Use `evaluate_multilabel.py`. No arm may define its own splits, thresholds, or metrics.

- **Splitting:** `site_grouped_cv`. Samples are seasonal repeats of a smaller number of
  sites and are not independent — **all seasons of a site go to one side of every
  split**. This applies to the inner validation split too.
- **Loss:** `BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(y_train))`, computed on
  training rows only.
- **Thresholds:** `tune_thresholds_mcc` per target, on inner-validation rows only.
- **Metrics:** macro F2, macro average precision, macro MCC, plus the per-target table.
- **Seeds:** 5 seeds per arm. Report mean ± SD across seeds and folds, and a paired
  bootstrap CI for `plus_network − plus_null` within each model.

The `plus_network` vs `plus_null` contrast is the result. `tree_only` is context.

---

## 6. Deliverables

1. `cross_domain_edges.py` — the shared loader and validators.
2. Patched `build_cross_domain_graph.py` — null edge CSV with permuted weights, and
   `discover_edges()` factored out for per-fold use.
3. `run_phylospec_arms.py` — three (or four) arms, one results table.
4. `run_metadr_arms.py` — three arms, one results table.
5. `results_cross_domain_injection.csv` — one row per (model, arm, seed, fold) with all
   metrics, plus the fold-level edge-set statistics from §3.
6. Two figures:
   - `fig_arm_comparison.png` — macro F2 and macro AP per arm per model, points for each
     seed with a median tick, `plus_network` visually dominant, `plus_null` adjacent.
   - `fig_edge_stability.png` — per-fold edge counts, Jaccard overlap matrix between the
     K edge sets, and the distribution of `rho_conditioned` against
     `patristic_distance` for core edges.
7. `REPORT_cross_domain_injection.md` — the arm table, the edge-stability numbers, the
   placement-quality fraction for MetaDR, the `fc1` dimensions proving the Phylo-Spec
   arms are parameter-matched, and every limitation named in §A.6 and §B.5.

---

## 7. Prohibitions

- **Do not write a modified Newick file for Phylo-Spec.** The injection is in
  `conv_order`. If you find yourself serialising a graph, you have taken a wrong turn.
- **Do not reconstruct a tree from the fused distance matrix.** A reconstruction differs
  from the original everywhere, and the contrast would confound the injection with
  reconstruction error.
- **Do not run a with-network arm without its parameter-matched null.** A with-vs-
  baseline difference alone is not evidence; the model got bigger.
- **Do not let the arms differ in anything but the injection** — same features, same
  splits, same seeds, same loss, same thresholds, same epoch budget.
- **Do not pass `eff_*` or `warning_*` columns to `--covariates`.** The script now
  refuses them. Do not use `--allow-outcome-covariates` to get past it.
- **Do not select `--scale`, `--topk`, or `--fdr` by test performance.** Inner CV or
  fixed a priori, and state which.
- **Do not silently drop an edge whose endpoint is missing from the feature list.**
  Raise, report the count, and ask before proceeding.
- **Do not fuzzy-match feature names to tip labels.** Exact match or stop.
- **Do not report a MetaDR null result as evidence against cross-domain structure.** Its
  channel discards edge weight and caps degree; it can only fail to detect.
- If any assertion in §2, §4, or §A.3 fires, **stop and report** rather than relaxing it.

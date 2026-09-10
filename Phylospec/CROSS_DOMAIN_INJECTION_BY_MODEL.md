# Injecting the Bacteria × Archaea network into DeepPhylo, Phylo-Spec and MetaDR

`build_cross_domain_graph.py` produces one object: a fused **distance matrix**
`D_fused` over the kept features, in which retained syntrophic pairs have been pulled
together by an added branch and all other distances re-derived as geodesics.

Only DeepPhylo consumes anything derived from a distance matrix. Phylo-Spec and MetaDR
consume the Newick tree, in two different and incompatible ways. So "with syntrophy"
has to be *re-expressed* per model, and the three expressions are not equally faithful.
This file specifies each one.

Common rule for all three arms: the **without-syntrophy** arm must be the same code
path with the injection parameter set to its neutral value, never a separate script.

---

## 1. DeepPhylo — drop-in

**What it eats.** `X.npy` (n_samples × n_taxa), `y.npy`, `embeding.npy`
(n_taxa × d). The embedding is loaded as `nn.Embedding.from_pretrained(..., freeze=True)`,
passed through a `Linear(d → hidden)`, transposed, then a `Conv1d(kernel=7)` runs
**along the taxon axis**.

**Injection.** Replace the patristic PCA with the fused PCA. Nothing else changes:

```
without:  np.save("emb.npy", embed(Dp,      dim))   # scale = inf
with:     np.save("emb.npy", embed(Dfused,  dim))   # scale = 0.25
```

Both are already emitted by `build_cross_domain_graph.py` as
`<p>_treeonly_DeepPhylo_embeding.npy` and `<p>_DeepPhylo_embeding.npy`.

**Two things to get right.**

- The `Conv1d` over the taxon axis means the **column order of `X` is itself a prior**.
  Keep columns in tree post-order in both arms, and keep it *identical* between arms —
  otherwise you have changed two things at once.
- `input_size = embedding.weight.size(0) - 1` in the model implies a reserved row 0.
  Confirm how `main_all.py` builds its `ids` tensor before trusting the alignment;
  `make_deepphylo_inputs.py` and `build_cross_domain_graph.py` both emit exactly
  `n_taxa` rows, with no pad row.
- `embed()` is a plain PCA of the distance matrix, not classical MDS (no double
  centring). This is a pre-existing quirk, applied identically to both arms, so it does
  not confound the contrast — but classical MDS is the more defensible transform if you
  are reporting the embedding itself.

---

## 2. Phylo-Spec — the shortcut cannot be handed over

**What it eats.** The `.nwk` file. `data_processing` derives a post-order traversal
(`conv_order`) and per-node scalars (`node_weights`); the model multiplies each leaf's
feature map by `node_weights[leaf]` and each accumulated internal feature map by
`node_weights[parent]`. Its entire inductive bias is **strictly nested clade
aggregation**.

**Why the fused object is unusable.** Adding a bacterium–archaeon branch creates a
cycle. A cyclic graph has no post-order and no Newick serialisation. Do **not** solve
this by running neighbour-joining or UPGMA on `D_fused` to get a tree back: the
re-derived tree differs from the original everywhere, so the tree-only baseline stops
being nested and any difference you measure is confounded with "NJ reconstruction
error".

**Two legitimate arms, which test different things.**

*Arm 2a — node weights (matched prior, weak channel).* Keep the tree exactly. For each
internal node `v`, compute the fraction `f(v)` of retained cross-domain edges that have
at least one endpoint inside `clade(v)`, and set

```
w'(v) = w(v) * (1 + alpha * f(v))          alpha = 0  ->  baseline, bit-for-bit
```

This is nested and cheap. Be aware of its limit: the smallest clade containing both
members of a cross-domain pair is their LCA, which for any Bacteria × Archaea pair is
very deep — under a domain graft it is the invented root itself, making the signal
degenerate. Under a SEPP-built tree it is a real node but still near the base. So this
arm carries genuinely less information than DeepPhylo's, and a null result here is
weak evidence.

*Arm 2b — edge features (strong channel, different experiment).* For each retained
edge `(b, a)`, append one column to the abundance CSV holding a per-sample partnership
activity, e.g. `min(clr_b, clr_a)` or the product of the two relative abundances. These
are appended as extra tip-less features, so the tree is untouched and the without arm
simply omits the columns.

This changes the **feature set**, not the prior, so it is not the same experiment as
DeepPhylo's. Report it as its own arm and say so; do not pool it with 2a.

**Pre-existing code issues to fix first** (from `model_input_requirements.csv`):
`data_processing.py::tree_p()` is missing `import tempfile` and
`from ete3 import PhyloTree`; `match_leaf_nodes()` uses substring matching, so numeric
feature IDs cross-match. Use the `--safe-ids` output of
`reduce_features_for_tree_models.py`.

---

## 3. MetaDR — the prior is only an ordering

**What it eats.** An abundance CSV and a `.nwk` (ete3 `format=1`). The tree is used
exactly once, and only for its tip sequence:

```
level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
post_order  = [leaf.name for leaf in tree.traverse("postorder")  if leaf.is_leaf()]
```

Columns are reordered to that sequence, zero-padded to `ceil(sqrt(p))**2`, reshaped into
a square, log-transformed base 4, quantile-binned into 10 levels, and fed to a 2-D CNN.
**No branch length ever reaches the model.** The prior is purely which features end up
adjacent in the folded square.

**Injection.** Convert `D_fused` into an ordering rather than a weighting: hierarchical
clustering (average linkage) on `D_fused` with **optimal leaf ordering**
(`scipy.cluster.hierarchy.optimal_leaf_ordering`), then use that leaf sequence as the
column order.

```
without:  column order = post-order traversal of the tree
with:     column order = optimal-leaf-ordering seriation of D_fused
```

Because the shortcut shortens exactly the syntrophic pairs, seriation places those
partners near each other in the sequence, and the 2-D fold puts them in the same
convolution window. That is the only channel MetaDR has.

Note the fold makes adjacency anisotropic: features consecutive in the sequence are
row-neighbours, but features `sqrt(p)` apart become column-neighbours, which is
arbitrary. Set `zigzag=True` so row transitions are continuous, and use the same setting
in both arms.

**Bugs that will stop the script before any of this matters.**

- `abundance_file = args.c` / `tree_file = args.t` — argparse defines `-c/--abundance`
  and `-t/--tree`, so the attributes are `args.abundance` and `args.tree`. As written
  this raises `AttributeError` on the first run.
- `epochs = 1` is hard-coded. MetaDR is effectively untrained; benchmarking against it
  at this setting measures nothing. Raise it and use the same budget in both arms.
- `nn.Linear(1800, 500)` is hard-coded to one feature count and must be recomputed for
  your `p`.
- The quantile bins are computed on the flattened array *including the zero padding*, so
  the padding shifts every bin edge. With `p` far from a perfect square this is
  substantial. Compute quantiles on the unpadded values.

---

## 4. What this means for the benchmark

The three arms inject the same network through three channels of very different
bandwidth: frozen per-taxon vectors (DeepPhylo), a scalar per internal node
(Phylo-Spec 2a), and a permutation of the feature axis (MetaDR). Therefore:

- **Valid comparison:** within a model, with vs without. This is what the nested
  parameterisation buys you.
- **Not valid:** attributing a DeepPhylo-vs-MetaDR difference to the syntrophy
  information. The channels differ, the models differ, and the two are not separable
  in this design.
- **Reporting:** state the injection mechanism per model in the results table, and treat
  a null in Phylo-Spec 2a or MetaDR as uninformative about syntrophy rather than as
  evidence against it.

Evaluate every arm with `evaluate_multilabel.py` — site-grouped folds, per-target MCC
thresholds, macro-averaged MCC and F2 — so the five imbalanced targets are scored
identically across arms.

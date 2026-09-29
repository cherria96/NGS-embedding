# Data Investigation Summary

Source notebook: `data/data_investigation.ipynb`. All statistical tests below use
999-permutation significance tests and Spearman (rank) correlation throughout.

---

## 1. Microbial community × chemistry correlation (genus-level, top 30)

Spearman correlation between every ARC/BAC genus and every `Sub`/`M` chemistry
parameter (9,724 pairs tested, requiring ≥15 paired samples), ranked by |ρ|.

- **All 30 top hits are BAC genera** — BAC has far more taxa than ARC (179 vs.
  25), so it dominates the ranking even before considering effect size.
- Strongest associations (|ρ| ≈ 0.68–0.76) are with **sCOD** and **ALK** —
  e.g. `Syntrophorhabdus` / `Fermentimonas` vs. sCOD (ρ ≈ ∓0.76),
  `Aminicenantales` / `Macellibacteroides` / `Zoogloea` / `Exilispira` vs. ALK.
- Biologically sensible: sCOD/ALK-correlated genera are known syntrophic
  acetogens / VFA-degraders in anaerobic digestion systems, consistent with
  their expected functional role.
- The 21 unique genera behind these 30 pairs split into **two clearly
  anti-correlated blocks** (13 vs. 8 genera) in the genus–genus abundance
  heatmap.

---

## 2. Are those two abundance clusters also phylogenetically distinct?

The same 13-vs-8 cluster split (derived from abundance-correlation
clustering) was tested with **ANOSIM** (Clarke 1993), once against
correlation-distance and once against phylogenetic (patristic) distance.

| distance metric | within-group | between-group | ANOSIM R | p |
|---|---|---|---|---|
| abundance correlation (1 − ρ) | mean ρ = 0.632 | mean ρ = −0.596 | **1.000** | 0.001 |
| phylogenetic (patristic) | mean dist = 0.504 | mean dist = 0.556 | **0.185** | 0.040 |

**Interpretation**: the co-abundance split is essentially perfect (R = 1.0) —
by construction, since it *is* the clustering criterion. On phylogeny, the
same two groups are only weakly, if nominally significantly, separated. So
the two co-abundance blocks are **not simply explained by shared ancestry**
— the tight multicollinearity between genera in the same block is more
likely driven by shared ecological niche or function across less-related
lineages than by common descent.

---

## 3. Mantel test: relative abundance vs. phylogenetic distance, by rank

Tests whether phylogenetically closer taxa have more correlated abundance
profiles across samples, run separately per domain and per taxonomic rank.

| domain | ASV | Class | Order | Family | Genus |
|---|---|---|---|---|---|
| **ARC** | r=0.119, p=0.024 ✱ | r=0.395, p=0.001 ✱ | r=0.395, p=0.001 ✱ | r=0.295, p=0.001 ✱ | r=0.276, p=0.001 ✱ |
| **BAC** | r=0.002, p=0.872 | r=0.008, p=0.839 | r=0.014, p=0.625 | r=0.008, p=0.747 | r=-0.004, p=0.815 |

✱ significant at p < 0.05

**Interpretation**:
- **Archaea show a real, significant phylogenetic signal at every rank** —
  strongest at Class/Order (r ≈ 0.40), tapering somewhat toward Genus
  (r ≈ 0.28). Archaeal community structure (dominated by a handful of
  methanogen lineages) is phylogenetically conserved: relatedness predicts
  co-abundance.
- **Bacteria show no phylogenetic signal at any rank** (r ≈ 0, always
  non-significant) — the much larger, functionally diverse bacterial
  community's abundance patterns are essentially independent of phylogeny.
- This ARC-vs-BAC contrast holds at every resolution tested, and reinforces
  finding #2: where phylogenetic signal does exist (archaea), it is real but
  distinguishable from the pure co-abundance clustering seen among BAC's
  chemistry-linked genera.

---

## Appendix: how each rank's tree was built

All trees start from the same source: each domain's raw QIIME2 outputs in
`data/qiimeresult/{ARC,BAC}/` — `rooted-tree.nwk` (the ASV-level phylogeny)
and `metadata.tsv` (ASV taxonomy, despite the filename). Everything above
Genus/ASV is a **tree-aware collapse** of that same ASV tree, not a
naive collapse-by-taxonomy-string — a taxon that is scattered across the
tree gets split into its maximal monophyletic blocks (e.g.
`Methanolinea__b1`, `__b2`, ...) instead of being merged into a group with
no defensible position on the tree.

| level | tree used | how it's built |
|---|---|---|
| **ASV** | `data/qiimeresult/{domain}/rooted-tree.nwk` | raw QIIME2 fragment-insertion tree, no collapsing |
| **Genus** | `output/genus_tree/genus_autorun/{domain}_tree.nwk` | `run_genus_tree_pipeline.py --mode genus` → `Phylospec/reduce_features_for_tree_models.py --mode genus --rank genus` |
| **Class / Order / Family** | `output/genus_tree/taxlevel_autorun/{class,order,family}/{domain}_tree.nwk` | same script, `--rank {class,order,family}` |

The collapsing script (`reduce_features_for_tree_models.py`) does, per rank:

1. Prevalence/abundance-filters ASVs (archaea get a protected, more lenient
   threshold via `--protect archaea` so low-abundance methanogens survive a
   bacteria-tuned filter).
2. Prunes the ASV tree down to the surviving ASVs.
3. Groups ASVs by their taxonomic label **at the chosen rank**, then finds
   each group's maximal monophyletic blocks on the pruned tree.
4. Collapses each monophyletic block into a single tip (summing member
   abundances); a non-monophyletic label becomes multiple tips
   (`Label__b1`, `Label__b2`, ...). ASVs unclassified at that rank are kept
   as their own tips — "unclassified" is not a clade, so they are never
   lumped together.

The `--rank` option (added this session) generalizes what was previously a
genus-only collapse; `--rank genus` (the default) reproduces the pre-existing
`genus_autorun` output byte-for-byte, so nothing that already depended on it
changed. Command used for the shallower ranks:

```bash
python Phylospec/reduce_features_for_tree_models.py \
  -c output/genus_tree/genus_autorun/{domain}_input_abundance.csv \
  -t data/qiimeresult/{DOMAIN}/rooted-tree.nwk \
  -x output/genus_tree/genus_autorun/{domain}_input_taxonomy.csv \
  -o output/genus_tree/taxlevel_autorun/{rank}/{domain} \
  --mode genus --rank {class,order,family} --safe-ids --protect archaea
```

Resulting feature (tip) counts per domain × rank:

| domain | Class | Order | Family | Genus |
|---|---|---|---|---|
| ARC | 21 | 21 | 34 | 42 |
| BAC | 102 | 159 | 242 | 389 |

Section 2's phylogenetic-distance test above works differently again: it
needs distances between the specific 21 *named* genera from the correlation
heatmap, most of which aren't tree tips in `genus_autorun` (that tree's tip
set is filtered/named independently). Instead, each genus's full ASV set is
pulled directly from `qiimeresult/BAC/metadata.tsv`'s taxonomy strings
(`g__<genus>`), and genus-to-genus distance is the **mean pairwise patristic
distance** (on the raw ASV-level `rooted-tree.nwk`) across every ASV pair
between the two genera — i.e. an average-linkage distance computed directly
from the ASV tree, not a pre-built genus tree.

# Experimental plan

## Dataset

34 anaerobic digestion (AD) sites × 4 seasons (Summer/Fall/Winter/Spring), delivered as separate
bacterial and archaeal QIIME2 artifacts (filtered feature table + SILVA taxonomy + rooted tree
each). Bacteria and archaea align to 138 paired samples (two bacteria-only samples, `1-32'` and
`4-24`, have no archaeal partner and drop out; a few `b`/`'` replicate facilities are retained
within their site). Raw dimensionality is large — ~13,385 bacterial ASVs and ~1,776 archaeal
ASVs — so aggressive prevalence filtering is applied per training fold (≥15% prevalence ⇒
≈450 BAC + ≈70 ARC features).

Sample IDs follow the `{round}-{No}` convention used throughout `data/fastq/`, `data/qiimeresult/`,
and `data/final/metadata.csv` (e.g. `1-2` = round 1 (Summer), site No. 2 (BSN); `2-32'` = round 2
(Fall), site No. 32' (JJY)).

### Raw source: `data/final/Dat_chem.xlsx`

Physicochemical characterization of each digester's influent substrates and effluent, sheet by sheet:

- **`Si_1R`–`Si_4R`** (Summer/Fall/Winter/Spring anaerobic digester input): one row per substrate
  fed to a site's digester that season. Where a site is fed multiple substrates (e.g. site `BSN`
  is fed `1SC`, `2SC`, `Fww`), each substrate gets its own row; the `Site` column is only filled on
  the first row of the group. The trailing `"{site} avg."` row is the Q-weighted average across
  that site's substrates and is excluded from downstream processing (it's derived, not raw data).
- **`Se`**: digester effluent physicochemical results. Row 2 (`S`/`F`/`W`/`Sp`) labels the season
  block for each parameter; season maps to round as `S`=`Si_1R`, `F`=`Si_2R`, `W`=`Si_3R`,
  `Sp`=`Si_4R`.
- **`ADPs_140Si,Se,분석`**: reference table of which substrate type is fed to each digester in each
  season (columns I–L) and the substrate-name → acronym legend (columns T–U, e.g. `1S(C)` =
  primary sludge, `Fww` = food wastewater, `HM` = human manure).

## `data/final/metadata.csv`

Tidy, per-site-per-season chemical feature table built from `Dat_chem.xlsx`, meant to be joined to
the microbiome tables on `SampleID` and used as ML features/covariates. 140 rows × 108 columns
(35 site codes × 4 seasons, minus 2 seasons a site couldn't be visited due to dredging works, plus
2 one-off QC replicate rows that only exist for Summer). `std`, `Cation`, `Anion`, `Pellet`, `DNA`,
`NGS`, `QPCR` columns and the `"{site} avg."` rows from the source sheets are excluded.

Columns:

| Group | Columns | Source |
|---|---|---|
| Identity | `SampleID`, `Round`, `Season`, `Site`, `No` | `Se` (`No`/`Site`), derived (`SampleID` = `Round`-`No`) |
| Digester operation | `HRT_d`, `T_C` | `Se` |
| Feed summary | `Q_total_Tpy` (sum of that season's substrate flows), `substrate_type` (e.g. `1SC/2SC/Fww`) | derived from `Si_*R` |
| `substrate1_*` … `substrate3_*` | `name`, `Q`, `pH`, `ALK`, `TS`, `VS`, `TSS`, `VSS`, `COD`, `sCOD`, `TC`, `Protein`, `Lipid`, `TVFAs`, `TVFAs_plus_EtOH`, `EtOH`, `HAc`, `HPro`, `i_HBut`, `HBut`, `i_HVal`, `HVal`, `i_HCap`, `HCap` | `Si_1R`–`Si_4R`, one block per substrate fed that site/season (blank if the site had fewer substrates that season) |
| `eff_*` | same physicochemical panel as substrates, plus `TKN`, `TAN`, `CO2`, `CH4`, `H2S`, `O2` | `Se` |

Notes:
- Sites `ADP` and `YCG` are missing their Spring row (facility inaccessible; flagged `x` in `ADPs_140Si,Se,분석`).
- `BSIb` and `DGYb` (QC replicate facilities) only have a Summer row — the source notes them as excluded from Fall onward.
- Empty cells are blank strings, not `0`/`NaN` — impute/cast before model training.

### Genomic data

- **`data/final/Dat_ARC.xlsx`, `data/final/Dat_BAC.xlsx`**: 16S rRNA taxonomic classification and
  relative-abundance summaries derived from the QIIME2 ASV tables (archaeal and bacterial
  domains, respectively). Sheet `OTUs` lists each ASV's `Feature ID` and full lineage
  (`Domain`→`Species`). For each taxonomic rank (`P`=Phylum, `C`=Class, `O`=Order, `F`=Family,
  `G`=Genus, `S`=Species) there are three wide sheets, one row per taxon and one column per
  `SampleID` (e.g. `1-1-BSSG`): `{rank}_read` (read counts), `{rank}(%)` (relative abundance
  within sample), and `{rank}_rank(%)` (percentile rank within sample).
- **`data/qiimeresult/ARC/`, `data/qiimeresult/BAC/`**: the underlying QIIME2 analysis artifacts
  that `Dat_ARC.xlsx`/`Dat_BAC.xlsx` were summarized from (per domain). Key outputs:
  - `table_filtered.qza` — prevalence-filtered ASV feature table (the ASV × sample count matrix).
  - `silva_16S_taxonomy.qza`/`.qzv` — SILVA-based taxonomic classification per ASV.
  - `rooted-tree.qza` — rooted phylogenetic tree over the ASVs, used by the hierarchy-aware models.
  
  The directories also retain the intermediate DADA2/denoising and QC artifacts
  (`dada2_table.qza`, `dada2_rep_seqs.qza`, `dada2_stats.qzv`, `alpha_rarefaction.qzv`,
  `primer_trimmed.qzv`, `seq_filtered.qza`, `aligned-rep-seqs.qza`, `unrooted-tree.qza`, etc.)
  from the QIIME2 pipeline run.

## Experimental plan

### Three-tier comparison

| Tier | Model(s) | Encodes | Named gap |
|---|---|---|---|
| Baseline | RF / MLP on single-rank abundance + covariates | one taxonomic level | drops other levels → information loss + rank-choice bias |
| Hierarchy-aware | PopPhy-CNN, DeepPhylo, Phylo-Spec | within-tree structure | ignores cross-domain bacteria↔archaea syntrophy |
| Proposed | dual-domain hierarchy encoder + syntrophic cross-attention | within-tree and cross-domain syntrophic interface | — |

### Benchmark reimplementations

Faithful but deliberately small reimplementations, used as the hierarchy-aware tier:

- **PopPhy-CNN** (Reiman 2020): tree → 2D matrix populated with abundance → 2D-CNN.
- **DeepPhylo** (Wang 2024): per-OTU embeddings = PCA of the patristic-distance matrix
  (evolutionary distance, not just topology); dual abundance + conv modules.
- **Phylo-Spec** (Zhang 2025): bottom-up fusion of per-rank abundance features.

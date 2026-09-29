"""Manually curated role hypotheses for the order blocks that carry signal.

Curated 2026-09-23. Rules kept from the earlier exp5 curation:

* Annotation is keyed on the **block**, not the order name, because the
  monophyly-consistent rank tree splits several orders into blocks with
  completely different genus content (``Bacteroidales__b9`` is Proteiniphilum
  only; ``Bacteroidales__b4`` is a 53-ASV mixture). Transferring one block's
  role to its order would be wrong.
* ``role`` is a hypothesis about the *named cultured relatives in that block*,
  never a measurement. These are 16S amplicons: no gene, transcript, pathway
  or flux was observed, and DNA from inactive immigrant cells is amplified
  exactly like DNA from active residents.
* ``basis`` states what kind of evidence the source is. ``expectation`` means
  no primary source was retrieved for this block in this session and the role
  is a taxonomic expectation only; it is reported, not hidden.
* Uncultured lineages with no cultured representative keep ``unassigned``.
  A candidate-phylum label is not a function.
"""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))


# (domain, block_label, role, basis, source, note)
EVIDENCE = [
    # --- Archaea ------------------------------------------------------------
    ("ARC", "Methanomicrobiales", "hydrogenotrophic methanogenesis", "isolate genome + community evidence",
     "https://journals.asm.org/doi/10.1128/genomeA.00876-14",
     "Block is Methanospirillum/Methanoculleus/Methanolinea/Methanosphaerula/Methanoregula/Methanocorpusculum. "
     "The Methanolinea tarda genome supports hydrogenotrophy and formate use; Methanoculleus becomes prominent "
     "under high ammonium in digester communities (PMC8510171)."),
    ("ARC", "Methanosarciniales__b2", "acetoclastic methanogenesis", "culture/community evidence",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/",
     "13 Methanosaeta (Methanothrix) ASVs. Obligate acetoclastic; reported as the ammonia-sensitive end of the "
     "methanogen spectrum."),
    ("ARC", "Methanosarciniales__b3", "acetoclastic methanogenesis", "culture/community evidence",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/", "Single Methanosaeta ASV separated by the tree."),
    ("ARC", "Methanosarciniales__b4", "acetoclastic methanogenesis", "culture/community evidence",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/",
     "Single Methanosaeta ASV separated from block b2 by the rank tree. Methanosaeta-dominated reactors have the "
     "lowest reported resilience to free ammonia; Smithella, Methanosaetaceae and Methanospirillaceae were "
     "inhibited together in ammonia dosing of digester sludge (PubMed 30278382)."),
    ("ARC", "Methanosarciniales__b5", "versatile methanogenesis (acetoclastic, hydrogenotrophic, methylotrophic)",
     "culture/community evidence; species dependent", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/",
     "5 Methanosarcina + 3 Methanomethylovorans ASVs. Methanosarcina occupies the intermediate/tolerant position "
     "in ammonium perturbation experiments and is repeatedly described as the ammonia-robust methanogen; "
     "Methanomethylovorans is methylotrophic. Genus identity alone does not resolve which pathway is active here."),
    ("ARC", "Methanomassiliicoccales", "hydrogen-dependent methyl reduction", "culture evidence outside digesters",
     "https://pubmed.ncbi.nlm.nih.gov/22859731/",
     "Methanomassiliicoccus/Ca. Methanoplasma. A cultured representative reduced methanol with hydrogen; neither "
     "substrate alone supported methane formation."),
    ("ARC", "Methanofastidiosales", "hydrogen-dependent methylated-thiol reduction", "genome-inferred potential",
     "https://pubmed.ncbi.nlm.nih.gov/26943620/",
     "5 Ca. Methanofastidiosum ASVs. Reconstructed genomes support methylated-thiol reduction; activity of these "
     "ASVs is not measured."),
    ("ARC", "Methanomethyliales", "methyl-reducing methanogenesis", "genome-inferred potential", "",
     "2 Ca. Methanomethylicus ASVs. Expectation from the described lineage; no primary source retrieved here."),

    # --- Bacteria: proteolytic / amino-acid fermenting ----------------------
    ("BAC", "Bacteroidales__b9", "proteolytic fermentation; acetate utilisation under ammonia stress",
     "culture evidence + reactor community evidence", "https://pubmed.ncbi.nlm.nih.gov/16280479/",
     "5 Proteiniphilum ASVs. Proteolytic anaerobes isolated from a UASB reactor; Proteiniphilum and Methanothrix "
     "harundinacea were reported as dominant acetate utilisers under strong ammonia stress "
     "(PMC9853277). That report makes Proteiniphilum's direction here consistent with, not diagnostic of, ammonia stress."),
    ("BAC", "Bacteroidales__b1", "saccharolytic/proteolytic fermentation", "expectation", "",
     "6 Dysgonomonadaceae + Fermentimonas + 1 uncultured. Family-level expectation for fermentative Bacteroidota; "
     "no block-specific primary source retrieved."),
    ("BAC", "Bacteroidales__b4", "mixed fermentation", "expectation", "",
     "53 ASVs spanning Rikenellaceae RC9, Paludibacter, Prevotella, Bacteroides, Petrimonas, Parabacteroides and "
     "uncultured members. Deliberately not given a single role: this block is a taxonomic mixture."),
    ("BAC", "Clostridia__b7", "amino-acid fermentation", "expectation", "",
     "23 Fastidiosipila ASVs. Described members ferment amino acids; no block-specific primary source retrieved."),
    ("BAC", "Peptostreptococcales_Tissierellales__b1", "proteolytic and amine/amino-acid fermentation",
     "expectation", "",
     "25 ASVs: Anaerovorax, Tissierella, Tepidimicrobium, Guggenheimella, Alkaliphilus and unclassified "
     "Anaerovoracaceae. The named genera are described amino-acid/amine fermenters; block-level activity is not measured."),
    ("BAC", "Synergistales__b1", "amino-acid degradation", "culture/genome evidence outside digesters",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC3387794/",
     "Thermovirga + Acetomicrobium + uncultured Synergistaceae. The Thermovirga oil-well isolate uses amino acids "
     "and proteinaceous substrates; application to these ASVs is a hypothesis."),
    ("BAC", "Synergistales__b2", "amino-acid degradation", "expectation", "",
     "Aminobacterium, Lactivibrio, Syner-01 and uncultured Synergistaceae. Same family expectation as b1; the two "
     "blocks are separated by the tree and carry opposite coefficients in the acid-high model."),

    # --- Bacteria: syntrophs ------------------------------------------------
    ("BAC", "Syntrophomonadales", "syntrophic fatty-acid (butyrate) oxidation",
     "digester isotope/genome evidence", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6687939/",
     "10 Syntrophomonas + Syntrophothermus + uncultured. Stable-isotope-informed metagenomics identified "
     "syntrophic butyrate degraders; Syntrophomonas is also repeatedly recovered alongside syntrophic acetate "
     "oxidisers in ammonia-loaded systems (PMC9600251)."),
    ("BAC", "Syntrophales__b1", "syntrophic propionate oxidation", "culture + ammonia-dosing evidence",
     "https://pubmed.ncbi.nlm.nih.gov/30278382/",
     "12 Smithella + 3 Syntrophus + 3 uncultured. Smithella propionica is a propionate-oxidising syntroph; "
     "ammonia dosing of digester sludge inhibited Smithella together with Methanosaetaceae and Methanospirillaceae."),
    ("BAC", "Syntrophales__b2", "syntrophic oxidation (family-level)", "expectation", "",
     "3 uncultured Syntrophaceae ASVs; genus unresolved, so the propionate role of b1 is not transferred."),
    ("BAC", "Syntrophorhabdales", "syntrophic aromatic-compound oxidation", "expectation", "",
     "3 Syntrophorhabdus ASVs. Described members oxidise aromatic compounds in syntrophy; no block-specific "
     "primary source retrieved."),
    ("BAC", "Desulfotomaculales", "syntrophic propionate oxidation / sulfate reduction", "expectation", "",
     "Single Pelotomaculum ASV; described members oxidise propionate syntrophically."),
    ("BAC", "Kosmotogales", "saccharolytic Thermotogota; syntroph-associated", "expectation", "",
     "2 Mesotoga ASVs."),
    ("BAC", "Petrotogales", "thermophilic saccharolytic Thermotogota", "expectation", "",
     "Defluviitoga + SC103 + AUTHM297. Typically enriched in thermophilic digesters."),

    # --- Bacteria: other fermenters ----------------------------------------
    ("BAC", "Caldicoprobacterales__b1", "xylanolytic/proteolytic thermophilic fermentation", "expectation", "",
     "5 Caldicoprobacter ASVs."),
    ("BAC", "Caldicoprobacterales__b2", "xylanolytic/proteolytic thermophilic fermentation", "expectation", "",
     "4 Caldicoprobacter ASVs; separated from b1 by the tree."),
    ("BAC", "Acholeplasmatales__b1", "fermentative Mollicutes", "expectation", "", "7 Acholeplasma ASVs."),
    ("BAC", "Acholeplasmatales__b2", "fermentative Mollicutes", "expectation", "", "2 Acholeplasma ASVs."),
    ("BAC", "Christensenellales__b2", "fermentation", "expectation", "",
     "29 Christensenellaceae R-7 group ASVs. Frequently abundant in digesters and repeatedly named among "
     "uncultured Clostridia in syntrophic acetate-oxidising consortia (PMC4769498), but not identified to a "
     "cultured acetate oxidiser here; no acetate-oxidising role is assigned."),
    ("BAC", "Clostridia__b2", "cellulolytic/saccharolytic fermentation", "expectation", "",
     "2 Ruminiclostridium + 2 HN-HF0106 ASVs."),
    ("BAC", "Oscillospirales__b1", "fermentation", "expectation", "",
     "[Eubacterium] coprostanoligenes group, Ruminococcus, uncultured."),
    ("BAC", "Spirochaetales", "fermentation", "expectation", "", "Sphaerochaeta, Treponema, uncultured."),
    ("BAC", "Anaerolineales", "fermentative Chloroflexi", "expectation", "",
     "Anaerolineaceae, Longilinea, Flexilinea, ADurb.Bin120."),
    ("BAC", "Sphingobacteriales__b2", "fermentation", "expectation", "", "Lentimicrobiaceae/Lentimicrobium."),

    # --- Bacteria: aerobic / facultative, influent-associated ---------------
    ("BAC", "Burkholderiales", "aerobic/denitrifying activated-sludge taxa; likely influent-associated",
     "digester immigration evidence", "https://www.nature.com/articles/s41598-017-09303-0",
     "54 ASVs: Comamonadaceae, Rhodocyclaceae, Dechloromonas, Thauera, Zoogloea, SC-I-84. Several of the most "
     "abundant digester populations are inactive immigrants arriving with the feed, and inactive populations were "
     "associated with aerobes/facultative anaerobes. A coefficient on this block may track feed and retention "
     "rather than digester metabolism."),
    ("BAC", "Micrococcales", "activated-sludge Actinobacteriota; likely influent-associated",
     "digester immigration evidence", "https://www.nature.com/articles/s41598-017-09303-0",
     "5 Tetrasphaera + Intrasporangiaceae + Microbacterium + Galbitalea. Same immigration caveat as Burkholderiales."),
    ("BAC", "Rhodobacterales", "aerobic/facultative Alphaproteobacteria; likely influent-associated",
     "digester immigration evidence", "https://www.nature.com/articles/s41598-017-09303-0",
     "Rhodobacter, Paracoccus, Rhodobacteraceae."),
    ("BAC", "Xanthomonadales", "aerobic Gammaproteobacteria; likely influent-associated",
     "digester immigration evidence", "https://www.nature.com/articles/s41598-017-09303-0",
     "Thermomonas, Dokdonella, Ahniella."),
    ("BAC", "Microtrichales", "aerobic Actinobacteriota; likely influent-associated",
     "digester immigration evidence", "https://www.nature.com/articles/s41598-017-09303-0", "IMCC26207, uncultured."),
]

# Lineages deliberately left unassigned. Listed so their absence from EVIDENCE
# is a recorded decision rather than an oversight.
CONTEXT = [
    ("ARC", "Bathyarchaeia", "https://pubmed.ncbi.nlm.nih.gov/26494757/",
     "Methane-metabolism potential was identified in particular genomes, not across this broad lineage; other "
     "bathyarchaeotal genomes support acetogenic potential instead (PubMed 26626228). A 16S label does not resolve function."),
    ("ARC", "Woesearchaeales__b1/b2", "", "No cultured representative; no role assigned."),
    ("BAC", "Bacteroidales__b7/b8", "",
     "16 Bacteroidetes vadinHA17 ASVs across two blocks. Digester-associated uncultured lineage; no role assigned "
     "even though b7 carries a large negative coefficient in both models."),
    ("BAC", "MBA03, D8A_2, MVP_15, JS1, WS1, SBR1031, SJA_15, SJA_28, CK_2C2_2, WCHB1_41, DTU014",
     "", "Uncultured lineage labels with no cultured representative in this dataset; no role assigned."),
    ("BAC", "Cloacimonadales, Aminicenantales, Caldatribacteriales, Caldisericales, Hydrogenedentiales, Izemoplasmatales",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC2293186/",
     "Candidate-phylum lineages. One reconstructed Cloacamonas genome suggests amino-acid fermentation and "
     "possible syntrophy; no role is transferred to the other labels here."),
    ("BAC", "SAR324_clade(Marine_group_B)", "",
     "Uncultured clade whose described habitat is marine, not digesters. Carries a consistent negative coefficient "
     "in both models; no role assigned and the assignment itself should be treated cautiously."),
]

# Background statements used in the report's discussion; each is a claim about
# the literature, not about this cohort.
BACKGROUND = [
    ("ammonia_methanogen_spectrum", "https://www.sciencedirect.com/science/article/abs/pii/S2214714419302107",
     "Ammonia inhibition review: acetoclastic methanogens are the most ammonia-sensitive group, hydrogenotrophic "
     "methanogens the most tolerant, with Methanosarcina intermediate."),
    ("sao_replaces_acetoclastic", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7334858/",
     "Syntrophic acetate oxidation coupled to hydrogenotrophic methanogenesis can replace acetoclastic "
     "methanogenesis when ammonia rises."),
    ("sao_consortia_members", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9600251/",
     "Syntrophic acetate-oxidising consortia enriched from full-scale mesophilic digesters are taxonomically "
     "diverse, with unclassified Clostridia and Syntrophomonas recurring."),
    ("sao_full_scale_digesters", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4769498/",
     "Low- vs high-ammonia full-scale digesters differ in bacterial composition and fhs profiles, with novel "
     "candidate syntrophic acetate oxidisers among uncultured Clostridia."),
    ("propionate_syntroph_inhibition", "https://pubmed.ncbi.nlm.nih.gov/30278382/",
     "Ammonia dosing of digester sludge inhibited syntrophic propionate oxidation, with Smithella, "
     "Methanosaetaceae and Methanospirillaceae among the inhibited populations."),
    ("immigration", "https://www.nature.com/articles/s41598-017-09303-0",
     "Several of the most abundant populations in full-scale digesters are inactive immigrants from the feed; "
     "inactive populations were associated with aerobes and facultative anaerobes."),
]

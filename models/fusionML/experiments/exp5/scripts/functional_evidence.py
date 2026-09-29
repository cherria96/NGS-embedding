"""Conservative, manually reviewed literature annotations; not inferred gene content.

Only named genera with retrieved primary evidence receive role hypotheses.
Unmatched genera and broad labels remain unassigned. Accessed 2026-09-22.
"""


# Locate the repository independently of the script's directory depth.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_REPO_ROOT = next(p for p in _RepositoryPath(__file__).resolve().parents
                  if (p / "main.py").is_file() and (p / "configs").is_dir())
if str(_REPO_ROOT) not in _repository_sys.path:
    _repository_sys.path.insert(0, str(_REPO_ROOT))

EVIDENCE = [
    ('Methanosaeta', 'acetoclastic methanogenesis', 'culture/community evidence; strain/context dependent', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/', 'Ammonium perturbations shifted active methanogens from Methanosaeta toward other pathways; this does not establish a universal response direction.'),
    ('Methanoculleus', 'hydrogenotrophic methanogenesis', 'community activity evidence', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/', 'Hydrogenotrophic Methanoculleus became prominent under high ammonium in the tested digesters.'),
    ('Methanosarcina', 'multiple methanogenic pathways', 'community activity evidence; species dependent', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8510171/', 'Methanosarcina occupied an intermediate response in the ammonium experiment; genus identity alone does not resolve the pathway in this dataset.'),
    ('Methanolinea', 'hydrogenotrophic methanogenesis', 'isolate genome evidence', 'https://journals.asm.org/doi/10.1128/genomeA.00876-14', 'The Methanolinea tarda isolate genome supports hydrogenotrophic methanogenesis and formate utilization.'),
    ('Methanomassiliicoccus', 'hydrogen-dependent methyl reduction', 'culture evidence outside digesters', 'https://pubmed.ncbi.nlm.nih.gov/22859731/', 'A cultured representative reduced methanol with hydrogen; neither substrate alone supported methane formation.'),
    ('Candidatus_Methanofastidiosum', 'hydrogen-dependent methylated-thiol reduction', 'genome-inferred potential', 'https://pubmed.ncbi.nlm.nih.gov/26943620/', 'Reconstructed genomes support methylated-thiol reduction; activity of the study ASVs is not measured.'),
    ('Syntrophomonas', 'syntrophic fatty-acid oxidation', 'digester isotope/genome evidence', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6687939/', 'Stable-isotope-informed metagenomics identified syntrophic butyrate-degrading populations; exact substrate use remains population dependent.'),
    ('Proteiniphilum', 'proteolytic fermentation', 'culture evidence in wastewater reactor', 'https://pubmed.ncbi.nlm.nih.gov/16280479/', 'Proteolytic anaerobic isolates were recovered from a brewery-wastewater UASB reactor.'),
    ('Thermovirga', 'amino-acid degradation', 'culture/genome evidence outside digesters', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3387794/', 'The oil-well isolate uses amino acids and proteinaceous substrates; application to these ASVs is a hypothesis.'),
]
# Broad Bathyarchaeia and Cloacimonadota/W5 are deliberately NOT assigned
# genus-wide functions. Context references are reported without mapping them
# onto unidentified populations or resolving Cloacamonas/Cloacimonas synonyms.
CONTEXT = [
    ('Bathyarchaeia', 'https://pubmed.ncbi.nlm.nih.gov/26494757/', 'Methane-metabolism potential was identified in particular genomes, not all members of this broad lineage.'),
    ('Bathyarchaeia', 'https://pubmed.ncbi.nlm.nih.gov/26626228/', 'Other bathyarchaeotal genomes support distinct substrate preferences and acetogenic potential; a broad 16S label does not resolve function.'),
    ('Cloacimonadota', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC2293186/', 'One reconstructed Cloacamonas genome suggests amino-acid fermentation and possible syntrophy; no role is automatically transferred to W5 or Cloacimonas labels here.'),
]

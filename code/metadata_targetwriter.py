import numpy as np
import pandas as pd

# Load original metadata file
df = pd.read_csv('data/final/metadata.csv')

# Parse numeric values
pH = pd.to_numeric(df['eff_pH'], errors='coerce')
ALK = pd.to_numeric(df['eff_ALK'], errors='coerce')
TVFAs = pd.to_numeric(df['eff_TVFAs'], errors='coerce')
HAc = pd.to_numeric(df['eff_HAc'], errors='coerce')
HPro = pd.to_numeric(df['eff_HPro'], errors='coerce')
TAN = pd.to_numeric(df['eff_TAN'], errors='coerce')
CH4 = pd.to_numeric(df['eff_CH4'], errors='coerce')

# Ratios
vfa_alk_ratio = np.where(
    ALK > 0, TVFAs / ALK, np.where((ALK == 0) & (TVFAs > 0), 999.0, np.nan)
)
hpro_hac_ratio = np.where(HAc > 0, HPro / HAc, np.nan)

# Generate warning flag columns
df['warning_acid_base_balance'] = np.where((pH < 6.5) | (pH > 8.0), 1, 0)
df['warning_buffer_capacity'] = np.where(vfa_alk_ratio >= 0.30, 1, 0)
df['warning_acid_accumulation'] = np.where(
    ((hpro_hac_ratio > 1.4) & (HPro > 0.1)) | (HPro >= 0.8), 1, 0
)
df['warning_ammonia_toxicity'] = np.where(TAN > 2.5, 1, 0)
df['warning_biogas_quality'] = np.where(CH4 < 50.0, 1, 0)

# Save updated dataset
df.to_csv('metadata.csv', index=False)
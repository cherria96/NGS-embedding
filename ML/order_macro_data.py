"""Audited Order workbook integration; metadata is never a predictor."""
import numpy as np
import pandas as pd
from order_macro_models import TARGETS, CATS, NUMS, numeric
from order_macro_utils import label_report

def inspect(data_dir, out, group_map):
    m = pd.read_csv(data_dir/'metadata.csv', dtype={'SampleID': str, 'No': str})
    required = ['SampleID', 'Site', 'Season', 'Round', 'No'] + TARGETS
    if set(required)-set(m.columns):
        raise ValueError(f'Missing required metadata columns: {set(required)-set(m.columns)}')
    if m.SampleID.isna().any() or m.SampleID.duplicated().any() or m[['Site', 'Season']].isna().any().any():
        raise ValueError('Missing/duplicate metadata IDs or missing Site/Season')
    if not m[TARGETS].isin([0, 1]).all().all():
        raise ValueError('Targets must contain only 0 or 1')
    if not (m.SampleID == m.Round.astype(str) + '-' + m.No).all():
        raise ValueError('SampleID disagrees with Round/No')
    seasons = {1: 'Summer', 2: 'Fall', 3: 'Winter', 4: 'Spring'}
    if not (m.Round.map(seasons) == m.Season).all():
        raise ValueError('Round/Season mismatch')
    if m.groupby('No').Site.nunique().max() != 1 or m.groupby('Site').No.nunique().max() != 1:
        raise ValueError('Site/No mapping is inconsistent')
    m['SiteGroup'] = m.Site.replace(group_map)
    m = m.set_index('SampleID')
    sheets, structures, mappings = {}, [], []
    order_source = []
    for domain in ['BAC', 'ARC']:
        path = data_dir/f'Dat_{domain}.xlsx'
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            frame = pd.read_excel(book, sheet_name=sheet)
            structures.append({'file': path.name, 'sheet': sheet, 'data_rows': len(frame), 'columns': len(frame.columns), 'leading_columns': '|'.join(map(str, frame.columns[:8]))})
        otu = pd.read_excel(book, sheet_name='OTUs')
        frame = pd.read_excel(book, sheet_name='O(%)').set_index('Order')
        if frame.index.isna().any() or frame.index.duplicated().any():
            raise ValueError('Missing/duplicate Order names')
        if set(frame.index) != set(otu.Order.fillna('unidentified')):
            raise ValueError('Order sheet and OTU Order annotation disagree')
        frame = frame.apply(pd.to_numeric, errors='raise')
        if not np.isfinite(frame.values).all() or (frame.values < 0).any():
            raise ValueError('Missing or invalid Order percentage: cannot assume absence')
        if not np.allclose(frame.sum(axis=0), 100, atol=.1):
            raise ValueError('Order percentages do not sum to approximately 100')
        original = frame.columns.tolist()
        ids = []
        for full in original:
            parts = str(full).split('-', 2)
            if len(parts) != 3:
                raise ValueError(f'Unexpected genomic ID: {full}')
            sid = '-'.join(parts[:2]); ids.append(sid)
            mappings.append({'domain': domain, 'rank': 'Order', 'workbook_sample': full,
                'SampleID': sid, 'workbook_suffix': parts[2].removesuffix('-'+domain), 'original_suffix': parts[2],
                'Site': m.loc[sid, 'Site'] if sid in m.index else None})
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate normalized workbook sample IDs')
        frame.columns = ids
        frame = frame.T
        order_source.append({'domain': domain, 'original_OTUs': len(otu),
                              'unique_Orders': len(frame.columns)})
        frame.columns = [f'{domain}_O::{taxon}' for taxon in frame.columns]
        sheets[f'{domain}_O'] = frame
        book.close()
    pd.DataFrame(order_source).to_csv(out/'order_source_counts.csv', index=False)
    mapping = pd.DataFrame(mappings)
    mapping[mapping.workbook_suffix != mapping.original_suffix].to_csv(out/'sample_id_corrections.csv', index=False)
    if mapping.groupby('SampleID').workbook_suffix.nunique().max() > 1:
        raise ValueError('BAC/ARC site suffix disagreement')
    if mapping.groupby(['domain', 'Site']).workbook_suffix.nunique().max() > 1:
        raise ValueError('Workbook suffix changes within metadata site')
    pd.DataFrame(structures).to_csv(out/'file_structure.csv', index=False)
    pd.DataFrame(mappings).to_csv(out/'sample_id_mapping.csv', index=False)
    all_ids = sorted(set(m.index).union(*(set(f.index) for f in sheets.values())))
    audit = pd.DataFrame(index=pd.Index(all_ids, name='SampleID'))
    audit['metadata'] = audit.index.isin(m.index)
    for block, frame in sheets.items():
        audit[block] = audit.index.isin(frame.index)
    audit['included'] = audit.all(axis=1)
    audit.to_csv(out/'sample_audit.csv')
    print('Unmatched samples (excluded and recorded):', audit.index[~audit.included].tolist(), flush=True)
    original_targets = m[TARGETS].to_numpy(int)
    label_rows = label_report(original_targets, 'all_metadata')
    simultaneous = []
    for scope, yy in [('all_metadata', original_targets)]:
        simultaneous.extend({'scope': scope, 'simultaneous_labels': n, 'samples': int((yy.sum(axis=1) == n).sum()), 'percent': float(100*(yy.sum(axis=1) == n).mean())} for n in range(6))
    included = m.index[m.index.isin(audit.index[audit.included])]
    m = m.loc[included].copy()
    coercions = []
    for c in NUMS:
        parsed = m[c].map(numeric).replace([np.inf, -np.inf], np.nan)
        for sid in m.index[m[c].notna() & pd.to_numeric(m[c], errors='coerce').isna()]:
            coercions.append({'SampleID': sid, 'column': c, 'original': m.loc[sid, c], 'parsed': parsed.loc[sid]})
        m[c] = parsed
    pd.DataFrame(coercions, columns=['SampleID', 'column', 'original', 'parsed']).to_csv(out/'numeric_coercions.csv', index=False)
    X = pd.concat([m[CATS+NUMS]]+[f.loc[included] for f in sheets.values()], axis=1)
    y = m[TARGETS].to_numpy(int)
    label_rows += label_report(y, 'matched')
    simultaneous.extend({'scope': 'matched', 'simultaneous_labels': n, 'samples': int((y.sum(axis=1) == n).sum()), 'percent': float(100*(y.sum(axis=1) == n).mean())} for n in range(6))
    pd.DataFrame(simultaneous).to_csv(out/'simultaneous_labels.csv', index=False)
    pd.DataFrame({'missing': X.isna().sum(), 'missing_percent': 100*X.isna().mean()}).to_csv(out/'missing_features.csv')
    pd.concat([m[['Site', 'SiteGroup', 'Season', 'Round', 'No']+TARGETS], X], axis=1).to_csv(out/'cleaned_merged.csv')
    m.groupby(['SiteGroup', 'Site']).agg(samples=('Season', 'size'), seasons=('Season', lambda s: '|'.join(s))).to_csv(out/'samples_per_site.csv')
    source = ['eff_pH', 'eff_ALK', 'eff_TVFAs', 'eff_HPro', 'eff_HAc', 'eff_TAN', 'eff_CH4']
    m[source].isna().to_csv(out/'target_source_missingness.csv')
    assert all(c.startswith(('BAC_O::', 'ARC_O::')) for c in X.columns)
    assert not set(X.columns) & set(m.columns)
    return m, X, y, label_rows, structures


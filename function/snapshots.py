"""Include the split source packages when refreshing an experiment snapshot."""
from pathlib import Path
import hashlib
import shutil


def source_files(root):
    root = Path(root)
    paths = [root / 'main.py', root / 'models/__init__.py', root / 'datasets/__init__.py']
    for package in ['configs', 'function', 'models/fusionML', 'datasets/datapreprocess', 'datasets/graph', 'function/investigation_20260921']:
        paths.extend((root / package).glob('*.py'))
    return sorted(set(paths))


def snapshot_current_sources(root, destination, prefix=''):
    root, destination = Path(root), Path(destination)
    records = []
    for source in source_files(root):
        relative = Path(prefix) / source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        records.append({'source': str(source.relative_to(root)), 'path': str(relative), 'copy': str(relative),
                        'sha256': digest, 'bytes': target.stat().st_size,
                        'role': 'current code context, not certified original training version'})
    # A frozen snapshot has its own legacy import namespace, independent of
    # the live repository's root layout. Record each compatibility file too.
    legacy_sources = [(source, Path('fusionML') / source.name)
                      for source in (root / 'models/fusionML').glob('*.py')]
    legacy_sources += [(source, Path('fusionML/investigation_20260921') / source.name)
                       for source in (root / 'function/investigation_20260921').glob('*.py')]
    legacy_sources += [(source, Path('fusionML/configs') / source.name)
                       for source in (root / 'configs/fusionML').glob('*.yaml')]
    for source, legacy_path in legacy_sources:
        relative = Path(prefix) / legacy_path
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        records.append({'source': str(source.relative_to(root)), 'path': str(relative), 'copy': str(relative),
                        'sha256': digest, 'bytes': target.stat().st_size,
                        'role': 'current code context, not certified original training version'})
    return records

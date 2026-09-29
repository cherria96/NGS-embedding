"""fusionML: domain-constrained log-contrast fusion regression for AD microbiome data.

Implements the model and validation protocol in
docs/superpowers/specs/ad-microbial-model-design.md, items 9-18.
"""

__version__ = "0.1.0"

# Preserve historical pickle/import module names after the repository split.
if __name__ == "fusionML":
    import importlib as _importlib
    import sys as _sys
    for _name in ("estimator", "groups", "search", "interactions", "single_domain"):
        _module = _importlib.import_module("models.fusionML." + _name)
        _sys.modules["fusionML." + _name] = _module
        globals()[_name] = _module

import sys as _sys
_sys.modules.setdefault("fusionML", _sys.modules[__name__])

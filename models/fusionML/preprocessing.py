"""Compatibility alias; implementation: datasets.datapreprocess.preprocessing."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("datasets.datapreprocess.preprocessing")

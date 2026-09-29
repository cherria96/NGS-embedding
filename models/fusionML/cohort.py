"""Compatibility alias; implementation: datasets.datapreprocess.cohort."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("datasets.datapreprocess.cohort")

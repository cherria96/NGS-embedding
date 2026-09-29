"""Compatibility alias; implementation: function.model_io."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("function.model_io")

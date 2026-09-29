#!/usr/bin/env python3
"""Entry point: python fusionML/run_model.py --config config.yaml --mode train

(design-spec item 17's "one usable entry point" example)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionML.cli import main

if __name__ == "__main__":
    sys.exit(main())

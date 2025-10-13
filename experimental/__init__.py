# experimental/__init__.py
"""
Experimental modules for Binoculars-T4-gpu.

This file intentionally avoids importing heavy submodules (e.g., torch/transformers)
at package import time. Submodules are imported lazily when accessed.

Common submodules:
- falcon_t4_binoculars
- falcon_a100_binoculars  (optional)
- inference               (CLI module: python -m experimental.inference)
"""

from importlib import import_module

__all__ = ["falcon_t4_binoculars", "falcon_a100_binoculars", "inference"]

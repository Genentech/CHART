"""
Archetype annotation for CHART.

Optional: the model calls need ``pip install ".[annotation]"`` and a key,
and no later stage reads what this one writes.
"""

from .workflow import STEPS, run_annotation

__all__ = [
    'run_annotation',
    'STEPS'
]

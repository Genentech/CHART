"""
Combining per-well tables into across-wells tables for CHART.
"""

from .aggregate import AGGREGATION_LEVELS, METHODS, aggregate
from .wells import INDEX_LEVELS, LEVELS, concat_wells, reindex_for_level
from .workflow import (
    OUTPUT_NAME,
    SUFFIXES,
    TARGETS,
    CombineResult,
    MissingWellError,
    combine_target,
    resolve_targets,
    run_combining
)

__all__ = [
    'run_combining',
    'combine_target',
    'resolve_targets',
    'TARGETS',
    'SUFFIXES',
    'OUTPUT_NAME',
    'LEVELS',
    'INDEX_LEVELS',
    'aggregate',
    'AGGREGATION_LEVELS',
    'METHODS',
    'CombineResult',
    'MissingWellError',
    'concat_wells',
    'reindex_for_level'
]

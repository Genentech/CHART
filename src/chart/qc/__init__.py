"""
Quality control analysis for CHART.

The names re-exported here are the public interface of this subpackage.
Anything not listed is internal plumbing and may change.
"""

from .io import (
    load_objects,
    save_filters,
    save_segmentation_filters
)
from .plotting import (
    plot_nuclear_size,
    plot_cell_size
)
from .correlation_analysis import (
    compute_pheno_correlations,
    compute_pheno_sbs_correlations
)
from .segmentation_error_detection import detect_segmentation_errors
from .workflow import (
    COMPONENTS,
    run_qc,
    process_well,
    WellQCResult,
    ComponentUnavailableError
)

__all__ = [
    'run_qc',
    'process_well',
    'COMPONENTS',
    'WellQCResult',
    'ComponentUnavailableError',
    'load_objects',
    'save_filters',
    'plot_nuclear_size',
    'plot_cell_size',
    'compute_pheno_correlations',
    'compute_pheno_sbs_correlations',
    'detect_segmentation_errors',
    'save_segmentation_filters'
]

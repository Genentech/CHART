"""
Outlier cell detection and removal for CHART.
"""

from .prefilters import PREFILTERS, PrefilterResult, robust_cv, apply_prefilters
from .detection import apply_outlier_filters, detect_outliers_per_gene
from .plotting import plot_outlier_positions, plot_outlier_statistics
from .workflow import (
    OUTLIER,
    REASON_COLUMN,
    OutlierFilterRecord,
    OutlierResult,
    filter_list,
    load_outlier_record,
    run_outlier_filtering
)

__all__ = [
    'run_outlier_filtering',
    'OutlierResult',
    'OutlierFilterRecord',
    'load_outlier_record',
    'filter_list',
    'REASON_COLUMN',
    'OUTLIER',
    'PREFILTERS',
    'PrefilterResult',
    'apply_prefilters',
    'robust_cv',
    'detect_outliers_per_gene',
    'apply_outlier_filters',
    'plot_outlier_statistics',
    'plot_outlier_positions'
]

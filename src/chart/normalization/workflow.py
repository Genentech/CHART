"""
The normalisation step: one normalised table for the whole screen.

Reads the combined feature table written by the combine step, normalises
it across wells, and writes the result, optionally aggregated to guide or
gene level.
"""

import gc
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from ..combining.aggregate import AGGREGATION_LEVELS, aggregate
from ..io.config import AllwellsConfig, load_allwells_config
from ..io.tables import load_table, save_table
from .quantile import normalize_across_wells
from .validation import plot_normalization

logger = logging.getLogger(__name__)

#: The normalisation methods available, by the name the config uses.
METHODS = {
    'quantile': normalize_across_wells,
}

INPUT_NAME = 'allwells'
INPUT_SUFFIX = '-features'
OUTPUT_SUFFIX = '-features_{level}'


@dataclass
class NormalizationResult:
    """Summary of a normalisation run."""
    method: str = ''
    rows: int = 0
    columns: int = 0
    output: Optional[str] = None
    aggregated: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def resolve_method(name: str):
    """Look up a normalisation method by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown normalization_type {name!r}. "
                         f"Available methods: {', '.join(sorted(METHODS))}")
    return METHODS[name]


def run_normalization(config: Union[str, Dict[str, Any], AllwellsConfig],
                      method: Optional[str] = None,
                      feature_columns: Optional[Sequence[str]] = None,
                      aggregate_levels: Sequence[str] = (),
                      aggregation_method: str = 'median',
                      validate: bool = False,
                      save_plots: bool = True,
                      plots_dir: Optional[str] = None,
                      random_state: Optional[int] = None) -> NormalizationResult:
    """Normalise the combined feature table.

    Args:
        config: An :class:`~chart.io.config.AllwellsConfig`, or a
                configuration mapping to build one from
        method: Which method to use; the default comes from the config's
                ``normalization_type``
        feature_columns: Columns to normalise; the default is every numeric
                         column
        aggregate_levels: Also write the table aggregated to these levels;
                          see :data:`~chart.combining.aggregate.AGGREGATION_LEVELS`
        aggregation_method: How to combine the rows of a group
        validate: Draw the before-and-after distributions
        save_plots: Whether to write that figure
        plots_dir: Where to write it (default: the configured plot
                   directory)
        random_state: Seed for choosing which features to draw

    Returns:
        A :class:`NormalizationResult`.
    """
    config = load_allwells_config(config)
    method = method or config.normalization_type
    normalize = resolve_method(method)

    unknown = [level for level in aggregate_levels
               if level not in AGGREGATION_LEVELS]
    if unknown:
        raise ValueError(f"Cannot aggregate at level(s) {', '.join(unknown)}. "
                         f"Available levels: {', '.join(AGGREGATION_LEVELS)}")

    result = NormalizationResult(method=method)
    logger.info(f"Normalising with method '{method}'")
    logger.info(f"Reading from {config.combined_path}")
    logger.info(f"Writing to {config.normalized_path}")

    data = load_table(config.combined_path, INPUT_NAME, INPUT_SUFFIX)

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        data.to_parquet(tmp_file.name)
        temp_input_file = tmp_file.name
    del data
    gc.collect()

    try:
        data = pd.read_parquet(temp_input_file)
        normalized = normalize(data,
                               feature_columns=list(feature_columns)
                               if feature_columns else None)
    finally:
        os.unlink(temp_input_file)

    result.rows, result.columns = normalized.shape
    result.output = save_table(normalized, config.normalized_path, INPUT_NAME,
                               OUTPUT_SUFFIX.format(level='cell'))

    for level in aggregate_levels:
        aggregated = aggregate(normalized, level, method=aggregation_method)
        result.aggregated[level] = save_table(
            aggregated, config.normalized_path, INPUT_NAME,
            OUTPUT_SUFFIX.format(level=level))

    if validate:
        logger.info("Validating normalisation...")
        # The original is read again rather than kept, since holding both
        # tables is what this step can least afford.
        original = load_table(config.combined_path, INPUT_NAME, INPUT_SUFFIX)
        plot_normalization(original, normalized,
                           random_state=random_state,
                           save_plots=save_plots,
                           output_dir=plots_dir or config.plots_path)
        del original
        gc.collect()

    return result

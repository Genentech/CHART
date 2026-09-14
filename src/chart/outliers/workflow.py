"""
The outlier filtering step: drop cells that do not behave like their peers.

Reads the normalised table written by the normalise step, prefilters the
columns, flags outlier cells per gene, and writes the filter list, the
filtered table, and any requested aggregations.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .. import __version__
from ..combining.aggregate import AGGREGATION_LEVELS, aggregate
from ..io.config import AllwellsConfig, load_allwells_config
from ..io.tables import load_table, save_table
from .prefilters import PREFILTERS, PrefilterResult, apply_prefilters
from .detection import apply_outlier_filters, detect_outliers_per_gene
from .plotting import plot_outlier_positions, plot_outlier_statistics

logger = logging.getLogger(__name__)

#: The table this step reads, written by the normalise step.
INPUT_NAME = 'allwells'
INPUT_SUFFIX = '-features_cell'

#: The objects table, read only to place cells within their wells.
OBJECTS_SUFFIX = '-objects'

#: What this step writes.  Kept from cellmapp.
OUTPUT_SUFFIX = '-features_{level}_outlier_filtered'
FILTER_FILE = 'outlier_filters.parquet'
RECORD_FILE = 'outlier_filters.json'

#: Names every cell missing from the output, and why it went: the name of
#: the prefilter that dropped it, or :data:`OUTLIER` if the forest flagged
#: it.  cellmapp listed only the flagged cells, which left the prefiltered
#: ones unaccounted for.
REASON_COLUMN = 'reason'
OUTLIER = 'outlier'


@dataclass
class OutlierFilterRecord:
    """What produced the outlier filter list.
    """
    created: str = ''
    chart_version: str = ''
    source: str = ''
    index_levels: List[str] = field(default_factory=list)
    cells_prefiltered: int = 0
    outliers: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    dropped_columns: Dict[str, str] = field(default_factory=dict)
    #: Reason to the number of cells removed for it, summarising the
    #: filter list alongside it.
    removed_cells: Dict[str, int] = field(default_factory=dict)

    def save(self, filter_dir: str) -> str:
        os.makedirs(filter_dir, exist_ok=True)
        self.created = datetime.now().isoformat(timespec='seconds')
        self.chart_version = __version__
        path = os.path.join(filter_dir, RECORD_FILE)
        with open(path, 'w') as handle:
            json.dump(asdict(self), handle, indent=2)
        logger.info(f"Saved outlier filter record: {path}")
        return path


def load_outlier_record(filter_dir: str) -> Optional[OutlierFilterRecord]:
    """Read the sidecar for the outlier filter list, if there is one."""
    path = os.path.join(filter_dir, RECORD_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        raw = json.load(handle)
    known = {f for f in OutlierFilterRecord.__dataclass_fields__}
    return OutlierFilterRecord(**{k: v for k, v in raw.items() if k in known})


def filter_list(index_names: Sequence[Any], prefilter: PrefilterResult,
                outlier_cells: pd.DataFrame) -> pd.DataFrame:
    """Every cell missing from the output, with the reason it went.

    Args:
        index_names: The index levels of the input table, which become the
                     columns identifying a cell
        prefilter: The account from :func:`~chart.outliers.apply_prefilters`
        outlier_cells: The cells the forest flagged

    Returns:
        One row per removed cell: a column for each index level, plus
        :data:`REASON_COLUMN`.  Applying the whole list to the input table
        reproduces the rows of the output.
    """
    columns = [str(name) for name in index_names]
    frames = []

    for reason, cells in prefilter.dropped_cells.items():
        if len(cells):
            frame = cells.to_frame(index=False)
            frame[REASON_COLUMN] = reason
            frames.append(frame)

    if len(outlier_cells):
        frame = outlier_cells.copy()
        frame[REASON_COLUMN] = OUTLIER
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=[*columns, REASON_COLUMN])
    return pd.concat(frames, axis=0, ignore_index=True)[[*columns, REASON_COLUMN]]


@dataclass
class OutlierResult:
    """Summary of an outlier filtering run."""
    cells_in: int = 0
    columns_in: int = 0
    cells_prefiltered: int = 0
    columns_prefiltered: int = 0
    outliers: int = 0
    cells_out: int = 0
    filter_file: Optional[str] = None
    output: Optional[str] = None
    aggregated: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def run_outlier_filtering(config: Union[str, Dict[str, Any], AllwellsConfig],
                          aggregate_levels: Sequence[str] = ('guide',),
                          aggregation_method: str = 'median',
                          skip_prefilters: Sequence[str] = (),
                          save_plots: bool = True,
                          plots_dir: Optional[str] = None) -> OutlierResult:
    """Prefilter the normalised table and remove outlier cells.

    Args:
        config: An :class:`~chart.io.config.AllwellsConfig`, or a
                configuration mapping to build one from
        aggregate_levels: Also write the filtered table aggregated to these
                          levels.  Defaults to guide level, as cellmapp did
        aggregation_method: How to combine the rows of a group
        skip_prefilters: Column and row prefilters not to run; see
                      :data:`~chart.outliers.columns.PREFILTERS`
        save_plots: Whether to draw and write the plots
        plots_dir: Where to write them (default: the configured plot
                   directory)

    Returns:
        An :class:`OutlierResult`.
    """
    config = load_allwells_config(config)

    unknown = [level for level in aggregate_levels
               if level not in AGGREGATION_LEVELS]
    if unknown:
        raise ValueError(f"Cannot aggregate at level(s) {', '.join(unknown)}. "
                         f"Available levels: {', '.join(AGGREGATION_LEVELS)}")

    logger.info(f"Reading from {config.normalized_path}")
    logger.info(f"Writing to {config.outlier_filtered_path}")

    data = load_table(config.normalized_path, INPUT_NAME, INPUT_SUFFIX)
    result = OutlierResult(cells_in=len(data), columns_in=len(data.columns))

    prefiltered, prefilter = apply_prefilters(
        data,
        missing_value_threshold=config.missing_value_threshold,
        rcv_threshold=config.rcv_threshold,
        skip=skip_prefilters)
    result.cells_prefiltered = len(prefiltered)
    result.columns_prefiltered = len(prefiltered.columns)

    if prefiltered.empty or not len(prefiltered.columns):
        raise ValueError(
            f"Prefiltering left {len(prefiltered)} cells and "
            f"{len(prefiltered.columns)} columns, so there is nothing to "
            f"analyse. Check the missing_value_threshold and rcv_threshold "
            f"settings, or skip the prefilters that are removing everything "
            f"({', '.join(PREFILTERS)})."
        )

    outlier_cells = detect_outliers_per_gene(
        prefiltered,
        contamination=config.contamination,
        random_state=config.random_state,
        n_estimators=config.n_estimators)
    result.outliers = len(outlier_cells)

    # The filter list is written whether or not anything was flagged, so
    # that an empty one is distinguishable from a step that never ran.
    os.makedirs(config.filters_path, exist_ok=True)
    result.filter_file = os.path.join(config.filters_path, FILTER_FILE)
    removed = filter_list(data.index.names, prefilter, outlier_cells)
    removed.to_parquet(result.filter_file, index=False)
    by_reason = removed[REASON_COLUMN].value_counts().to_dict()
    logger.info(f"Saved {len(removed)} removed cells to {result.filter_file}: "
                + (', '.join(f"{count} {reason}"
                             for reason, count in by_reason.items())
                   or 'none'))

    OutlierFilterRecord(
        source=os.path.join(config.normalized_path,
                            f"{INPUT_NAME}{INPUT_SUFFIX}.parquet"),
        index_levels=[str(n) for n in prefiltered.index.names],
        cells_prefiltered=len(prefiltered),
        outliers=len(outlier_cells),
        parameters={'contamination': config.contamination,
                    'random_state': config.random_state,
                    'n_estimators': config.n_estimators,
                    'missing_value_threshold': config.missing_value_threshold,
                    'rcv_threshold': config.rcv_threshold,
                    'skipped_prefilters': list(skip_prefilters)},
        dropped_columns=dict(prefilter.dropped_columns),
        removed_cells={str(reason): int(count)
                       for reason, count in by_reason.items()}
    ).save(config.filters_path)

    if save_plots:
        plot_dir = plots_dir or config.plots_path
        plot_outlier_statistics(prefiltered, outlier_cells, plot_dir)
        _plot_positions(config, outlier_cells, plot_dir)

    filtered = apply_outlier_filters(prefiltered, outlier_cells)
    result.cells_out = len(filtered)
    result.output = save_table(filtered, config.outlier_filtered_path,
                               INPUT_NAME, OUTPUT_SUFFIX.format(level='cell'))

    for level in aggregate_levels:
        aggregated = aggregate(filtered, level, method=aggregation_method)
        result.aggregated[level] = save_table(
            aggregated, config.outlier_filtered_path, INPUT_NAME,
            OUTPUT_SUFFIX.format(level=level))

    return result


def _plot_positions(config: AllwellsConfig, outlier_cells: pd.DataFrame,
                    plot_dir: str) -> None:
    """Draw where the outliers sit, if the objects table is available.

    The positions come from the combined objects table rather than the
    normalised features, which carry no coordinates.  Its absence is not
    an error: the plots are a convenience, and the step's real output does
    not depend on them.
    """
    if len(outlier_cells) == 0:
        return

    path = os.path.join(config.combined_path,
                        f"{INPUT_NAME}{OBJECTS_SUFFIX}.parquet")
    if not os.path.exists(path):
        logger.warning(f"Not plotting outlier positions: {path} is not there, "
                       f"so there are no centroid coordinates to plot")
        return

    cell_data = load_table(config.combined_path, INPUT_NAME, OBJECTS_SUFFIX)
    plot_outlier_positions(outlier_cells, cell_data, plot_dir)

"""
Filtering workflow.

This module orchestrates the per-well sequence: loading the QC filter
lists, applying them to the object and feature tables, and writing the
filtered results.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from ..io.config import BywellConfig, load_bywell_config
from .features import filter_features, merge_objects_features
from ..io.tables import load_table, save_table
from .io import load_filter_lists, rename_columns
from .labels import apply_filters, arrange_index, drop_unassigned

logger = logging.getLogger(__name__)

#: What can be filtered.  ``objects`` keeps every column, ``features``
#: keeps only the feature columns.
TARGETS = ('objects', 'features')


class MissingFiltersError(RuntimeError):
    """A well was asked for but QC left no filter lists for it."""


@dataclass
class WellFilterResult:
    """Summary of a filtering run for a single well."""
    well: str
    objects_kept: Optional[int] = None
    objects_total: Optional[int] = None
    features_kept: Optional[int] = None
    features_total: Optional[int] = None
    merged_written: bool = False
    skipped: bool = False
    error: Optional[str] = None


def _enabled(targets: Sequence[str], name: str) -> bool:
    """Whether *name* should be filtered for this request."""
    return 'all' in targets or name in targets


def process_well(config: Union[Dict[str, Any], BywellConfig],
                 well: str,
                 targets: Sequence[str] = ('all',),
                 require_filters: bool = True) -> WellFilterResult:
    """Apply the QC filters for a single well.

    Args:
        config: A :class:`~chart.io.config.BywellConfig`, or a
                configuration mapping to build one from
        well: Well identifier
        targets: What to filter; ``('all',)`` does everything in
                 :data:`TARGETS`
        require_filters: Raise :class:`MissingFiltersError` when the well
                         has no QC filter lists.  Set ``False`` to skip
                         such wells with a warning instead.

    Returns:
        A :class:`WellFilterResult` with the object counts before and after.
    """
    config = load_bywell_config(config)
    result = WellFilterResult(well=well)

    logger.info(f"Working on well {well}")
    logger.info(f"Filtering: {targets}")

    inclusion, exclusion = load_filter_lists(config.filters_path, well)
    if not inclusion and not exclusion:
        if require_filters:
            raise MissingFiltersError(
                f"well {well} has no QC filter lists in {config.filters_path}; "
                f"run the qc step first")
        logger.warning(f"Skipping well {well}: no QC filter lists in "
                       f"{config.filters_path}")
        result.skipped = True
        return result

    if config.premerged:
        return _process_premerged(config, well, result, inclusion, exclusion)

    objects = features = None

    if _enabled(targets, 'objects'):
        objects = arrange_index(rename_columns(
            load_table(config.merged_path, well, '-objects',
                       config.objects_pattern),
            config.schema.rename_map()))
        if config.drop_unassigned:
            objects = drop_unassigned(objects)
        result.objects_total = len(objects)
        objects = apply_filters(objects, inclusion, exclusion)
        result.objects_kept = len(objects)
        save_table(objects, config.filtered_path, well, '-objects')

    if _enabled(targets, 'features'):
        features = arrange_index(rename_columns(
            load_table(config.merged_path, well, '-features',
                       config.features_pattern),
            config.schema.rename_map()))
        if config.drop_unassigned:
            features = drop_unassigned(features)
        result.features_total = len(features)
        features = filter_features(features,
                                   keep_patterns=config.schema.feature_patterns,
                                   exclude_patterns=config.exclude_patterns)
        features = apply_filters(features, inclusion, exclusion)
        result.features_kept = len(features)
        save_table(features, config.filtered_path, well, '-features')

    # The across-wells step expects a combined table as well.  It is built
    # from the frames already in memory rather than by re-reading them.
    if objects is not None and features is not None:
        merged = merge_objects_features(objects, features)
        merged = filter_features(merged,
                                 keep_patterns=config.schema.feature_patterns,
                                 exclude_patterns=config.exclude_patterns)
        save_table(merged, config.filtered_path, well)
        result.merged_written = True

    logger.info(f"Filtering completed for well {well}")
    return result


def _process_premerged(config: BywellConfig,
                       well: str,
                       result: WellFilterResult,
                       inclusion: Sequence[Any],
                       exclusion: Sequence[Any]) -> WellFilterResult:
    """Filter a single input file that already holds objects and features.

    There is one file rather than two, so ``merged_pattern`` names it.
    """
    data = load_table(config.merged_path, well, '', config.merged_pattern)
    data = rename_columns(data, {**(config.column_mapping or {}),
                                 **config.schema.rename_map()},
                          config.column_prefix_mapping)

    if 'label' not in data.columns:
        data['label'] = data.index

    data = arrange_index(data)
    if config.drop_unassigned:
        data = drop_unassigned(data)

    result.objects_total = result.features_total = len(data)
    filtered = apply_filters(data, inclusion, exclusion)
    result.objects_kept = result.features_kept = len(filtered)
    save_table(filtered, config.filtered_path, well, '-objects')

    features = filter_features(filtered, exclude_patterns=config.exclude_patterns)
    save_table(features, config.filtered_path, well, '-features')
    save_table(features, config.filtered_path, well)
    result.merged_written = True

    logger.info(f"Filtering completed for well {well}")
    return result


def run_filtering(config: Union[Dict[str, Any], BywellConfig],
                  wells: Optional[Sequence[str]] = None,
                  targets: Sequence[str] = ('all',),
                  require_filters: Optional[bool] = None,
                  continue_on_error: bool = False) -> List[WellFilterResult]:
    """Apply the QC filters across wells.

    Args:
        config: A :class:`~chart.io.config.BywellConfig`, or a
                configuration mapping to build one from
        wells: Wells to process; ``None`` processes every well in the
               config
        targets: What to filter; ``('all',)`` does everything
        require_filters: Fail when a well has no filter lists.  The default
                         requires them only when *wells* was given, so
                         naming a well makes its filters mandatory while a
                         config-supplied list tolerates a gap
        continue_on_error: Log and carry on when a well fails

    Returns:
        One :class:`WellFilterResult` per well, in the order processed.
    """
    config = load_bywell_config(config)
    if require_filters is None:
        require_filters = wells is not None

    if wells is None:
        wells = config.well_names()

    logger.info(f"Filtering {len(wells)} wells: {list(wells)}")
    if config.premerged:
        logger.info("Using pre-merged input files ("
                    + (config.merged_pattern or '{well}.parquet') + ")")

    results = []
    for well in wells:
        logger.info(f"=== Filtering well {well} ===")
        try:
            results.append(process_well(config, well, targets=targets,
                                        require_filters=require_filters))
        except Exception as e:
            if not continue_on_error:
                raise
            logger.error(f"Error filtering well {well}: {e}", exc_info=True)
            results.append(WellFilterResult(well=well, error=str(e)))

    failed = [r.well for r in results if r.error]
    skipped = [r.well for r in results if r.skipped]
    logger.info(f"Filtering complete: "
                f"{len(results) - len(failed) - len(skipped)}/{len(results)} wells filtered")
    if skipped:
        logger.warning(f"Wells skipped for having no filter lists: {skipped}")
    if failed:
        logger.warning(f"Wells that failed: {failed}")

    return results

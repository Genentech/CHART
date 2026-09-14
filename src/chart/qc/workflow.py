"""
Quality control workflow.

This module orchestrates the per-well QC sequence: loading objects,
plotting size distributions, detecting segmentation errors, computing
correlations between imaging modalities, and saving the resulting filters.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ..io.config import (
    BywellConfig,
    join_path,
    load_bywell_config
)
from ..io.filters import FilterEntry, FilterManifest
from .io import (
    SCALLOPS_AVAILABLE,
    load_objects,
    load_precomputed_filters,
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

logger = logging.getLogger(__name__)

COMPONENTS = ('plots', 'segmentation', 'pheno', 'pheno_sbs', 'filters')


class ComponentUnavailableError(RuntimeError):
    """A component was explicitly requested but its prerequisites are missing."""


@dataclass
class WellQCResult:
    """Summary of a QC run for a single well."""
    well: str
    total_objects: int = 0
    sgrna_assigned: int = 0
    segmentation_errors: Optional[int] = None
    pheno_passing: Optional[int] = None
    pheno_total: Optional[int] = None
    pheno_sbs_passing: Optional[int] = None
    pheno_sbs_total: Optional[int] = None
    skipped: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _enabled(components: Sequence[str], name: str) -> bool:
    """Whether component *name* should run for this request."""
    return 'all' in components or name in components


def _requested(components: Sequence[str], name: str) -> bool:
    """Whether the caller named this component explicitly, rather than via 'all'."""
    return name in components


def _skip(result: 'WellQCResult', components: Sequence[str], name: str, reason: str) -> None:
    """Record that a component could not run."""
    if _requested(components, name):
        raise ComponentUnavailableError(f"'{name}' was requested for well {result.well} but {reason}")
    logger.warning(f"Skipping '{name}' for well {result.well}: {reason}")
    result.skipped.append(name)


def get_well_paths(config: Union[Dict[str, Any], BywellConfig], well: str) -> Dict[str, str]:
    """Get all required paths for a well from configuration."""
    config = load_bywell_config(config)
    well_config = config.well(well)

    filter_dir = config.filters_path
    qc_dir = config.reports_path

    if not filter_dir.startswith('s3://'):
        os.makedirs(filter_dir, exist_ok=True)
    if not qc_dir.startswith('s3://'):
        os.makedirs(qc_dir, exist_ok=True)

    def scallops_path(value: str) -> str:
        """Resolve a scallops-relative directory, or '' when unconfigured."""
        return join_path(config.scallops_dir, value) if value else ''

    return {
        'objects_dir': config.merged_path,
        'pheno_dir': scallops_path(well_config.pheno_dir),
        'pheno_to_sbs_dir': scallops_path(well_config.pheno_sbs_registered_dir),
        'sbs_dir': scallops_path(well_config.sbs_dir),
        'sbs_feature_dir': scallops_path(well_config.sbs_feature_dir),
        'filter_dir': filter_dir,
        'qc_dir': qc_dir
    }


def process_well(config: Union[Dict[str, Any], BywellConfig],
                 well: str,
                 components: Sequence[str] = ('all',),
                 save_plots: bool = True,
                 plots_dir: Optional[str] = None) -> WellQCResult:
    """Run the QC sequence for a single well.

    Args:
        config: A :class:`~chart.io.config.BywellConfig`, or a configuration
                mapping to build one from
        well: Well identifier
        components: Which components to run; ``('all',)`` runs everything.
                    Individual names are listed in :data:`COMPONENTS`.
        save_plots: Whether to write plots to *plots_dir*
        plots_dir: Where to write plots (default: ``{qc_dir}/plots``)

    Returns:
        A :class:`WellQCResult` with the counts from each component that ran.

    Raises:
        Whatever the underlying analysis raises; see :func:`run_qc` for
        per-well error isolation.
    """
    config = load_bywell_config(config)
    well_config = config.well(well)
    paths = get_well_paths(config, well)
    qc_dir = paths['qc_dir']
    if plots_dir is None:
        plots_dir = os.path.join(qc_dir, 'plots')

    logger.info(f"Working on well {well}")
    logger.info(f"QC directory: {qc_dir}")
    logger.info(f"Running components: {components}")

    objects = load_objects(paths['objects_dir'], well,
                           premerged=config.premerged,
                           column_mapping=config.column_mapping)

    sgRNA_count = sum(objects['sgRNA'].notna())
    logger.info(f"Objects with sgRNA assignment: {sgRNA_count}")

    result = WellQCResult(well=well,
                          total_objects=len(objects),
                          sgrna_assigned=int(sgRNA_count))

    filter_entries: List[FilterEntry] = []

    # Plot size distributions
    if _enabled(components, 'plots'):
        logger.info("Plotting nuclear and cell size distributions...")
        plot_nuclear_size(objects, well, save_plots, plots_dir)
        plot_cell_size(objects, well, save_plots, plots_dir)

    # Detect segmentation errors
    if _enabled(components, 'segmentation'):
        logger.info("Detecting segmentation errors...")
        high_ratio_errors = detect_segmentation_errors(
            objects, well,
            ratio_threshold=config.thresholds.segmentation_error_threshold,
            save_plots=save_plots,
            output_dir=plots_dir
        )
        entry = save_segmentation_filters(high_ratio_errors, well, paths['filter_dir'])
        if entry is not None:
            filter_entries.append(entry)
        result.segmentation_errors = int(high_ratio_errors.sum())

    dapi_channel_names = well_config.dapi_channel_names
    if dapi_channel_names:
        logger.info(f"DAPI channel names: {dapi_channel_names}")

    # Compute pheno correlations
    phenocor_labels = None
    if _enabled(components, 'pheno'):
        if not paths['pheno_dir']:
            _skip(result, components, 'pheno', "no pheno_dir is configured for this well")
        elif not SCALLOPS_AVAILABLE:
            _skip(result, components, 'pheno', "scallops is not installed")
        elif not dapi_channel_names:
            _skip(result, components, 'pheno',
                  "no dapi_channel_names is configured for this well")
        elif len(dapi_channel_names) < 2:
            _skip(result, components, 'pheno',
                  f"only {len(dapi_channel_names)} DAPI channel(s) configured, need at least 2")
        else:
            logger.info("Computing pheno correlations...")
            cor_threshold = config.thresholds.pheno_correlation
            logger.info(f"Using pheno correlation threshold: {cor_threshold}")

            pass_thre, mean_cors, num_passing, n_scored = compute_pheno_correlations(
                objects, paths['pheno_dir'], well,
                cor_threshold, save_plots, plots_dir,
                dapi_channel_names=dapi_channel_names)
            phenocor_labels = objects.index[pass_thre]
            result.pheno_passing = int(num_passing)
            result.pheno_total = int(n_scored)
            logger.info(f"Pheno correlation results: {num_passing}/{n_scored} objects passed threshold ({num_passing/n_scored*100:.1f}%)")

    # Compute pheno-SBS correlations
    phenosbscor_labels = None
    if _enabled(components, 'pheno_sbs'):
        missing_dirs = [name for name in ('pheno_to_sbs_dir', 'sbs_dir', 'sbs_feature_dir')
                        if not paths[name]]
        if missing_dirs:
            _skip(result, components, 'pheno_sbs',
                  f"these directories are not configured: {', '.join(missing_dirs)}")
        elif not SCALLOPS_AVAILABLE:
            _skip(result, components, 'pheno_sbs', "scallops is not installed")
        elif well_config.pheno_sbs_dapi_channel is None and not dapi_channel_names:
            _skip(result, components, 'pheno_sbs',
                  "neither pheno_sbs_dapi_channel nor dapi_channel_names is "
                  "configured for this well")
        else:
            logger.info("Computing pheno-SBS correlations...")
            cor_threshold = config.thresholds.pheno_sbs_correlation
            logger.info(f"Using pheno-SBS correlation threshold: {cor_threshold}")
            pheno_dapi_ch = well_config.resolve_pheno_sbs_dapi_channel()
            logger.info(f"Using pheno DAPI channel: {pheno_dapi_ch}")
            pass_thre_phenosbs, phenosbs_cors, num_passing_phenosbs, n_scored_phenosbs = compute_pheno_sbs_correlations(
                objects, paths['pheno_to_sbs_dir'], paths['sbs_dir'], paths['sbs_feature_dir'],
                well, cor_threshold, save_plots, plots_dir,
                pheno_dapi_channel=pheno_dapi_ch)
            phenosbscor_labels = pass_thre_phenosbs.index[pass_thre_phenosbs]
            result.pheno_sbs_passing = int(num_passing_phenosbs)
            result.pheno_sbs_total = int(n_scored_phenosbs)
            logger.info(f"Pheno-SBS correlation results: {num_passing_phenosbs}/{n_scored_phenosbs} objects passed threshold ({num_passing_phenosbs/n_scored_phenosbs*100:.1f}%)")

    # Save filters
    if _enabled(components, 'filters'):
        logger.info("Saving filters...")
        if phenocor_labels is None and phenosbscor_labels is None:
            logger.warning(f"No correlation filters were produced for well {well}; "
                           f"nothing to save beyond any precomputed filters")
        filter_entries += save_filters(phenocor_labels, phenosbscor_labels,
                                       well, paths['filter_dir'])
        if config.precomputed_filters:
            filter_entries += load_precomputed_filters(config.precomputed_filters,
                                                       well, paths['filter_dir'])

    # Written even when nothing was produced: an empty manifest says this
    # run made no filters, which an absent one cannot distinguish from an
    # earlier run's.
    FilterManifest(
        well=well,
        components_run=[c for c in COMPONENTS
                        if _enabled(components, c) and c not in result.skipped],
        components_skipped=list(result.skipped),
        filters=filter_entries
    ).save(paths['filter_dir'])
    if not filter_entries:
        logger.warning(f"No filters were produced for well {well}; the "
                       f"manifest records that, so filtering will not apply "
                       f"an earlier run's")

    logger.info(f"Quality control analysis completed for well {well}")
    return result


def run_qc(config: Union[Dict[str, Any], BywellConfig],
           wells: Optional[Sequence[str]] = None,
           components: Sequence[str] = ('all',),
           save_plots: bool = True,
           plots_dir: Optional[str] = None,
           continue_on_error: bool = False) -> List[WellQCResult]:
    """Run QC across wells.

    Args:
        config: A :class:`~chart.io.config.BywellConfig`, or a configuration
                mapping to build one from
        wells: Wells to process; ``None`` processes every well in the config
        components: Which components to run; ``('all',)`` runs everything
        save_plots: Whether to write plots
        plots_dir: Where to write plots (default: ``{qc_dir}/plots``)
        continue_on_error: Log and carry on when a well fails.  Defaults to
                           ``False`` so that a failure stops the run rather
                           than producing a partial set of filters.

    Returns:
        One :class:`WellQCResult` per well, in the order processed.  Wells
        that failed have their *error* field set.
    """
    config = load_bywell_config(config)

    if wells is None:
        wells = config.well_names()

    logger.info(f"Processing {len(wells)} wells for QC analysis: {list(wells)}")
    logger.info(f"Running components: {components}")
    if config.premerged:
        logger.info("Using pre-merged input files ({well}.parquet)")
    if config.column_mapping:
        logger.info(f"Column mapping active ({len(config.column_mapping)} entries)")

    results = []
    for well in wells:
        logger.info(f"=== Processing well {well} ===")
        try:
            results.append(process_well(config, well,
                                        components=components,
                                        save_plots=save_plots,
                                        plots_dir=plots_dir))
        except Exception as e:
            if not continue_on_error:
                raise
            logger.error(f"Error processing well {well}: {e}", exc_info=True)
            results.append(WellQCResult(well=well, error=str(e)))

    failed = [r.well for r in results if r.error]
    logger.info(f"QC complete: {len(results) - len(failed)}/{len(results)} wells succeeded")
    if failed:
        logger.warning(f"Wells that failed: {failed}")

    return results

"""
Preprocessing workflow.

This module runs the preprocessing steps for a set of wells.  Each step
lives in its own subpackage, such as :mod:`chart.qc`; this layer only
decides which of them run, and in what order.

The steps come in two halves: the by-well steps run once per well, and the
across-wells steps run once over the whole well list.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .io.config import BywellConfig, PreprocessingConfig, load_preprocessing_config
from .combining import CombineResult, run_combining
from .filtering import WellFilterResult, run_filtering
from .normalization import NormalizationResult, run_normalization
from .outliers import OutlierResult, run_outlier_filtering
from .qc import WellQCResult, run_qc

logger = logging.getLogger(__name__)

#: Steps that run once per well.  Merging is still to be ported from
#: cellmapp.
BYWELL_STEPS = ('qc', 'filter')

#: Steps that run once over the whole well list.
ALLWELLS_STEPS = ('combine', 'normalize', 'outliers')

#: cellmapp's outlier step aggregated to guide level unless told otherwise,
#: while its normalise step never aggregated at all.  Both defaults are
#: kept, so a run reproduces what cellmapp produced.
DEFAULT_NORMALIZE_AGGREGATION = ()
DEFAULT_OUTLIER_AGGREGATION = ('guide',)

#: Every step, in the order they run.
STEPS = BYWELL_STEPS + ALLWELLS_STEPS


class StaleInputError(RuntimeError):
    """A step's input was not produced by the step that feeds it.

    Raised rather than reading what an earlier run left behind, which
    would succeed while describing the wrong data.
    """


@dataclass
class PreprocessingResult:
    """What each step produced, for the steps that ran."""
    qc: List[WellQCResult] = field(default_factory=list)
    filtering: List[WellFilterResult] = field(default_factory=list)
    combining: List[CombineResult] = field(default_factory=list)
    normalization: Optional[NormalizationResult] = None
    outliers: Optional[OutlierResult] = None

    @property
    def failed(self) -> List[str]:
        """What failed, named by well for the by-well steps and by step for
        the across-wells ones."""
        failures = [r.well for r in (*self.qc, *self.filtering) if r.error]
        failures += [f"combine ({r.target})" for r in self.combining if r.error]
        for step, result in (('normalize', self.normalization),
                             ('outliers', self.outliers)):
            if result is not None and result.error:
                failures.append(step)
        return failures


#: The combine target that normalisation reads.
_NORMALIZATION_INPUT = 'features'


def _stale_outlier_input(normalization: Optional[NormalizationResult]) -> Optional[str]:
    """Why outlier filtering cannot trust its input, or ``None``.

    The same reasoning as :func:`_stale_normalization_input`, one step
    further down: a normalise step that wrote nothing leaves the previous
    run's table for this step to read.
    """
    if normalization is None or normalization.output:
        return None

    reason = normalization.error or 'it produced no output'
    return (f"Not filtering outliers: the normalize step did not write its "
            f"table in this run ({reason}), so the only input available "
            f"would be left over from an earlier run. Fix what stopped the "
            f"normalize step, or ask for outliers on its own if an older "
            f"table really is intended.")


def _stale_normalization_input(combining: Sequence[CombineResult]) -> Optional[str]:
    """Why normalisation cannot trust its input, or ``None`` when it can.

    Normalisation reads one table that the combine step writes.  A skipped
    combine leaves whatever an earlier run wrote in place, so running on
    that would quietly normalise data that does not match this run's
    filters.  An absent file at least fails; a stale one would not.
    """
    combined = next((r for r in combining
                     if r.target == _NORMALIZATION_INPUT), None)
    if combined is None or combined.output:
        # Either combine did not run, in which case there is nothing to go
        # on and the caller is trusted, or it wrote the table.
        return None

    reason = combined.error or combined.skipped or 'it produced no output'
    return (f"Not normalising: the combine step did not write the "
            f"{_NORMALIZATION_INPUT} table in this run ({reason}), so the "
            f"only input available would be left over from an earlier run. "
            f"Fix what stopped the combine step, or ask for normalize on "
            f"its own if an older table really is intended.")


#: Which field of a filter result says a combine target was written for a
#: well.  A target the filter step did not run leaves the field unset, so
#: the well has last run's table for it and this run's for the others.
_TARGET_WRITTEN: Dict[str, Callable[[WellFilterResult], bool]] = {
    'objects': lambda r: r.objects_kept is not None,
    'features': lambda r: r.features_kept is not None,
    'merged': lambda r: r.merged_written,
}


def _wells_for_filtering(qc: Sequence[WellQCResult]) -> List[str]:
    """The wells QC handled, minus the ones it failed on.

    Only reachable with ``continue_on_error``, since QC would otherwise
    have stopped the run.  The failure is already recorded, so a well is
    excluded with a warning rather than raising again.
    """
    failed = [r.well for r in qc if r.error]
    if not failed:
        return [r.well for r in qc]

    logger.warning(f"Not filtering {len(failed)} of {len(qc)} wells whose QC "
                   f"failed in this run: {', '.join(failed)}. Filtering them "
                   f"would apply whatever filters an earlier run left "
                   f"behind.")
    return [r.well for r in qc if not r.error]


def _wells_for_combining(wells: Sequence[str],
                         filtering: Sequence[WellFilterResult],
                         named: bool) -> Dict[str, List[str]]:
    """Work out, per target, which wells the filter step wrote in this run.

    Checking the well alone is not enough: a run that filtered only the
    objects target writes one table per well and leaves the others as they
    were, so the well is current for one target and stale for the rest.

    Returns:
        The wells to combine for each target in
        :data:`~chart.combining.TARGETS`.  A target with no wells is one
        the filter step did not produce.

    Raises:
        StaleInputError: if a well was named explicitly, or if no target
            has any well left
    """
    if not filtering:
        # The filter step did not run, so there is nothing to check against
        # and the caller is trusted.
        return {target: list(wells) for target in _TARGET_WRITTEN}

    usable = [r for r in filtering if not r.skipped and not r.error]
    by_target = {}

    for target, was_written in _TARGET_WRITTEN.items():
        written = {r.well for r in usable if was_written(r)}
        stale = [well for well in wells if well not in written]
        if not stale:
            by_target[target] = list(wells)
            continue

        detail = (f"the filter step did not write the {target} table for "
                  f"{len(stale)} of {len(wells)} wells in this run: "
                  f"{', '.join(stale)}")
        if named:
            raise StaleInputError(f"Cannot combine: {detail}. Combining them "
                                  f"would use whatever an earlier run left "
                                  f"on disk.")

        by_target[target] = [well for well in wells if well in written]
        if by_target[target]:
            logger.warning(f"Combining only the wells written in this run, so "
                           f"that an earlier run's output is not mixed in: "
                           f"{detail}")

    if not any(by_target.values()):
        raise StaleInputError(
            f"Cannot combine: the filter step wrote no table for any of "
            f"{', '.join(wells)} in this run, so every table on disk is "
            f"from an earlier run."
        )
    return by_target


def resolve_steps(steps: Sequence[str]) -> List[str]:
    """Expand ``all`` into the individual steps, keeping pipeline order."""
    if 'all' in steps:
        return list(STEPS)

    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        raise ValueError(f"Unknown preprocessing step(s): {', '.join(unknown)}. "
                         f"Available steps: {', '.join(STEPS)}")
    return [step for step in STEPS if step in steps]


def run_preprocessing(config: Union[str, Dict[str, Any], BywellConfig,
                                    PreprocessingConfig],
                      wells: Optional[Sequence[str]] = None,
                      steps: Sequence[str] = ('all',),
                      components: Sequence[str] = ('all',),
                      targets: Sequence[str] = ('all',),
                      level: str = 'cell',
                      aggregate_levels: Optional[Sequence[str]] = None,
                      aggregation_method: str = 'median',
                      skip_prefilters: Sequence[str] = (),
                      validate: bool = False,
                      save_plots: bool = True,
                      plots_dir: Optional[str] = None,
                      continue_on_error: bool = False) -> PreprocessingResult:
    """Run the requested preprocessing steps.

    Args:
        config: A :class:`~chart.io.config.PreprocessingConfig`, or a
                configuration mapping or file path to build one from
        wells: Wells to process; ``None`` processes every well in the config
        steps: Which steps to run; ``('all',)`` runs everything in
               :data:`STEPS`
        components: For the QC step, which components to run
        targets: For the filter step, what to filter
        level: For the combine step, which index levels to keep
        aggregate_levels: Which aggregated tables the normalize and outliers
                          steps write alongside the cell-level one.  The
                          default is each step's own: see
                          :data:`DEFAULT_NORMALIZE_AGGREGATION` and
                          :data:`DEFAULT_OUTLIER_AGGREGATION`
        aggregation_method: How to combine the rows of a group
        skip_prefilters: For the outliers step, which prefilters
                      not to run
        validate: For the normalize step, draw the before-and-after
                  distributions
        save_plots: Whether to write plots
        plots_dir: Where to write plots (default: ``{qc_dir}/plots``)
        continue_on_error: Log and carry on when a well fails

    Returns:
        A :class:`PreprocessingResult` holding the results of each step
        that ran.
    """
    config = load_preprocessing_config(config)
    named_steps = 'all' not in steps
    steps = resolve_steps(steps)

    # An across-wells step writes one file for the whole screen, so running
    # it on a few named wells would overwrite that file with a subset.  It
    # takes asking for by name.
    if wells is not None and not named_steps:
        held_back = [step for step in steps if step in ALLWELLS_STEPS]
        if held_back:
            steps = [step for step in steps if step not in ALLWELLS_STEPS]
            logger.info(f"Not running {', '.join(held_back)}: combining writes one "
                        f"file for every well, and only some wells were named. "
                        f"Process all wells, or ask for the step by name to "
                        f"combine just these.")

    logger.info(f"Running steps: {', '.join(steps)}")

    result = PreprocessingResult()

    if 'qc' in steps:
        result.qc = run_qc(config.bywell,
                           wells=wells,
                           components=components,
                           save_plots=save_plots,
                           plots_dir=plots_dir,
                           continue_on_error=continue_on_error)

    if 'filter' in steps:
        # Naming wells makes their filter lists mandatory, whether or not
        # the QC step narrowed the list afterwards.
        filter_wells = _wells_for_filtering(result.qc) if result.qc else wells
        if result.qc and not filter_wells:
            logger.error("Not filtering: QC failed for every well in this run")
        else:
            result.filtering = run_filtering(config.bywell,
                                             wells=filter_wells,
                                             targets=targets,
                                             require_filters=wells is not None,
                                             continue_on_error=continue_on_error)

    if 'combine' in steps:
        # A well list from the config is a statement about the screen, not a
        # request for those particular wells, so a well the filter step never
        # reached is a warning there and an error when it was named.
        named = wells is not None
        combine_wells = _wells_for_combining(
            list(wells) if named else config.bywell.well_names(),
            result.filtering, named)
        result.combining = run_combining(config.allwells,
                                         wells=combine_wells,
                                         input_dir=config.bywell.filtered_path,
                                         level=level,
                                         require_wells=named,
                                         continue_on_error=continue_on_error)

    if 'normalize' in steps:
        stale = _stale_normalization_input(result.combining)
        if stale:
            if not continue_on_error:
                raise StaleInputError(stale)
            logger.error(stale)
            result.normalization = NormalizationResult(error=stale)
        else:
            try:
                result.normalization = run_normalization(
                    config.allwells,
                    aggregate_levels=(aggregate_levels
                                      if aggregate_levels is not None
                                      else DEFAULT_NORMALIZE_AGGREGATION),
                    aggregation_method=aggregation_method,
                    validate=validate,
                    save_plots=save_plots,
                    plots_dir=plots_dir)
            except Exception as exc:
                if not continue_on_error:
                    raise
                logger.error(f"Normalisation failed: {exc}")
                result.normalization = NormalizationResult(error=str(exc))

    if 'outliers' in steps:
        stale = _stale_outlier_input(result.normalization)
        if stale:
            if not continue_on_error:
                raise StaleInputError(stale)
            logger.error(stale)
            result.outliers = OutlierResult(error=stale)
        else:
            try:
                result.outliers = run_outlier_filtering(
                    config.allwells,
                    aggregate_levels=(aggregate_levels
                                      if aggregate_levels is not None
                                      else DEFAULT_OUTLIER_AGGREGATION),
                    aggregation_method=aggregation_method,
                    skip_prefilters=skip_prefilters,
                    save_plots=save_plots,
                    plots_dir=plots_dir)
            except Exception as exc:
                if not continue_on_error:
                    raise
                logger.error(f"Outlier filtering failed: {exc}")
                result.outliers = OutlierResult(error=str(exc))

    return result

"""
By-well preprocessing workflow.

This module runs the preprocessing steps for a set of wells.  Each step
lives in its own subpackage, such as :mod:`chart.qc`; this layer only
decides which of them run, and in what order.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .io.config import BywellConfig, load_bywell_config
from .filtering import WellFilterResult, run_filtering
from .qc import WellQCResult, run_qc

logger = logging.getLogger(__name__)

#: The by-well steps, in the order they run.  Merging is still to be
#: ported from cellmapp.
STEPS = ('qc', 'filter')


@dataclass
class PreprocessingResult:
    """What each step produced, for the steps that ran."""
    qc: List[WellQCResult] = field(default_factory=list)
    filtering: List[WellFilterResult] = field(default_factory=list)

    @property
    def failed(self) -> List[str]:
        """Wells that failed, in any step."""
        return [r.well for r in (*self.qc, *self.filtering) if r.error]


def resolve_steps(steps: Sequence[str]) -> List[str]:
    """Expand ``all`` into the individual steps, keeping pipeline order."""
    if 'all' in steps:
        return list(STEPS)

    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        raise ValueError(f"Unknown preprocessing step(s): {', '.join(unknown)}. "
                         f"Available steps: {', '.join(STEPS)}")
    return [step for step in STEPS if step in steps]


def run_preprocessing(config: Union[Dict[str, Any], BywellConfig],
                      wells: Optional[Sequence[str]] = None,
                      steps: Sequence[str] = ('all',),
                      components: Sequence[str] = ('all',),
                      targets: Sequence[str] = ('all',),
                      save_plots: bool = True,
                      plots_dir: Optional[str] = None,
                      continue_on_error: bool = False) -> PreprocessingResult:
    """Run the requested by-well preprocessing steps.

    Args:
        config: A :class:`~chart.io.config.BywellConfig`, or a
                configuration mapping to build one from
        wells: Wells to process; ``None`` processes every well in the config
        steps: Which steps to run; ``('all',)`` runs everything in
               :data:`STEPS`
        components: For the QC step, which components to run
        targets: For the filter step, what to filter
        save_plots: Whether to write plots
        plots_dir: Where to write plots (default: ``{qc_dir}/plots``)
        continue_on_error: Log and carry on when a well fails

    Returns:
        A :class:`PreprocessingResult` holding the results of each step
        that ran.
    """
    config = load_bywell_config(config)
    steps = resolve_steps(steps)
    logger.info(f"Running steps: {', '.join(steps)}")

    result = PreprocessingResult()

    if 'qc' in steps:
        result.qc = run_qc(config,
                           wells=wells,
                           components=components,
                           save_plots=save_plots,
                           plots_dir=plots_dir,
                           continue_on_error=continue_on_error)

    if 'filter' in steps:
        result.filtering = run_filtering(config,
                                         wells=wells,
                                         targets=targets,
                                         continue_on_error=continue_on_error)

    return result

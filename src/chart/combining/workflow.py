"""
The combine step: one table per target, across all wells.

Reads the filtered per-well tables written by the filter step and writes
``allwells{suffix}.parquet`` for each.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..io.config import AllwellsConfig, load_allwells_config
from ..io.tables import load_table, save_table, table_exists
from .wells import LEVELS, concat_wells

logger = logging.getLogger(__name__)

#: What can be combined, and the file suffix each one uses.
SUFFIXES = {'objects': '-objects', 'features': '-features', 'merged': ''}
TARGETS = tuple(SUFFIXES)

#: The stem of the combined files, kept from cellmapp because the
#: normalisation step reads them by name.
OUTPUT_NAME = 'allwells'


class MissingWellError(RuntimeError):
    """A well was asked for but the filter step left no table for it."""


@dataclass
class CombineResult:
    """Summary of combining one target."""
    target: str
    wells: List[str] = field(default_factory=list)
    objects: int = 0
    columns: int = 0
    output: Optional[str] = None
    skipped: Optional[str] = None
    error: Optional[str] = None


def resolve_targets(targets: Sequence[str]) -> List[str]:
    """Expand ``all`` into the individual targets."""
    if 'all' in targets:
        return list(TARGETS)

    unknown = [target for target in targets if target not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown combine target(s): {', '.join(unknown)}. "
                         f"Available targets: {', '.join(TARGETS)}")
    return [target for target in TARGETS if target in targets]


def combine_target(input_dir: str,
                   output_dir: str,
                   wells: Sequence[str],
                   target: str,
                   level: str = 'cell',
                   require_wells: bool = True) -> CombineResult:
    """Combine one target across *wells*.

    Args:
        input_dir: Where the filtered per-well tables are
        output_dir: Where to write the combined table
        wells: Wells to combine
        target: One of :data:`TARGETS`
        level: Which index levels to keep; see
               :data:`~chart.combining.wells.INDEX_LEVELS`
        require_wells: Fail when a well has no table, rather than skipping it

    Returns:
        A :class:`CombineResult`.  ``skipped`` is set, without an error, when
        no well had a table for this target, which means the filter step was
        not asked to produce it.

    Raises:
        MissingWellError: if a well has no table and *require_wells* is set
    """
    suffix = SUFFIXES[target]
    result = CombineResult(target=target)

    present = [well for well in wells if table_exists(input_dir, well, suffix)]
    missing = [well for well in wells if well not in present]

    if not present:
        result.skipped = (f"no {target} table was found for any well in "
                          f"{input_dir}; the filter step did not produce one")
        logger.warning(f"Not combining {target}: {result.skipped}")
        return result

    if missing:
        detail = (f"{target} table missing for {len(missing)} of {len(wells)} "
                  f"wells in {input_dir}: {', '.join(missing)}")
        if require_wells:
            raise MissingWellError(f"Cannot combine {target}: {detail}")
        logger.warning(f"Combining only the wells that have one: {detail}")

    logger.info(f"Combining {target} for {len(present)} wells: {', '.join(present)}")
    tables = {well: load_table(input_dir, well, suffix) for well in present}
    combined = concat_wells(tables, level)

    result.wells = present
    result.objects, result.columns = combined.shape
    result.output = save_table(combined, output_dir, OUTPUT_NAME, suffix)
    return result


def run_combining(config: Union[str, Dict[str, Any], AllwellsConfig],
                  wells: Union[Sequence[str], Mapping[str, Sequence[str]]],
                  input_dir: str,
                  targets: Sequence[str] = ('all',),
                  level: str = 'cell',
                  require_wells: bool = True,
                  continue_on_error: bool = False) -> List[CombineResult]:
    """Combine every requested target across *wells*.

    Args:
        config: An :class:`~chart.io.config.AllwellsConfig`, or a
                configuration mapping to build one from
        wells: Wells to combine, either one list for every target or a
               mapping from target to wells.  This is the whole point of
               the step, so it is required rather than defaulting to the
               configured wells
        input_dir: Where the filtered per-well tables are, normally
                   ``BywellConfig.filtered_path``
        targets: Which targets to combine; ``('all',)`` does each of
                 :data:`TARGETS` that the filter step produced
        level: Which index levels to keep
        require_wells: Fail when one of *wells* has no table.  Callers pass
                       ``False`` when the well list came from the config
                       rather than from the user
        continue_on_error: Log and carry on when a target fails

    Returns:
        A :class:`CombineResult` per target attempted.
    """
    config = load_allwells_config(config)
    targets = resolve_targets(targets)

    if not wells:
        raise ValueError("No wells to combine")
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. "
                         f"Available levels: {', '.join(LEVELS)}")

    logger.info(f"Combining {', '.join(targets)} at {level} level")
    logger.info(f"Reading from {input_dir}")
    logger.info(f"Writing to {config.combined_path}")

    results = []
    for target in targets:
        target_wells = (list(wells.get(target, ()))
                        if isinstance(wells, Mapping) else list(wells))
        if not target_wells:
            reason = "no well has one from this run"
            logger.warning(f"Not combining {target}: {reason}")
            results.append(CombineResult(target=target, skipped=reason))
            continue

        logger.info(f"--- Combining {target} across {len(target_wells)} "
                    f"wells ---")
        try:
            results.append(combine_target(input_dir, config.combined_path,
                                          target_wells, target, level=level,
                                          require_wells=require_wells))
        except Exception as exc:
            if not continue_on_error:
                raise
            logger.error(f"Failed to combine {target}: {exc}")
            results.append(CombineResult(target=target, error=str(exc)))

    return results

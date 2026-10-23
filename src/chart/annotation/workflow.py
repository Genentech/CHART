"""
The archetype annotation stage.

Optional, and run between ``chart archetypes`` and ``chart effects``:
nothing after it reads what it writes.  It describes each archetype from
the grids the crops step drew, with a language model, in the steps of
:mod:`.steps`.  :mod:`.llm` says which keys it reads.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from ..archetypes.crops import CELLS_PER_GRID, CROPS_DIR, TOP_CELLS_NAME
from ..io.config import (ANNOTATION_SECTION, ARCHETYPES_SECTION,
                         AnnotationConfig, ArchetypesConfig,
                         load_annotation_config, load_archetypes_config)
from . import steps as annotation_steps
from .llm import check_llm_credentials

logger = logging.getLogger(__name__)

#: The steps, in the order they have to run.
STEPS = ('annotate', 'baseline', 'synthesize', 'refine', 'dashboard')

#: Grids described per archetype; the first one is the one compared.
GRIDS = 2


def channel_grids(archetypes: ArchetypesConfig, channel: str) -> Dict[str, List[Path]]:
    """Each archetype's grid images, as the crops step drew them, leaving
    out the archetypes ``archetypes_to_remove`` lists."""
    directory = Path(archetypes.report_path(channel)) / CROPS_DIR
    listing = directory / f'{TOP_CELLS_NAME}.parquet'
    if not listing.exists():
        raise FileNotFoundError(f"No {listing.name} in {directory}; run the "
                                f"crops step of chart archetypes for {channel}")

    removed = {name for names in archetypes.archetypes_to_remove.values()
               for name in names}
    shown = pd.read_parquet(listing, columns=['Archetype', 'Rank'])
    grids = {}
    for archetype, ranks in shown.groupby('Archetype', sort=False)['Rank']:
        if f'{channel}_{archetype}' in removed:
            continue
        number = archetype[len('Archetype'):]
        rows = min(GRIDS, -(-int(ranks.max()) // CELLS_PER_GRID))
        paths = [directory / f'archetype{number}_{channel}_grid_{row}.png'
                 for row in range(1, rows + 1)]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{listing.name} lists archetype {number}, "
                                    f"but {', '.join(missing)} is not in "
                                    f"{directory}; rerun the crops step")
        grids[number] = paths
    return grids


def _normal_ids(annotation: AnnotationConfig, channel: str,
                grids: Dict[str, List[Path]]) -> List[str]:
    """The archetypes the baseline step compares every other with."""
    numbers = annotation.normal_archetypes.get(channel) or []
    normal = [str(n) for n in (numbers if isinstance(numbers, list) else [numbers])]
    if not normal or any(number not in grids for number in normal):
        raise ValueError(f"'{ANNOTATION_SECTION}.normal_archetypes' has to name "
                         f"archetypes of {channel} that the crops step drew and "
                         f"'{ARCHETYPES_SECTION}.archetypes_to_remove' keeps; "
                         f"it names {', '.join(normal) or 'none'}")
    return normal


def annotate_channel(archetypes: ArchetypesConfig, annotation: AnnotationConfig,
                     channel: str, steps: Sequence[str],
                     resume: bool = False) -> str:
    """Run *steps* on one channel, returning the directory they wrote to."""
    grids = channel_grids(archetypes, channel)
    normal = _normal_ids(annotation, channel, grids) if 'baseline' in steps else []
    directory = Path(annotation.channel_path(channel))

    for step in steps:
        logger.info(f"{channel}: {step}")
        if step == 'annotate':
            annotation_steps.annotate(annotation.model, channel, grids,
                                      directory, resume)
        elif step == 'baseline':
            annotation_steps.baseline(annotation.model, channel, grids, normal,
                                      directory)
        elif step == 'synthesize':
            annotation_steps.synthesize(annotation.model, channel, grids,
                                        directory)
        elif step == 'refine':
            annotation_steps.refine(annotation.model, channel, grids, directory,
                                    annotation.similarity_threshold,
                                    annotation.max_pairs_per_archetype)
        else:
            annotation_steps.dashboard(channel, grids, directory)
    return str(directory)


def run_annotation(config: Union[str, Dict[str, Any]],
                   channels: Optional[Sequence[str]] = None,
                   steps: Sequence[str] = ('all',),
                   resume: bool = False,
                   continue_on_error: bool = False) -> Dict[str, Optional[str]]:
    """Describe each channel's archetypes from their crops.

    Args:
        config: A configuration file or mapping; the channels, crops and
                removal list come from its archetypes section
        channels: Channels to run; ``None`` runs every channel in the config
        steps: Which of :data:`STEPS` to run; ``('all',)`` runs them all
        resume: For the annotate step, keep the reports already written
        continue_on_error: Log and carry on when a channel fails

    Returns:
        Each channel's annotation directory, or ``None`` where it failed.
    """
    archetypes = load_archetypes_config(config)
    annotation = load_annotation_config(config)
    channels = list(channels) if channels else list(archetypes.channels)
    if not channels:
        raise ValueError(f"No channels to run: name them with --channel or "
                         f"under '{ARCHETYPES_SECTION}.channels'")

    wanted = STEPS if 'all' in steps else tuple(s for s in STEPS if s in steps)
    if wanted != ('dashboard',):
        logger.info(f"Provider: {check_llm_credentials(model=annotation.model)}")

    outputs: Dict[str, Optional[str]] = {}
    for channel in channels:
        try:
            outputs[channel] = annotate_channel(archetypes, annotation, channel,
                                                wanted, resume)
        except Exception as error:
            if not continue_on_error:
                raise
            outputs[channel] = None
            logger.error(f"Annotation failed for channel {channel}: {error}")

    failed = [channel for channel, output in outputs.items() if output is None]
    logger.info(f"Annotation complete: {len(outputs) - len(failed)}/"
                f"{len(outputs)} channels succeeded")
    return outputs

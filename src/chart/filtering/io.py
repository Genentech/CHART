"""
Input/output for filtering.

Reads the filter lists written by QC and the object and feature tables
written by the merge step, and writes the filtered results.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..io.filters import INCLUSION, load_manifest, manifest_from_directory

logger = logging.getLogger(__name__)


def load_filter_lists(filter_dir: str,
                      well: str) -> Tuple[List[pd.Series], List[pd.Series]]:
    """Load the QC filter lists for a well.

    The manifest QC writes says which filters exist and whether each one
    includes or excludes.  A directory without a manifest is read by
    looking for the filenames QC has always produced; anything outside
    that set needs a manifest to be seen at all.

    Returns:
        The inclusion and exclusion filters, as lists of label Series.
        Either may be empty.
    """
    manifest = load_manifest(filter_dir, well)
    if manifest is None:
        manifest = manifest_from_directory(filter_dir, well)
        if manifest.filters:
            logger.warning(f"No filter manifest for well {well}; falling back "
                           f"to the filter files present in {filter_dir}")
    else:
        logger.info(f"Filter manifest written {manifest.created} "
                    f"by CHART {manifest.chart_version}")
        if manifest.components_skipped:
            logger.warning(
                f"QC skipped {', '.join(manifest.components_skipped)} for well "
                f"{well}, so no filters from those components are applied"
            )

    inclusion, exclusion = [], []
    for entry in manifest.filters:
        # A file named in the manifest is expected to exist, so a missing one
        # is an error rather than something to pass over.
        labels = pd.read_parquet(os.path.join(filter_dir, entry.file))['label']
        target = inclusion if entry.role == INCLUSION else exclusion
        target.append(labels)
        logger.info(f"Loaded {entry.role} filter '{entry.name}': "
                    f"{len(labels)} labels")

    return inclusion, exclusion


def _reject_collisions(columns: pd.Index, renames: Dict[str, str]) -> None:
    """Refuse a rename that would give two columns the same name.

    A table may well hold a column already called ``label`` that means
    something other than the cell identifier.  Renaming onto it would
    leave the steps reading whichever of the two came first, so the
    clash has to be settled outside CHART.
    """
    produced: Dict[str, List[str]] = {}
    for source in columns:
        produced.setdefault(renames.get(source, source), []).append(source)

    # Duplicates the table already had are left alone; only a name this
    # rename is about to create counts.
    clashes = {name: sources for name, sources in produced.items()
               if len(sources) > 1 and any(s in renames for s in sources)}
    if clashes:
        detail = '; '.join(
            f"'{name}' from " + ' and '.join(f"'{source}'" for source in sources)
            for name, sources in clashes.items())
        raise ValueError(
            f"Renaming would give two columns the same name: {detail}. "
            f"Rename or drop the column standing in the way before CHART "
            f"reads the table.")


def rename_columns(data: pd.DataFrame,
                   column_mapping: Optional[Dict[str, str]] = None,
                   column_prefix_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Bring column names into CHART's vocabulary.

    Raises:
        ValueError: If a rename would collide with a column already
            present under that name.
    """
    if column_mapping:
        present = {k: v for k, v in column_mapping.items() if k in data.columns}
        if present:
            _reject_collisions(data.columns, present)
            data = data.rename(columns=present)
            logger.info(f"Renamed {len(present)} columns via column_mapping")

    if column_prefix_mapping:
        renames = {}
        for col in data.columns:
            for old_prefix, new_prefix in column_prefix_mapping.items():
                if col.startswith(old_prefix):
                    renames[col] = new_prefix + col[len(old_prefix):].lower()
                    break
        if renames:
            _reject_collisions(data.columns, renames)
            data = data.rename(columns=renames)
            logger.info(f"Prefix-renamed {len(renames)} columns")

    return data



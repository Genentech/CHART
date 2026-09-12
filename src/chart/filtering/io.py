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


def rename_columns(data: pd.DataFrame,
                   column_mapping: Optional[Dict[str, str]] = None,
                   column_prefix_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Bring column names into CHART's vocabulary.
    """
    if column_mapping:
        present = {k: v for k, v in column_mapping.items() if k in data.columns}
        if present:
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
            data = data.rename(columns=renames)
            logger.info(f"Prefix-renamed {len(renames)} columns")

    return data


def load_table(input_dir: str, well: str, suffix: str = '') -> pd.DataFrame:
    """Read ``{well}{suffix}.parquet`` from *input_dir*."""
    path = os.path.join(input_dir, f"{well}{suffix}.parquet")
    logger.info(f"Reading {path}")
    data = pd.read_parquet(path)
    logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    return data


def save_table(data: pd.DataFrame, output_dir: str, well: str,
               suffix: str = '') -> str:
    """Write *data* to ``{well}{suffix}.parquet`` in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{well}{suffix}.parquet")
    data.to_parquet(path)
    logger.info(f"Saved {len(data)} rows to {path}")
    return path

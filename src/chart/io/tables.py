"""
Reading and writing the parquet tables that steps pass between each other.

Files are named ``{name}{suffix}.parquet``, where the name is usually a
well identifier and the suffix says which table it is.  A caller whose
inputs are named differently supplies a pattern holding ``{well}``.

A directory may be local or a URL such as ``s3://bucket/prefix``.  The
latter needs fsspec and the driver for that filesystem (``s3fs`` for S3),
which is why the import sits inside :func:`exists` rather than at the top.
"""

import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def exists(path: str) -> bool:
    """Whether *path* is present, locally or on the filesystem it names."""
    if '://' not in path:
        return os.path.exists(path)

    import fsspec
    filesystem, inner = fsspec.core.url_to_fs(path)
    return filesystem.exists(inner)


def list_dir(path: str) -> List[str]:
    """The filenames directly inside *path*, or nothing if it is absent."""
    if '://' not in path:
        return os.listdir(path) if os.path.isdir(path) else []

    import fsspec
    filesystem, inner = fsspec.core.url_to_fs(path)
    if not filesystem.isdir(inner):
        return []
    return [entry.rsplit('/', 1)[-1] for entry in filesystem.ls(inner)]


def table_path(input_dir: str, name: str, suffix: str = '',
               pattern: Optional[str] = None) -> str:
    """Where the table for *name* lives.

    Args:
        input_dir: The directory holding it, local or a URL
        name: Usually a well identifier
        suffix: Which table it is
        pattern: A filename holding ``{well}``, used in place of the
                 ``{name}{suffix}.parquet`` convention

    Returns:
        The full path, whether or not anything is there.

    Raises:
        ValueError: If the pattern names a format CHART cannot read, or
            holds a placeholder other than ``{well}``.
    """
    if pattern is None:
        return os.path.join(input_dir, f"{name}{suffix}.parquet")

    if not pattern.endswith('.parquet'):
        raise ValueError(f"Filename pattern '{pattern}' does not name a "
                         f"parquet file, which is the only format CHART "
                         f"reads; convert the inputs first")
    try:
        filename = pattern.format(well=name)
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Filename pattern '{pattern}' holds a placeholder "
                         f"CHART cannot fill ({exc}); '{{well}}' is the only "
                         f"one available") from exc

    return os.path.join(input_dir, filename)


def load_table(input_dir: str, name: str, suffix: str = '',
               pattern: Optional[str] = None) -> pd.DataFrame:
    """Read the table for *name* from *input_dir*."""
    path = table_path(input_dir, name, suffix, pattern)
    if not exists(path):
        raise FileNotFoundError(f"No table at {path}")

    logger.info(f"Reading {path}")
    data = pd.read_parquet(path)
    logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    return data


def save_table(data: pd.DataFrame, output_dir: str, name: str,
               suffix: str = '') -> str:
    """Write *data* to ``{name}{suffix}.parquet`` in *output_dir*."""
    if '://' not in output_dir:
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}{suffix}.parquet")
    data.to_parquet(path)
    logger.info(f"Saved {len(data)} rows to {path}")
    return path


def table_exists(input_dir: str, name: str, suffix: str = '',
                 pattern: Optional[str] = None) -> bool:
    """Whether the table for *name* is present in *input_dir*."""
    return exists(table_path(input_dir, name, suffix, pattern))

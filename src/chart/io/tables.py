"""
Reading and writing the parquet tables that steps pass between each other.

Files are named ``{name}{suffix}.parquet``, where the name is usually a
well identifier and the suffix says which table it is.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def load_table(input_dir: str, name: str, suffix: str = '') -> pd.DataFrame:
    """Read ``{name}{suffix}.parquet`` from *input_dir*."""
    path = os.path.join(input_dir, f"{name}{suffix}.parquet")
    logger.info(f"Reading {path}")
    data = pd.read_parquet(path)
    logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    return data


def save_table(data: pd.DataFrame, output_dir: str, name: str,
               suffix: str = '') -> str:
    """Write *data* to ``{name}{suffix}.parquet`` in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}{suffix}.parquet")
    data.to_parquet(path)
    logger.info(f"Saved {len(data)} rows to {path}")
    return path


def table_exists(input_dir: str, name: str, suffix: str = '') -> bool:
    """Whether ``{name}{suffix}.parquet`` is present in *input_dir*."""
    return os.path.exists(os.path.join(input_dir, f"{name}{suffix}.parquet"))

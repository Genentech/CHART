"""
Cosine similarity between rows of a reduced space.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(centered: pd.DataFrame) -> pd.DataFrame:
    """Compare every row of *centered* with every other.

    Args:
        centered: A space centred on the controls, rows by components

    Returns:
        A square matrix indexed and columned by the rows of *centered*,
        holding their pairwise cosine similarity.

    Raises:
        ValueError: if *centered* has no rows or no columns
    """
    if not len(centered) or not len(centered.columns):
        raise ValueError(f"Cannot compare rows: the table is "
                         f"{len(centered)} by {len(centered.columns)}")

    if centered.index.has_duplicates:
        duplicates = int(centered.index.duplicated().sum())
        logger.warning(f"{duplicates} duplicate row label(s): the matrix will "
                       f"have repeated labels, and selecting one will return "
                       f"several")

    values = centered.to_numpy(dtype=np.float64)

    # A row of zeros is a guide sitting exactly on the control centroid.
    # It has no direction, and sklearn scores it 0 against everything,
    # which reads as 'unrelated' rather than 'no signal'.
    at_origin = int(np.isclose(np.linalg.norm(values, axis=1), 0).sum())
    if at_origin:
        logger.warning(f"{at_origin} of {len(centered)} rows sit on the "
                       f"control centroid and have no direction; they score 0 "
                       f"against everything, which is indistinguishable from "
                       f"being unrelated")

    logger.info(f"Comparing {len(centered)} rows over "
                f"{len(centered.columns)} components")

    return pd.DataFrame(cosine_similarity(values),
                        index=centered.index,
                        columns=centered.index)

"""
Principal component analysis over a feature table.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Percentage of the variance the retained components should cover.
DEFAULT_VARIANCE_THRESHOLD = 90.0

#: Rows per batch when fitting.
DEFAULT_BATCH_SIZE = 2000

#: Fewer rows than this and the variance is undefined.
MIN_SAMPLES = 2

#: The index level naming the gene a guide targets.
GENE_LEVEL = 'Gene'

#: Genes that carry no perturbation: the non-targeting controls, and the
#: olfactory receptors, which are not expressed in these cells.
CONTROL_GENE = 'NTC'
CONTROL_PREFIX = 'OR'


class NoComponentsError(ValueError):
    """The variance threshold retained no components."""


class NoControlsError(ValueError):
    """The table holds no control guides to centre on."""


def generate_pca_space(data: pd.DataFrame,
                       variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
                       batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    """Standardise *data* and reduce it to its leading principal components.

    Args:
        data: Feature table, samples by features
        variance_threshold: Percentage of variance to cover, 0 to 100
        batch_size: Rows per batch while fitting

    Returns:
        The components, indexed as *data* and with columns ``PC1``, ``PC2``
        and so on.

    Raises:
        ValueError: if *data* has fewer than :data:`MIN_SAMPLES` rows
        NoComponentsError: if the threshold retains no components
    """
    if len(data) < MIN_SAMPLES:
        raise ValueError(f"Cannot run PCA on {len(data)} row(s): at least "
                         f"{MIN_SAMPLES} are needed for the variance to be "
                         f"defined")

    logger.info(f"PCA on {len(data)} rows and {len(data.columns)} features, "
                f"keeping components up to {variance_threshold}% of variance")

    scaled = StandardScaler().fit_transform(data)

    # Only the fit is batched.  Scaling builds the whole matrix and the
    # transform below runs on all of it, so the peak is the full matrix
    # either way; see OPEN_ISSUES.md.
    ipca = IncrementalPCA(batch_size=batch_size).fit(scaled)
    transformed = ipca.transform(scaled)

    explained = ipca.explained_variance_ratio_ * 100
    cumulative = explained.cumsum()
    n_components = int((cumulative <= variance_threshold).sum())

    if n_components == 0:
        raise NoComponentsError(
            f"No components retained: PC1 alone explains "
            f"{explained[0]:.2f}% of the variance, which already exceeds the "
            f"{variance_threshold}% threshold, and the count only takes "
            f"components that stay below it. Raise variance_threshold above "
            f"{explained[0]:.2f}.")

    logger.info(f"Keeping {n_components} of {len(explained)} components, "
                f"explaining {cumulative[n_components - 1]:.2f}% of the "
                f"variance")

    return pd.DataFrame(transformed[:, :n_components],
                        index=data.index,
                        columns=[f'PC{i + 1}' for i in range(n_components)])


def control_mask(genes: Union[pd.Index, pd.Series]) -> np.ndarray:
    """Which entries of *genes* name a control.

    Args:
        genes: The gene each row targets

    Returns:
        A boolean array, true where the gene is :data:`CONTROL_GENE` or
        starts with :data:`CONTROL_PREFIX`.
    """
    values = pd.Index(genes).astype(str)
    return np.asarray((values == CONTROL_GENE)
                      | values.str.startswith(CONTROL_PREFIX, na=False))


def center_on_controls(pca_data: pd.DataFrame) -> pd.DataFrame:
    """Shift the PCA space so the control centroid sits at the origin.

    Args:
        pca_data: A PCA space indexed with a :data:`GENE_LEVEL` level

    Returns:
        The same space, with the control centroid subtracted.

    Raises:
        KeyError: if the index has no :data:`GENE_LEVEL` level
        NoControlsError: if no row is a control
    """
    if GENE_LEVEL not in (pca_data.index.names or ()):
        raise KeyError(f"Cannot centre on controls: the table has no "
                       f"'{GENE_LEVEL}' index level, only "
                       f"{', '.join(str(n) for n in pca_data.index.names or ())}")

    genes = pca_data.index.get_level_values(GENE_LEVEL)
    controls = control_mask(genes)

    if not controls.any():
        raise NoControlsError(
            f"Cannot centre on controls: none of the {len(pca_data)} rows "
            f"target {CONTROL_GENE} or a gene starting with "
            f"{CONTROL_PREFIX}. Genes seen: "
            f"{', '.join(str(g) for g in pd.unique(genes)[:5]) or 'none'}"
            + (', and more' if pca_data.index.get_level_values(
                GENE_LEVEL).nunique() > 5 else '') + ".")

    centroid = pca_data[controls].mean(axis=0)
    logger.info(f"Centring on {int(controls.sum())} control guides of "
                f"{len(pca_data)}")
    return pca_data - centroid

"""
Dimensionality reduction, shared by the steps that need a reduced space.
"""

from .channels import (
    GEOMETRY_PATTERNS,
    NoFeaturesError,
    channel_features,
    write_channel_pca
)
from .cosinesim import cosine_similarity_matrix
from .pca import (
    CONTROL_GENE,
    CONTROL_PREFIX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_VARIANCE_THRESHOLD,
    GENE_LEVEL,
    NoComponentsError,
    NoControlsError,
    center_on_controls,
    control_mask,
    generate_pca_space
)

__all__ = [
    'generate_pca_space',
    'channel_features',
    'write_channel_pca',
    'NoFeaturesError',
    'GEOMETRY_PATTERNS',
    'center_on_controls',
    'control_mask',
    'cosine_similarity_matrix',
    'NoComponentsError',
    'NoControlsError',
    'CONTROL_GENE',
    'CONTROL_PREFIX',
    'GENE_LEVEL',
    'DEFAULT_VARIANCE_THRESHOLD',
    'DEFAULT_BATCH_SIZE'
]

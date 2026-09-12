"""
Checking that normalisation did what it was supposed to.

Quantile normalisation should leave every well with the same distribution
for a given feature, so the check is to draw the per-well distributions
before and after and look at them.
"""

import logging
import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def choose_features(data: pd.DataFrame, count: int = 5,
                    random_state: Optional[int] = None) -> List[str]:
    """Pick *count* numeric columns at random, or all of them if fewer.
    """
    numeric = data.select_dtypes(include=[np.number]).columns
    if len(numeric) <= count:
        return list(numeric)
    rng = np.random.default_rng(random_state)
    return list(rng.choice(numeric, count, replace=False))


def plot_normalization(original: pd.DataFrame,
                       normalized: pd.DataFrame,
                       well_column: str = 'Well',
                       features: Optional[Sequence[str]] = None,
                       sample_features: int = 5,
                       random_state: Optional[int] = None,
                       save_plots: bool = False,
                       output_dir: Optional[str] = None) -> None:
    """Draw per-well distributions before and after normalisation.

    Args:
        original: The table as it was before normalising
        normalized: The table after normalising
        well_column: The index level naming the well
        features: Features to draw; the default samples them
        sample_features: How many to sample when *features* is not given
        random_state: Seed for that sampling
        save_plots: Whether to write the figure
        output_dir: Where to write it
    """
    if features is None:
        features = choose_features(original, sample_features, random_state)
    features = [f for f in features if f in normalized.columns]
    if not features:
        logger.warning("No features in common between the two tables; "
                       "nothing to plot")
        return

    logger.info(f"Validating normalisation with features: {', '.join(features)}")

    fig, axes = plt.subplots(len(features), 2,
                             figsize=(12, 3 * len(features)), squeeze=False)

    for i, feature in enumerate(features):
        for column, (frame, label) in enumerate(((original, 'Original'),
                                                 (normalized, 'Normalized'))):
            sns.kdeplot(data=frame, x=feature, hue=well_column, ax=axes[i, column])
            axes[i, column].set_title(f'{label}: {feature}')
            axes[i, column].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'normalization_validation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved validation plots: {plot_path}")

    plt.show()
    plt.close(fig)

    missing = int(normalized.isna().sum().sum())
    logger.info(f"Original shape {original.shape}, "
                f"normalised shape {normalized.shape}")
    if missing:
        logger.warning(f"The normalised data holds {missing} missing value(s)")

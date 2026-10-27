"""
Plots describing which guides were dropped and why.
"""

import logging
import os
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .controls import GUIDE_LEVEL

logger = logging.getLogger(__name__)

#: The two components the control scatter is drawn in.
PLOT_COMPONENTS = ('PC1', 'PC2')


def _save(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_detector_votes(votes: pd.Series, output_dir: str) -> None:
    """Bar chart of how many detectors flagged each control guide.

    Args:
        votes: Detector count per guide, indexed by the label to show
        output_dir: Where to write the plot
    """
    if not len(votes):
        logger.info("No control guide was flagged by any detector, so there "
                    "is no vote plot to draw")
        return

    ordered = votes.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(ordered)), ordered.to_numpy(),
                  color='skyblue', alpha=0.7)
    ax.set_xlabel('Guide')
    ax.set_ylabel('Detectors flagging the guide')
    ax.set_title('Control guides by detector agreement')
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([str(label) for label in ordered.index],
                       rotation=45, ha='right')
    for bar, count in zip(bars, ordered.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    _save(fig, output_dir, 'control_guide_detector_votes.png')


def plot_control_outliers(controls_pca: pd.DataFrame,
                          outlier_guides: Sequence[str],
                          output_dir: str) -> None:
    """Scatter the control guides, marking the ones being dropped.

    Args:
        controls_pca: The control guides in the reduced space
        outlier_guides: The guide names being dropped
        output_dir: Where to write the plot
    """
    missing = [c for c in PLOT_COMPONENTS if c not in controls_pca.columns]
    if missing:
        logger.warning(f"Not drawing the control scatter: the reduced space "
                       f"has no {', '.join(missing)}, only "
                       f"{', '.join(map(str, controls_pca.columns))}")
        return

    is_outlier = controls_pca.index.get_level_values(GUIDE_LEVEL).isin(
        list(outlier_guides))

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, mask, colour, size in (
            ('Control', ~is_outlier, '#9F8170', 20),
            ('Outlier control', is_outlier, '#FF7900', 50)):
        if mask.any():
            ax.scatter(controls_pca.loc[mask, PLOT_COMPONENTS[0]],
                       controls_pca.loc[mask, PLOT_COMPONENTS[1]],
                       c=colour, label=label, s=size, alpha=0.7)

    ax.set_xlabel(PLOT_COMPONENTS[0])
    ax.set_ylabel(PLOT_COMPONENTS[1])
    ax.set_title('Control guides, with the ones being dropped marked')
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_dir, 'control_guide_outliers.png')


def plot_similarity_comparison(scores: pd.DataFrame, threshold: float,
                               output_dir: str) -> None:
    """Overlay the guides' similarity to their gene and to the controls.

    Args:
        scores: The ``scores`` of a
                :class:`~chart.guides.perturbations.SimilarityResult`
        threshold: The cut-off being applied, drawn as a line
        output_dir: Where to write the plot
    """
    from .perturbations import NTC_SIMILARITY_COLUMN, SIMILARITY_COLUMN

    if not len(scores):
        logger.info("No guide was scored, so there is no similarity plot")
        return

    finite = pd.concat([scores[SIMILARITY_COLUMN],
                        scores[NTC_SIMILARITY_COLUMN]]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    for column, label, colour in (
            (SIMILARITY_COLUMN, 'Same gene', '#1F77B4'),
            (NTC_SIMILARITY_COLUMN, 'Non-targeting controls', '#9F8170')):
        values = scores[column].dropna()
        if len(values):
            ax.hist(values, bins=30, range=(finite.min(), finite.max()),
                    color=colour, alpha=0.6, label=label)

    ax.axvline(threshold, color='k', linestyle='--',
               label=f'Threshold {threshold}')
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Guides')
    ax.legend(frameon=False)
    _save(fig, output_dir, 'guide_similarity_comparison.png')

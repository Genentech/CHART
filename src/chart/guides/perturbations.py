"""
Finding perturbation guides that disagree with their own gene.
A guide is scored by its mean cosine similarity to the other guides
against the same gene.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from ..dimred import CONTROL_GENE, GENE_LEVEL, control_mask
from .controls import GUIDE_LEVEL

logger = logging.getLogger(__name__)

#: Each guide's mean similarity to the others against its gene.
SIMILARITY_COLUMN = 'Similarity'

#: Its mean similarity to the non-targeting controls, for the plot.  This
#: uses the non-targeting controls only, not the wider control set the
#: centring uses.
NTC_SIMILARITY_COLUMN = 'Similarity to NTC'


@dataclass
class SimilarityResult:
    """Scores, and what could not be scored."""
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Genes with a single guide, which cannot be compared with anything.
    single_guide_genes: List[str] = field(default_factory=list)
    #: The guides those genes carry, which no threshold can exclude.
    unscored_guides: int = 0


def guide_similarities(cosine: pd.DataFrame) -> SimilarityResult:
    """Score each perturbation guide against the others targeting its gene.

    Args:
        cosine: The square similarity matrix, indexed by Guide and Gene

    Returns:
        A :class:`SimilarityResult` whose ``scores`` carry
        :data:`SIMILARITY_COLUMN` and :data:`NTC_SIMILARITY_COLUMN`,
        indexed as *cosine*.
    """
    genes = cosine.index.get_level_values(GENE_LEVEL)
    values = cosine.to_numpy()

    ntc = np.asarray(genes == CONTROL_GENE)
    if not ntc.any():
        logger.warning(f"No {CONTROL_GENE} guides, so "
                       f"'{NTC_SIMILARITY_COLUMN}' will be empty; it is only "
                       f"plotted, so the filtering is unaffected")
        ntc_means = np.full(len(cosine), np.nan)
    else:
        ntc_means = values[:, ntc].mean(axis=1)

    targets = pd.unique(np.asarray(genes)[~control_mask(genes)])
    logger.info(f"Scoring guides against {len(targets)} non-control gene(s)")

    result = SimilarityResult()
    positions, scores, controls_scores = [], [], []

    for gene in targets:
        rows = np.flatnonzero(np.asarray(genes) == gene)
        if len(rows) < 2:
            result.single_guide_genes.append(str(gene))
            result.unscored_guides += len(rows)
            continue

        block = values[np.ix_(rows, rows)]
        # Every guide's similarity to the others against this gene: the
        # row total less the guide's similarity to itself.
        off_diagonal = (block.sum(axis=1) - np.diag(block)) / (len(rows) - 1)

        positions.append(rows)
        scores.append(off_diagonal)
        controls_scores.append(ntc_means[rows])

    if result.single_guide_genes:
        logger.warning(
            f"{len(result.single_guide_genes)} gene(s) carry a single guide, "
            f"so their {result.unscored_guides} guide(s) have nothing to be "
            f"compared with and cannot be excluded by this filter: "
            f"{', '.join(result.single_guide_genes[:5])}"
            f"{', and more' if len(result.single_guide_genes) > 5 else ''}")

    if not positions:
        logger.warning("No gene carries more than one guide, so no guide "
                       "could be scored")
        result.scores = pd.DataFrame(
            columns=[SIMILARITY_COLUMN, NTC_SIMILARITY_COLUMN],
            index=cosine.index[[]])
        return result

    order = np.concatenate(positions)
    result.scores = pd.DataFrame(
        {SIMILARITY_COLUMN: np.concatenate(scores),
         NTC_SIMILARITY_COLUMN: np.concatenate(controls_scores)},
        index=cosine.index[order])
    logger.info(f"Scored {len(result.scores)} guide(s)")
    return result


def guides_below_threshold(scores: pd.DataFrame,
                           threshold: float) -> List[str]:
    """The guides agreeing with their gene less than *threshold*.

    Args:
        scores: The ``scores`` of a :class:`SimilarityResult`
        threshold: The cosine similarity a guide must reach

    Returns:
        The names of the guides to exclude.
    """
    if not len(scores):
        return []

    below = scores[scores[SIMILARITY_COLUMN] < threshold]
    guides = pd.unique(below.index.get_level_values(GUIDE_LEVEL)).tolist()
    logger.info(f"{len(guides)} of {len(scores)} scored guide(s) fall below "
                f"{threshold} "
                f"({len(guides) / len(scores) * 100:.2f}%)")
    return guides

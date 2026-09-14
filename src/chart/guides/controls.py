"""
Finding control guides that do not behave like controls.

Control guides carry no perturbation, so they should sit together near
the centre of the space. 
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ..dimred import control_mask

logger = logging.getLogger(__name__)

#: The detectors that vote.
ISOLATION_FOREST = 'isolation_forest'
LOCAL_OUTLIER_FACTOR = 'local_outlier_factor'
ELLIPTIC_ENVELOPE = 'elliptic_envelope'
METHODS = (ISOLATION_FOREST, LOCAL_OUTLIER_FACTOR, ELLIPTIC_ENVELOPE)

#: How many of them must agree before a guide is dropped.
MIN_METHODS = 2

#: The index level naming a guide.
GUIDE_LEVEL = 'Guide'

#: Defaults carried over from cellmapp, which has no config key for them.
N_ESTIMATORS = 100
RANDOM_STATE = 42
N_NEIGHBORS = 20


class NoControlGuidesError(ValueError):
    """The table holds no control guides to examine."""


@dataclass
class ControlResult:
    """What the vote found."""
    controls: int = 0
    #: Detector to the number of guides it flagged.
    flagged: Dict[str, int] = field(default_factory=dict)
    #: Detector to the reason it did not run.
    failed: Dict[str, str] = field(default_factory=dict)
    #: Guide to the number of detectors that flagged it.
    votes: Dict[str, int] = field(default_factory=dict)
    #: The guides at least :data:`MIN_METHODS` detectors agreed on.
    outliers: List[str] = field(default_factory=list)

    @property
    def methods_run(self) -> List[str]:
        return [method for method in METHODS if method not in self.failed]


def subset_controls(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the control guides.

    Args:
        data: A guide-level table indexed by Guide and Gene

    Returns:
        The rows whose gene is a control.

    Raises:
        NoControlGuidesError: if no row is a control
    """
    from ..dimred import CONTROL_GENE, CONTROL_PREFIX, GENE_LEVEL

    if GENE_LEVEL not in (data.index.names or ()):
        raise KeyError(f"Cannot select controls: the table has no "
                       f"'{GENE_LEVEL}' index level, only "
                       f"{', '.join(str(n) for n in data.index.names or ())}")

    controls = data[control_mask(data.index.get_level_values(GENE_LEVEL))]
    if not len(controls):
        raise NoControlGuidesError(
            f"No control guides among {len(data)} rows: none target "
            f"{CONTROL_GENE} or a gene starting with {CONTROL_PREFIX}. "
            f"Without controls there is no reference to filter against.")

    logger.info(f"{len(controls)} control guides of {len(data)}")
    return controls


def detect_isolation_forest(controls: pd.DataFrame,
                            n_estimators: int = N_ESTIMATORS,
                            random_state: int = RANDOM_STATE) -> pd.Index:
    """Flag controls that are easy to isolate in the original features."""
    forest = IsolationForest(n_estimators=n_estimators, contamination='auto',
                             random_state=random_state)
    return controls.index[forest.fit_predict(controls) == -1]


def detect_local_outlier_factor(controls_pca: pd.DataFrame,
                                n_neighbors: int = N_NEIGHBORS) -> pd.Index:
    """Flag controls whose local density is below their neighbours'."""
    if len(controls_pca) <= n_neighbors:
        logger.warning(f"Only {len(controls_pca)} control guides for "
                       f"n_neighbors={n_neighbors}; the neighbourhood is the "
                       f"whole set, so the factor is close to meaningless")
    factor = LocalOutlierFactor(n_jobs=-1, n_neighbors=n_neighbors)
    return controls_pca.index[factor.fit_predict(controls_pca) == -1]


def detect_elliptic_envelope(controls_pca: pd.DataFrame) -> pd.Index:
    """Flag controls outside an ellipse fitted to the bulk of them."""
    envelope = EllipticEnvelope()
    return controls_pca.index[envelope.fit_predict(controls_pca) == -1]


def count_votes(flagged: Dict[str, pd.Index]) -> Dict[str, int]:
    """How many detectors flagged each guide.

    Args:
        flagged: Detector to the rows it flagged

    Returns:
        Guide name to the number of detectors that flagged it.
    """
    votes: Counter = Counter()
    for rows in flagged.values():
        votes.update(pd.unique(rows.get_level_values(GUIDE_LEVEL)))
    return dict(votes)


def consensus(votes: Dict[str, int],
              min_methods: int = MIN_METHODS) -> List[str]:
    """The guides that enough detectors agreed on.

    Args:
        votes: Guide name to the number of detectors that flagged it
        min_methods: How many detectors must agree

    Returns:
        The guide names reaching *min_methods*, in the order first seen.
    """
    return [guide for guide, count in votes.items() if count >= min_methods]


def find_outlier_controls(controls: pd.DataFrame,
                          controls_pca: pd.DataFrame,
                          min_methods: int = MIN_METHODS,
                          n_estimators: int = N_ESTIMATORS,
                          random_state: int = RANDOM_STATE,
                          n_neighbors: int = N_NEIGHBORS) -> ControlResult:
    """Run the three detectors over the controls and take the vote.

    Args:
        controls: The control guides in the original feature space
        controls_pca: The same guides in the reduced space
        min_methods: How many detectors must agree
        n_estimators: Trees in the isolation forest
        random_state: Seed for the isolation forest
        n_neighbors: Neighbourhood size for the local outlier factor

    Returns:
        A :class:`ControlResult`.
    """
    result = ControlResult(controls=len(controls))
    detectors = (
        (ISOLATION_FOREST, lambda: detect_isolation_forest(
            controls, n_estimators=n_estimators, random_state=random_state)),
        (LOCAL_OUTLIER_FACTOR, lambda: detect_local_outlier_factor(
            controls_pca, n_neighbors=n_neighbors)),
        (ELLIPTIC_ENVELOPE, lambda: detect_elliptic_envelope(controls_pca)),
    )

    flagged: Dict[str, pd.Index] = {}
    for name, run in detectors:
        try:
            rows = run()
        except Exception as error:
            result.failed[name] = f"{type(error).__name__}: {error}"
            logger.warning(f"{name} did not run ({type(error).__name__}: "
                           f"{error}); the vote will be taken over the "
                           f"remaining detectors")
            continue
        flagged[name] = rows
        result.flagged[name] = len(rows)
        logger.info(f"{name} flagged {len(rows)} of {len(controls)} controls")

    if not flagged:
        raise RuntimeError(
            f"No control detector ran: "
            + '; '.join(f"{name} {reason}"
                        for name, reason in result.failed.items()))

    if len(flagged) < min_methods:
        logger.warning(f"Only {len(flagged)} detector(s) ran but "
                       f"{min_methods} must agree, so every guide they both "
                       f"flag will be dropped and no other can be")

    result.votes = count_votes(flagged)
    result.outliers = consensus(result.votes, min_methods=min_methods)
    logger.info(f"{len(result.outliers)} control guide(s) flagged by at least "
                f"{min_methods} of {len(flagged)} detector(s)")
    return result


def warn_on_shared_guide_names(data: pd.DataFrame) -> None:
    """Warn if a guide name appears against more than one gene.
    """
    from ..dimred import GENE_LEVEL

    if GUIDE_LEVEL not in (data.index.names or ()):
        return
    pairs = data.index.to_frame(index=False)[[GUIDE_LEVEL, GENE_LEVEL]]
    genes_per_guide = pairs.drop_duplicates().groupby(GUIDE_LEVEL).size()
    shared = genes_per_guide[genes_per_guide > 1]
    if len(shared):
        logger.warning(f"{len(shared)} guide name(s) appear against more than "
                       f"one gene: {', '.join(str(g) for g in shared.index[:5])}"
                       f"{', and more' if len(shared) > 5 else ''}. Filtering "
                       f"matches on the name alone, so removing one removes "
                       f"every gene it appears against")


def excluded_rows(data: pd.DataFrame, guides: Sequence[str]) -> pd.DataFrame:
    """The index entries of *data* whose guide is in *guides*."""
    if not len(guides):
        return data.index[[]].to_frame(index=False)
    rows = data.index.get_level_values(GUIDE_LEVEL).isin(list(guides))
    return data.index[rows].to_frame(index=False)

"""
The guide filtering step: drop guides that cannot be trusted.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .. import __version__
from ..combining.aggregate import aggregate
from ..dimred import (GENE_LEVEL, center_on_controls, cosine_similarity_matrix,
                      generate_pca_space)
from ..io.config import AllwellsConfig, GuideConfig, SchemaConfig
from ..io.tables import load_table, save_table
from .controls import (GUIDE_LEVEL, MIN_METHODS, ControlResult,
                       find_outlier_controls, subset_controls,
                       warn_on_shared_guide_names)
from .perturbations import (SimilarityResult, guide_similarities,
                            guides_below_threshold)
from .plotting import (plot_control_outliers, plot_detector_votes,
                       plot_similarity_comparison)

logger = logging.getLogger(__name__)

#: The tables this step reads, both written by the outlier step.
INPUT_NAME = 'allwells'
GUIDE_INPUT_SUFFIX = '-features_guide_outlier_filtered'
CELL_INPUT_SUFFIX = '-features_cell_outlier_filtered'

#: What this step writes.  Kept from cellmapp.
OUTPUT_SUFFIX = '-features_{level}'
FILTER_FILE = 'guide_filters.parquet'
RECORD_FILE = 'guide_filters.json'

#: Names every guide missing from the output, and why it went.
REASON_COLUMN = 'reason'
CONTROL = 'control'
PERTURBATION = 'perturbation'
LOW_CELLS = 'low_cells'
REASONS = (CONTROL, PERTURBATION, LOW_CELLS)


@dataclass
class GuideFilterRecord:
    """What produced the guide filter list."""
    created: str = ''
    chart_version: str = ''
    guide_source: str = ''
    cell_source: str = ''
    index_levels: List[str] = field(default_factory=list)
    controls: int = 0
    #: Detector to the number of control guides it flagged.
    detectors: Dict[str, int] = field(default_factory=dict)
    #: Detector to the reason it did not run, so a vote taken over fewer
    #: methods than intended is visible afterwards.
    detectors_failed: Dict[str, str] = field(default_factory=dict)
    guides_scored: int = 0
    single_guide_genes: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    #: Reason to the number of guides removed for it.
    removed_guides: Dict[str, int] = field(default_factory=dict)

    def save(self, filter_dir: str) -> str:
        os.makedirs(filter_dir, exist_ok=True)
        self.created = datetime.now().isoformat(timespec='seconds')
        self.chart_version = __version__
        path = os.path.join(filter_dir, RECORD_FILE)
        with open(path, 'w') as handle:
            json.dump(asdict(self), handle, indent=2)
        logger.info(f"Saved guide filter record: {path}")
        return path


def load_guide_record(filter_dir: str) -> Optional[GuideFilterRecord]:
    """Read the sidecar for the guide filter list, if there is one."""
    path = os.path.join(filter_dir, RECORD_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        raw = json.load(handle)
    known = set(GuideFilterRecord.__dataclass_fields__)
    return GuideFilterRecord(**{k: v for k, v in raw.items() if k in known})


@dataclass
class GuideResult:
    """Summary of a guide filtering run."""
    guides_in: int = 0
    controls: int = 0
    control_outliers: int = 0
    guides_scored: int = 0
    perturbation_outliers: int = 0
    low_cell_guides: int = 0
    cells_in: int = 0
    cells_out: int = 0
    filter_file: Optional[str] = None
    output: Optional[str] = None
    aggregated: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def _guide_pairs(index: pd.MultiIndex,
                 guides: Sequence[str]) -> pd.DataFrame:
    """The unique Guide and Gene pairs of *index* named in *guides*."""
    columns = [GUIDE_LEVEL, GENE_LEVEL]
    if not len(guides):
        return pd.DataFrame(columns=columns)
    frame = index.to_frame(index=False)[columns].drop_duplicates()
    return frame[frame[GUIDE_LEVEL].isin(list(guides))].reset_index(drop=True)


def filter_list(removed: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every guide missing from the output, with the reason it went.

    Args:
        removed: Reason to the Guide and Gene pairs removed for it

    Returns:
        One row per removed guide, with :data:`REASON_COLUMN`.
    """
    columns = [GUIDE_LEVEL, GENE_LEVEL, REASON_COLUMN]
    frames = []
    for reason, pairs in removed.items():
        if len(pairs):
            frame = pairs.copy()
            frame[REASON_COLUMN] = reason
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, axis=0, ignore_index=True)[columns]


def _truncate_for_envelope(controls_pca: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing components so the envelope has more rows than columns.

    EllipticEnvelope cannot fit a covariance with at least as many
    dimensions as observations.  Both it and the local outlier factor see
    the truncated space.
    """
    if len(controls_pca.columns) < len(controls_pca):
        return controls_pca
    keep = max(len(controls_pca) - 1, 1)
    logger.info(f"Truncating the control space from "
                f"{len(controls_pca.columns)} to {keep} component(s): the "
                f"envelope needs fewer dimensions than the "
                f"{len(controls_pca)} guides it is fitted to")
    return controls_pca.iloc[:, :keep]


def run_guide_filtering(allwells: AllwellsConfig,
                        guides: GuideConfig,
                        threshold: Optional[float] = None,
                        min_cells_per_guide: Optional[int] = None,
                        min_methods: int = MIN_METHODS,
                        aggregation_method: str = 'median',
                        save_plots: bool = True,
                        plots_dir: Optional[str] = None) -> GuideResult:
    """Drop untrustworthy guides and write the surviving tables.

    Args:
        allwells: Where the outlier step left its tables
        guides: Where this step writes, and its thresholds
        threshold: Cosine similarity a guide must reach (default: the
                   configured ``cosine_similarity_threshold``)
        min_cells_per_guide: Cells a guide needs to be kept (default: the
                             configured ``min_cells_per_guide``)
        min_methods: How many detectors must agree on a control guide
        aggregation_method: How to combine the rows of a group
        save_plots: Whether to draw and write the plots
        plots_dir: Where to write them (default: the configured directory)

    Returns:
        A :class:`GuideResult`.
    """
    threshold = (guides.cosine_similarity_threshold if threshold is None
                 else threshold)
    min_cells = (guides.min_cells_per_guide if min_cells_per_guide is None
                 else min_cells_per_guide)
    plot_dir = plots_dir or guides.plots_path

    logger.info(f"Reading from {allwells.outlier_filtered_path}")
    logger.info(f"Writing to {guides.guide_filtered_path}")

    guide_data = load_table(allwells.outlier_filtered_path, INPUT_NAME,
                            GUIDE_INPUT_SUFFIX)
    result = GuideResult(guides_in=len(guide_data))
    warn_on_shared_guide_names(guide_data)

    control_result, control_guides = _run_controls(
        guide_data, min_methods=min_methods, save_plots=save_plots,
        plot_dir=plot_dir, schema=guides.schema)
    result.controls = control_result.controls
    result.control_outliers = len(control_guides)

    kept = guide_data[~guide_data.index.get_level_values(GUIDE_LEVEL)
                      .isin(control_guides)]
    similarity, perturbation_guides = _run_perturbations(
        kept, threshold=threshold, save_plots=save_plots, plot_dir=plot_dir,
        schema=guides.schema)
    result.guides_scored = len(similarity.scores)
    result.perturbation_outliers = len(perturbation_guides)

    return _run_combine(allwells, guides, result, guide_data, control_result,
                        similarity, control_guides, perturbation_guides,
                        threshold=threshold, min_cells=min_cells,
                        min_methods=min_methods,
                        aggregation_method=aggregation_method)


def _run_controls(guide_data: pd.DataFrame, min_methods: int,
                  save_plots: bool, plot_dir: str,
                  schema: Optional[SchemaConfig] = None):
    """Phase one: the control guides that do not behave like controls."""
    controls = subset_controls(guide_data, schema)

    # The space is built from every guide, so the controls are placed
    # against the full spread rather than only against each other.
    pca = generate_pca_space(guide_data)
    controls_pca = _truncate_for_envelope(pca.loc[controls.index])

    control_result = find_outlier_controls(controls, controls_pca,
                                           min_methods=min_methods)

    if save_plots:
        labels = controls.index.to_frame(index=False)
        labels = dict(zip(labels[GUIDE_LEVEL],
                          labels[GENE_LEVEL].astype(str) + '_'
                          + labels[GUIDE_LEVEL].astype(str)))
        votes = pd.Series(
            {labels.get(guide, guide): count
             for guide, count in control_result.votes.items()},
            dtype='int64')
        plot_detector_votes(votes, plot_dir)
        plot_control_outliers(controls_pca, control_result.outliers, plot_dir)

    return control_result, control_result.outliers


def _run_perturbations(kept: pd.DataFrame, threshold: float,
                       save_plots: bool, plot_dir: str,
                       schema: Optional[SchemaConfig] = None):
    """Phase two: the guides that disagree with their own gene."""
    # A second space, because the first was built with the outlier
    # controls still in it and they move the centroid everything is
    # measured from.
    centered = center_on_controls(generate_pca_space(kept), schema)
    similarity = guide_similarities(cosine_similarity_matrix(centered), schema)

    if save_plots:
        plot_similarity_comparison(similarity.scores, threshold, plot_dir)

    return similarity, guides_below_threshold(similarity.scores, threshold)


def _run_combine(allwells: AllwellsConfig, guides: GuideConfig,
                 result: GuideResult, guide_data: pd.DataFrame,
                 control_result: ControlResult,
                 similarity: SimilarityResult,
                 control_guides: Sequence[str],
                 perturbation_guides: Sequence[str],
                 threshold: float, min_cells: int, min_methods: int,
                 aggregation_method: str) -> GuideResult:
    """Phase three: apply both lists to the cells and write everything."""
    cells = load_table(allwells.outlier_filtered_path, INPUT_NAME,
                       CELL_INPUT_SUFFIX)
    result.cells_in = len(cells)

    excluded = list(control_guides) + list(perturbation_guides)
    surviving = cells[~cells.index.get_level_values(GUIDE_LEVEL)
                      .isin(excluded)]
    logger.info(f"Removed {len(cells) - len(surviving)} cell(s) carrying the "
                f"{len(excluded)} excluded guide(s)")

    counts = surviving.index.get_level_values(GUIDE_LEVEL).value_counts()
    low_cell_guides = counts[counts < min_cells].index.tolist()
    result.low_cell_guides = len(low_cell_guides)
    if low_cell_guides:
        surviving = surviving[~surviving.index.get_level_values(GUIDE_LEVEL)
                              .isin(low_cell_guides)]
    logger.info(f"{len(low_cell_guides)} guide(s) carry fewer than "
                f"{min_cells} cells")

    if surviving.empty:
        raise ValueError(
            f"Guide filtering removed every cell of {result.cells_in}: "
            f"{result.control_outliers} control guide(s), "
            f"{result.perturbation_outliers} perturbation guide(s) and "
            f"{result.low_cell_guides} guide(s) below {min_cells} cells. "
            f"Check cosine_similarity_threshold ({threshold}) and "
            f"min_cells_per_guide ({min_cells}).")

    # The list is written whether or not anything was removed, so that an
    # empty one is distinguishable from a step that never ran.
    os.makedirs(guides.filters_path, exist_ok=True)
    removed = filter_list({
        CONTROL: _guide_pairs(guide_data.index, control_guides),
        PERTURBATION: _guide_pairs(guide_data.index, perturbation_guides),
        LOW_CELLS: _guide_pairs(cells.index, low_cell_guides)})
    result.filter_file = os.path.join(guides.filters_path, FILTER_FILE)
    removed.to_parquet(result.filter_file, index=False)
    by_reason = removed[REASON_COLUMN].value_counts().to_dict()
    logger.info(f"Saved {len(removed)} removed guide(s) to "
                f"{result.filter_file}: "
                + (', '.join(f"{count} {reason}"
                             for reason, count in by_reason.items())
                   or 'none'))

    GuideFilterRecord(
        guide_source=os.path.join(allwells.outlier_filtered_path,
                                  f"{INPUT_NAME}{GUIDE_INPUT_SUFFIX}.parquet"),
        cell_source=os.path.join(allwells.outlier_filtered_path,
                                 f"{INPUT_NAME}{CELL_INPUT_SUFFIX}.parquet"),
        index_levels=[str(n) for n in cells.index.names],
        controls=control_result.controls,
        detectors=dict(control_result.flagged),
        detectors_failed=dict(control_result.failed),
        guides_scored=len(similarity.scores),
        single_guide_genes=list(similarity.single_guide_genes),
        parameters={'cosine_similarity_threshold': threshold,
                    'min_cells_per_guide': min_cells,
                    'min_methods': min_methods,
                    'detectors_run': control_result.methods_run},
        removed_guides={str(reason): int(count)
                        for reason, count in by_reason.items()}
    ).save(guides.filters_path)

    result.cells_out = len(surviving)
    result.output = save_table(surviving, guides.guide_filtered_path,
                               INPUT_NAME, OUTPUT_SUFFIX.format(level='cell'))

    # Gene level is the median of the guide medians, not of the cells, so
    # a guide with many cells does not speak for the whole gene.  Chaining
    # the two aggregations is what makes it so.
    per_guide = aggregate(surviving, 'guide', method=aggregation_method)
    result.aggregated['guide'] = save_table(
        per_guide, guides.guide_filtered_path, INPUT_NAME,
        OUTPUT_SUFFIX.format(level='guide'))

    per_gene = aggregate(per_guide, 'gene', method=aggregation_method)
    result.aggregated['gene'] = save_table(
        per_gene, guides.guide_filtered_path, INPUT_NAME,
        OUTPUT_SUFFIX.format(level='gene'))

    return result

"""
Programs: sets of archetypes that genes move together, found by sparse PCA
on the genes' pooled effects.  The genes are then grouped into modules by
how often they cluster together across refits, and every cell is scored on
the programs.
"""

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import SparsePCA
from sklearn.neighbors import kneighbors_graph

from .. import __version__
from ..archetypes.crops import with_boxes
from ..combining.workflow import OUTPUT_NAME, SUFFIXES
from ..dimred.channels import INDEX_COLUMNS
from ..effects.workflow import POOLED, _write_record, clr_transform
from ..io.config import (ARCHETYPES_SECTION, join_path, load_allwells_config,
                         load_archetypes_config, load_effects_config,
                         load_programs_config)
from ..io.tables import load_table, save_table, table_exists, table_path

logger = logging.getLogger(__name__)

STEPS = ('matrix', 'fit', 'modules', 'cells')

MATRIX, LOADINGS, SCORES = 'matrix', 'loadings', 'scores'
CONSENSUS, MODULES = 'consensus', 'modules'
CELL_SCORES, TOP_CELLS = 'cell_scores', 'top_cells'

SPCA_MAX_ITER = 500
SPCA_TOL = 1e-4
#: SparsePCA's default, which its transform uses.
RIDGE_ALPHA = 0.01

N_SEEDS = 10
RUNS_PER_SEED = 15
SUBSAMPLE_FRAC = 0.85
HC_LINKAGES = ('ward', 'average')
HC_N_CLUSTERS = tuple(range(12, 30, 2))
GRAPH_N_NEIGHBORS = (5, 8, 10, 12)
GRAPH_RESOLUTIONS = (0.8, 1.0, 1.2, 1.5, 1.8)
CORE_THRESHOLD = 0.15

TOP_N_CELLS = 100
TOP_N_PURE = 20


def program_names(k: int) -> List[str]:
    return [f'P{i + 1}' for i in range(k)]


def effect_matrix(pooled: pd.DataFrame, qval_threshold: float) -> pd.DataFrame:
    """Each gene's signed effect on each archetype, keeping the genes and
    archetypes with any q-value below *qval_threshold*."""
    pivot = pooled.pivot(index='Gene', columns='Archetype', values='signed_w1')
    sig = pooled.pivot(index='Gene', columns='Archetype', values='qval') < qval_threshold
    matrix = pivot.loc[sig.any(axis=1), sig.any(axis=0)]
    if matrix.empty:
        raise ValueError(f"No gene has a q-value below {qval_threshold}")
    untested = matrix.index[matrix.isna().any(axis=1)]
    if len(untested):
        raise ValueError(f"{len(untested)} genes were not tested on every "
                         f"archetype kept, the first being {untested[0]}; "
                         f"their wells hold no controls in some channel")
    return matrix


def fit_spca(X: np.ndarray, K: int, rs: int,
             n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    model = SparsePCA(n_components=K, random_state=rs, max_iter=SPCA_MAX_ITER,
                      tol=SPCA_TOL, n_jobs=n_jobs)
    S = model.fit_transform(X)
    return S, model.components_.T


def project(table: pd.DataFrame, loadings: pd.DataFrame) -> pd.DataFrame:
    """Scores on each program as SparsePCA's transform gives them, but
    without subtracting the average gene first, so that genes and cells
    share a scale."""
    L = loadings.to_numpy()
    inverse = np.linalg.inv(L.T @ L + RIDGE_ALPHA * np.eye(L.shape[1]))
    return pd.DataFrame(table[loadings.index].to_numpy() @ L @ inverse,
                        index=table.index, columns=loadings.columns)


def fit_programs(matrix: pd.DataFrame, k: int, seed: int,
                 n_jobs: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The programs' loadings on each archetype, each turned so that its
    largest loading is positive, and each gene's scores on them."""
    _, L = fit_spca(matrix.to_numpy(), k, seed, n_jobs)
    L[:, L[np.abs(L).argmax(axis=0), np.arange(k)] < 0] *= -1
    loadings = pd.DataFrame(L, index=matrix.columns, columns=program_names(k))
    return loadings, project(matrix, loadings)


def align_to_reference(L_new, L_ref):
    """Align L_new components to L_ref via cosine similarity + Hungarian.
    Returns (permutation, signs) such that L_new[:, perm] * signs aligns to L_ref."""
    n_ref = np.linalg.norm(L_ref, axis=0, keepdims=True) + 1e-15
    n_new = np.linalg.norm(L_new, axis=0, keepdims=True) + 1e-15
    C = (L_ref / n_ref).T @ (L_new / n_new)

    row_ind, col_ind = linear_sum_assignment(-np.abs(C))
    signs = np.sign(C[row_ind, col_ind])
    signs[signs == 0] = 1.0
    return col_ind, signs


def cluster_hierarchical(S: np.ndarray) -> List[np.ndarray]:
    """Hierarchical clusterings of the scores, each program z-scored so
    that all contribute equally."""
    col_std = S.std(axis=0)
    col_std[col_std == 0] = 1.0
    S = (S - S.mean(axis=0)) / col_std
    results = []
    for link in HC_LINKAGES:
        Z = linkage(pdist(S, metric='euclidean'), method=link)
        for nc in HC_N_CLUSTERS:
            results.append(fcluster(Z, t=nc, criterion='maxclust') - 1)
    return results


def cluster_graph(S: np.ndarray, seed: int) -> List[np.ndarray]:
    """Leiden clusterings of a nearest-neighbour graph of the scores."""
    results = []
    for nn in GRAPH_N_NEIGHBORS:
        A = kneighbors_graph(S, n_neighbors=nn, mode='connectivity', include_self=False)
        A = A.maximum(A.T)
        sources, targets = A.nonzero()
        g = ig.Graph(n=S.shape[0], edges=list(zip(sources.tolist(), targets.tolist())),
                     directed=False)
        g.simplify()
        for res in GRAPH_RESOLUTIONS:
            part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                            resolution_parameter=res, seed=seed)
            results.append(np.array(part.membership))
    return results


def gene_modules(matrix: pd.DataFrame, loadings: pd.DataFrame, n_modules: int,
                 seed: int, n_jobs: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group the genes by how often they cluster together when the programs
    are refitted on part of them.

    Returns:
        The share of clusterings that put each pair of genes together, and
        each gene's module with how firmly it belongs there.
    """
    X, L_ref = matrix.to_numpy(), loadings.to_numpy()
    n_pert, K = len(X), L_ref.shape[1]
    n_sub = int(n_pert * SUBSAMPLE_FRAC)
    rng = np.random.default_rng(seed)
    C_num = np.zeros((n_pert, n_pert), dtype=np.float64)
    C_den = np.zeros((n_pert, n_pert), dtype=np.float64)

    runs = N_SEEDS * RUNS_PER_SEED
    logger.info(f"Modules: {runs} refits on {n_sub} of {n_pert} genes")
    for seed_idx in range(N_SEEDS):
        for sub_iter in range(RUNS_PER_SEED):
            idx = np.sort(rng.choice(n_pert, size=n_sub, replace=False))
            S_sub, L_sub = fit_spca(X[idx], K, seed + seed_idx + sub_iter, n_jobs)
            perm, signs = align_to_reference(L_sub, L_ref)
            S_aligned = S_sub[:, perm] * signs[np.newaxis, :]

            clusterings = cluster_hierarchical(S_aligned) + cluster_graph(S_aligned, seed)
            C_den[np.ix_(idx, idx)] += len(clusterings)
            np.fill_diagonal(C_den, 0)
            for labels in clusterings:
                same = (labels[:, None] == labels[None, :]).astype(np.float64)
                np.fill_diagonal(same, 0)
                C_num[np.ix_(idx, idx)] += same

            done = seed_idx * RUNS_PER_SEED + sub_iter + 1
            if done % 10 == 0:
                logger.info(f"Modules: {done}/{runs} refits")

    C = np.divide(C_num, C_den, out=np.zeros_like(C_num), where=C_den > 0)
    np.fill_diagonal(C, 1.0)
    D = np.clip(1.0 - C, 0, None)
    np.fill_diagonal(D, 0)
    final_labels = fcluster(linkage(squareform(D, checks=False), method='average'),
                            t=n_modules, criterion='maxclust') - 1

    diag = []
    for i in range(n_pert):
        my_cluster = final_labels[i]
        same_mask = final_labels == my_cluster
        same_mask[i] = False
        mean_own = C[i, same_mask].mean() if same_mask.any() else 1.0

        max_other, best_other_cluster = 0.0, -1
        for c in range(n_modules):
            c_mask = final_labels == c
            if c == my_cluster or not c_mask.any():
                continue
            mean_c = C[i, c_mask].mean()
            if mean_c > max_other:
                max_other, best_other_cluster = mean_c, c

        diag.append({'module': my_cluster, 'mean_consensus_own': mean_own,
                     'max_consensus_other': max_other,
                     'best_other_module': best_other_cluster,
                     'margin': mean_own - max_other})

    modules = pd.DataFrame(diag, index=matrix.index)
    modules['assignment'] = np.where(modules['margin'] < CORE_THRESHOLD, 'weak', 'core')
    return pd.DataFrame(C, index=matrix.index, columns=matrix.index), modules


def cell_scores(weights: Dict[str, pd.DataFrame],
                loadings: pd.DataFrame) -> pd.DataFrame:
    """Every cell's scores on the programs, from the log-ratios of its
    weights in each channel.  A cell missing from a channel counts as
    weighting that channel's archetypes equally."""
    parts = []
    for channel, table in weights.items():
        table = table.set_index(list(INDEX_COLUMNS))
        parts.append(pd.DataFrame(clr_transform(table.to_numpy(dtype=np.float64)),
                                  index=table.index,
                                  columns=[f'{channel}_{c}' for c in table.columns]))
    combined = pd.concat(parts, axis=1, join='outer')
    missing = loadings.index.difference(combined.columns)
    if len(missing):
        raise ValueError(f"The programs load on {', '.join(missing[:3])}"
                         f"{' and others' if len(missing) > 3 else ''}, which "
                         f"the weights do not hold; rerun from the matrix step "
                         f"if the weights have changed")
    return project(combined[loadings.index].fillna(0.0), loadings)


def example_cells(scores: pd.DataFrame, objects: pd.DataFrame,
                  bbox_columns: Sequence[str]) -> pd.DataFrame:
    """For each program, the purest of its highest-scoring cells, with their
    bounding boxes.

    A cell's purity is its percentile on the program less its mean
    percentile on the others.
    """
    sorted_vals = {prog: np.sort(scores[prog].to_numpy()) for prog in scores.columns}

    def scores_to_pct(prog, values):
        sv = sorted_vals[prog]
        return 100.0 * np.searchsorted(sv, values, side='left') / len(sv)

    ranked = []
    for program in scores.columns:
        top_cells = scores.nlargest(TOP_N_CELLS, program)
        other_pcts = np.column_stack([scores_to_pct(p, top_cells[p].to_numpy())
                                      for p in scores.columns if p != program])
        top_cells = top_cells.assign(
            Purity=scores_to_pct(program, top_cells[program].to_numpy())
            - other_pcts.mean(axis=1))
        pure_top = top_cells.nlargest(TOP_N_PURE, 'Purity')
        ranked.append(pd.DataFrame({
            'Program': program,
            'Rank': np.arange(1, len(pure_top) + 1),
            'Score': pure_top[program].to_numpy(),
            'Purity': pure_top['Purity'].to_numpy(),
            **{name: pure_top.index.get_level_values(name) for name in INDEX_COLUMNS}}))
    return with_boxes(pd.concat(ranked, ignore_index=True), objects, bbox_columns)


def _load(directory: str, name: str, remedy: str) -> pd.DataFrame:
    if not table_exists(directory, name):
        raise FileNotFoundError(f"No {name}.parquet in {directory}; {remedy}")
    return load_table(directory, name)


def _save(table: pd.DataFrame, directory: str, name: str, **record: Any) -> str:
    output = save_table(table, directory, name)
    _write_record(join_path(directory, f'{name}.json'),
                  {'created': datetime.now().isoformat(timespec='seconds'),
                   'chart_version': __version__, **record})
    return output


def run_programs(config: Union[str, Dict[str, Any]],
                 steps: Sequence[str] = ('all',),
                 n_jobs: Optional[int] = None) -> Dict[str, str]:
    """Run the programs steps in order, each reading what the one before
    it wrote.

    Args:
        config: A configuration file or mapping; the pooled effects, the
                weights and the objects table come from its other sections
        steps: Which of :data:`STEPS` to run; ``('all',)`` runs them all
        n_jobs: Workers for the sparse PCA, in place of the configured ones

    Returns:
        The main table each step wrote.
    """
    programs = load_programs_config(config)
    if n_jobs is not None:
        programs = replace(programs, n_jobs=n_jobs)
    wanted = STEPS if 'all' in steps else tuple(s for s in STEPS if s in steps)
    directory = programs.outputs_path
    outputs: Dict[str, str] = {}

    if 'matrix' in wanted:
        effects = load_effects_config(config).outputs_path
        matrix = effect_matrix(_load(effects, POOLED, "run 'chart effects' first"),
                               programs.qval_threshold)
        logger.info(f"{matrix.shape[0]} genes and {matrix.shape[1]} archetypes "
                    f"have a q-value below {programs.qval_threshold}")
        outputs['matrix'] = _save(matrix, directory, MATRIX,
                                  effects=table_path(effects, POOLED),
                                  qval_threshold=programs.qval_threshold)

    if 'fit' in wanted:
        matrix = _load(directory, MATRIX, 'run the matrix step')
        logger.info(f"Fitting {programs.k} programs")
        loadings, scores = fit_programs(matrix, programs.k, programs.seed,
                                        programs.n_jobs)
        save_table(loadings, directory, LOADINGS)
        outputs['fit'] = _save(scores, directory, SCORES, k=programs.k,
                               seed=programs.seed)

    if 'modules' in wanted:
        matrix = _load(directory, MATRIX, 'run the matrix step')
        loadings = _load(directory, LOADINGS, 'run the fit step')
        scores = _load(directory, SCORES, 'run the fit step')
        if (list(loadings.index) != list(matrix.columns)
                or list(scores.index) != list(matrix.index)):
            raise ValueError(f"The loadings in {directory} were fitted on "
                             f"another matrix; rerun the fit step")
        consensus, modules = gene_modules(matrix, loadings, programs.n_modules,
                                          programs.seed, programs.n_jobs)
        save_table(consensus, directory, CONSENSUS)
        outputs['modules'] = _save(modules, directory, MODULES,
                                   n_modules=programs.n_modules,
                                   seed=programs.seed,
                                   refits=N_SEEDS * RUNS_PER_SEED)

    if 'cells' in wanted:
        archetypes = load_archetypes_config(config)
        loadings = _load(directory, LOADINGS, 'run the fit step')
        removed = sorted(set(loadings.index) & {
            name for names in archetypes.archetypes_to_remove.values() for name in names})
        if removed:
            raise ValueError(f"The programs load on {', '.join(removed)}, which "
                             f"'{ARCHETYPES_SECTION}.archetypes_to_remove' now "
                             f"leaves out; rerun 'chart effects', then the "
                             f"programs from the matrix step")
        paths = {channel: archetypes.weights_path(channel)
                 for channel in archetypes.channels}
        weights = {}
        for channel, path in paths.items():
            logger.info(f"Reading {path}")
            weights[channel] = pd.read_parquet(path)
        scores = cell_scores(weights, loadings)
        outputs['cells'] = _save(scores.reset_index(), directory, CELL_SCORES,
                                 weights=paths)
        objects = load_table(load_allwells_config(config).combined_path,
                             OUTPUT_NAME, SUFFIXES['objects'])
        save_table(example_cells(scores, objects, archetypes.schema.bbox_columns),
                   directory, TOP_CELLS)

    return outputs
